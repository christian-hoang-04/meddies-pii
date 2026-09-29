from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Protocol

from meddies_pii.bioes_inference import PiiSpanDetector, SpanDetection
from meddies_pii.bioes_inference.file_identity import file_sha256
from meddies_pii.languages import normalize_language
from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    PageRegion,
    ReviewedRedaction,
)
from meddies_pii.pdf_redaction.errors import (
    InputDocumentError,
    OutputPathError,
    RedactionApplyError,
    VerificationError,
)
from meddies_pii.pdf_redaction.geometry import map_detections_to_regions
from meddies_pii.regex_runtime import RegexLanguageSource
from meddies_pii.regex_runtime import apply as apply_regex_runtime
from meddies_pii.taxonomy import PiiLabel, require_pii_label

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from meddies_pii.pdf_redaction.document import (
        DocumentAdapter,
        DocumentInspection,
    )
    from meddies_pii.pdf_redaction.extraction import NativePdfExtractor
    from meddies_pii.pdf_redaction.ocr import GeometryOcr
    from meddies_pii.pdf_redaction.routing import PageRouteDecision
    from meddies_pii.pdf_redaction.verification import VerificationReport
    from meddies_pii.pdf_redaction.writers import RedactionWriteResult

DEFAULT_MAX_SOURCE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_TEXT_CHARS = 2_000_000
"""Generous for complete clinical narratives, bounded for in-process PHI handling."""


class RedactionWriter(Protocol):
    @property
    def writer_id(self) -> str: ...

    def write(
        self,
        source: Path,
        regions: Sequence[PageRegion],
        destination: Path,
    ) -> RedactionWriteResult: ...


class PdfVerifier(Protocol):
    def verify(
        self,
        written: RedactionWriteResult,
        *,
        known_canaries: Sequence[str],
    ) -> VerificationReport: ...


@dataclass(frozen=True, slots=True)
class RedactionPreview:
    source_path: Path = field(repr=False)
    source_bytes: bytes = field(repr=False)
    source_sha256: str
    pages: tuple[GeometryPage, ...] = field(repr=False)
    detections: tuple[SpanDetection, ...] = field(repr=False)
    suggestions: tuple[ReviewedRedaction, ...]
    regex_manifest_sha256: str | None = None
    regex_language: str | None = None
    regex_language_source: RegexLanguageSource | None = None


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    source_path: Path = field(repr=False)
    source_bytes: bytes = field(repr=False)
    source_sha256: str
    pages: tuple[GeometryPage, ...] = field(repr=False)
    decisions: tuple[PageRouteDecision, ...]
    raster_sha256: tuple[str, ...]


class PdfGeometryPreparer:
    def __init__(
        self,
        *,
        document_adapter: DocumentAdapter,
        native_extractor: NativePdfExtractor,
        ocr: GeometryOcr,
        max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    ) -> None:
        if max_source_bytes <= 0:
            msg = "max_source_bytes must be positive"
            raise ValueError(msg)
        self._document_adapter = document_adapter
        self._native_extractor = native_extractor
        self._ocr = ocr
        self._max_source_bytes = max_source_bytes

    def prepare(self, source: Path) -> PreparedDocument:
        source = Path(source)
        source_bytes = _read_source_once(source, max_bytes=self._max_source_bytes)
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        with _materialized_source(source_bytes) as snapshot:
            inspection = self._document_adapter.inspect(snapshot)
            native_pages = self._native_extractor.extract(snapshot)
            _validate_page_alignment(inspection, native_pages)

            ocr_indices = tuple(
                page.signals.page_index for page in inspection.pages if page.decision.requires_full_page_ocr
            )
            artifacts = self._document_adapter.render_pages(
                snapshot,
                page_indices=ocr_indices,
            )
            ocr_pages = {artifact.raster.page_index: self._ocr.extract(artifact.raster) for artifact in artifacts}
            pages = tuple(ocr_pages.get(page.page_index, page) for page in native_pages)
        if tuple(page.page_index for page in pages) != tuple(range(len(pages))):
            msg = "prepared page indices are not contiguous"
            raise RuntimeError(msg)
        return PreparedDocument(
            source_path=source,
            source_bytes=source_bytes,
            source_sha256=source_sha256,
            pages=pages,
            decisions=tuple(page.decision for page in inspection.pages),
            raster_sha256=tuple(artifact.sha256 for artifact in artifacts),
        )


@dataclass(frozen=True, slots=True)
class VerifiedRedactionOutput:
    path: Path
    sha256: str
    writer_id: str
    page_count: int
    verification_codes: tuple[str, ...]
    verification_scope: str = "reviewed_values"
    regex_manifest_sha256: str | None = None
    regex_language: str | None = None
    regex_language_source: RegexLanguageSource | None = None


class PipelineVerificationError(VerificationError):
    def __init__(self, failed_codes: Sequence[str]) -> None:
        normalized = tuple(dict.fromkeys(failed_codes))
        if not normalized:
            normalized = ("verification_failed_without_finding",)
        super().__init__(
            "redacted PDF failed independent verification",
            stage="verify",
            code="independent_verification_failed",
        )
        self.failed_codes = normalized


class ExtractedTextLimitError(InputDocumentError):
    def __init__(self) -> None:
        super().__init__(
            "prepared PDF text exceeds extraction ceiling",
            stage="input",
            code="extracted_text_too_large",
        )


class PdfRedactionPipeline:
    """Preview and apply reviewed redactions with optional regex recovery.

    A caller language is normalized through the Meddies language registry. Without
    one, the pipeline detects one language from the full prepared document and uses it
    for every page. An undecided document disables language-scoped organization packs
    while language-independent authoritative rules remain enabled.
    """

    def __init__(
        self,
        *,
        detector: PiiSpanDetector,
        writer: RedactionWriter,
        verifier: PdfVerifier,
        language: str | None = None,
        regex_postprocess: bool = True,
    ) -> None:
        if language is not None and not isinstance(language, str):
            msg = "language must be a string or None"
            raise TypeError(msg)
        if not isinstance(regex_postprocess, bool):
            msg = "regex_postprocess must be a boolean"
            raise TypeError(msg)
        self._detector = detector
        self._writer = writer
        self._verifier = verifier
        self._language = normalize_language(language).code if language is not None else None
        self._regex_postprocess = regex_postprocess

    def preview(self, prepared: PreparedDocument) -> RedactionPreview:
        normalized_pages = prepared.pages
        if not normalized_pages:
            msg = "redaction preview requires at least one page"
            raise ValueError(msg)
        if sum(len(page.text) for page in normalized_pages) > DEFAULT_MAX_EXTRACTED_TEXT_CHARS:
            raise ExtractedTextLimitError
        nonblank_indices = tuple(index for index, page in enumerate(normalized_pages) if page.text.strip())
        empty = SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False)
        aligned_detections = [empty] * len(normalized_pages)
        if nonblank_indices:
            self._detector.load()
            detected = self._detector.detect([normalized_pages[index].text for index in nonblank_indices])
            if len(detected) != len(nonblank_indices):
                msg = "detector result count does not match nonblank pages"
                raise RuntimeError(msg)
            for index, detection in zip(nonblank_indices, detected, strict=True):
                aligned_detections[index] = detection
        detections = tuple(aligned_detections)
        regex_result = apply_regex_runtime(
            tuple(page.text for page in normalized_pages),
            tuple(detection.spans for detection in detections),
            language=self._language,
            enabled=self._regex_postprocess,
        )
        detections = tuple(
            replace(detection, spans=spans) for detection, spans in zip(detections, regex_result.spans, strict=True)
        )
        regions = map_detections_to_regions(normalized_pages, detections)
        return RedactionPreview(
            source_path=prepared.source_path,
            source_bytes=prepared.source_bytes,
            source_sha256=prepared.source_sha256,
            pages=normalized_pages,
            detections=detections,
            suggestions=_reviewed_suggestions(detections, regions),
            regex_manifest_sha256=regex_result.manifest_sha256,
            regex_language=regex_result.regex_language,
            regex_language_source=regex_result.regex_language_source,
        )

    def apply_verified(
        self,
        *,
        preview: RedactionPreview,
        reviewed: Sequence[ReviewedRedaction],
        destination: Path,
    ) -> VerifiedRedactionOutput:
        reviewed = tuple(reviewed)
        if not reviewed:
            msg = "at least one reviewed redaction is required"
            raise ValueError(msg)
        regions = tuple(region for item in reviewed for region in item.to_regions())
        known_values = tuple(dict.fromkeys(item.value for item in reviewed))
        if not known_values:
            msg = "reviewed redactions contain no values"
            raise ValueError(msg)

        destination = Path(destination)
        _require_destination_available(preview.source_path, destination)
        _require_source_identity(preview.source_path, preview.source_sha256)
        with _private_workspace(destination.parent) as workspace:
            snapshot = workspace / "source.pdf"
            staged_output = workspace / "redacted.pdf"
            _write_private_file(snapshot, preview.source_bytes)
            written = self._writer.write(snapshot, regions, staged_output)
            if written.output_path.resolve() != staged_output.resolve():
                msg = "writer returned output outside private staging"
                raise RedactionApplyError(
                    msg,
                    stage="apply",
                    code="writer_output_path_mismatch",
                )
            try:
                staged_output.chmod(0o600)
            except OSError as error:
                msg = "writer output permissions could not be restricted"
                raise RedactionApplyError(
                    msg,
                    stage="apply",
                    code="writer_output_permissions_failed",
                    cause_type=type(error).__name__,
                ) from None
            if written.source_sha256 != preview.source_sha256:
                msg = "writer used a different source PDF than the preview"
                raise InputDocumentError(
                    msg,
                    stage="input",
                    code="writer_source_identity_mismatch",
                )
            report = self._verifier.verify(
                written,
                known_canaries=known_values,
            )
            failed_codes = tuple(finding.code for finding in report.findings if not finding.passed)
            staged_sha256 = _source_sha256(staged_output)
            if report.output_sha256 != written.output_sha256 or staged_sha256 != written.output_sha256:
                failed_codes += ("verification_output_identity_mismatch",)
            if not report.passed or failed_codes:
                raise PipelineVerificationError(failed_codes)
            _promote_verified_output(staged_output, destination)

        return VerifiedRedactionOutput(
            path=destination,
            sha256=report.output_sha256,
            writer_id=written.writer_id,
            page_count=written.page_count,
            verification_codes=tuple(finding.code for finding in report.findings),
            regex_manifest_sha256=preview.regex_manifest_sha256,
            regex_language=preview.regex_language,
            regex_language_source=preview.regex_language_source,
        )


def _validate_page_alignment(
    inspection: DocumentInspection,
    native_pages: Sequence[GeometryPage],
) -> None:
    expected = tuple(range(len(inspection.pages)))
    observed = tuple(page.page_index for page in native_pages)
    if observed != expected:
        msg = "native extraction page indices do not match document inspection"
        raise RuntimeError(msg)


def _reviewed_suggestions(
    detections: Sequence[SpanDetection],
    regions: Sequence[PageRegion],
) -> tuple[ReviewedRedaction, ...]:
    regions_by_span: dict[tuple[int, int, int, PiiLabel], list[PageRegion]] = {}
    for region in regions:
        key = (
            region.page_index,
            region.span_start,
            region.span_end,
            region.label,
        )
        regions_by_span.setdefault(key, []).append(region)

    suggestions: list[ReviewedRedaction] = []
    for page_index, detection in enumerate(detections):
        for span in detection.spans:
            label = require_pii_label(span.label)
            key = (page_index, span.start, span.end, label)
            grouped_regions = regions_by_span.pop(key, None)
            if not grouped_regions:
                msg = "detected span has no mapped geometry"
                raise RuntimeError(msg)
            suggestions.append(
                ReviewedRedaction(
                    value=span.text,
                    label=label,
                    page_index=page_index,
                    quads=tuple(region.quad for region in grouped_regions),
                    span_start=span.start,
                    span_end=span.end,
                ),
            )
    if regions_by_span:
        msg = "mapped region has no source detection"
        raise RuntimeError(msg)
    return tuple(suggestions)


def _source_sha256(source: Path) -> str:
    try:
        return file_sha256(source)
    except OSError as error:
        msg = "source PDF is not readable"
        raise InputDocumentError(
            msg,
            stage="input",
            code="source_unreadable",
            cause_type=type(error).__name__,
        ) from None


def _read_source_once(source: Path, *, max_bytes: int) -> bytes:
    try:
        with source.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError as error:
        msg = "source PDF is not readable"
        raise InputDocumentError(
            msg,
            stage="input",
            code="source_unreadable",
            cause_type=type(error).__name__,
        ) from None
    if len(payload) > max_bytes:
        msg = "PDF input exceeds size limit"
        raise InputDocumentError(
            msg,
            stage="input",
            code="source_too_large",
        )
    return payload


@contextmanager
def _materialized_source(source_bytes: bytes) -> Iterator[Path]:
    with _private_workspace() as root:
        snapshot = root / "source.pdf"
        _write_private_file(snapshot, source_bytes)
        yield snapshot


@contextmanager
def _private_workspace(parent: Path | None = None) -> Iterator[Path]:
    with TemporaryDirectory(prefix=".meddies-pdf-", dir=parent) as directory:
        root = Path(directory)
        root.chmod(0o700)
        yield root


def _write_private_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _require_destination_available(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        msg = "redaction output must use a different path"
        raise OutputPathError(
            msg,
            stage="output",
            code="in_place_output",
        )
    if destination.exists():
        msg = "redaction output path must not already exist"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_exists",
        )
    if not destination.parent.is_dir():
        msg = "redaction output parent directory does not exist"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_parent_missing",
        )


def _promote_verified_output(staged: Path, destination: Path) -> None:
    try:
        os.link(staged, destination)
    except FileExistsError as error:
        msg = "redaction output path was created before verified promotion"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_exists",
            cause_type=type(error).__name__,
        ) from error
    except OSError as error:
        msg = "verified redaction output could not be promoted"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_promotion_failed",
            cause_type=type(error).__name__,
        ) from error


def _require_source_identity(source: Path, expected_sha256: str) -> None:
    if _source_sha256(source) != expected_sha256:
        msg = "source PDF changed after geometry preparation"
        raise InputDocumentError(
            msg,
            stage="input",
            code="source_identity_changed",
        )
