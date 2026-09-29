from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, override

import pytest

from meddies_pii.bioes_inference import SpanDetection
from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    GeometryToken,
    PageRegion,
    Point,
    Quad,
)
from meddies_pii.pdf_redaction.errors import (
    InputDocumentError,
    OutputPathError,
    RedactionApplyError,
)
from meddies_pii.pdf_redaction.pipeline import (
    PdfRedactionPipeline,
    PipelineVerificationError,
    PreparedDocument,
)
from meddies_pii.pdf_redaction.verification import (
    VerificationFinding,
    VerificationReport,
)
from meddies_pii.pdf_redaction.writers import RedactionWriteResult
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from meddies_pii.pdf_redaction.pipeline import RedactionWriter


def _quad() -> Quad:
    return Quad((Point(1, 1), Point(10, 1), Point(10, 5), Point(1, 5)))


def _page() -> GeometryPage:
    return GeometryPage(
        page_index=0,
        width_pt=100,
        height_pt=100,
        text="Name",
        tokens=(
            GeometryToken(
                text="Name",
                quad=_quad(),
                start=0,
                end=4,
                source="native",
                confidence=None,
            ),
        ),
    )


def _prepared(source: Path, pages: tuple[GeometryPage, ...] = (_page(),)) -> PreparedDocument:
    source_bytes = source.read_bytes()
    return PreparedDocument(
        source_path=source,
        source_bytes=source_bytes,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        pages=pages,
        decisions=(),
        raster_sha256=(),
    )


class _Detector:
    def load(self) -> None:
        pass

    # reason: this mirrors the `Detector` protocol, whose `detect` the pipeline calls on the detector instance
    # reason: it is constructed with, so the bound-method form is the API being stood in for.
    def detect(self, texts: Sequence[str]) -> list[SpanDetection]:  # ruff: ignore[no-self-use]
        return [
            SpanDetection(
                spans=(CharSpan(0, 4, "Name", "human_name"),),
                bucket=16,
                num_tokens=1,
                truncated=False,
            )
            for _ in texts
        ]


class _PassVerifier:
    # reason: this mirrors the `Verifier` protocol, whose `verify` the pipeline calls on the verifier instance
    # reason: it is constructed with, so the bound-method form is the API being stood in for.
    def verify(  # ruff: ignore[no-self-use]
        self,
        written: RedactionWriteResult,
        *,
        known_canaries: Sequence[str],
    ) -> VerificationReport:
        del known_canaries
        return VerificationReport(
            output_path=written.output_path,
            output_sha256=written.output_sha256,
            passed=True,
            findings=(VerificationFinding(gate="writer_safety", passed=True, code="destructive", detail="synthetic"),),
            observations=(),
        )


class _OutsideWriter:
    writer_id = "outside"

    def write(self, source: Path, regions: Sequence[PageRegion], destination: Path) -> RedactionWriteResult:
        del regions, destination
        return RedactionWriteResult(
            source_path=source,
            output_path=source,
            writer_id=self.writer_id,
            safety="destructive",
            page_count=1,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )


class _WrongSourceWriter:
    writer_id = "wrong-source"

    def write(self, source: Path, regions: Sequence[PageRegion], destination: Path) -> RedactionWriteResult:
        del regions
        destination.write_bytes(b"redacted")
        return RedactionWriteResult(
            source_path=source,
            output_path=destination,
            writer_id=self.writer_id,
            safety="destructive",
            page_count=1,
            source_sha256="0" * 64,
            output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )


def _pipeline(writer: RedactionWriter) -> PdfRedactionPipeline:
    return PdfRedactionPipeline(detector=_Detector(), writer=writer, verifier=_PassVerifier())


def test_preview_accepts_an_all_blank_document_without_loading_detector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")

    class NoLoadDetector:
        # reason: this mirrors the `Detector` protocol, whose `load` the pipeline calls on the detector instance
        # reason: it is constructed with; the whole point is that reaching it as a bound method fails the test.
        def load(self) -> None:  # ruff: ignore[no-self-use]
            msg = "blank pages must not load the detector"
            raise AssertionError(msg)

        # reason: this mirrors the `Detector` protocol, whose `detect` the pipeline calls on the detector
        # reason: instance it is constructed with; reaching it as a bound method is what fails the test.
        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:  # ruff: ignore[no-self-use]
            msg = f"blank pages must not detect: {texts}"
            raise AssertionError(msg)

    pipeline = PdfRedactionPipeline(detector=NoLoadDetector(), writer=_OutsideWriter(), verifier=_PassVerifier())
    blank = GeometryPage(0, 100, 100, "", ())

    preview = pipeline.preview(_prepared(source, (blank,)))

    assert preview.detections[0].spans == ()
    assert preview.suggestions == ()


def test_preview_rejects_empty_documents_and_misaligned_detector_results(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    pipeline = _pipeline(_OutsideWriter())

    with pytest.raises(ValueError, match="at least one page"):
        pipeline.preview(_prepared(source, ()))

    class MissingDetector(_Detector):
        @override
        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
            del texts
            return []

    with pytest.raises(RuntimeError, match="result count"):
        PdfRedactionPipeline(
            detector=MissingDetector(),
            writer=_OutsideWriter(),
            verifier=_PassVerifier(),
        ).preview(_prepared(source))


def test_apply_rejects_writer_staging_escape_and_cleans_private_workspace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"source")
    pipeline = _pipeline(_OutsideWriter())
    preview = pipeline.preview(_prepared(source))

    with pytest.raises(RedactionApplyError) as raised:
        pipeline.apply_verified(preview=preview, reviewed=preview.suggestions, destination=destination)

    assert raised.value.code == "writer_output_path_mismatch"
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".meddies-pdf-*"))


def test_apply_rejects_writer_source_identity_mismatch_and_cleans_workspace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"source")
    pipeline = _pipeline(_WrongSourceWriter())
    preview = pipeline.preview(_prepared(source))

    with pytest.raises(InputDocumentError) as raised:
        pipeline.apply_verified(preview=preview, reviewed=preview.suggestions, destination=destination)

    assert raised.value.code == "writer_source_identity_mismatch"
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".meddies-pdf-*"))


@pytest.mark.parametrize(
    ("destination_name", "setup", "expected_code"),
    [
        ("source.pdf", None, "in_place_output"),
        ("exists.pdf", b"occupied", "output_exists"),
        ("missing/target.pdf", None, "output_parent_missing"),
    ],
)
def test_apply_rejects_unsafe_destination_paths_before_writing(
    tmp_path: Path,
    destination_name: str,
    setup: bytes | None,
    expected_code: str,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    destination = tmp_path / destination_name
    if setup is not None:
        destination.write_bytes(setup)
    pipeline = _pipeline(_OutsideWriter())
    preview = pipeline.preview(_prepared(source))

    with pytest.raises(OutputPathError) as raised:
        pipeline.apply_verified(preview=preview, reviewed=preview.suggestions, destination=destination)

    assert raised.value.code == expected_code


def test_pipeline_verification_error_deduplicates_failure_codes() -> None:
    error = PipelineVerificationError(("scan", "scan", "writer"))

    assert error.failed_codes == ("scan", "writer")
