from __future__ import annotations

# ruff: file-ignore[no-self-use]
# reason: stateless detector doubles retain the bound load and detect method shape required by the pipeline protocol.
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest

from meddies_pii.bioes_inference import SpanDetection
from meddies_pii.pdf_redaction.benchmark.corpus import generate_challenge_corpus
from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    GeometryToken,
    PageRegion,
    Point,
    Quad,
    ReviewedRedaction,
)
from meddies_pii.pdf_redaction.document import (
    DocumentInspection,
    PymupdfDocumentAdapter,
    RasterArtifact,
)
from meddies_pii.pdf_redaction.document import (
    PdfSource as DocumentPdfSource,
)
from meddies_pii.pdf_redaction.errors import InputDocumentError, OutputPathError
from meddies_pii.pdf_redaction.extraction import (
    NativeGeometryPrecision,
    PymupdfExtractor,
)
from meddies_pii.pdf_redaction.extraction import (
    PdfSource as NativePdfSource,
)
from meddies_pii.pdf_redaction.pipeline import (
    DEFAULT_MAX_EXTRACTED_TEXT_CHARS,
    ExtractedTextLimitError,
    PdfGeometryPreparer,
    PdfRedactionPipeline,
    PipelineVerificationError,
    PreparedDocument,
)
from meddies_pii.pdf_redaction.verification import (
    VerificationFinding,
    VerificationReport,
)
from meddies_pii.pdf_redaction.writers import RedactionWriteResult
from meddies_pii.regex_runtime import regex_manifest
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.pdf_redaction.ocr import RasterPage


def _quad(x0: float, y0: float, x1: float, y1: float) -> Quad:
    return Quad(
        points=(
            Point(x0, y0),
            Point(x1, y0),
            Point(x1, y1),
            Point(x0, y1),
        ),
    )


def _page() -> GeometryPage:
    text = "Patient Maya Sato"
    return GeometryPage(
        page_index=0,
        width_pt=612,
        height_pt=792,
        text=text,
        tokens=(
            GeometryToken(
                text="Maya",
                quad=_quad(80, 100, 110, 112),
                start=8,
                end=12,
                source="native",
                confidence=None,
            ),
            GeometryToken(
                text="Sato",
                quad=_quad(114, 100, 142, 112),
                start=13,
                end=17,
                source="native",
                confidence=None,
            ),
        ),
    )


def _blank_page(page_index: int = 1) -> GeometryPage:
    return GeometryPage(
        page_index=page_index,
        width_pt=612,
        height_pt=792,
        text="",
        tokens=(),
    )


def _page_with_word_geometry(text: str) -> GeometryPage:
    tokens: list[GeometryToken] = []
    cursor = 0
    for index, word in enumerate(text.split()):
        start = text.index(word, cursor)
        end = start + len(word)
        left = 20.0 + index * 80.0
        tokens.append(
            GeometryToken(
                text=word,
                quad=_quad(left, 100.0, left + 72.0, 112.0),
                start=start,
                end=end,
                source="native",
                confidence=None,
            ),
        )
        cursor = end
    return GeometryPage(
        page_index=0,
        width_pt=max(1000.0, 20.0 + len(tokens) * 80.0),
        height_pt=792.0,
        text=text,
        tokens=tuple(tokens),
    )


def _prepared(source: Path, *pages: GeometryPage) -> PreparedDocument:
    return PreparedDocument(
        source_path=source,
        source_bytes=source.read_bytes(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pages=pages,
        decisions=(),
        raster_sha256=(),
    )


def test_pipeline_rejects_extracted_text_above_the_ceiling_before_detection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    oversized_page = replace(_blank_page(0), text="x" * (DEFAULT_MAX_EXTRACTED_TEXT_CHARS + 1))
    pipeline = PdfRedactionPipeline(detector=_Detector(), writer=_Writer(), verifier=_Verifier(passes=True))

    with pytest.raises(ExtractedTextLimitError) as raised:
        pipeline.preview(_prepared(source, oversized_page))

    assert raised.value.code == "extracted_text_too_large"


def test_pipeline_accepts_extracted_text_at_the_ceiling(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    page_at_limit = replace(_blank_page(0), text=" " * DEFAULT_MAX_EXTRACTED_TEXT_CHARS)
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
        regex_postprocess=False,
    )

    preview = pipeline.preview(_prepared(source, page_at_limit))

    assert preview.pages == (page_at_limit,)


class _Detector:
    def __init__(self, *, truncated: bool = False) -> None:
        self.loaded = 0
        self.truncated = truncated

    def load(self) -> None:
        self.loaded += 1

    def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
        assert texts == ["Patient Maya Sato"]
        return [
            SpanDetection(
                spans=(
                    CharSpan(
                        start=8,
                        end=17,
                        text="Maya Sato",
                        label="human_name",
                    ),
                ),
                bucket=512,
                num_tokens=4,
                truncated=self.truncated,
            ),
        ]


class _Writer:
    writer_id = "fake-destructive"

    def __init__(self) -> None:
        self.regions = ()

    def write(
        self,
        source: Path,
        regions: Sequence[PageRegion],
        destination: Path,
    ) -> RedactionWriteResult:
        self.regions = tuple(regions)
        destination.write_bytes(b"%PDF-1.7\n% redacted\n")
        return RedactionWriteResult(
            source_path=source,
            output_path=destination,
            writer_id=self.writer_id,
            safety="destructive",
            page_count=1,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )


class _Verifier:
    def __init__(self, *, passes: bool) -> None:
        self.passes = passes
        self.known_values: tuple[str, ...] = ()

    def verify(
        self,
        written: RedactionWriteResult,
        *,
        known_canaries: Sequence[str],
    ) -> VerificationReport:
        self.known_values = tuple(known_canaries)
        finding = VerificationFinding(
            gate="writer_safety",
            passed=self.passes,
            code="destructive_writer" if self.passes else "unsafe_writer",
            detail="synthetic verifier result",
        )
        return VerificationReport(
            output_path=written.output_path,
            output_sha256=written.output_sha256,
            passed=self.passes,
            findings=(finding,),
            observations=(),
        )


def test_pipeline_maps_detected_text_and_returns_only_verified_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    detector = _Detector()
    writer = _Writer()
    verifier = _Verifier(passes=True)
    pipeline = PdfRedactionPipeline(
        detector=detector,
        writer=writer,
        verifier=verifier,
    )

    preview = pipeline.preview(_prepared(source, _page()))
    output = pipeline.apply_verified(
        preview=preview,
        reviewed=preview.suggestions,
        destination=destination,
    )

    assert detector.loaded == 1
    assert len(preview.suggestions) == 1
    assert len(preview.suggestions[0].to_regions()) == 2
    assert writer.regions == preview.suggestions[0].to_regions()
    assert verifier.known_values == ("Maya Sato",)
    assert output.path == destination
    assert output.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert output.writer_id == "fake-destructive"
    assert output.verification_scope == "reviewed_values"


def test_pipeline_detects_vietnamese_and_suggests_hospital_without_model_spans(
    tmp_path: Path,
) -> None:
    text = "Chuyển đến Bệnh viện Đa khoa Sông Xanh để điều trị."
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    class EmptyDetector:
        def load(self) -> None:
            return None

        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
            assert texts == [text]
            return [SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False)]

    writer = _Writer()
    pipeline = PdfRedactionPipeline(
        detector=EmptyDetector(),
        writer=writer,
        verifier=_Verifier(passes=True),
    )

    page = _page_with_word_geometry(text)
    preview = pipeline.preview(_prepared(source, page))
    output = pipeline.apply_verified(
        preview=preview,
        reviewed=preview.suggestions,
        destination=destination,
    )

    assert [(item.value, item.label) for item in preview.suggestions] == [("Bệnh viện Đa khoa Sông Xanh", "company_name")]
    hospital_start = text.index("Bệnh viện")
    hospital_end = hospital_start + len("Bệnh viện Đa khoa Sông Xanh")
    hospital_tokens = tuple(
        token for token in page.tokens if max(token.start, hospital_start) < min(token.end, hospital_end)
    )
    assert len(writer.regions) == len(hospital_tokens)
    assert all((region.span_start, region.span_end) == (hospital_start, hospital_end) for region in writer.regions)
    assert preview.regex_language == "vi"
    assert preview.regex_language_source == "detected"
    assert output.regex_language == "vi"
    assert output.regex_language_source == "detected"


def test_pipeline_regex_escape_hatch_preserves_model_only_detections(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    writer = _Writer()
    verifier = _Verifier(passes=True)
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=writer,
        verifier=verifier,
        regex_postprocess=False,
    )

    preview = pipeline.preview(_prepared(source, _page()))
    output = pipeline.apply_verified(
        preview=preview,
        reviewed=preview.suggestions,
        destination=destination,
    )

    expected = SpanDetection(
        spans=(CharSpan(8, 17, "Maya Sato", "human_name"),),
        bucket=512,
        num_tokens=4,
        truncated=False,
    )
    assert preview.detections == (expected,)
    assert [(item.value, item.label) for item in preview.suggestions] == [("Maya Sato", "human_name")]
    assert writer.regions == preview.suggestions[0].to_regions()
    assert verifier.known_values == ("Maya Sato",)
    assert destination.read_bytes() == b"%PDF-1.7\n% redacted\n"
    assert output.sha256 == "f006a4051e9219f2968df91aa23946afe9c08435b8edc990705cdd2e7100f878"
    assert preview.regex_manifest_sha256 is None
    assert preview.regex_language is None
    assert output.regex_manifest_sha256 is None
    assert output.regex_language is None


def test_pipeline_detects_english_without_masking_hospital_problem_list(
    tmp_path: Path,
) -> None:
    text = "Hospital Problem List\nThe patient is stable and ready for the next stage of treatment."
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    class EmptyDetector:
        def load(self) -> None:
            return None

        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
            assert texts == [text]
            return [SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False)]

    preview = PdfRedactionPipeline(
        detector=EmptyDetector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
    ).preview(_prepared(source, _page_with_word_geometry(text)))

    assert preview.suggestions == ()
    assert preview.regex_language == "en"
    assert preview.regex_language_source == "detected"


def test_pipeline_detects_one_language_from_the_full_document(tmp_path: Path) -> None:
    expected_texts = ("The patient is stable.", "Continue with care and follow-up.")
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    class EmptyDetector:
        def load(self) -> None:
            return None

        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
            assert texts == list(expected_texts)
            return [
                SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False),
                SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False),
            ]

    pages = (
        _page_with_word_geometry(expected_texts[0]),
        replace(_page_with_word_geometry(expected_texts[1]), page_index=1),
    )
    preview = PdfRedactionPipeline(
        detector=EmptyDetector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
    ).preview(_prepared(source, *pages))

    assert preview.regex_language == "en"
    assert preview.regex_language_source == "detected"


def test_pipeline_fails_closed_for_unknown_language_organization_packs(
    tmp_path: Path,
) -> None:
    text = (
        "Diagnosis: Hospital Santa Aurora. Contact alpha@example.invalid; Portal: "
        "https://portal.example.invalid/case?access_token=Abc12345. "
        "API key: sk_test_Abc12345."
    )
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    class EmptyDetector:
        def load(self) -> None:
            return None

        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
            assert texts == [text]
            return [SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False)]

    english_preview = PdfRedactionPipeline(
        detector=EmptyDetector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
        language="en",
    ).preview(_prepared(source, _page_with_word_geometry(text)))
    unknown_language_preview = PdfRedactionPipeline(
        detector=EmptyDetector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
    ).preview(_prepared(source, _page_with_word_geometry(text)))

    authoritative_suggestions = [
        ("alpha@example.invalid", "email_address"),
        (
            "https://portal.example.invalid/case?access_token=Abc12345",
            "private_url",
        ),
        ("sk_test_Abc12345", "secret"),
    ]
    assert [(item.value, item.label) for item in english_preview.suggestions] == authoritative_suggestions
    assert [(item.value, item.label) for item in unknown_language_preview.suggestions] == authoritative_suggestions


def test_pipeline_warns_and_fails_closed_for_an_undecidable_short_fragment(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    text = "Bệnh viện X."
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    class EmptyDetector:
        def load(self) -> None:
            return None

        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
            assert texts == [text]
            return [SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False)]

    preview = PdfRedactionPipeline(
        detector=EmptyDetector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
    ).preview(_prepared(source, _page_with_word_geometry(text)))

    assert preview.suggestions == ()
    assert preview.regex_language is None
    assert preview.regex_language_source == "undetected"
    assert "organization regex packs remain disabled" in caplog.text


def test_caller_language_bypasses_document_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Chuyển đến Bệnh viện Đa khoa Sông Xanh để điều trị."
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    class EmptyDetector:
        def load(self) -> None:
            return None

        def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
            assert texts == [text]
            return [SpanDetection(spans=(), bucket=0, num_tokens=0, truncated=False)]

    def fail_if_called(_: str) -> str | None:
        msg = "caller language must bypass detection"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "meddies_pii.regex_runtime.detect_language",
        fail_if_called,
    )
    preview = PdfRedactionPipeline(
        detector=EmptyDetector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
        language=" Vietnamese ",
    ).preview(_prepared(source, _page_with_word_geometry(text)))

    assert [(item.value, item.label) for item in preview.suggestions] == [("Bệnh viện Đa khoa Sông Xanh", "company_name")]
    assert preview.regex_language == "vi"
    assert preview.regex_language_source == "caller"


@pytest.mark.parametrize("language", ["", "vi,en", "Klingon"])
def test_pipeline_rejects_noncanonicalizable_language(language: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Meddies PII language"):
        PdfRedactionPipeline(
            detector=_Detector(),
            writer=_Writer(),
            verifier=_Verifier(passes=True),
            language=language,
        )


def test_pipeline_records_the_regex_manifest_on_preview_and_verified_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
    )

    preview = pipeline.preview(_prepared(source, _page()))
    output = pipeline.apply_verified(
        preview=preview,
        reviewed=preview.suggestions,
        destination=destination,
    )

    expected_manifest = str(regex_manifest()["sha256"])
    assert preview.regex_manifest_sha256 == expected_manifest
    assert output.regex_manifest_sha256 == expected_manifest
    assert preview.regex_language is None
    assert output.regex_language is None


def test_pipeline_writes_and_verifies_the_same_manual_review_collection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    writer = _Writer()
    verifier = _Verifier(passes=True)
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=writer,
        verifier=verifier,
    )
    preview = pipeline.preview(_prepared(source, _page()))
    manual = ReviewedRedaction(
        value="Manual addition",
        label="company_name",
        page_index=0,
        quads=(
            _quad(200, 100, 260, 112),
            _quad(200, 116, 245, 128),
        ),
        span_start=0,
        span_end=6,
    )
    reviewed = (preview.suggestions[0], manual)

    pipeline.apply_verified(
        preview=preview,
        reviewed=reviewed,
        destination=destination,
    )

    assert writer.regions == tuple(region for item in reviewed for region in item.to_regions())
    assert verifier.known_values == ("Maya Sato", "Manual addition")
    assert len(manual.to_regions()) == 2


def test_pipeline_keeps_output_private_until_all_gates_pass(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    class InspectingVerifier(_Verifier):
        @override
        def verify(
            self,
            written: RedactionWriteResult,
            *,
            known_canaries: Sequence[str],
        ) -> VerificationReport:
            assert not destination.exists()
            assert written.output_path != destination
            if os.name == "posix":
                assert written.output_path.stat().st_mode & 0o777 == 0o600
            return super().verify(written, known_canaries=known_canaries)

    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=_Writer(),
        verifier=InspectingVerifier(passes=True),
    )
    preview = pipeline.preview(_prepared(source, _page()))

    pipeline.apply_verified(
        preview=preview,
        reviewed=preview.suggestions,
        destination=destination,
    )

    assert destination.exists()
    if os.name == "posix":
        assert destination.stat().st_mode & 0o777 == 0o600
    assert not tuple(tmp_path.glob(".meddies-pdf-*"))


def test_pipeline_rejects_caller_destination_unsafe_clean_unsafe_aba(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    unsafe = b"%PDF-1.7\n% unsafe\n"

    class DestinationAbaVerifier(_Verifier):
        @override
        def verify(
            self,
            written: RedactionWriteResult,
            *,
            known_canaries: Sequence[str],
        ) -> VerificationReport:
            assert not destination.exists()
            destination.write_bytes(unsafe)
            destination.write_bytes(written.output_path.read_bytes())
            destination.write_bytes(unsafe)
            return super().verify(written, known_canaries=known_canaries)

    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=_Writer(),
        verifier=DestinationAbaVerifier(passes=True),
    )
    preview = pipeline.preview(_prepared(source, _page()))

    with pytest.raises(OutputPathError) as raised:
        pipeline.apply_verified(
            preview=preview,
            reviewed=preview.suggestions,
            destination=destination,
        )

    assert raised.value.code == "output_exists"
    assert destination.read_bytes() == unsafe
    assert not tuple(tmp_path.glob(".meddies-pdf-*"))


def test_pipeline_writes_snapshot_a_when_original_path_performs_aba(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source_a = b"%PDF-1.7\n% source A\n"
    source_b = b"%PDF-1.7\n% source B\n"
    source.write_bytes(source_a)

    class SourceAbaWriter(_Writer):
        @override
        def write(
            self,
            source: Path,
            regions: Sequence[PageRegion],
            destination: Path,
        ) -> RedactionWriteResult:
            assert source != outer_source
            assert source.read_bytes() == source_a
            outer_source.write_bytes(source_b)
            outer_source.write_bytes(source_a)
            self.regions = tuple(regions)
            destination.write_bytes(source.read_bytes())
            return RedactionWriteResult(
                source_path=source,
                output_path=destination,
                writer_id=self.writer_id,
                safety="destructive",
                page_count=1,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            )

    outer_source = source
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=SourceAbaWriter(),
        verifier=_Verifier(passes=True),
    )
    preview = pipeline.preview(_prepared(source, _page()))

    pipeline.apply_verified(
        preview=preview,
        reviewed=preview.suggestions,
        destination=destination,
    )

    assert destination.read_bytes() == source_a
    assert source.read_bytes() == source_a


def test_pipeline_fails_closed_on_truncation_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    detector = _Detector(truncated=True)
    writer = _Writer()
    pipeline = PdfRedactionPipeline(
        detector=detector,
        writer=writer,
        verifier=_Verifier(passes=True),
    )

    with pytest.raises(RuntimeError, match="detector_truncated"):
        pipeline.preview(_prepared(source, _page()))

    assert writer.regions == ()


def test_pipeline_deletes_unverified_output_and_preserves_hash_only_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=_Writer(),
        verifier=_Verifier(passes=False),
    )
    preview = pipeline.preview(_prepared(source, _page()))

    with pytest.raises(PipelineVerificationError) as raised:
        pipeline.apply_verified(
            preview=preview,
            reviewed=preview.suggestions,
            destination=destination,
        )

    assert raised.value.failed_codes == ("unsafe_writer",)
    assert not destination.exists()
    assert "Maya Sato" not in repr(raised.value)


def test_pipeline_skips_blank_pages_without_changing_detection_alignment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=_Writer(),
        verifier=_Verifier(passes=True),
    )

    preview = pipeline.preview(_prepared(source, _page(), _blank_page()))

    assert len(preview.detections) == 2
    assert preview.detections[1].spans == ()
    assert preview.detections[1].num_tokens == 0


def test_pipeline_rejects_source_changed_after_preview(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source A\n")
    writer = _Writer()
    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=writer,
        verifier=_Verifier(passes=True),
    )
    preview = pipeline.preview(_prepared(source, _page()))
    source.write_bytes(b"%PDF-1.7\n% source B\n")

    with pytest.raises(InputDocumentError) as raised:
        pipeline.apply_verified(
            preview=preview,
            reviewed=preview.suggestions,
            destination=destination,
        )

    assert raised.value.code == "source_identity_changed"
    assert writer.regions == ()
    assert not destination.exists()


def test_pipeline_deletes_output_when_verifier_raises(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "redacted.pdf"
    source.write_bytes(b"%PDF-1.7\n% source\n")

    # reason: this double never reaches its parameters — it raises on entry, which is the point of
    # reason: the test. Both names are contract anyway: `pipeline.py:244-246` calls
    # reason: `self._verifier.verify(..., known_canaries=known_values)`, so a double that dropped or
    # reason: renamed `known_canaries` would stop being callable by the code under test.
    class RaisingVerifier:
        def verify(
            self,
            written: RedactionWriteResult,  # ruff: ignore[unused-method-argument]
            *,
            known_canaries: Sequence[str],  # ruff: ignore[unused-method-argument]
        ) -> VerificationReport:
            msg = "verifier unavailable"
            raise RuntimeError(msg)

    pipeline = PdfRedactionPipeline(
        detector=_Detector(),
        writer=_Writer(),
        verifier=RaisingVerifier(),
    )
    preview = pipeline.preview(_prepared(source, _page()))

    with pytest.raises(RuntimeError, match="verifier unavailable"):
        pipeline.apply_verified(
            preview=preview,
            reviewed=preview.suggestions,
            destination=destination,
        )

    assert not destination.exists()


def test_geometry_preparer_uses_native_only_on_trusted_pages_and_one_ocr_pass(
    tmp_path: Path,
) -> None:
    source = tmp_path / "challenge.pdf"
    fixture = generate_challenge_corpus()
    source.write_bytes(fixture.pdf_bytes)

    class Ocr:
        def __init__(self) -> None:
            self.page_indices: list[int] = []

        def extract(self, raster: RasterPage) -> GeometryPage:
            self.page_indices.append(raster.page_index)
            return GeometryPage(
                page_index=raster.page_index,
                width_pt=raster.width_pt,
                height_pt=raster.height_pt,
                text="Raster",
                tokens=(
                    GeometryToken(
                        text="Raster",
                        quad=_quad(10, 10, 50, 22),
                        start=0,
                        end=6,
                        source="ocr",
                        confidence=0.9,
                    ),
                ),
            )

    ocr = Ocr()
    prepared = PdfGeometryPreparer(
        document_adapter=PymupdfDocumentAdapter(dpi=72),
        native_extractor=PymupdfExtractor(),
        ocr=ocr,
    ).prepare(source)

    assert tuple(page.page_index for page in prepared.pages) == tuple(range(10))
    assert ocr.page_indices == [4, 5, 6, 7, 8, 9]
    assert all(token.source == "native" for page in prepared.pages[:4] for token in page.tokens)
    assert all(token.source == "ocr" for page in prepared.pages[4:] for token in page.tokens)
    assert len(prepared.raster_sha256) == 6
    assert not any(canary.raw_value in repr(prepared) for canary in fixture.gold.canaries)


def test_geometry_preparer_uses_one_private_snapshot_through_source_aba(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source_a = b"%PDF-1.7\n% source A\n"
    source_b = b"%PDF-1.7\n% source B\n"
    source.write_bytes(source_a)
    original = source
    observed_paths: list[Path] = []

    # reason: every method below is reached through attribute lookup on an INSTANCE —
    # reason: `self._document_adapter.inspect(...)`, `.render_pages(...)`, `self._ocr.extract(...)`
    # reason: at `pipeline.py:107-118` — so a staticmethod would not be reachable and `self` is
    # reason: contract even where the body ignores it. The unused parameters absorb the real call
    # reason: shapes for the same reason.
    class MutatingAdapter:
        def inspect(self, source: DocumentPdfSource) -> DocumentInspection:
            assert isinstance(source, Path)
            snapshot = source
            observed_paths.append(snapshot)
            assert snapshot != original
            assert snapshot.read_bytes() == source_a
            if os.name == "posix":
                assert snapshot.parent.stat().st_mode & 0o777 == 0o700
                assert snapshot.stat().st_mode & 0o777 == 0o600
            original.write_bytes(source_b)
            return DocumentInspection(pages=(), attachment_count=0)

        def render_pages(self, source: DocumentPdfSource, *, page_indices: Sequence[int]) -> tuple[RasterArtifact, ...]:
            assert isinstance(source, Path)
            snapshot = source
            observed_paths.append(snapshot)
            assert snapshot.read_bytes() == source_a
            assert not page_indices
            return ()

    class RestoringExtractor:
        geometry_precision: NativeGeometryPrecision = "character_quad"

        def extract(self, source: NativePdfSource) -> tuple[GeometryPage, ...]:
            snapshot = Path(source)
            observed_paths.append(snapshot)
            assert snapshot.read_bytes() == source_a
            original.write_bytes(source_a)
            return ()

    class NoOcr:
        def extract(self, raster: RasterPage) -> GeometryPage:  # ruff: ignore[unused-method-argument]
            msg = "empty document must not OCR"
            raise AssertionError(msg)

    prepared = PdfGeometryPreparer(
        document_adapter=MutatingAdapter(),
        native_extractor=RestoringExtractor(),
        ocr=NoOcr(),
    ).prepare(source)

    assert prepared.source_bytes == source_a
    assert prepared.source_sha256 == hashlib.sha256(source_a).hexdigest()
    assert len(set(observed_paths)) == 1
    assert not observed_paths[0].exists()
    assert source.read_bytes() == source_a
    assert source_a.decode() not in repr(prepared)


def test_geometry_preparer_rejects_source_over_configured_size_before_adapters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"123456789")

    class UnusedAdapter:
        def inspect(self, source: DocumentPdfSource) -> DocumentInspection:  # ruff: ignore[unused-method-argument]
            msg = "oversized input reached the document adapter"
            raise AssertionError(msg)

        def render_pages(self, source: DocumentPdfSource, *, page_indices: Sequence[int]) -> tuple[RasterArtifact, ...]:  # ruff: ignore[unused-method-argument]
            msg = "oversized input reached the document adapter"
            raise AssertionError(msg)

    class UnusedExtractor:
        geometry_precision: NativeGeometryPrecision = "character_quad"

        def extract(self, source: NativePdfSource) -> tuple[GeometryPage, ...]:  # ruff: ignore[unused-method-argument]
            msg = "oversized input reached the native extractor"
            raise AssertionError(msg)

    class UnusedOcr:
        def extract(self, raster: RasterPage) -> GeometryPage:  # ruff: ignore[unused-method-argument]
            msg = "oversized input reached OCR"
            raise AssertionError(msg)

    with pytest.raises(InputDocumentError) as raised:
        PdfGeometryPreparer(
            document_adapter=UnusedAdapter(),
            native_extractor=UnusedExtractor(),
            ocr=UnusedOcr(),
            max_source_bytes=8,
        ).prepare(source)

    assert raised.value.code == "source_too_large"
    assert repr(raised.value) == "InputDocumentError('PDF input exceeds size limit')"
