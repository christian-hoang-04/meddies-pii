from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import override

import pymupdf
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

from meddies_pii.pdf_redaction.contracts import PageRegion, Point, Quad
from meddies_pii.pdf_redaction.verification import (
    IndependentPdfVerifier,
    SubprocessToolRunner,
    ToolRun,
    VerificationToolchain,
)
from meddies_pii.pdf_redaction.writers import (
    PyMuPdfRedactionWriter,
    RasterRebuildWriter,
    UnsafeOverlayWriter,
)

CANARY = "SYNTHETIC-VERIFY-CANARY-73"
UNICODE_CANARY = "Tiếng Việt hồ sơ 73"


class _FakeQpdfAndTesseractRunner:
    def __init__(
        self,
        *,
        qpdf_json: bytes = b'{"objects": {}}',
        expanded_payload: bytes = b"%PDF-1.7\n% independently expanded\n",
        ocr_output: bytes = b"",
        attachment_listing_output: bytes = b"",
    ) -> None:
        self.timeouts: list[float] = []
        self.calls: list[tuple[str, ...]] = []
        self._real = SubprocessToolRunner()
        self._qpdf_json = qpdf_json
        self._expanded_payload = expanded_payload
        self._ocr_output = ocr_output
        self._attachment_listing_output = attachment_listing_output

    def run(self, args: tuple[str, ...], *, timeout_seconds: float) -> ToolRun:
        self.timeouts.append(timeout_seconds)
        self.calls.append(args)
        tool = Path(args[0]).name
        if tool == "qpdf":
            if "--qdf" in args:
                Path(args[-1]).write_bytes(self._expanded_payload)
            if "--json" in args:
                stdout = self._qpdf_json
            elif "--list-attachments" in args:
                stdout = self._attachment_listing_output
            else:
                stdout = b""
            return ToolRun(tool, "completed", 0, stdout, b"", 0.25)
        if tool == "tesseract":
            return ToolRun(tool, "completed", 0, self._ocr_output, b"", 0.5)
        return self._real.run(args, timeout_seconds=timeout_seconds)


class _MutatingRunner(_FakeQpdfAndTesseractRunner):
    def __init__(self, output: Path) -> None:
        super().__init__()
        self._output = output
        self._mutated = False

    @override
    def run(self, args: tuple[str, ...], *, timeout_seconds: float) -> ToolRun:
        result = super().run(args, timeout_seconds=timeout_seconds)
        if Path(args[0]).name == "tesseract" and not self._mutated:
            with self._output.open("ab") as stream:
                stream.write(b"\n% changed during verification\n")
            self._mutated = True
        return result


def _make_text_pdf(path: Path) -> PageRegion:
    canvas = Canvas(str(path), pagesize=letter, invariant=1)
    canvas.setFont("Helvetica", 16)
    canvas.drawString(72, 700, CANARY)
    canvas.drawString(72, 650, "safe content remains visible")
    canvas.save()

    with pymupdf.open(path) as document:
        box = document[0].search_for(CANARY)[0]
    return PageRegion(
        page_index=0,
        quad=Quad(
            points=(
                Point(box.x0, box.y0),
                Point(box.x1, box.y0),
                Point(box.x1, box.y1),
                Point(box.x0, box.y1),
            ),
        ),
        label="secret",
        span_start=0,
        span_end=len(CANARY),
    )


def test_verifier_catches_overlay_negative_control_with_real_poppler(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "unsafe.pdf"
    region = _make_text_pdf(source)
    written = UnsafeOverlayWriter().write(source, (region,), output)

    report = IndependentPdfVerifier().verify(written, known_canaries=(CANARY,))

    text_finding = report.finding("text_extraction")
    assert not text_finding.passed
    assert text_finding.code == "known_text_recoverable"
    assert text_finding.matched_value_sha256
    assert report.finding("writer_safety").code == "unsafe_writer"
    if shutil.which("qpdf") is None:
        assert report.finding("structure").code == "tool_unavailable"
    assert not report.passed


def test_verifier_orchestrates_every_gate_with_bounded_tool_calls(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    runner = _FakeQpdfAndTesseractRunner()
    verifier = IndependentPdfVerifier(
        toolchain=VerificationToolchain(timeout_seconds=2.5, render_dpi=72),
        runner=runner,
    )

    report = verifier.verify(written, known_canaries=(CANARY,))

    assert report.passed
    assert {finding.gate for finding in report.findings} == {
        "writer_safety",
        "fresh_output",
        "structure",
        "expanded_objects",
        "risky_objects",
        "text_extraction",
        "reopen_render",
        "ocr",
    }
    assert runner.timeouts
    assert set(runner.timeouts) == {2.5}


def test_verifier_rejects_output_changed_during_tool_checks(tmp_path: Path) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)

    report = IndependentPdfVerifier(runner=_MutatingRunner(output)).verify(
        written,
        known_canaries=(CANARY,),
    )

    assert not report.passed
    assert report.finding("fresh_output").code == "output_identity_mismatch"
    assert report.output_sha256 != written.output_sha256


def test_subprocess_adapter_terminates_tools_at_the_timeout() -> None:
    run = SubprocessToolRunner().run(
        (sys.executable, "-c", "import time; time.sleep(1)"),
        timeout_seconds=0.01,
    )

    assert run.status == "timeout"
    assert run.returncode is None
    assert run.duration_ms < 500


def test_verifier_exposes_risky_object_cause_without_raw_canary(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    runner = _FakeQpdfAndTesseractRunner(
        qpdf_json=b'{"objects": {"1 0 R": {"/OpenAction": {}}}}',
    )

    report = IndependentPdfVerifier(runner=runner).verify(
        written,
        known_canaries=(CANARY,),
    )

    risk = report.finding("risky_objects")
    assert not risk.passed
    assert risk.code == "risky_objects_present"
    assert risk.risky_markers == ("/OpenAction",)
    assert CANARY not in repr(report)


def test_attachment_listing_banner_does_not_false_fail_clean_json(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    runner = _FakeQpdfAndTesseractRunner(
        qpdf_json=b'{"attachments": {}, "objects": {}}',
        attachment_listing_output=b"qpdf 11 attachment inventory: none\n",
    )

    report = IndependentPdfVerifier(runner=runner).verify(
        written,
        known_canaries=(CANARY,),
    )

    assert report.finding("risky_objects").passed
    assert not any("--list-attachments" in call for call in runner.calls)
    assert sum("--json" in call for call in runner.calls) == 1


@pytest.mark.parametrize(
    "attachments",
    [
        {"synthetic-private-note.txt": {"filespec": "12 0 R"}},
        [{"filespec": "12 0 R"}],
    ],
    ids=("qpdf-mapping", "conservative-list-variant"),
)
def test_nonempty_json_attachment_inventory_fails_without_exposing_values(
    tmp_path: Path,
    attachments: object,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    raw_name = "synthetic-private-note.txt"
    qpdf_json = json.dumps({"attachments": attachments, "objects": {}}).encode()

    report = IndependentPdfVerifier(
        runner=_FakeQpdfAndTesseractRunner(qpdf_json=qpdf_json),
    ).verify(written, known_canaries=(CANARY,))

    risk = report.finding("risky_objects")
    assert not risk.passed
    assert risk.code == "risky_objects_present"
    assert raw_name not in repr(report)


def test_verifier_hashes_object_and_ocr_leaks_instead_of_returning_values(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    encoded = CANARY.encode()
    runner = _FakeQpdfAndTesseractRunner(
        expanded_payload=b"%PDF-1.7\n" + encoded,
        ocr_output=encoded,
    )

    report = IndependentPdfVerifier(runner=runner).verify(
        written,
        known_canaries=(CANARY,),
    )

    assert report.finding("expanded_objects").code == "known_text_in_expanded_objects"
    assert report.finding("ocr").code == "known_text_visible_to_ocr"
    assert report.finding("expanded_objects").matched_value_sha256
    assert report.finding("ocr").matched_value_sha256
    assert CANARY not in repr(report)


def test_verifier_detects_canonically_equivalent_unicode_ocr_leak(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    decomposed = unicodedata.normalize("NFD", UNICODE_CANARY).encode()

    report = IndependentPdfVerifier(
        runner=_FakeQpdfAndTesseractRunner(ocr_output=decomposed),
    ).verify(written, known_canaries=(UNICODE_CANARY,))

    finding = report.finding("ocr")
    assert not finding.passed
    assert finding.code == "known_text_visible_to_ocr"
    assert UNICODE_CANARY not in repr(report)


def test_real_poppler_finds_no_extractable_text_in_raster_rebuild(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "raster.pdf"
    region = _make_text_pdf(source)
    written = RasterRebuildWriter(dpi=90).write(source, (region,), output)

    report = IndependentPdfVerifier().verify(written, known_canaries=(CANARY,))

    assert report.finding("text_extraction").passed
    assert report.finding("reopen_render").passed


def test_verifier_finds_qpdf_unicode_string_values_without_exposing_them(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    qpdf_json = ('{"objects":{"1 0 R":{"value":"u:' + UNICODE_CANARY + '"}}}').encode()

    report = IndependentPdfVerifier(
        runner=_FakeQpdfAndTesseractRunner(qpdf_json=qpdf_json),
    ).verify(written, known_canaries=(UNICODE_CANARY,))

    risk = report.finding("risky_objects")
    assert not risk.passed
    assert risk.code == "known_text_in_object_inventory"
    assert risk.matched_value_sha256
    assert UNICODE_CANARY not in repr(report)


def test_verifier_decodes_utf16_qpdf_binary_strings_without_exposing_them(
    tmp_path: Path,
) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    encoded = (b"\xfe\xff" + UNICODE_CANARY.encode("utf-16-be")).hex().upper()
    qpdf_json = ('{"objects":{"1 0 R":{"value":"b:' + encoded + '"}}}').encode()

    report = IndependentPdfVerifier(
        runner=_FakeQpdfAndTesseractRunner(qpdf_json=qpdf_json),
    ).verify(written, known_canaries=(UNICODE_CANARY,))

    risk = report.finding("risky_objects")
    assert not risk.passed
    assert risk.code == "known_text_in_object_inventory"
    assert risk.matched_value_sha256
    assert UNICODE_CANARY not in repr(report)


def test_verifier_decodes_utf8_bom_qpdf_binary_strings(tmp_path: Path) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)
    encoded = (b"\xef\xbb\xbf" + UNICODE_CANARY.encode()).hex().upper()
    qpdf_json = ('{"objects":{"1 0 R":{"value":"b:' + encoded + '"}}}').encode()

    report = IndependentPdfVerifier(
        runner=_FakeQpdfAndTesseractRunner(qpdf_json=qpdf_json),
    ).verify(written, known_canaries=(UNICODE_CANARY,))

    risk = report.finding("risky_objects")
    assert not risk.passed
    assert risk.code == "known_text_in_object_inventory"
    assert risk.matched_value_sha256
    assert UNICODE_CANARY not in repr(report)


def test_verifier_fails_closed_on_invalid_qpdf_json(tmp_path: Path) -> None:
    if shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is required for the real verifier integration test")
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    written = PyMuPdfRedactionWriter().write(source, (region,), output)

    report = IndependentPdfVerifier(
        runner=_FakeQpdfAndTesseractRunner(qpdf_json=b'{"objects":'),
    ).verify(written, known_canaries=(CANARY,))

    risk = report.finding("risky_objects")
    assert not risk.passed
    assert risk.code == "object_inventory_invalid_json"
