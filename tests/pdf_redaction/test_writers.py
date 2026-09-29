from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import pymupdf
import pypdfium2 as pdfium
import pytest
from PIL.Image import Image
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

import meddies_pii.pdf_redaction.writer_overlay as writer_overlay_module
import meddies_pii.pdf_redaction.writer_pymupdf as writer_pymupdf_module
import meddies_pii.pdf_redaction.writer_raster as writer_raster_module
from meddies_pii.pdf_redaction.contracts import PageRegion, Point, Quad
from meddies_pii.pdf_redaction.errors import (
    InputDocumentError,
    OutputPathError,
    RedactionApplyError,
)
from meddies_pii.pdf_redaction.writers import (
    PyMuPdfRedactionWriter,
    RasterRebuildWriter,
    UnsafeOverlayWriter,
)

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.pdf_redaction.pipeline import RedactionWriter

CANARY = "SYNTHETIC-CANARY-42"


def _make_text_pdf(path: Path) -> PageRegion:
    canvas = Canvas(str(path), pagesize=letter, invariant=1)
    canvas.setFont("Helvetica", 16)
    canvas.drawString(72, 700, CANARY)
    canvas.drawString(72, 650, "ordinary clinical text")
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


def _make_replacement_pdf(path: Path) -> None:
    canvas = Canvas(str(path), pagesize=letter, invariant=1)
    canvas.setFont("Helvetica", 16)
    canvas.drawString(72, 700, "replacement record")
    canvas.save()


def _make_rotated_text_pdf(path: Path, base: Path) -> PageRegion:
    canvas = Canvas(str(base), pagesize=letter, invariant=1)
    canvas.setFont("Helvetica", 16)
    canvas.drawString(72, 100, CANARY)
    canvas.save()
    writer = PdfWriter(clone_from=base)
    writer.pages[0].rotate(90)
    writer.write(path)

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


def test_pymupdf_writer_removes_covered_text_and_paints_black(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)

    result = PyMuPdfRedactionWriter().write(source, (region,), output)

    assert result.output_path == output
    assert result.safety == "destructive"
    assert result.verification_state == "unverified"
    assert CANARY not in (PdfReader(output).pages[0].extract_text() or "")
    assert "ordinary clinical text" in (PdfReader(output).pages[0].extract_text() or "")

    document = pdfium.PdfDocument(output)
    image = document[0].render(scale=2).to_pil()
    box = region.quad.bounds()
    center = (round(box.x0 + box.x1), round(box.y0 + box.y1))
    red, green, blue = image.convert("RGB").getpixel(center)
    assert max(red, green, blue) < 20


def test_raster_writer_discards_source_objects_and_paints_black(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "raster-redacted.pdf"
    region = _make_text_pdf(source)

    result = RasterRebuildWriter(dpi=90).write(source, (region,), output)

    assert result.safety == "destructive"
    assert result.verification_state == "unverified"
    output_page = PdfReader(output).pages[0]
    assert output_page.extract_text() in {None, ""}
    embedded_image = output_page.images[0].image
    if not isinstance(embedded_image, Image):
        msg = "raster reconstruction must embed an image"
        raise TypeError(msg)
    assert embedded_image.size == (765, 990)
    assert CANARY.encode() not in output.read_bytes()

    document = pdfium.PdfDocument(output)
    assert document[0].get_size() == pdfium.PdfDocument(source)[0].get_size()
    image = document[0].render(scale=2).to_pil()
    box = region.quad.bounds()
    center = (round(box.x0 + box.x1), round(box.y0 + box.y1))
    red, green, blue = image.convert("RGB").getpixel(center)
    assert max(red, green, blue) < 20


def test_raster_writer_never_promotes_output_before_identity_is_known(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "raster-redacted.pdf"
    region = _make_text_pdf(source)
    real_sha256 = writer_raster_module.file_sha256

    def fail_temporary_identity(path: Path) -> str:
        if path.parent.name.startswith(".meddies-pdf-"):
            msg = "synthetic identity failure"
            raise OSError(msg)
        return real_sha256(path)

    monkeypatch.setattr(
        writer_raster_module,
        "file_sha256",
        fail_temporary_identity,
    )

    with pytest.raises(RedactionApplyError) as error:
        RasterRebuildWriter(dpi=90).write(source, (region,), output)

    assert error.value.code == "raster_rebuild_failed"
    assert not output.exists()
    assert not tuple(tmp_path.glob(".meddies-pdf-*.working"))


def test_raster_product_writer_has_no_unredacted_reference_capability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    empty_redaction = tmp_path / "empty-redaction.pdf"
    _make_text_pdf(source)
    writer = RasterRebuildWriter(dpi=90)

    assert not hasattr(writer, "write_reference")
    with pytest.raises(InputDocumentError, match="at least one redaction region"):
        writer.write(source, (), empty_redaction)

    assert not empty_redaction.exists()


def test_destructive_writer_outputs_are_owner_read_write_only(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX permission bits are required")
    source = tmp_path / "source.pdf"
    region = _make_text_pdf(source)

    for index, writer in enumerate((PyMuPdfRedactionWriter(), RasterRebuildWriter(dpi=90))):
        output = tmp_path / f"redacted-{index}.pdf"
        writer.write(source, (region,), output)
        assert output.stat().st_mode & 0o777 == 0o600


def test_writer_working_files_stay_inside_owner_only_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX permission bits are required")
    tmp_path.chmod(0o755)
    source = tmp_path / "source.pdf"
    region = _make_text_pdf(source)
    observed_parent_modes: list[int] = []
    promote = writer_pymupdf_module._promote_temporary

    def inspect_private_stage(temporary: Path, destination: Path) -> None:
        observed_parent_modes.append(temporary.parent.stat().st_mode & 0o777)
        promote(temporary, destination)

    for module in (writer_pymupdf_module, writer_raster_module, writer_overlay_module):
        monkeypatch.setattr(module, "_promote_temporary", inspect_private_stage)

    for index, writer in enumerate((
        PyMuPdfRedactionWriter(),
        RasterRebuildWriter(dpi=90),
        UnsafeOverlayWriter(),
    )):
        writer.write(source, (region,), tmp_path / f"redacted-{index}.pdf")

    assert observed_parent_modes == [0o700] * 3
    assert not tuple(tmp_path.glob(".meddies-pdf-*.working"))


def test_public_writer_consumes_and_reports_one_source_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pdf"
    replacement = tmp_path / "replacement.pdf"
    output = tmp_path / "redacted.pdf"
    region = _make_text_pdf(source)
    source_a = source.read_bytes()
    _make_replacement_pdf(replacement)
    source_b = replacement.read_bytes()
    snapshot = writer_pymupdf_module._snapshot_source

    def snapshot_then_replace(original: Path, staged: Path) -> str:
        source_sha256 = snapshot(original, staged)
        original.write_bytes(source_b)
        return source_sha256

    monkeypatch.setattr(writer_pymupdf_module, "_snapshot_source", snapshot_then_replace)

    result = PyMuPdfRedactionWriter().write(source, (region,), output)
    output_text = PdfReader(output).pages[0].extract_text() or ""

    assert result.source_sha256 == hashlib.sha256(source_a).hexdigest()
    assert "ordinary clinical text" in output_text
    assert "replacement record" not in output_text


def test_overlay_negative_control_stays_extractable_beneath_black_mask(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "unsafe-overlay.pdf"
    region = _make_text_pdf(source)

    result = UnsafeOverlayWriter().write(source, (region,), output)

    assert result.safety == "unsafe_overlay"
    assert result.verification_state == "unverified"
    assert CANARY in (PdfReader(output).pages[0].extract_text() or "")

    document = pdfium.PdfDocument(output)
    image = document[0].render(scale=2).to_pil()
    box = region.quad.bounds()
    center = (round(box.x0 + box.x1), round(box.y0 + box.y1))
    red, green, blue = image.convert("RGB").getpixel(center)
    assert max(red, green, blue) < 20


def test_writer_reports_malformed_input_as_document_error(tmp_path: Path) -> None:
    source = tmp_path / "malformed.pdf"
    source.write_bytes(b"not a PDF")
    output = tmp_path / "output.pdf"
    region = PageRegion(
        page_index=0,
        quad=Quad(
            points=(
                Point(1, 1),
                Point(2, 1),
                Point(2, 2),
                Point(1, 2),
            ),
        ),
        label="secret",
        span_start=0,
        span_end=1,
    )

    with pytest.raises(InputDocumentError) as error:
        PyMuPdfRedactionWriter().write(source, (region,), output)

    assert error.value.stage == "input"
    assert error.value.code == "malformed_document"
    assert not output.exists()


def test_writer_refuses_to_replace_an_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "existing.pdf"
    region = _make_text_pdf(source)
    output.write_bytes(b"keep this file")

    with pytest.raises(OutputPathError) as error:
        PyMuPdfRedactionWriter().write(source, (region,), output)

    assert error.value.stage == "output"
    assert error.value.code == "output_exists"
    assert output.read_bytes() == b"keep this file"


def test_pymupdf_writer_scrubs_attachments_and_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base.pdf"
    source = tmp_path / "source-with-payloads.pdf"
    output = tmp_path / "scrubbed.pdf"
    region = _make_text_pdf(base)
    writer = PdfWriter(clone_from=base)
    writer.add_attachment("synthetic.txt", CANARY.encode())
    writer.add_metadata({"/Subject": CANARY})
    writer.write(source)

    PyMuPdfRedactionWriter().write(source, (region,), output)

    redacted = PdfReader(output)
    assert redacted.attachments == {}
    metadata = redacted.metadata or {}
    assert CANARY not in " ".join(str(value) for value in metadata.values())


def test_other_writers_report_malformed_input_as_document_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "malformed.pdf"
    source.write_bytes(b"not a PDF")
    region = PageRegion(
        page_index=0,
        quad=Quad(
            points=(
                Point(1, 1),
                Point(2, 1),
                Point(2, 2),
                Point(1, 2),
            ),
        ),
        label="secret",
        span_start=0,
        span_end=1,
    )

    for index, writer in enumerate((RasterRebuildWriter(), UnsafeOverlayWriter())):
        with pytest.raises(InputDocumentError) as error:
            writer.write(source, (region,), tmp_path / f"output-{index}.pdf")

        assert error.value.code == "malformed_document"


def test_destructive_writers_transform_unrotated_regions_on_rotated_pages(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rotated.pdf"
    region = _make_rotated_text_pdf(source, tmp_path / "base.pdf")
    bounds = region.quad.bounds()
    display_center = (
        round(letter[1] - ((bounds.y0 + bounds.y1) / 2)),
        round((bounds.x0 + bounds.x1) / 2),
    )

    for writer, filename in (
        (PyMuPdfRedactionWriter(), "object-aware.pdf"),
        (RasterRebuildWriter(dpi=72), "raster.pdf"),
    ):
        output = tmp_path / filename
        writer.write(source, (region,), output)

        assert CANARY not in (PdfReader(output).pages[0].extract_text() or "")
        document = pdfium.PdfDocument(output)
        image = document[0].render(scale=1).to_pil().convert("RGB")
        assert max(image.getpixel(display_center)) < 20


@pytest.mark.parametrize(
    ("writer", "writer_id"),
    [
        (PyMuPdfRedactionWriter(), "pymupdf-object-aware"),
        (RasterRebuildWriter(dpi=90), "pdfium-raster-rebuild-90dpi"),
        (UnsafeOverlayWriter(), "pypdf-overlay-negative-control"),
    ],
)
def test_writer_identity_is_stable_across_internal_implementation_modules(
    writer: RedactionWriter,
    writer_id: str,
) -> None:
    assert writer.writer_id == writer_id
