from __future__ import annotations

import hashlib
from io import BytesIO
from itertools import starmap
from typing import TYPE_CHECKING, cast

import fitz
import pymupdf
import pytest
from PIL import Image

from meddies_pii.pdf_redaction.benchmark.corpus import generate_challenge_corpus
from meddies_pii.pdf_redaction.document import (
    DEFAULT_MEANINGFUL_RASTER_AREA_RATIO,
    DocumentAdapterError,
    PdfiumDocumentAdapter,
    PymupdfDocumentAdapter,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_generated_corpus_routes_from_page_evidence_without_exposing_text() -> None:
    fixture = generate_challenge_corpus()

    inspection = PymupdfDocumentAdapter().inspect(fixture.pdf_bytes)

    assert tuple(page.decision.content_route for page in inspection.pages) == (
        "native_trusted",
        "native_trusted",
        "native_trusted",
        "native_trusted",
        "hybrid_untrusted",
        "image_only",
        "image_only",
        "image_only",
        "hybrid_mixed",
        "hybrid_untrusted",
    )
    assert inspection.attachment_count == 1

    hidden_ocr = inspection.pages[4]
    assert hidden_ocr.invisible_native_characters > 0
    assert hidden_ocr.signals.has_meaningful_raster
    assert hidden_ocr.signals.trust_diagnostics == ("invisible_text_layer",)

    for page in inspection.pages[5:8]:
        assert page.signals.visible_native_characters == 0
        assert page.signals.has_meaningful_raster

    mixed = inspection.pages[8]
    assert mixed.signals.visible_native_characters > 0
    assert mixed.raster_area_ratio >= DEFAULT_MEANINGFUL_RASTER_AREA_RATIO

    hostile = inspection.pages[9]
    assert hostile.invisible_native_characters > 0
    assert hostile.annotation_count > 0
    assert hostile.widget_count > 0
    assert set(hostile.signals.trust_diagnostics) >= {
        "annotation_object",
        "form_widget",
        "invisible_text_layer",
    }

    safe_repr = repr(inspection)
    assert not any(canary.raw_value in safe_repr for canary in fixture.gold.canaries)

    rotated = PymupdfDocumentAdapter(dpi=72).render_page(fixture.pdf_bytes, page_index=3)
    assert (rotated.raster.width_px, rotated.raster.height_px) == (792, 612)
    assert rotated.raster.map_pixel(0, 0).x == pytest.approx(0)
    assert rotated.raster.map_pixel(0, 0).y == pytest.approx(792)


def test_pdfium_adapter_routes_the_corpus_conservatively() -> None:
    fixture = generate_challenge_corpus()

    inspection = PdfiumDocumentAdapter().inspect(fixture.pdf_bytes)

    assert tuple(page.decision.content_route for page in inspection.pages) == (
        "native_trusted",
        "native_trusted",
        "native_trusted",
        "native_trusted",
        "hybrid_untrusted",
        "image_only",
        "image_only",
        "image_only",
        "hybrid_mixed",
        "hybrid_untrusted",
    )
    assert inspection.attachment_count == 1
    assert all(page.decision.requires_full_page_ocr for page in inspection.pages[4:])
    hostile = inspection.pages[9]
    assert hostile.annotation_count == 3
    assert hostile.widget_count == 1
    assert set(hostile.signals.trust_diagnostics) >= {
        "annotation_object",
        "form_widget",
        "invisible_text_layer",
    }
    assert not any(canary.raw_value in repr(inspection) for canary in fixture.gold.canaries)


def test_raster_area_threshold_is_explicit_and_configurable() -> None:
    fixture = generate_challenge_corpus()

    default = PymupdfDocumentAdapter().inspect(fixture.pdf_bytes).pages[8]
    strict = PymupdfDocumentAdapter(meaningful_raster_area_ratio=0.5).inspect(fixture.pdf_bytes).pages[8]
    zero_threshold_native = PymupdfDocumentAdapter(meaningful_raster_area_ratio=0.0).inspect(fixture.pdf_bytes).pages[0]

    assert default.signals.has_meaningful_raster
    assert not strict.signals.has_meaningful_raster
    assert strict.decision.content_route == "native_trusted"
    assert not zero_threshold_native.signals.has_meaningful_raster


def test_document_backends_report_the_same_overlapping_raster_area(
    tmp_path: Path,
) -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), "black").save(image_bytes, format="PNG")
    source = tmp_path / "overlapping-images.pdf"
    with fitz.open() as document:
        page = document.new_page(width=100, height=100)
        page.insert_image(fitz.Rect(0, 0, 60, 40), stream=image_bytes.getvalue())
        page.insert_image(fitz.Rect(30, 20, 90, 80), stream=image_bytes.getvalue())
        document.save(source)

    pymupdf_page = PymupdfDocumentAdapter().inspect(source).pages[0]
    pdfium_page = PdfiumDocumentAdapter().inspect(source).pages[0]

    assert pymupdf_page.displayed_image_count == 2
    assert pdfium_page.displayed_image_count == 2
    assert pymupdf_page.raster_area_ratio == pytest.approx(0.54)
    assert pdfium_page.raster_area_ratio == pytest.approx(pymupdf_page.raster_area_ratio)


def test_zero_opacity_text_is_counted_as_invisible_not_visible(tmp_path: Path) -> None:
    source = tmp_path / "zero-opacity.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=300)
        page.insert_text((20, 40), "VISIBLE", fill_opacity=1)
        page.insert_text((20, 80), "HIDDEN", fill_opacity=0)
        document.save(source)

    page = PymupdfDocumentAdapter().inspect(source).pages[0]

    assert page.signals.visible_native_characters == 7
    assert page.invisible_native_characters == 6
    assert page.signals.trust_diagnostics == ("invisible_text_layer",)


def test_pdfium_treats_zero_opacity_text_as_untrusted(tmp_path: Path) -> None:
    source = tmp_path / "pdfium-zero-opacity.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=300)
        page.insert_text((20, 40), "VISIBLE", fill_opacity=1)
        page.insert_text((20, 80), "HIDDEN", fill_opacity=0)
        document.save(source)

    page = PdfiumDocumentAdapter().inspect(source).pages[0]

    assert page.signals.visible_native_characters == 7
    assert page.invisible_native_characters == 6
    assert page.decision.content_route == "hybrid_untrusted"
    assert page.signals.trust_diagnostics == ("invisible_text_layer",)


@pytest.mark.parametrize(
    ("rotation", "expected_corners"),
    [
        (0, ((0.0, 0.0), (200.0, 0.0), (200.0, 300.0), (0.0, 300.0))),
        (90, ((0.0, 300.0), (0.0, 0.0), (200.0, 0.0), (200.0, 300.0))),
        (
            180,
            ((200.0, 300.0), (0.0, 300.0), (0.0, 0.0), (200.0, 0.0)),
        ),
        (270, ((200.0, 0.0), (200.0, 300.0), (0.0, 300.0), (0.0, 0.0))),
    ],
)
def test_fixed_dpi_raster_maps_display_pixels_to_unrotated_pdf_points(
    tmp_path: Path,
    rotation: int,
    expected_corners: tuple[tuple[float, float], ...],
) -> None:
    source = tmp_path / f"rotation-{rotation}.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=300)
        page.draw_rect(fitz.Rect(20, 30, 180, 270), color=(0, 0, 0))
        page.set_rotation(rotation)
        document.save(source)

    first = PymupdfDocumentAdapter(dpi=72).render_page(source, page_index=0)
    second = PymupdfDocumentAdapter(dpi=72).render_page(source, page_index=0)

    assert first.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert first.png_bytes == second.png_bytes
    assert first.sha256 == second.sha256
    assert first.sha256 == hashlib.sha256(first.png_bytes).hexdigest()
    assert first.raster.image is first.png_bytes
    assert first.raster.width_pt == 200
    assert first.raster.height_pt == 300

    pixel_corners = (
        (0.0, 0.0),
        (float(first.raster.width_px), 0.0),
        (float(first.raster.width_px), float(first.raster.height_px)),
        (0.0, float(first.raster.height_px)),
    )
    mapped = tuple(starmap(first.raster.map_pixel, pixel_corners))
    mapped_coordinates = tuple(coordinate for point in mapped for coordinate in (point.x, point.y))
    expected_coordinates = tuple(coordinate for point in expected_corners for coordinate in point)
    assert mapped_coordinates == pytest.approx(expected_coordinates)


@pytest.mark.parametrize(
    ("rotation", "expected_corners"),
    [
        (0, ((0.0, 0.0), (200.0, 0.0), (200.0, 300.0), (0.0, 300.0))),
        (90, ((0.0, 300.0), (0.0, 0.0), (200.0, 0.0), (200.0, 300.0))),
        (
            180,
            ((200.0, 300.0), (0.0, 300.0), (0.0, 0.0), (200.0, 0.0)),
        ),
        (270, ((200.0, 0.0), (200.0, 300.0), (0.0, 300.0), (0.0, 0.0))),
    ],
)
def test_pdfium_raster_maps_display_pixels_to_unrotated_pdf_points(
    tmp_path: Path,
    rotation: int,
    expected_corners: tuple[tuple[float, float], ...],
) -> None:
    source = tmp_path / f"pdfium-rotation-{rotation}.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=300)
        page.draw_rect(fitz.Rect(20, 30, 180, 270), color=(0, 0, 0))
        page.set_rotation(rotation)
        document.save(source)

    first = PdfiumDocumentAdapter(dpi=72).render_page(source, page_index=0)
    second = PdfiumDocumentAdapter(dpi=72).render_page(source, page_index=0)

    assert first.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert first.png_bytes == second.png_bytes
    assert first.sha256 == hashlib.sha256(first.png_bytes).hexdigest()
    assert first.raster.image is first.png_bytes
    assert first.raster.width_pt == 200
    assert first.raster.height_pt == 300
    pixel_corners = (
        (0.0, 0.0),
        (float(first.raster.width_px), 0.0),
        (float(first.raster.width_px), float(first.raster.height_px)),
        (0.0, float(first.raster.height_px)),
    )
    mapped = tuple(starmap(first.raster.map_pixel, pixel_corners))
    mapped_coordinates = tuple(coordinate for point in mapped for coordinate in (point.x, point.y))
    expected_coordinates = tuple(coordinate for point in expected_corners for coordinate in point)
    assert mapped_coordinates == pytest.approx(expected_coordinates)


def test_document_errors_never_echo_source_bytes() -> None:
    private_marker = "SYNTH-PRIVATE-ERROR-MARKER"
    source = (private_marker + " is not a PDF").encode()

    with pytest.raises(DocumentAdapterError) as caught:
        PymupdfDocumentAdapter().inspect(source)

    assert private_marker not in str(caught.value)
    assert private_marker not in repr(caught.value)


def test_pdfium_errors_identify_adapter_without_echoing_source_bytes() -> None:
    private_marker = "SYNTH-PDFIUM-PRIVATE-ERROR-MARKER"
    source = (private_marker + " is not a PDF").encode()

    with pytest.raises(DocumentAdapterError) as caught:
        PdfiumDocumentAdapter().inspect(source)

    assert "pdfium document adapter failed" in str(caught.value)
    assert private_marker not in str(caught.value)
    assert private_marker not in repr(caught.value)


def test_raster_pixel_limit_fails_before_rendering_unbounded_page(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oversized.pdf"
    with fitz.open() as document:
        document.new_page(width=200, height=300)
        document.save(source)

    with pytest.raises(DocumentAdapterError) as caught:
        PymupdfDocumentAdapter(dpi=72, max_raster_pixels=59_999).render_page(source, page_index=0)

    assert caught.value.stage == "raster_pixel_limit"


def test_bulk_render_opens_once_and_preserves_requested_page_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "three-pages.pdf"
    with fitz.open() as document:
        for rotation in (0, 90, 180):
            page = document.new_page(width=200, height=300)
            page.draw_rect(fitz.Rect(20, 30, 180, 270), color=(0, 0, 0))
            page.set_rotation(rotation)
        document.save(source)

    original_open = cast("Callable[..., pymupdf.Document]", pymupdf.open)
    open_calls = 0

    def counted_open(*args: object, **kwargs: object) -> pymupdf.Document:
        nonlocal open_calls
        open_calls += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(pymupdf, "open", counted_open)

    artifacts = PymupdfDocumentAdapter(dpi=72).render_pages(source, page_indices=(2, 0, 1))

    assert open_calls == 1
    assert tuple(artifact.raster.page_index for artifact in artifacts) == (2, 0, 1)


def test_pdfium_bulk_render_preserves_order_and_enforces_pixel_bound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pdfium-three-pages.pdf"
    with fitz.open() as document:
        for rotation in (0, 90, 180):
            page = document.new_page(width=200, height=300)
            page.draw_rect(fitz.Rect(20, 30, 180, 270), color=(0, 0, 0))
            page.set_rotation(rotation)
        document.save(source)

    artifacts = PdfiumDocumentAdapter(dpi=72).render_pages(source, page_indices=(2, 0, 1))

    assert tuple(artifact.raster.page_index for artifact in artifacts) == (2, 0, 1)
    with pytest.raises(DocumentAdapterError) as caught:
        PdfiumDocumentAdapter(dpi=72, max_raster_pixels=59_999).render_page(source, page_index=0)
    assert caught.value.stage == "raster_pixel_limit"


def test_pdfium_bulk_render_rejects_duplicate_indices_before_parsing_source() -> None:
    with pytest.raises(DocumentAdapterError) as caught:
        PdfiumDocumentAdapter().render_pages(b"not a PDF and must not be parsed", page_indices=(0, 0))

    assert caught.value.stage == "duplicate_page_index"


def test_bulk_render_rejects_duplicate_indices_before_opening_source() -> None:
    with pytest.raises(DocumentAdapterError) as caught:
        PymupdfDocumentAdapter().render_pages(b"not a PDF and must not be parsed", page_indices=(0, 0))

    assert caught.value.stage == "duplicate_page_index"


def test_adapter_rejects_invalid_render_dpi() -> None:
    with pytest.raises(ValueError, match="dpi must be a positive integer"):
        PymupdfDocumentAdapter(dpi=0)


def test_adapter_rejects_invalid_raster_pixel_limit() -> None:
    with pytest.raises(ValueError, match="max_raster_pixels must be a positive integer"):
        PymupdfDocumentAdapter(max_raster_pixels=0)


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_adapter_rejects_invalid_raster_area_ratio(ratio: float) -> None:
    with pytest.raises(
        ValueError,
        match="meaningful_raster_area_ratio must be between zero and one",
    ):
        PymupdfDocumentAdapter(meaningful_raster_area_ratio=ratio)
