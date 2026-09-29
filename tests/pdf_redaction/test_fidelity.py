from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import shutil
from typing import TYPE_CHECKING

import pymupdf
import pytest

from meddies_pii.pdf_redaction.benchmark.fidelity import (
    measure_render_fidelity,
    write_raster_reference,
)
from meddies_pii.pdf_redaction.contracts import PageRegion, Point, Quad
from meddies_pii.pdf_redaction.writers import RasterRebuildWriter

if TYPE_CHECKING:
    from pathlib import Path


# reason: Page size, rotation, count, text, and rectangles are independent PDF-fixture axes used by fidelity tests.
def _write_pdf(  # ruff: ignore[too-many-arguments]
    path: Path,
    *,
    width: float = 200.0,
    height: float = 300.0,
    rotation: int = 0,
    page_count: int = 1,
    include_stable_text: bool = True,
    rectangles: tuple[tuple[pymupdf.Rect, tuple[float, float, float]], ...] = (),
) -> None:
    with pymupdf.open() as document:
        for _ in range(page_count):
            page = document.new_page(width=width, height=height)
            if include_stable_text:
                page.insert_text((20.0, 40.0), "stable clinical text", fontsize=12.0)
            for rectangle, color in rectangles:
                page.draw_rect(rectangle, color=color, fill=color, width=0.0)
            page.set_rotation(rotation)
        document.save(path)


def _region(rectangle: pymupdf.Rect, *, page_index: int = 0) -> PageRegion:
    return PageRegion(
        page_index=page_index,
        quad=Quad(
            points=(
                Point(rectangle.x0, rectangle.y0),
                Point(rectangle.x1, rectangle.y0),
                Point(rectangle.x1, rectangle.y1),
                Point(rectangle.x0, rectangle.y1),
            ),
        ),
        label="secret",
        span_start=0,
        span_end=1,
    )


def test_identical_render_has_no_unexplained_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)
    shutil.copyfile(source, output)

    result = measure_render_fidelity(source, output, ())

    assert result.render_dpi == 72
    assert result.pixel_difference_threshold == 16
    assert result.mask_dilation_pixels == 2
    assert result.total_outside_mask_pixels == 200 * 300
    assert result.changed_outside_mask_pixels == 0
    assert result.changed_outside_mask_rate == 0.0
    assert result.unexplained_changed_regions == ()


def test_changes_inside_dilated_oracle_mask_are_ignored(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    masked_rectangle = pymupdf.Rect(60.0, 80.0, 140.0, 140.0)
    _write_pdf(source, rectangles=((masked_rectangle, (0.0, 0.5, 0.8)),))
    _write_pdf(output, rectangles=((masked_rectangle, (0.0, 0.0, 0.0)),))

    result = measure_render_fidelity(source, output, (_region(masked_rectangle),))

    assert 0 < result.total_outside_mask_pixels < 200 * 300
    assert result.changed_outside_mask_pixels == 0
    assert result.unexplained_changed_regions == ()


def test_disconnected_changes_outside_mask_are_reported_separately(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    first = pymupdf.Rect(30.0, 100.0, 50.0, 120.0)
    second = pymupdf.Rect(150.0, 220.0, 180.0, 250.0)
    _write_pdf(source)
    _write_pdf(
        output,
        rectangles=((first, (0.0, 0.0, 0.0)), (second, (0.0, 0.0, 0.0))),
    )

    result = measure_render_fidelity(source, output, ())

    assert result.changed_outside_mask_pixels > 1_000
    assert result.changed_outside_mask_rate > 0.01
    assert len(result.unexplained_changed_regions) == 2
    assert result.unexplained_changed_regions[0].page_index == 0
    assert result.unexplained_changed_regions[0].x0_px <= 30
    assert result.unexplained_changed_regions[0].x1_px >= 50
    assert result.unexplained_changed_regions[1].x0_px <= 150
    assert result.unexplained_changed_regions[1].x1_px >= 180


def test_blank_output_is_not_certified_as_faithful(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    blank_output = tmp_path / "blank.pdf"
    _write_pdf(
        source,
        rectangles=((pymupdf.Rect(20.0, 80.0, 180.0, 260.0), (0.2, 0.4, 0.7)),),
    )
    _write_pdf(blank_output, include_stable_text=False)

    result = measure_render_fidelity(source, blank_output, ())

    assert result.changed_outside_mask_pixels > 20_000
    assert result.changed_outside_mask_rate > 0.3
    assert result.unexplained_changed_regions


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_oracle_mask_follows_rotated_page_coordinates(tmp_path: Path, rotation: int) -> None:
    source = tmp_path / f"source-{rotation}.pdf"
    output = tmp_path / f"output-{rotation}.pdf"
    masked_rectangle = pymupdf.Rect(60.0, 80.0, 140.0, 140.0)
    _write_pdf(
        source,
        rotation=rotation,
        rectangles=((masked_rectangle, (0.0, 0.5, 0.8)),),
    )
    _write_pdf(
        output,
        rotation=rotation,
        rectangles=((masked_rectangle, (0.0, 0.0, 0.0)),),
    )

    result = measure_render_fidelity(source, output, (_region(masked_rectangle),))

    assert result.total_outside_mask_pixels < 200 * 300
    assert result.changed_outside_mask_pixels == 0
    assert result.unexplained_changed_regions == ()


def test_page_count_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "two-pages.pdf"
    output = tmp_path / "one-page.pdf"
    _write_pdf(source, page_count=2)
    _write_pdf(output)

    with pytest.raises(ValueError, match="same page count"):
        measure_render_fidelity(source, output, ())


def test_display_size_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "different-size.pdf"
    _write_pdf(source)
    _write_pdf(output, width=201.0)

    with pytest.raises(ValueError, match="same display size"):
        measure_render_fidelity(source, output, ())


def test_raster_redaction_has_zero_drift_against_same_writer_reference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    reference = tmp_path / "reference.pdf"
    redacted = tmp_path / "redacted.pdf"
    masked_rectangle = pymupdf.Rect(60.0, 80.0, 140.0, 140.0)
    _write_pdf(
        source,
        rotation=90,
        rectangles=((masked_rectangle, (0.0, 0.5, 0.8)),),
    )
    region = _region(masked_rectangle)
    writer = RasterRebuildWriter(dpi=90)
    write_raster_reference(source, reference, dpi=90)
    writer.write(source, (region,), redacted)

    redaction_fidelity = measure_render_fidelity(
        reference,
        redacted,
        (region,),
        mask_geometry_source=source,
    )
    rasterization_fidelity = measure_render_fidelity(source, reference, ())

    assert redaction_fidelity.changed_outside_mask_pixels == 0
    assert redaction_fidelity.unexplained_changed_regions == ()
    assert rasterization_fidelity.changed_outside_mask_pixels > 0
