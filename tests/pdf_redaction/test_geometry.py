from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import pytest

from meddies_pii.bioes_inference import SpanDetection
from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    GeometryToken,
    Point,
    Quad,
)
from meddies_pii.pdf_redaction.geometry import (
    SpanMappingError,
    map_detection_to_regions,
    map_detections_to_regions,
    map_span_to_regions,
    rectangle_union_area,
)
from meddies_pii.spans import CharSpan


def _quad(x0: float, y0: float, x1: float, y1: float) -> Quad:
    return Quad(
        points=(
            Point(x0, y0),
            Point(x1, y0 + 2.0),
            Point(x1, y1),
            Point(x0, y1 - 2.0),
        ),
    )


def _page() -> GeometryPage:
    text = "José  ID-42\nMaya"
    return GeometryPage(
        page_index=3,
        width_pt=200.0,
        height_pt=100.0,
        text=text,
        tokens=(
            GeometryToken("José", _quad(10, 10, 35, 20), 0, 4, "native", None),
            GeometryToken("ID-42", _quad(40, 10, 75, 20), 6, 11, "native", None),
            GeometryToken("Maya", _quad(10, 30, 40, 40), 12, 16, "native", None),
        ),
    )


def test_maps_half_open_unicode_span_across_unmapped_whitespace() -> None:
    page = _page()
    span = CharSpan(
        start=0,
        end=11,
        text="José  ID-42",
        label="id_number",
    )

    regions = map_span_to_regions(page, span)

    assert tuple(region.quad for region in regions) == (
        page.tokens[0].quad,
        page.tokens[1].quad,
    )
    assert all(region.page_index == 3 for region in regions)
    assert all((region.span_start, region.span_end) == (0, 11) for region in regions)


def test_rejects_span_with_unmapped_non_whitespace() -> None:
    page = _page()
    broken_page = GeometryPage(
        page_index=page.page_index,
        width_pt=page.width_pt,
        height_pt=page.height_pt,
        text=page.text,
        tokens=(page.tokens[0], page.tokens[2]),
    )
    span = CharSpan(0, 11, "José  ID-42", "id_number")

    with pytest.raises(SpanMappingError, match="unmapped_non_whitespace"):
        map_span_to_regions(broken_page, span)


def test_rejects_invalid_offset_contract_without_echoing_span_text() -> None:
    span = CharSpan(6, 99, "ID-42", "id_number")

    with pytest.raises(SpanMappingError, match="invalid_offsets") as caught:
        map_span_to_regions(_page(), span)

    assert "ID-42" not in str(caught.value)


def test_rejects_truncated_or_missing_detector_results() -> None:
    page = _page()
    truncated = SpanDetection(spans=(), bucket=512, num_tokens=16, truncated=True)

    with pytest.raises(SpanMappingError, match="detector_truncated"):
        map_detection_to_regions(page, truncated)
    with pytest.raises(SpanMappingError, match="detection_cardinality_mismatch"):
        map_detections_to_regions((page,), ())


def test_rectangle_union_area_counts_overlaps_once_and_ignores_empty_input() -> None:
    rectangles = (
        (0.0, 0.0, 60.0, 40.0),
        (30.0, 20.0, 90.0, 80.0),
    )

    assert rectangle_union_area(rectangles) == pytest.approx(5_400.0)
    assert rectangle_union_area(()) == 0.0
