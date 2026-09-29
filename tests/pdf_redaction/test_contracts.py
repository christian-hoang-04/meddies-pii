from __future__ import annotations

import pytest

from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    GeometryToken,
    PageRegion,
    Point,
    Quad,
    Rect,
    ReviewedRedaction,
)


def test_geometry_page_rejects_token_offset_mismatch() -> None:
    token = GeometryToken(
        text="Alicia",
        quad=Quad(
            points=(
                Point(10.0, 10.0),
                Point(40.0, 10.0),
                Point(40.0, 20.0),
                Point(10.0, 20.0),
            ),
        ),
        start=0,
        end=5,
        source="native",
        confidence=None,
    )

    with pytest.raises(ValueError, match="token text does not match page text"):
        GeometryPage(
            page_index=0,
            width_pt=100.0,
            height_pt=100.0,
            text="Alice",
            tokens=(token,),
        )


def test_quad_bounds_preserve_rotated_polygon_extent() -> None:
    quad = Quad(
        points=(
            Point(20.0, 10.0),
            Point(40.0, 20.0),
            Point(30.0, 40.0),
            Point(10.0, 30.0),
        ),
    )

    assert quad.bounds() == Rect(x0=10.0, y0=10.0, x1=40.0, y1=40.0)


def test_quad_reports_signed_double_area_from_perimeter_order() -> None:
    counterclockwise = Quad(
        points=(
            Point(10.0, 10.0),
            Point(40.0, 10.0),
            Point(40.0, 30.0),
            Point(10.0, 30.0),
        ),
    )
    clockwise = Quad(
        points=(
            counterclockwise.points[3],
            counterclockwise.points[2],
            counterclockwise.points[1],
            counterclockwise.points[0],
        ),
    )

    assert counterclockwise.signed_double_area() == pytest.approx(1_200.0)
    assert clockwise.signed_double_area() == pytest.approx(-1_200.0)


def test_geometry_contract_rejects_self_crossing_redaction_quad() -> None:
    bow_tie = Quad(
        points=(
            Point(10.0, 10.0),
            Point(40.0, 10.0),
            Point(10.0, 40.0),
            Point(40.0, 40.0),
        ),
    )

    with pytest.raises(ValueError, match="perimeter order with positive area"):
        PageRegion(
            page_index=0,
            quad=bow_tie,
            label="secret",
            span_start=0,
            span_end=1,
        )


def test_reviewed_redaction_couples_value_label_page_and_geometry() -> None:
    first_quad = Quad(
        points=(
            Point(10.0, 10.0),
            Point(40.0, 10.0),
            Point(40.0, 20.0),
            Point(10.0, 20.0),
        ),
    )
    second_quad = Quad(
        points=(
            Point(44.0, 10.0),
            Point(70.0, 10.0),
            Point(70.0, 20.0),
            Point(44.0, 20.0),
        ),
    )

    reviewed = ReviewedRedaction(
        value="Alicia",
        label="human_name",
        page_index=2,
        quads=(first_quad, second_quad),
        span_start=4,
        span_end=10,
    )

    assert reviewed.to_regions() == (
        PageRegion(
            page_index=2,
            quad=first_quad,
            label="human_name",
            span_start=4,
            span_end=10,
        ),
        PageRegion(
            page_index=2,
            quad=second_quad,
            label="human_name",
            span_start=4,
            span_end=10,
        ),
    )
    assert "Alicia" not in repr(reviewed)


def test_reviewed_redaction_rejects_blank_values() -> None:
    with pytest.raises(ValueError, match="reviewed value must be non-empty"):
        ReviewedRedaction(
            value="",
            label="human_name",
            page_index=0,
            quads=(
                Quad(
                    points=(
                        Point(10.0, 10.0),
                        Point(40.0, 10.0),
                        Point(40.0, 20.0),
                        Point(10.0, 20.0),
                    ),
                ),
            ),
            span_start=0,
            span_end=1,
        )


def test_reviewed_redaction_rejects_empty_geometry_group() -> None:
    with pytest.raises(ValueError, match="at least one geometry quad"):
        ReviewedRedaction(
            value="Alicia",
            label="human_name",
            page_index=0,
            quads=(),
            span_start=0,
            span_end=6,
        )
