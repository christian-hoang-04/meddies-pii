from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from meddies_pii.pdf_redaction.contracts import PageRegion, Quad, Rect

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.pdf_redaction.document import DocumentInspection


@dataclass(frozen=True, slots=True)
class OracleRegion:
    region: PageRegion
    value_index: int
    channel: str

    def __post_init__(self) -> None:
        if self.value_index < 0:
            msg = "value_index must be non-negative"
            raise ValueError(msg)
        if not self.channel or not self.channel.replace("_", "").isalnum():
            msg = "channel must be a safe identifier"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PageQuad:
    page_index: int
    quad: Quad
    label: str | None = None

    def __post_init__(self) -> None:
        if self.page_index < 0:
            msg = "page_index must be non-negative"
            raise ValueError(msg)
        _require_positive_rect(self.quad.bounds())


@dataclass(frozen=True, slots=True)
class SpatialCoverage:
    gold_regions_fully_covered: int
    gold_region_recall: float
    gold_regions_partially_covered: int
    gold_region_partial_recall: float
    undercoverage_rate: float
    overredaction_rate: float
    negative_control_touches: int


def measure_spatial_coverage(
    *,
    gold: Sequence[PageQuad],
    applied: Sequence[PageQuad],
    negative_controls: Sequence[PageQuad],
) -> SpatialCoverage:
    if not gold:
        msg = "spatial coverage requires gold regions"
        raise ValueError(msg)
    applied_by_page: dict[int, list[Rect]] = {}
    for region in applied:
        applied_by_page.setdefault(region.page_index, []).append(region.quad.bounds())

    fully_covered = 0
    partially_covered = 0
    for region in gold:
        bounds = region.quad.bounds()
        area = _area(bounds)
        intersection = _clipped_union_area(
            bounds,
            applied_by_page.get(region.page_index, ()),
        )
        if intersection > 0.0:
            partially_covered += 1
        if math.isclose(intersection, area, rel_tol=0.0, abs_tol=1e-6):
            fully_covered += 1

    gold_by_page: dict[int, list[Rect]] = {}
    for region in gold:
        gold_by_page.setdefault(region.page_index, []).append(region.quad.bounds())
    pages = set(gold_by_page) | set(applied_by_page)
    gold_area = sum(_union_area(gold_by_page.get(page, ())) for page in pages)
    applied_area = sum(_union_area(applied_by_page.get(page, ())) for page in pages)
    covered_area = sum(
        _intersection_union_area(gold_by_page.get(page, ()), applied_by_page.get(page, ())) for page in pages
    )

    control_touches = sum(
        _clipped_union_area(
            control.quad.bounds(),
            applied_by_page.get(control.page_index, ()),
        )
        > 0.0
        for control in negative_controls
    )
    return SpatialCoverage(
        gold_regions_fully_covered=fully_covered,
        gold_region_recall=fully_covered / len(gold),
        gold_regions_partially_covered=partially_covered,
        gold_region_partial_recall=partially_covered / len(gold),
        undercoverage_rate=max(0.0, (gold_area - covered_area) / gold_area),
        overredaction_rate=max(0.0, (applied_area - covered_area) / gold_area),
        negative_control_touches=control_touches,
    )


def model_coverage_breakdown(
    *,
    oracle_regions: Sequence[OracleRegion],
    applied: Sequence[PageQuad],
    inspection: DocumentInspection,
) -> dict[str, object]:
    def summarize(
        items: Sequence[OracleRegion],
        scoped_applied: Sequence[PageQuad],
    ) -> dict[str, object]:
        coverage = measure_spatial_coverage(
            gold=tuple(PageQuad(page_index=item.region.page_index, quad=item.region.quad) for item in items),
            applied=scoped_applied,
            negative_controls=(),
        )
        return {
            "gold_regions": len(items),
            "fully_covered": coverage.gold_regions_fully_covered,
            "full_recall": coverage.gold_region_recall,
            "partially_covered": coverage.gold_regions_partially_covered,
            "partial_recall": coverage.gold_region_partial_recall,
            "undercoverage_rate": coverage.undercoverage_rate,
            "overredaction_rate": coverage.overredaction_rate,
        }

    page_indices = sorted({item.region.page_index for item in oracle_regions})
    route_by_page = {page.decision.page_index: page.decision.content_route for page in inspection.pages}
    routes = sorted(set(route_by_page.values()))
    labels = sorted({item.region.label for item in oracle_regions})
    return {
        "by_page": [
            {
                "page_index": page_index,
                **summarize(
                    tuple(item for item in oracle_regions if item.region.page_index == page_index),
                    tuple(region for region in applied if region.page_index == page_index),
                ),
            }
            for page_index in page_indices
        ],
        "by_route": [
            {
                "route": route,
                **summarize(
                    tuple(item for item in oracle_regions if route_by_page[item.region.page_index] == route),
                    tuple(region for region in applied if route_by_page[region.page_index] == route),
                ),
            }
            for route in routes
        ],
        "by_label": [
            {
                "label": label,
                **summarize(
                    tuple(item for item in oracle_regions if item.region.label == label),
                    tuple(region for region in applied if region.label == label),
                ),
            }
            for label in labels
        ],
    }


def _clipped_union_area(clip: Rect, rectangles: Sequence[Rect]) -> float:
    clipped = tuple(
        intersection for rectangle in rectangles if (intersection := _intersection(clip, rectangle)) is not None
    )
    return _union_area(clipped)


def _intersection_union_area(left: Sequence[Rect], right: Sequence[Rect]) -> float:
    intersections = tuple(
        intersection
        for left_rectangle in left
        for right_rectangle in right
        if (intersection := _intersection(left_rectangle, right_rectangle)) is not None
    )
    return _union_area(intersections)


def _union_area(rectangles: Sequence[Rect]) -> float:
    if not rectangles:
        return 0.0

    x_values = sorted({value for rectangle in rectangles for value in (rectangle.x0, rectangle.x1)})
    area = 0.0
    for x0, x1 in itertools.pairwise(x_values):
        if x1 <= x0:
            continue
        intervals = sorted(
            (rectangle.y0, rectangle.y1) for rectangle in rectangles if rectangle.x0 < x1 and rectangle.x1 > x0
        )
        if not intervals:
            continue
        covered_y = 0.0
        active_start, active_end = intervals[0]
        for start, end in intervals[1:]:
            if start > active_end:
                covered_y += active_end - active_start
                active_start, active_end = start, end
            else:
                active_end = max(active_end, end)
        covered_y += active_end - active_start
        area += (x1 - x0) * covered_y
    return area


def _intersection(left: Rect, right: Rect) -> Rect | None:
    x0 = max(left.x0, right.x0)
    y0 = max(left.y0, right.y0)
    x1 = min(left.x1, right.x1)
    y1 = min(left.y1, right.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Rect(x0=x0, y0=y0, x1=x1, y1=y1)


def _area(rectangle: Rect) -> float:
    _require_positive_rect(rectangle)
    return (rectangle.x1 - rectangle.x0) * (rectangle.y1 - rectangle.y0)


def _require_positive_rect(rectangle: Rect) -> None:
    coordinates = (rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y1)
    if not all(math.isfinite(value) for value in coordinates):
        msg = "quad bounds must be finite"
        raise ValueError(msg)
    if rectangle.x1 <= rectangle.x0 or rectangle.y1 <= rectangle.y0:
        msg = "quad bounds must have positive area"
        raise ValueError(msg)
