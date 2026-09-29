from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from meddies_pii.pdf_redaction.contracts import GeometryPage, PageRegion
from meddies_pii.taxonomy import require_pii_label

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.bioes_inference import SpanDetection
    from meddies_pii.spans import CharSpan


class SpanMappingError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        page_index: int | None = None,
        span_index: int | None = None,
    ) -> None:
        location = "document"
        if page_index is not None:
            location = f"page {page_index}"
        if span_index is not None:
            location = f"{location}, span {span_index}"
        super().__init__(f"span mapping failed at {location}: {reason}")
        self.reason = reason
        self.page_index = page_index
        self.span_index = span_index


def rectangle_union_area(
    rectangles: Sequence[tuple[float, float, float, float]],
) -> float:
    if not rectangles:
        return 0.0

    x_coordinates = sorted({x for rectangle in rectangles for x in rectangle[::2]})
    area = 0.0
    for left, right in itertools.pairwise(x_coordinates):
        if right <= left:
            continue
        intervals = sorted((bottom, top) for x0, bottom, x1, top in rectangles if x0 < right and x1 > left)
        if not intervals:
            continue
        covered_y = 0.0
        current_bottom, current_top = intervals[0]
        for bottom, top in intervals[1:]:
            if bottom > current_top:
                covered_y += current_top - current_bottom
                current_bottom, current_top = bottom, top
            else:
                current_top = max(current_top, top)
        covered_y += current_top - current_bottom
        area += (right - left) * covered_y
    return area


def map_span_to_regions(page: GeometryPage, span: CharSpan) -> tuple[PageRegion, ...]:
    if span.start < 0 or span.end <= span.start or span.end > len(page.text):
        msg = "invalid_offsets"
        raise SpanMappingError(msg, page_index=page.page_index)
    if page.text[span.start : span.end] != span.text:
        msg = "text_offset_mismatch"
        raise SpanMappingError(msg, page_index=page.page_index)
    try:
        label = require_pii_label(span.label)
    except ValueError:
        msg = "invalid_label"
        raise SpanMappingError(msg, page_index=page.page_index) from None

    covered = [False] * (span.end - span.start)
    regions: list[PageRegion] = []
    for token in page.tokens:
        overlap_start = max(span.start, token.start)
        overlap_end = min(span.end, token.end)
        if overlap_start >= overlap_end:
            continue
        for index in range(overlap_start, overlap_end):
            covered[index - span.start] = True
        regions.append(
            PageRegion(
                page_index=page.page_index,
                quad=token.quad,
                label=label,
                span_start=span.start,
                span_end=span.end,
            ),
        )

    for offset, character in enumerate(span.text):
        if not character.isspace() and not covered[offset]:
            msg = "unmapped_non_whitespace"
            raise SpanMappingError(msg, page_index=page.page_index)
    if not regions:
        msg = "empty_region_set"
        raise SpanMappingError(msg, page_index=page.page_index)
    return tuple(regions)


def map_detection_to_regions(page: GeometryPage, detection: SpanDetection) -> tuple[PageRegion, ...]:
    if detection.truncated:
        msg = "detector_truncated"
        raise SpanMappingError(msg, page_index=page.page_index)

    regions: list[PageRegion] = []
    for span_index, span in enumerate(detection.spans):
        # reason: each mapping failure is enriched with the current span index before the next independent span is tried.
        try:
            regions.extend(map_span_to_regions(page, span))
        except SpanMappingError as error:  # ruff: ignore[try-except-in-loop]
            raise SpanMappingError(
                error.reason,
                page_index=page.page_index,
                span_index=span_index,
            ) from None
    return tuple(regions)


def map_detections_to_regions(
    pages: Sequence[GeometryPage],
    detections: Sequence[SpanDetection],
) -> tuple[PageRegion, ...]:
    if len(pages) != len(detections):
        msg = "detection_cardinality_mismatch"
        raise SpanMappingError(msg)
    if any(previous.page_index >= current.page_index for previous, current in itertools.pairwise(pages)):
        msg = "pages_not_strictly_ordered"
        raise SpanMappingError(msg)

    regions: list[PageRegion] = []
    for page, detection in zip(pages, detections, strict=True):
        regions.extend(map_detection_to_regions(page, detection))
    return tuple(regions)
