from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from meddies_pii.taxonomy import PiiLabel

TextSource = Literal["native", "ocr"]
ContentRoute = Literal["native_trusted", "hybrid_mixed", "hybrid_untrusted", "image_only"]
DocumentRisk = Literal["static", "dynamic_renderable", "unsupported"]
RedactionPolicy = Literal["surgical_static", "secure_share"]


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class Quad:
    points: tuple[Point, Point, Point, Point]

    def bounds(self) -> Rect:
        xs = tuple(point.x for point in self.points)
        ys = tuple(point.y for point in self.points)
        return Rect(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    def signed_double_area(self) -> float:
        return sum(
            current.x * following.y - following.x * current.y
            for current, following in zip(
                self.points,
                self.points[1:] + self.points[:1],
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class GeometryToken:
    text: str
    quad: Quad
    start: int
    end: int
    source: TextSource
    confidence: float | None

    def __post_init__(self) -> None:
        if not self.text or self.start < 0 or self.end <= self.start:
            msg = "geometry token text and offsets must be non-empty"
            raise ValueError(msg)
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            msg = "geometry token confidence must be between zero and one"
            raise ValueError(msg)
        _validate_positive_quad(self.quad)


@dataclass(frozen=True, slots=True)
class GeometryPage:
    page_index: int
    width_pt: float
    height_pt: float
    text: str
    tokens: tuple[GeometryToken, ...]

    def __post_init__(self) -> None:
        if self.page_index < 0:
            msg = "page_index must be non-negative"
            raise ValueError(msg)
        if not self._valid_dimension(self.width_pt) or not self._valid_dimension(self.height_pt):
            msg = "page dimensions must be finite and positive"
            raise ValueError(msg)

        previous_end = 0
        for token in self.tokens:
            if token.start < previous_end or token.end < token.start:
                msg = "token offsets must be ordered and non-overlapping"
                raise ValueError(msg)
            if self.text[token.start : token.end] != token.text:
                msg = "token text does not match page text at offsets"
                raise ValueError(msg)
            self._validate_quad(token.quad)
            previous_end = token.end

    @staticmethod
    def _valid_dimension(value: float) -> bool:
        return math.isfinite(value) and value > 0

    def _validate_quad(self, quad: Quad) -> None:
        for point in quad.points:
            if not math.isfinite(point.x) or not math.isfinite(point.y):
                msg = "token polygon coordinates must be finite"
                raise ValueError(msg)
            if not 0 <= point.x <= self.width_pt or not 0 <= point.y <= self.height_pt:
                msg = "token polygon must be inside the page"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PageRegion:
    page_index: int
    quad: Quad
    label: PiiLabel
    span_start: int
    span_end: int

    def __post_init__(self) -> None:
        if self.page_index < 0:
            msg = "page_index must be non-negative"
            raise ValueError(msg)
        if self.span_start < 0 or self.span_end <= self.span_start:
            msg = "region span offsets must be non-empty"
            raise ValueError(msg)
        _validate_positive_quad(self.quad)


@dataclass(frozen=True, slots=True)
class ReviewedRedaction:
    value: str = field(repr=False)
    label: PiiLabel
    page_index: int
    quads: tuple[Quad, ...]
    span_start: int
    span_end: int

    def __post_init__(self) -> None:
        if not self.value:
            msg = "reviewed value must be non-empty"
            raise ValueError(msg)
        if not self.quads:
            msg = "reviewed redaction requires at least one geometry quad"
            raise ValueError(msg)
        self.to_regions()

    def to_regions(self) -> tuple[PageRegion, ...]:
        return tuple(
            PageRegion(
                page_index=self.page_index,
                quad=quad,
                label=self.label,
                span_start=self.span_start,
                span_end=self.span_end,
            )
            for quad in self.quads
        )


def _validate_positive_quad(quad: Quad) -> None:
    if any(not math.isfinite(coordinate) for point in quad.points for coordinate in (point.x, point.y)):
        msg = "polygon coordinates must be finite"
        raise ValueError(msg)
    if quad.signed_double_area() <= 0.0:
        msg = "polygon must use perimeter order with positive area"
        raise ValueError(msg)
