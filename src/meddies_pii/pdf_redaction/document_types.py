from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from meddies_pii.pdf_redaction.ocr_types import RasterPage
    from meddies_pii.pdf_redaction.routing import PageRouteDecision, PageRouteSignals

PdfSource = str | Path | bytes

DEFAULT_RENDER_DPI = 300
DEFAULT_MEANINGFUL_RASTER_AREA_RATIO = 0.01
DEFAULT_MAX_RASTER_PIXELS = 25_000_000


class DocumentAdapterError(RuntimeError):
    def __init__(self, stage: str, *, page_index: int | None = None) -> None:
        location = "document" if page_index is None else f"page {page_index}"
        super().__init__(f"pymupdf document adapter failed at {location}: {stage}")
        self.stage = stage
        self.page_index = page_index


class PdfiumDocumentAdapterError(DocumentAdapterError):
    def __init__(self, stage: str, *, page_index: int | None = None) -> None:
        location = "document" if page_index is None else f"page {page_index}"
        RuntimeError.__init__(self, f"pdfium document adapter failed at {location}: {stage}")
        self.stage = stage
        self.page_index = page_index


@dataclass(frozen=True, slots=True)
class PageInspection:
    signals: PageRouteSignals
    decision: PageRouteDecision
    raster_area_ratio: float
    displayed_image_count: int
    invisible_native_characters: int
    replacement_characters: int
    annotation_count: int
    widget_count: int
    rotation_degrees: int


@dataclass(frozen=True, slots=True)
class DocumentInspection:
    pages: tuple[PageInspection, ...]
    attachment_count: int


@dataclass(frozen=True, slots=True)
class RasterArtifact:
    raster: RasterPage = field(repr=False)
    png_bytes: bytes = field(repr=False)
    sha256: str
    dpi: int


class DocumentAdapter(Protocol):
    def inspect(self, source: PdfSource) -> DocumentInspection: ...

    def render_pages(self, source: PdfSource, *, page_indices: Sequence[int]) -> tuple[RasterArtifact, ...]: ...


@dataclass(frozen=True, slots=True)
class TextEvidence:
    visible_characters: int
    invisible_characters: int
    replacement_characters: int


def validate_render_settings(*, dpi: int, meaningful_raster_area_ratio: float, max_raster_pixels: int) -> None:
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        msg = "dpi must be a positive integer"
        raise ValueError(msg)
    if isinstance(max_raster_pixels, bool) or not isinstance(max_raster_pixels, int) or max_raster_pixels <= 0:
        msg = "max_raster_pixels must be a positive integer"
        raise ValueError(msg)
    if not math.isfinite(meaningful_raster_area_ratio) or not 0.0 <= meaningful_raster_area_ratio <= 1.0:
        msg = "meaningful_raster_area_ratio must be between zero and one"
        raise ValueError(msg)


def validate_page_indices(page_indices: Sequence[int]) -> tuple[int, ...]:
    indices = tuple(page_indices)
    seen: set[int] = set()
    for page_index in indices:
        if isinstance(page_index, bool) or not isinstance(page_index, int):
            msg = "invalid_page_index"
            raise DocumentAdapterError(msg)
        if page_index < 0:
            msg = "page_index_out_of_range"
            raise DocumentAdapterError(msg, page_index=page_index)
        if page_index in seen:
            msg = "duplicate_page_index"
            raise DocumentAdapterError(msg, page_index=page_index)
        seen.add(page_index)
    return indices


def trust_diagnostics(*, evidence: TextEvidence, annotation_count: int, widget_count: int) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if evidence.invisible_characters:
        diagnostics.append("invisible_text_layer")
    if evidence.replacement_characters:
        diagnostics.append("replacement_character")
    if annotation_count:
        diagnostics.append("annotation_object")
    if widget_count:
        diagnostics.append("form_widget")
    return tuple(diagnostics)


def require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = "expected mapping"
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def require_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        msg = "expected sequence"
        raise ValueError(msg)
    return cast("Sequence[object]", value)


def require_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "expected integer"
        raise ValueError(msg)
    return value


def require_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "expected real number"
        raise ValueError(msg)
    result = float(value)
    if not math.isfinite(result):
        msg = "expected finite real number"
        raise ValueError(msg)
    return result
