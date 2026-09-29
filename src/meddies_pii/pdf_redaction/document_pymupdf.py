from __future__ import annotations

import hashlib
import importlib
import math
from typing import TYPE_CHECKING, Protocol, cast

from meddies_pii.pdf_redaction.document_types import (
    DEFAULT_MAX_RASTER_PIXELS,
    DEFAULT_MEANINGFUL_RASTER_AREA_RATIO,
    DEFAULT_RENDER_DPI,
    DocumentAdapterError,
    DocumentInspection,
    PageInspection,
    PdfSource,
    RasterArtifact,
    TextEvidence,
    require_int,
    require_mapping,
    require_real,
    require_sequence,
    trust_diagnostics,
    validate_render_settings,
)
from meddies_pii.pdf_redaction.geometry import rectangle_union_area
from meddies_pii.pdf_redaction.ocr import PixelToPageTransform, RasterPage
from meddies_pii.pdf_redaction.routing import PageRouteSignals, classify_page

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_FILLED_TEXT_FLAG = 1 << 4
_STROKED_TEXT_FLAG = 1 << 3
_VISIBLE_TEXT_FLAGS = _FILLED_TEXT_FLAG | _STROKED_TEXT_FLAG
_REPLACEMENT_CHARACTER = "\ufffd"


class _Rect(Protocol):
    width: float
    height: float


class _Matrix(Protocol):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


class _Pixmap(Protocol):
    width: int
    height: int

    def tobytes(self, output: str) -> bytes: ...


class _Page(Protocol):
    cropbox: _Rect
    rect: _Rect
    rotation: int
    derotation_matrix: _Matrix

    def annot_xrefs(self) -> Sequence[object]: ...

    def widgets(self) -> Sequence[object] | None: ...

    def get_pixmap(self, *, dpi: int, alpha: bool) -> _Pixmap: ...

    def get_text(self, option: str, **kwargs: object) -> object: ...

    def get_image_info(self) -> object: ...


class _Document(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _Page: ...

    def embfile_count(self) -> int: ...

    def close(self) -> None: ...


class _PymupdfModule(Protocol):
    TEXTFLAGS_RAWDICT: int
    TEXT_PRESERVE_IMAGES: int
    open: Callable[..., _Document]


class PymupdfDocumentAdapter:
    def __init__(
        self,
        *,
        dpi: int = DEFAULT_RENDER_DPI,
        meaningful_raster_area_ratio: float = DEFAULT_MEANINGFUL_RASTER_AREA_RATIO,
        max_raster_pixels: int = DEFAULT_MAX_RASTER_PIXELS,
    ) -> None:
        validate_render_settings(
            dpi=dpi,
            meaningful_raster_area_ratio=meaningful_raster_area_ratio,
            max_raster_pixels=max_raster_pixels,
        )
        self._dpi = dpi
        self._meaningful_raster_area_ratio = meaningful_raster_area_ratio
        self._max_raster_pixels = max_raster_pixels

    def inspect(self, source: PdfSource) -> DocumentInspection:
        pymupdf = _load_pymupdf()
        # reason: PymupdfDocumentAda's try keeps inspect page with close; splitting would mix coordinate frames.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            document = _open_document(pymupdf, source)
            try:
                pages = tuple(self._inspect_page(document[index], index) for index in range(len(document)))
                return DocumentInspection(
                    pages=pages,
                    attachment_count=int(document.embfile_count()),
                )
            finally:
                document.close()
        except DocumentAdapterError:
            raise
        # reason: PyMuPDF exposes backend-specific parse failures; the adapter boundary returns one stable domain error.
        except Exception:  # ruff: ignore[blind-except]
            msg = "document_read"
            raise DocumentAdapterError(msg) from None

    def render_page(self, source: PdfSource, *, page_index: int) -> RasterArtifact:
        return self.render_pages(source, page_indices=(page_index,))[0]

    def render_pages(self, source: PdfSource, *, page_indices: Sequence[int]) -> tuple[RasterArtifact, ...]:
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

        if not indices:
            return ()

        pymupdf = _load_pymupdf()
        # reason: PymupdfDocumentAda's try keeps render page with close; splitting would mix coordinate frames.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            document = _open_document(pymupdf, source)
            try:
                for page_index in indices:
                    if page_index >= len(document):
                        msg = "page_index_out_of_range"
                        raise DocumentAdapterError(msg, page_index=page_index)
                return tuple(self._render_page(document[page_index], page_index) for page_index in indices)
            finally:
                document.close()
        except DocumentAdapterError:
            raise
        # reason: document opening and indexed page rendering share PyMuPDF's unstable exception surface.
        except Exception:  # ruff: ignore[blind-except]
            msg = "document_read"
            raise DocumentAdapterError(msg) from None

    def _inspect_page(self, page: _Page, page_index: int) -> PageInspection:
        # reason: PymupdfDocumentAda's try keeps inspect text with inspect; splitting would mix coordinate frames.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            evidence = _inspect_text(page)
            raster_area_ratio, displayed_image_count = _inspect_rasters(page)
            annotation_count = len(page.annot_xrefs())
            widget_count = sum(1 for _ in page.widgets() or ())
            diagnostics = trust_diagnostics(
                evidence=evidence,
                annotation_count=annotation_count,
                widget_count=widget_count,
            )
            signals = PageRouteSignals(
                page_index=page_index,
                visible_native_characters=evidence.visible_characters,
                has_meaningful_raster=(
                    raster_area_ratio > 0.0 and raster_area_ratio >= self._meaningful_raster_area_ratio
                ),
                native_text_trusted=not diagnostics,
                trust_diagnostics=diagnostics,
            )
            return PageInspection(
                signals=signals,
                decision=classify_page(signals),
                raster_area_ratio=raster_area_ratio,
                displayed_image_count=displayed_image_count,
                invisible_native_characters=evidence.invisible_characters,
                replacement_characters=evidence.replacement_characters,
                annotation_count=annotation_count,
                widget_count=widget_count,
                rotation_degrees=int(page.rotation),
            )
        except DocumentAdapterError:
            raise
        # reason: PyMuPDF page inspection failures are normalized to the page-scoped adapter contract.
        except Exception:  # ruff: ignore[blind-except]
            msg = "page_inspection"
            raise DocumentAdapterError(msg, page_index=page_index) from None

    def _render_page(self, page: _Page, page_index: int) -> RasterArtifact:
        # reason: PymupdfDocumentAda's try keeps get pixmap with to page; splitting would mix coordinate frames.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            cropbox = page.cropbox
            width_pt = float(cropbox.width)
            height_pt = float(cropbox.height)
            display_width_pt = float(page.rect.width)
            display_height_pt = float(page.rect.height)
            estimated_width_px = math.ceil(display_width_pt * self._dpi / 72.0)
            estimated_height_px = math.ceil(display_height_pt * self._dpi / 72.0)
            if estimated_width_px * estimated_height_px > self._max_raster_pixels:
                msg = "raster_pixel_limit"
                # reason: keep this resource guard inside the PyMuPDF page-render translation boundary.
                raise DocumentAdapterError(msg, page_index=page_index)  # ruff: ignore[raise-within-try]

            pixmap = page.get_pixmap(dpi=self._dpi, alpha=False)
            png_bytes = bytes(pixmap.tobytes("png"))
            derotation = page.derotation_matrix
            pixel_to_page = PixelToPageTransform(
                a=float(derotation.a) * display_width_pt / pixmap.width,
                b=float(derotation.b) * display_width_pt / pixmap.width,
                c=float(derotation.c) * display_height_pt / pixmap.height,
                d=float(derotation.d) * display_height_pt / pixmap.height,
                e=float(derotation.e),
                f=float(derotation.f),
            )
            raster = RasterPage(
                page_index=page_index,
                image=png_bytes,
                width_px=int(pixmap.width),
                height_px=int(pixmap.height),
                width_pt=width_pt,
                height_pt=height_pt,
                pixel_to_page=pixel_to_page,
            )
            return RasterArtifact(
                raster=raster,
                png_bytes=png_bytes,
                sha256=hashlib.sha256(png_bytes).hexdigest(),
                dpi=self._dpi,
            )
        except DocumentAdapterError:
            raise
        # reason: pixmap creation, PNG conversion, and geometry conversion share one stable page-render failure contract.
        except Exception:  # ruff: ignore[blind-except]
            msg = "page_render"
            raise DocumentAdapterError(msg, page_index=page_index) from None


def _load_pymupdf() -> _PymupdfModule:
    try:
        module = importlib.import_module("pymupdf")
    except ModuleNotFoundError:
        msg = "dependency_unavailable"
        raise DocumentAdapterError(msg) from None
    return cast("_PymupdfModule", module)


def _open_document(module: _PymupdfModule, source: PdfSource) -> _Document:
    open_pdf = module.open
    if isinstance(source, bytes):
        return open_pdf(stream=source, filetype="pdf")
    return open_pdf(str(source))


def _inspect_text(page: _Page) -> TextEvidence:
    pymupdf = _load_pymupdf()
    text_flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    raw = require_mapping(page.get_text("rawdict", flags=text_flags, sort=False))
    visible_characters = 0
    invisible_characters = 0
    replacement_characters = 0

    for raw_block in require_sequence(raw.get("blocks")):
        block = require_mapping(raw_block)
        if block.get("type") != 0:
            continue
        for raw_line in require_sequence(block.get("lines")):
            line = require_mapping(raw_line)
            for raw_span in require_sequence(line.get("spans")):
                span = require_mapping(raw_span)
                alpha = require_int(span.get("alpha"))
                character_flags = require_int(span.get("char_flags"))
                is_painted = alpha > 0 and bool(character_flags & _VISIBLE_TEXT_FLAGS)
                for raw_character in require_sequence(span.get("chars")):
                    character = require_mapping(raw_character)
                    value = character.get("c")
                    if not isinstance(value, str) or not value:
                        msg = "invalid character"
                        raise ValueError(msg)
                    if value.isspace():
                        continue
                    if value == _REPLACEMENT_CHARACTER:
                        replacement_characters += 1
                    if is_painted:
                        visible_characters += 1
                    else:
                        invisible_characters += 1

    return TextEvidence(
        visible_characters=visible_characters,
        invisible_characters=invisible_characters,
        replacement_characters=replacement_characters,
    )


def _inspect_rasters(page: _Page) -> tuple[float, int]:
    rectangles: list[tuple[float, float, float, float]] = []
    cropbox = page.cropbox
    page_width = float(cropbox.width)
    page_height = float(cropbox.height)

    images = require_sequence(page.get_image_info())
    for raw_image in images:
        image = require_mapping(raw_image)
        raw_bbox = require_sequence(image.get("bbox"))
        if len(raw_bbox) != 4:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            msg = "invalid image bounds"
            raise ValueError(msg)
        x0, y0, x1, y1 = (require_real(value) for value in raw_bbox)
        clipped = (
            max(0.0, min(page_width, x0)),
            max(0.0, min(page_height, y0)),
            max(0.0, min(page_width, x1)),
            max(0.0, min(page_height, y1)),
        )
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            rectangles.append(clipped)

    page_area = page_width * page_height
    if page_area <= 0.0:
        msg = "invalid page area"
        raise ValueError(msg)
    return min(1.0, rectangle_union_area(rectangles) / page_area), len(images)
