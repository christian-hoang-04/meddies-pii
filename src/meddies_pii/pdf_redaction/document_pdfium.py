from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import hashlib
import importlib
import io
import math
from collections.abc import Callable, Iterable, Sequence, Sized
from typing import Protocol, TypeAlias, cast, runtime_checkable

from meddies_pii.pdf_redaction.document_pdfium_support import (
    _inspect_pdfium_page_objects,
    _load_pdfium,
    _pdfium_pixel_to_page_transform,
    _PdfiumDocument,
    _PdfiumModule,
    _validate_page_indices,
    require_box,
)
from meddies_pii.pdf_redaction.document_types import (
    DEFAULT_MAX_RASTER_PIXELS,
    DEFAULT_MEANINGFUL_RASTER_AREA_RATIO,
    DEFAULT_RENDER_DPI,
    DocumentAdapterError,
    DocumentInspection,
    PageInspection,
    PdfiumDocumentAdapterError,
    PdfSource,
    RasterArtifact,
    require_real,
    trust_diagnostics,
    validate_render_settings,
)
from meddies_pii.pdf_redaction.ocr import RasterPage
from meddies_pii.pdf_redaction.routing import PageRouteSignals, classify_page


@runtime_checkable
class _PypdfPage(Protocol):
    def get(self, key: str, default: object) -> object: ...


@runtime_checkable
class _PypdfReader(Protocol):
    pages: Sequence[_PypdfPage]
    attachments: Sized

    def close(self) -> None: ...


_PypdfReaderFactory: TypeAlias = Callable[..., object]
"""The pypdf entry points this module reaches through `getattr`.

Named so the calls type-check against a declared shape rather than against `object`. The reader factory takes `strict` as a
keyword, which `Callable` cannot spell, so only its arguments stay unconstrained.

"""
_PypdfGetter: TypeAlias = Callable[[str, object], object]
_PypdfObjectResolver: TypeAlias = Callable[[], object]


def _open_pypdf_reader(source: PdfSource) -> _PypdfReader:
    module = importlib.import_module("pypdf")
    reader_factory: _PypdfReaderFactory | None = getattr(module, "PdfReader", None)
    if not callable(reader_factory):
        msg = "pypdf.PdfReader is unavailable"
        raise ValueError(msg)
    reader: object
    if isinstance(source, bytes):
        reader = reader_factory(io.BytesIO(source), strict=False)
    else:
        reader = reader_factory(source, strict=False)
    if not isinstance(reader, _PypdfReader):
        msg = "pypdf.PdfReader returned an invalid reader"
        raise ValueError(msg)
    return reader


def _inspect_pypdf_annotations(page: _PypdfPage) -> tuple[int, int]:
    annotations: object = page.get("/Annots", ())
    if not isinstance(annotations, Iterable):
        msg = "invalid PDF annotations"
        raise ValueError(msg)
    annotation_count = 0
    widget_count = 0
    for reference in annotations:
        get_object: _PypdfObjectResolver | None = getattr(reference, "get_object", None)
        if not callable(get_object):
            msg = "invalid PDF annotation reference"
            raise ValueError(msg)
        annotation: object = get_object()
        get: _PypdfGetter | None = getattr(annotation, "get", None)
        if not callable(get):
            msg = "invalid PDF annotation"
            raise ValueError(msg)
        if str(get("/Subtype", "")) == "/Widget":
            widget_count += 1
        else:
            annotation_count += 1
    return annotation_count, widget_count


class PdfiumDocumentAdapter:
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
        try:
            pdfium = cast("_PdfiumModule", importlib.import_module("pypdfium2"))
        except ModuleNotFoundError:
            msg = "dependency_unavailable"
            raise PdfiumDocumentAdapterError(msg) from None

        pdfium_document: _PdfiumDocument | None = None
        pypdf_reader: _PypdfReader | None = None
        # reason: PdfiumDocumentAdap's try keeps inspect page with pypdf reader; splitting would mix coordinate frames.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            pdfium_document = pdfium.PdfDocument(source)
            pypdf_reader = _open_pypdf_reader(source)
            if len(pdfium_document) != len(pypdf_reader.pages):
                msg = "PDF parsers disagree on page count"
                # reason: keep parser disagreement inside the cleanup and adapter-translation boundary for both handles.
                raise ValueError(msg)  # ruff: ignore[raise-within-try]
            pages = tuple(
                self._inspect_page(
                    pdfium_document,
                    pypdf_reader.pages[page_index],
                    page_index,
                )
                for page_index in range(len(pdfium_document))
            )
            return DocumentInspection(
                pages=pages,
                attachment_count=len(pypdf_reader.attachments),
            )
        except DocumentAdapterError:
            raise
        # reason: Pdfium and pypdf expose version-specific failures; this adapter translates
        # reason: all of them to one stable code.
        except Exception:  # ruff: ignore[blind-except]
            msg = "document_read"
            raise PdfiumDocumentAdapterError(msg) from None
        finally:
            if pypdf_reader is not None:
                pypdf_reader.close()
            if pdfium_document is not None:
                pdfium_document.close()

    def render_page(self, source: PdfSource, *, page_index: int) -> RasterArtifact:
        return self.render_pages(source, page_indices=(page_index,))[0]

    def render_pages(self, source: PdfSource, *, page_indices: Sequence[int]) -> tuple[RasterArtifact, ...]:
        indices = _validate_page_indices(page_indices)
        if not indices:
            return ()

        try:
            pdfium = cast("_PdfiumModule", importlib.import_module("pypdfium2"))
        except ModuleNotFoundError:
            msg = "dependency_unavailable"
            raise PdfiumDocumentAdapterError(msg) from None

        document: _PdfiumDocument | None = None
        try:
            document = pdfium.PdfDocument(source)
            document.init_forms()
            for page_index in indices:
                if page_index >= len(document):
                    msg = "page_index_out_of_range"
                    # reason: the open Pdfium document must close before this page-scoped domain error escapes unchanged.
                    raise PdfiumDocumentAdapterError(msg, page_index=page_index)  # ruff: ignore[raise-within-try]
            return tuple(self._render_page(document, page_index) for page_index in indices)
        except DocumentAdapterError:
            raise
        # reason: Pdfium can fail while opening, initializing forms, or rendering; callers consume the stable adapter code.
        except Exception:  # ruff: ignore[blind-except]
            msg = "document_read"
            raise PdfiumDocumentAdapterError(msg) from None
        finally:
            if document is not None:
                document.close()

    def _inspect_page(
        self,
        document: _PdfiumDocument,
        pypdf_page: _PypdfPage,
        page_index: int,
    ) -> PageInspection:
        page = document[page_index]
        text_page = page.get_textpage()
        try:
            evidence, raster_area_ratio, displayed_image_count = _inspect_pdfium_page_objects(
                page,
                text_page,
                pdfium=_load_pdfium(),
            )
            annotation_count, widget_count = _inspect_pypdf_annotations(pypdf_page)
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
                rotation_degrees=int(page.get_rotation()),
            )
        # reason: page inspection crosses Pdfium and pypdf APIs whose exception sets are not stable across releases.
        except Exception:  # ruff: ignore[blind-except]
            msg = "page_inspection"
            raise PdfiumDocumentAdapterError(msg, page_index=page_index) from None
        finally:
            text_page.close()
            page.close()

    # reason: render and close share PdfiumDocumentAdap's state; extraction would mix coordinate frames.
    def _render_page(self, document: _PdfiumDocument, page_index: int) -> RasterArtifact:  # ruff: ignore[too-many-locals]
        page = document[page_index]
        bitmap = None
        image = None
        rgb_image = None
        # reason: PdfiumDocumentAdap's try keeps render with close; splitting would mix coordinate frames.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            crop_left, crop_bottom, crop_right, crop_top = require_box(page.get_cropbox())
            width_pt = crop_right - crop_left
            height_pt = crop_top - crop_bottom
            display_width_pt, display_height_pt = (require_real(value) for value in page.get_size())
            if width_pt <= 0.0 or height_pt <= 0.0:
                msg = "invalid page area"
                # reason: invalid geometry must pass through the page-render translation and cleanup boundary.
                raise ValueError(msg)  # ruff: ignore[raise-within-try]
            estimated_width_px = math.ceil(display_width_pt * self._dpi / 72.0)
            estimated_height_px = math.ceil(display_height_pt * self._dpi / 72.0)
            if estimated_width_px * estimated_height_px > self._max_raster_pixels:
                msg = "raster_pixel_limit"
                # reason: the open page must close before this page-scoped resource-limit error escapes unchanged.
                raise PdfiumDocumentAdapterError(msg, page_index=page_index)  # ruff: ignore[raise-within-try]

            bitmap = page.render(
                scale=self._dpi / 72.0,
                may_draw_forms=True,
                draw_annots=True,
                fill_color=(255, 255, 255, 255),
                rev_byteorder=True,
            )
            image = bitmap.to_pil()
            rgb_image = image.convert("RGB")
            buffer = io.BytesIO()
            rgb_image.save(buffer, format="PNG", optimize=False, compress_level=6)
            png_bytes = buffer.getvalue()
            width_px, height_px = rgb_image.size
            raster = RasterPage(
                page_index=page_index,
                image=png_bytes,
                width_px=width_px,
                height_px=height_px,
                width_pt=width_pt,
                height_pt=height_pt,
                pixel_to_page=_pdfium_pixel_to_page_transform(
                    rotation_degrees=int(page.get_rotation()),
                    width_pt=width_pt,
                    height_pt=height_pt,
                    width_px=width_px,
                    height_px=height_px,
                ),
            )
            return RasterArtifact(
                raster=raster,
                png_bytes=png_bytes,
                sha256=hashlib.sha256(png_bytes).hexdigest(),
                dpi=self._dpi,
            )
        except DocumentAdapterError:
            raise
        # reason: Pdfium, Pillow, and buffer encoding errors all map to the same page-scoped render failure contract.
        except Exception:  # ruff: ignore[blind-except]
            msg = "page_render"
            raise PdfiumDocumentAdapterError(msg, page_index=page_index) from None
        finally:
            if rgb_image is not None:
                rgb_image.close()
            if image is not None:
                image.close()
            if bitmap is not None:
                bitmap.close()
            page.close()
