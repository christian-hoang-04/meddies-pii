from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import ctypes
import importlib
import io
from typing import TYPE_CHECKING, Protocol, cast

from meddies_pii.pdf_redaction.document_types import (
    PdfiumDocumentAdapterError,
    PdfSource,
    TextEvidence,
    require_real,
    require_sequence,
)
from meddies_pii.pdf_redaction.geometry import rectangle_union_area
from meddies_pii.pdf_redaction.ocr import PixelToPageTransform

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from PIL.Image import Image as PillowImage
    from pypdf import PdfReader
    from pypdf._page import PageObject

QUARTER_TURN_DEGREES = 90
HALF_TURN_DEGREES = 180

_REPLACEMENT_CHARACTER = "\ufffd"


class _PdfiumBitmap(Protocol):
    def to_pil(self) -> PillowImage: ...

    def close(self) -> None: ...


class _PdfiumRender(Protocol):
    def __call__(
        self,
        *,
        scale: float,
        may_draw_forms: bool,
        draw_annots: bool,
        fill_color: tuple[int, int, int, int],
        rev_byteorder: bool,
    ) -> _PdfiumBitmap: ...


class _PdfiumTextPage(Protocol):
    raw: object

    def close(self) -> None: ...


class _PdfiumPage(Protocol):
    render: _PdfiumRender

    def get_cropbox(self) -> object: ...

    def get_size(self) -> tuple[float, float]: ...

    def get_rotation(self) -> int: ...

    def get_textpage(self) -> _PdfiumTextPage: ...

    def get_objects(self, *, max_depth: int, textpage: _PdfiumTextPage) -> Iterable[object]: ...

    def close(self) -> None: ...


class _PdfiumDocument(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _PdfiumPage: ...

    def init_forms(self) -> None: ...

    def close(self) -> None: ...


class _PdfiumModule(Protocol):
    PdfDocument: Callable[[PdfSource], _PdfiumDocument]
    PdfTextObj: type[object]
    PdfImage: type[object]


class _PdfiumTextObject(Protocol):
    raw: object

    def extract(self) -> object: ...


class _PdfiumImageObject(Protocol):
    def get_bounds(self) -> object: ...


class _PdfiumRawModule(Protocol):
    FPDF_TEXTRENDERMODE_FILL: int
    FPDF_TEXTRENDERMODE_FILL_CLIP: int
    FPDF_TEXTRENDERMODE_FILL_STROKE: int
    FPDF_TEXTRENDERMODE_FILL_STROKE_CLIP: int
    FPDF_TEXTRENDERMODE_STROKE: int
    FPDF_TEXTRENDERMODE_STROKE_CLIP: int
    FPDFTextObj_GetTextRenderMode: Callable[[object], int]
    FPDFPageObj_GetFillColor: Callable[..., int]
    FPDFPageObj_GetStrokeColor: Callable[..., int]


def _open_pypdf(reader_type: type[PdfReader], source: PdfSource) -> PdfReader:
    if isinstance(source, bytes):
        return reader_type(io.BytesIO(source), strict=False)
    return reader_type(source, strict=False)


def _validate_page_indices(page_indices: Sequence[int]) -> tuple[int, ...]:
    indices = tuple(page_indices)
    seen: set[int] = set()
    for page_index in indices:
        if isinstance(page_index, bool) or not isinstance(page_index, int):
            msg = "invalid_page_index"
            raise PdfiumDocumentAdapterError(msg)
        if page_index < 0:
            msg = "page_index_out_of_range"
            raise PdfiumDocumentAdapterError(msg, page_index=page_index)
        if page_index in seen:
            msg = "duplicate_page_index"
            raise PdfiumDocumentAdapterError(msg, page_index=page_index)
        seen.add(page_index)
    return indices


def _pdfium_pixel_to_page_transform(
    *,
    rotation_degrees: int,
    width_pt: float,
    height_pt: float,
    width_px: int,
    height_px: int,
) -> PixelToPageTransform:
    rotation = rotation_degrees % 360
    if rotation not in {0, 90, 180, 270}:
        msg = "unsupported page rotation"
        raise ValueError(msg)
    display_width_pt = height_pt if rotation in {90, 270} else width_pt
    display_height_pt = width_pt if rotation in {90, 270} else height_pt
    scale_x = display_width_pt / width_px
    scale_y = display_height_pt / height_px
    if rotation == 0:
        return PixelToPageTransform(scale_x, 0.0, 0.0, scale_y, 0.0, 0.0)
    if rotation == QUARTER_TURN_DEGREES:
        return PixelToPageTransform(0.0, -scale_x, scale_y, 0.0, 0.0, height_pt)
    if rotation == HALF_TURN_DEGREES:
        return PixelToPageTransform(-scale_x, 0.0, 0.0, -scale_y, width_pt, height_pt)
    return PixelToPageTransform(0.0, scale_x, -scale_y, 0.0, width_pt, 0.0)


# reason: Object bounds, crop box, text evidence, and trust flags share one PDFium page frame.
def _inspect_pdfium_page_objects(  # ruff: ignore[too-many-locals]
    page: _PdfiumPage,
    text_page: _PdfiumTextPage,
    *,
    pdfium: _PdfiumModule,
) -> tuple[TextEvidence, float, int]:
    visible_characters = 0
    invisible_characters = 0
    replacement_characters = 0
    image_rectangles: list[tuple[float, float, float, float]] = []

    crop_left, crop_bottom, crop_right, crop_top = require_box(page.get_cropbox())
    page_width = crop_right - crop_left
    page_height = crop_top - crop_bottom
    if page_width <= 0.0 or page_height <= 0.0:
        msg = "invalid page area"
        raise ValueError(msg)

    for page_object in page.get_objects(max_depth=15, textpage=text_page):
        if isinstance(page_object, pdfium.PdfTextObj):
            text_object = cast("_PdfiumTextObject", page_object)
            value = text_object.extract()
            if not isinstance(value, str):
                msg = "invalid text object"
                raise ValueError(msg)
            nonspace = tuple(character for character in value if not character.isspace())
            replacement_characters += nonspace.count(_REPLACEMENT_CHARACTER)
            if _pdfium_text_object_is_visible(text_object):
                visible_characters += len(nonspace)
            else:
                invisible_characters += len(nonspace)
        elif isinstance(page_object, pdfium.PdfImage):
            image_object = cast("_PdfiumImageObject", page_object)
            x0, y0, x1, y1 = (require_real(value) for value in require_sequence(image_object.get_bounds()))
            clipped = (
                max(crop_left, min(crop_right, x0)) - crop_left,
                max(crop_bottom, min(crop_top, y0)) - crop_bottom,
                max(crop_left, min(crop_right, x1)) - crop_left,
                max(crop_bottom, min(crop_top, y1)) - crop_bottom,
            )
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                image_rectangles.append(clipped)

    page_area = page_width * page_height
    raster_area_ratio = min(1.0, rectangle_union_area(image_rectangles) / page_area)
    return (
        TextEvidence(
            visible_characters=visible_characters,
            invisible_characters=invisible_characters,
            replacement_characters=replacement_characters,
        ),
        raster_area_ratio,
        len(image_rectangles),
    )


def require_box(value: object) -> tuple[float, float, float, float]:
    raw_box = require_sequence(value)
    if len(raw_box) != 4:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "invalid page box"
        raise ValueError(msg)
    left, bottom, right, top = (require_real(coordinate) for coordinate in raw_box)
    return left, bottom, right, top


def _pdfium_text_object_is_visible(page_object: _PdfiumTextObject) -> bool:

    pdfium_raw = cast("_PdfiumRawModule", importlib.import_module("pypdfium2.raw"))

    raw_object = page_object.raw
    render_mode = int(pdfium_raw.FPDFTextObj_GetTextRenderMode(raw_object))
    fill_modes = {
        pdfium_raw.FPDF_TEXTRENDERMODE_FILL,
        pdfium_raw.FPDF_TEXTRENDERMODE_FILL_CLIP,
        pdfium_raw.FPDF_TEXTRENDERMODE_FILL_STROKE,
        pdfium_raw.FPDF_TEXTRENDERMODE_FILL_STROKE_CLIP,
    }
    stroke_modes = {
        pdfium_raw.FPDF_TEXTRENDERMODE_STROKE,
        pdfium_raw.FPDF_TEXTRENDERMODE_STROKE_CLIP,
        pdfium_raw.FPDF_TEXTRENDERMODE_FILL_STROKE,
        pdfium_raw.FPDF_TEXTRENDERMODE_FILL_STROKE_CLIP,
    }

    painted = False
    if render_mode in fill_modes:
        fill_alpha = ctypes.c_uint()
        red = ctypes.c_uint()
        green = ctypes.c_uint()
        blue = ctypes.c_uint()
        if pdfium_raw.FPDFPageObj_GetFillColor(
            raw_object,
            ctypes.byref(red),
            ctypes.byref(green),
            ctypes.byref(blue),
            ctypes.byref(fill_alpha),
        ):
            painted = fill_alpha.value > 0
    if render_mode in stroke_modes:
        stroke_alpha = ctypes.c_uint()
        red = ctypes.c_uint()
        green = ctypes.c_uint()
        blue = ctypes.c_uint()
        if pdfium_raw.FPDFPageObj_GetStrokeColor(
            raw_object,
            ctypes.byref(red),
            ctypes.byref(green),
            ctypes.byref(blue),
            ctypes.byref(stroke_alpha),
        ):
            painted = painted or stroke_alpha.value > 0
    return painted


def _load_pdfium() -> _PdfiumModule:
    return cast("_PdfiumModule", importlib.import_module("pypdfium2"))


def _inspect_pypdf_annotations(page: PageObject) -> tuple[int, int]:
    annotations = page.get("/Annots", ())
    annotation_count = 0
    widget_count = 0
    for reference in annotations:
        annotation = reference.get_object()
        if str(annotation.get("/Subtype", "")) == "/Widget":
            widget_count += 1
        else:
            annotation_count += 1
    return annotation_count, widget_count
