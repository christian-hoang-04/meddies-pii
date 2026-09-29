from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import importlib
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    GeometryToken,
    Point,
    Quad,
)
from meddies_pii.pdf_redaction.document_types import require_sequence

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PdfSource = str | Path
NativeGeometryPrecision = Literal["character_quad", "axis_aligned_character_box"]


MAX_UNICODE_CODEPOINT = 0x10FFFF


class NativeExtractionError(RuntimeError):
    def __init__(self, engine: str, stage: str, *, page_index: int | None = None) -> None:
        location = "document" if page_index is None else f"page {page_index}"
        super().__init__(f"{engine} native extraction failed at {location}: {stage}")
        self.engine = engine
        self.stage = stage
        self.page_index = page_index


class NativePdfExtractor(Protocol):
    geometry_precision: NativeGeometryPrecision

    def extract(self, source: PdfSource) -> tuple[GeometryPage, ...]: ...


class _PdfiumTextPage(Protocol):
    raw: object

    def count_chars(self) -> int: ...

    def get_charbox(self, index: int) -> object: ...

    def close(self) -> None: ...


class _PdfiumPage(Protocol):
    def get_bbox(self) -> object: ...

    def get_textpage(self) -> _PdfiumTextPage: ...

    def close(self) -> None: ...


class _PdfiumDocument(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _PdfiumPage: ...

    def close(self) -> None: ...


class _PdfiumModule(Protocol):
    PdfDocument: Callable[[str], _PdfiumDocument]


class _PdfiumRawModule(Protocol):
    FPDFText_GetUnicode: Callable[[object, int], int]


class _PymupdfPoint(Protocol):
    x: float
    y: float


class _PymupdfQuad(Protocol):
    ul: _PymupdfPoint
    ur: _PymupdfPoint
    lr: _PymupdfPoint
    ll: _PymupdfPoint


class _PymupdfRect(Protocol):
    width: float
    height: float


class _PymupdfPage(Protocol):
    cropbox: _PymupdfRect

    def get_text(self, option: str, **kwargs: object) -> object: ...


class _PymupdfDocument(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _PymupdfPage: ...

    def close(self) -> None: ...


class _PymupdfModule(Protocol):
    def open(self, path: str) -> _PymupdfDocument: ...

    def recover_char_quad(
        self,
        direction: tuple[float, float],
        span: dict[str, object],
        character: dict[str, object],
    ) -> _PymupdfQuad: ...


@dataclass(frozen=True, slots=True)
class _Glyph:
    text: str
    quad: Quad


class PymupdfExtractor:
    geometry_precision: NativeGeometryPrecision = "character_quad"

    def __init__(self, *, sort: bool = False) -> None:
        self._sort = sort

    def extract(self, source: PdfSource) -> tuple[GeometryPage, ...]:
        try:
            pymupdf = cast("_PymupdfModule", importlib.import_module("pymupdf"))
        except ModuleNotFoundError:
            msg = "pymupdf"
            raise NativeExtractionError(msg, "dependency_unavailable") from None

        try:
            document = pymupdf.open(str(source))
            try:
                return tuple(self._extract_page(document[index], index) for index in range(len(document)))
            finally:
                document.close()
        except NativeExtractionError:
            raise
        # reason: PyMuPDF parse failures are backend-specific; extraction exposes one stable document-read error.
        except Exception:  # ruff: ignore[blind-except]
            msg = "pymupdf"
            raise NativeExtractionError(msg, "document_read") from None

    # reason: Text blocks, glyph boxes, and page bounds share one PyMuPDF coordinate frame.
    def _extract_page(self, page: _PymupdfPage, page_index: int) -> GeometryPage:  # ruff: ignore[too-many-locals]
        pymupdf = cast("_PymupdfModule", importlib.import_module("pymupdf"))

        # reason: Glyph parsing and page-shape failures share one try so coordinates map to one page diagnostic.
        try:  # ruff: ignore[too-many-nested-blocks,too-many-statements-in-try-clause]
            cropbox = page.cropbox
            width_pt = float(cropbox.width)
            height_pt = float(cropbox.height)
            raw = _require_mapping(page.get_text("rawdict", sort=self._sort))
            blocks = require_sequence(raw.get("blocks"))
            glyph_blocks: list[list[list[_Glyph]]] = []

            for raw_block in blocks:
                block = _require_mapping(raw_block)
                if block.get("type") != 0:
                    continue
                glyph_lines: list[list[_Glyph]] = []
                for raw_line in require_sequence(block.get("lines")):
                    line = _require_mapping(raw_line)
                    line_direction = _require_pair(line.get("dir"))
                    glyphs: list[_Glyph] = []
                    for raw_span in require_sequence(line.get("spans")):
                        span = _require_mapping(raw_span)
                        for raw_character in require_sequence(span.get("chars")):
                            character = _require_mapping(raw_character)
                            text = character.get("c")
                            if not isinstance(text, str) or not text:
                                msg = "invalid character"
                                # reason: malformed backend text must become the enclosing page-scoped extraction error.
                                raise ValueError(msg)  # ruff: ignore[raise-within-try]
                            recovered = pymupdf.recover_char_quad(line_direction, span, character)
                            glyphs.append(
                                _Glyph(
                                    text=text,
                                    quad=Quad(
                                        points=(
                                            _external_point(recovered.ul),
                                            _external_point(recovered.ur),
                                            _external_point(recovered.lr),
                                            _external_point(recovered.ll),
                                        ),
                                    ),
                                ),
                            )
                    if glyphs:
                        glyph_lines.append(glyphs)
                if glyph_lines:
                    glyph_blocks.append(glyph_lines)

            return _page_from_glyph_blocks(
                page_index=page_index,
                width_pt=width_pt,
                height_pt=height_pt,
                glyph_blocks=glyph_blocks,
            )
        # reason: every PyMuPDF text or geometry failure maps to the same page-scoped extraction contract.
        except Exception:  # ruff: ignore[blind-except]
            msg = "pymupdf"
            raise NativeExtractionError(msg, "page_geometry", page_index=page_index) from None


class PdfiumExtractor:
    geometry_precision: NativeGeometryPrecision = "axis_aligned_character_box"

    def extract(self, source: PdfSource) -> tuple[GeometryPage, ...]:
        try:
            pdfium = cast("_PdfiumModule", importlib.import_module("pypdfium2"))
        except ModuleNotFoundError:
            msg = "pdfium"
            raise NativeExtractionError(msg, "dependency_unavailable") from None

        try:
            document = pdfium.PdfDocument(str(source))
            try:
                return tuple(self._extract_page(document[index], index) for index in range(len(document)))
            finally:
                document.close()
        except NativeExtractionError:
            raise
        # reason: PDFium parse failures are backend-specific; extraction exposes one stable document-read error.
        except Exception:  # ruff: ignore[blind-except]
            msg = "pdfium"
            raise NativeExtractionError(msg, "document_read") from None

    # reason: Character boxes, text decoding, and page bounds share one PDFium coordinate frame.
    @staticmethod
    def _extract_page(page: _PdfiumPage, page_index: int) -> GeometryPage:  # ruff: ignore[too-many-locals]
        pdfium_raw = cast("_PdfiumRawModule", importlib.import_module("pypdfium2.raw"))

        # reason: Text-page lifetime and glyph parsing share one try so failures retain the page index.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            page_left, page_bottom, page_right, page_top = _require_box(page.get_bbox())
            width_pt = page_right - page_left
            height_pt = page_top - page_bottom
            glyphs: list[_Glyph] = []

            text_page = page.get_textpage()
            try:
                for char_index in range(text_page.count_chars()):
                    codepoint = int(pdfium_raw.FPDFText_GetUnicode(text_page.raw, char_index))
                    if codepoint == 0 or codepoint > MAX_UNICODE_CODEPOINT:
                        msg = "invalid unicode map"
                        raise ValueError(msg)
                    text = chr(codepoint)

                    if text.isspace():
                        zero = Point(0.0, 0.0)
                        quad = Quad(points=(zero, zero, zero, zero))
                    else:
                        left, bottom, right, top = _require_box(text_page.get_charbox(char_index))
                        quad = Quad(
                            points=(
                                Point(left - page_left, page_top - top),
                                Point(right - page_left, page_top - top),
                                Point(right - page_left, page_top - bottom),
                                Point(left - page_left, page_top - bottom),
                            ),
                        )
                    glyphs.append(_Glyph(text=text, quad=quad))
            finally:
                text_page.close()

            glyph_blocks = [[glyphs]] if glyphs else []
            return _page_from_glyph_blocks(
                page_index=page_index,
                width_pt=width_pt,
                height_pt=height_pt,
                glyph_blocks=glyph_blocks,
            )
        # reason: every PDFium text or geometry failure maps to the same page-scoped extraction contract.
        except Exception:  # ruff: ignore[blind-except]
            msg = "pdfium"
            raise NativeExtractionError(msg, "page_geometry", page_index=page_index) from None


def _page_from_glyph_blocks(
    *,
    page_index: int,
    width_pt: float,
    height_pt: float,
    glyph_blocks: Sequence[Sequence[Sequence[_Glyph]]],
) -> GeometryPage:
    text_parts: list[str] = []
    tokens: list[GeometryToken] = []
    cursor = 0

    for block_index, lines in enumerate(glyph_blocks):
        if block_index:
            text_parts.append("\n\n")
            cursor += 2
        for line_index, glyphs in enumerate(lines):
            if line_index:
                text_parts.append("\n")
                cursor += 1
            for glyph in glyphs:
                start = cursor
                text_parts.append(glyph.text)
                cursor += len(glyph.text)
                if not glyph.text.isspace():
                    if glyph.quad.signed_double_area() <= 0.0:
                        msg = "polygon must use perimeter order with positive area"
                        raise ValueError(msg)
                    tokens.append(
                        GeometryToken(
                            text=glyph.text,
                            quad=glyph.quad,
                            start=start,
                            end=cursor,
                            source="native",
                            confidence=None,
                        ),
                    )

    return GeometryPage(
        page_index=page_index,
        width_pt=width_pt,
        height_pt=height_pt,
        text="".join(text_parts),
        tokens=tuple(tokens),
    )


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = "expected mapping"
        raise ValueError(msg)
    return cast("dict[str, object]", value)


def _require_pair(value: object) -> tuple[float, float]:
    values = require_sequence(value)
    if len(values) != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "expected pair"
        raise ValueError(msg)
    pair = (_require_real(values[0]), _require_real(values[1]))
    if not all(math.isfinite(item) for item in pair):
        msg = "pair must be finite"
        raise ValueError(msg)
    return pair


def _require_box(value: object) -> tuple[float, float, float, float]:
    values = require_sequence(value)
    if len(values) != 4:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "expected box"
        raise ValueError(msg)
    box = tuple(_require_real(item) for item in values)
    if not all(math.isfinite(item) for item in box):
        msg = "box must be finite"
        raise ValueError(msg)
    return cast("tuple[float, float, float, float]", box)


def _external_point(value: _PymupdfPoint) -> Point:
    x = float(value.x)
    y = float(value.y)
    if not math.isfinite(x) or not math.isfinite(y):
        msg = "point must be finite"
        raise ValueError(msg)
    return Point(x=x, y=y)


def _require_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = "expected numeric value"
        raise ValueError(msg)
    return float(value)
