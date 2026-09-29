from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import csv
import math
from dataclasses import dataclass
from io import StringIO
from numbers import Real
from typing import TYPE_CHECKING, cast

from meddies_pii.pdf_redaction.contracts import GeometryPage, GeometryToken, Quad
from meddies_pii.pdf_redaction.document_types import require_sequence
from meddies_pii.pdf_redaction.ocr_backends import RapidOcrEngineFactory
from meddies_pii.pdf_redaction.ocr_types import (
    OcrAdapterError,
    RapidEngine,
    RapidOcrRuntimeConfig,
    RasterPage,
    TesseractRunner,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


TESSERACT_WORD_LEVEL = 5


@dataclass(frozen=True, slots=True)
class _OcrWord:
    text: str
    confidence: float
    polygon_px: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]


class RapidOcrAdapter:
    def __init__(self, *, engine_factory: Callable[[], RapidEngine] | None = None) -> None:
        self._engine_factory = engine_factory or RapidOcrEngineFactory(
            RapidOcrRuntimeConfig(backend="openvino", model_tier="tiny"),
        )
        self._engine: RapidEngine | None = None

    def extract(self, raster: RasterPage) -> GeometryPage:
        try:
            output = self._get_engine()(
                raster.image,
                return_word_box=True,
                return_single_char_box=False,
            )
            line_texts, words = _parse_rapid_output(output)
            return _page_from_aligned_words(raster, line_texts, words)
        except OcrAdapterError as error:
            raise _page_scoped_error(error, raster.page_index) from None
        # reason: arbitrary RapidOCR engine and output-shape failures must become the page-scoped adapter error.
        except Exception:  # ruff: ignore[blind-except]
            msg = "rapidocr"
            raise OcrAdapterError(msg, "invalid_output", page_index=raster.page_index) from None

    def _get_engine(self) -> RapidEngine:
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine


class TesseractTsvAdapter:
    def __init__(self, *, runner_factory: Callable[[], TesseractRunner]) -> None:
        self._runner_factory = runner_factory
        self._runner: TesseractRunner | None = None

    def extract(self, raster: RasterPage) -> GeometryPage:
        try:
            tsv = self._get_runner().run(raster.image)
            line_texts, words = _parse_tesseract_tsv(tsv)
            return _page_from_aligned_words(raster, line_texts, words)
        except OcrAdapterError as error:
            raise _page_scoped_error(error, raster.page_index) from None
        # reason: arbitrary Tesseract runner and TSV-shape failures must become the page-scoped adapter error.
        except Exception:  # ruff: ignore[blind-except]
            msg = "tesseract"
            raise OcrAdapterError(msg, "invalid_tsv", page_index=raster.page_index) from None

    def _get_runner(self) -> TesseractRunner:
        if self._runner is None:
            self._runner = self._runner_factory()
        return self._runner


def _page_scoped_error(error: OcrAdapterError, page_index: int) -> OcrAdapterError:
    if error.page_index is not None:
        return error
    return OcrAdapterError(error.engine, error.stage, page_index=page_index)


def _parse_rapid_output(
    output: object,
) -> tuple[tuple[str, ...], tuple[tuple[_OcrWord, ...], ...]]:
    line_texts_raw = getattr(output, "txts", None)
    word_results_raw = getattr(output, "word_results", None)
    if line_texts_raw is None:
        return (), ()
    line_values = require_sequence(line_texts_raw)
    word_values = require_sequence(word_results_raw)

    line_texts: list[str] = []
    for value in line_values:
        if not isinstance(value, str):
            msg = "invalid line text"
            raise ValueError(msg)
        line_texts.append(value)
    if len(line_texts) != len(word_values):
        msg = "line and word result counts differ"
        raise ValueError(msg)

    word_lines: list[tuple[_OcrWord, ...]] = []
    for line_words_raw in word_values:
        line_words = require_sequence(line_words_raw)
        parsed_line: list[_OcrWord] = []
        for value in line_words:
            parts = require_sequence(value)
            if len(parts) != 3 or not isinstance(parts[0], str) or not parts[0]:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
                msg = "invalid word result"
                raise ValueError(msg)
            confidence = _require_confidence(parts[1])
            polygon = _require_polygon(parts[2])
            parsed_line.append(
                _OcrWord(
                    text=parts[0],
                    confidence=confidence,
                    polygon_px=polygon,
                ),
            )
        word_lines.append(tuple(parsed_line))
    return tuple(line_texts), tuple(word_lines)


def _parse_tesseract_tsv(
    tsv: str,
) -> tuple[tuple[str, ...], tuple[tuple[_OcrWord, ...], ...]]:
    reader = csv.DictReader(StringIO(tsv), delimiter="\t")
    required_fields = {
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    }
    if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
        msg = "missing TSV fields"
        raise ValueError(msg)

    line_texts: list[str] = []
    word_lines: list[tuple[_OcrWord, ...]] = []
    active_line: tuple[int, int, int, int] | None = None
    active_text: list[str] = []
    active_words: list[_OcrWord] = []

    for row in reader:
        if _parse_int(row.get("level")) != TESSERACT_WORD_LEVEL:
            continue
        text = row.get("text")
        if not text or text.isspace():
            continue
        line_key = (
            _parse_int(row.get("page_num")),
            _parse_int(row.get("block_num")),
            _parse_int(row.get("par_num")),
            _parse_int(row.get("line_num")),
        )
        if active_line is not None and line_key != active_line:
            line_texts.append(" ".join(active_text))
            word_lines.append(tuple(active_words))
            active_text = []
            active_words = []
        active_line = line_key
        active_text.append(text)

        left = _parse_float(row.get("left"))
        top = _parse_float(row.get("top"))
        width = _parse_float(row.get("width"))
        height = _parse_float(row.get("height"))
        if width <= 0 or height <= 0:
            msg = "invalid TSV box"
            raise ValueError(msg)
        confidence = _require_confidence(_parse_float(row.get("conf")) / 100.0)
        active_words.append(
            _OcrWord(
                text=text,
                confidence=confidence,
                polygon_px=(
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left, top + height),
                ),
            ),
        )

    if active_line is not None:
        line_texts.append(" ".join(active_text))
        word_lines.append(tuple(active_words))
    return tuple(line_texts), tuple(word_lines)


def _page_from_aligned_words(
    raster: RasterPage,
    line_texts: Sequence[str],
    word_lines: Sequence[Sequence[_OcrWord]],
) -> GeometryPage:
    page_text = "\n".join(line_texts)
    tokens: list[GeometryToken] = []
    line_start = 0

    for line_text, line_words in zip(line_texts, word_lines, strict=True):
        covered = [False] * len(line_text)
        search_start = 0
        for word in line_words:
            position = line_text.find(word.text, search_start)
            if position < 0:
                msg = "unmapped word result"
                raise ValueError(msg)
            token_start = line_start + position
            token_end = token_start + len(word.text)
            point_0, point_1, point_2, point_3 = word.polygon_px
            quad = Quad(
                points=(
                    raster.map_pixel(*point_0),
                    raster.map_pixel(*point_1),
                    raster.map_pixel(*point_2),
                    raster.map_pixel(*point_3),
                ),
            )
            if quad.signed_double_area() <= 0.0:
                msg = "polygon must use perimeter order with positive area"
                raise ValueError(msg)
            tokens.append(
                GeometryToken(
                    text=word.text,
                    quad=quad,
                    start=token_start,
                    end=token_end,
                    source="ocr",
                    confidence=word.confidence,
                ),
            )
            for index in range(position, position + len(word.text)):
                covered[index] = True
            search_start = position + len(word.text)

        if any(not character.isspace() and not covered[index] for index, character in enumerate(line_text)):
            msg = "unmapped line text"
            raise ValueError(msg)
        line_start += len(line_text) + 1
    return GeometryPage(
        page_index=raster.page_index,
        width_pt=raster.width_pt,
        height_pt=raster.height_pt,
        text=page_text,
        tokens=tuple(tokens),
    )


def _require_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = "confidence must be numeric"
        raise ValueError(msg)
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        msg = "confidence must be between zero and one"
        raise ValueError(msg)
    return confidence


def _require_polygon(
    value: object,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    points = require_sequence(value)
    if len(points) != 4:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "polygon must have four points"
        raise ValueError(msg)
    parsed = tuple(_require_pair(point) for point in points)
    return cast(
        "tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]",
        parsed,
    )


def _require_pair(value: object) -> tuple[float, float]:
    coordinates = require_sequence(value)
    if len(coordinates) != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "point must have two coordinates"
        raise ValueError(msg)
    x = _require_real(coordinates[0])
    y = _require_real(coordinates[1])
    if not math.isfinite(x) or not math.isfinite(y):
        msg = "point coordinates must be finite"
        raise ValueError(msg)
    return x, y


def _require_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = "expected numeric value"
        raise ValueError(msg)
    return float(value)


def _parse_int(value: object) -> int:
    if not isinstance(value, str):
        msg = "missing TSV integer"
        raise ValueError(msg)
    try:
        return int(value)
    except ValueError:
        msg = "invalid TSV integer"
        raise ValueError(msg) from None


def _parse_float(value: object) -> float:
    if not isinstance(value, str):
        msg = "missing TSV number"
        raise ValueError(msg)
    try:
        number = float(value)
    except ValueError:
        msg = "invalid TSV number"
        raise ValueError(msg) from None
    if not math.isfinite(number):
        msg = "TSV number must be finite"
        raise ValueError(msg)
    return number
