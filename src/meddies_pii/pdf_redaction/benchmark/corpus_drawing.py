from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import hashlib
import importlib
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from meddies_pii.pdf_redaction.benchmark.corpus_records import (
    _DOMAIN,
    _FIXED_PDF_DATE,
    _FONT_CANDIDATES,
    _HEIGHT_PT,
    _WIDTH_PT,
    Degradation,
)
from meddies_pii.pdf_redaction.contracts import Point, Quad
from meddies_pii.taxonomy import PII_LABELS, PiiLabel

if TYPE_CHECKING:
    from collections.abc import Callable


def _fitz_point(x: float, y: float) -> fitz.Point:
    factory = cast("Callable[[float, float], fitz.Point]", fitz.Point)
    return factory(x, y)


def _fitz_rect(x0: float, y0: float, x1: float, y1: float) -> fitz.Rect:
    factory = cast("Callable[[float, float, float, float], fitz.Rect]", fitz.Rect)
    return factory(x0, y0, x1, y1)


def _fitz_font(path: Path | None = None) -> fitz.Font:
    factory = cast("Callable[..., fitz.Font]", fitz.Font)
    return factory("helv") if path is None else factory(fontfile=str(path))


def _font_text_length(font: fitz.Font, text: str, *, fontsize: int) -> float:
    method = cast("Callable[..., float]", font.text_length)
    return method(text, fontsize=fontsize)


def _page_insert_font(page: fitz.Page, *, fontname: str, fontfile: str) -> None:
    method = cast("Callable[..., object]", page.insert_font)
    method(fontname=fontname, fontfile=fontfile)


def _font_has_glyph(font: fitz.Font, codepoint: int) -> int:
    method = cast("Callable[[int], int]", font.has_glyph)
    return method(codepoint)


def _document_xref_length(document: fitz.Document) -> int:
    method = cast("Callable[[], int]", document.xref_length)
    return method()


def _document_xref_object(document: fitz.Document, xref: int) -> str:
    method = cast("Callable[[int], str]", document.xref_object)
    return method(xref)


def _document_xref_set_key(document: fitz.Document, xref: int, key: str, value: str) -> None:
    method = cast("Callable[[int, str, str], object]", document.xref_set_key)
    method(xref, key, value)


def _page_insert_image(page: fitz.Page, rectangle: fitz.Rect, png: bytes) -> None:
    method = cast("Callable[..., object]", page.insert_image)
    method(rectangle, stream=png)


def _page_set_rotation(page: fitz.Page, degrees: int) -> None:
    method = cast("Callable[[int], object]", page.set_rotation)
    method(degrees)


def _document_tobytes(document: fitz.Document, **kwargs: object) -> bytes:
    method = cast("Callable[..., bytes]", document.tobytes)
    return method(**kwargs)


def _document_close(document: fitz.Document) -> None:
    method = cast("Callable[[], object]", document.close)
    method()


def _document_set_xml_metadata(document: fitz.Document, xml: str) -> None:
    method = cast("Callable[[str], object]", document.set_xml_metadata)
    method(xml)


def _fitz_widget() -> fitz.Widget:
    factory = cast("Callable[[], fitz.Widget]", fitz.Widget)
    return factory()


def _annotation_set_info(annotation: fitz.Annot, **kwargs: str) -> None:
    method = cast("Callable[..., object]", annotation.set_info)
    method(**kwargs)


# reason: draw all raster keeps draw/x at its adapter seam; bundling would hide required inputs.
def _draw_all_raster_labels(  # ruff: ignore[too-many-arguments]
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    placements: list[tuple[PiiLabel, str, tuple[tuple[float, float], ...]]],
    series: str,
    *,
    start_y: int,
    x: int = 150,
) -> None:
    values: dict[PiiLabel, str] = {
        "address": f"{series} 88 Đường Thử Nghiệm, Hà Nội",
        "company_name": f"Công ty Tổng Hợp {series}",
        "date": f"2098-02-{10 + int(series[-1]):02d}",
        "email_address": f"{series.lower()}.scan@example.invalid",
        "human_name": f"Nguyễn Tổng Hợp {series}",
        "id_number": f"SYNTH-SCAN-ID-{series}-22",
        "phone_number": f"+84 900 100 {200 + int(series[-1])}",
        "private_url": f"private.example.invalid/scan/{series.lower()}",
        "secret": f"SYNTH-SCAN-SECRET-{series}",
    }
    for row, label in enumerate(PII_LABELS):
        value = values[label]
        quad = _draw_raster_value(
            draw,
            font,
            prefix=f"{label.replace('_', ' ').title()}: ",
            value=value,
            xy=(x, start_y + row * 175),
        )
        placements.append((label, value, quad))


# reason: draw degraded keeps image/seed at its adapter seam; bundling would hide required inputs.
def _draw_degraded_value(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    image: Image.Image,
    font: ImageFont.FreeTypeFont,
    value: str,
    xy: tuple[int, int],
    degradation: Degradation,
    seed: int,
) -> tuple[tuple[float, float], ...]:
    x, y = xy
    bbox = font.getbbox(value)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pad = 35
    crop_size = (
        int(min(760, text_width + 2 * pad)),
        int(text_height + 2 * pad),
    )
    crop = Image.new("RGB", crop_size, "white")
    crop_draw = ImageDraw.Draw(crop)
    fill: str | tuple[int, int, int] = "black"
    if degradation == "low_contrast":
        fill = (185, 185, 185)
    crop_draw.text((pad, pad - bbox[1]), value, fill=fill, font=font)
    text_box: tuple[float, float, float, float] = (
        pad,
        pad,
        min(crop.width - pad, pad + text_width),
        pad + text_height,
    )

    if degradation == "150_dpi":
        low_size = (max(1, crop.width * 2 // 3), max(1, crop.height * 2 // 3))
        crop = crop.resize(low_size, Image.Resampling.LANCZOS).resize(crop_size, Image.Resampling.BICUBIC)
    elif degradation == "skew":
        angle = 4.0
        crop = crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
        text_box = _rotate_box(text_box, crop_size, -math.radians(angle))
    elif degradation == "blur_noise":
        crop = crop.filter(ImageFilter.GaussianBlur(radius=2.2))
        raw = bytearray(crop.tobytes())
        state = seed
        for index, channel in enumerate(raw):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            raw[index] = max(0, min(255, channel + (state % 17) - 8))
        crop = Image.frombytes("RGB", crop.size, bytes(raw))
    elif degradation == "low_contrast":
        crop = ImageEnhance.Contrast(crop).enhance(0.65)

    image.paste(crop, xy)
    return tuple((x + px, y + py) for px, py in _box_to_points(text_box))


def _new_scan_canvas(*, dpi: int = 216) -> tuple[Image.Image, list[tuple[PiiLabel, str, tuple[tuple[float, float], ...]]]]:
    width = round(_WIDTH_PT * dpi / 72)
    height = round(_HEIGHT_PT * dpi / 72)
    return Image.new("RGB", (width, height), "white"), []


def _draw_scan_heading(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, text: str) -> None:
    draw.text((150, 110), text, fill=(15, 65, 70), font=font)
    draw.line((150, 205, 1680, 205), fill=(15, 65, 70), width=4)


def _draw_raster_value(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    *,
    prefix: str,
    value: str,
    xy: tuple[int, int],
) -> tuple[tuple[float, float], ...]:
    draw.text(xy, prefix + value, fill="black", font=font)
    prefix_width = draw.textlength(prefix, font=font)
    value_bbox = draw.textbbox((xy[0] + prefix_width, xy[1]), value, font=font)
    return _box_to_points(value_bbox)


def _heading(page: fitz.Page, title: str, *, font_path: Path | None = None) -> None:
    font_name = "helv"
    if font_path is not None:
        font_name = "benchmarkunicode"
        _page_insert_font(page, fontname=font_name, fontfile=str(font_path))
    page.insert_text((46, 48), title, fontname=font_name, fontsize=15)
    page.draw_line(_fitz_point(46, 60), _fitz_point(566, 60), color=(0.1, 0.35, 0.38))


def _resolve_raster_font() -> Path:
    for candidate in _FONT_CANDIDATES:
        if candidate.is_file() and _font_supports_challenge_text(candidate):
            return candidate
    try:
        reportlab = importlib.import_module("reportlab")
        reportlab_init = getattr(reportlab, "__file__", None)
    except ImportError as error:
        msg = "no deterministic raster font is available"
        raise RuntimeError(msg) from error
    if not isinstance(reportlab_init, str):
        msg = "installed ReportLab package path is unavailable"
        raise RuntimeError(msg)
    fallback = Path(reportlab_init).parent / "fonts" / "Vera.ttf"
    if fallback.is_file() and _font_supports_challenge_text(fallback):
        return fallback
    msg = "no deterministic TrueType font with Vietnamese glyph coverage is available"
    raise RuntimeError(msg)


def _font_supports_challenge_text(path: Path) -> bool:
    font = _fitz_font(path)
    return all(_font_has_glyph(font, ord(character)) > 0 for character in "Đệễộữ•")


def _raster_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(path), size=size)
    if not isinstance(font, ImageFont.FreeTypeFont):
        msg = "benchmark raster font must be a TrueType font"
        raise TypeError(msg)
    return font


def _digest(seed: int, fixture_id: str, channel: str, value: str) -> str:
    payload = b"\0".join((str(seed).encode(), fixture_id.encode(), channel.encode(), value.encode()))
    return hashlib.sha256(_DOMAIN + payload).hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _page_index(page: fitz.Page) -> int:
    page_index = page.number
    if not isinstance(page_index, int):
        msg = "benchmark page must belong to an open document"
        raise RuntimeError(msg)
    return page_index


def _rect_quad(x0: float, y0: float, x1: float, y1: float) -> Quad:
    return Quad(points=(Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)))


def _fitz_rect_quad(rect: fitz.Rect) -> Quad:
    return _rect_quad(rect.x0, rect.y0, rect.x1, rect.y1)


def _box_to_points(
    box: tuple[float, float, float, float],
) -> tuple[tuple[float, float], ...]:
    x0, y0, x1, y1 = box
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _rotate_box(
    box: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    angle: float,
) -> tuple[float, float, float, float]:
    cx = canvas_size[0] / 2
    cy = canvas_size[1] / 2
    rotated: list[tuple[float, float]] = []
    for x, y in _box_to_points(box):
        dx = x - cx
        dy = y - cy
        rotated.append((
            cx + dx * math.cos(angle) - dy * math.sin(angle),
            cy + dx * math.sin(angle) + dy * math.cos(angle),
        ))
    xs = tuple(point[0] for point in rotated)
    ys = tuple(point[1] for point in rotated)
    return min(xs), min(ys), max(xs), max(ys)


def _pixel_quad_to_page(quad: tuple[tuple[float, float], ...], image_size: tuple[int, int]) -> Quad:
    points = tuple(Point(x * _WIDTH_PT / image_size[0], y * _HEIGHT_PT / image_size[1]) for x, y in quad)
    return Quad(points=cast("tuple[Point, Point, Point, Point]", points))


def _pixel_quad_to_rect(
    quad: tuple[tuple[float, float], ...],
    image_size: tuple[int, int],
    destination: fitz.Rect,
) -> Quad:
    points = tuple(
        Point(
            destination.x0 + x * destination.width / image_size[0],
            destination.y0 + y * destination.height / image_size[1],
        )
        for x, y in quad
    )
    return Quad(points=cast("tuple[Point, Point, Point, Point]", points))


def _fix_embedded_file_dates(document: fitz.Document) -> None:
    for xref in range(1, _document_xref_length(document)):
        if "/Type /EmbeddedFile" not in _document_xref_object(document, xref):
            continue
        _document_xref_set_key(document, xref, "Params/CreationDate", f"({_FIXED_PDF_DATE})")
        _document_xref_set_key(document, xref, "Params/ModDate", f"({_FIXED_PDF_DATE})")
