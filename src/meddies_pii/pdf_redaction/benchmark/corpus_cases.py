from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, cast

import pymupdf as fitz
from PIL import Image, ImageDraw

from meddies_pii.pdf_redaction.benchmark.corpus_drawing import (
    _annotation_set_info,
    _digest,
    _document_set_xml_metadata,
    _draw_all_raster_labels,
    _draw_degraded_value,
    _draw_raster_value,
    _draw_scan_heading,
    _fitz_point,
    _fitz_rect,
    _fitz_rect_quad,
    _fitz_widget,
    _fix_embedded_file_dates,
    _font_text_length,
    _heading,
    _new_scan_canvas,
    _page_index,
    _page_insert_image,
    _page_set_rotation,
    _pixel_quad_to_page,
    _pixel_quad_to_rect,
    _png_bytes,
    _raster_font,
    _rect_quad,
)
from meddies_pii.pdf_redaction.benchmark.corpus_records import (
    _FIXED_PDF_DATE,
    _TEXT_WIDGET_TYPE,
    Degradation,
    NegativeControlGold,
)
from meddies_pii.pdf_redaction.contracts import Point, Quad
from meddies_pii.taxonomy import PII_LABELS, PiiLabel

if TYPE_CHECKING:
    from meddies_pii.pdf_redaction.benchmark.corpus_builder import _CorpusBuilder


class _MutableWidget(Protocol):
    field_name: str | None
    field_type: int | None
    field_value: str | None
    rect: fitz.Rect | None


# reason: add all labels keeps builder/row height at its adapter seam; bundling would hide required inputs.
def _add_all_labels(  # ruff: ignore[too-many-arguments]
    builder: _CorpusBuilder,
    page: fitz.Page,
    series: str,
    *,
    start_y: float,
    x: float = 54,
    row_height: float = 32,
) -> None:
    page_index = _page_index(page)
    values: dict[PiiLabel, str] = {
        "address": f"{series} 14 Lantern Way, Test City",
        "company_name": f"Synthetic Orbit Clinic {series}",
        "date": f"2099-01-{10 + page_index:02d}",
        "email_address": f"{series.lower()}.patient@example.invalid",
        "human_name": f"Mx Synthetic {series}",
        "id_number": f"SYNTH-ID-{series}-4491",
        "phone_number": f"+1 555 010 {1000 + page_index:04d}",
        "private_url": f"private.example.invalid/{series.lower()}/care",
        "secret": f"SYNTH-SECRET-{series}-K4N7",
    }
    for row, label in enumerate(PII_LABELS):
        builder.add_native_canary(
            page,
            label=label,
            value=values[label],
            baseline=(x, start_y + row * row_height),
            prefix=f"{label.replace('_', ' ').title()}: ",
        )


def _record_raster_placements(
    builder: _CorpusBuilder,
    page_index: int,
    image_size: tuple[int, int],
    placements: list[tuple[PiiLabel, str, tuple[tuple[float, float], ...]]],
) -> None:
    for label, value, pixel_quad in placements:
        builder.record_canary(
            page_index=page_index,
            label=label,
            value=value,
            channel="visible_raster",
            quads=(_pixel_quad_to_page(pixel_quad, image_size),),
        )


def _record_raster_control(
    builder: _CorpusBuilder,
    page_index: int,
    image_size: tuple[int, int],
    value: str,
    pixel_quad: tuple[tuple[float, float], ...],
) -> None:
    fixture_id = f"p{page_index + 1:02d}-decoy-01"
    control = NegativeControlGold(
        fixture_id=fixture_id,
        page_index=page_index,
        digest=_digest(builder.seed, fixture_id, "negative_control", value),
        quads=(_pixel_quad_to_page(pixel_quad, image_size),),
        raw_value=value,
    )
    builder.controls.append(control)
    builder.page_records[page_index].negative_controls.append(fixture_id)


# reason: add multiline keeps builder/prefix at its adapter seam; bundling would hide required inputs.
def _add_multiline_native_canary(  # ruff: ignore[too-many-arguments]
    builder: _CorpusBuilder,
    page: fitz.Page,
    *,
    label: PiiLabel,
    value: str,
    segments: tuple[str, ...],
    baselines: tuple[tuple[float, float], ...],
    prefix: str,
) -> None:
    if "".join(segments) != value or len(segments) != len(baselines):
        msg = "multiline canary segments must reconstruct the raw value"
        raise ValueError(msg)
    fontsize = 10
    quads: list[Quad] = []
    for index, (segment, baseline) in enumerate(zip(segments, baselines, strict=True)):
        line_prefix = prefix if index == 0 else ""
        point = _fitz_point(*baseline)
        page.insert_text(point, line_prefix + segment, fontname="helv", fontsize=fontsize)
        x0 = point.x + _font_text_length(builder.font, line_prefix, fontsize=fontsize)
        x1 = x0 + _font_text_length(builder.font, segment, fontsize=fontsize)
        y0 = point.y - builder.font.ascender * fontsize
        y1 = point.y - builder.font.descender * fontsize
        quads.append(_rect_quad(x0, y0, x1, y1))
    builder.record_canary(
        page_index=_page_index(page),
        label=label,
        value=value,
        channel="visible_native",
        quads=tuple(quads),
    )


# reason: add rotated keeps builder/prefix at its adapter seam; bundling would hide required inputs.
def _add_rotated_native_canary(  # ruff: ignore[too-many-arguments]
    builder: _CorpusBuilder,
    page: fitz.Page,
    *,
    label: PiiLabel,
    value: str,
    baseline: tuple[float, float],
    prefix: str,
) -> None:
    fontsize = 10
    point = _fitz_point(*baseline)
    page.insert_text(
        point,
        prefix + value,
        fontname="helv",
        fontsize=fontsize,
        rotate=90,
    )
    advance_start = _font_text_length(builder.font, prefix, fontsize=fontsize)
    advance_end = advance_start + _font_text_length(builder.font, value, fontsize=fontsize)
    vertical_top = -builder.font.ascender * fontsize
    vertical_bottom = -builder.font.descender * fontsize
    quad = Quad(
        points=(
            Point(point.x + vertical_top, point.y - advance_start),
            Point(point.x + vertical_top, point.y - advance_end),
            Point(point.x + vertical_bottom, point.y - advance_end),
            Point(point.x + vertical_bottom, point.y - advance_start),
        ),
    )
    builder.record_canary(
        page_index=_page_index(page),
        label=label,
        value=value,
        channel="visible_native",
        quads=(quad,),
    )


def _build_native_discharge(builder: _CorpusBuilder) -> None:
    page = builder.new_page(("native words are authoritative", "single column"))
    _heading(page, "DISCHARGE SUMMARY — SYNTHETIC")
    _add_all_labels(builder, page, "N1", start_y=105)
    page.insert_text(
        (54, 410),
        "Primary diagnosis: community-acquired pneumonia, resolved.",
        fontname="helv",
        fontsize=10,
    )
    builder.add_native_control(page, "age 54; oxygen 2 L/min", (54, 450))


def _build_native_form(builder: _CorpusBuilder) -> None:
    page = builder.new_page(("native form text is authoritative", "empty cells retained"))
    _heading(page, "PATIENT REGISTRATION FORM — SYNTHETIC")
    for x in (46, 306):
        for y in range(86, 406, 40):
            page.draw_rect(_fitz_rect(x, y, x + 250, y + 32), color=(0.4, 0.4, 0.4))
    _add_all_labels(builder, page, "N2", start_y=105, x=56, row_height=34)
    page.insert_text((328, 107), "[ ] unchecked", fontname="helv", fontsize=9)
    page.insert_text((328, 141), "EMPTY:", fontname="helv", fontsize=9)
    builder.add_native_control(page, "HbA1c 6.4%; 500 mg", (328, 190))


def _build_native_two_column(builder: _CorpusBuilder) -> None:
    page = builder.new_page(("native words are authoritative", "two-column order trap"))
    _heading(page, "CLINICAL HISTORY — SYNTHETIC")
    page.insert_text((46, 82), "HISTORY", fontname="helv", fontsize=9)
    page.insert_text((326, 82), "HISTORY", fontname="helv", fontsize=9)
    builder.add_native_canary(
        page,
        label="human_name",
        value="Dr Synthetic N3",
        baseline=(46, 120),
        prefix="Clinician: ",
    )
    builder.add_native_canary(
        page,
        label="address",
        value="31 Split Column Lane",
        baseline=(326, 120),
        prefix="Address line 1: ",
    )
    builder.add_native_canary(
        page,
        label="address",
        value="Ward Nine, Test City",
        baseline=(326, 140),
        prefix="Address line 2: ",
    )
    page.insert_text((46, 170), "Left paragraph continues here.", fontname="helv", fontsize=10)
    page.insert_text((326, 170), "Right paragraph continues here.", fontname="helv", fontsize=10)
    builder.add_native_control(page, "sodium 139 mmol/L", (46, 720))


def _build_native_rotated(builder: _CorpusBuilder) -> None:
    page = builder.new_page(("native words are authoritative", "page rotation normalized"))
    _heading(
        page,
        "ROTATED CLINICAL NOTE — SYNTHETIC",
        font_path=builder.raster_font_path,
    )
    builder.add_native_canary(
        page,
        label="id_number",
        value="SYNTH-ROT-4401",
        baseline=(72, 150),
        prefix="Record: ",
    )
    _add_rotated_native_canary(
        builder,
        page,
        label="phone_number",
        value="+1 555 010 4404",
        baseline=(540, 650),
        prefix="Rotated phone: ",
    )
    page.insert_text(
        (72, 190),
        "Unicode punctuation: café, résumé, em dash — synthetic.",
        fontname="benchmarkunicode",
        fontsize=10,
    )
    builder.add_native_control(page, "pulse 72 bpm", (72, 230))
    _page_set_rotation(page, 90)
    builder.page_records[_page_index(page)].rotation_degrees = 90


def _build_hybrid_hidden_ocr(builder: _CorpusBuilder) -> None:
    image, placements = _new_scan_canvas()
    draw = ImageDraw.Draw(image)
    font = _raster_font(builder.raster_font_path, 34)
    _draw_scan_heading(draw, font, "VISIBLE SCAN — SYNTHETIC")
    placement = _draw_raster_value(
        draw,
        font,
        prefix="Visible patient: ",
        value="Visible Synthetic P5",
        xy=(150, 370),
    )
    placements.append(("human_name", "Visible Synthetic P5", placement))
    page = builder.add_raster_page(
        image,
        ("visible image and hidden text disagree", "hidden layer is untrusted"),
        source_dpi=216,
    )
    for label, value, pixel_quad in placements:
        builder.record_canary(
            page_index=_page_index(page),
            label=label,
            value=value,
            channel="visible_raster",
            quads=(_pixel_quad_to_page(pixel_quad, image.size),),
        )
    builder.add_native_canary(
        page,
        label="human_name",
        value="Hidden Different P5",
        baseline=(50, 140),
        channel="hidden_text",
        render_mode=3,
    )
    builder.add_native_control(page, "dose 20 mg", (54, 730))


def _build_image_clean(builder: _CorpusBuilder) -> None:
    image, placements = _new_scan_canvas(dpi=300)
    draw = ImageDraw.Draw(image)
    font = _raster_font(builder.raster_font_path, 30)
    _draw_scan_heading(draw, font, "ENGLISH + TIẾNG VIỆT — SYNTHETIC")
    _draw_all_raster_labels(draw, font, placements, "S1", start_y=300)
    control_value = "huyết áp 120/80"
    control_quad = _draw_raster_value(draw, font, prefix="Negative control: ", value=control_value, xy=(150, 2070))
    page = builder.add_raster_page(
        image,
        ("no native text objects", "clean 300 dpi scan"),
        source_dpi=300,
    )
    page_index = _page_index(page)
    _record_raster_placements(builder, page_index, image.size, placements)
    _record_raster_control(builder, page_index, image.size, control_value, control_quad)


def _build_image_degraded(builder: _CorpusBuilder) -> None:
    width, height = 1836, 2376
    image = Image.new("RGB", (width, height), "white")
    font = _raster_font(builder.raster_font_path, 34)
    values: tuple[tuple[PiiLabel, str, tuple[int, int], Degradation], ...] = (
        ("human_name", "Nguyễn Mờ P7", (120, 360), "150_dpi"),
        ("id_number", "SYNTH-SKEW-P7-91", (1020, 360), "skew"),
        (
            "email_address",
            "blur.p7 [at] example [dot] invalid",
            (120, 1510),
            "blur_noise",
        ),
        ("phone_number", "(+84) 900•111•P7", (980, 1510), "low_contrast"),
    )
    placements: list[tuple[PiiLabel, str, tuple[tuple[float, float], ...]]] = []
    draw = ImageDraw.Draw(image)
    draw.text((120, 100), "FOUR DEGRADATION ISLANDS", fill="black", font=font)
    for label, value, xy, degradation in values:
        quad = _draw_degraded_value(image, font, value, xy, degradation, builder.seed)
        placements.append((label, value, quad))
    control_value = "glucose 108 mg/dL"
    control_quad = _draw_raster_value(draw, font, prefix="Negative control: ", value=control_value, xy=(120, 2200))
    page = builder.add_raster_page(
        image,
        ("no native text objects", "four isolated degradation factors"),
        source_dpi=216,
    )
    page_index = _page_index(page)
    builder.page_records[page_index].degradations = tuple(item[3] for item in values)
    _record_raster_placements(builder, page_index, image.size, placements)
    _record_raster_control(builder, page_index, image.size, control_value, control_quad)


def _build_image_complex(builder: _CorpusBuilder) -> None:
    image, placements = _new_scan_canvas()
    draw = ImageDraw.Draw(image)
    font = _raster_font(builder.raster_font_path, 28)
    bold = _raster_font(builder.raster_font_path, 38)
    _draw_scan_heading(draw, bold, "COMPLEX CLINICAL SCAN")
    draw.rectangle((75, 240, 290, 2050), outline=(20, 70, 100), width=8)
    draw.text((105, 320), "S\nI\nD\nE\nB\nA\nR", fill=(20, 70, 100), font=font)
    draw.ellipse((1270, 250, 1710, 580), outline=(170, 30, 30), width=12)
    draw.text((1360, 385), "STAMP", fill=(170, 30, 30), font=bold)
    for index in range(4):
        y = 665 + index * 70
        draw.rectangle((1100, y, 1140, y + 40), outline="black", width=4)
        draw.text((1170, y), f"clinical option {index + 1}", fill="black", font=font)
    _draw_all_raster_labels(draw, font, placements, "S2", start_y=280, x=350)
    draw.line((1000, 1980, 1690, 1870, 1770, 1970), fill="black", width=6)
    draw.text((1120, 2020), "signature-like mark: policy only", fill="black", font=font)
    control_value = "temperature 37.1 C"
    control_quad = _draw_raster_value(draw, font, prefix="Negative control: ", value=control_value, xy=(1040, 2200))
    page = builder.add_raster_page(
        image,
        ("no native text objects", "PII adjacent to stamps and non-text marks"),
        source_dpi=216,
    )
    page_index = _page_index(page)
    _record_raster_placements(builder, page_index, image.size, placements)
    _record_raster_control(builder, page_index, image.size, control_value, control_quad)


def _build_multilingual_hybrid(builder: _CorpusBuilder) -> None:
    page = builder.new_page((
        "trustworthy native text plus scan fragment",
        "full-page OCR preserves mixed order",
    ))
    _heading(
        page,
        "BILINGUAL APPENDIX / PHỤ LỤC SONG NGỮ",
        font_path=builder.raster_font_path,
    )
    _add_multiline_native_canary(
        builder,
        page,
        label="private_url",
        value="private.example.invalid/care/team",
        segments=("private.example.invalid/", "care/team"),
        baselines=((54, 130), (116, 147)),
        prefix="Portal: ",
    )
    _add_multiline_native_canary(
        builder,
        page,
        label="email_address",
        value="multiline.p9@example.invalid",
        segments=("multiline.p9@", "example.invalid"),
        baselines=((54, 180), (102, 197)),
        prefix="Email: ",
    )
    _add_multiline_native_canary(
        builder,
        page,
        label="secret",
        value="SYNTH-WRAPPED-SECRET-P9",
        segments=("SYNTH-WRAPPED-", "SECRET-P9"),
        baselines=((54, 230), (106, 247)),
        prefix="Secret: ",
    )
    page.insert_text(
        (54, 285),
        "Public decoy URL: https://www.who.int",
        fontname="helv",
        fontsize=10,
    )
    fragment = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(fragment)
    font = _raster_font(builder.raster_font_path, 28)
    visible = _draw_raster_value(
        draw,
        font,
        prefix="Bệnh nhân: ",
        value="Nguyễn Tổng Hợp P9",
        xy=(55, 110),
    )
    address = _draw_raster_value(
        draw,
        font,
        prefix="Địa chỉ: ",
        value="90 Đường Thử Nghiệm",
        xy=(55, 210),
    )
    destination = _fitz_rect(54, 320, 558, 712)
    fragment_bytes = _png_bytes(fragment)
    _page_insert_image(page, destination, fragment_bytes)
    page_record = builder.page_records[_page_index(page)]
    page_record.source_asset_digest = hashlib.sha256(fragment_bytes).hexdigest()
    page_record.source_dpi = 128
    raster_entries: tuple[tuple[PiiLabel, str, tuple[tuple[float, float], ...]], ...] = (
        ("human_name", "Nguyễn Tổng Hợp P9", visible),
        ("address", "90 Đường Thử Nghiệm", address),
    )
    for label, value, pixel_quad in raster_entries:
        builder.record_canary(
            page_index=_page_index(page),
            label=label,
            value=value,
            channel="visible_raster",
            quads=(_pixel_quad_to_rect(pixel_quad, fragment.size, destination),),
        )
    builder.add_native_control(page, "public URL https://www.who.int", (54, 755))


def _build_hostile_page(builder: _CorpusBuilder) -> None:
    page = builder.new_page((
        "PII exists outside page paint",
        "dynamic and hidden objects require sanitization",
    ))
    _heading(page, "HOSTILE PDF OBJECTS — SYNTHETIC")
    builder.add_native_canary(
        page,
        label="secret",
        value="SYNTH-INVISIBLE-P10",
        baseline=(54, 130),
        channel="hidden_text",
        render_mode=3,
        fixture_suffix="hidden",
    )
    widget_value = "SYNTH-FORM-P10-771"
    widget = cast("_MutableWidget", _fitz_widget())
    widget.field_name = "synthetic_private_id"
    widget.field_type = _TEXT_WIDGET_TYPE
    widget.field_value = widget_value
    widget_rect = _fitz_rect(54, 170, 310, 202)
    widget.rect = widget_rect
    page.add_widget(cast("fitz.Widget", widget))
    builder.record_canary(
        page_index=_page_index(page),
        label="id_number",
        value=widget_value,
        channel="form",
        quads=(_fitz_rect_quad(widget_rect),),
        object_locator="AcroForm.synthetic_private_id",
        fixture_suffix="form",
    )
    annotation_value = "SYNTH-ANNOTATION-P10-SECRET"
    annotation = page.add_text_annot(_fitz_point(360, 185), annotation_value)
    _annotation_set_info(
        annotation,
        title="Synthetic reviewer",
        creationDate=_FIXED_PDF_DATE,
        modDate=_FIXED_PDF_DATE,
    )
    builder.record_canary(
        page_index=_page_index(page),
        label="secret",
        value=annotation_value,
        channel="annotation",
        quads=(_fitz_rect_quad(annotation.rect),),
        object_locator="page[9].annotation[0].content",
        fixture_suffix="annotation",
    )
    link_value = "https://private.example.invalid/p10/link-secret"
    link_rect = _fitz_rect(54, 250, 360, 280)
    page.insert_text((54, 270), "Synthetic private link target", fontname="helv", fontsize=10)
    page.insert_link({"kind": fitz.LINK_URI, "from": link_rect, "uri": link_value})
    builder.record_canary(
        page_index=_page_index(page),
        label="private_url",
        value=link_value,
        channel="link",
        quads=(_fitz_rect_quad(link_rect),),
        object_locator="page[9].link[0].uri",
        fixture_suffix="link",
    )
    builder.add_native_control(page, "ordinary number 42", (54, 720))


def _add_hostile_document_objects(builder: _CorpusBuilder) -> None:
    page_index = 9
    metadata_value = "SYNTH-METADATA-P10-NAME"
    metadata = cast("dict[str, str] | None", builder.document.metadata) or {}
    metadata.update({
        "title": "Meddies PII PDF Redaction Benchmark",
        "author": "Meddies Research",
        "subject": metadata_value,
        "keywords": "synthetic, pii, redaction",
        "creator": "Meddies deterministic corpus generator",
        "producer": "Meddies deterministic corpus generator",
        "creationDate": _FIXED_PDF_DATE,
        "modDate": _FIXED_PDF_DATE,
    })
    builder.document.set_metadata(metadata)
    builder.record_canary(
        page_index=page_index,
        label="human_name",
        value=metadata_value,
        channel="metadata",
        object_locator="Info.Subject",
        fixture_suffix="metadata",
    )

    xmp_value = "SYNTH-XMP-P10-ID-8842"
    xmp = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:meddies="https://meddies.ai/ns/benchmark/" '
        f'meddies:syntheticCanary="{xmp_value}"/>'
        '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )
    _document_set_xml_metadata(builder.document, xmp)
    builder.record_canary(
        page_index=page_index,
        label="id_number",
        value=xmp_value,
        channel="xmp",
        object_locator="Catalog.Metadata",
        fixture_suffix="xmp",
    )

    attachment_value = "SYNTH-ATTACHMENT-P10-SECRET"
    attachment = f"Synthetic attachment\nsecret={attachment_value}\n".encode()
    builder.document.embfile_add(
        "synthetic-private-note.txt",
        attachment,
        filename="synthetic-private-note.txt",
        ufilename="synthetic-private-note.txt",
        desc="Synthetic hostile attachment",
    )
    _fix_embedded_file_dates(builder.document)
    builder.record_canary(
        page_index=page_index,
        label="secret",
        value=attachment_value,
        channel="attachment",
        object_locator="EmbeddedFiles.synthetic-private-note.txt",
        fixture_suffix="attachment",
    )
