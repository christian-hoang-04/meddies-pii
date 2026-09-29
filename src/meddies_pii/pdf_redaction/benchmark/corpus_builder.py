from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pymupdf as fitz

from meddies_pii.pdf_redaction.benchmark.corpus_drawing import (
    _digest,
    _document_close,
    _document_tobytes,
    _fitz_font,
    _fitz_point,
    _font_text_length,
    _page_index,
    _page_insert_image,
    _png_bytes,
    _rect_quad,
    _resolve_raster_font,
)
from meddies_pii.pdf_redaction.benchmark.corpus_records import (
    _FIXED_PDF_DATE,
    _HEIGHT_PT,
    _PAGE_ROUTES,
    _WIDTH_PT,
    PAGE_CLASSES,
    CanaryGold,
    CorpusFixture,
    CorpusGold,
    Degradation,
    GoldChannel,
    NegativeControlGold,
    PageGold,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image

    from meddies_pii.pdf_redaction.contracts import Quad
    from meddies_pii.taxonomy import PiiLabel


@dataclass(slots=True)
class _PageRecord:
    route_reasons: tuple[str, ...]
    negative_controls: list[str] = field(default_factory=list)
    rotation_degrees: int = 0
    degradations: tuple[Degradation, ...] = ()
    source_asset_digest: str | None = None
    source_dpi: int | None = None


class _CorpusBuilder:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        open_document = cast("Callable[[], fitz.Document]", fitz.open)
        self.document = open_document()
        self.canaries: list[CanaryGold] = []
        self.controls: list[NegativeControlGold] = []
        self.page_records: list[_PageRecord] = []
        self.font = _fitz_font()
        self.raster_font_path = _resolve_raster_font()

    def new_page(self, reasons: tuple[str, ...]) -> fitz.Page:
        page = self.document.new_page(width=_WIDTH_PT, height=_HEIGHT_PT)
        self.page_records.append(_PageRecord(route_reasons=reasons))
        return page

    # reason: _CorpusBuilder exposes page/fixture as its public contract; bundling would break callers.
    def add_native_canary(  # ruff: ignore[too-many-arguments]
        self,
        page: fitz.Page,
        *,
        label: PiiLabel,
        value: str,
        baseline: tuple[float, float],
        prefix: str = "",
        fontsize: int = 10,
        channel: GoldChannel = "visible_native",
        render_mode: int = 0,
        fixture_suffix: str | None = None,
    ) -> CanaryGold:
        point = _fitz_point(*baseline)
        page.insert_text(
            point,
            prefix + value,
            fontname="helv",
            fontsize=fontsize,
            render_mode=render_mode,
        )
        x0 = point.x + _font_text_length(self.font, prefix, fontsize=fontsize)
        x1 = x0 + _font_text_length(self.font, value, fontsize=fontsize)
        y0 = point.y - self.font.ascender * fontsize
        y1 = point.y - self.font.descender * fontsize
        return self.record_canary(
            page_index=_page_index(page),
            label=label,
            value=value,
            channel=channel,
            quads=(_rect_quad(x0, y0, x1, y1),),
            fixture_suffix=fixture_suffix,
        )

    # reason: _CorpusBuilder exposes page index/fixture as its public contract; bundling would break callers.
    def record_canary(  # ruff: ignore[too-many-arguments]
        self,
        *,
        page_index: int,
        label: PiiLabel,
        value: str,
        channel: GoldChannel,
        quads: tuple[Quad, ...] = (),
        object_locator: str | None = None,
        fixture_suffix: str | None = None,
    ) -> CanaryGold:
        occurrence = 1 + sum(item.page_index == page_index and item.label == label for item in self.canaries)
        suffix = fixture_suffix or f"{occurrence:02d}"
        fixture_id = f"p{page_index + 1:02d}-{label}-{suffix}"
        canary = CanaryGold(
            fixture_id=fixture_id,
            page_index=page_index,
            label=label,
            channel=channel,
            digest=_digest(self.seed, fixture_id, channel, value),
            quads=quads,
            object_locator=object_locator,
            raw_value=value,
        )
        self.canaries.append(canary)
        return canary

    def add_native_control(
        self,
        page: fitz.Page,
        value: str,
        baseline: tuple[float, float],
    ) -> None:
        prefix = "Negative control: "
        fontsize = 9
        point = _fitz_point(*baseline)
        page.insert_text(point, prefix + value, fontname="helv", fontsize=fontsize)
        x0 = point.x + _font_text_length(self.font, prefix, fontsize=fontsize)
        x1 = x0 + _font_text_length(self.font, value, fontsize=fontsize)
        y0 = point.y - self.font.ascender * fontsize
        y1 = point.y - self.font.descender * fontsize
        page_index = _page_index(page)
        fixture_id = f"p{page_index + 1:02d}-decoy-01"
        control = NegativeControlGold(
            fixture_id=fixture_id,
            page_index=page_index,
            digest=_digest(self.seed, fixture_id, "negative_control", value),
            quads=(_rect_quad(x0, y0, x1, y1),),
            raw_value=value,
        )
        self.controls.append(control)
        self.page_records[page_index].negative_controls.append(fixture_id)

    def add_raster_page(
        self,
        image: Image.Image,
        reasons: tuple[str, ...],
        *,
        source_dpi: int,
    ) -> fitz.Page:
        page = self.new_page(reasons)
        png = _png_bytes(image)
        _page_insert_image(page, page.rect, png)
        self.page_records[_page_index(page)].source_asset_digest = hashlib.sha256(png).hexdigest()
        self.page_records[_page_index(page)].source_dpi = source_dpi
        return page

    def set_fixed_metadata(self) -> None:
        self.document.set_metadata({
            "title": "Meddies PII PDF Redaction Benchmark",
            "author": "Meddies Research",
            "creator": "Meddies deterministic corpus generator",
            "producer": "Meddies deterministic corpus generator",
            "creationDate": _FIXED_PDF_DATE,
            "modDate": _FIXED_PDF_DATE,
        })

    def finish(self) -> CorpusFixture:
        pdf_bytes = _document_tobytes(
            self.document,
            garbage=4,
            clean=True,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
            no_new_id=True,
            use_objstms=1,
            compression_effort=100,
        )
        _document_close(self.document)
        pages = tuple(
            PageGold(
                page_index=index,
                page_class=page_class,
                route=_PAGE_ROUTES[index],
                route_reasons=record.route_reasons,
                negative_controls=tuple(record.negative_controls),
                rotation_degrees=record.rotation_degrees,
                degradations=record.degradations,
                source_asset_digest=record.source_asset_digest,
                source_dpi=record.source_dpi,
            )
            for index, (page_class, record) in enumerate(zip(PAGE_CLASSES, self.page_records, strict=True))
        )
        return CorpusFixture(
            pdf_bytes=pdf_bytes,
            gold=CorpusGold(
                seed=self.seed,
                pages=pages,
                canaries=tuple(self.canaries),
                negative_controls=tuple(self.controls),
            ),
        )
