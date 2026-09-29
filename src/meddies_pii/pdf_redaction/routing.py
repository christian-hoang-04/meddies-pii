from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from meddies_pii.pdf_redaction.contracts import ContentRoute

PageContentClass = Literal["native_text", "image_only", "hybrid_mixed", "untrusted_native"]


@dataclass(frozen=True, slots=True)
class PageRouteSignals:
    page_index: int
    visible_native_characters: int
    has_meaningful_raster: bool
    native_text_trusted: bool
    trust_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.page_index < 0:
            msg = "page_index must be non-negative"
            raise ValueError(msg)
        if self.visible_native_characters < 0:
            msg = "visible_native_characters must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PageRouteDecision:
    page_index: int
    content_class: PageContentClass
    content_route: ContentRoute
    requires_full_page_ocr: bool
    diagnostics: tuple[str, ...]


def classify_page(signals: PageRouteSignals) -> PageRouteDecision:
    if not signals.native_text_trusted:
        diagnostics = signals.trust_diagnostics or ("native_text_untrusted",)
        return PageRouteDecision(
            page_index=signals.page_index,
            content_class="untrusted_native",
            content_route="hybrid_untrusted",
            requires_full_page_ocr=True,
            diagnostics=diagnostics,
        )

    if signals.has_meaningful_raster and signals.visible_native_characters:
        return PageRouteDecision(
            page_index=signals.page_index,
            content_class="hybrid_mixed",
            content_route="hybrid_mixed",
            requires_full_page_ocr=True,
            diagnostics=("meaningful_raster_content",),
        )

    if signals.has_meaningful_raster:
        return PageRouteDecision(
            page_index=signals.page_index,
            content_class="image_only",
            content_route="image_only",
            requires_full_page_ocr=True,
            diagnostics=("no_usable_native_text",),
        )

    diagnostics = () if signals.visible_native_characters else ("blank_page",)
    return PageRouteDecision(
        page_index=signals.page_index,
        content_class="native_text",
        content_route="native_trusted",
        requires_full_page_ocr=False,
        diagnostics=diagnostics,
    )
