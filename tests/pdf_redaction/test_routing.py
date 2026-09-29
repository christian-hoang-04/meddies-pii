from __future__ import annotations

import pytest

from meddies_pii.pdf_redaction.routing import PageRouteSignals, classify_page


@pytest.mark.parametrize(
    ("signals", "content_class", "content_route", "full_page_ocr"),
    [
        (
            PageRouteSignals(
                page_index=0,
                visible_native_characters=48,
                has_meaningful_raster=False,
                native_text_trusted=True,
            ),
            "native_text",
            "native_trusted",
            False,
        ),
        (
            PageRouteSignals(
                page_index=1,
                visible_native_characters=0,
                has_meaningful_raster=True,
                native_text_trusted=True,
            ),
            "image_only",
            "image_only",
            True,
        ),
        (
            PageRouteSignals(
                page_index=2,
                visible_native_characters=48,
                has_meaningful_raster=True,
                native_text_trusted=True,
            ),
            "hybrid_mixed",
            "hybrid_mixed",
            True,
        ),
        (
            PageRouteSignals(
                page_index=3,
                visible_native_characters=48,
                has_meaningful_raster=False,
                native_text_trusted=False,
                trust_diagnostics=("broken_unicode_map",),
            ),
            "untrusted_native",
            "hybrid_untrusted",
            True,
        ),
    ],
)
# reason: pytest binds this parameter from the parametrize argnames tuple, so the boolean is labelled at every call site.
def test_classifies_pages_with_correctness_first_ocr_routes(
    signals: PageRouteSignals,
    content_class: str,
    content_route: str,
    full_page_ocr: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
) -> None:
    decision = classify_page(signals)

    assert decision.content_class == content_class
    assert decision.content_route == content_route
    assert decision.requires_full_page_ocr is full_page_ocr
