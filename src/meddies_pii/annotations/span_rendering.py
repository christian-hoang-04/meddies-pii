"""HTML rendering for annotated character spans."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _css_label(label: str) -> str:
    return label.replace("_", "-")


def snippet_html(
    text: str,
    spans: Sequence[CharSpan],
    *,
    focus_span: CharSpan | None = None,
    context: int = 180,
) -> str:
    """Return an escaped/highlighted text window around one span.

    Returns:
        An HTML window of ``context`` characters either side of the focus span, with every span
        that overlaps the window highlighted and ellipses marking a truncated edge. Empty text
        gives an empty string, and text with no span at all is escaped and truncated rather
        than dropped. Span offsets are rebased onto the window, so they index what is rendered.

    """
    if not text:
        return ""
    if focus_span is None and spans:
        focus_span = spans[0]
    if focus_span is None:
        return escape(_truncate_text(text, context * 2), quote=False)

    start = max(0, focus_span.start - context)
    end = min(len(text), focus_span.end + context)
    window = text[start:end]
    window_spans = [
        CharSpan(
            start=max(0, span.start - start),
            end=min(end, span.end) - start,
            text=span.text,
            label=span.label,
        )
        for span in spans
        if span.end > start and span.start < end
    ]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + highlight_spans(window, window_spans) + suffix


def highlight_spans(text: str, spans: Sequence[CharSpan]) -> str:
    """Escape text and wrap valid, non-overlapping spans in label marks.

    Returns:
        The escaped text with each accepted span wrapped in its label mark. Spans are taken in
        start order and clamped to the text; one that starts before the cursor, or that clamps
        to zero width, is skipped. Every piece is escaped before it is joined, so span text
        drawn from a document cannot inject markup into the rendering.

    """
    pieces: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label)):
        start = max(0, min(span.start, len(text)))
        end = max(start, min(span.end, len(text)))
        if start < cursor or start == end:
            continue
        pieces.append(escape(text[cursor:start], quote=False))
        label = _css_label(span.label)
        pieces.append(
            f'<mark class="pii label-{label}" title="{escape(span.label)}">'
            f"{escape(text[start:end], quote=False)}"
            f'<span class="tag">{escape(span.label)}</span>'
            "</mark>",
        )
        cursor = end
    pieces.append(escape(text[cursor:], quote=False))
    return "".join(pieces)
