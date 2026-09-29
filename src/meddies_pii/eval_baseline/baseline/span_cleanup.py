"""Shared span post-processing for token-classification adapters.

Encoder NER models (openmed/DeBERTa, opf) emit noisy subword-level tags: a span
carries leading whitespace and trailing punctuation, and a single entity is often
split across word-boundary ``B-`` tags. ``clean_spans`` repairs both — trim the
boundaries to the entity core, then merge same-label fragments separated only by
whitespace — so exact-boundary scoring is fair. Used by every encoder adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

_LEAD_STRIP = " \t\n\r\f\v"
"""Leading: whitespace only (an entity may legitimately start with punctuation, e.g.

a quoted secret). Trailing: whitespace + sentence punctuation, which an over-extended span picks up from the following
token.

"""
_TRAIL_STRIP = " \t\n\r\f\v.,;:!?"


def trim_span(text: str, span: CharSpan) -> CharSpan | None:
    """Shrink a span's bounds past leading whitespace / trailing punctuation.

    Returns None if nothing survives the trim (the span was pure junk).

    Returns:
        A new span over the trimmed bounds with its surface re-sliced from the text, or ``None``
        when the trim consumes it entirely. The surface is re-sliced rather than trimmed as a
        string, so ``text[start:end]`` still equals the span's own text afterwards.

    """
    start, end = span.start, span.end
    while start < end and text[start] in _LEAD_STRIP:
        start += 1
    while end > start and text[end - 1] in _TRAIL_STRIP:
        end -= 1
    if end <= start:
        return None
    return CharSpan(start=start, end=end, text=text[start:end], label=span.label)


def merge_adjacent_same_label(text: str, spans: list[CharSpan]) -> list[CharSpan]:
    """Merge same-label spans whose only separation is whitespace (or overlap).

    Reconstructs an entity the model fragmented across word-boundary ``B-`` tags
    (``Mister`` | ``Enie`` | ``Idilia`` -> ``Mister Enie Idilia``). A non-empty,
    non-whitespace gap (``", "``) keeps them distinct.

    text[prev.end:span.start] is "" when adjacent/overlapping (reverse slice), the literal gap text otherwise.

    Returns:
        The spans in start order, with each run of same-label whitespace-separated neighbours
        collapsed into one span whose surface is re-sliced across the whole merged range. The
        end is taken as the maximum of the two, so a contained span cannot shorten the span it
        merges into.

    """
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[CharSpan] = []
    for span in ordered:
        if merged:
            prev = merged[-1]
            gap = text[prev.end : span.start]
            if span.label == prev.label and not gap.strip():
                new_end = max(prev.end, span.end)
                merged[-1] = CharSpan(
                    start=prev.start,
                    end=new_end,
                    text=text[prev.start : new_end],
                    label=prev.label,
                )
                continue
        merged.append(span)
    return merged


def clean_spans(text: str, spans: Sequence[CharSpan]) -> list[CharSpan]:
    """Trim boundaries, then merge same-label whitespace-separated fragments.

    Returns:
        The cleaned spans. Trimming runs first and drops anything that does not survive, so the
        merge never sees a junk span that would bridge two real ones across what was actually
        punctuation.

    """
    trimmed = [trimmed_span for span in spans if (trimmed_span := trim_span(text, span)) is not None]
    return merge_adjacent_same_label(text, trimmed)
