"""Neutral character-offset span contract shared across Meddies PII capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, TypeGuard


@dataclass(frozen=True, slots=True)
class CharSpan:
    """A labeled text interval with offsets measured in Python characters."""

    start: int
    end: int
    text: str
    label: str


class CharSpanValue(TypedDict):
    """JSON-compatible persisted representation of a :class:`CharSpan`."""

    start: int
    end: int
    text: str
    label: str


def char_span_to_dict(span: CharSpan) -> CharSpanValue:
    """Convert a span into the JSON-compatible evaluation artifact shape.

    Returns:
        The span's start, end, surface text and label as a plain JSON-compatible mapping. The
        surface travels with the offsets so an artifact stays readable and checkable without
        the source document beside it.

    """
    return {
        "start": span.start,
        "end": span.end,
        "text": span.text,
        "label": span.label,
    }


def _is_string_keyed_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def char_span_from_value(value: object) -> CharSpan | None:
    """Parse one persisted span, returning ``None`` for every malformed value.

    Evaluation artifacts are best-effort inputs. Invalid values are discarded rather
    than coerced, so malformed JSON cannot become a different scored span.

    Returns:
        The parsed span, or ``None`` for anything that is not a string-keyed mapping carrying
        the four fields at their expected types. Nothing is coerced and nothing raises, because
        a coerced value would enter the scoring as a span that no system actually predicted.

    """
    if not _is_string_keyed_mapping(value):
        return None

    start = value.get("start")
    end = value.get("end")
    text = value.get("text")
    label = value.get("label")
    # reason: Text, offsets, label, and metadata form one CharSpan input schema; predicate helpers would scatter it.
    if (
        not isinstance(start, int)  # ruff: ignore[too-many-boolean-expressions]
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(text, str)
        or not isinstance(label, str)
    ):
        return None
    return CharSpan(start=start, end=end, text=text, label=label)
