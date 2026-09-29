"""Tagged-text parsing and label JSON helpers."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from meddies_pii.annotations.bioes import ENTITY_LABELS
from meddies_pii.spans import CharSpan
from meddies_pii.tags import fix_labels

if TYPE_CHECKING:
    from collections.abc import Sequence

VALID_LABELS = frozenset(ENTITY_LABELS)
_STRAY_LABEL_RE = re.compile(
    r"</?(?:" + "|".join(re.escape(label) for label in sorted(VALID_LABELS)) + r")>",
    re.IGNORECASE,
)
"""A stray `<label>` / `</label>` fragment the model emitted without a `[value]` wrapper (e.g.

`TP.HCM<address>` — it forgot the brackets). These are generation artifacts, never real clinical text, so the parser skips
them instead of leaking them into the raw text as training noise (which used to force a whole rich doc to be rejected as a
"leftover_label_marker").

"""


@dataclass(frozen=True, slots=True)
class ParsedTaggedDocument:
    text: str
    normalized_text: str
    had_label_repairs: bool
    raw: str
    spans: tuple[CharSpan, ...]


def parse_tagged_text(text: str, *, normalize_labels: bool = False) -> ParsedTaggedDocument:
    normalized = fix_labels(text) if normalize_labels else text
    had_label_repairs = normalized != text
    raw_parts: list[str] = []
    spans: list[CharSpan] = []
    i = 0

    while i < len(normalized):
        if normalized[i] == "[":
            close = normalized.find("]", i + 1)
            if close != -1 and close + 1 < len(normalized) and normalized[close + 1] == "<":
                label_close = normalized.find(">", close + 2)
                if label_close != -1:
                    entity = normalized[i + 1 : close]
                    label = normalized[close + 2 : label_close].strip().lower()
                    if label in VALID_LABELS:
                        start = len("".join(raw_parts))
                        raw_parts.append(entity)
                        end = start + len(entity)
                        spans.append(CharSpan(start=start, end=end, text=entity, label=label))
                        i = label_close + 1
                        continue
        if normalized[i] == "<":
            stray = _STRAY_LABEL_RE.match(normalized, i)
            if stray is not None:
                i = stray.end()
                continue
        raw_parts.append(normalized[i])
        i += 1

    raw = "".join(raw_parts)
    return ParsedTaggedDocument(
        text=text,
        normalized_text=normalized,
        had_label_repairs=had_label_repairs,
        raw=raw,
        spans=tuple(spans),
    )


def spans_to_label_json(spans: Sequence[CharSpan]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for span in spans:
        key = (span.text, span.label)
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(span.label, []).append(span.text)
    return out


def label_json_from_string(label_text: str | None) -> dict[str, list[str]]:
    if not label_text:
        return {}
    value = json.loads(label_text)
    if not isinstance(value, dict):
        msg = "label JSON must be an object"
        raise ValueError(msg)
    normalized: dict[str, list[str]] = {}
    for label, values in value.items():
        if not isinstance(label, str) or not isinstance(values, list):
            msg = "label JSON must map string labels to lists"
            raise ValueError(msg)
        if not all(isinstance(item, str) for item in values):
            msg = "label JSON values must be strings"
            raise ValueError(msg)
        normalized[label] = values
    return normalized


def normalize_label_json(label_json: dict[str, list[str]]) -> dict[str, list[str]]:
    return {key: sorted(dict.fromkeys(values)) for key, values in sorted(label_json.items())}
