from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from meddies_pii.annotations.source_mapping import normalize_native_label
from meddies_pii.eval_baseline.adapters.opf import OPF_LABEL_FOLD
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_BIOES_PREFIXES = frozenset({"B", "I", "E", "S"})


def bioes_token_entities_to_spans(text: str, entities: Sequence[Mapping[str, object]]) -> list[CharSpan]:
    spans: list[CharSpan] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    previous_index: int | None = None

    for token_index, prefix, label, start, end in _valid_bioes_entities(text, entities):
        if previous_index is not None and token_index != previous_index + 1:
            _append_span(spans, text, current_label, current_start, current_end)
            current_label = None
            current_start = None
            current_end = None

        if prefix == "S":
            _append_span(spans, text, current_label, current_start, current_end)
            _append_span(spans, text, label, start, end)
            current_label = None
            current_start = None
            current_end = None
        elif prefix == "B":
            _append_span(spans, text, current_label, current_start, current_end)
            current_label = label
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == label:
                current_end = end
            else:
                _append_span(spans, text, current_label, current_start, current_end)
                current_label = label
                current_start = start
                current_end = end
        elif prefix == "E":
            if current_label == label and current_start is not None:
                _append_span(spans, text, label, current_start, end)
            else:
                _append_span(spans, text, current_label, current_start, current_end)
                _append_span(spans, text, label, start, end)
            current_label = None
            current_start = None
            current_end = None

        previous_index = token_index

    _append_span(spans, text, current_label, current_start, current_end)
    return _deduplicate_spans(spans)


def token_logits_to_bioes_spans(
    text: str,
    logits: Sequence[Sequence[float]],
    offsets: Sequence[tuple[int, int]],
    id2label: Mapping[int, str],
) -> list[CharSpan]:
    entities: list[dict[str, object]] = []
    for token_index, (token_scores, (start, end)) in enumerate(zip(logits, offsets, strict=True)):
        if not token_scores:
            continue
        label_id = max(range(len(token_scores)), key=token_scores.__getitem__)
        label = id2label.get(label_id)
        if label is None:
            continue
        entities.append({
            "entity": label,
            "index": token_index,
            "start": start,
            "end": end,
        })
    return bioes_token_entities_to_spans(text, entities)


def _valid_bioes_entities(text: str, entities: Sequence[Mapping[str, object]]) -> list[tuple[int, str, str, int, int]]:
    valid: list[tuple[int, str, str, int, int]] = []
    for fallback_index, entity in enumerate(entities):
        parsed = _parse_bioes_entity(text, entity, fallback_index)
        if parsed is not None:
            valid.append(parsed)
    return sorted(valid, key=operator.itemgetter(0, 3, 4))


def _parse_bioes_entity(
    text: str,
    entity: Mapping[str, object],
    fallback_index: int,
) -> tuple[int, str, str, int, int] | None:
    raw_label = entity.get("entity")
    if not isinstance(raw_label, str):
        return None
    prefix, label = _split_bioes_label(raw_label)
    if prefix not in _BIOES_PREFIXES or label not in OPF_LABEL_FOLD:
        return None

    start = entity.get("start")
    end = entity.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end <= start or end > len(text):
        return None

    raw_index = entity.get("index")
    token_index = raw_index if isinstance(raw_index, int) else fallback_index
    return token_index, prefix, label, start, end


def _split_bioes_label(label: str) -> tuple[str, str]:
    if "-" not in label:
        return "", normalize_native_label(label)
    prefix, raw_base = label.split("-", 1)
    return prefix.upper(), normalize_native_label(raw_base)


def _append_span(
    spans: list[CharSpan],
    text: str,
    label: str | None,
    start: int | None,
    end: int | None,
) -> None:
    if label is None or start is None or end is None:
        return
    if label not in OPF_LABEL_FOLD or not (0 <= start < end <= len(text)):
        return
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append(CharSpan(start=start, end=end, text=text[start:end], label=label))


def _deduplicate_spans(spans: Sequence[CharSpan]) -> list[CharSpan]:
    deduplicated: list[CharSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for span in spans:
        key = (span.start, span.end, span.label)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(span)
    return deduplicated
