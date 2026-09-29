"""Legacy Meddies row parsing for the PII-label migration."""

from __future__ import annotations

from collections.abc import Mapping

from meddies_pii.annotations.tagged_text import parse_tagged_text
from meddies_pii.spans import CharSpan
from meddies_pii.training.bioes.data.source_payloads import parse_serialized_collection


def row_uid(row: Mapping[str, object], default: str = "row-0") -> str:
    return str(row.get("uid") or row.get("source") or default)


def plain_text(row: Mapping[str, object]) -> str:
    raw = row.get("raw")
    if isinstance(raw, str) and "]<" not in raw:
        return raw
    text = row.get("text")
    if isinstance(text, str):
        return parse_tagged_text(text).raw if "]<" in text else text
    return str(raw or "")


def _locate_span(raw: str, value: str, start: object = None, end: object = None) -> tuple[int, int] | None:
    if isinstance(start, int) and isinstance(end, int) and raw[start:end] == value:
        return start, end
    found = raw.find(value)
    if found == -1:
        return None
    return found, found + len(value)


def original_spans(row: Mapping[str, object], raw: str) -> list[CharSpan]:
    spans = parse_serialized_collection(row.get("spans"))
    if not isinstance(spans, list):
        return []
    out: list[CharSpan] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        label = span.get("label") or span.get("category")
        value = span.get("text") or span.get("value")
        if not isinstance(label, str) or not isinstance(value, str):
            continue
        located = _locate_span(raw, value, span.get("start"), span.get("end"))
        if located is None:
            continue
        out.append(
            CharSpan(
                start=located[0],
                end=located[1],
                text=value,
                label=label.strip().lower(),
            ),
        )
    return out


def _parse_inline_tags(text: str) -> tuple[str, list[CharSpan]]:
    """Parse migration tags while unwrapping unlabeled bracket placeholders.

    Returns:
        The de-tagged text and its spans, offsets indexing the de-tagged result. A bracket with
        no label is unwrapped rather than dropped, so its contents stay in the text as ordinary
        words and contribute no span -- legacy rows used brackets both as markup and as literal
        punctuation, and deleting them would remove real content.

    """
    raw_parts: list[str] = []
    spans: list[CharSpan] = []
    index = 0
    while index < len(text):
        if text[index] == "[":
            close = text.find("]", index + 1)
            if close != -1:
                entity = text[index + 1 : close]
                start = len("".join(raw_parts))
                raw_parts.append(entity)
                end = start + len(entity)
                cursor = close + 1
                if cursor < len(text) and text[cursor] == "<":
                    label_close = text.find(">", cursor + 1)
                    if label_close != -1:
                        spans.append(
                            CharSpan(
                                start=start,
                                end=end,
                                text=entity,
                                label=text[cursor + 1 : label_close].strip().lower(),
                            ),
                        )
                        index = label_close + 1
                        continue
                index = cursor
                continue
        raw_parts.append(text[index])
        index += 1
    return "".join(raw_parts), spans


def current_tagged_spans(row: Mapping[str, object], raw: str) -> list[CharSpan]:
    text = row.get("text")
    if isinstance(text, str) and "]<" in text:
        parsed_raw, spans = _parse_inline_tags(text)
        if parsed_raw == raw:
            return spans
        relocated: list[CharSpan] = []
        for span in spans:
            located = _locate_span(raw, span.text, span.start, span.end)
            if located is not None:
                relocated.append(
                    CharSpan(
                        start=located[0],
                        end=located[1],
                        text=span.text,
                        label=span.label,
                    ),
                )
        return relocated
    raw_field = row.get("raw")
    if isinstance(raw_field, str) and "]<" in raw_field:
        return list(parse_tagged_text(raw_field).spans)
    return []
