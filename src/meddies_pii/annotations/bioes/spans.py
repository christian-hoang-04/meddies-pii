"""Decode BIOES token labels into character spans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from meddies_pii.annotations.bioes.vocabulary import IGNORE_INDEX
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence


def _trim_whitespace_bounds(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


# reason: decode bioes from orders trim bounds before close active; helper seams would misattribute row errors.
def decode_bioes_from_offsets(  # ruff: ignore[complex-structure]
    raw: str,
    offset_mapping: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    id_to_label: dict[int, str],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[CharSpan, ...]:
    spans: list[CharSpan] = []
    active_label: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is None or active_start is None or active_end is None:
            active_label = active_start = active_end = None
            return
        active_start, active_end = _trim_whitespace_bounds(raw, active_start, active_end)
        spans.append(
            CharSpan(
                start=active_start,
                end=active_end,
                text=raw[active_start:active_end],
                label=active_label,
            ),
        )
        active_label = active_start = active_end = None

    for (start, end), label_id in zip(offset_mapping, label_ids, strict=True):
        if label_id == ignore_index or start == end == 0:
            close_active()
            continue
        label = id_to_label[int(label_id)]
        if label == "O":
            close_active()
            continue
        prefix, entity = label.split("-", 1)
        if prefix == "S":
            close_active()
            trim_start, trim_end = _trim_whitespace_bounds(raw, start, end)
            spans.append(CharSpan(start=trim_start, end=trim_end, text=raw[trim_start:trim_end], label=entity))
            continue
        if prefix == "B":
            close_active()
            active_label = entity
            active_start = start
            active_end = end
            continue
        if prefix == "I":
            if active_label != entity or active_start is None:
                close_active()
                active_label = entity
                active_start = start
                active_end = end
            else:
                active_end = end
            continue
        if prefix == "E":
            if active_label != entity or active_start is None:
                trim_start, trim_end = _trim_whitespace_bounds(raw, start, end)
                spans.append(CharSpan(start=trim_start, end=trim_end, text=raw[trim_start:trim_end], label=entity))
            else:
                active_end = end
                close_active()
            continue
    close_active()
    return tuple(spans)
