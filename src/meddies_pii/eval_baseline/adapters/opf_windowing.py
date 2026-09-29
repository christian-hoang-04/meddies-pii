from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TokenizedOpfDoc:
    example_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PackedWindow:
    example_id: str
    token_ids: tuple[int, ...]
    offsets: tuple[int, ...]
    valid_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class PackedWindowBatch:
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[bool, ...], ...]
    windows: tuple[PackedWindow, ...]


def pack_token_windows(
    docs: Sequence[TokenizedOpfDoc],
    *,
    window_size: int,
    batch_size: int,
    pad_token_id: int,
) -> tuple[PackedWindowBatch, ...]:
    if window_size <= 0:
        msg = "window_size must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    windows: list[PackedWindow] = []
    for doc in docs:
        for start in range(0, len(doc.token_ids), window_size):
            token_slice = doc.token_ids[start : start + window_size]
            windows.append(
                PackedWindow(
                    example_id=doc.example_id,
                    token_ids=tuple(token_slice),
                    offsets=tuple(range(start, start + len(token_slice))),
                    valid_mask=tuple(True for _ in token_slice),
                ),
            )

    batches: list[PackedWindowBatch] = []
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        max_len = max((len(window.token_ids) for window in batch_windows), default=0)
        input_ids: list[tuple[int, ...]] = []
        attention_mask: list[tuple[bool, ...]] = []
        padded_windows: list[PackedWindow] = []
        for window in batch_windows:
            pad = max_len - len(window.token_ids)
            input_ids.append((*window.token_ids, *(pad_token_id for _ in range(pad))))
            attention_mask.append((*window.valid_mask, *(False for _ in range(pad))))
            padded_windows.append(
                PackedWindow(
                    example_id=window.example_id,
                    token_ids=(*window.token_ids, *(pad_token_id for _ in range(pad))),
                    offsets=window.offsets,
                    valid_mask=(*window.valid_mask, *(False for _ in range(pad))),
                ),
            )
        batches.append(
            PackedWindowBatch(
                input_ids=tuple(input_ids),
                attention_mask=tuple(attention_mask),
                windows=tuple(padded_windows),
            ),
        )
    return tuple(batches)


def reassemble_window_values(
    batches: Sequence[PackedWindowBatch],
    values_by_batch: Sequence[Sequence[Sequence[T]]],
) -> dict[str, tuple[T, ...]]:
    if len(batches) != len(values_by_batch):
        msg = "values_by_batch must align with batches"
        raise ValueError(msg)

    values_by_example: dict[str, dict[int, T]] = {}
    for batch, batch_values in zip(batches, values_by_batch, strict=True):
        if len(batch.windows) != len(batch_values):
            msg = "batch value rows must align with packed windows"
            raise ValueError(msg)
        for window, row_values in zip(batch.windows, batch_values, strict=True):
            if len(row_values) < len(window.valid_mask):
                msg = "window values shorter than attention mask"
                raise ValueError(msg)
            target = values_by_example.setdefault(window.example_id, {})
            for position, is_valid in enumerate(window.valid_mask):
                if not is_valid:
                    continue
                if position >= len(window.offsets):
                    continue
                target[window.offsets[position]] = row_values[position]

    return {
        example_id: tuple(value for _offset, value in sorted(offset_values.items()))
        for example_id, offset_values in sorted(values_by_example.items())
    }
