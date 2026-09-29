"""Tokenizer-length measurement used by augmentation acceptance gates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meddies_pii.training.bioes.data.record_schema import BatchLengthTokenizer


def token_lengths(texts: Sequence[str], *, tokenizer: BatchLengthTokenizer, batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, (str, bytes)) or not isinstance(input_ids, Sequence):
            msg = "Tokenizer field 'input_ids' must be a sequence"
            raise TypeError(msg)
        for row_index, row in enumerate(input_ids):
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                msg = f"Tokenizer field 'input_ids' must contain sequences; invalid row {row_index}"
                raise TypeError(msg)
            lengths.append(len(row))
    return lengths
