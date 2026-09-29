from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from meddies_pii.training.bioes.data.preparation import PreparedRow


# reason: the tokenizer arrives from `AutoTokenizer.from_pretrained`, which the pinned transformers declares as
# reason: `Unknown | TokenizersBackend | None | SentencePieceBackend`. There is no single honest static type to write,
# reason: and spelling the union out would add an `Unknown` arm and a `None` the pinned version never returns.
def _collate(
    batch: Sequence[PreparedRow],
    tokenizer: Any,  # ruff: ignore[any-type]
    device: str,
    *,
    pad_to_length: int | None = None,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    max_len = max(len(row.tokenized.input_ids) for row in batch)
    if pad_to_length is not None:
        if pad_to_length < max_len:
            msg = f"pad_to_length={pad_to_length} is shorter than batch max length {max_len}"
            raise ValueError(msg)
        max_len = pad_to_length
    input_ids = []
    attention_mask = []
    labels = []
    for row in batch:
        pad = max_len - len(row.tokenized.input_ids)
        input_ids.append(row.tokenized.input_ids + [pad_id] * pad)
        attention_mask.append(row.tokenized.attention_mask + [0] * pad)
        labels.append(row.tokenized.labels + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _batches(rows: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    for index in range(0, len(rows), batch_size):
        yield list(rows[index : index + batch_size])
