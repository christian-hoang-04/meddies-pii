"""BIOES vocabulary construction and padding-label masking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from meddies_pii.taxonomy import PII_LABELS

if TYPE_CHECKING:
    from collections.abc import Sequence

ENTITY_LABELS: tuple[str, ...] = tuple(sorted(PII_LABELS))
IGNORE_INDEX = -100


def build_bioes_label_space(entity_labels: Sequence[str]) -> tuple[str, ...]:
    vocab = ["O"]
    for label in entity_labels:
        vocab.extend((f"B-{label}", f"I-{label}", f"E-{label}", f"S-{label}"))
    return tuple(vocab)


def build_label_to_id(label_vocab: Sequence[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(label_vocab)}


def mask_padding_labels(
    labels: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]],
    ignore_index: int = IGNORE_INDEX,
) -> list[list[int]]:
    return [
        [label if mask else ignore_index for label, mask in zip(label_row, mask_row, strict=True)]
        for label_row, mask_row in zip(labels, attention_mask, strict=True)
    ]
