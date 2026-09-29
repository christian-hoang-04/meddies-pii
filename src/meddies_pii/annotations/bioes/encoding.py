"""Tokenize source text and encode aligned character spans as BIOES labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from meddies_pii.annotations.bioes.vocabulary import (
    ENTITY_LABELS,
    IGNORE_INDEX,
    build_bioes_label_space,
    build_label_to_id,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from meddies_pii.spans import CharSpan


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    raw: str
    spans: tuple[CharSpan, ...]
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    offset_mapping: list[tuple[int, int]]
    truncated: bool = False


def _token_indices_for_span(offset_mapping: Sequence[tuple[int, int]], span: CharSpan) -> list[int]:
    token_indices: list[int] = []
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end == 0:
            continue
        if end <= span.start or start >= span.end:
            continue
        token_indices.append(idx)
    return token_indices


def encode_bioes_for_offsets(
    offset_mapping: Sequence[tuple[int, int]],
    spans: Sequence[CharSpan],
    label_to_id: dict[str, int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    labels = [label_to_id["O"] if not (start == end == 0) else ignore_index for start, end in offset_mapping]
    for span in spans:
        token_indices = _token_indices_for_span(offset_mapping, span)
        if not token_indices:
            msg = f"Span was not aligned to any token: {span}"
            raise ValueError(msg)
        if len(token_indices) == 1:
            sequence = [f"S-{span.label}"]
        elif len(token_indices) == 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            sequence = [f"B-{span.label}", f"E-{span.label}"]
        else:
            sequence = [
                f"B-{span.label}",
                *([f"I-{span.label}"] * (len(token_indices) - 2)),
                f"E-{span.label}",
            ]
        for token_index, tag in zip(token_indices, sequence, strict=True):
            existing = labels[token_index]
            if existing not in {ignore_index, label_to_id["O"]}:
                msg = f"Overlapping token assignment at index {token_index}"
                raise ValueError(msg)
            labels[token_index] = label_to_id[tag]
    return labels


def _tokenizer_sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"Tokenizer field {field!r} must be a sequence"
        raise TypeError(msg)
    return value


def _tokenizer_integer_sequence(value: object, field: str) -> list[int]:
    items = _tokenizer_sequence(value, field)
    integers: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            msg = f"Tokenizer field {field!r} must contain integers"
            raise TypeError(msg)
        integers.append(item)
    return integers


def _tokenizer_offset_sequence(value: object, field: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for pair_value in _tokenizer_sequence(value, field):
        pair = _tokenizer_sequence(pair_value, field)
        if len(pair) != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            msg = f"Tokenizer field {field!r} must contain offset pairs"
            raise ValueError(msg)
        start, end = pair
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            msg = f"Tokenizer field {field!r} offsets must be integers"
            raise TypeError(msg)
        offsets.append((start, end))
    return offsets


def _validate_tokenizer_alignment(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
) -> None:
    if not input_ids:
        msg = "Tokenizer returned no input IDs"
        raise ValueError(msg)
    lengths = {len(input_ids), len(attention_mask), len(offset_mapping)}
    if len(lengths) != 1:
        msg = (
            "Tokenizer fields must have matching lengths: "
            f"input_ids={len(input_ids)} "
            f"attention_mask={len(attention_mask)} "
            f"offset_mapping={len(offset_mapping)}"
        )
        raise ValueError(
            msg,
        )


# reason: tokenize and orders validate before label space; helper seams would fragment diagnostics.
def tokenize_and_align(  # ruff: ignore[too-many-locals]
    tokenizer: PreTrainedTokenizerBase,
    raw: str,
    spans: Sequence[CharSpan],
    *,
    max_length: int,
    entity_labels: Sequence[str] = ENTITY_LABELS,
) -> TokenizedExample:
    encoded = tokenizer(
        raw,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    if not isinstance(encoded, Mapping):
        msg = "Tokenizer output must be a mapping"
        raise TypeError(msg)
    encoded_data: Mapping[str, object] = encoded
    input_id_values = _tokenizer_sequence(encoded_data.get("input_ids"), "input_ids")
    has_overflow_rows = bool(input_id_values) and isinstance(input_id_values[0], Sequence)

    if has_overflow_rows:
        input_id_rows = [_tokenizer_integer_sequence(row, "input_ids") for row in input_id_values]
        attention_rows = [
            _tokenizer_integer_sequence(row, "attention_mask")
            for row in _tokenizer_sequence(encoded_data.get("attention_mask"), "attention_mask")
        ]
        offset_rows = [
            _tokenizer_offset_sequence(row, "offset_mapping")
            for row in _tokenizer_sequence(encoded_data.get("offset_mapping"), "offset_mapping")
        ]
        row_counts = {len(input_id_rows), len(attention_rows), len(offset_rows)}
        if len(row_counts) != 1:
            msg = "Tokenizer overflow fields must have matching row counts"
            raise ValueError(msg)
        input_ids = input_id_rows[0]
        attention_mask = attention_rows[0]
        offset_mapping = offset_rows[0]
        truncated = len(input_id_rows) > 1
    else:
        input_ids = _tokenizer_integer_sequence(input_id_values, "input_ids")
        attention_mask = _tokenizer_integer_sequence(encoded_data.get("attention_mask"), "attention_mask")
        offset_mapping = _tokenizer_offset_sequence(encoded_data.get("offset_mapping"), "offset_mapping")
        num_truncated_tokens = encoded_data.get("num_truncated_tokens", 0)
        if isinstance(num_truncated_tokens, bool) or not isinstance(num_truncated_tokens, int):
            msg = "Tokenizer field 'num_truncated_tokens' must be an integer"
            raise TypeError(msg)
        truncated = num_truncated_tokens > 0

    _validate_tokenizer_alignment(input_ids, attention_mask, offset_mapping)
    label_vocab = build_bioes_label_space(entity_labels)
    label_to_id = build_label_to_id(label_vocab)
    labels = encode_bioes_for_offsets(offset_mapping, spans, label_to_id)
    return TokenizedExample(
        raw=raw,
        spans=tuple(spans),
        input_ids=list(input_ids),
        attention_mask=list(attention_mask),
        labels=labels,
        offset_mapping=offset_mapping,
        truncated=truncated,
    )
