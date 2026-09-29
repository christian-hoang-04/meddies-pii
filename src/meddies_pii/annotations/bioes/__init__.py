"""Public BIOES annotation primitives."""

from __future__ import annotations

from meddies_pii.annotations.bioes.encoding import (
    TokenizedExample,
    encode_bioes_for_offsets,
    tokenize_and_align,
)
from meddies_pii.annotations.bioes.spans import decode_bioes_from_offsets
from meddies_pii.annotations.bioes.viterbi import (
    VITERBI_BIAS_KEYS,
    normalize_viterbi_transition_biases,
    viterbi_decode_logits,
    viterbi_decode_numpy,
    zero_viterbi_transition_biases,
)
from meddies_pii.annotations.bioes.vocabulary import (
    ENTITY_LABELS,
    IGNORE_INDEX,
    build_bioes_label_space,
    build_label_to_id,
    mask_padding_labels,
)

__all__ = (
    "ENTITY_LABELS",
    "IGNORE_INDEX",
    "VITERBI_BIAS_KEYS",
    "TokenizedExample",
    "build_bioes_label_space",
    "build_label_to_id",
    "decode_bioes_from_offsets",
    "encode_bioes_for_offsets",
    "mask_padding_labels",
    "normalize_viterbi_transition_biases",
    "tokenize_and_align",
    "viterbi_decode_logits",
    "viterbi_decode_numpy",
    "zero_viterbi_transition_biases",
)
