"""Constrained Viterbi decoding for BIOES logits."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the dependency is optional at runtime and deferred to the call that needs it.
# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
from functools import lru_cache
from typing import TYPE_CHECKING

from meddies_pii.annotations.bioes.vocabulary import IGNORE_INDEX

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

VITERBI_BIAS_KEYS: tuple[str, ...] = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


def _parse_bioes_state(label: str) -> tuple[str, str | None]:
    if label == "O":
        return ("O", None)
    prefix, entity = label.split("-", 1)
    return (prefix, entity)


# reason: bioes state and lru cache share bioes constraint's state; extraction would misattribute row errors.
@lru_cache(maxsize=32)
def _bioes_constraint_cache(  # ruff: ignore[complex-structure]
    label_vocab: tuple[str, ...],
    bias_values: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
) -> tuple[list[float], list[float], list[list[float]]]:
    states = [_parse_bioes_state(label) for label in label_vocab]
    neg_inf = float("-inf")
    (
        transition_bias_background_stay,
        transition_bias_background_to_start,
        transition_bias_inside_to_continue,
        transition_bias_inside_to_end,
        transition_bias_end_to_background,
        transition_bias_end_to_start,
    ) = bias_values

    def start_allowed(state: tuple[str, str | None]) -> bool:
        prefix, _entity = state
        return prefix in {"O", "B", "S"}

    def end_allowed(state: tuple[str, str | None]) -> bool:
        prefix, _entity = state
        return prefix in {"O", "E", "S"}

    def transition_allowed(prev: tuple[str, str | None], cur: tuple[str, str | None]) -> bool:
        prev_prefix, prev_entity = prev
        cur_prefix, cur_entity = cur
        if cur_prefix in {"O", "B", "S"}:
            return prev_prefix in {"O", "E", "S"}
        if cur_prefix in {"I", "E"}:
            return prev_prefix in {"B", "I"} and prev_entity == cur_entity
        return False

    # reason: BIOES start/end/transition cases use ordered exits; merging would obscure forbidden-state precedence.
    def transition_bias(prev: tuple[str, str | None], cur: tuple[str, str | None]) -> float:  # ruff: ignore[too-many-return-statements]
        prev_prefix, _prev_entity = prev
        cur_prefix, _cur_entity = cur
        if prev_prefix == "O":
            if cur_prefix == "O":
                return transition_bias_background_stay
            if cur_prefix in {"B", "S"}:
                return transition_bias_background_to_start
            return 0.0
        if prev_prefix in {"B", "I"}:
            if cur_prefix == "I":
                return transition_bias_inside_to_continue
            if cur_prefix == "E":
                return transition_bias_inside_to_end
            return 0.0
        if prev_prefix in {"E", "S"}:
            if cur_prefix == "O":
                return transition_bias_end_to_background
            if cur_prefix in {"B", "S"}:
                return transition_bias_end_to_start
            return 0.0
        return 0.0

    start_scores = [0.0 if start_allowed(state) else neg_inf for state in states]
    end_scores = [0.0 if end_allowed(state) else neg_inf for state in states]
    transition_scores = [
        [transition_bias(prev, cur) if transition_allowed(prev, cur) else neg_inf for cur in states] for prev in states
    ]
    return start_scores, end_scores, transition_scores


def zero_viterbi_transition_biases() -> dict[str, float]:
    return dict.fromkeys(VITERBI_BIAS_KEYS, 0.0)


def normalize_viterbi_transition_biases(
    transition_biases: Mapping[str, float | int] | None,
) -> dict[str, float]:
    if transition_biases is None:
        return zero_viterbi_transition_biases()
    extra = set(transition_biases) - set(VITERBI_BIAS_KEYS)
    missing = set(VITERBI_BIAS_KEYS) - set(transition_biases)
    if extra or missing:
        msg = (
            "Viterbi transition_biases must contain exactly "
            f"{sorted(VITERBI_BIAS_KEYS)} (missing={sorted(missing)}, extra={sorted(extra)})"
        )
        raise ValueError(
            msg,
        )
    values: list[float] = []
    for key in VITERBI_BIAS_KEYS:
        value = transition_biases[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"Viterbi transition bias {key!r} must be numeric"
            raise ValueError(msg)
        values.append(float(value))
    return dict(zip(VITERBI_BIAS_KEYS, values, strict=True))


def _normalize_viterbi_transition_biases(
    transition_biases: Mapping[str, float | int] | None,
) -> tuple[float, float, float, float, float, float]:
    """Return the six biases in VITERBI_BIAS_KEYS order.

    The exact positional order _bioes_constraint_cache unpacks them into. The prior body read a stale `illegal_*` key
    scheme that the 2026-06-14 module consolidation (c7cb67f) left behind; those keys never exist in the normalized dict,
    so every call KeyError-ed (viterbi_decode_logits was untested, so it went unnoticed until an eval consumer).

    Returns:
        The six biases as a tuple in ``VITERBI_BIAS_KEYS`` order, which is exactly the order the
        constraint cache unpacks them into. Position is the contract here, so reordering the
        tuple or the key list without the other silently mislabels every transition rather than
        failing.

    """
    normalized = normalize_viterbi_transition_biases(transition_biases)
    return (
        normalized["transition_bias_background_stay"],
        normalized["transition_bias_background_to_start"],
        normalized["transition_bias_inside_to_continue"],
        normalized["transition_bias_inside_to_end"],
        normalized["transition_bias_end_to_background"],
        normalized["transition_bias_end_to_start"],
    )


# reason: viterbi decode owns normalize and bioes cache together; splitting would misattribute row errors.
def viterbi_decode_logits(  # ruff: ignore[too-many-locals]
    logits: torch.Tensor,
    id_to_label: dict[int, str],
    offset_mapping: Sequence[tuple[int, int]],
    *,
    ignore_index: int = IGNORE_INDEX,
    transition_biases: Mapping[str, float | int] | None = None,
) -> list[int]:
    import torch

    if logits.ndim != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = f"Expected [tokens, labels] logits, got shape {tuple(logits.shape)}"
        raise ValueError(msg)

    sorted_ids = sorted(id_to_label)
    label_vocab = tuple(id_to_label[idx] for idx in sorted_ids)
    bias_values = _normalize_viterbi_transition_biases(transition_biases)
    start_scores, end_scores, transition_scores = _bioes_constraint_cache(label_vocab, bias_values)

    valid_positions = [idx for idx, (start, end) in enumerate(offset_mapping) if not (start == end == 0)]
    decoded = [ignore_index] * len(offset_mapping)
    if not valid_positions:
        return decoded

    emission = logits[valid_positions].to(torch.float32)
    device = emission.device
    start = torch.tensor(start_scores, dtype=torch.float32, device=device)
    end = torch.tensor(end_scores, dtype=torch.float32, device=device)
    transitions = torch.tensor(transition_scores, dtype=torch.float32, device=device)

    scores = emission[0] + start
    backpointers: list[torch.Tensor] = []

    for step in range(1, emission.size(0)):
        candidate_scores = scores.unsqueeze(1) + transitions + emission[step].unsqueeze(0)
        best_scores, best_prev = candidate_scores.max(dim=0)
        backpointers.append(best_prev)
        scores = best_scores

    scores += end
    best_last = int(scores.argmax().item())
    best_path = [best_last]
    for best_prev in reversed(backpointers):
        best_last = int(best_prev[best_last].item())
        best_path.append(best_last)
    best_path.reverse()

    for position, label_id in zip(valid_positions, best_path, strict=True):
        decoded[position] = label_id
    return decoded


# reason: viterbi decode coordinates normalize with bioes cache; extra seams would misattribute row errors.
def viterbi_decode_numpy(  # ruff: ignore[too-many-locals]
    logits: object,
    id_to_label: dict[int, str],
    offset_mapping: Sequence[tuple[int, int]],
    *,
    ignore_index: int = IGNORE_INDEX,
    transition_biases: Mapping[str, float | int] | None = None,
) -> list[int]:
    import numpy as np

    matrix = np.asarray(logits, dtype=np.float32)
    if matrix.ndim != 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = f"Expected [tokens, labels] logits, got shape {tuple(matrix.shape)}"
        raise ValueError(msg)

    sorted_ids = sorted(id_to_label)
    label_vocab = tuple(id_to_label[idx] for idx in sorted_ids)
    expected_shape = (len(offset_mapping), len(label_vocab))
    if matrix.shape != expected_shape:
        msg = f"Logits shape must match offsets and labels: expected={expected_shape} actual={tuple(matrix.shape)}"
        raise ValueError(
            msg,
        )
    bias_values = _normalize_viterbi_transition_biases(transition_biases)
    start_scores, end_scores, transition_scores = _bioes_constraint_cache(label_vocab, bias_values)

    valid_positions = [idx for idx, (start, end) in enumerate(offset_mapping) if not (start == end == 0)]
    decoded = [ignore_index] * len(offset_mapping)
    if not valid_positions:
        return decoded

    emission = matrix[valid_positions]
    start = np.asarray(start_scores, dtype=np.float32)
    end = np.asarray(end_scores, dtype=np.float32)
    transitions = np.asarray(transition_scores, dtype=np.float32)
    scores = emission[0] + start
    backpointers: list[list[int]] = []

    for step in range(1, emission.shape[0]):
        candidate_scores = scores[:, None] + transitions + emission[step][None, :]
        best_prev = np.argmax(candidate_scores, axis=0)
        scores = candidate_scores[best_prev, np.arange(candidate_scores.shape[1])]
        backpointers.append([int(label_id) for label_id in best_prev])

    best_last = int(np.argmax(scores + end))
    best_path = [best_last]
    for best_prev in reversed(backpointers):
        best_last = int(best_prev[best_last])
        best_path.append(best_last)
    best_path.reverse()

    for position, label_id in zip(valid_positions, best_path, strict=True):
        decoded[position] = label_id
    return decoded
