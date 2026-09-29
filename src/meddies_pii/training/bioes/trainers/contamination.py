"""Runtime probes for packed-attention contamination.

The BIOES packing contract is only valid when changing one packed document
cannot change another document's logits.  These probes compare the target
document logits under a clean prefix and a mutated prefix while keeping the
target tokens fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .packing import PackedTrainingUnit, collate_packed_units


@dataclass(frozen=True, slots=True)
class PackedAttentionContaminationResult:
    passed: bool
    target_row_uid: str
    token_start: int
    token_end: int
    compared_tokens: int
    max_abs_diff: float
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def packed_row_token_range(
    unit: PackedTrainingUnit,
    *,
    row_uid: str,
) -> tuple[int, int]:
    for row_range in unit.row_ranges:
        if row_range.uid == row_uid:
            return row_range.start, row_range.end
    msg = f"row_uid={row_uid!r} is not present in packed unit"
    raise ValueError(msg)


# reason: run packed exposes model/rtol as its public contract; bundling would break callers.
def run_packed_attention_contamination_probe(  # ruff: ignore[too-many-arguments]
    model: torch.nn.Module,
    *,
    clean_unit: PackedTrainingUnit,
    mutated_unit: PackedTrainingUnit,
    target_row_uid: str,
    device: str,
    atol: float = 1e-6,
    rtol: float = 0.0,
) -> PackedAttentionContaminationResult:
    """Return whether target logits are invariant to a different packed prefix.

    Returns:
        The comparison result. Invariance is the PASS: the target row's logits must not move
        when a different row is packed before it, because a difference means attention
        crossed the row boundary and one training example leaked into another's loss.

    Raises:
        ValueError: When the two units are not comparable -- unequal token length, a target
            row whose token range moved between them, or target tokens that are not
            byte-identical. All three are checked BEFORE any model call, because a probe run
            on mismatched units would report a logit difference caused by the mismatch and
            read as contamination that is not there.

    """
    if len(clean_unit.input_ids) != len(mutated_unit.input_ids):
        msg = "clean and mutated units must have equal token length"
        raise ValueError(msg)
    token_start, token_end = packed_row_token_range(clean_unit, row_uid=target_row_uid)
    mutated_start, mutated_end = packed_row_token_range(mutated_unit, row_uid=target_row_uid)
    if (token_start, token_end) != (mutated_start, mutated_end):
        msg = "target row token range changed between clean/mutated units"
        raise ValueError(msg)
    if clean_unit.input_ids[token_start:token_end] != mutated_unit.input_ids[token_start:token_end]:
        msg = "target row tokens must be identical for contamination probe"
        raise ValueError(msg)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        clean_logits = model(**collate_packed_units([clean_unit], device=device))["logits"]
        mutated_logits = model(**collate_packed_units([mutated_unit], device=device))["logits"]
    if was_training:
        model.train()

    clean_target = clean_logits[:, token_start:token_end, :]
    mutated_target = mutated_logits[:, token_start:token_end, :]
    diff = (clean_target - mutated_target).abs()
    max_abs_diff = float(diff.max().detach().cpu().item()) if diff.numel() else 0.0
    passed = bool(torch.allclose(clean_target, mutated_target, atol=atol, rtol=rtol))
    return PackedAttentionContaminationResult(
        passed=passed,
        target_row_uid=target_row_uid,
        token_start=token_start,
        token_end=token_end,
        compared_tokens=token_end - token_start,
        max_abs_diff=max_abs_diff,
        atol=atol,
        rtol=rtol,
    )
