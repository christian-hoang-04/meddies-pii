"""Immutable, CPU-renderable contract for the PII350 2xA100 DDP smoke lane.

This module contains no Modal, CUDA, model, or dataset imports.  The paid lane is
separate so reviewing either command cannot accidentally launch it.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra with no macOS wheels, so it loads inside the call that needs it.
# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
import argparse
import json
import math
from hashlib import sha256
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from .base_selection import CANDIDATES, GATE_SEED
from .config import LORA_TARGET_MODULES
from .full_run import ENCODER_FULL_RUN_IMAGE_PACKAGES, render_pii350_scout_contract
from .pins import (
    PACKED_DATASET_ID,
    PACKED_DATASET_REVISION,
    PACKED_MANIFEST_SHA256,
    PACKED_SHARD_COUNT,
    PACKED_UNIT_COUNT,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

WORLD_SIZE = 2
LOCAL_BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
OPTIMIZER_STEPS = 10
DDP_SMOKE_CONFIRMATION = "LAUNCH_PII350_DDP_A100_40GB_X2_SMOKE"
DDP_SMOKE_PRIMARY_ACTION = "HA_AUTHORIZE_PII350_R128A256_B128_DDP_A10040GBX2_10STEP_SMOKE"
MODAL_PROFILE = "diffusionllm"
"""Ha selected the profile only after the contract was scoped.

It is deliberately part of the immutable digest so the reviewed dispatch cannot drift accounts.

"""
GPU_SPEC = "A100-40GB:2"
HARD_TIMEOUT_SECONDS = 1_800
TRAINING_DEADLINE_SECONDS = 1_500
BF16_PARAMETER_DRIFT_TOLERANCE = 0.0
"""DDP all-reduces the fp32 LoRA/head gradients and runs the same optimizer on every rank.

So any non-zero post-update drift is a correctness failure.

"""
DDP_GRADIENT_REDUCTION = "no_sync_then_explicit_all_reduce_mean_v1"
"""``HiddenStateTokenTagger`` invokes an Unsloth-reentrant-checkpointed encoder once per packed document.

The number of documents varies by rank and batch, which is outside stock DDP's supported repeated-reentrant-checkpoint
case. Keep DDP's initial replica broadcast, but reduce the complete local gradient set explicitly after the outer
forward/backward finishes under ``no_sync``.

"""
A100_40GB_RATE_USD_PER_SECOND = 0.000583
CPU_RATE_USD_PER_SECOND = 4 * 0.0000131
MEMORY_RATE_USD_PER_SECOND = 64 * 0.00000222
ALL_IN_RATE_USD_PER_SECOND = 2 * A100_40GB_RATE_USD_PER_SECOND + CPU_RATE_USD_PER_SECOND + MEMORY_RATE_USD_PER_SECOND


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contract_digest(contract: Mapping[str, Any]) -> str:
    """Hash a contract regardless of whether its digest field is already present.

    Returns:
        The canonical SHA-256 of the contract with ``config_digest`` removed, so a rendered
        contract and the same contract before its digest was attached hash identically.

    """
    body = {key: value for key, value in contract.items() if key != "config_digest"}
    return _canonical_digest(body)


def matched_constant3_contract() -> dict[str, Any]:
    """Use the existing live 3e-4 scout as the optimizer/mechanics source of truth.

    Returns:
        The rendered ``lr3e-4`` PII350 scout contract, whose optimizer block and mechanics the
        DDP smoke contract copies rather than restating.

    """
    return render_pii350_scout_contract("lr3e-4")


def render_ddp_smoke_contract() -> dict[str, Any]:
    """Render the single approved two-rank experiment; never infer a launch default.

    Returns:
        The complete immutable contract with its ``config_digest`` attached last, so the digest
        covers every other field. ``require_ddp_smoke_execute`` compares against this exact dict.

    """
    scout = matched_constant3_contract()
    candidate = CANDIDATES["pii350"]
    body: dict[str, Any] = {
        "contract_version": 1,
        "candidate_key": "pii350",
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "seed": GATE_SEED,
        "runtime_packages": list(ENCODER_FULL_RUN_IMAGE_PACKAGES),
        "topology": {
            "gpu": GPU_SPEC,
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "model_parallelism": False,
        },
        "packed_dataset": {
            "id": PACKED_DATASET_ID,
            "revision": PACKED_DATASET_REVISION,
            "manifest_sha256": PACKED_MANIFEST_SHA256,
            "shard_count": PACKED_SHARD_COUNT,
            "packed_unit_count": PACKED_UNIT_COUNT,
            "config": "packed",
            "max_length": 8192,
            "segment_isolation": "lfm2_segment_isolation_v1",
        },
        "model_mechanics": {
            "backend": "unsloth",
            "precision": "bfloat16",
            "gradient_checkpointing": "unsloth",
            "lora": {
                "rank": 128,
                "alpha": 256,
                "dropout": 0.0,
                "target_modules": list(LORA_TARGET_MODULES),
            },
            "head": "Dropout(0.1)+Linear(1024,37,bfloat16)",
        },
        "training": {
            "optimizer_steps": OPTIMIZER_STEPS,
            "evaluation": False,
            "checkpoint": False,
            "optimizer": {
                "name": "adamw",
                "lr": 3e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "fused": False,
                "scheduler": None,
            },
            "loss_reduction": (
                "explicit_gradient_bucket_all_reduce_mean_of_"
                "world_size_times_local_active_labels_over_global_active_labels"
            ),
        },
        "attestation": {
            "parameter_drift_tolerance": BF16_PARAMETER_DRIFT_TOLERANCE,
            "steps": [1, 10],
            "require_exact_world_size": True,
            "require_rank_coverage": True,
            "require_packed_segment_isolation": True,
            "interconnect_evidence": (
                "pre_torchrun_uuid_inventory_plus_raw_nvlink_status_and_symmetric_peer_access_nccl_readiness_v1"
            ),
            "gradient_reduction": DDP_GRADIENT_REDUCTION,
            "require_step_one_parameter_gradient_diagnostics": True,
            "step_one_diagnostic_persistence": ("rank_zero_atomic_json_with_fresh_nested_parent_v1"),
        },
        "budget": {
            "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
            "training_deadline_seconds": TRAINING_DEADLINE_SECONDS,
            "shutdown_reserve_seconds": HARD_TIMEOUT_SECONDS - TRAINING_DEADLINE_SECONDS,
            "deadline_scope": (
                "outer_modal_function_entry_through_child_initialization_and_training; "
                "absolute_monotonic_deadline_shared_by_torchrun_children_in_one_container"
            ),
            "all_in_rate_usd_per_second": ALL_IN_RATE_USD_PER_SECOND,
            "pricing_basis": "two A100-40GB GPUs plus 4 physical CPU cores plus 64 GiB memory",
        },
        "manual_launch": {
            "automatic_launch": False,
            "modal_profile": MODAL_PROFILE,
            "requires_confirmation": DDP_SMOKE_CONFIRMATION,
            "requires_primary_action": DDP_SMOKE_PRIMARY_ACTION,
            "source_constant3_scout_digest": scout["config_digest"],
        },
    }
    return {**body, "config_digest": _canonical_digest(body)}


def require_ddp_smoke_execute(
    contract: Mapping[str, Any],
    *,
    execute: bool,
    confirmation: str,
    primary_action: str,
) -> None:
    """Reject all partial, stale, or accidental paid dispatches.

    Raises:
        RuntimeError: If the contract is not byte-identical to ``render_ddp_smoke_contract()``,
            or ``execute`` is false, or either the confirmation phrase or the primary action
            differs from the constant the contract pins. Each of the four is refused separately.

    """
    expected = render_ddp_smoke_contract()
    if dict(contract) != expected:
        msg = "PII350 DDP smoke requires the exact immutable contract"
        raise RuntimeError(msg)
    if not execute:
        msg = "PII350 DDP smoke requires --execute"
        raise RuntimeError(msg)
    if confirmation != DDP_SMOKE_CONFIRMATION:
        msg = "PII350 DDP smoke requires the exact confirmation"
        raise RuntimeError(msg)
    if primary_action != DDP_SMOKE_PRIMARY_ACTION:
        msg = "PII350 DDP smoke requires the explicit primary action"
        raise RuntimeError(msg)


def rank_window(*, global_step: int, rank: int) -> tuple[int, int]:
    """Return one rank's half-open packed-unit range in globally seeded order.

    Returns:
        The half-open ``(start, end)`` packed-unit indices this rank owns at this step.

    Raises:
        ValueError: If the step is not positive and one-indexed, or the rank is outside the
            pinned world size.

    """
    if global_step <= 0 or rank not in range(WORLD_SIZE):
        msg = "rank window requires a positive one-indexed step and a valid rank"
        raise ValueError(msg)
    start = (global_step - 1) * GLOBAL_BATCH_SIZE + rank * LOCAL_BATCH_SIZE
    return start, start + LOCAL_BATCH_SIZE


def validate_rank_windows(*, global_step: int, windows: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Prove exact coverage and no overlap before an update consumes the batch.

    Returns:
        The step, the single global window the rank windows span, and the expected per-rank
        windows -- the proof itself, for the receipt.

    Raises:
        RuntimeError: If any two rank ranges overlap, or the given windows are not exactly the
            ones ``rank_window`` derives for this step.

    """
    flattened = sorted(windows)
    if any(left[1] > right[0] for left, right in pairwise(flattened)):
        msg = "DDP rank packed-unit ranges overlap"
        raise RuntimeError(msg)
    expected = [rank_window(global_step=global_step, rank=rank) for rank in range(WORLD_SIZE)]
    if list(windows) != expected:
        msg = "DDP rank packed-unit ranges do not exactly cover the global batch"
        raise RuntimeError(msg)
    return {
        "global_step": global_step,
        "global_window": (expected[0][0], expected[-1][1]),
        "rank_windows": expected,
    }


def scale_local_token_mean_loss(loss: Tensor, *, local_active: Tensor, global_active: Tensor | int) -> Tensor:
    """Scale rank token means so DDP's gradient average equals the global token mean.

    Returns:
        The loss multiplied by ``world_size * local_active / global_active``, which cancels
        DDP's own averaging over ranks and leaves the global token mean.

    Raises:
        RuntimeError: If this rank or the run as a whole has no active labels, which would make
            the scale factor degenerate rather than merely small.

    """
    local_count = int(local_active.detach().item()) if hasattr(local_active, "detach") else int(local_active)
    global_count = global_active if isinstance(global_active, int) else int(global_active.detach().item())
    if local_count <= 0 or global_count <= 0:
        msg = "DDP token-mean loss requires active labels on every rank and globally"
        raise RuntimeError(msg)
    import torch

    local_tensor = torch.as_tensor(local_active, device=loss.device, dtype=loss.dtype)
    global_tensor = torch.as_tensor(global_active, device=loss.device, dtype=loss.dtype)
    return loss * (WORLD_SIZE * local_tensor / global_tensor)


def packed_unit_identities(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Give every packed unit a stable identity without copying CUDA tensors.

    Returns:
        One SHA-256 per row, taken over the row's ``row_uids`` (or ``uid``), in row order.

    Raises:
        RuntimeError: If two rows hash to the same identity, which means the global packed batch
            repeats a unit.

    """
    identities: list[str] = []
    for row in rows:
        source = row.get("row_uids", row.get("uid"))
        encoded = json.dumps(source, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        identities.append(sha256(encoded.encode()).hexdigest())
    if len(identities) != len(set(identities)):
        msg = "DDP global packed batch repeats a packed-unit identity"
        raise RuntimeError(msg)
    return identities


def ordered_identity_digest(identities: Sequence[str]) -> str:
    return sha256(json.dumps(list(identities), separators=(",", ":")).encode()).hexdigest()


def validate_rank_identity_coverage(*, global_identities: Sequence[str], rank_identities: Sequence[Sequence[str]]) -> str:
    """Prove the two rank slices reconstruct exactly one ordered global batch.

    Returns:
        The ordered digest of the global identities, which the receipt records as the proof.

    Raises:
        RuntimeError: If the global batch is not the pinned size, or a rank slice is not the
            pinned local size, or the concatenated rank slices are not the global list in order
            and without repeats.

    """
    if len(global_identities) != GLOBAL_BATCH_SIZE:
        msg = "DDP global identity proof requires exactly 128 packed units"
        raise RuntimeError(msg)
    flattened = [identity for identities in rank_identities for identity in identities]
    if any(len(identities) != LOCAL_BATCH_SIZE for identities in rank_identities):
        msg = "DDP rank identity proof requires 64 packed units per rank"
        raise RuntimeError(msg)
    if flattened != list(global_identities) or len(flattened) != len(set(flattened)):
        msg = "DDP rank identities overlap or do not exactly cover the global batch"
        raise RuntimeError(msg)
    return ordered_identity_digest(global_identities)


def aggregate_real_tokens_per_second(*, global_real_tokens: int, rank_step_seconds: Sequence[float]) -> float:
    """Report throughput against the slowest synchronized rank, not rank zero.

    Returns:
        Real tokens per second over the longest rank duration, which is what the synchronized
        step actually cost.

    Raises:
        RuntimeError: If the token count is not positive, the durations are not one per rank, or
            the slowest duration is not positive.

    """
    if global_real_tokens <= 0 or len(rank_step_seconds) != WORLD_SIZE or max(rank_step_seconds) <= 0:
        msg = "DDP aggregate throughput requires two positive rank durations"
        raise RuntimeError(msg)
    return global_real_tokens / max(rank_step_seconds)


def validate_collective_readiness(
    *,
    peer_access_records: Sequence[Mapping[str, Any]],
    observed_all_reduce_sum: int,
) -> dict[str, Any]:
    """Require bidirectional peer access and the deterministic two-rank NCCL sum.

    Returns:
        The ordered peer-access records and the all-reduce inputs, expected sum and observed sum,
        so the receipt carries the evidence rather than only the verdict.

    Raises:
        RuntimeError: If there is not one record per rank, the ranks are not exactly the pinned
            set, a record names the wrong peer, peer access is not granted in both directions, or
            the observed all-reduce sum differs from the deterministic expected one.

    """
    if len(peer_access_records) != WORLD_SIZE:
        msg = "DDP readiness probe requires one peer-access record per rank"
        raise RuntimeError(msg)
    ordered = sorted(peer_access_records, key=lambda record: int(record["rank"]))
    expected_ranks = list(range(WORLD_SIZE))
    if [record.get("rank") for record in ordered] != expected_ranks:
        msg = "DDP readiness probe ranks are not exactly {0, 1}"
        raise RuntimeError(msg)
    for record in ordered:
        rank = int(record["rank"])
        if record.get("peer_rank") != WORLD_SIZE - 1 - rank:
            msg = "DDP readiness probe peer rank is invalid"
            raise RuntimeError(msg)
        if record.get("can_access_peer") is not True:
            msg = "DDP readiness probe requires bidirectional CUDA peer access"
            raise RuntimeError(msg)
    expected_sum = sum(range(1, WORLD_SIZE + 1))
    if observed_all_reduce_sum != expected_sum:
        msg = f"DDP readiness probe NCCL all-reduce returned {observed_all_reduce_sum}, expected {expected_sum}"
        raise RuntimeError(
            msg,
        )
    return {
        "peer_access": [dict(record) for record in ordered],
        "nccl_all_reduce": {
            "inputs_by_rank": list(range(1, WORLD_SIZE + 1)),
            "expected_sum": expected_sum,
            "observed_sum": observed_all_reduce_sum,
        },
    }


def absolute_training_deadline(*, outer_started_monotonic: float) -> float:
    """Create the one soft deadline shared by the Modal parent and torchrun children.

    Returns:
        The absolute monotonic instant training must stop by, so parent and children compare
        against one value rather than each starting its own budget.

    Raises:
        ValueError: If the start instant is not finite and non-negative.

    """
    if not math.isfinite(outer_started_monotonic) or outer_started_monotonic < 0:
        msg = "outer_started_monotonic must be a finite non-negative value"
        raise ValueError(msg)
    return outer_started_monotonic + TRAINING_DEADLINE_SECONDS


def training_deadline_reached(*, absolute_deadline_monotonic: float, now_monotonic: float) -> bool:
    """Charge parent validation/spawn and child setup against one absolute deadline.

    Returns:
        True once the current monotonic instant has reached the absolute deadline.

    Raises:
        ValueError: If either instant is not finite.

    """
    if not math.isfinite(absolute_deadline_monotonic) or not math.isfinite(now_monotonic):
        msg = "deadline values must be finite"
        raise ValueError(msg)
    return now_monotonic >= absolute_deadline_monotonic


def render_ddp_smoke_preflight_command() -> str:
    return (
        f"MODAL_PROFILE={MODAL_PROFILE} uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.pii350_ddp_smoke --preflight-assets"
    )


def render_ddp_smoke_prewarm_command() -> str:
    """Render the CPU-only online hydration prerequisite for the DDP receipt.

    Returns:
        The exact ``modal run --prewarm-assets`` command line, profile included, so the operator
        does not assemble it by hand.

    """
    return (
        f"MODAL_PROFILE={MODAL_PROFILE} uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.pii350_ddp_smoke --prewarm-assets"
    )


def render_ddp_smoke_launch_command() -> str:
    contract = render_ddp_smoke_contract()
    return (
        f"MODAL_PROFILE={MODAL_PROFILE} uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.pii350_ddp_smoke --execute "
        f"--confirmation {DDP_SMOKE_CONFIRMATION} --primary-action {DDP_SMOKE_PRIMARY_ACTION} "
        f"--config-digest {contract['config_digest']}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-contract", action="store_true")
    parser.add_argument("--render-preflight-command", action="store_true")
    parser.add_argument("--render-prewarm-command", action="store_true")
    parser.add_argument("--render-launch-command", action="store_true")
    args = parser.parse_args()
    selected = sum((
        args.render_contract,
        args.render_preflight_command,
        args.render_prewarm_command,
        args.render_launch_command,
    ))
    if selected != 1:
        msg = "choose exactly one render flag"
        raise SystemExit(msg)
    if args.render_contract:
        print(json.dumps(render_ddp_smoke_contract(), indent=2, sort_keys=True))
    elif args.render_preflight_command:
        print(render_ddp_smoke_preflight_command())
    elif args.render_prewarm_command:
        print(render_ddp_smoke_prewarm_command())
    else:
        print(render_ddp_smoke_launch_command())


if __name__ == "__main__":
    _main()
