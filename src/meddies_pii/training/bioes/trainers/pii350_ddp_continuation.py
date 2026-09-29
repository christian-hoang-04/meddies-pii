"""Pure contracts for topology-flexible PII350 checkpoint continuation."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import argparse
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from meddies_pii.json_types import is_str_mapping

from .base_selection import CANDIDATES, GATE_SEED
from .config import LORA_TARGET_MODULES
from .full_run import render_pii350_scout_contract
from .pins import PACKED_DATASET_ID, PACKED_DATASET_REVISION, PACKED_MANIFEST_SHA256

if TYPE_CHECKING:
    from torch import Tensor

COST_ABS_TOLERANCE = 1e-9
SHA256_HEX_LENGTH = 64

WORLD_SIZE = 4
"""Default compatibility topology; execution is profile-selected."""
LOCAL_BATCH_SIZE = 32
SUPPORTED_WORLD_SIZES = (2, 4)
GLOBAL_BATCH_SIZE = 128
EPOCH_CURSOR = 120_036
EPOCH_TERMINAL_STEP = 938
FINAL_GLOBAL_START = 119_936
FINAL_GLOBAL_BATCH_SIZE = 100
FINAL_LOCAL_BATCH_SIZE = 25
"""Per-rank batch size on the epoch's final, short global batch.

``rank_windows`` derives this at runtime as ``FINAL_GLOBAL_BATCH_SIZE // world_size``. The name
exists so a plan or a receipt can state the four-rank value without re-deriving it; the pinned
equality is checked in ``tests/bioes/test_pii350_ddp_continuation.py``.

"""
SOURCE_STEP = 60
SOURCE_CURSOR = 7_680
INTERRUPTION_COST_RECEIPT_SCHEMA_VERSION = 1
PREEMPTED_WAVE2_SOURCE_CHECKPOINT_DIGEST = "10c305bfc0d4c33ec16a192574c631366461bb8955903035ade5424dea4172e2"
PREEMPTED_WAVE2_SOURCE_STEP = 200
PREEMPTED_WAVE2_SOURCE_CURSOR = 25_600
PREEMPTED_WAVE2_OBSERVED_MODAL_COST_USD = 10.5361509104
CHECKPOINT_SCHEMA_VERSION = 2
RNG_TRANSITION = "single_rank_to_four_rank_v1"
GPU_SPEC = "A100-40GB:4"
HARD_TIMEOUT_SECONDS = 10_800
SHUTDOWN_RESERVE_SECONDS = 600
A100_40GB_RATE_USD_PER_SECOND = 0.000583
CPU_RATE_USD_PER_SECOND = 0.0000131
MEMORY_RATE_USD_PER_SECOND = 0.00000222
WAVE_ONE_FAILED_CONTAINER_CARRY_USD = 2.16724464
"""First-wave sunk carry: two failed A100 containers ($2.0543248 + $0.11291984)."""
WAVE_ONE_LIVE_REMAINING_BALANCE_USD = 11.0
EXECUTION_TOPOLOGIES = {
    "meddiesresearch": {
        "world_size": 2,
        "gpu": "A100-40GB:2",
        "cpu": 4.0,
        "memory_mib": 64 * 1024,
        "timeout_seconds": 7_900,
    },
    "huyhoang0411ha": {
        "world_size": 4,
        "gpu": "A100-40GB:4",
        "cpu": 8.0,
        "memory_mib": 128 * 1024,
        "timeout_seconds": 10_800,
    },
    "meddies-ocr": {
        "world_size": 4,
        "gpu": "A100-40GB:4",
        "cpu": 8.0,
        "memory_mib": 128 * 1024,
        "timeout_seconds": 10_800,
    },
}


def topology_all_in_rate(topology: Mapping[str, Any]) -> float:
    return (
        int(topology["world_size"]) * A100_40GB_RATE_USD_PER_SECOND
        + float(topology["cpu"]) * CPU_RATE_USD_PER_SECOND
        + int(topology["memory_mib"]) / 1024 * MEMORY_RATE_USD_PER_SECOND
    )


def authorization_tokens(world_size: int) -> dict[str, str]:
    try:
        return {
            "confirmation": TRAIN_CONFIRMATION_BY_WORLD_SIZE[world_size],
            "primary_action": PRIMARY_ACTION_BY_WORLD_SIZE[world_size],
        }
    except KeyError as error:
        msg = "continuation authorization requires a supported world size"
        raise RuntimeError(msg) from error


SLICE_TIMEOUT_SCHEDULE: dict[str, tuple[int, ...]] = {
    "meddiesresearch": (7_900,),
    "huyhoang0411ha": (10_800, 7_000, 6_000, 4_000),
    "meddies-ocr": (10_800, 4_000),
}
TRAIN_CONFIRMATION_BY_WORLD_SIZE = {
    2: "LAUNCH_PII350_DDP2_CONTINUATION",
    4: "LAUNCH_PII350_DDP4_CONTINUATION",
}
PRIMARY_ACTION_BY_WORLD_SIZE = {
    2: "HA_AUTHORIZE_PII350_R128A256_B128_LR4E4_DDP2_CONTINUATION",
    4: "HA_AUTHORIZE_PII350_R128A256_B128_LR4E4_DDP4_CONTINUATION",
}
TRAIN_CONFIRMATION = TRAIN_CONFIRMATION_BY_WORLD_SIZE[4]
"""Compatibility exports for historical four-rank callers.

Profile-bound launch surfaces must obtain their pair through ``authorization_tokens``.

"""
PRIMARY_ACTION = PRIMARY_ACTION_BY_WORLD_SIZE[4]
EVAL_ROWS = 1_700
EVALUATION_CREDITS: dict[str, float] = {
    "diffusionllm": 4.0,
    "meddies-pii": 3.8,
    "huyhoang041100": 3.0,
    "meddies-run": 2.9,
}
WAVE_ORDER = ("meddiesresearch", "huyhoang0411ha", "meddies-ocr")
SEGMENTS: dict[str, float] = {
    "meddiesresearch": WAVE_ONE_LIVE_REMAINING_BALANCE_USD,
    "huyhoang0411ha": 30.0,
    "meddies-ocr": 71.0,
}
"""This is Modal's reported live balance.

It already reflects earlier spend.

"""
CPU_AUXILIARY_RATE_USD_PER_SECOND = 2 * 0.0000131 + 8 * 0.00000222
"""Each CPU surface is limited to 1,800 seconds at 2 CPU and 8 GiB.

A normal wave permits four calls (preflight, source receipt, upload, download); OCR reserves twelve because its three
planned resumable slices repeat transport.

"""
AUXILIARY_BILLING_RESERVE_USD: dict[str, float] = {
    "meddiesresearch": 0.40,
    "huyhoang0411ha": 0.40,
    "meddies-ocr": 1.20,
}
AUXILIARY_MAX_CPU_CALLS: dict[str, int] = {
    "meddiesresearch": 4,
    "huyhoang0411ha": 4,
    "meddies-ocr": 12,
}
CHECKPOINT_REQUIRED_FILES = (
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "classifier.pt",
    "optimizer.pt",
    "rng.pt",
    "metadata.json",
)


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def trajectory_contract() -> dict[str, Any]:
    """Keep construction on the already-proven PII350 Unsloth path.

    This continuation changes process topology, not the model implementation.

    Returns:
        The trajectory block: model pin, seed, backend, LoRA and head shape, optimizer,
        packed dataset pin, and the baseline scout contract it inherits. Everything here is
        hashed into ``trajectory_digest``, so any edit changes every downstream digest.

    """
    candidate = CANDIDATES["pii350"]
    return {
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "seed": GATE_SEED,
        "backend": "unsloth==2026.7.4",
        "precision": "bfloat16",
        "lora": {
            "rank": 128,
            "alpha": 256,
            "dropout": 0.0,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "head": "Dropout(0.1)+Linear(1024,37,bfloat16)",
        "optimizer": {
            "name": "adamw",
            "lr": 4e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "scheduler": None,
        },
        "packed_dataset": {
            "id": PACKED_DATASET_ID,
            "revision": PACKED_DATASET_REVISION,
            "manifest_sha256": PACKED_MANIFEST_SHA256,
            "segment_isolation": "lfm2_segment_isolation_v1",
        },
        "baseline_full_run_contract": render_pii350_scout_contract("lr4e-4"),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "loss": "global_token_mean_cross_entropy",
        "label_count": 37,
    }


def render_segment_contract(profile: str) -> dict[str, Any]:
    if profile not in SEGMENTS:
        msg = "continuation requires one approved training profile"
        raise ValueError(msg)
    trajectory = trajectory_contract()
    topology = EXECUTION_TOPOLOGIES[profile]
    world_size = topology["world_size"]
    if not isinstance(world_size, int):
        msg = "continuation topology world size is invalid"
        raise RuntimeError(msg)
    execution = {
        "trajectory_digest": _digest(trajectory),
        "profile": profile,
        "gpu": topology["gpu"],
        "world_size": world_size,
        "local_batch_size": GLOBAL_BATCH_SIZE // world_size,
        "cpu": topology["cpu"],
        "memory_mib": topology["memory_mib"],
        "timeout_seconds": topology["timeout_seconds"],
        "all_in_rate_usd_per_second": topology_all_in_rate(topology),
        "shutdown_reserve_seconds": SHUTDOWN_RESERVE_SECONDS,
        "all_in_ceiling_usd": SEGMENTS[profile],
        "budget_basis": (
            "reported_modal_live_remaining_balance_usd"
            if profile == "meddiesresearch"
            else "approved_wave_all_in_ceiling_usd"
        ),
        "historical_failed_container_carry_usd": (
            WAVE_ONE_FAILED_CONTAINER_CARRY_USD if profile == "meddiesresearch" else 0.0
        ),
        "checkpoint_interval_steps": 50,
        "slice_timeout_schedule_seconds": list(SLICE_TIMEOUT_SCHEDULE[profile]),
        "wave_order": list(WAVE_ORDER),
        "epoch_terminal": {
            "step": EPOCH_TERMINAL_STEP,
            "cursor": EPOCH_CURSOR,
            "final_global_batch_size": FINAL_GLOBAL_BATCH_SIZE,
        },
        "auxiliary_billing_reserve": {
            "usd": AUXILIARY_BILLING_RESERVE_USD[profile],
            "cpu_rate_usd_per_second": CPU_AUXILIARY_RATE_USD_PER_SECOND,
            "max_cpu_calls": AUXILIARY_MAX_CPU_CALLS[profile],
            "per_call_timeout_seconds": 1_800,
            "basis": "bounded 2CPU+8GiB preflight/receipt/transport plus volume-network uncertainty",
        },
    }
    return {
        "trajectory": trajectory,
        "trajectory_digest": _digest(trajectory),
        "execution": execution,
        "execution_contract_digest": _digest(execution),
        "source": {"step": SOURCE_STEP, "cursor": SOURCE_CURSOR},
        "manual_launch": {
            **authorization_tokens(world_size),
            "automatic_retry": False,
        },
    }


def launch_digest(execution_contract_digest: str, source_receipt: Mapping[str, Any]) -> str:
    """Bind the reviewed paid action to one immutable parent checkpoint.

    Returns:
        The digest over the execution contract and the source receipt's checkpoint digest,
        optimizer step, packed cursor and world size. Changing any one of those five yields a
        different launch identity, which is what stops a resume from silently retargeting.

    Raises:
        RuntimeError: If the execution contract digest is empty or not a string, or if any
            of the four required source-receipt fields is None. It refuses rather than
            hashing a partial receipt into a launch identity that would look valid.

    """
    fields = ("checkpoint_digest", "optimizer_step", "packed_cursor", "world_size")
    if not isinstance(execution_contract_digest, str) or not execution_contract_digest:
        msg = "launch digest requires an execution contract digest"
        raise RuntimeError(msg)
    if any(source_receipt.get(field) is None for field in fields):
        msg = "launch digest requires a complete verified source receipt"
        raise RuntimeError(msg)
    return _digest({
        "execution_contract_digest": execution_contract_digest,
        "source_checkpoint_digest": source_receipt["checkpoint_digest"],
        "optimizer_step": source_receipt["optimizer_step"],
        "packed_cursor": source_receipt["packed_cursor"],
        "world_size": source_receipt["world_size"],
    })


def launch_attempt_digest(
    execution_contract_digest: str,
    source_launch_digest: str,
    launch_attempt_nonce: str,
) -> str:
    """Bind one human-approved paid attempt, so a Modal retry has no new identity.

    Returns:
        The digest over the execution contract digest, the source launch digest and the
        attempt nonce. A Modal retry of the same approved attempt reuses the same nonce and
        so recomputes the same identity; a new attempt needs a new approval.

    Raises:
        RuntimeError: If any of the three inputs is not lowercase hexadecimal of its exact
            length -- 64 for both digests, 32 for the nonce. Each is checked separately and
            reported separately; an uppercase or short value is refused, not normalized.

    """

    def is_lower_hex(value: object, *, length: int) -> bool:
        return (
            isinstance(value, str) and len(value) == length and all(character in "0123456789abcdef" for character in value)
        )

    if not is_lower_hex(execution_contract_digest, length=64):
        msg = "launch attempt requires an execution contract digest"
        raise RuntimeError(msg)
    if not is_lower_hex(source_launch_digest, length=64):
        msg = "launch attempt requires a source launch digest"
        raise RuntimeError(msg)
    if not is_lower_hex(launch_attempt_nonce, length=32):
        msg = "launch attempt requires one 32-character lowercase hexadecimal nonce"
        raise RuntimeError(msg)
    return _digest({
        "execution_contract_digest": execution_contract_digest,
        "source_launch_digest": source_launch_digest,
        "launch_attempt_nonce": launch_attempt_nonce,
    })


def preflight_digest(
    execution_contract_digest: str,
    *,
    packed_paths_sha256: str,
    shard_count: int,
    encoder_checkpoint_attestation: Mapping[str, Any],
) -> str:
    if not execution_contract_digest or not packed_paths_sha256 or shard_count <= 0:
        msg = "preflight digest requires verified packed inventory"
        raise RuntimeError(msg)
    if not encoder_checkpoint_attestation:
        msg = "preflight digest requires encoder attestation"
        raise RuntimeError(msg)
    return _digest({
        "execution_contract_digest": execution_contract_digest,
        "packed_paths_sha256": packed_paths_sha256,
        "shard_count": shard_count,
        "encoder_checkpoint_attestation": dict(encoder_checkpoint_attestation),
    })


def checkpoint_directory_name(*, step: int, terminal: bool) -> str:
    if step < SOURCE_STEP:
        msg = "continuation checkpoint step predates the source lineage"
        raise ValueError(msg)
    return f"terminal-step-{step:08d}" if terminal else f"step-{step:08d}"


def validate_wave_source(profile: str, source_receipt: Mapping[str, Any]) -> float:
    """Allow an in-wave resume or exactly the prior renewable-credit wave.

    Returns:
        The cumulative all-in cost carried forward: the source's own figure when this is a
        resume of the same wave, and 0.0 when the parent is the prior wave, whose spend sat
        against the prior wave's ceiling.

    Raises:
        RuntimeError: If the profile is not in the wave order; if the source receipt is
            epoch-complete, which ends training rather than continuing it; if the parent wave
            is neither this wave nor exactly its predecessor; or if the source's cumulative
            cost is negative.

    """
    if profile not in WAVE_ORDER:
        msg = "continuation wave profile is invalid"
        raise RuntimeError(msg)
    if source_receipt.get("epoch_complete") is True:
        msg = "epoch-terminal continuation refuses further training"
        raise RuntimeError(msg)
    source_profile = source_receipt.get("wave_profile")
    index = WAVE_ORDER.index(profile)
    if index == 0:
        if source_profile not in {None, profile}:
            msg = "first continuation wave has an invalid parent wave"
            raise RuntimeError(msg)
    elif source_profile not in {WAVE_ORDER[index - 1], profile}:
        msg = "continuation wave must follow its exact prior wave or resume itself"
        raise RuntimeError(msg)
    cumulative = float(source_receipt.get("wave_cumulative_all_in_cost_usd", 0.0))
    if cumulative < 0:
        msg = "continuation source has an invalid cumulative cost"
        raise RuntimeError(msg)
    return cumulative if source_profile == profile else 0.0


def _require_finite_nonnegative_cost(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or type(value) not in {int, float} or not isfinite(value) or value < 0:
        msg = f"interruption cost receipt has an invalid {field}"
        raise RuntimeError(msg)
    return float(value)


def validate_interruption_cost_receipt(
    profile: str,
    source_receipt: Mapping[str, Any],
    interruption_cost_receipt: Mapping[str, Any],
) -> float:
    """Return a reconciled preemption cost only for the exact resumed lineage.

    Returns:
        The receipt's cumulative all-in cost, once it is proven to describe this launch and
        to be internally consistent.

    Raises:
        RuntimeError: If the receipt is not a mapping; if its schema version, profile,
            execution contract digest, source checkpoint digest or source launch digest does
            not match the launch this call recomputes; if any of the three cost fields is not
            a finite non-negative number; if the retry cost is not positive, since a resume
            must reserve it; if the cumulative cost is not the sum of observed and retry
            within tolerance; or, for the one pinned preempted Wave-2 source, if the observed
            cost is below the Modal spend already on record for it.

    """
    if not isinstance(interruption_cost_receipt, Mapping):
        msg = "continuation requires an interruption cost receipt"
        raise RuntimeError(msg)
    contract = render_segment_contract(profile)
    source_launch = launch_digest(contract["execution_contract_digest"], source_receipt)
    required = {
        "schema_version": INTERRUPTION_COST_RECEIPT_SCHEMA_VERSION,
        "profile": profile,
        "execution_contract_digest": contract["execution_contract_digest"],
        "source_checkpoint_digest": source_receipt.get("checkpoint_digest"),
        "source_launch_digest": source_launch,
    }
    if any(interruption_cost_receipt.get(key) != value for key, value in required.items()):
        msg = "interruption cost receipt does not bind this exact resumed launch"
        raise RuntimeError(msg)
    observed_cost = _require_finite_nonnegative_cost(
        interruption_cost_receipt.get("observed_modal_cost_usd"),
        field="observed_modal_cost_usd",
    )
    retry_cost = _require_finite_nonnegative_cost(
        interruption_cost_receipt.get("automatic_retry_cost_usd"),
        field="automatic_retry_cost_usd",
    )
    cumulative_cost = _require_finite_nonnegative_cost(
        interruption_cost_receipt.get("wave_cumulative_all_in_cost_usd"),
        field="wave_cumulative_all_in_cost_usd",
    )
    if retry_cost <= 0:
        msg = "interruption cost receipt must reserve the automatic retry cost"
        raise RuntimeError(msg)
    if abs(cumulative_cost - (observed_cost + retry_cost)) > COST_ABS_TOLERANCE:
        msg = "interruption cost receipt cumulative cost is inconsistent"
        raise RuntimeError(msg)
    is_preempted_wave2_source = (
        profile == "huyhoang0411ha"
        and source_receipt.get("checkpoint_digest") == PREEMPTED_WAVE2_SOURCE_CHECKPOINT_DIGEST
        and source_receipt.get("optimizer_step") == PREEMPTED_WAVE2_SOURCE_STEP
        and source_receipt.get("packed_cursor") == PREEMPTED_WAVE2_SOURCE_CURSOR
    )
    if is_preempted_wave2_source and observed_cost < PREEMPTED_WAVE2_OBSERVED_MODAL_COST_USD:
        msg = "interruption cost receipt cannot be lower than observed Modal spend"
        raise RuntimeError(msg)
    return cumulative_cost


def requires_interruption_cost_receipt(profile: str, source_receipt: Mapping[str, object]) -> bool:
    return (
        profile == "huyhoang0411ha"
        and source_receipt.get("checkpoint_digest") == PREEMPTED_WAVE2_SOURCE_CHECKPOINT_DIGEST
        and source_receipt.get("optimizer_step") == PREEMPTED_WAVE2_SOURCE_STEP
        and source_receipt.get("packed_cursor") == PREEMPTED_WAVE2_SOURCE_CURSOR
    )


def effective_wave_carried_cost(
    profile: str,
    source_receipt: Mapping[str, Any],
    *,
    interruption_cost_receipt: Mapping[str, Any] | None = None,
) -> float:
    """Return only spend that remains outside the contract's budget basis.

    The $11 live Modal balance already includes every earlier Wave-1 cost.

    Returns:
        The spend to charge against this wave's ceiling: the wave's own carried cost on a
        same-wave resume, 0.0 for the balance-based profile, otherwise that profile's
        auxiliary reserve. When an interruption receipt is supplied the result is the larger
        of that base and the reconciled figure, so reconciliation can only raise the charge.

    Raises:
        RuntimeError: If this source requires a reconciled interruption cost receipt and none
            was given, plus everything ``validate_wave_source`` and
            ``validate_interruption_cost_receipt`` refuse, both of which run here.

    """
    carried = validate_wave_source(profile, source_receipt)
    if source_receipt.get("wave_profile") == profile:
        base_carried_cost = carried
    elif profile == "meddiesresearch":
        base_carried_cost = 0.0
    else:
        base_carried_cost = AUXILIARY_BILLING_RESERVE_USD[profile]
    if interruption_cost_receipt is None:
        if requires_interruption_cost_receipt(profile, source_receipt):
            msg = "preempted Wave-2 resume requires a reconciled interruption cost receipt"
            raise RuntimeError(msg)
        return base_carried_cost
    reconciled_cost = validate_interruption_cost_receipt(profile, source_receipt, interruption_cost_receipt)
    return max(base_carried_cost, reconciled_cost)


def select_slice_timeout_seconds(profile: str, *, carried_cost_usd: float) -> int:
    if profile not in SEGMENTS or carried_cost_usd < 0:
        msg = "slice selection has invalid profile or carried cost"
        raise RuntimeError(msg)
    ceiling = SEGMENTS[profile]
    for timeout in SLICE_TIMEOUT_SCHEDULE[profile]:
        if carried_cost_usd + timeout * topology_all_in_rate(EXECUTION_TOPOLOGIES[profile]) <= ceiling:
            return timeout
    msg = "no approved hard-timeout denomination fits the remaining wave balance"
    raise RuntimeError(msg)


def require_slice_admission(profile: str, *, carried_cost_usd: float, slice_timeout_seconds: int) -> None:
    selected = select_slice_timeout_seconds(profile, carried_cost_usd=carried_cost_usd)
    if slice_timeout_seconds != selected:
        msg = "continuation slice timeout is not the largest approved denomination that fits"
        raise RuntimeError(msg)
    if carried_cost_usd + slice_timeout_seconds * topology_all_in_rate(EXECUTION_TOPOLOGIES[profile]) > SEGMENTS[profile]:
        msg = "continuation hard timeout can exceed the wave ceiling"
        raise RuntimeError(msg)
    if slice_timeout_seconds <= SHUTDOWN_RESERVE_SECONDS:
        msg = "continuation slice cannot retain the 600-second persistence reserve"
        raise RuntimeError(msg)


CADENCE_INITIAL_STEP = 100
CADENCE_INTERVAL_STEPS = 50
EARLY_STOP_F1_DEGRADATION = 0.005
EARLY_STOP_CONSECUTIVE_DEGRADATIONS = 2
_CADENCE_RESULT_KEYS = {
    "checkpoint_digest",
    "trajectory_digest",
    "evaluation_contract_digest",
    "comparison_contract_digest",
    "optimizer_step",
    "fixed_nine_exact_typed",
}
_CADENCE_METRIC_KEYS = {"precision", "recall", "f1"}


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _preregistered_early_stop_decision(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Mirror the benchmark's pure cadence decision without importing its runner.

    Returns:
        ``{"status": "early_stop", ...}`` naming the best checkpoint, the one that triggered
        the stop and the degradation count, once two consecutive results degrade on F1 AND
        do not improve recall; otherwise ``{"status": "continue", ...}`` with the best
        checkpoint so far. Best tracks highest F1, breaking ties on recall.

    Raises:
        RuntimeError: If the history is empty, which is a missing measurement rather than a
            decision to continue.

    """
    if not history:
        msg = "early-stop receipt history is empty"
        raise RuntimeError(msg)

    best: dict[str, Any] | None = None
    consecutive_degradations = 0
    for result in history:
        metrics = result["fixed_nine_exact_typed"]
        if not isinstance(metrics, Mapping):
            msg = "early-stop receipt metrics are malformed"
            raise RuntimeError(msg)
        normalized = {
            "checkpoint_digest": result["checkpoint_digest"],
            "optimizer_step": result["optimizer_step"],
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }
        if best is None:
            best = normalized
            continue

        degraded = normalized["f1"] <= best["f1"] - EARLY_STOP_F1_DEGRADATION and normalized["recall"] <= best["recall"]
        consecutive_degradations = consecutive_degradations + 1 if degraded else 0
        if normalized["f1"] > best["f1"] or (normalized["f1"] == best["f1"] and normalized["recall"] > best["recall"]):
            best = normalized
        if consecutive_degradations >= EARLY_STOP_CONSECUTIVE_DEGRADATIONS:
            return {
                "status": "early_stop",
                "reason": "two_consecutive_f1_and_recall_degradations",
                "best_checkpoint": best,
                "trigger_checkpoint": normalized,
                "consecutive_degradations": consecutive_degradations,
            }

    if best is None:
        msg = "early-stop receipt history did not produce a best checkpoint"
        raise RuntimeError(msg)
    return {
        "status": "continue",
        "reason": "patience_not_exhausted",
        "best_checkpoint": best,
        "consecutive_degradations": consecutive_degradations,
    }


def _validate_cadence_history(
    history: object,
    *,
    trajectory_digest: str,
    comparison_contract_digest: str,
    source_step: int,
) -> list[Mapping[str, Any]]:
    if not isinstance(history, list):
        msg = "early-stop receipt history must be a JSON array"
        raise RuntimeError(msg)
    required_steps = list(
        range(
            CADENCE_INITIAL_STEP,
            source_step // CADENCE_INTERVAL_STEPS * CADENCE_INTERVAL_STEPS + 1,
            CADENCE_INTERVAL_STEPS,
        ),
    )
    if not required_steps:
        msg = "post-step60 continuation requires a cadence checkpoint"
        raise RuntimeError(msg)
    if len(history) != len(required_steps):
        msg = "early-stop receipt lacks complete cadence coverage"
        raise RuntimeError(msg)

    seen_checkpoints: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for result, required_step in zip(history, required_steps, strict=True):
        if not is_str_mapping(result) or set(result) != _CADENCE_RESULT_KEYS:
            msg = "early-stop receipt cadence result schema is invalid"
            raise RuntimeError(msg)
        checkpoint_digest = result["checkpoint_digest"]
        evaluation_contract_digest = result["evaluation_contract_digest"]
        optimizer_step = result["optimizer_step"]
        metrics = result["fixed_nine_exact_typed"]
        # reason: validate cadence keeps trajectory/comparison in one gate; helper predicates would scatter the rule.
        if (
            not _is_sha256(checkpoint_digest)  # ruff: ignore[too-many-boolean-expressions]
            or checkpoint_digest in seen_checkpoints
            or not _is_sha256(result["trajectory_digest"])
            or not _is_sha256(evaluation_contract_digest)
            or not _is_sha256(result["comparison_contract_digest"])
            or result["trajectory_digest"] != trajectory_digest
            or result["comparison_contract_digest"] != comparison_contract_digest
            or type(optimizer_step) is not int
            or optimizer_step != required_step
            or not is_str_mapping(metrics)
            or set(metrics) != _CADENCE_METRIC_KEYS
        ):
            msg = "early-stop receipt cadence identity or sequence is invalid"
            raise RuntimeError(msg)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in (metrics["precision"], metrics["recall"], metrics["f1"])
        ):
            msg = "early-stop receipt cadence metrics are invalid"
            raise RuntimeError(msg)
        seen_checkpoints.add(checkpoint_digest)
        validated.append(result)
    return validated


# reason: validate early coordinates validate with early stop; extra seams would fragment diagnostics.
def validate_early_stop_receipt(  # ruff: ignore[complex-structure]
    contract: Mapping[str, Any],
    *,
    source_step: int,
    receipt: Mapping[str, Any] | None,
) -> None:
    if source_step == SOURCE_STEP:
        if receipt is not None:
            msg = "step-60 source must not carry an early-stop receipt"
            raise RuntimeError(msg)
        return
    if not isinstance(receipt, Mapping):
        msg = "post-step60 continuation requires an early-stop continue receipt"
        raise RuntimeError(msg)
    required = {
        "schema_version",
        "trajectory_digest",
        "comparison_contract_digest",
        "history",
        "history_digest",
        "decision",
        "status",
        "last_evaluated_optimizer_step",
        "decision_digest",
    }
    if set(receipt) != required or receipt.get("schema_version") != 1:
        msg = "early-stop receipt schema is invalid"
        raise RuntimeError(msg)
    trajectory_digest = receipt.get("trajectory_digest")
    comparison_contract_digest = receipt.get("comparison_contract_digest")
    if (
        not isinstance(trajectory_digest, str)
        or not _is_sha256(trajectory_digest)
        or trajectory_digest != contract.get("trajectory_digest")
        or not isinstance(comparison_contract_digest, str)
        or not _is_sha256(comparison_contract_digest)
    ):
        msg = "early-stop receipt identity drifted"
        raise RuntimeError(msg)

    history = _validate_cadence_history(
        receipt.get("history"),
        trajectory_digest=trajectory_digest,
        comparison_contract_digest=comparison_contract_digest,
        source_step=source_step,
    )
    if receipt.get("last_evaluated_optimizer_step") != history[-1]["optimizer_step"]:
        msg = "early-stop receipt lacks complete cadence coverage"
        raise RuntimeError(msg)
    decision = receipt.get("decision")
    expected_decision = _preregistered_early_stop_decision(history)
    if not isinstance(decision, Mapping) or dict(decision) != expected_decision:
        msg = "early-stop receipt decision is invalid"
        raise RuntimeError(msg)
    if receipt.get("status") != expected_decision["status"] or receipt.get("status") != "continue":
        msg = "early-stop receipt does not authorize continuation"
        raise RuntimeError(msg)

    if receipt.get("history_digest") != _canonical_sha256(history):
        msg = "early-stop receipt history digest is invalid"
        raise RuntimeError(msg)
    digest_body = {key: value for key, value in receipt.items() if key != "decision_digest"}
    if receipt.get("decision_digest") != _canonical_sha256(digest_body):
        msg = "early-stop receipt decision digest is invalid"
        raise RuntimeError(msg)


def rank_windows(
    *,
    cursor: int,
    global_batch_size: int | None = None,
    world_size: int = WORLD_SIZE,
) -> list[tuple[int, int]]:
    if global_batch_size is None:
        global_batch_size = FINAL_GLOBAL_BATCH_SIZE if cursor == FINAL_GLOBAL_START else GLOBAL_BATCH_SIZE
    if (
        cursor < 0
        or global_batch_size <= 0
        or global_batch_size > GLOBAL_BATCH_SIZE
        or world_size not in SUPPORTED_WORLD_SIZES
        or global_batch_size % world_size
    ):
        msg = "continuation rank windows require a positive runtime-wide global batch no larger than 128"
        raise ValueError(msg)
    if cursor != EPOCH_CURSOR and cursor % GLOBAL_BATCH_SIZE:
        msg = "nonterminal continuation cursors must be global-batch aligned"
        raise ValueError(msg)
    local_batch_size = global_batch_size // world_size
    return [(cursor + rank * local_batch_size, cursor + (rank + 1) * local_batch_size) for rank in range(world_size)]


def commit_update(
    *,
    step: int,
    cursor: int,
    rank_consensus: bool,
    consumed_units: int = GLOBAL_BATCH_SIZE,
) -> tuple[int, int]:
    if step < SOURCE_STEP or cursor < SOURCE_CURSOR or not rank_consensus:
        msg = "continuation update requires source-aligned all-rank consensus"
        raise RuntimeError(msg)
    if cursor >= EPOCH_CURSOR:
        msg = "epoch-terminal continuation cannot consume more packed units"
        raise RuntimeError(msg)
    if consumed_units not in {GLOBAL_BATCH_SIZE, FINAL_GLOBAL_BATCH_SIZE} or cursor + consumed_units > EPOCH_CURSOR:
        msg = "continuation update has an invalid packed-unit coverage window"
        raise RuntimeError(msg)
    rank_windows(cursor=cursor, global_batch_size=consumed_units)
    return step + 1, cursor + consumed_units


def derived_rank_seed(parent_checkpoint_digest: str, rank: int, *, world_size: int = WORLD_SIZE) -> int:
    if not parent_checkpoint_digest or world_size not in SUPPORTED_WORLD_SIZES or rank not in range(world_size):
        msg = "rank seed requires a parent digest and a valid runtime rank"
        raise ValueError(msg)
    return int.from_bytes(
        sha256(f"{GATE_SEED}:{parent_checkpoint_digest}:world{world_size}:rank{rank}".encode()).digest()[:8],
        "big",
    )


def rng_transition_kind(source_world_size: int, execution_world_size: int) -> str:
    if source_world_size not in {1, 2, 4} or execution_world_size not in SUPPORTED_WORLD_SIZES:
        msg = "continuation checkpoint world size is unsupported"
        raise RuntimeError(msg)
    transitions = {
        (1, 2): "single_rank_to_two_rank_v1",
        (1, 4): RNG_TRANSITION,
        (2, 2): "two_rank_to_two_rank_v1",
        (2, 4): "two_rank_to_four_rank_v1",
        (4, 4): "four_rank_to_four_rank_v1",
    }
    try:
        return transitions[source_world_size, execution_world_size]
    except KeyError as error:
        msg = "continuation cannot reduce execution world size"
        raise RuntimeError(msg) from error


# reason: checkpoint exposes contract/wave as its public contract; bundling would break callers.
def checkpoint_metadata_v2(  # ruff: ignore[too-many-arguments]
    contract: Mapping[str, Any],
    *,
    parent_checkpoint_digest: str | None,
    step: int,
    cursor: int,
    lifecycle_state: Literal["resume_source", "update_committed", "terminal"],
    rank_rng_states: Sequence[Mapping[str, Any]],
    execution_world_size: int = WORLD_SIZE,
    source_world_size: int | None = None,
    wave_profile: str | None = None,
    wave_cumulative_all_in_cost_usd: float = 0.0,
) -> dict[str, Any]:
    if execution_world_size not in SUPPORTED_WORLD_SIZES:
        msg = "continuation checkpoint world size is unsupported"
        raise RuntimeError(msg)
    source_world_size = source_world_size if source_world_size is not None else execution_world_size
    transition_kind = rng_transition_kind(source_world_size, execution_world_size)
    partial_terminal = lifecycle_state == "terminal" and step == EPOCH_TERMINAL_STEP and cursor == EPOCH_CURSOR
    if step < SOURCE_STEP or cursor < SOURCE_CURSOR or (cursor != step * GLOBAL_BATCH_SIZE and not partial_terminal):
        msg = "checkpoint step and global packed cursor are inconsistent"
        raise RuntimeError(msg)
    if lifecycle_state == "resume_source":
        if step != SOURCE_STEP or cursor != SOURCE_CURSOR or not parent_checkpoint_digest or rank_rng_states:
            msg = "source transition requires the exact step-60 parent and no invented rank states"
            raise RuntimeError(msg)
        transition: dict[str, Any] = {
            "kind": transition_kind,
            "rank_seeds": [
                {
                    "rank": rank,
                    "seed": derived_rank_seed(parent_checkpoint_digest, rank, world_size=execution_world_size),
                }
                for rank in range(execution_world_size)
            ],
        }
    else:
        if len(rank_rng_states) != execution_world_size:
            msg = "continuation checkpoint requires one CPU/CUDA RNG state per runtime rank"
            raise RuntimeError(msg)
        transition = {
            "kind": transition_kind,
            "rank_mapping": list(range(execution_world_size)),
        }
    if wave_profile is not None and wave_profile not in WAVE_ORDER:
        msg = "continuation checkpoint wave profile is invalid"
        raise RuntimeError(msg)
    if wave_cumulative_all_in_cost_usd < 0:
        msg = "continuation checkpoint cumulative cost is invalid"
        raise RuntimeError(msg)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "trajectory_digest": contract["trajectory_digest"],
        "execution_contract_digest": contract["execution_contract_digest"],
        "parent_checkpoint_digest": parent_checkpoint_digest,
        "optimizer_step": step,
        "packed_cursor": cursor,
        "lifecycle_state": lifecycle_state,
        "world_size": execution_world_size,
        "rank_mapping": list(range(execution_world_size)),
        "rng_transition": transition,
        "rank_rng_states": [dict(state) for state in rank_rng_states],
        "epoch_complete": partial_terminal,
        "wave_profile": wave_profile,
        "wave_cumulative_all_in_cost_usd": wave_cumulative_all_in_cost_usd,
    }


def explicit_gradient_mean(gradients: Sequence[Tensor]) -> Tensor:
    if len(gradients) != WORLD_SIZE:
        msg = "explicit gradient mean requires exactly four rank gradients"
        raise RuntimeError(msg)
    result = gradients[0].clone()
    for gradient in gradients[1:]:
        result.add_(gradient)
    return result.div_(WORLD_SIZE)


def max_parameter_drift(parameters: Sequence[Any]) -> float:
    if not parameters:
        msg = "parameter drift requires rank values"
        raise ValueError(msg)
    reference = parameters[0]
    return (
        max(float((parameter.float() - reference.float()).abs().max().item()) for parameter in parameters[1:])
        if len(parameters) > 1
        else 0.0
    )


def require_zero_parameter_drift(parameters: Sequence[Any]) -> None:
    # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
    # reason: equality, so any tolerance here would let a real drift or contamination pass the gate.
    if max_parameter_drift(parameters) != 0.0:  # ruff: ignore[float-equality-comparison]
        msg = "four-rank trainable parameter drift is non-zero"
        raise RuntimeError(msg)


def build_manifest(files: Mapping[str, Path]) -> dict[str, Any]:
    entries = []
    for name, path in sorted(files.items()):
        if not path.is_file() or Path(name).is_absolute() or ".." in Path(name).parts:
            msg = "checkpoint manifest contains an invalid file path"
            raise RuntimeError(msg)
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    return {"schema_version": 1, "files": entries, "committed": True}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_digest(metadata: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    if manifest.get("committed") is not True:
        msg = "checkpoint digest requires a committed manifest"
        raise RuntimeError(msg)
    if "checkpoint_digest" in metadata:
        msg = "checkpoint metadata must not contain checkpoint_digest"
        raise RuntimeError(msg)
    manifest_body = {key: value for key, value in manifest.items() if key != "checkpoint_digest"}
    if set(manifest_body) != {"schema_version", "committed", "files"}:
        msg = "checkpoint digest requires the canonical manifest body"
        raise RuntimeError(msg)
    return _digest({"metadata": dict(metadata), "manifest": manifest_body})


def verify_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if manifest.get("committed") is not True or not isinstance(files, list) or not files:
        msg = "checkpoint manifest is absent or uncommitted"
        raise RuntimeError(msg)
    expected = set()
    for item in files:
        if not isinstance(item, Mapping):
            msg = "checkpoint manifest file entry is invalid"
            raise RuntimeError(msg)
        name, byte_count, digest = (
            item.get("path"),
            item.get("bytes"),
            item.get("sha256"),
        )
        if not isinstance(name, str) or not isinstance(byte_count, int) or not isinstance(digest, str):
            msg = "checkpoint manifest file entry is incomplete"
            raise RuntimeError(msg)
        path = root / name
        if not path.is_file() or path.stat().st_size != byte_count:
            msg = "checkpoint file size verification failed"
            raise RuntimeError(msg)
        if _file_sha256(path) != digest:
            msg = "checkpoint file SHA-256 verification failed"
            raise RuntimeError(msg)
        expected.add(name)
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    if actual != expected:
        msg = "checkpoint manifest has missing or extra files"
        raise RuntimeError(msg)


def verify_checkpoint_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_digest: str | None = None,
) -> str:
    """Verify bytes first, then bind the manifest to its canonical metadata.

    ``manifest.json`` intentionally does not hash itself.  The metadata hash is
    present in the manifest only, avoiding a cryptographic cycle where metadata
    would need to contain the hash of a manifest that hashes metadata.

    Returns:
        The digest recomputed from the on-disk metadata and manifest -- not the embedded
        value, which is only ever compared against this one.

    Raises:
        RuntimeError: From ``verify_manifest_files``, if the manifest is uncommitted or lists
            no files, if an entry lacks a path, byte count or digest, if a file is missing or
            its size or SHA-256 disagrees with the entry, or if the directory holds any file
            the manifest does not list -- an extra file is refused, not ignored. Then here, if
            the manifest carries no checkpoint digest, if ``metadata.json`` is unreadable or is
            not an object, or if the recomputed digest differs from the embedded one or from
            ``expected_digest`` when the caller supplied it.

    """
    verify_manifest_files(root, manifest)
    embedded_digest = manifest.get("checkpoint_digest")
    if not isinstance(embedded_digest, str) or not embedded_digest:
        msg = "checkpoint manifest lacks its checkpoint digest"
        raise RuntimeError(msg)
    metadata_path = root / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = "checkpoint metadata is unreadable"
        raise RuntimeError(msg) from error
    if not isinstance(metadata, Mapping):
        msg = "checkpoint metadata is not an object"
        raise RuntimeError(msg)
    recomputed = checkpoint_digest(metadata, manifest)
    if embedded_digest != recomputed or (expected_digest is not None and expected_digest != recomputed):
        msg = "checkpoint manifest digest verification failed"
        raise RuntimeError(msg)
    return recomputed


def budget_allows_next_step(
    *,
    elapsed_seconds: float,
    predicted_step_seconds: float,
    ceiling_usd: float,
    rate_usd_per_second: float,
    reserve_seconds: float,
) -> bool:
    if (
        min(
            elapsed_seconds,
            predicted_step_seconds,
            ceiling_usd,
            rate_usd_per_second,
            reserve_seconds,
        )
        < 0
    ):
        msg = "budget values must be non-negative"
        raise ValueError(msg)
    return (elapsed_seconds + predicted_step_seconds + reserve_seconds) * rate_usd_per_second <= ceiling_usd


def assign_evaluation_checkpoints(checkpoints: Sequence[str]) -> list[tuple[str, str]]:
    profiles = sorted(EVALUATION_CREDITS, key=lambda profile: (-EVALUATION_CREDITS[profile], profile))
    return [(checkpoint, profiles[index % len(profiles)]) for index, checkpoint in enumerate(checkpoints)]


def render_preflight_command(profile: str) -> str:
    render_segment_contract(profile)
    return f"MODAL_PROFILE={profile} uv run modal run --detach --timestamps -m \
meddies_pii.training.bioes.modal.pii350_ddp_continuation --preflight --profile {profile}"


def render_cpu_receipt_command(
    profile: str,
    source_root: str = "/artifacts/full-runs/pii350/2e679221fd7c4ec6925e13f27cf96769/checkpoints/step-00000060",
) -> str:
    render_segment_contract(profile)
    return f"MODAL_PROFILE={profile} uv run modal run --detach --timestamps -m \
meddies_pii.training.bioes.modal.pii350_ddp_continuation --cpu-receipt --profile {profile} \
--source-root '{source_root}'"


def render_upload_command(profile: str, checkpoint_root: str, remote_prefix: str, source_receipt_json: str) -> str:
    render_segment_contract(profile)
    if not source_receipt_json:
        msg = "upload render requires the prior CPU source receipt JSON"
        raise ValueError(msg)
    return f"MODAL_PROFILE={profile} uv run modal run --detach --timestamps -m \
meddies_pii.training.bioes.modal.pii350_ddp_continuation --upload --profile {profile} \
--checkpoint-root '{checkpoint_root}' --remote-prefix '{remote_prefix}' --source-receipt-json \
'{source_receipt_json}'"


def render_download_command(
    profile: str,
    remote_prefix: str,
    destination: str,
    revision: str,
    *,
    purpose: str = "training",
) -> str:
    render_segment_contract(profile)
    if purpose not in {"training", "evaluation"} or not revision:
        msg = "download render requires immutable revision and training/evaluation purpose"
        raise ValueError(msg)
    return f"MODAL_PROFILE={profile} uv run modal run --detach --timestamps -m \
meddies_pii.training.bioes.modal.pii350_ddp_continuation --download --profile {profile} \
--remote-prefix '{remote_prefix}' --destination '{destination}' --revision '{revision}' --purpose \
{purpose}"


def render_train_command(profile: str) -> str:
    contract = render_segment_contract(profile)
    manual = contract["manual_launch"]
    return f"MODAL_PROFILE={profile} uv run modal run --detach --timestamps -m \
meddies_pii.training.bioes.modal.pii350_ddp_continuation --train --profile {profile} --execute \
--confirmation {manual['confirmation']} --primary-action {manual['primary_action']} \
--execution-contract-digest {contract['execution_contract_digest']} --source-receipt-json \
'<SOURCE_RECEIPT_JSON>' --source-launch-digest '<LAUNCH_DIGEST>' --launch-attempt-nonce \
'<FRESH_32_HEX_NONCE>' --preflight-receipt-json '<PREFLIGHT_RECEIPT_JSON>' \
--interruption-cost-receipt-json '<REQUIRED_FOR_STEP200_PREEMPTION_RESUME>' \
--early-stop-receipt-json '<REQUIRED_AFTER_STEP60>'"


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--render",
        choices=("contract", "preflight", "receipt", "upload", "download", "train"),
        required=True,
    )
    args = parser.parse_args()
    value: Any = (
        render_segment_contract(args.profile)
        if args.render == "contract"
        else render_preflight_command(args.profile)
        if args.render == "preflight"
        else render_cpu_receipt_command(args.profile)
        if args.render == "receipt"
        else render_upload_command(
            args.profile,
            "<CHECKPOINT_ROOT>",
            "<REMOTE_PREFIX>",
            "<CPU_SOURCE_RECEIPT_JSON>",
        )
        if args.render == "upload"
        else render_download_command(args.profile, "<REMOTE_PREFIX>", "<DESTINATION>", "<IMMUTABLE_REVISION>")
        if args.render == "download"
        else render_train_command(args.profile)
    )
    print(json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)


if __name__ == "__main__":
    _main()
