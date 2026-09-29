"""Topology-flexible PII350 continuation lifecycle."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra with no macOS wheels, so it loads inside the call that needs it.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from .pii350_ddp_continuation import (
    EPOCH_CURSOR,
    GLOBAL_BATCH_SIZE,
    SOURCE_STEP,
    rank_windows,
    rng_transition_kind,
)

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class ResumeState:
    step: int
    cursor: int
    parent_checkpoint_digest: str
    source_world_size: int


@dataclass(frozen=True, slots=True)
class FourRankPlan:
    step: int
    """Compatibility name; world_size is runtime-selected."""
    cursor: int
    parent_checkpoint_digest: str
    rng_transition: str
    windows: tuple[tuple[int, int], ...]
    world_size: int


def topology_transition(source_world_size: int, execution_world_size: int) -> str:
    return rng_transition_kind(source_world_size, execution_world_size)


def begin_four_rank_resume(state: ResumeState, *, execution_world_size: int = 4) -> FourRankPlan:
    if state.cursor == EPOCH_CURSOR:
        msg = "epoch-terminal continuation refuses further training"
        raise RuntimeError(msg)
    if state.step < SOURCE_STEP or state.cursor != state.step * GLOBAL_BATCH_SIZE or not state.parent_checkpoint_digest:
        msg = "continuation resume state is not an exact checkpoint lineage"
        raise RuntimeError(msg)
    return FourRankPlan(
        state.step,
        state.cursor,
        state.parent_checkpoint_digest,
        topology_transition(state.source_world_size, execution_world_size),
        tuple(rank_windows(cursor=state.cursor, world_size=execution_world_size)),
        execution_world_size,
    )


def require_finite_tensors(tensors: Sequence[Tensor]) -> None:
    import torch

    if not tensors or any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        msg = "continuation has non-finite loss or gradient tensors"
        raise RuntimeError(msg)


def mean_four_rank_gradient_tensors(gradients: Sequence[Tensor], *, world_size: int = 4) -> Tensor:
    if len(gradients) != world_size:
        msg = "explicit DDP gradient mean requires one tensor per rank"
        raise RuntimeError(msg)
    require_finite_tensors(gradients)
    result = gradients[0].clone()
    for gradient in gradients[1:]:
        result.add_(gradient)
    return result.div_(world_size)


# reason: commit four rank exposes plan/consumed as its public contract; bundling would break callers.
def commit_four_rank_update(  # ruff: ignore[too-many-arguments]
    plan: FourRankPlan,
    *,
    finite_loss: bool,
    finite_gradients: bool,
    gradient_drift: float,
    parameter_drift: float,
    rank_successes: Sequence[bool],
    consumed_units: int = GLOBAL_BATCH_SIZE,
) -> ResumeState:
    if len(rank_successes) != plan.world_size or not all(rank_successes):
        msg = "continuation update lacks rank consensus"
        raise RuntimeError(msg)
    # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
    # reason: equality, so any tolerance here would let a real drift pass the gate.
    if (
        not finite_loss
        or not finite_gradients
        or gradient_drift != 0.0  # ruff: ignore[float-equality-comparison]
        or parameter_drift != 0.0  # ruff: ignore[float-equality-comparison]
    ):
        msg = "continuation update has invalid distributed tensors"
        raise RuntimeError(msg)
    if (
        consumed_units <= 0
        or consumed_units > GLOBAL_BATCH_SIZE
        or consumed_units % plan.world_size
        or plan.cursor + consumed_units > EPOCH_CURSOR
    ):
        msg = "continuation update has an invalid packed-unit count"
        raise RuntimeError(msg)
    return ResumeState(
        plan.step + 1,
        plan.cursor + consumed_units,
        plan.parent_checkpoint_digest,
        plan.world_size,
    )


def require_runtime_rank_rng_records(records: Sequence[Mapping[str, Any]], *, world_size: int) -> None:
    if len(records) != world_size or [record.get("rank") for record in records] != list(range(world_size)):
        msg = "continuation checkpoint requires exactly one ordered RNG state per runtime rank"
        raise RuntimeError(msg)
    if any(
        not isinstance(record.get("cpu_rng"), str) or not isinstance(record.get("cuda_rng"), str) for record in records
    ):
        msg = "continuation rank RNG state is incomplete"
        raise RuntimeError(msg)


def require_four_rank_rng_records(records: Sequence[Mapping[str, Any]]) -> None:
    require_runtime_rank_rng_records(records, world_size=4)


def projected_budget_stop(
    *,
    worker_elapsed_seconds: float,
    predicted_step_seconds: float,
    all_in_rate_usd_per_second: float,
    ceiling_usd: float,
    persistence_reserve_seconds: float,
) -> bool:
    values = (
        worker_elapsed_seconds,
        predicted_step_seconds,
        all_in_rate_usd_per_second,
        ceiling_usd,
        persistence_reserve_seconds,
    )
    if any(not isfinite(value) or value < 0 for value in values):
        msg = "continuation budget values must be finite and non-negative"
        raise ValueError(msg)
    return (
        worker_elapsed_seconds + predicted_step_seconds + persistence_reserve_seconds
    ) * all_in_rate_usd_per_second >= ceiling_usd


def load_pii350_composed_runtime(
    contract: Mapping[str, Any],
    *,
    checkpoint_attestation: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    from .full_run_runtime import _build_tagger

    baseline = contract.get("baseline_full_run_contract")
    if not isinstance(baseline, Mapping):
        msg = "continuation requires the exact baseline PII350 full-run contract"
        raise RuntimeError(msg)
    return _build_tagger(
        "pii350",
        contract=baseline,
        expected_encoder_checkpoint_attestation=checkpoint_attestation,
    )


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def require_a100_40gb(torch: Any, *, world_size: int) -> list[dict[str, Any]]:  # ruff: ignore[any-type]
    if world_size not in {2, 4} or not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        msg = "continuation requires exactly its contract-visible CUDA devices"
        raise RuntimeError(msg)
    minimum, maximum = 39 * 1024**3, 40 * 1024**3
    inventory = []
    for index in range(world_size):
        properties = torch.cuda.get_device_properties(index)
        if "A100" not in str(properties.name) or not minimum <= int(properties.total_memory) <= maximum:
            msg = "continuation requires A100-40GB devices"
            raise RuntimeError(msg)
        inventory.append({
            "index": index,
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
        })
    return inventory


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def require_four_a100_40gb(torch: Any) -> list[dict[str, Any]]:  # ruff: ignore[any-type]
    """Assert the four-rank A100-40GB topology the pii350 continuation is pinned to.

    Use this at the entry of a four-rank continuation rather than passing ``world_size=4`` to
    ``require_a100_40gb`` by hand; the pinned topology is named here once. Generic callers whose
    world size is a variable should keep using ``require_a100_40gb``.

    Returns:
        The per-device inventory the topology check collected.

    """
    return require_a100_40gb(torch, world_size=4)


def should_save_checkpoint(*, step: int, terminal: bool, rank_consensus: bool) -> bool:
    return rank_consensus and (terminal or step % 50 == 0)


def rank_rng_records(cpu_states: Sequence[bytes], cuda_states: Sequence[bytes]) -> list[dict[str, Any]]:
    if len(cpu_states) != len(cuda_states) or len(cpu_states) not in {2, 4}:
        msg = "continuation must gather CPU and CUDA RNG from every runtime rank"
        raise RuntimeError(msg)
    return [
        {
            "rank": rank,
            "cpu_rng": cpu_states[rank].hex(),
            "cuda_rng": cuda_states[rank].hex(),
        }
        for rank in range(len(cpu_states))
    ]
