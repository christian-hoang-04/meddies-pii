"""Pure durability and budget controls shared by isolated full-run runtimes."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .full_run import FULL_RUN_ALL_IN_RATE_USD_PER_SECOND
from .pins import (
    EVAL_DATASET_REVISION,
    PACKED_DATASET_REVISION,
    PACKED_MANIFEST_SHA256,
    PACKED_UNIT_COUNT,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

StopReason = Literal[
    "training",
    "deadline_reached",
    "epoch_complete",
    "optimizer_step_cap_reached",
]


@dataclass(frozen=True, slots=True)
class RuntimeResumeState:
    candidate_key: str
    config_digest: str
    packed_revision: str
    packed_manifest_sha256: str
    eval_revision: str
    packed_cursor: int
    optimizer_step: int


class RunWriter:
    def __init__(self, root: str | Path, *, commit: Callable[[], None] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.events_path = self.root / "events.jsonl"
        self._commit = commit
        self.runtime_context: dict[str, Any] = {}

    # reason: this is a failure-context bag, not a typed record: callers pass whatever the failing stage knows,
    # reason: it lands in the already-`dict[str, Any]` runtime_context, and it leaves through `json.dumps`. A
    # reason: union here would have to name every caller's locals and would still not constrain the payload.
    def set_runtime_context(self, **values: Any) -> None:  # ruff: ignore[any-type]
        """Retain failure context without committing the volume per event."""
        self.runtime_context.update(values)

    def append(self, event: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(event), sort_keys=True)
        print(encoded, flush=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def atomic_json(self, relative_path: str, value: Mapping[str, Any]) -> Path:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        if self._commit is not None:
            self._commit()
        return target


def validate_resume_state(state: RuntimeResumeState, *, candidate_key: str, config_digest: str) -> None:
    if state.candidate_key != candidate_key or state.config_digest != config_digest:
        msg = "resume candidate or config digest mismatch"
        raise RuntimeError(msg)
    if state.packed_revision != PACKED_DATASET_REVISION or state.packed_manifest_sha256 != PACKED_MANIFEST_SHA256:
        msg = "resume packed pin mismatch"
        raise RuntimeError(msg)
    if state.eval_revision != EVAL_DATASET_REVISION:
        msg = "resume eval pin mismatch"
        raise RuntimeError(msg)
    if not 0 <= state.packed_cursor <= PACKED_UNIT_COUNT or state.optimizer_step < 0:
        msg = "resume cursor or step is invalid"
        raise RuntimeError(msg)


# reason: next step reason exposes elapsed/optimizer as its public contract; bundling would break callers.
def next_step_reason(  # ruff: ignore[too-many-arguments]
    *,
    elapsed_seconds: float,
    predicted_next_step_seconds: float,
    training_deadline_seconds: float,
    epoch_complete: bool,
    optimizer_step: int = 0,
    optimizer_step_cap: int | None = None,
) -> StopReason:
    """Stop before the next update for epoch, explicit cap, or Modal deadline.

    Returns:
        The lifecycle reason, checked in a fixed precedence: a complete epoch, then the
        optimizer-step cap, then the deadline, and ``"training"`` when none applies. The
        deadline test adds the predicted next step to the elapsed time, so it stops before the
        step that would overrun rather than after -- a run that decided afterwards would have
        already spent the overrun.

    Raises:
        ValueError: If any of the three time inputs is negative, or if the optimizer step is
            negative or the cap is not positive. A zero cap is refused rather than treated as
            "stop immediately", because a caller meaning no cap passes ``None``.

    """
    if min(elapsed_seconds, predicted_next_step_seconds, training_deadline_seconds) < 0:
        msg = "deadline inputs must be non-negative"
        raise ValueError(msg)
    if optimizer_step < 0 or (optimizer_step_cap is not None and optimizer_step_cap <= 0):
        msg = "optimizer step cap inputs are invalid"
        raise ValueError(msg)
    if epoch_complete:
        return "epoch_complete"
    if optimizer_step_cap is not None and optimizer_step >= optimizer_step_cap:
        return "optimizer_step_cap_reached"
    if elapsed_seconds + predicted_next_step_seconds >= training_deadline_seconds:
        return "deadline_reached"
    return "training"


def cost_fields(elapsed_seconds: float, gpu_only_rate_usd_per_second: float) -> dict[str, float | None]:
    if elapsed_seconds < 0 or gpu_only_rate_usd_per_second < 0:
        msg = "cost inputs must be non-negative"
        raise ValueError(msg)
    return {
        "estimated_gpu_only_cost_usd": elapsed_seconds * gpu_only_rate_usd_per_second,
        "estimated_all_in_cost_usd": elapsed_seconds * FULL_RUN_ALL_IN_RATE_USD_PER_SECOND,
        "platform_actual_cost_usd": None,
    }


def checkpoint_metadata(state: RuntimeResumeState, *, reason: StopReason) -> dict[str, Any]:
    return {**asdict(state), "lifecycle_state": reason}
