"""Matrix, budget, and receipt logic for the PII350 checkpoint benchmark.

One verified checkpoint owns one complete 17-cell matrix in one Modal
workspace. This module holds the parts of that contract that need no Modal
handle and no live volume: the benchmark request shape, matrix coverage
requirements, cadence history and its decision receipts, cost prediction and
fill ordering, cross-workspace result transfer manifests, and cell scoring.

``scripts/ops/run_pii350_checkpoint_benchmark.py`` keeps the Modal app, the
volume-backed persistence, and the GPU functions that call into this module.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the sibling is imported inside the call so a test patching it on its own module is seen; binding it at
# reason: import time would bypass that seam.
# ruff: file-ignore[type-check-without-type-error]
# reason: all 8 sites sit in a function that already raises the same type more than once - checked by AST,
# reason: 2 to 9 same-type raises per function - so converting only the isinstance-guarded raise would split
# reason: one failure class across two exception types on the shape of the guard. Each states one contract:
# reason: a malformed ledger, receipt, cadence result, or cell list is refused, never a caller type error.
import json
import os
import shlex
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.eval_baseline.baseline.datasets import (
    EVAL_DATASETS,
    EVAL_EXPECTED_ROWS,
    EXTERNAL_DATASET_REVISION,
    V2_DATASET_REVISION,
)
from meddies_pii.evaluation.identity import (
    EvaluationContract,
    canonical_sha256,
    file_sha256,
    is_sha256,
)
from meddies_pii.file_locks import exclusive_file_lock
from meddies_pii.json_types import is_str_mapping
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import VerifiedCheckpoint

GIT_SHA_HEX_LENGTH = 40
MIN_PARTIAL_OPTIMIZER_STEP = 100

GPU = "A10G"

CPU = 8

MEMORY_MIB = 32_768

TIMEOUT_SECONDS = 6 * 60 * 60

PROGRESS_EVERY = 2_000

INTERNAL_CONTROL_PROGRESS_EVERY = 100

FULL_ROWS = 263_785

A10G_ALL_IN_RATE_USD_PER_SECOND = 0.00048184
"""Published A10G GPU + 8 CPU + 32 GiB memory rate: 0.000306 + (8 * 0.0000131) + (32 * 0.00000222) USD/s."""

BOOTSTRAP_DATASET = "v2-eval"

BOOTSTRAP_MAX_ROWS = 1_700

PREDICTED_CELL_MARGIN = 2.0

MINIMUM_PERSISTENCE_RESERVE_SECONDS = 300

CADENCE_INTERVAL_STEPS = 50

EARLY_STOP_F1_DEGRADATION = 0.005

EARLY_STOP_CONSECUTIVE_DEGRADATIONS = 2

FULL_GOLD_SPANS = 1_601_262

STEP60_BASELINE_PURPOSE = "step60_baseline"

STEP60_BASELINE_OPTIMIZER_STEP = 60

STEP60_PARTIAL_16OF17_PURPOSE = "step60_partial_16of17"

CHECKPOINT_PARTIAL_16OF17_PURPOSE = "checkpoint_partial_16of17"

FILL_MISSING_CELLS_PURPOSE = "fill_missing_cells"
"""A fill run completes an existing generation one named cell at a time.

The signed approval carries the cell list and the executor runs exactly that list, so an approval for two cells can never
be spent on the whole matrix.

"""

FILL_PRIOR_SECONDS_PER_ROW = 0.03
"""A fill list need not contain the bootstrap cell, so it has no cheap first cell to measure.

This pinned prior replaces the observed-rate projector for fill pre-dispatch checks: ~33 rows/s against ~60 rows/s observed
on a healthy A10G. It is a dispatch gate only. The mid-cell stop at batch boundaries (budget_ceiling_reached) remains the
real ceiling, so a wrong prior can waste at most one partial cell and can never overrun the approved ceiling.

"""

STEP60_PARTIAL_16OF17_EXCLUDED_CELLS = ("v2-eval-challenge",)

STEP60_PARTIAL_16OF17_DATASETS = tuple(
    dataset for dataset in EVAL_DATASETS if dataset not in STEP60_PARTIAL_16OF17_EXCLUDED_CELLS
)

STEP60_PARTIAL_16OF17_ROWS = 260_385

TERMINAL_STEP146_CHECKPOINT_DIGEST = "73a490a6a90da19d077951a673fa451e3350dc81f3e09ba594ee86e182985c76"

TERMINAL_STEP146_OPTIMIZER_STEP = 146

TERMINAL_STEP146_PACKED_CURSOR = 18_688

TERMINAL_STEP146_WORLD_SIZE = 2

SINGLE_MISSING_CELL_RESUME_PROGRESS_WINDOW_SECONDS = 60

V2_EVAL_REUSE_POLICY = "reuse_only_if_identity_verified_else_rerun"

V2_CHALLENGE_REUSE_POLICY = "require_existing_identity_verified_no_rerun"

STEP60_REUSED_CELLS = ("v2-eval", "v2-eval-challenge")

STEP60_REUSE_RECEIPT_SCHEMA_VERSION = 1

HF_CACHE_MOUNT = "/cache/huggingface"

PROFILE_ENVIRONMENT_KEY = "MODAL_PROFILE"

PROFILE_ALL_IN_CEILING_USD = {
    "diffusionllm": 4.0,
    "huyhoang0411ha": 8.5,
    "meddies-pii": 3.8,
    "huyhoang041100": 3.0,
    "meddies-run": 2.9,
}
"""These ceilings are an approved workspace ledger, not caller preferences.

A request has to match its selected workspace exactly, which prevents a CLI payload from quietly borrowing budget allocated
to another workspace.

"""

EVALUATION_PROFILES = tuple(PROFILE_ALL_IN_CEILING_USD)

LAUNCH_APPROVAL_SCHEMA_VERSION = 1

APPROVAL_LEDGER_SCHEMA_VERSION = 1

APPROVAL_PRIVATE_KEY_ENV = "PII350_BENCHMARK_APPROVAL_KEY_FILE"

MODELED_PRIOR_DIFFUSION_ATTEMPTS = (
    {
        "modal_app_id": "ap-n3kCBBhWGfbQgSmr1pmlcZ",
        "modeled_all_in_cost_usd": "0.00849154082499184",
        "kind": "observed_modeled_cost",
    },
    {
        "modal_app_id": "ap-Zs5taYwsHLrb1fDPwLw3Y7",
        "modeled_all_in_cost_usd": "1.0482259330791703",
        "kind": "observed_modeled_cost",
    },
    {
        "modal_app_id": "ap-N8DjFZFbLU8odxP9dma0TN",
        "modeled_all_in_cost_usd": "0.144552",
        "kind": "conservative_modeled_upper_bound",
    },
    {
        "modal_app_id": "cache-preflight-reserve",
        "modeled_all_in_cost_usd": "0.00357816",
        "kind": "reserve",
    },
    {
        "modal_app_id": "auxiliary-uncertainty-reserve",
        "modeled_all_in_cost_usd": "0.010000",
        "kind": "reserve",
    },
)

MODELED_PRIOR_DIFFUSION_TOTAL_USD = "1.214847633904162"

TRITON_COMPILER_APT_PACKAGE = "gcc"

TRITON_C_COMPILER = "/usr/bin/gcc"

RESULT_TRANSFER_REPO_ID = "Meddies/pii350-trajectories-private"
"""Cross-workspace result transfer.

Completed per-cell results move through the private trajectory repo so a second workspace can resume a generation instead
of paying A10G time again. The transfer never mints identity: the paid resume path still validates generation identity
before it reuses any imported cell.

"""

RESULT_TRANSFER_ROOT = "results-transfer"

RESULT_TRANSFER_FILES_PREFIX = "files"

RESULT_TRANSFER_MANIFEST_FILENAME = "manifest.json"

RESULT_TRANSFER_SCHEMA_VERSION = 1

CELL_SCORE_SUMMARY_SCHEMA_VERSION = 1

RESULT_SHARD = "full"

RESULT_CELL_FILENAMES = (
    f"{RESULT_SHARD}.jsonl",
    f"{RESULT_SHARD}.meta.json",
    f"{RESULT_SHARD}.done",
)
"""The completion marker is written last at the destination.

So an interrupted import can never leave a cell that reads as done without its verified bytes.

"""

RESULT_TRANSFER_RECEIPT_KEYS = frozenset({
    "schema_version",
    "checkpoint_digest",
    "trajectory_digest",
    "evaluation_generation_digest",
    "model",
    "source_profile",
    "cells",
    "files",
    "total_bytes",
    "manifest_digest",
    "repo_id",
    "repo_prefix",
    "hf_commit_sha",
    "receipt_digest",
})


def _decimal_usd(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        msg = f"{field} must be a decimal USD value"
        raise ValueError(msg)
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        msg = f"{field} must be a decimal USD value"
        raise ValueError(msg) from error
    if not decimal.is_finite() or decimal < 0:
        msg = f"{field} must be a nonnegative finite USD value"
        raise ValueError(msg)
    return decimal


def _approval_ledger_template() -> dict[str, Any]:
    return {
        "schema_version": APPROVAL_LEDGER_SCHEMA_VERSION,
        "prior_modeled_attempts_by_profile": {
            "diffusionllm": [dict(entry) for entry in MODELED_PRIOR_DIFFUSION_ATTEMPTS],
            **{profile: [] for profile in EVALUATION_PROFILES if profile != "diffusionllm"},
        },
        "reservations": [],
    }


def _validate_prior_attempts(value: object, *, profile: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        msg = "approval ledger prior modeled spend is invalid"
        raise RuntimeError(msg)
    normalized: list[dict[str, str]] = []
    ids: set[str] = set()
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {
            "modal_app_id",
            "modeled_all_in_cost_usd",
            "kind",
        }:
            msg = "approval ledger prior modeled spend is invalid"
            raise RuntimeError(msg)
        app_id = entry.get("modal_app_id")
        kind = entry.get("kind")
        if not isinstance(app_id, str) or not app_id or not isinstance(kind, str) or not kind:
            msg = "approval ledger prior modeled spend is invalid"
            raise RuntimeError(msg)
        if app_id in ids:
            msg = "approval ledger repeats a prior app inventory entry"
            raise RuntimeError(msg)
        ids.add(app_id)
        normalized.append({
            "modal_app_id": app_id,
            "modeled_all_in_cost_usd": str(
                _decimal_usd(
                    entry.get("modeled_all_in_cost_usd"),
                    field="prior modeled spend",
                ),
            ),
            "kind": kind,
        })
    if (
        profile == "diffusionllm"
        and tuple(normalized[: len(MODELED_PRIOR_DIFFUSION_ATTEMPTS)]) != MODELED_PRIOR_DIFFUSION_ATTEMPTS
    ):
        msg = "approval ledger diffusion modeled-spend inventory drifted"
        raise RuntimeError(msg)
    return normalized


def _validate_approval_ledger(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "prior_modeled_attempts_by_profile",
        "reservations",
    }:
        msg = "approval ledger is malformed"
        raise RuntimeError(msg)
    if value.get("schema_version") != APPROVAL_LEDGER_SCHEMA_VERSION:
        msg = "approval ledger schema is invalid"
        raise RuntimeError(msg)
    prior_by_profile = value.get("prior_modeled_attempts_by_profile")
    if not is_str_mapping(prior_by_profile) or set(prior_by_profile) != set(EVALUATION_PROFILES):
        msg = "approval ledger profile inventory is invalid"
        raise RuntimeError(msg)
    normalized_prior = {
        profile: _validate_prior_attempts(prior_by_profile[profile], profile=profile) for profile in EVALUATION_PROFILES
    }
    reservations = value.get("reservations")
    if not isinstance(reservations, list):
        msg = "approval ledger reservations are invalid"
        raise RuntimeError(msg)
    normalized_reservations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_nonces: set[str] = set()
    seen_external_ids: set[str] = set()
    for reservation in reservations:
        if not isinstance(reservation, Mapping) or set(reservation) != {
            "approval_digest",
            "profile",
            "launch_nonce",
            "purpose",
            "status",
            "new_attempt_ceiling_usd",
            "external_ledger_approval_id",
        }:
            msg = "approval ledger reservation is invalid"
            raise RuntimeError(msg)
        approval_digest = reservation.get("approval_digest")
        profile = reservation.get("profile")
        nonce = reservation.get("launch_nonce")
        purpose = reservation.get("purpose")
        status = reservation.get("status")
        external_id = reservation.get("external_ledger_approval_id")
        # reason: validate approval keeps settled/approval in one gate; helper predicates would scatter the rule.
        if (
            not is_sha256(approval_digest)  # ruff: ignore[too-many-boolean-expressions]
            or approval_digest in seen_ids
            or profile not in EVALUATION_PROFILES
            or not is_sha256(nonce)
            or nonce in seen_nonces
            or not isinstance(purpose, str)
            or not isinstance(external_id, str)
            or not external_id
            or external_id in seen_external_ids
            or status not in {"active", "settled", "released"}
        ):
            msg = "approval ledger reservation is invalid"
            raise RuntimeError(msg)
        seen_ids.add(approval_digest)
        seen_nonces.add(nonce)
        seen_external_ids.add(external_id)
        normalized_reservations.append({
            "approval_digest": approval_digest,
            "profile": profile,
            "launch_nonce": nonce,
            "purpose": purpose,
            "status": status,
            "new_attempt_ceiling_usd": str(
                _decimal_usd(
                    reservation.get("new_attempt_ceiling_usd"),
                    field="reservation ceiling",
                ),
            ),
            "external_ledger_approval_id": external_id,
        })
    return {
        "schema_version": APPROVAL_LEDGER_SCHEMA_VERSION,
        "prior_modeled_attempts_by_profile": normalized_prior,
        "reservations": normalized_reservations,
    }


def _read_approval_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _approval_ledger_template()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = "approval ledger is malformed"
        raise RuntimeError(msg) from error
    return _validate_approval_ledger(value)


def _atomic_local_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    try:
        Path(temp_path).replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def _with_approval_ledger_lock[Result](ledger_path: str | Path, operation: Callable[[Path], Result]) -> Result:
    """Keep a stable inode locked while the ledger itself is atomically replaced.

    Returns:
        Whatever ``operation`` returns when handed the ledger path, evaluated while the lock is held.

    Raises:
        RuntimeError: If no ledger path was given, or another process already holds the lock, which
            is how concurrent issuance is refused rather than serialised.

    """
    if not ledger_path:
        msg = "approval ledger path is required"
        raise RuntimeError(msg)
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        with exclusive_file_lock(lock_path, blocking=False):
            return operation(path)
    except BlockingIOError as error:
        msg = "approval ledger is busy; concurrent issuance is refused"
        raise RuntimeError(msg) from error


def inspect_approval_ledger(ledger_path: str | Path) -> dict[str, Any]:
    """Read a validated local issuer ledger without touching Modal.

    Returns:
        The ledger in its normalized form, with every prior modeled attempt and reservation checked.

    """
    return _with_approval_ledger_lock(ledger_path, _read_approval_ledger)


def _require_active_ledger_reservation(ledger: Mapping[str, Any], approval: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a paid dispatch to one still-live external issuer reservation.

    Returns:
        A copy of the matching reservation, so a later mutation of the ledger cannot reach the caller.

    Raises:
        RuntimeError: If the ledger holds no reservation for this approval digest, or the one it holds
            is not active or disagrees with the approval on profile, purpose, nonce, or external ID.

    """
    for reservation in ledger["reservations"]:
        if reservation["approval_digest"] != approval["approval_digest"]:
            continue
        if (
            reservation["status"] != "active"
            or reservation["profile"] != approval["profile"]
            or reservation["purpose"] != approval["purpose"]
            or reservation["launch_nonce"] != approval["launch_nonce"]
            or reservation["external_ledger_approval_id"] != approval["external_ledger_approval_id"]
        ):
            msg = "approval ledger reservation does not match an active approval"
            raise RuntimeError(msg)
        return dict(reservation)
    msg = "approval ledger has no reservation for this approval"
    raise RuntimeError(msg)


# reason: this cross-process control signal is re-exported by the runner and recorded by its stable domain name.
class DuplicateLaunchNonce(RuntimeError):  # ruff: ignore[error-suffix-on-exception-name]
    """A cross-container replay lost the atomic Modal Dict admission race."""


# reason: budget stop is a persisted terminal outcome, not an unexpected error, and is part of the runner API.
class BudgetStop(RuntimeError):  # ruff: ignore[error-suffix-on-exception-name]
    """Fail closed after persisting a terminal budget-stop record."""


# reason: budget refusal is a planned admission outcome re-exported by the runner under this domain name.
class BudgetRefusal(RuntimeError):  # ruff: ignore[error-suffix-on-exception-name]
    """The next cell cannot begin within the requested all-in budget."""


# reason: Cadence metrics, budget state, and trajectory identity produce one early-stop decision.
def cadence_early_stop_decision(  # ruff: ignore[complex-structure,too-many-branches,too-many-locals]
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered control-only patience rule to cadence metrics.

    Returns:
        Either an ``early_stop`` decision naming the best and triggering checkpoints, or a ``continue``
        decision, each carrying the run of consecutive degradations that produced it.

    Raises:
        ValueError: If the history is empty, a result has the wrong field set, an identity digest is
            not a SHA-256, a checkpoint repeats, a step is off the 50-step cadence or out of order, the
            history mixes trajectories, or a metric is not a finite probability.
        RuntimeError: If a non-empty validated history fails to produce a best checkpoint.

    """
    if not history:
        msg = "cadence history must contain at least one checkpoint"
        raise ValueError(msg)

    required_result_keys = {
        "checkpoint_digest",
        "trajectory_digest",
        "evaluation_contract_digest",
        "comparison_contract_digest",
        "optimizer_step",
        "fixed_nine_exact_typed",
    }
    identity: tuple[str, str] | None = None
    prior_step: int | None = None
    seen_checkpoints: set[str] = set()
    best: dict[str, Any] | None = None
    consecutive_degradations = 0

    for result in history:
        if set(result) != required_result_keys:
            msg = "cadence result has missing or unexpected fields"
            raise ValueError(msg)
        checkpoint_digest = result["checkpoint_digest"]
        trajectory_digest = result["trajectory_digest"]
        evaluation_contract_digest = result["evaluation_contract_digest"]
        comparison_contract_digest = result["comparison_contract_digest"]
        step = result["optimizer_step"]
        metrics = result["fixed_nine_exact_typed"]
        # reason: cadence early keeps precision/recall in one gate; helper predicates would scatter the rule.
        if (
            not is_sha256(checkpoint_digest)  # ruff: ignore[too-many-boolean-expressions]
            or checkpoint_digest in seen_checkpoints
            or not is_sha256(trajectory_digest)
            or not is_sha256(evaluation_contract_digest)
            or not is_sha256(comparison_contract_digest)
            or type(step) is not int
            or step < CADENCE_INTERVAL_STEPS
            or step % CADENCE_INTERVAL_STEPS != 0
            or not isinstance(metrics, Mapping)
            or set(metrics) != {"precision", "recall", "f1"}
        ):
            msg = "cadence result identity or metric shape is invalid"
            raise ValueError(msg)
        metric_values = {name: metrics[name] for name in ("precision", "recall", "f1")}
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in metric_values.values()
        ):
            msg = "cadence result metrics must be finite probabilities"
            raise ValueError(msg)
        if identity is None:
            identity = (trajectory_digest, comparison_contract_digest)
        elif identity != (trajectory_digest, comparison_contract_digest):
            msg = "cadence history mixes trajectory or comparison identities"
            raise ValueError(msg)
        if prior_step is not None and step != prior_step + CADENCE_INTERVAL_STEPS:
            msg = "cadence history is missing or out of order"
            raise ValueError(msg)

        normalized: dict[str, Any] = {
            "checkpoint_digest": checkpoint_digest,
            "optimizer_step": step,
            "precision": float(metric_values["precision"]),
            "recall": float(metric_values["recall"]),
            "f1": float(metric_values["f1"]),
        }
        if best is None:
            best = normalized
        else:
            degraded = (
                normalized["f1"] <= best["f1"] - EARLY_STOP_F1_DEGRADATION and normalized["recall"] <= best["recall"]
            )
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
        seen_checkpoints.add(checkpoint_digest)
        prior_step = step

    if best is None:
        msg = "cadence history did not produce a best checkpoint"
        raise RuntimeError(msg)
    return {
        "status": "continue",
        "reason": "patience_not_exhausted",
        "best_checkpoint": best,
        "consecutive_degradations": consecutive_degradations,
    }


def comparison_contract_digest(evaluation: EvaluationContract, *, fixture_digest: str) -> str:
    """Identify invariant control inputs while excluding checkpoint-specific model bytes.

    Returns:
        The SHA-256 over dataset, batching, and evaluation identity with ``model`` dropped, so the
        digest stays equal across every checkpoint in one trajectory.

    Raises:
        ValueError: If the internal control fixture digest is not a SHA-256.

    """
    if not is_sha256(fixture_digest):
        msg = "internal control fixture digest is invalid"
        raise ValueError(msg)
    evaluation_payload = dict(evaluation.to_payload())
    evaluation_payload.pop("model", None)
    return canonical_sha256({
        "schema_version": 1,
        "purpose": "internal_control",
        "dataset": {
            "name": "v2-eval",
            "revision": V2_DATASET_REVISION,
            "rows": EVAL_EXPECTED_ROWS["v2-eval"],
            "fixture_digest": fixture_digest,
        },
        "batching": {
            "max_batch_size": 8,
            "max_batch_tokens": 32_768,
            "max_sequence_length": 8192,
        },
        "evaluation": evaluation_payload,
    })


CADENCE_RECEIPT_SCHEMA_VERSION = 1

CADENCE_INITIAL_STEP = 100


def _cadence_history(value: object, *, role: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        msg = f"{role} must be a JSON array"
        raise ValueError(msg)
    history: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            msg = f"{role} contains a non-object result"
            raise ValueError(msg)
        # reason: `isinstance(entry, Mapping)` narrows an `object` to an unknown key type, which `dict()` has no
        # reason: overload for. The repo's `is_str_mapping` would narrow it but adds a string-key check, and two
        # reason: callers in run_pii350_checkpoint_benchmark.py do not run the key-set check that would subsume it.
        history.append(dict(entry))
    return history


def _require_complete_cadence_history(
    history: Sequence[Mapping[str, Any]],
    *,
    trajectory_digest: str,
    stable_comparison_digest: str,
    expected_last_step: int,
) -> None:
    if expected_last_step < CADENCE_INITIAL_STEP or (expected_last_step % CADENCE_INTERVAL_STEPS != 0):
        msg = "current cadence step is invalid"
        raise ValueError(msg)
    if not history:
        msg = "cadence history is required after step 100"
        raise ValueError(msg)
    cadence_early_stop_decision(history)
    steps = [entry.get("optimizer_step") for entry in history]
    expected_steps = list(range(CADENCE_INITIAL_STEP, expected_last_step + 1, CADENCE_INTERVAL_STEPS))
    if steps != expected_steps:
        msg = "cadence history has a rollback, omission, or reordering"
        raise ValueError(msg)
    if any(
        entry.get("trajectory_digest") != trajectory_digest
        or entry.get("comparison_contract_digest") != stable_comparison_digest
        for entry in history
    ):
        msg = "cadence history does not match the current trajectory"
        raise ValueError(msg)


def build_cadence_decision_receipt(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the cumulative cadence history to its pure early-stop decision.

    Returns:
        The receipt body plus a ``decision_digest`` over it, so the history and the decision it
        produced cannot be separated afterwards.

    """
    normalized_history = _cadence_history(list(history), role="cadence history")
    decision = cadence_early_stop_decision(normalized_history)
    last = normalized_history[-1]
    body = {
        "schema_version": CADENCE_RECEIPT_SCHEMA_VERSION,
        "trajectory_digest": last["trajectory_digest"],
        "comparison_contract_digest": last["comparison_contract_digest"],
        "history": normalized_history,
        "history_digest": canonical_sha256(normalized_history),
        "decision": decision,
        "status": decision["status"],
        "last_evaluated_optimizer_step": last["optimizer_step"],
    }
    return {**body, "decision_digest": canonical_sha256(body)}


def verify_cadence_decision_receipt(receipt: object) -> dict[str, Any]:
    """Reject a receipt whose history, decision, or digest has changed.

    Returns:
        The receipt rebuilt from its own history, which equals the argument or the call has raised.

    Raises:
        ValueError: If the receipt is not an object, has the wrong field set or schema, carries an
            empty or non-SHA-256 history or digest, has a non-integer last step, fails the
            completeness check, or does not equal the receipt its own history rebuilds.

    """
    if not is_str_mapping(receipt):
        msg = "cadence decision receipt must be an object"
        raise ValueError(msg)
    required_keys = {
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
    if set(receipt) != required_keys:
        msg = "cadence decision receipt has missing or unexpected fields"
        raise ValueError(msg)
    if receipt.get("schema_version") != CADENCE_RECEIPT_SCHEMA_VERSION:
        msg = "cadence decision receipt schema is invalid"
        raise ValueError(msg)
    history = _cadence_history(receipt.get("history"), role="receipt history")
    if not history:
        msg = "cadence decision receipt history is empty"
        raise ValueError(msg)
    if (
        not is_sha256(receipt.get("trajectory_digest"))
        or not is_sha256(receipt.get("comparison_contract_digest"))
        or receipt.get("history_digest") != canonical_sha256(history)
        or not is_sha256(receipt.get("decision_digest"))
    ):
        msg = "cadence decision receipt digest is invalid"
        raise ValueError(msg)
    last_step = receipt.get("last_evaluated_optimizer_step")
    if type(last_step) is not int:
        msg = "cadence decision receipt last step is invalid"
        raise ValueError(msg)
    _require_complete_cadence_history(
        history,
        trajectory_digest=str(receipt["trajectory_digest"]),
        stable_comparison_digest=str(receipt["comparison_contract_digest"]),
        expected_last_step=last_step,
    )
    expected = build_cadence_decision_receipt(history)
    if dict(receipt) != expected:
        msg = "cadence decision receipt decision is invalid"
        raise ValueError(msg)
    return expected


def _read_cadence_history_at_paths(history_path: Path, receipt_path: Path, *, role: str) -> list[dict[str, Any]] | None:
    if not history_path.exists() and not receipt_path.exists():
        return None
    if not history_path.is_file() or not receipt_path.is_file():
        msg = f"{role} is incomplete"
        raise RuntimeError(msg)
    try:
        history_value = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"{role} is invalid"
        raise RuntimeError(msg) from error
    if not isinstance(history_value, Mapping) or set(history_value) != {"history"}:
        msg = f"{role} is invalid"
        raise RuntimeError(msg)
    history = _cadence_history(history_value.get("history"), role=role)
    receipt = _read_receipt(receipt_path, role=f"{role} decision receipt")
    verified = verify_cadence_decision_receipt(receipt)
    if history != verified["history"]:
        msg = f"{role} disagrees with its receipt"
        raise RuntimeError(msg)
    return history


def _require_prior_cadence_history(
    prior_history: object,
    *,
    durable_history: list[dict[str, Any]] | None,
    trajectory_digest: str,
    stable_comparison_digest: str,
    current_step: int,
) -> list[dict[str, Any]]:
    prior = _cadence_history(prior_history, role="prior cadence history")
    if current_step == CADENCE_INITIAL_STEP:
        if prior or durable_history is not None:
            msg = "step 100 control requires an empty prior history"
            raise RuntimeError(msg)
        return prior
    if current_step < CADENCE_INITIAL_STEP:
        msg = "internal control cannot start before step 100"
        raise RuntimeError(msg)
    if durable_history is not None and prior != durable_history:
        msg = "prior cadence history disagrees with durable history"
        raise RuntimeError(msg)
    _require_complete_cadence_history(
        prior,
        trajectory_digest=trajectory_digest,
        stable_comparison_digest=stable_comparison_digest,
        expected_last_step=current_step - CADENCE_INTERVAL_STEPS,
    )
    return prior


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    repo_id: str
    revision: str
    artifact_path: str
    checkpoint_digest: str
    trajectory_digest: str
    max_all_in_usd: float
    all_in_rate_usd_per_second: float
    reserve_seconds_per_cell: int
    profile: str

    def __post_init__(self) -> None:
        if not self.repo_id or not self.revision or not self.artifact_path or not self.profile:
            msg = "benchmark request requires artifact identity and profile"
            raise ValueError(msg)
        if not is_sha256(self.checkpoint_digest):
            msg = "benchmark request checkpoint digest is invalid"
            raise ValueError(msg)
        if not is_sha256(self.trajectory_digest):
            msg = "benchmark request trajectory digest is invalid"
            raise ValueError(msg)
        if len(self.revision) != GIT_SHA_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            msg = "benchmark request revision must be an immutable Git SHA"
            raise ValueError(msg)
        path = Path(self.artifact_path)
        if path.is_absolute() or ".." in path.parts:
            msg = "benchmark request artifact path must remain below its root"
            raise ValueError(msg)
        profile_ceiling = PROFILE_ALL_IN_CEILING_USD.get(self.profile)
        if profile_ceiling is None:
            msg = "profile is not an approved evaluation workspace"
            raise ValueError(msg)
        if (
            self.max_all_in_usd <= 0
            or self.max_all_in_usd > profile_ceiling
            or self.all_in_rate_usd_per_second != A10G_ALL_IN_RATE_USD_PER_SECOND
            or self.reserve_seconds_per_cell < MINIMUM_PERSISTENCE_RESERVE_SECONDS
        ):
            msg = (
                "benchmark budget must not exceed the profile-pinned all-in ceiling, "
                "A10G rate, and minimum persistence reserve"
            )
            raise ValueError(msg)


def checkpoint_model_name(request: BenchmarkRequest) -> str:
    return f"pii350-checkpoint-{request.checkpoint_digest[:16]}"


def _request_json(request: BenchmarkRequest) -> str:
    return json.dumps(asdict(request), sort_keys=True, separators=(",", ":"))


def render_cache_preflight_command(profile: str) -> str:
    require_evaluation_profile(profile)
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action cache_preflight --profile {shlex.quote(profile)}"
    )


def render_triton_compiler_preflight_command(profile: str) -> str:
    require_evaluation_profile(profile)
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action triton_compiler_preflight --profile {shlex.quote(profile)}"
    )


def render_cpu_receipt_command(request: BenchmarkRequest) -> str:
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(request.profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action receipt --request-json {shlex.quote(_request_json(request))}"
    )


def render_issue_approval_command(request: BenchmarkRequest, *, purpose: str, launch_nonce: str, ledger_path: str) -> str:
    """Render the local-only signing command. The private key path stays in env.

    Returns:
        The shell command that issues the approval, every interpolated value quoted for the shell.

    Raises:
        ValueError: If the purpose is not one of the five the issuer signs, or the launch nonce is
            not a SHA-256.

    """
    if purpose not in {
        "full_benchmark",
        STEP60_BASELINE_PURPOSE,
        STEP60_PARTIAL_16OF17_PURPOSE,
        CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        "internal_control",
    }:
        msg = "approval purpose is invalid"
        raise ValueError(msg)
    if not is_sha256(launch_nonce):
        msg = "launch approval nonce is invalid"
        raise ValueError(msg)
    return (
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        "--action issue_approval "
        f"--request-json {shlex.quote(_request_json(request))} "
        f"--approval-purpose {shlex.quote(purpose)} "
        f"--launch-nonce {shlex.quote(launch_nonce)} "
        f"--ledger-path {shlex.quote(ledger_path)}"
    )


def render_step60_reuse_receipt_command(request: BenchmarkRequest) -> str:
    """Render the CPU-only gate separately for inspection or recovery.

    Returns:
        The shell command that produces the step-60 reuse receipt, with request and profile quoted.

    """
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(request.profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action step60_reuse_receipt --request-json {shlex.quote(_request_json(request))}"
    )


def require_fill_cells(cells: object) -> tuple[str, ...]:
    """Validate an explicit fill cell list and pin it to canonical matrix order.

    Canonical order keeps one cell set to one contract digest, so the same
    request cannot mint two different approvals by reordering its own list.

    Returns:
        The requested cells in ``EVAL_DATASETS`` order, whatever order they arrived in.

    Raises:
        ValueError: If the argument is not a non-string sequence, holds a non-string element, is
            empty, repeats a cell, or names a cell outside the matrix.

    """
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        msg = "fill_missing_cells requires an explicit cell list"
        raise ValueError(msg)
    requested = [cell for cell in cells if isinstance(cell, str)]
    if len(requested) != len(cells):
        msg = "fill_missing_cells cell list must contain cell names"
        raise ValueError(msg)
    if not requested:
        msg = "fill_missing_cells requires at least one cell"
        raise ValueError(msg)
    if len(set(requested)) != len(requested):
        msg = "fill_missing_cells cell list repeats a cell"
        raise ValueError(msg)
    unknown = sorted(set(requested) - set(EVAL_DATASETS))
    if unknown:
        msg = f"fill_missing_cells cell list has unknown cells: {unknown}"
        raise ValueError(msg)
    selected = set(requested)
    return tuple(dataset for dataset in EVAL_DATASETS if dataset in selected)


def benchmark_contract(request: BenchmarkRequest, *, purpose: str, cells: Sequence[str] | None = None) -> dict[str, Any]:
    if purpose not in {
        "full_benchmark",
        STEP60_BASELINE_PURPOSE,
        STEP60_PARTIAL_16OF17_PURPOSE,
        CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        FILL_MISSING_CELLS_PURPOSE,
        "internal_control",
    }:
        msg = "benchmark purpose is invalid"
        raise ValueError(msg)
    if (cells is not None) != (purpose == FILL_MISSING_CELLS_PURPOSE):
        msg = "only a fill_missing_cells contract binds an explicit cell list"
        raise ValueError(msg)
    if purpose == FILL_MISSING_CELLS_PURPOSE:
        datasets = require_fill_cells(cells)
        rows = sum(EVAL_EXPECTED_ROWS[dataset] for dataset in datasets)
    elif purpose in {"full_benchmark", STEP60_BASELINE_PURPOSE}:
        datasets = EVAL_DATASETS
        rows = FULL_ROWS
    elif purpose in {
        STEP60_PARTIAL_16OF17_PURPOSE,
        CHECKPOINT_PARTIAL_16OF17_PURPOSE,
    }:
        datasets = STEP60_PARTIAL_16OF17_DATASETS
        rows = STEP60_PARTIAL_16OF17_ROWS
    else:
        datasets = ("v2-eval",)
        rows = EVAL_EXPECTED_ROWS["v2-eval"]
    body: dict[str, Any] = {
        "schema_version": 1,
        "purpose": purpose,
        "checkpoint": {
            "repo_id": request.repo_id,
            "revision": request.revision,
            "artifact_path": request.artifact_path,
            "checkpoint_digest": request.checkpoint_digest,
            "trajectory_digest": request.trajectory_digest,
        },
        "model": "LiquidAI/LFM2.5-Encoder-350M-PII-Detector+PEFT-LoRA-r128-a256+Meddies-37-head",
        "datasets": list(datasets),
        "rows_by_cell": {dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in datasets},
        "total_rows": rows,
        "v2_revision": V2_DATASET_REVISION,
        "external_revision": EXTERNAL_DATASET_REVISION,
        "gpu": GPU,
        "batching": {
            "max_batch_size": 8,
            "max_batch_tokens": 32_768,
            "max_sequence_length": 8192,
        },
        "budget": {
            "max_all_in_usd": request.max_all_in_usd,
            "all_in_rate_usd_per_second": A10G_ALL_IN_RATE_USD_PER_SECOND,
            "persistence_reserve_seconds": request.reserve_seconds_per_cell,
            "minimum_persistence_reserve_seconds": MINIMUM_PERSISTENCE_RESERVE_SECONDS,
        },
        "profile": request.profile,
    }
    if purpose == STEP60_BASELINE_PURPOSE:
        body["expected_full9_gold_spans"] = FULL_GOLD_SPANS
        body["v2_eval_policy"] = V2_EVAL_REUSE_POLICY
        body["v2_eval_challenge_policy"] = V2_CHALLENGE_REUSE_POLICY
    if purpose in {
        STEP60_PARTIAL_16OF17_PURPOSE,
        CHECKPOINT_PARTIAL_16OF17_PURPOSE,
    }:
        body["excluded_cells"] = list(STEP60_PARTIAL_16OF17_EXCLUDED_CELLS)
    return {**body, "contract_digest": canonical_sha256(body)}


def require_full_matrix(contract: Mapping[str, Any]) -> None:
    if contract.get("purpose") != "full_benchmark" or tuple(contract.get("datasets", ())) != EVAL_DATASETS:
        msg = "full benchmark must schedule the exact 17-cell EVAL_DATASETS matrix"
        raise RuntimeError(msg)
    if contract.get("rows_by_cell") != EVAL_EXPECTED_ROWS or contract.get("total_rows") != FULL_ROWS:
        msg = "full benchmark row schedule drifted"
        raise RuntimeError(msg)
    if (
        contract.get("v2_revision") != V2_DATASET_REVISION
        or contract.get("external_revision") != EXTERNAL_DATASET_REVISION
    ):
        msg = "full benchmark dataset revision drifted"
        raise RuntimeError(msg)


def require_step60_checkpoint(optimizer_step: int) -> None:
    if optimizer_step != STEP60_BASELINE_OPTIMIZER_STEP:
        msg = "step-60 baseline requires optimizer step 60"
        raise RuntimeError(msg)


def require_checkpoint_partial_optimizer_step(optimizer_step: int) -> None:
    if optimizer_step != STEP60_BASELINE_OPTIMIZER_STEP and (
        optimizer_step < MIN_PARTIAL_OPTIMIZER_STEP or optimizer_step % CADENCE_INTERVAL_STEPS != 0
    ):
        msg = "checkpoint partial 16/17 requires step 60 or a 50-step cadence from step 100"
        raise RuntimeError(msg)


def require_checkpoint_partial_checkpoint(request: BenchmarkRequest, verified: VerifiedCheckpoint) -> None:
    """Bind a generic partial evaluation to the verified checkpoint identity.

    Raises:
        RuntimeError: If the verified revision disagrees with the request; or, for a terminal
            checkpoint, if its digest, step, packed cursor, or world size is not the allowlisted
            step-146 identity. A non-terminal checkpoint is delegated to the cadence-step rule.

    """
    if verified.artifact.revision != request.revision:
        msg = "verified checkpoint revision does not match benchmark request"
        raise RuntimeError(msg)
    if verified.lifecycle_state == "terminal":
        if (
            request.checkpoint_digest != TERMINAL_STEP146_CHECKPOINT_DIGEST
            or verified.optimizer_step != TERMINAL_STEP146_OPTIMIZER_STEP
            or verified.packed_cursor != TERMINAL_STEP146_PACKED_CURSOR
            or verified.world_size != TERMINAL_STEP146_WORLD_SIZE
        ):
            msg = "terminal checkpoint does not match the allowlisted step-146 identity"
            raise RuntimeError(msg)
        return
    require_checkpoint_partial_optimizer_step(verified.optimizer_step)


def require_step60_baseline_matrix(contract: Mapping[str, Any]) -> None:
    # reason: Step-60 baseline requires 17 cells, pinned purpose, and row totals in one matrix gate.
    if (
        contract.get("purpose") != STEP60_BASELINE_PURPOSE  # ruff: ignore[too-many-boolean-expressions]
        or tuple(contract.get("datasets", ())) != EVAL_DATASETS
        or contract.get("rows_by_cell") != EVAL_EXPECTED_ROWS
        or contract.get("total_rows") != FULL_ROWS
        or contract.get("expected_full9_gold_spans") != FULL_GOLD_SPANS
        or contract.get("v2_eval_policy") != V2_EVAL_REUSE_POLICY
        or contract.get("v2_eval_challenge_policy") != V2_CHALLENGE_REUSE_POLICY
        or contract.get("v2_revision") != V2_DATASET_REVISION
        or contract.get("external_revision") != EXTERNAL_DATASET_REVISION
    ):
        msg = "step-60 baseline must schedule the exact 17-cell matrix"
        raise RuntimeError(msg)


def require_step60_partial_16of17_matrix(contract: Mapping[str, Any]) -> None:
    # reason: Step-60 partial reuse requires 16 cells plus one sanctioned gap in one matrix gate.
    if (
        contract.get("purpose") != STEP60_PARTIAL_16OF17_PURPOSE  # ruff: ignore[too-many-boolean-expressions]
        or tuple(contract.get("datasets", ())) != STEP60_PARTIAL_16OF17_DATASETS
        or contract.get("rows_by_cell")
        != {dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in STEP60_PARTIAL_16OF17_DATASETS}
        or contract.get("total_rows") != STEP60_PARTIAL_16OF17_ROWS
        or contract.get("excluded_cells") != list(STEP60_PARTIAL_16OF17_EXCLUDED_CELLS)
        or contract.get("v2_revision") != V2_DATASET_REVISION
        or contract.get("external_revision") != EXTERNAL_DATASET_REVISION
    ):
        msg = "step-60 partial 16/17 benchmark must schedule its exact fixed matrix"
        raise RuntimeError(msg)


def require_checkpoint_partial_16of17_matrix(contract: Mapping[str, Any]) -> None:
    # reason: require keeps purpose/rows by in one gate; helper predicates would scatter the rule.
    if (
        contract.get("purpose") != CHECKPOINT_PARTIAL_16OF17_PURPOSE  # ruff: ignore[too-many-boolean-expressions]
        or tuple(contract.get("datasets", ())) != STEP60_PARTIAL_16OF17_DATASETS
        or contract.get("rows_by_cell")
        != {dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in STEP60_PARTIAL_16OF17_DATASETS}
        or contract.get("total_rows") != STEP60_PARTIAL_16OF17_ROWS
        or contract.get("excluded_cells") != list(STEP60_PARTIAL_16OF17_EXCLUDED_CELLS)
        or contract.get("v2_revision") != V2_DATASET_REVISION
        or contract.get("external_revision") != EXTERNAL_DATASET_REVISION
    ):
        msg = "checkpoint partial 16/17 benchmark must schedule its exact fixed matrix"
        raise RuntimeError(msg)


def require_fill_missing_cells_matrix(contract: Mapping[str, Any], *, cells: Sequence[str]) -> None:
    expected = require_fill_cells(cells)
    # reason: require fill keeps purpose/rows by in one gate; helper predicates would scatter the rule.
    if (
        contract.get("purpose") != FILL_MISSING_CELLS_PURPOSE  # ruff: ignore[too-many-boolean-expressions]
        or tuple(contract.get("datasets", ())) != expected
        or contract.get("rows_by_cell") != {dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in expected}
        or contract.get("total_rows") != sum(EVAL_EXPECTED_ROWS[dataset] for dataset in expected)
        or contract.get("v2_revision") != V2_DATASET_REVISION
        or contract.get("external_revision") != EXTERNAL_DATASET_REVISION
    ):
        msg = "fill_missing_cells benchmark must schedule its exact approved cell list"
        raise RuntimeError(msg)


def require_fillable_cells(scheduled_datasets: Sequence[str], missing_datasets: Sequence[str]) -> None:
    """Refuse to pay a second time for a cell this generation already completed.

    Raises:
        RuntimeError: If any scheduled dataset is already complete in this generation.

    """
    missing = set(missing_datasets)
    already_complete = tuple(dataset for dataset in scheduled_datasets if dataset not in missing)
    if already_complete:
        msg = f"fill_missing_cells refuses cells already completed in this generation: {list(already_complete)}"
        raise RuntimeError(
            msg,
        )


def _parse_cells_argument(value: str) -> tuple[str, ...]:
    return require_fill_cells([cell.strip() for cell in value.split(",") if cell.strip()])


def approved_fill_cells(approval: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the cell list the signature covers; the executor runs only this.

    Returns:
        The approved cells in canonical matrix order, taken from the signed body rather than from
        anything the caller supplied.

    Raises:
        RuntimeError: If the approval was issued for some other purpose than a fill run.

    """
    if approval.get("purpose") != FILL_MISSING_CELLS_PURPOSE:
        msg = "launch approval does not authorize a fill run"
        raise RuntimeError(msg)
    return require_fill_cells(approval.get("cells"))


def fill_missing_cells_coverage_report(*, filled_cells: Sequence[str], generation_digest: str) -> dict[str, Any]:
    """Report only the cells that this fill run was approved to score.

    Returns:
        The filled cells, their count against the full matrix, the generation digest, and a
        ``eligible_for_full_matrix_safety_veto`` that is always False for a fill run.

    """
    filled = list(filled_cells)
    return {
        "coverage_cells": f"{len(filled)}/{len(EVAL_DATASETS)}",
        "filled_cells": filled,
        "evaluation_generation_digest": generation_digest,
        "eligible_for_full_matrix_safety_veto": False,
    }


def step60_baseline_coverage_report(*, v2_eval_reused: bool) -> dict[str, Any]:
    return {
        "coverage_cells": "17/17",
        "v2_eval": {
            "status": "reused" if v2_eval_reused else "rerun",
            "policy": V2_EVAL_REUSE_POLICY,
        },
        "v2_eval_challenge": {
            "status": "reused",
            "policy": V2_CHALLENGE_REUSE_POLICY,
        },
        "eligible_for_full_matrix_safety_veto": True,
    }


def step60_partial_16of17_coverage_report() -> dict[str, Any]:
    return {
        "coverage_cells": "16/17",
        "excluded_cells": list(STEP60_PARTIAL_16OF17_EXCLUDED_CELLS),
        "eligible_for_full_matrix_safety_veto": False,
    }


def fill_execution_order(cells: Sequence[str]) -> tuple[str, ...]:
    """Run the cheapest approved cell first so a stop loses the least work.

    Identity stays canonical: the contract and the signed approval keep matrix
    order. Only the execution schedule is reordered.

    Returns:
        The same cells ordered by expected row count, ties broken by name so the order is stable.

    """
    return tuple(sorted(cells, key=lambda dataset: (EVAL_EXPECTED_ROWS[dataset], dataset)))


def predict_fill_cell_seconds(*, next_rows: int) -> float:
    """Project a fill cell from a pinned prior instead of the bootstrap rule.

    An approved fill list need not contain the bounded bootstrap cell, so the
    shared projector's cold-start branch would refuse the very first cell after
    the paid lease is already active. The prior is deliberately independent of
    observed rates, which early in a run carry kernel-compile overhead.

    Returns:
        The projected seconds for the cell, the row count times the pinned per-row prior.

    Raises:
        ValueError: If the row count is not positive.

    """
    if next_rows <= 0:
        msg = "next cell rows must be positive"
        raise ValueError(msg)
    return next_rows * FILL_PRIOR_SECONDS_PER_ROW


def fill_budget_allows_next_cell(*, elapsed_seconds: float, next_rows: int, request: BenchmarkRequest) -> float:
    predicted_next_cell_seconds = predict_fill_cell_seconds(next_rows=next_rows)
    projected = (
        elapsed_seconds + predicted_next_cell_seconds + request.reserve_seconds_per_cell
    ) * A10G_ALL_IN_RATE_USD_PER_SECOND
    if projected > request.max_all_in_usd:
        msg = f"fill budget refuses next cell: projected=${projected:.4f} ceiling=${request.max_all_in_usd:.4f}"
        raise BudgetRefusal(
            msg,
        )
    return predicted_next_cell_seconds


def single_missing_cell_resume_admission(
    *,
    scheduled_datasets: Sequence[str],
    validated_completed_cells: Sequence[str],
    missing_cells: Sequence[str],
    elapsed_seconds: float,
    request: BenchmarkRequest,
) -> dict[str, Any] | None:
    """Admit one identity-proven missing shard without stale-cell timing reuse.

    Returns:
        The admission record naming the single missing cell, the reused cells, and the reserve and
        progress windows; or None when the matrix is not in the exactly-one-missing shape, which
        leaves the decision to the caller rather than refusing it here.

    Raises:
        BudgetRefusal: If the seconds left under the ceiling do not cover the persistence reserve
            plus the progress window, so the dispatch is refused before it can spend anything.

    """
    scheduled = tuple(scheduled_datasets)
    completed = tuple(validated_completed_cells)
    missing = tuple(missing_cells)
    expected_missing = tuple(dataset for dataset in scheduled if dataset not in completed)
    # reason: single missing keeps scheduled/completed in one gate; helper predicates would scatter the rule.
    if (
        not scheduled  # ruff: ignore[too-many-boolean-expressions]
        or len(set(scheduled)) != len(scheduled)
        or len(set(completed)) != len(completed)
        or tuple(dataset for dataset in scheduled if dataset not in missing) != completed
        or missing != expected_missing
        or len(missing) != 1
    ):
        return None
    remaining_seconds = request.max_all_in_usd / A10G_ALL_IN_RATE_USD_PER_SECOND - elapsed_seconds
    required_seconds = request.reserve_seconds_per_cell + SINGLE_MISSING_CELL_RESUME_PROGRESS_WINDOW_SECONDS
    if remaining_seconds < required_seconds:
        msg = (
            "single-missing-cell resume refuses dispatch: "
            f"remaining_seconds={remaining_seconds:.4f} "
            f"required_seconds={required_seconds}"
        )
        raise BudgetRefusal(msg)
    return {
        "mode": "single_missing_cell_resume",
        "hard_ceiling_usd": request.max_all_in_usd,
        "reused_cells": list(completed),
        "missing_cells": list(missing),
        "persistence_reserve_seconds": request.reserve_seconds_per_cell,
        "progress_window_seconds": SINGLE_MISSING_CELL_RESUME_PROGRESS_WINDOW_SECONDS,
    }


def require_complete_or_single_missing_partial_matrix(
    *,
    scheduled_datasets: Sequence[str],
    validated_completed_cells: Sequence[str],
    missing_cells: Sequence[str],
    elapsed_seconds: float,
    request: BenchmarkRequest,
) -> dict[str, Any]:
    """Classify a fixed partial matrix without allowing multi-cell repair drift.

    A caller must also expect `BudgetRefusal`, which the single-missing-cell admission raises and
    this function propagates unchanged.

    Returns:
        ``fresh_exact16`` when nothing is complete, ``aggregate_only`` when everything is, or the
        single-missing-cell admission record. No other shape is reachable.

    Raises:
        RuntimeError: If the schedule is empty, repeats a cell, names a completed cell it never
            scheduled, or disagrees with its own missing list; and again when two or more cells are
            missing, which is the multi-cell repair this refuses.

    """
    scheduled = tuple(scheduled_datasets)
    completed = tuple(validated_completed_cells)
    missing = tuple(missing_cells)
    expected_missing = tuple(dataset for dataset in scheduled if dataset not in completed)
    if (
        not scheduled
        or len(set(scheduled)) != len(scheduled)
        or len(set(completed)) != len(completed)
        or any(dataset not in scheduled for dataset in completed)
        or missing != expected_missing
    ):
        msg = "checkpoint partial 16/17 requires a complete identity-proven matrix or an exact fresh/one-cell state"
        raise RuntimeError(
            msg,
        )
    if not completed and missing == scheduled:
        return {"mode": "fresh_exact16"}
    if completed == scheduled and not missing:
        return {"mode": "aggregate_only"}
    resume = single_missing_cell_resume_admission(
        scheduled_datasets=scheduled,
        validated_completed_cells=completed,
        missing_cells=missing,
        elapsed_seconds=elapsed_seconds,
        request=request,
    )
    if resume is not None:
        return resume
    msg = (
        "checkpoint partial 16/17 requires a complete identity-proven matrix, "
        "a fresh exact16 matrix, or exactly one missing cell"
    )
    raise RuntimeError(msg)


def require_expected_trajectory(request: BenchmarkRequest, observed_trajectory_digest: str) -> None:
    if observed_trajectory_digest != request.trajectory_digest:
        msg = "checkpoint trajectory does not match the requested frozen run"
        raise RuntimeError(msg)


def require_evaluation_profile(profile: str) -> None:
    if profile not in EVALUATION_PROFILES:
        msg = "profile is not an approved evaluation workspace"
        raise ValueError(msg)


def require_local_profile_name(profile: str) -> None:
    require_evaluation_profile(profile)
    if os.environ.get(PROFILE_ENVIRONMENT_KEY) != profile:
        msg = "local MODAL_PROFILE does not match the benchmark request"
        raise RuntimeError(msg)


def require_local_profile(request: BenchmarkRequest) -> None:
    require_local_profile_name(request.profile)


TRITON_COMPILER_PROOF_SOURCE = "int triton_preflight(void) { return 0; }\n"

TRITON_COMPILER_PROOF_ARGUMENTS = (
    "-shared",
    "-fPIC",
    "-x",
    "c",
    "-",
    "-o",
    "<output>",
)


def evaluation_generation_digest_from_root(root: Path) -> str | None:
    generation_root = root if root.parent.name == "evaluation-generations" else root.parent
    if generation_root.parent.name == "evaluation-generations" and is_sha256(generation_root.name):
        return generation_root.name
    return None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_receipt(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"{role} is absent or invalid"
        raise RuntimeError(msg) from error
    if not isinstance(payload, Mapping):
        msg = f"{role} must be an object"
        raise RuntimeError(msg)
    return payload


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _result_cell_relative_paths(model: str, dataset: str) -> tuple[str, ...]:
    """Name the one shard trio that `read_matrix_results` treats as a done cell.

    Returns:
        The three repo-relative paths - result, sidecar, completion marker - for that cell.

    """
    return tuple(f"results/{model}/{dataset}/{name}" for name in RESULT_CELL_FILENAMES)


def _result_transfer_prefix(checkpoint_digest: str, generation_digest: str) -> str:
    if not is_sha256(checkpoint_digest) or not is_sha256(generation_digest):
        msg = "result transfer namespace digest is invalid"
        raise RuntimeError(msg)
    return f"{RESULT_TRANSFER_ROOT}/{checkpoint_digest}/{generation_digest}"


def _completed_transfer_cells(root: Path, model: str) -> list[str]:
    """Accept only whole cells: a half-written trio is refused, never truncated.

    Returns:
        The datasets whose full trio is present on disk, in matrix order. A cell with no file at all
        is skipped silently, because it was never started.

    Raises:
        RuntimeError: If a cell has some but not all of its three files, naming the missing ones; or
            if no cell is complete, since there would be nothing to transfer.

    """
    cells: list[str] = []
    for dataset in EVAL_DATASETS:
        relatives = _result_cell_relative_paths(model, dataset)
        present = [relative for relative in relatives if (root / relative).is_file()]
        if not present:
            continue
        if len(present) != len(relatives):
            missing = sorted(set(relatives) - set(present))
            msg = f"result cell {dataset} is incomplete and cannot be transferred: {missing}"
            raise RuntimeError(msg)
        cells.append(dataset)
    if not cells:
        msg = "evaluation generation has no completed result cell to transfer"
        raise RuntimeError(msg)
    return cells


def _transfer_file_entries(root: Path, model: str, cells: Sequence[str]) -> list[dict[str, Any]]:
    """Build transfer entries while checking each result against its sidecar.

    Returns:
        One entry per file across every cell, each carrying its repo-relative path, byte count, and
        SHA-256.

    Raises:
        RuntimeError: If a cell's sidecar records a result digest other than the one its result file
            hashes to.

    """
    entries: list[dict[str, Any]] = []
    for dataset in cells:
        digests: dict[str, str] = {}
        for relative in _result_cell_relative_paths(model, dataset):
            path = root / relative
            digest = file_sha256(str(path))
            digests[relative] = digest
            entries.append({
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
            })
        sidecar = _read_receipt(
            root / f"results/{model}/{dataset}/{RESULT_SHARD}.meta.json",
            role=f"{dataset} result sidecar",
        )
        if sidecar.get("result_sha256") != digests[f"results/{model}/{dataset}/{RESULT_SHARD}.jsonl"]:
            msg = f"result cell {dataset} disagrees with its sidecar result digest"
            raise RuntimeError(msg)
    return sorted(entries, key=lambda entry: str(entry["path"]))


def _result_transfer_manifest(*, request: BenchmarkRequest, generation_digest: str, root: Path) -> dict[str, Any]:
    model = checkpoint_model_name(request)
    cells = _completed_transfer_cells(root, model)
    files = _transfer_file_entries(root, model, cells)
    body = {
        "schema_version": RESULT_TRANSFER_SCHEMA_VERSION,
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": request.trajectory_digest,
        "evaluation_generation_digest": generation_digest,
        "model": model,
        "source_profile": request.profile,
        "cells": cells,
        "files": files,
        "total_bytes": sum(int(entry["bytes"]) for entry in files),
    }
    return {**body, "manifest_digest": canonical_sha256(body)}


def _require_commit_sha(commit: object) -> str:
    oid = getattr(commit, "oid", None)
    if (
        not isinstance(oid, str)
        or len(oid) != GIT_SHA_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in oid)
    ):
        msg = "result transfer upload did not return an immutable commit SHA"
        raise RuntimeError(msg)
    return oid


def _empty_exact_counts() -> dict[str, int]:
    return {"tp": 0, "pred_total": 0, "gold_total": 0}


def _empty_containment_counts() -> dict[str, int]:
    return {"precision_tp": 0, "recall_tp": 0, "pred_total": 0, "gold_total": 0}


def _add_counts(total: dict[str, int], observed: Mapping[str, int]) -> None:
    for key, value in observed.items():
        total[key] += int(value)


def _score_cell_rows(result_path: Path) -> dict[str, Any]:
    """Recount one persisted cell with the same primitives the aggregate uses.

    The aggregate reports rounded per-cell metrics only, so a cross-generation
    micro total has to be rebuilt from raw counts. Averaging per-cell F1 would
    silently reweight cells by count rather than by rows scored.

    Returns:
        Raw exact and containment counts for the cell, the same pair restricted to the supported
        label set and over all nine labels, plus per-label and per-language breakdowns and the row
        count, so a caller can sum across cells before computing any metric.

    """
    from meddies_pii.eval_baseline.baseline.datasets import canonical_language
    from meddies_pii.evaluation.span_metrics import (
        containment_span_counts,
        containment_span_prf_from_counts,
        exact_span_counts,
        exact_span_prf_from_counts,
        typed_span_counts_by_label,
    )
    from meddies_pii.jsonl import read_jsonl
    from meddies_pii.spans import char_span_from_value

    def spans(raw: object) -> list[Any]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            return []
        return [span for value in raw if (span := char_span_from_value(value)) is not None]

    supported = frozenset(PII_LABEL_SET)
    totals = {
        "exact": _empty_exact_counts(),
        "containment": _empty_containment_counts(),
        "full9_exact": _empty_exact_counts(),
        "full9_containment": _empty_containment_counts(),
    }
    by_label: dict[str, dict[str, dict[str, int]]] = {
        label: {
            "exact": _empty_exact_counts(),
            "containment": _empty_containment_counts(),
        }
        for label in sorted(PII_LABEL_SET)
    }
    by_language: dict[str, dict[str, Any]] = {}
    rows = 0
    for row in read_jsonl(result_path):
        predicted = spans(row.get("pred_spans"))
        gold = spans(row.get("gold_spans"))
        supported_gold = [span for span in gold if span.label in supported]
        exact = exact_span_counts(predicted, supported_gold)
        containment = containment_span_counts(predicted, supported_gold)
        _add_counts(totals["exact"], exact)
        _add_counts(totals["containment"], containment)
        _add_counts(totals["full9_exact"], exact_span_counts(predicted, gold))
        _add_counts(totals["full9_containment"], containment_span_counts(predicted, gold))
        label_exact, label_containment = typed_span_counts_by_label(predicted, gold)
        for label in PII_LABEL_SET:
            _add_counts(by_label[label]["exact"], label_exact[label])
            _add_counts(by_label[label]["containment"], label_containment[label])
        language = canonical_language(str(row.get("language", "unknown")))
        bucket = by_language.setdefault(
            language,
            {
                "rows": 0,
                "exact": _empty_exact_counts(),
                "containment": _empty_containment_counts(),
            },
        )
        bucket["rows"] = int(bucket["rows"]) + 1
        _add_counts(bucket["exact"], exact)
        _add_counts(bucket["containment"], containment)
        rows += 1
    return {
        "rows": rows,
        "counts": totals,
        "counts_by_label": by_label,
        "counts_by_language": {language: by_language[language] for language in sorted(by_language)},
        "metrics": {
            "exact": exact_span_prf_from_counts(totals["exact"]),
            "containment": containment_span_prf_from_counts(totals["containment"]),
        },
    }


# reason: validated keeps result beside result cell; splitting would let sample setup drift.
def _validated_transfer_receipt(  # ruff: ignore[complex-structure]
    payload: object,
    *,
    request: BenchmarkRequest,
) -> dict[str, Any]:
    """Refuse any receipt that does not bind this checkpoint and its own bytes.

    Returns:
        The receipt with its file inventory normalized to exactly path, bytes, and SHA-256 per entry.

    Raises:
        RuntimeError: If the receipt has the wrong field set, its digest does not cover its own body,
            it names a different checkpoint, trajectory, model, or repo than this request, its cell
            list is empty or repeats or leaves the matrix, its namespace disagrees with its digests,
            a file entry is malformed, the file set is not exactly the trio for every named cell, or
            the byte total does not equal the sum of its entries.

    """
    if not isinstance(payload, Mapping) or set(payload) != RESULT_TRANSFER_RECEIPT_KEYS:
        msg = "result transfer receipt has missing or unexpected fields"
        raise RuntimeError(msg)
    receipt = dict(payload)
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if receipt["receipt_digest"] != canonical_sha256(body):
        msg = "result transfer receipt digest is invalid"
        raise RuntimeError(msg)
    model = checkpoint_model_name(request)
    cells = receipt["cells"]
    # reason: validated keeps checkpoint/trajectory in one gate; helper predicates would scatter the rule.
    if (
        receipt["schema_version"] != RESULT_TRANSFER_SCHEMA_VERSION  # ruff: ignore[too-many-boolean-expressions]
        or receipt["checkpoint_digest"] != request.checkpoint_digest
        or receipt["trajectory_digest"] != request.trajectory_digest
        or not is_sha256(receipt["evaluation_generation_digest"])
        or receipt["model"] != model
        or receipt["repo_id"] != RESULT_TRANSFER_REPO_ID
        or not isinstance(cells, list)
        or not cells
        or len(set(cells)) != len(cells)
        or any(dataset not in EVAL_DATASETS for dataset in cells)
        or not is_sha256(receipt["manifest_digest"])
    ):
        msg = "result transfer receipt does not bind this benchmark request"
        raise RuntimeError(msg)
    if receipt["repo_prefix"] != _result_transfer_prefix(
        request.checkpoint_digest,
        str(receipt["evaluation_generation_digest"]),
    ):
        msg = "result transfer receipt namespace does not match its digests"
        raise RuntimeError(msg)
    expected_paths = sorted(relative for dataset in cells for relative in _result_cell_relative_paths(model, str(dataset)))
    files = receipt["files"]
    if not isinstance(files, list):
        msg = "result transfer receipt file inventory is invalid"
        raise RuntimeError(msg)
    normalized: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            msg = "result transfer receipt file entry is invalid"
            raise RuntimeError(msg)
        byte_count = entry["bytes"]
        if (
            not isinstance(entry["path"], str)
            or type(byte_count) is not int
            or byte_count < 0
            or not is_sha256(entry["sha256"])
        ):
            msg = "result transfer receipt file entry is invalid"
            raise RuntimeError(msg)
        normalized.append({
            "path": entry["path"],
            "bytes": byte_count,
            "sha256": entry["sha256"],
        })
    if [str(entry["path"]) for entry in normalized] != expected_paths:
        msg = "result transfer receipt file set has a missing or unexpected cell file"
        raise RuntimeError(msg)
    if receipt["total_bytes"] != sum(int(entry["bytes"]) for entry in normalized):
        msg = "result transfer receipt byte total is invalid"
        raise RuntimeError(msg)
    receipt["files"] = normalized
    return receipt


# reason: step60 reuse keeps request/scored at its adapter seam; bundling would hide required inputs.
def _step60_reuse_receipt_body(  # ruff: ignore[too-many-arguments]
    *,
    request: BenchmarkRequest,
    optimizer_step: int,
    contract_digest: str,
    evaluation_digest: str,
    shard_identity_digest_by_cell: Mapping[str, str],
    result_sha256_by_cell: Mapping[str, str],
    scored_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the no-rerun cells to the checkpoint, fixture, bytes, and scores.

    Returns:
        The receipt body binding the reused cells to the checkpoint, trajectory, contract, fixture
        identity, result bytes, and the scores those bytes produced.

    Raises:
        RuntimeError: If ``v2-eval-challenge`` is not among the reused cells, since the user excluded
            its rerun and reuse is the only way it can be covered; if a reused cell has no shard
            identity; or if any identity digest is not a SHA-256.

    """
    reused_cells = tuple(dataset for dataset in STEP60_REUSED_CELLS if dataset in result_sha256_by_cell)
    if "v2-eval-challenge" not in reused_cells:
        msg = "step-60 baseline cannot prove existing v2-eval-challenge artifact identity; refusing user-excluded rerun"
        raise RuntimeError(
            msg,
        )
    if set(shard_identity_digest_by_cell) != set(reused_cells):
        msg = "step-60 reuse receipt has incomplete shard identities"
        raise RuntimeError(msg)
    if not (
        is_sha256(contract_digest)
        and is_sha256(evaluation_digest)
        and all(is_sha256(value) for value in shard_identity_digest_by_cell.values())
        and all(is_sha256(value) for value in result_sha256_by_cell.values())
    ):
        msg = "step-60 reuse receipt identity digest is invalid"
        raise RuntimeError(msg)
    return {
        "schema_version": STEP60_REUSE_RECEIPT_SCHEMA_VERSION,
        "purpose": STEP60_BASELINE_PURPOSE,
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": request.trajectory_digest,
        "optimizer_step": optimizer_step,
        "contract_digest": contract_digest,
        "evaluation_contract_digest": evaluation_digest,
        "reused_cells": list(reused_cells),
        "rows_by_cell": {dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in reused_cells},
        "dataset_shard_identity_digest_by_cell": dict(shard_identity_digest_by_cell),
        "result_sha256_by_cell": dict(result_sha256_by_cell),
        "scored_report_digest": canonical_sha256(scored_report),
    }
