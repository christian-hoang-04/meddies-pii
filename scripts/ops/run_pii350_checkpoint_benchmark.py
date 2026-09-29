"""Evaluate one verified PII350 checkpoint on the pinned 17-cell matrix.

This runner intentionally owns one complete checkpoint matrix in one Modal
workspace.  It never reruns reference models and never calls the 1,700-row
control a full benchmark.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the heavy dependency loads in the subcommand that needs it, so starting the script does not pay for it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[type-check-without-type-error]
# reason: all 4 sites sit in a function that already raises the same type more than once - checked by AST,
# reason: 6 to 20 same-type raises per function - so converting only the isinstance-guarded raise would split
# reason: one failure class across two exception types on the shape of the guard. Two of the four are inside
# reason: `verify_launch_approval`, whose 17 raises are one contract: this launch approval is refused.
# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a
# reason: script rather than a package member. An `__init__.py` would declare this directory a package
# reason: it is not.
# ruff: file-ignore[import-private-name]
# reason: all 29 sites are ONE import block, pulling this runner's own implementation module's helpers so
# reason: the evidence it writes is produced by the same code the verifier reads. Re-implementing any of
# reason: them here would let the runner and the verifier drift apart silently, which is the failure the
# reason: digest checks exist to prevent. The fix that satisfies the rule is a public re-export inside
# reason: `src/`, which this lane does not own, so it is declared and flagged rather than worked around.
# ruff: file-ignore[suspicious-subprocess-import]
# reason: the Triton compiler preflight shells out to the C compiler to prove it can link, which is the
# reason: only way to observe that capability before a billed GPU run depends on it. The one call site
# reason: carries its own argv reason below.
# ruff: file-ignore[raise-within-try]
# reason: every site is a refusal raised at the point its condition is detected, inside a `try` that turns
# reason: refusals into a recorded outcome. Hoisting the raise past the `try`, which is what the rule asks
# reason: for, separates each guard from the condition that justifies it — on authorization code whose
# reason: readability IS the audit surface. Same argument the type-check declaration above records.
import base64
import json
import os
import shlex
import shutil
import stat
import statistics
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import modal

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from huggingface_hub import HfApi

from meddies_pii.eval_baseline.baseline.datasets import (
    EVAL_DATASETS,
    EVAL_EXPECTED_ROWS,
    EXTERNAL_DATASET_REVISION,
    V2_DATASET_REVISION,
)
from meddies_pii.eval_baseline.pii350_release.checkpoint_benchmark import (
    A10G_ALL_IN_RATE_USD_PER_SECOND,
    APPROVAL_LEDGER_SCHEMA_VERSION,
    APPROVAL_PRIVATE_KEY_ENV,
    BOOTSTRAP_DATASET,
    BOOTSTRAP_MAX_ROWS,
    CADENCE_INITIAL_STEP,
    CADENCE_INTERVAL_STEPS,
    CADENCE_RECEIPT_SCHEMA_VERSION,
    CELL_SCORE_SUMMARY_SCHEMA_VERSION,
    CHECKPOINT_PARTIAL_16OF17_PURPOSE,
    CPU,
    EARLY_STOP_CONSECUTIVE_DEGRADATIONS,
    EARLY_STOP_F1_DEGRADATION,
    EVALUATION_PROFILES,
    FILL_MISSING_CELLS_PURPOSE,
    FILL_PRIOR_SECONDS_PER_ROW,
    FULL_GOLD_SPANS,
    FULL_ROWS,
    GPU,
    HF_CACHE_MOUNT,
    INTERNAL_CONTROL_PROGRESS_EVERY,
    LAUNCH_APPROVAL_SCHEMA_VERSION,
    MEMORY_MIB,
    MINIMUM_PERSISTENCE_RESERVE_SECONDS,
    MODELED_PRIOR_DIFFUSION_ATTEMPTS,
    MODELED_PRIOR_DIFFUSION_TOTAL_USD,
    PREDICTED_CELL_MARGIN,
    PROFILE_ALL_IN_CEILING_USD,
    PROFILE_ENVIRONMENT_KEY,
    PROGRESS_EVERY,
    RESULT_CELL_FILENAMES,
    RESULT_SHARD,
    RESULT_TRANSFER_FILES_PREFIX,
    RESULT_TRANSFER_MANIFEST_FILENAME,
    RESULT_TRANSFER_RECEIPT_KEYS,
    RESULT_TRANSFER_REPO_ID,
    RESULT_TRANSFER_ROOT,
    RESULT_TRANSFER_SCHEMA_VERSION,
    SINGLE_MISSING_CELL_RESUME_PROGRESS_WINDOW_SECONDS,
    STEP60_BASELINE_OPTIMIZER_STEP,
    STEP60_BASELINE_PURPOSE,
    STEP60_PARTIAL_16OF17_DATASETS,
    STEP60_PARTIAL_16OF17_EXCLUDED_CELLS,
    STEP60_PARTIAL_16OF17_PURPOSE,
    STEP60_PARTIAL_16OF17_ROWS,
    STEP60_REUSE_RECEIPT_SCHEMA_VERSION,
    STEP60_REUSED_CELLS,
    TERMINAL_STEP146_CHECKPOINT_DIGEST,
    TERMINAL_STEP146_OPTIMIZER_STEP,
    TERMINAL_STEP146_PACKED_CURSOR,
    TERMINAL_STEP146_WORLD_SIZE,
    TIMEOUT_SECONDS,
    TRITON_C_COMPILER,
    TRITON_COMPILER_APT_PACKAGE,
    TRITON_COMPILER_PROOF_ARGUMENTS,
    TRITON_COMPILER_PROOF_SOURCE,
    V2_CHALLENGE_REUSE_POLICY,
    V2_EVAL_REUSE_POLICY,
    BenchmarkRequest,
    BudgetRefusal,
    BudgetStop,
    DuplicateLaunchNonce,
    _add_counts,
    _approval_ledger_template,
    _atomic_bytes,
    _atomic_json,
    _atomic_local_json,
    _cadence_history,
    _completed_transfer_cells,
    _decimal_usd,
    _empty_containment_counts,
    _empty_exact_counts,
    _parse_cells_argument,
    _read_approval_ledger,
    _read_cadence_history_at_paths,
    _read_receipt,
    _request_json,
    _require_active_ledger_reservation,
    _require_commit_sha,
    _require_complete_cadence_history,
    _require_prior_cadence_history,
    _result_cell_relative_paths,
    _result_transfer_manifest,
    _result_transfer_prefix,
    _score_cell_rows,
    _step60_reuse_receipt_body,
    _transfer_file_entries,
    _validate_approval_ledger,
    _validate_prior_attempts,
    _validated_transfer_receipt,
    _with_approval_ledger_lock,
    approved_fill_cells,
    benchmark_contract,
    build_cadence_decision_receipt,
    cadence_early_stop_decision,
    checkpoint_model_name,
    comparison_contract_digest,
    evaluation_generation_digest_from_root,
    fill_budget_allows_next_cell,
    fill_execution_order,
    fill_missing_cells_coverage_report,
    inspect_approval_ledger,
    predict_fill_cell_seconds,
    render_cache_preflight_command,
    render_cpu_receipt_command,
    render_issue_approval_command,
    render_step60_reuse_receipt_command,
    render_triton_compiler_preflight_command,
    require_checkpoint_partial_16of17_matrix,
    require_checkpoint_partial_checkpoint,
    require_checkpoint_partial_optimizer_step,
    require_complete_or_single_missing_partial_matrix,
    require_evaluation_profile,
    require_expected_trajectory,
    require_fill_cells,
    require_fill_missing_cells_matrix,
    require_fillable_cells,
    require_full_matrix,
    require_local_profile,
    require_local_profile_name,
    require_step60_baseline_matrix,
    require_step60_checkpoint,
    require_step60_partial_16of17_matrix,
    single_missing_cell_resume_admission,
    step60_baseline_coverage_report,
    step60_partial_16of17_coverage_report,
    verify_cadence_decision_receipt,
)
from meddies_pii.evaluation.identity import (
    EvaluationContract,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    is_sha256,
)
from meddies_pii.json_types import is_str_mapping
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
    use_pinned_debian_snapshot,
)
from meddies_pii.taxonomy import PII_LABEL_SET

__all__ = (
    "A10G_ALL_IN_RATE_USD_PER_SECOND",
    "APPROVAL_LEDGER_SCHEMA_VERSION",
    "APPROVAL_PRIVATE_KEY_ENV",
    "BOOTSTRAP_DATASET",
    "BOOTSTRAP_MAX_ROWS",
    "CADENCE_INITIAL_STEP",
    "CADENCE_INTERVAL_STEPS",
    "CADENCE_RECEIPT_SCHEMA_VERSION",
    "CELL_SCORE_SUMMARY_SCHEMA_VERSION",
    "CHECKPOINT_PARTIAL_16OF17_PURPOSE",
    "CPU",
    "EARLY_STOP_CONSECUTIVE_DEGRADATIONS",
    "EARLY_STOP_F1_DEGRADATION",
    "EVALUATION_PROFILES",
    "FILL_MISSING_CELLS_PURPOSE",
    "FILL_PRIOR_SECONDS_PER_ROW",
    "FULL_GOLD_SPANS",
    "FULL_ROWS",
    "GPU",
    "HF_CACHE_MOUNT",
    "INTERNAL_CONTROL_PROGRESS_EVERY",
    "LAUNCH_APPROVAL_SCHEMA_VERSION",
    "MEMORY_MIB",
    "MINIMUM_PERSISTENCE_RESERVE_SECONDS",
    "MODELED_PRIOR_DIFFUSION_ATTEMPTS",
    "MODELED_PRIOR_DIFFUSION_TOTAL_USD",
    "PREDICTED_CELL_MARGIN",
    "PROFILE_ALL_IN_CEILING_USD",
    "PROFILE_ENVIRONMENT_KEY",
    "PROGRESS_EVERY",
    "RESULT_CELL_FILENAMES",
    "RESULT_SHARD",
    "RESULT_TRANSFER_FILES_PREFIX",
    "RESULT_TRANSFER_MANIFEST_FILENAME",
    "RESULT_TRANSFER_RECEIPT_KEYS",
    "RESULT_TRANSFER_REPO_ID",
    "RESULT_TRANSFER_ROOT",
    "RESULT_TRANSFER_SCHEMA_VERSION",
    "SINGLE_MISSING_CELL_RESUME_PROGRESS_WINDOW_SECONDS",
    "STEP60_BASELINE_OPTIMIZER_STEP",
    "STEP60_BASELINE_PURPOSE",
    "STEP60_PARTIAL_16OF17_DATASETS",
    "STEP60_PARTIAL_16OF17_EXCLUDED_CELLS",
    "STEP60_PARTIAL_16OF17_PURPOSE",
    "STEP60_PARTIAL_16OF17_ROWS",
    "STEP60_REUSED_CELLS",
    "STEP60_REUSE_RECEIPT_SCHEMA_VERSION",
    "TERMINAL_STEP146_CHECKPOINT_DIGEST",
    "TERMINAL_STEP146_OPTIMIZER_STEP",
    "TERMINAL_STEP146_PACKED_CURSOR",
    "TERMINAL_STEP146_WORLD_SIZE",
    "TIMEOUT_SECONDS",
    "TRITON_COMPILER_APT_PACKAGE",
    "TRITON_COMPILER_PROOF_ARGUMENTS",
    "TRITON_COMPILER_PROOF_SOURCE",
    "TRITON_C_COMPILER",
    "V2_CHALLENGE_REUSE_POLICY",
    "V2_EVAL_REUSE_POLICY",
    "BenchmarkRequest",
    "BudgetRefusal",
    "BudgetStop",
    "DuplicateLaunchNonce",
    "_add_counts",
    "_approval_ledger_template",
    "_atomic_bytes",
    "_atomic_json",
    "_atomic_local_json",
    "_cadence_history",
    "_completed_transfer_cells",
    "_decimal_usd",
    "_empty_containment_counts",
    "_empty_exact_counts",
    "_parse_cells_argument",
    "_read_approval_ledger",
    "_read_cadence_history_at_paths",
    "_read_receipt",
    "_request_json",
    "_require_active_ledger_reservation",
    "_require_commit_sha",
    "_require_complete_cadence_history",
    "_require_prior_cadence_history",
    "_result_cell_relative_paths",
    "_result_transfer_manifest",
    "_result_transfer_prefix",
    "_score_cell_rows",
    "_step60_reuse_receipt_body",
    "_transfer_file_entries",
    "_validate_approval_ledger",
    "_validate_prior_attempts",
    "_validated_transfer_receipt",
    "_with_approval_ledger_lock",
    "approved_fill_cells",
    "benchmark_contract",
    "build_cadence_decision_receipt",
    "cadence_early_stop_decision",
    "checkpoint_model_name",
    "comparison_contract_digest",
    "evaluation_generation_digest_from_root",
    "fill_budget_allows_next_cell",
    "fill_execution_order",
    "fill_missing_cells_coverage_report",
    "inspect_approval_ledger",
    "predict_fill_cell_seconds",
    "render_cache_preflight_command",
    "render_cpu_receipt_command",
    "render_issue_approval_command",
    "render_step60_reuse_receipt_command",
    "render_triton_compiler_preflight_command",
    "require_checkpoint_partial_16of17_matrix",
    "require_checkpoint_partial_checkpoint",
    "require_checkpoint_partial_optimizer_step",
    "require_complete_or_single_missing_partial_matrix",
    "require_evaluation_profile",
    "require_expected_trajectory",
    "require_fill_cells",
    "require_fill_missing_cells_matrix",
    "require_fillable_cells",
    "require_full_matrix",
    "require_local_profile",
    "require_local_profile_name",
    "require_step60_baseline_matrix",
    "require_step60_checkpoint",
    "require_step60_partial_16of17_matrix",
    "single_missing_cell_resume_admission",
    "step60_baseline_coverage_report",
    "step60_partial_16of17_coverage_report",
    "verify_cadence_decision_receipt",
)
"""Re-exported so the Modal entrypoints and the benchmark tests read one namespace while the logic itself.

stays importable.

"""

APP_NAME = "meddies-pii350-checkpoint-benchmark"
BOOTSTRAP_CELL_SECONDS = 1_200.0
CHECKPOINT_CACHE_MOUNT = "/cache/checkpoints"
BENCHMARK_MOUNT = "/benchmark"
APPROVAL_PUBLIC_KEY_HEX = "b5e76d7fa824434237695af5b8fdac2909d1b6bab7d87bf3203e0b94c7039462"
"""The evaluator image only needs this public verification key.

The matching private key stays on the local issuer machine and is read from the environment selected key file only for the
local `issue_approval` action.

"""
APPROVAL_ISSUER_KEY_ID = f"ed25519:{canonical_sha256({'public_key_hex': APPROVAL_PUBLIC_KEY_HEX})}"
TRITON_COMPILER_APT_PACKAGES = ("gcc=4:12.2.0-3",)
"""The pinned install form is a literal tuple because the audit scanner in.

scripts/ops/modal_runtime_direct_dependency_inventory.py resolves only starred literal string tuples.

"""


# reason: approval body keeps request/cells at its adapter seam; bundling would hide required inputs.
def _approval_body(  # ruff: ignore[too-many-arguments]
    request: BenchmarkRequest,
    *,
    purpose: str,
    launch_nonce: str,
    external_ledger_approval_id: str,
    modeled_prior_attempts: Sequence[Mapping[str, object]],
    cells: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the body that the local issuer signs; never mint an approval here."""
    if (cells is not None) != (purpose == FILL_MISSING_CELLS_PURPOSE):
        msg = "only a fill_missing_cells approval binds a cell list"
        raise ValueError(msg)
    prior = [dict(entry) for entry in modeled_prior_attempts]
    prior_total = sum(
        (_decimal_usd(entry["modeled_all_in_cost_usd"], field="prior cost") for entry in prior),
        Decimal(0),
    )
    body = {
        "schema_version": LAUNCH_APPROVAL_SCHEMA_VERSION,
        "profile": request.profile,
        "checkpoint_digest": request.checkpoint_digest,
        "purpose": purpose,
        "launch_nonce": launch_nonce,
        "external_ledger_approval_id": external_ledger_approval_id,
        "modeled_prior_attempts": prior,
        "new_attempt_ceiling_usd": str(request.max_all_in_usd),
        "cumulative_modeled_ceiling_usd": str(prior_total + Decimal(str(request.max_all_in_usd))),
        "profile_allocation_usd": str(PROFILE_ALL_IN_CEILING_USD[request.profile]),
        "issuer_key_id": APPROVAL_ISSUER_KEY_ID,
    }
    if cells is not None:
        body["cells"] = list(require_fill_cells(cells))
    return body


def _load_issuer_private_key() -> Ed25519PrivateKey:
    key_file = os.environ.get(APPROVAL_PRIVATE_KEY_ENV)
    if not key_file:
        msg = f"{APPROVAL_PRIVATE_KEY_ENV} is required for local approval issuance"
        raise RuntimeError(msg)
    path = Path(key_file)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_symlink() or not path.is_file() or mode & 0o077:
            msg = "approval issuer private key file must be a regular 0600 file"
            raise RuntimeError(msg)
        raw = path.read_bytes()
    except OSError as error:
        msg = "approval issuer private key file is unavailable"
        raise RuntimeError(msg) from error
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = serialization.load_pem_private_key(raw, password=None)
    except Exception as error:
        msg = "approval issuer private key is invalid"
        raise RuntimeError(msg) from error
    if not isinstance(key, Ed25519PrivateKey):
        msg = "approval issuer private key must be Ed25519"
        raise RuntimeError(msg)
    from cryptography.hazmat.primitives import serialization

    public_hex = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    if public_hex != APPROVAL_PUBLIC_KEY_HEX:
        msg = "approval issuer private key does not match the pinned public key"
        raise RuntimeError(msg)
    return key


def issue_launch_approval(
    request: BenchmarkRequest,
    *,
    purpose: str,
    launch_nonce: str,
    ledger_path: str | Path,
    cells: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Locally reserve one profile budget and sign an approval with Ed25519."""
    if purpose not in {
        "full_benchmark",
        STEP60_BASELINE_PURPOSE,
        STEP60_PARTIAL_16OF17_PURPOSE,
        CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        FILL_MISSING_CELLS_PURPOSE,
        "internal_control",
    }:
        msg = "approval purpose is invalid"
        raise RuntimeError(msg)
    if not is_sha256(launch_nonce):
        msg = "launch approval nonce is invalid"
        raise RuntimeError(msg)
    issuer_key = _load_issuer_private_key()

    def issue(path: Path) -> dict[str, Any]:
        """One profile may hold several live reservations at once.

        So every still-active ceiling has to count against the allocation alongside settled spend. This term is what bounds
        concurrent launches: settled cost is only recorded when a reservation closes, so without it two approvals could
        each pass on their own and together exceed the cap.

        """
        ledger = _read_approval_ledger(path)
        if any(reservation["launch_nonce"] == launch_nonce for reservation in ledger["reservations"]):
            msg = "approval ledger launch nonce was already issued"
            raise RuntimeError(msg)
        prior = ledger["prior_modeled_attempts_by_profile"][request.profile]
        prior_total = sum(
            (_decimal_usd(entry["modeled_all_in_cost_usd"], field="prior modeled spend") for entry in prior),
            Decimal(0),
        )
        active_total = sum(
            (
                _decimal_usd(
                    reservation["new_attempt_ceiling_usd"],
                    field="active reservation ceiling",
                )
                for reservation in ledger["reservations"]
                if reservation["profile"] == request.profile and reservation["status"] == "active"
            ),
            Decimal(0),
        )
        new_ceiling = Decimal(str(request.max_all_in_usd))
        allocation = Decimal(str(PROFILE_ALL_IN_CEILING_USD[request.profile]))
        if new_ceiling > allocation or prior_total + active_total + new_ceiling > allocation:
            msg = "approval ledger cumulative modeled ceiling exceeds profile allocation"
            raise RuntimeError(msg)
        external_id = f"ledger-{canonical_sha256({'profile': request.profile, 'nonce': launch_nonce, 'purpose': purpose})}"
        body = _approval_body(
            request,
            purpose=purpose,
            launch_nonce=launch_nonce,
            external_ledger_approval_id=external_id,
            modeled_prior_attempts=prior,
            cells=cells,
        )
        approval_digest = canonical_sha256(body)
        if any(reservation["approval_digest"] == approval_digest for reservation in ledger["reservations"]):
            msg = "approval ledger digest was already issued"
            raise RuntimeError(msg)
        signature = base64.b64encode(issuer_key.sign(canonical_json_bytes(body))).decode("ascii")
        approval = {**body, "approval_digest": approval_digest, "signature": signature}
        ledger["reservations"].append({
            "approval_digest": approval_digest,
            "profile": request.profile,
            "launch_nonce": launch_nonce,
            "purpose": purpose,
            "status": "active",
            "new_attempt_ceiling_usd": str(request.max_all_in_usd),
            "external_ledger_approval_id": external_id,
        })
        _atomic_local_json(path, ledger)
        return approval

    return _with_approval_ledger_lock(ledger_path, issue)


# reason: Signature, cost, nonce, profile, and cell checks produce one launch-approval verdict.
def verify_launch_approval(approval: object, *, request: BenchmarkRequest, purpose: str) -> dict[str, Any]:  # ruff: ignore[complex-structure,too-many-branches,too-many-statements]
    """Keyed on the caller's purpose.

    Never the approval's own claimed purpose, so a forged payload cannot choose which field set it is checked against.

    """
    if not is_str_mapping(approval):
        msg = "launch approval JSON is required"
        raise RuntimeError(msg)
    required = {
        "schema_version",
        "profile",
        "checkpoint_digest",
        "purpose",
        "launch_nonce",
        "external_ledger_approval_id",
        "modeled_prior_attempts",
        "new_attempt_ceiling_usd",
        "cumulative_modeled_ceiling_usd",
        "profile_allocation_usd",
        "issuer_key_id",
        "approval_digest",
        "signature",
    }
    if purpose == FILL_MISSING_CELLS_PURPOSE:
        required |= {"cells"}
    if set(approval) != required:
        msg = "launch approval has missing or unexpected fields"
        raise RuntimeError(msg)
    body = {key: approval[key] for key in required if key not in {"approval_digest", "signature"}}
    if approval.get("approval_digest") != canonical_sha256(body):
        msg = "launch approval digest is invalid"
        raise RuntimeError(msg)
    if approval.get("issuer_key_id") != APPROVAL_ISSUER_KEY_ID:
        msg = "launch approval issuer key ID is invalid"
        raise RuntimeError(msg)
    signature = approval.get("signature")
    if not isinstance(signature, str):
        msg = "launch approval signature is invalid"
        raise RuntimeError(msg)
    # reason: both names are referenced by the except clause below, so importing them inside the try
    # reason: made an ImportError raise NameError while handling it. They stay inside the function
    # reason: because cryptography is an optional extra, but above the try that uses them.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        signature_bytes = base64.b64decode(signature.encode("ascii"), validate=True)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(APPROVAL_PUBLIC_KEY_HEX)).verify(
            signature_bytes,
            canonical_json_bytes(body),
        )
    except (ValueError, TypeError, InvalidSignature) as error:
        msg = "launch approval signature is invalid"
        raise RuntimeError(msg) from error
    nonce = approval.get("launch_nonce")
    if not isinstance(nonce, str) or not is_sha256(nonce):
        msg = "launch approval nonce is invalid"
        raise RuntimeError(msg)
    if not isinstance(approval.get("external_ledger_approval_id"), str) or not approval["external_ledger_approval_id"]:
        msg = "launch approval lacks external-ledger approval"
        raise RuntimeError(msg)
    if (
        approval.get("profile") != request.profile
        or approval.get("checkpoint_digest") != request.checkpoint_digest
        or approval.get("purpose") != purpose
    ):
        msg = "launch approval does not bind this profile, checkpoint, and purpose"
        raise RuntimeError(msg)
    prior = approval.get("modeled_prior_attempts")
    if not isinstance(prior, list):
        msg = "launch approval prior modeled spend is invalid"
        raise RuntimeError(msg)
    ids: list[str] = []
    prior_total = Decimal(0)
    for entry in prior:
        if not isinstance(entry, Mapping) or set(entry) != {
            "modal_app_id",
            "modeled_all_in_cost_usd",
            "kind",
        }:
            msg = "launch approval prior attempt inventory is invalid"
            raise RuntimeError(msg)
        app_id = entry.get("modal_app_id")
        if not isinstance(app_id, str) or not app_id:
            msg = "launch approval prior attempt inventory is invalid"
            raise RuntimeError(msg)
        ids.append(app_id)
        prior_total += _decimal_usd(entry.get("modeled_all_in_cost_usd"), field="prior modeled spend")
    if len(ids) != len(set(ids)):
        msg = "launch approval repeats a prior app inventory entry"
        raise RuntimeError(msg)
    if (
        request.profile == "diffusionllm"
        and tuple(prior[: len(MODELED_PRIOR_DIFFUSION_ATTEMPTS)]) != MODELED_PRIOR_DIFFUSION_ATTEMPTS
    ):
        msg = "launch approval diffusion modeled-spend inventory drifted"
        raise RuntimeError(msg)
    new_ceiling = _decimal_usd(approval.get("new_attempt_ceiling_usd"), field="new attempt ceiling")
    cumulative = _decimal_usd(approval.get("cumulative_modeled_ceiling_usd"), field="cumulative ceiling")
    allocation = _decimal_usd(approval.get("profile_allocation_usd"), field="profile allocation")
    if (
        new_ceiling != Decimal(str(request.max_all_in_usd))
        or allocation != Decimal(str(PROFILE_ALL_IN_CEILING_USD[request.profile]))
        or cumulative != prior_total + new_ceiling
        or cumulative > allocation
    ):
        msg = "launch approval modeled-spend arithmetic or profile cap is invalid"
        raise RuntimeError(msg)
    if purpose == FILL_MISSING_CELLS_PURPOSE:
        try:
            canonical_cells = require_fill_cells(approval.get("cells"))
        except ValueError as error:
            msg = f"launch approval cell list is invalid: {error}"
            raise RuntimeError(msg) from error
        if list(canonical_cells) != approval.get("cells"):
            msg = "launch approval cell list is not in canonical matrix order"
            raise RuntimeError(msg)
    return dict(approval)


def activate_paid_lease_from_ledger(
    *,
    ledger_path: str | Path,
    request: BenchmarkRequest,
    purpose: str,
    approval: Mapping[str, Any],
) -> dict[str, Any]:
    """Hold the ledger lock through remote lease activation to close release races."""
    verified = verify_launch_approval(approval, request=request, purpose=purpose)

    def activate(path: Path) -> dict[str, Any]:
        ledger = _read_approval_ledger(path)
        _require_active_ledger_reservation(ledger, verified)
        return activate_launch_lease.remote(asdict(request), purpose, verified)

    return _with_approval_ledger_lock(ledger_path, activate)


def _close_approval_reservation(
    *,
    ledger_path: str | Path,
    approval_digest: str,
    status: str,
    settled_modeled_cost_usd: object | None = None,
) -> dict[str, Any]:
    if not is_sha256(approval_digest):
        msg = "approval digest is invalid"
        raise RuntimeError(msg)
    if status not in {"settled", "released"}:
        msg = "approval reservation status is invalid"
        raise RuntimeError(msg)

    def close(path: Path) -> dict[str, Any]:
        """Remote revocation is part of the state transition.

        If it fails, the local reservation remains active and cannot be misreported closed.

        """
        ledger = _read_approval_ledger(path)
        for reservation in ledger["reservations"]:
            if reservation["approval_digest"] != approval_digest:
                continue
            if reservation["status"] != "active":
                msg = "approval reservation is not active"
                raise RuntimeError(msg)
            revoke_launch_lease.remote(approval_digest)
            if status == "settled":
                cost = _decimal_usd(settled_modeled_cost_usd, field="settled modeled spend")
                if cost > _decimal_usd(reservation["new_attempt_ceiling_usd"], field="reservation ceiling"):
                    msg = "settled modeled spend exceeds the approved ceiling"
                    raise RuntimeError(msg)
                ledger["prior_modeled_attempts_by_profile"][reservation["profile"]].append({
                    "modal_app_id": reservation["external_ledger_approval_id"],
                    "modeled_all_in_cost_usd": str(cost),
                    "kind": "observed_modeled_cost",
                })
            reservation["status"] = status
            _atomic_local_json(path, ledger)
            return {"approval_digest": approval_digest, "status": status}
        msg = "approval reservation was not found"
        raise RuntimeError(msg)

    return _with_approval_ledger_lock(ledger_path, close)


def settle_launch_approval(ledger_path: str | Path, *, approval_digest: str, modeled_cost_usd: object) -> dict[str, Any]:
    return _close_approval_reservation(
        ledger_path=ledger_path,
        approval_digest=approval_digest,
        status="settled",
        settled_modeled_cost_usd=modeled_cost_usd,
    )


def release_launch_approval(ledger_path: str | Path, *, approval_digest: str) -> dict[str, Any]:
    return _close_approval_reservation(ledger_path=ledger_path, approval_digest=approval_digest, status="released")


def preflight_failure_root(request: BenchmarkRequest, *, purpose: str, approval: object) -> Path:
    attempt_digest = canonical_sha256({"request": asdict(request), "purpose": purpose, "approval": approval})
    return Path(BENCHMARK_MOUNT) / "preflight-failures" / attempt_digest


def launch_nonce_claim_path(approval: Mapping[str, Any]) -> Path:
    return Path(BENCHMARK_MOUNT) / "launch-nonce-claims" / str(approval["launch_nonce"]) / "claim.json"


def claim_launch_nonce(approval: Mapping[str, Any]) -> None:
    """Record the claim on the Volume as a post-claim audit trail.

    It has no admission role.

    """
    claim = {
        "approval_digest": approval["approval_digest"],
        "launch_nonce": approval["launch_nonce"],
        "profile": approval["profile"],
        "purpose": approval["purpose"],
    }
    try:
        claimed = nonce_claim_authority.put(claim["launch_nonce"], claim, skip_if_exists=True)
    except Exception as error:
        msg = "launch nonce authority is unavailable; refusing GPU admission"
        raise RuntimeError(msg) from error
    if claimed is not True:
        msg = "launch approval nonce was already claimed"
        raise DuplicateLaunchNonce(msg)
    path = launch_nonce_claim_path(approval)
    _atomic_json(
        path,
        {
            "status": "claimed",
            **claim,
            "external_ledger_approval_id": approval["external_ledger_approval_id"],
            "claim_authority": "modal_dict_put_skip_if_exists",
            "volume_record_role": "post_claim_audit_only",
        },
    )
    benchmark_volume.commit()


def _reload_benchmark_volume() -> None:
    """Refresh the Volume snapshot before a cross-container authorization read.

    Local unit-test doubles deliberately have no snapshot.

    """
    reload = getattr(benchmark_volume, "reload", None)
    if reload is None:
        return
    try:
        reload()
    except Exception as error:
        msg = "benchmark Volume reload failed before authorization read"
        raise RuntimeError(msg) from error


def launch_lease_path(approval_digest: str) -> Path:
    if not is_sha256(approval_digest):
        msg = "approval digest is invalid"
        raise RuntimeError(msg)
    return Path(BENCHMARK_MOUNT) / "launch-leases" / approval_digest / "lease.json"


def _read_launch_lease(approval_digest: str) -> dict[str, Any] | None:
    path = launch_lease_path(approval_digest)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = "launch approval lease is malformed"
        raise RuntimeError(msg) from error
    required = {"status", "approval_digest", "launch_nonce", "profile", "purpose"}
    # reason: Lease status, digest, nonce, profile, and purpose form one admission schema; helpers would scatter it.
    if (
        not isinstance(payload, Mapping)  # ruff: ignore[too-many-boolean-expressions]
        or set(payload) != required
        or payload.get("approval_digest") != approval_digest
        or payload.get("status") not in {"active", "released"}
        or not is_sha256(payload.get("launch_nonce"))
        or payload.get("profile") not in EVALUATION_PROFILES
        or not isinstance(payload.get("purpose"), str)
    ):
        msg = "launch approval lease is malformed"
        raise RuntimeError(msg)
    return dict(payload)


def persist_active_launch_lease(request: BenchmarkRequest, *, purpose: str, approval: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the remote, revocable lease before a GPU function can be dispatched."""
    verified = verify_launch_approval(approval, request=request, purpose=purpose)
    _reload_benchmark_volume()
    existing = _read_launch_lease(verified["approval_digest"])
    if existing is not None:
        msg = "launch approval lease was already activated or released"
        raise RuntimeError(msg)
    payload = {
        "status": "active",
        "approval_digest": verified["approval_digest"],
        "launch_nonce": verified["launch_nonce"],
        "profile": verified["profile"],
        "purpose": purpose,
    }
    _atomic_json(launch_lease_path(verified["approval_digest"]), payload)
    benchmark_volume.commit()
    return payload


def require_active_launch_lease(request: BenchmarkRequest, *, purpose: str, approval: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse GPU work unless the remote Volume says this signed approval is live."""
    verified = verify_launch_approval(approval, request=request, purpose=purpose)
    _reload_benchmark_volume()
    lease = _read_launch_lease(verified["approval_digest"])
    if lease is None or lease.get("status") != "active":
        msg = "launch approval lease is absent or revoked"
        raise RuntimeError(msg)
    if (
        lease["launch_nonce"] != verified["launch_nonce"]
        or lease["profile"] != request.profile
        or lease["purpose"] != purpose
    ):
        msg = "launch approval lease does not bind this launch"
        raise RuntimeError(msg)
    return lease


def persist_released_launch_lease(approval_digest: str) -> dict[str, Any]:
    """Write an irreversible remote revocation tombstone for a signed approval."""
    _reload_benchmark_volume()
    existing = _read_launch_lease(approval_digest)
    if existing is None:
        payload = {
            "status": "released",
            "approval_digest": approval_digest,
            "launch_nonce": "0" * 64,
            "profile": "diffusionllm",
            "purpose": "revoked_before_activation",
        }
    else:
        payload = {**existing, "status": "released"}
    _atomic_json(launch_lease_path(approval_digest), payload)
    benchmark_volume.commit()
    return payload


def _trajectory_control_root(trajectory_digest: str) -> Path:
    if not is_sha256(trajectory_digest):
        msg = "trajectory digest is invalid"
        raise ValueError(msg)
    return Path(BENCHMARK_MOUNT) / "trajectories" / trajectory_digest / "internal-control"


def _legacy_trajectory_history_path(trajectory_digest: str) -> Path:
    """Locate pre-generation history without making it eligible for a new run."""
    return _trajectory_control_root(trajectory_digest) / "history.json"


def _legacy_trajectory_decision_receipt_path(trajectory_digest: str) -> Path:
    return _trajectory_control_root(trajectory_digest) / "decision-receipt.json"


def _cadence_generation_root(trajectory_digest: str, stable_comparison_digest: str) -> Path:
    if not is_sha256(stable_comparison_digest):
        msg = "comparison contract digest is invalid"
        raise ValueError(msg)
    return _trajectory_control_root(trajectory_digest) / "generations" / stable_comparison_digest


def _trajectory_history_path(trajectory_digest: str, stable_comparison_digest: str) -> Path:
    return _cadence_generation_root(trajectory_digest, stable_comparison_digest) / "history.json"


def _trajectory_decision_receipt_path(trajectory_digest: str, stable_comparison_digest: str) -> Path:
    return _cadence_generation_root(trajectory_digest, stable_comparison_digest) / "decision-receipt.json"


def _read_legacy_durable_cadence_history(
    trajectory_digest: str,
) -> list[dict[str, Any]] | None:
    """Read the immutable pre-generation archive without admitting it to a run."""
    return _read_cadence_history_at_paths(
        _legacy_trajectory_history_path(trajectory_digest),
        _legacy_trajectory_decision_receipt_path(trajectory_digest),
        role="legacy durable cadence history",
    )


def _read_durable_cadence_history(trajectory_digest: str, stable_comparison_digest: str) -> list[dict[str, Any]] | None:
    return _read_cadence_history_at_paths(
        _trajectory_history_path(trajectory_digest, stable_comparison_digest),
        _trajectory_decision_receipt_path(trajectory_digest, stable_comparison_digest),
        role="durable cadence history",
    )


def _persist_cadence_decision(
    *,
    trajectory_digest: str,
    stable_comparison_digest: str,
    receipt: Mapping[str, Any],
) -> None:
    verified = verify_cadence_decision_receipt(receipt)
    if (
        verified["trajectory_digest"] != trajectory_digest
        or verified["comparison_contract_digest"] != stable_comparison_digest
    ):
        msg = "cadence receipt does not match its durable generation"
        raise ValueError(msg)
    _atomic_json(
        _trajectory_history_path(trajectory_digest, stable_comparison_digest),
        {"history": verified["history"]},
    )
    _atomic_json(
        _trajectory_decision_receipt_path(trajectory_digest, stable_comparison_digest),
        verified,
    )
    benchmark_volume.commit()


def checkpoint_output_root(request: BenchmarkRequest) -> Path:
    return Path(BENCHMARK_MOUNT) / "checkpoints" / request.checkpoint_digest


def evaluation_generation_root(request: BenchmarkRequest, evaluation_contract_digest: str) -> Path:
    """Namespace mutable evaluation outputs by their immutable evaluator identity."""
    if not is_sha256(evaluation_contract_digest):
        msg = "evaluation generation digest is invalid"
        raise ValueError(msg)
    return checkpoint_output_root(request) / "evaluation-generations" / evaluation_contract_digest


def render_full_benchmark_command(request: BenchmarkRequest, approval: Mapping[str, Any]) -> str:
    verify_launch_approval(approval, request=request, purpose="full_benchmark")
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(request.profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action full_benchmark --request-json {shlex.quote(_request_json(request))} "
        f"--approval-json {shlex.quote(json.dumps(dict(approval), sort_keys=True))}"
    )


def render_step60_baseline_command(request: BenchmarkRequest, approval: Mapping[str, Any]) -> str:
    verify_launch_approval(approval, request=request, purpose=STEP60_BASELINE_PURPOSE)
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(request.profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action step60_baseline --request-json {shlex.quote(_request_json(request))} "
        f"--approval-json {shlex.quote(json.dumps(dict(approval), sort_keys=True))}"
    )


def render_step60_partial_16of17_command(request: BenchmarkRequest, approval: Mapping[str, Any]) -> str:
    verify_launch_approval(approval, request=request, purpose=STEP60_PARTIAL_16OF17_PURPOSE)
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(request.profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action {STEP60_PARTIAL_16OF17_PURPOSE} "
        f"--request-json {shlex.quote(_request_json(request))} "
        f"--approval-json {shlex.quote(json.dumps(dict(approval), sort_keys=True))}"
    )


def render_checkpoint_partial_16of17_command(request: BenchmarkRequest, approval: Mapping[str, Any]) -> str:
    """Render the operator command line for a checkpoint partial 16-of-17 launch.

    Hold an issued Ed25519 approval for this purpose first; ``--action render_contract`` prints
    the free commands but deliberately withholds the paid ones. The rendered line carries the
    profile, the request and the approval, so it is the whole launch.

    Returns:
        The shell command, with every interpolated value quoted.

    """
    verify_launch_approval(approval, request=request, purpose=CHECKPOINT_PARTIAL_16OF17_PURPOSE)
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(request.profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action {CHECKPOINT_PARTIAL_16OF17_PURPOSE} "
        f"--request-json {shlex.quote(_request_json(request))} "
        f"--approval-json {shlex.quote(json.dumps(dict(approval), sort_keys=True))}"
    )


def render_internal_control_command(
    request: BenchmarkRequest,
    prior_history_json: str,
    approval: Mapping[str, Any],
) -> str:
    verify_launch_approval(approval, request=request, purpose="internal_control")
    try:
        prior_history = json.loads(prior_history_json)
    except json.JSONDecodeError as error:
        msg = "prior history JSON must be valid"
        raise ValueError(msg) from error
    _cadence_history(prior_history, role="prior cadence history")
    return (
        f"{PROFILE_ENVIRONMENT_KEY}={shlex.quote(request.profile)} "
        "uv run modal run scripts/ops/run_pii350_checkpoint_benchmark.py "
        f"--action internal_control --request-json {shlex.quote(_request_json(request))} "
        f"--prior-history-json {shlex.quote(prior_history_json)} "
        f"--approval-json {shlex.quote(json.dumps(dict(approval), sort_keys=True))}"
    )


def predict_next_cell_seconds(*, next_dataset: str, next_rows: int, observed_seconds_per_row: Sequence[float]) -> float:
    """Project the next cell from the median observed rate.

    The first cell pays one-time kernel-compile overhead, and a max-based projection would carry
    that outlier through the rest of the run. The signed ceiling and mid-cell stop still bound spend.

    Returns:
        The bootstrap allowance for the first bounded cell, otherwise the median rate with margin.

    Raises:
        ValueError: If a row count or observed per-row duration is not positive.
        RuntimeError: If the first cell is not the bounded bootstrap cell.

    """
    if next_rows <= 0:
        msg = "next cell rows must be positive"
        raise ValueError(msg)
    if not observed_seconds_per_row:
        if next_dataset != BOOTSTRAP_DATASET or next_rows > BOOTSTRAP_MAX_ROWS:
            msg = "first benchmark cell must use the bounded bootstrap cell"
            raise RuntimeError(msg)
        return BOOTSTRAP_CELL_SECONDS
    if any(seconds_per_row <= 0 for seconds_per_row in observed_seconds_per_row):
        msg = "observed seconds per row must be positive"
        raise ValueError(msg)
    return statistics.median(observed_seconds_per_row) * PREDICTED_CELL_MARGIN * next_rows


def budget_allows_next_cell(
    *,
    elapsed_seconds: float,
    next_dataset: str,
    next_rows: int,
    observed_seconds_per_row: Sequence[float],
    request: BenchmarkRequest,
) -> float:
    predicted_next_cell_seconds = predict_next_cell_seconds(
        next_dataset=next_dataset,
        next_rows=next_rows,
        observed_seconds_per_row=observed_seconds_per_row,
    )
    projected = (
        elapsed_seconds + predicted_next_cell_seconds + request.reserve_seconds_per_cell
    ) * A10G_ALL_IN_RATE_USD_PER_SECOND
    if projected > request.max_all_in_usd:
        msg = f"benchmark budget refuses next cell: projected=${projected:.4f} ceiling=${request.max_all_in_usd:.4f}"
        raise BudgetRefusal(
            msg,
        )
    return predicted_next_cell_seconds


def budget_ceiling_reached(*, elapsed_seconds: float, request: BenchmarkRequest) -> bool:
    return (elapsed_seconds + request.reserve_seconds_per_cell) * A10G_ALL_IN_RATE_USD_PER_SECOND >= request.max_all_in_usd


def internal_control_start_budget_seconds(*, elapsed_seconds: float, request: BenchmarkRequest) -> float:
    """Reserve bounded bootstrap, model load, inference, and persistence before load."""
    return budget_allows_next_cell(
        elapsed_seconds=elapsed_seconds,
        next_dataset=BOOTSTRAP_DATASET,
        next_rows=BOOTSTRAP_MAX_ROWS,
        observed_seconds_per_row=(),
        request=request,
    )


# reason: enforce internal exposes root/elapsed as its public contract; bundling would break callers.
def enforce_internal_control_progress_budget(  # ruff: ignore[too-many-arguments]
    *,
    root: Path,
    request: BenchmarkRequest,
    approval: Mapping[str, Any],
    done: int,
    total: int,
    elapsed_seconds: float,
) -> None:
    """Check revocation and all-in ceiling at each internal-control progress point."""
    require_active_launch_lease(request, purpose="internal_control", approval=approval)
    if budget_ceiling_reached(elapsed_seconds=elapsed_seconds, request=request):
        publish_budget_stop(
            root=root,
            request=request,
            dataset="v2-eval",
            rows_done=done,
            rows_total=total,
            elapsed_seconds=elapsed_seconds,
            predicted_next_cell_seconds=0.0,
        )
        msg = "internal control budget stop during inference"
        raise BudgetStop(msg)
    publish_progress(
        root=root,
        request=request,
        dataset="v2-eval",
        rows_done=done,
        rows_total=total,
    )


benchmark_volume = modal.Volume.from_name("meddies-pii350-checkpoint-benchmark", create_if_missing=True)
nonce_claim_authority = modal.Dict.from_name("meddies-pii350-checkpoint-nonce-claims", create_if_missing=True)
"""Modal Dict.put(..., skip_if_exists=True) is the cross-container compare-and-set authority. Volume files remain audit
evidence only; they are never admission.
"""
checkpoint_cache = modal.Volume.from_name("meddies-pii350-checkpoint-cache", create_if_missing=True)
huggingface_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
image = add_source_pythonpath(
    cast(
        "Any",
        use_pinned_debian_snapshot(
            modal.Image.from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"),
        )
        .apt_install(*TRITON_COMPILER_APT_PACKAGES)
        .pip_install(
            "torch==2.13.0",
            "transformers==5.11.0",
            "peft==0.19.1",
            "safetensors==0.8.0",
            "datasets==4.5.0",
            "pyarrow==23.0.1",
            "huggingface-hub==1.19.0",
            "cryptography==49.0.0",
        )
        .env({
            "HF_HOME": HF_CACHE_MOUNT,
            "HF_DATASETS_CACHE": f"{HF_CACHE_MOUNT}/datasets",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CC": TRITON_C_COMPILER,
        }),
    ),
).add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=None,
    cpu=1,
    memory=512,
    timeout=2 * 60,
    volumes={BENCHMARK_MOUNT: benchmark_volume},
)
def activate_launch_lease(
    request_payload: dict[str, Any],
    purpose: str,
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    """CPU-only authorization admission. It must finish before GPU dispatch."""
    return persist_active_launch_lease(BenchmarkRequest(**request_payload), purpose=purpose, approval=approval_payload)


@app.function(
    gpu=None,
    cpu=1,
    memory=512,
    timeout=2 * 60,
    volumes={BENCHMARK_MOUNT: benchmark_volume},
)
def revoke_launch_lease(approval_digest: str) -> dict[str, Any]:
    """CPU-only irreversible revocation for a locally released approval."""
    return persist_released_launch_lease(approval_digest)


def require_triton_c_compiler() -> str:
    """Fail before model loading when Triton cannot compile CUDA driver helpers."""
    configured = os.environ.get("CC", TRITON_C_COMPILER)
    compiler = str(Path(configured)) if Path(configured).is_absolute() else shutil.which(configured)
    if compiler is None or not os.access(compiler, os.X_OK):
        msg = f"Triton C compiler is unavailable: CC={configured!r}; expected executable {TRITON_C_COMPILER}"
        raise RuntimeError(msg)
    return compiler


def compile_triton_c_preflight() -> dict[str, str]:
    """Compile and link a throwaway C shared object with Triton's compiler."""
    compiler = require_triton_c_compiler()
    with tempfile.TemporaryDirectory(prefix="triton-compiler-preflight-") as directory:
        output = Path(directory) / "compiler-proof.so"
        command = [
            compiler,
            *TRITON_COMPILER_PROOF_ARGUMENTS[:-1],
            str(output),
        ]
        try:
            # reason: argv is a list, so no shell parses it. `compiler` comes from
            # reason: `require_triton_c_compiler()` and the rest are module constants — no caller input
            # reason: reaches this command, and the output path is inside a TemporaryDirectory.
            subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                command,
                input=TRITON_COMPILER_PROOF_SOURCE,
                text=True,
                capture_output=True,
                check=True,
                timeout=20,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            msg = "Triton C compiler failed to compile and link the preflight shared object"
            raise RuntimeError(msg) from error
        if not output.is_file() or output.stat().st_size <= 0:
            msg = "Triton C compiler did not produce a nonempty preflight shared object"
            raise RuntimeError(msg)
    proof = {
        "compiler": compiler,
        "source": TRITON_COMPILER_PROOF_SOURCE,
        "arguments": list(TRITON_COMPILER_PROOF_ARGUMENTS),
    }
    return {
        "cc": compiler,
        "proof": "c_shared_object_compile_link_v1",
        "proof_sha256": canonical_sha256(proof),
    }


@app.function(gpu=None, cpu=1, memory=512, timeout=2 * 60)
def triton_compiler_preflight() -> dict[str, str]:
    """Cheap CPU-only check of the exact compiler in the benchmark image."""
    payload = compile_triton_c_preflight()
    print("PII350_TRITON_COMPILER_PREFLIGHT::" + json.dumps(payload), flush=True)
    return payload


def publish_progress(
    *,
    root: Path,
    request: BenchmarkRequest,
    dataset: str,
    rows_done: int,
    rows_total: int,
) -> dict[str, Any]:
    payload = {
        "status": "running",
        "dataset": dataset,
        "rows_done": rows_done,
        "rows_total": rows_total,
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": request.trajectory_digest,
        "evaluation_generation_digest": evaluation_generation_digest_from_root(root),
    }
    _atomic_json(root / "progress" / f"{dataset}.json", payload)
    benchmark_volume.commit()
    print("PII350_BENCHMARK_PROGRESS::" + json.dumps(payload, sort_keys=True), flush=True)
    return payload


# reason: publish budget exposes root/prior as its public contract; bundling would break callers.
def publish_budget_stop(  # ruff: ignore[too-many-arguments]
    *,
    root: Path,
    request: BenchmarkRequest,
    dataset: str,
    rows_done: int,
    rows_total: int,
    elapsed_seconds: float,
    predicted_next_cell_seconds: float = 0.0,
    prior_completed_cell: str | None = None,
    prior_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "budget_stop",
        "dataset": dataset,
        "rows_done": rows_done,
        "rows_total": rows_total,
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": request.trajectory_digest,
        "evaluation_generation_digest": evaluation_generation_digest_from_root(root),
        "elapsed_seconds": elapsed_seconds,
        "predicted_next_cell_seconds": predicted_next_cell_seconds,
        "projected_all_in_cost_usd": (elapsed_seconds + predicted_next_cell_seconds + request.reserve_seconds_per_cell)
        * A10G_ALL_IN_RATE_USD_PER_SECOND,
        "ceiling_usd": request.max_all_in_usd,
        "persistence_reserve_seconds": request.reserve_seconds_per_cell,
        "prior_completed_cell": prior_completed_cell,
        "prior_progress": dict(prior_progress) if prior_progress is not None else None,
    }
    _atomic_json(root / "progress" / f"{dataset}.json", payload)
    _atomic_json(root / "budget_stop.json", payload)
    _atomic_json(root / "state.budget_stop.json", payload)
    benchmark_volume.commit()
    print(
        "PII350_BENCHMARK_BUDGET_STOP::" + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return payload


def _checkpoint_local_root(request: BenchmarkRequest) -> Path:
    root = (Path(CHECKPOINT_CACHE_MOUNT) / request.checkpoint_digest / request.artifact_path).resolve()
    if not root.is_dir():
        msg = "verified CPU checkpoint artifact is absent from cache volume"
        raise RuntimeError(msg)
    return root


def _receipt_path(request: BenchmarkRequest) -> Path:
    return checkpoint_output_root(request) / "receipt.json"


def _step60_reuse_receipt_path(request: BenchmarkRequest, evaluation_contract_digest: str | None = None) -> Path:
    if evaluation_contract_digest is None:
        return checkpoint_output_root(request) / "step60-reuse-receipt.json"
    return evaluation_generation_root(request, evaluation_contract_digest) / "step60-reuse-receipt.json"


def _cache_preflight_receipt_path(profile: str) -> Path:
    return Path(BENCHMARK_MOUNT) / "cache-preflight" / profile / "receipt.json"


def _cache_preflight_receipt(profile: str) -> Mapping[str, Any]:
    require_evaluation_profile(profile)
    receipt = _read_receipt(_cache_preflight_receipt_path(profile), role="CPU cache-preflight receipt")
    offline_cache = receipt.get("offline_cache")
    model = offline_cache.get("base_model") if isinstance(offline_cache, Mapping) else None
    datasets = offline_cache.get("datasets") if isinstance(offline_cache, Mapping) else None
    # reason: cache preflight keeps profile/id in one gate; helper predicates would scatter the rule.
    if (
        receipt.get("profile") != profile  # ruff: ignore[too-many-boolean-expressions]
        or not is_sha256(receipt.get("receipt_digest"))
        or not isinstance(offline_cache, Mapping)
        or not isinstance(model, Mapping)
        or model.get("id") != "LiquidAI/LFM2.5-Encoder-350M-PII-Detector"
        or model.get("revision") != "b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2"
        or not isinstance(model.get("snapshot_path"), str)
        or not Path(str(model["snapshot_path"])).is_dir()
        or not isinstance(datasets, Mapping)
        or datasets.get("v2_revision") != V2_DATASET_REVISION
        or datasets.get("external_revision") != EXTERNAL_DATASET_REVISION
        or datasets.get("rows_by_cell") != EVAL_EXPECTED_ROWS
        or datasets.get("total_rows") != FULL_ROWS
        or canonical_sha256({key: value for key, value in receipt.items() if key != "receipt_digest"})
        != receipt["receipt_digest"]
    ):
        msg = "CPU cache-preflight receipt does not satisfy the full matrix"
        raise RuntimeError(msg)
    return receipt


def _checkpoint_receipt(request: BenchmarkRequest) -> Mapping[str, Any]:
    receipt = _read_receipt(_receipt_path(request), role="CPU checkpoint receipt")
    cache_preflight = _cache_preflight_receipt(request.profile)
    if (
        receipt.get("checkpoint_digest") != request.checkpoint_digest
        or receipt.get("trajectory_digest") != request.trajectory_digest
        or receipt.get("revision") != request.revision
        or receipt.get("profile") != request.profile
        or receipt.get("cache_preflight_receipt_digest") != cache_preflight["receipt_digest"]
    ):
        msg = "CPU checkpoint receipt does not match the benchmark request"
        raise RuntimeError(msg)
    return receipt


def enable_offline_runtime(request: BenchmarkRequest) -> None:
    _checkpoint_receipt(request)
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })


def _cache_preflight_progress_path(profile: str) -> Path:
    return Path(BENCHMARK_MOUNT) / "cache-preflight" / profile / "progress.json"


def publish_cache_preflight_progress(*, profile: str, stage: str, cells_done: int, rows_done: int) -> None:
    payload = {
        "status": "running",
        "profile": profile,
        "stage": stage,
        "cells_done": cells_done,
        "cells_total": len(EVAL_DATASETS),
        "rows_done": rows_done,
        "rows_total": FULL_ROWS,
    }
    _atomic_json(_cache_preflight_progress_path(profile), payload)
    benchmark_volume.commit()
    print(
        "PII350_CACHE_PREFLIGHT_PROGRESS::" + json.dumps(payload, sort_keys=True),
        flush=True,
    )


def hydrate_offline_cache(profile: str) -> dict[str, Any]:
    """CPU-only preflight for every network-dependent GPU input."""
    from huggingface_hub import snapshot_download

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        BASE_MODEL_ID,
        BASE_MODEL_REVISION,
    )
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell

    publish_cache_preflight_progress(profile=profile, stage="base_model_start", cells_done=0, rows_done=0)
    snapshot_path = snapshot_download(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    publish_cache_preflight_progress(profile=profile, stage="base_model_ready", cells_done=0, rows_done=0)
    rows_by_cell: dict[str, int] = {}
    rows_done = 0
    for dataset in EVAL_DATASETS:
        rows = load_eval_cell(
            dataset,
            v2_limit=None,
            external_limit=None,
            v2_revision=V2_DATASET_REVISION,
            external_revision=EXTERNAL_DATASET_REVISION,
        )
        if len(rows) != EVAL_EXPECTED_ROWS[dataset]:
            msg = f"CPU cache hydration row drift for {dataset}: expected {EVAL_EXPECTED_ROWS[dataset]}, got {len(rows)}"
            raise RuntimeError(
                msg,
            )
        rows_by_cell[dataset] = len(rows)
        rows_done += len(rows)
        publish_cache_preflight_progress(
            profile=profile,
            stage=f"dataset:{dataset}",
            cells_done=len(rows_by_cell),
            rows_done=rows_done,
        )
    if sum(rows_by_cell.values()) != FULL_ROWS:
        msg = "CPU cache hydration total row count drifted"
        raise RuntimeError(msg)
    return {
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "snapshot_path": snapshot_path,
            "tokenizer_source": "same_pinned_base_snapshot",
        },
        "datasets": {
            "cache_path": f"{HF_CACHE_MOUNT}/datasets",
            "v2_revision": V2_DATASET_REVISION,
            "external_revision": EXTERNAL_DATASET_REVISION,
            "rows_by_cell": rows_by_cell,
            "total_rows": FULL_ROWS,
        },
    }


@app.function(
    gpu=None,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def cache_preflight(profile: str) -> dict[str, Any]:
    """CPU-only hydrate model, tokenizer, and all pinned matrix cells for one profile."""
    require_evaluation_profile(profile)
    offline_cache = hydrate_offline_cache(profile)
    body = {"schema_version": 1, "profile": profile, "offline_cache": offline_cache}
    receipt = {**body, "receipt_digest": canonical_sha256(body)}
    _atomic_json(_cache_preflight_receipt_path(profile), receipt)
    huggingface_cache.commit()
    benchmark_volume.commit()
    print("PII350_CACHE_PREFLIGHT::" + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


@app.function(
    gpu=None,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def checkpoint_receipt(request_payload: dict[str, Any]) -> dict[str, Any]:
    """CPU-only private-HF download and manifest gate; no model deserialization."""
    from huggingface_hub import snapshot_download

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        CheckpointArtifact,
        verify_checkpoint_artifact,
    )

    request = BenchmarkRequest(**request_payload)
    destination = Path(CHECKPOINT_CACHE_MOUNT) / request.checkpoint_digest
    snapshot_download(
        request.repo_id,
        revision=request.revision,
        local_dir=destination,
        allow_patterns=[f"{request.artifact_path}/**"],
    )
    root = destination / request.artifact_path
    verified = verify_checkpoint_artifact(
        root,
        CheckpointArtifact(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
        ),
    )
    require_expected_trajectory(request, verified.trajectory_digest)
    cache_preflight_receipt = _cache_preflight_receipt(request.profile)
    receipt = {
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": verified.trajectory_digest,
        "optimizer_step": verified.optimizer_step,
        "artifact_path": request.artifact_path,
        "repo_id": request.repo_id,
        "revision": request.revision,
        "profile": request.profile,
        "cache_preflight_receipt_digest": cache_preflight_receipt["receipt_digest"],
    }
    _atomic_json(_receipt_path(request), receipt)
    checkpoint_cache.commit()
    benchmark_volume.commit()
    print("PII350_CHECKPOINT_RECEIPT::" + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _hf_api() -> HfApi:
    from huggingface_hub import HfApi

    return HfApi()


def _hf_download(*, repo_id: str, path_in_repo: str, revision: str, destination: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        revision=revision,
        repo_type="model",
        local_dir=destination,
    )


def _download_transfer_file(*, path_in_repo: str, revision: str, destination: str) -> Path:
    """Treat any absent transfer object as a refusal, never as an empty import."""
    try:
        local = _hf_download(
            repo_id=RESULT_TRANSFER_REPO_ID,
            path_in_repo=path_in_repo,
            revision=revision,
            destination=destination,
        )
    except Exception as error:
        msg = f"result transfer object is absent from the pinned revision: {path_in_repo}"
        raise RuntimeError(msg) from error
    return Path(local)


@app.function(
    gpu=None,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BENCHMARK_MOUNT: benchmark_volume},
)
def export_cell_results(request_payload: dict[str, Any], generation_digest: str) -> dict[str, Any]:
    """CPU-only manifest-last export of completed cells from the source workspace.

    The manifest commits last, so a partially uploaded transfer can never be read as a complete one; the returned commit is
    the only importable revision.

    """
    request = BenchmarkRequest(**request_payload)
    root = evaluation_generation_root(request, generation_digest)
    manifest = _result_transfer_manifest(request=request, generation_digest=generation_digest, root=root)
    prefix = _result_transfer_prefix(request.checkpoint_digest, generation_digest)
    api = _hf_api()
    for entry in manifest["files"]:
        api.upload_file(
            path_or_fileobj=str(root / str(entry["path"])),
            path_in_repo=f"{prefix}/{RESULT_TRANSFER_FILES_PREFIX}/{entry['path']}",
            repo_id=RESULT_TRANSFER_REPO_ID,
            repo_type="model",
            commit_message=f"pii350 result transfer cell file {entry['path']}",
        )
    commit = api.upload_file(
        path_or_fileobj=canonical_json_bytes(manifest),
        path_in_repo=f"{prefix}/{RESULT_TRANSFER_MANIFEST_FILENAME}",
        repo_id=RESULT_TRANSFER_REPO_ID,
        repo_type="model",
        commit_message=(f"pii350 result transfer manifest {manifest['manifest_digest']}"),
    )
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    body |= {
        "manifest_digest": manifest["manifest_digest"],
        "repo_id": RESULT_TRANSFER_REPO_ID,
        "repo_prefix": prefix,
        "hf_commit_sha": _require_commit_sha(commit),
    }
    receipt = {**body, "receipt_digest": canonical_sha256(body)}
    print(
        "PII350_RESULT_TRANSFER_EXPORT::" + json.dumps(receipt, sort_keys=True),
        flush=True,
    )
    return receipt


@app.function(
    gpu=None,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={BENCHMARK_MOUNT: benchmark_volume},
)
def cell_score_summary(request_payload: dict[str, Any], generation_digest: str) -> dict[str, Any]:
    """CPU-only per-cell counts for one generation; no GPU and no model load."""
    request = BenchmarkRequest(**request_payload)
    root = evaluation_generation_root(request, generation_digest)
    model = checkpoint_model_name(request)
    cells: dict[str, Any] = {}
    for dataset in EVAL_DATASETS:
        directory = root / "results" / model / dataset
        result_path = directory / f"{RESULT_SHARD}.jsonl"
        done_path = directory / f"{RESULT_SHARD}.done"
        meta_path = directory / f"{RESULT_SHARD}.meta.json"
        if not (result_path.is_file() and done_path.is_file() and meta_path.is_file()):
            continue
        sidecar = _read_receipt(meta_path, role=f"{dataset} result sidecar")
        observed_digest = file_sha256(str(result_path))
        if sidecar.get("result_sha256") != observed_digest:
            msg = f"result cell {dataset} disagrees with its sidecar result digest"
            raise RuntimeError(msg)
        if sidecar.get("evaluation_contract_sha256") != generation_digest:
            msg = f"result cell {dataset} belongs to a different evaluation generation"
            raise RuntimeError(msg)
        scored = _score_cell_rows(result_path)
        if scored["rows"] != EVAL_EXPECTED_ROWS[dataset]:
            msg = f"result cell {dataset} row count drift: expected {EVAL_EXPECTED_ROWS[dataset]}, scored {scored['rows']}"
            raise RuntimeError(
                msg,
            )
        cells[dataset] = {
            "dataset": dataset,
            "result_sha256": observed_digest,
            "fixture_sha256": sidecar.get("fixture_sha256"),
            "evaluation_contract_sha256": generation_digest,
            "dataset_shard_identity_sha256": sidecar.get("dataset_shard_identity_sha256"),
            "elapsed_seconds": sidecar.get("elapsed_seconds"),
            **scored,
        }
    if not cells:
        msg = "evaluation generation has no completed result cell to score"
        raise RuntimeError(msg)
    body = {
        "schema_version": CELL_SCORE_SUMMARY_SCHEMA_VERSION,
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": request.trajectory_digest,
        "evaluation_generation_digest": generation_digest,
        "model": model,
        "profile": request.profile,
        "cells": cells,
    }
    summary = {**body, "summary_digest": canonical_sha256(body)}
    print("PII350_CELL_SCORE_SUMMARY::" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.function(
    gpu=None,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={BENCHMARK_MOUNT: benchmark_volume},
)
def import_cell_results(request_payload: dict[str, Any], receipt_payload: dict[str, Any], revision: str) -> dict[str, Any]:  # ruff: ignore[too-many-locals]
    """CPU-only verified import into the destination workspace benchmark Volume.

    Per cell the completion marker is written last, so an interrupted import never leaves a cell that reads as done without
    its verified bytes.

    """
    request = BenchmarkRequest(**request_payload)
    receipt = _validated_transfer_receipt(receipt_payload, request=request)
    if revision != receipt["hf_commit_sha"]:
        msg = "result transfer revision is not the immutable manifest commit of this receipt"
        raise RuntimeError(msg)
    generation_digest = str(receipt["evaluation_generation_digest"])
    model = str(receipt["model"])
    prefix = str(receipt["repo_prefix"])
    expected_manifest_body = {
        "schema_version": receipt["schema_version"],
        "checkpoint_digest": receipt["checkpoint_digest"],
        "trajectory_digest": receipt["trajectory_digest"],
        "evaluation_generation_digest": generation_digest,
        "model": model,
        "source_profile": receipt["source_profile"],
        "cells": receipt["cells"],
        "files": receipt["files"],
        "total_bytes": receipt["total_bytes"],
    }
    expected_manifest = {
        **expected_manifest_body,
        "manifest_digest": canonical_sha256(expected_manifest_body),
    }
    if expected_manifest["manifest_digest"] != receipt["manifest_digest"]:
        msg = "result transfer receipt disagrees with its own manifest digest"
        raise RuntimeError(msg)
    root = evaluation_generation_root(request, generation_digest)
    written: list[str] = []
    already_present: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pii350-result-transfer-") as directory:
        manifest_path = _download_transfer_file(
            path_in_repo=f"{prefix}/{RESULT_TRANSFER_MANIFEST_FILENAME}",
            revision=revision,
            destination=directory,
        )
        downloaded_manifest = _read_receipt(manifest_path, role="result transfer manifest")
        if dict(downloaded_manifest) != expected_manifest:
            msg = "result transfer manifest does not match the export receipt"
            raise RuntimeError(msg)
        payload_by_path: dict[str, bytes] = {}
        for entry in receipt["files"]:
            relative = str(entry["path"])
            local = _download_transfer_file(
                path_in_repo=f"{prefix}/{RESULT_TRANSFER_FILES_PREFIX}/{relative}",
                revision=revision,
                destination=directory,
            )
            payload = local.read_bytes()
            if len(payload) != int(entry["bytes"]) or file_sha256(str(local)) != entry["sha256"]:
                msg = f"result transfer file failed hash verification: {relative}"
                raise RuntimeError(msg)
            payload_by_path[relative] = payload
        for dataset in receipt["cells"]:
            for relative in _result_cell_relative_paths(model, str(dataset)):
                destination = root / relative
                payload = payload_by_path[relative]
                if destination.exists():
                    if destination.read_bytes() != payload:
                        msg = f"destination result file already exists with different bytes: {relative}"
                        raise RuntimeError(msg)
                    already_present.append(relative)
                    continue
                _atomic_bytes(destination, payload)
                written.append(relative)
    benchmark_volume.commit()
    body = {
        "schema_version": RESULT_TRANSFER_SCHEMA_VERSION,
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": request.trajectory_digest,
        "evaluation_generation_digest": generation_digest,
        "model": model,
        "destination_profile": request.profile,
        "cells": receipt["cells"],
        "files": receipt["files"],
        "total_bytes": receipt["total_bytes"],
        "manifest_digest": receipt["manifest_digest"],
        "repo_id": RESULT_TRANSFER_REPO_ID,
        "repo_prefix": prefix,
        "hf_commit_sha": revision,
        "source_receipt_digest": receipt["receipt_digest"],
        "written_files": written,
        "already_present_files": already_present,
    }
    imported = {**body, "receipt_digest": canonical_sha256(body)}
    print(
        "PII350_RESULT_TRANSFER_IMPORT::" + json.dumps(imported, sort_keys=True),
        flush=True,
    )
    return imported


def _evaluation_contract(request: BenchmarkRequest, trajectory_digest: str) -> EvaluationContract:
    from meddies_pii import __file__ as package_file
    from meddies_pii.eval_baseline.adapters import pii350_checkpoint
    from meddies_pii.eval_baseline.baseline.run import resolved_environment_fingerprint
    from meddies_pii.evaluation.identity import evaluation_contract, payload_artifact
    from meddies_pii.evaluation.runtime_provenance import (
        evaluation_runtime_source_artifact,
    )

    if package_file is None:
        msg = "cannot observe evaluation source"
        raise RuntimeError(msg)
    source = evaluation_runtime_source_artifact(Path(package_file).parent, __file__)
    return evaluation_contract(
        model=payload_artifact(
            "pii350-checkpoint",
            {
                "checkpoint_digest": request.checkpoint_digest,
                "trajectory_digest": trajectory_digest,
                "expected_trajectory_digest": request.trajectory_digest,
                "repo_id": request.repo_id,
                "revision": request.revision,
            },
        ),
        vendor_inference_source=payload_artifact(
            "standard-peft-unsloth-compatible-loader",
            {
                "base": pii350_checkpoint.BASE_MODEL_ID,
                "revision": pii350_checkpoint.BASE_MODEL_REVISION,
            },
        ),
        local_adapter_source=source,
        applied_label_prediction_contract=payload_artifact("pii350-label-prediction", "meddies-37-bioes-full-nine"),
        decoder_contract=payload_artifact("pii350-decoder", "viterbi-bioes-char-offset-v1"),
        resolved_runtime_environment=payload_artifact("resolved-runtime", resolved_environment_fingerprint()),
        scorer_contract=source,
        supported_labels=tuple(sorted(PII_LABEL_SET)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


def _persist_step60_reuse_receipt(*, request: BenchmarkRequest, body: Mapping[str, Any]) -> dict[str, Any]:
    receipt = {**body, "receipt_digest": canonical_sha256(body)}
    _atomic_json(
        _step60_reuse_receipt_path(request, str(body["evaluation_contract_digest"])),
        receipt,
    )
    benchmark_volume.commit()
    print(
        "PII350_STEP60_REUSE_RECEIPT::" + json.dumps(receipt, sort_keys=True),
        flush=True,
    )
    return receipt


def _require_step60_reuse_receipt(
    *,
    request: BenchmarkRequest,
    expected_receipt_digest: str,
    body: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not is_sha256(expected_receipt_digest):
        msg = "step-60 baseline requires a CPU reuse receipt digest"
        raise RuntimeError(msg)
    receipt = _read_receipt(
        _step60_reuse_receipt_path(request, str(body["evaluation_contract_digest"])),
        role="CPU step-60 reuse receipt",
    )
    expected = {**body, "receipt_digest": canonical_sha256(body)}
    if receipt != expected or receipt["receipt_digest"] != expected_receipt_digest:
        msg = "CPU step-60 reuse receipt does not match checkpoint, evaluation, result bytes, and scored rows"
        raise RuntimeError(
            msg,
        )
    return receipt


@app.function(
    gpu=None,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def step60_reuse_receipt(request_payload: dict[str, Any]) -> dict[str, Any]:
    """CPU-only proof that the user-excluded challenge will never be rerun."""
    request = BenchmarkRequest(**request_payload)
    enable_offline_runtime(request)
    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        CheckpointArtifact,
        verify_checkpoint_artifact,
    )
    from meddies_pii.eval_baseline.baseline.aggregate import aggregate_results
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
    from meddies_pii.eval_baseline.baseline.run import read_matrix_results
    from meddies_pii.evaluation.identity import dataset_shard_identity

    contract = benchmark_contract(request, purpose=STEP60_BASELINE_PURPOSE)
    require_step60_baseline_matrix(contract)
    verified = verify_checkpoint_artifact(
        _checkpoint_local_root(request),
        CheckpointArtifact(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
        ),
    )
    require_expected_trajectory(request, verified.trajectory_digest)
    require_step60_checkpoint(verified.optimizer_step)
    evaluation = _evaluation_contract(request, verified.trajectory_digest)
    identities: dict[str, Any] = {}
    for dataset in STEP60_REUSED_CELLS:
        rows = load_eval_cell(
            dataset,
            v2_limit=None,
            external_limit=None,
            v2_revision=V2_DATASET_REVISION,
            external_revision=EXTERNAL_DATASET_REVISION,
        )
        if len(rows) != EVAL_EXPECTED_ROWS[dataset]:
            msg = f"{dataset} row count drift: expected {EVAL_EXPECTED_ROWS[dataset]}, got {len(rows)}"
            raise RuntimeError(msg)
        identities[dataset] = dataset_shard_identity(evaluation, dataset=dataset, shard="full", rows=rows)
    matrix = read_matrix_results(
        evaluation_generation_root(request, evaluation.digest),
        checkpoint_model_name(request),
        STEP60_REUSED_CELLS,
        require_done=True,
        expected_rows={dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in STEP60_REUSED_CELLS},
        expected_evaluation_contract=evaluation,
        expected_dataset_shard_identities=identities,
    )
    if "v2-eval-challenge" in matrix.missing_datasets:
        msg = "step-60 baseline cannot prove existing v2-eval-challenge artifact identity; refusing user-excluded rerun"
        raise RuntimeError(
            msg,
        )
    scored_rows = matrix.rows_by_dataset
    report = aggregate_results(
        scored_rows,
        supported_labels=frozenset(PII_LABEL_SET),
        expected_evaluation_contract=evaluation,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
        timing_by_config=matrix.timing_by_dataset,
    )
    body = _step60_reuse_receipt_body(
        request=request,
        optimizer_step=verified.optimizer_step,
        contract_digest=contract["contract_digest"],
        evaluation_digest=evaluation.digest,
        shard_identity_digest_by_cell={
            dataset: matrix.shard_identities_by_dataset[dataset].digest for dataset in scored_rows
        },
        result_sha256_by_cell=matrix.result_sha256_by_dataset,
        scored_report=report,
    )
    return _persist_step60_reuse_receipt(request=request, body=body)


# reason: verify and verify share run complete's state; extraction would split cleanup from writes.
def _run_complete_matrix(  # ruff: ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
    request_payload: dict[str, Any],
    expected_checkpoint_digest: str,
    *,
    purpose: str,
    approval_payload: dict[str, Any] | None = None,
    expected_step60_reuse_receipt_digest: str | None = None,
) -> dict[str, Any]:
    """One workspace, one A10G, and one verified fixed matrix.

    Re-open and score the reusable result rows before paid inference. This proves the sidecar hash and evaluation identity
    bind valid rows, not only an existing file at the expected path.

    A reused v2 result has no new GPU timing. Seed subsequent budget refusals from the same conservative bootstrap
    allowance instead.

    """
    started = time.monotonic()
    request = BenchmarkRequest(**request_payload)
    if request.checkpoint_digest != expected_checkpoint_digest:
        msg = "caller checkpoint digest does not match benchmark request"
        raise RuntimeError(msg)
    root = preflight_failure_root(request, purpose=purpose, approval=approval_payload)
    last_completed_cell: str | None = None
    last_progress: Mapping[str, Any] | None = None
    resume_state: Mapping[str, Any] | None = None
    aggregate_only = False
    # reason: run complete's try keeps verify with verify; splitting would split cleanup from writes.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        approval = verify_launch_approval(approval_payload, request=request, purpose=purpose)
        require_active_launch_lease(request, purpose=purpose, approval=approval)
        claim_launch_nonce(approval)
        enable_offline_runtime(request)
        require_triton_c_compiler()
        import torch

        from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
            CheckpointArtifact,
            Pii350CheckpointAdapter,
            verify_checkpoint_artifact,
        )
        from meddies_pii.eval_baseline.baseline.aggregate import aggregate_results
        from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
        from meddies_pii.eval_baseline.baseline.run import (
            ShardSpec,
            assert_frozen_fixture,
            expected_matrix_shard_identities,
            read_matrix_results,
            run_shard,
        )
        from meddies_pii.evaluation.identity import dataset_shard_identity

        contract = benchmark_contract(request, purpose=purpose)
        if purpose == "full_benchmark":
            if expected_step60_reuse_receipt_digest is not None:
                msg = "full benchmark must not accept a step-60 reuse receipt"
                raise RuntimeError(msg)
            require_full_matrix(contract)
            scheduled_datasets = EVAL_DATASETS
        elif purpose == STEP60_BASELINE_PURPOSE:
            require_step60_baseline_matrix(contract)
            if expected_step60_reuse_receipt_digest is None:
                msg = "step-60 baseline requires a CPU reuse receipt digest"
                raise RuntimeError(msg)
            scheduled_datasets = EVAL_DATASETS
        elif purpose in {
            STEP60_PARTIAL_16OF17_PURPOSE,
            CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        }:
            if expected_step60_reuse_receipt_digest is not None:
                msg = "checkpoint partial 16/17 benchmark cannot use reuse receipt"
                raise RuntimeError(msg)
            if purpose == STEP60_PARTIAL_16OF17_PURPOSE:
                require_step60_partial_16of17_matrix(contract)
            else:
                require_checkpoint_partial_16of17_matrix(contract)
            scheduled_datasets = STEP60_PARTIAL_16OF17_DATASETS
        else:
            msg = "complete matrix purpose is invalid"
            raise RuntimeError(msg)
        verified = verify_checkpoint_artifact(
            _checkpoint_local_root(request),
            CheckpointArtifact(
                request.repo_id,
                request.revision,
                request.artifact_path,
                request.checkpoint_digest,
            ),
        )
        require_expected_trajectory(request, verified.trajectory_digest)
        if purpose in {STEP60_BASELINE_PURPOSE, STEP60_PARTIAL_16OF17_PURPOSE}:
            require_step60_checkpoint(verified.optimizer_step)
        elif purpose == CHECKPOINT_PARTIAL_16OF17_PURPOSE:
            require_checkpoint_partial_checkpoint(request, verified)
        evaluation = _evaluation_contract(request, verified.trajectory_digest)
        root = evaluation_generation_root(request, evaluation.digest)
        _atomic_json(
            root / "state.json",
            {
                "status": "startup",
                "contract": contract,
                "trajectory_digest": verified.trajectory_digest,
                "evaluation_generation_digest": evaluation.digest,
            },
        )
        model = checkpoint_model_name(request)
        reusable_rows: dict[str, Any] = {}
        prevalidated_reused_cells: set[str] = set()
        scheduled_identities: dict[str, Any] | None = None
        if purpose == STEP60_BASELINE_PURPOSE:
            preexisting_identities: dict[str, Any] = {}
            for dataset in STEP60_REUSED_CELLS:
                rows = load_eval_cell(
                    dataset,
                    v2_limit=None,
                    external_limit=None,
                    v2_revision=V2_DATASET_REVISION,
                    external_revision=EXTERNAL_DATASET_REVISION,
                )
                if len(rows) != EVAL_EXPECTED_ROWS[dataset]:
                    msg = f"{dataset} row count drift: expected {EVAL_EXPECTED_ROWS[dataset]}, got {len(rows)}"
                    raise RuntimeError(
                        msg,
                    )
                reusable_rows[dataset] = rows
                preexisting_identities[dataset] = dataset_shard_identity(
                    evaluation,
                    dataset=dataset,
                    shard="full",
                    rows=rows,
                )
            preexisting = read_matrix_results(
                root,
                model,
                STEP60_REUSED_CELLS,
                require_done=True,
                expected_rows={dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in STEP60_REUSED_CELLS},
                expected_evaluation_contract=evaluation,
                expected_dataset_shard_identities=preexisting_identities,
            )
            if "v2-eval-challenge" in preexisting.missing_datasets:
                msg = (
                    "step-60 baseline cannot prove existing v2-eval-challenge "
                    "artifact identity; refusing user-excluded rerun"
                )
                raise RuntimeError(msg)
            reused_report = aggregate_results(
                preexisting.rows_by_dataset,
                supported_labels=frozenset(PII_LABEL_SET),
                expected_evaluation_contract=evaluation,
                expected_dataset_shard_identities=preexisting.shard_identities_by_dataset,
                result_sha256_by_config=preexisting.result_sha256_by_dataset,
                timing_by_config=preexisting.timing_by_dataset,
            )
            reuse_receipt_body = _step60_reuse_receipt_body(
                request=request,
                optimizer_step=verified.optimizer_step,
                contract_digest=contract["contract_digest"],
                evaluation_digest=evaluation.digest,
                shard_identity_digest_by_cell={
                    dataset: preexisting.shard_identities_by_dataset[dataset].digest
                    for dataset in preexisting.rows_by_dataset
                },
                result_sha256_by_cell=preexisting.result_sha256_by_dataset,
                scored_report=reused_report,
            )
            _require_step60_reuse_receipt(
                request=request,
                # reason: the step-60 branch above already raised when this is None; ty drops that narrowing here.
                expected_receipt_digest=expected_step60_reuse_receipt_digest,  # ty: ignore[invalid-argument-type]
                body=reuse_receipt_body,
            )
            prevalidated_reused_cells = set(preexisting.rows_by_dataset)
        elif purpose in {
            STEP60_PARTIAL_16OF17_PURPOSE,
            CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        }:
            scheduled_identities = expected_matrix_shard_identities(evaluation, scheduled_datasets)
            preexisting = read_matrix_results(
                root,
                model,
                scheduled_datasets,
                require_done=True,
                expected_rows={dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in scheduled_datasets},
                expected_evaluation_contract=evaluation,
                expected_dataset_shard_identities=scheduled_identities,
            )
            prevalidated_reused_cells = set(preexisting.rows_by_dataset)
            partial_matrix_state = require_complete_or_single_missing_partial_matrix(
                scheduled_datasets=scheduled_datasets,
                validated_completed_cells=tuple(
                    dataset for dataset in scheduled_datasets if dataset in prevalidated_reused_cells
                ),
                missing_cells=preexisting.missing_datasets,
                elapsed_seconds=time.monotonic() - started,
                request=request,
            )
            aggregate_only = partial_matrix_state["mode"] == "aggregate_only"
            if partial_matrix_state["mode"] == "single_missing_cell_resume":
                resume_state = partial_matrix_state
                _atomic_json(
                    root / "resume.json",
                    {
                        "status": "resume_ready",
                        "checkpoint_digest": request.checkpoint_digest,
                        "trajectory_digest": verified.trajectory_digest,
                        "evaluation_generation_digest": evaluation.digest,
                        **resume_state,
                    },
                )
                benchmark_volume.commit()
        adapter: Any | None = None
        if not aggregate_only:
            adapter = Pii350CheckpointAdapter(verified)
            adapter.load()
            _atomic_json(
                root / "state.json",
                {
                    "status": "model_loaded",
                    "contract_digest": contract["contract_digest"],
                    "checkpoint_digest": request.checkpoint_digest,
                    "evaluation_generation_digest": evaluation.digest,
                },
            )
        else:
            _atomic_json(
                root / "state.json",
                {
                    "status": "aggregate_ready",
                    "contract_digest": contract["contract_digest"],
                    "checkpoint_digest": request.checkpoint_digest,
                    "evaluation_generation_digest": evaluation.digest,
                },
            )
            benchmark_volume.commit()
        cost_ledger: list[dict[str, Any]] = []
        observed_seconds_per_row: list[float] = []
        v2_eval_reused = False
        for dataset in () if aggregate_only else scheduled_datasets:
            elapsed_before_dispatch = time.monotonic() - started
            identity_validated_reuse = dataset in prevalidated_reused_cells
            if identity_validated_reuse:
                predicted_next_cell_seconds = 0.0
            elif resume_state is not None:
                predicted_next_cell_seconds = float(resume_state["progress_window_seconds"])
            else:
                try:
                    predicted_next_cell_seconds = budget_allows_next_cell(
                        elapsed_seconds=elapsed_before_dispatch,
                        next_dataset=dataset,
                        next_rows=EVAL_EXPECTED_ROWS[dataset],
                        observed_seconds_per_row=observed_seconds_per_row,
                        request=request,
                    )
                except BudgetRefusal as error:
                    last_progress = publish_budget_stop(
                        root=root,
                        request=request,
                        dataset=dataset,
                        rows_done=0,
                        rows_total=EVAL_EXPECTED_ROWS[dataset],
                        elapsed_seconds=elapsed_before_dispatch,
                        predicted_next_cell_seconds=predict_next_cell_seconds(
                            next_dataset=dataset,
                            next_rows=EVAL_EXPECTED_ROWS[dataset],
                            observed_seconds_per_row=observed_seconds_per_row,
                        ),
                        prior_completed_cell=last_completed_cell,
                        prior_progress=last_progress,
                    )
                    msg = "benchmark budget stop before dispatch"
                    raise BudgetStop(msg) from error
            rows = cast("Any", reusable_rows.get(dataset))
            if rows is None:
                rows = load_eval_cell(
                    dataset,
                    v2_limit=None,
                    external_limit=None,
                    v2_revision=V2_DATASET_REVISION,
                    external_revision=EXTERNAL_DATASET_REVISION,
                )
            if len(rows) != EVAL_EXPECTED_ROWS[dataset]:
                msg = f"{dataset} row count drift: expected {EVAL_EXPECTED_ROWS[dataset]}, got {len(rows)}"
                raise RuntimeError(msg)
            cell_started = time.monotonic()

            # reason: `cell` is already captured by default argument; `prior_cell` closes the same
            # reason: hazard for the accumulator beside it. The write `last_completed_cell = dataset`
            # reason: sits at the END of this loop body, so the captured value equals the late-bound
            # reason: one at every current call — but a reordering of that write would make this
            # reason: callback report the CURRENT cell as the PRIOR one inside budget-stop evidence.
            def progress(
                done: int,
                total: int,
                *,
                cell: str = dataset,
                prior_cell: str | None = last_completed_cell,
            ) -> None:
                nonlocal last_progress
                if done % PROGRESS_EVERY != 0 and done != total:
                    return
                elapsed_seconds = time.monotonic() - started
                require_active_launch_lease(request, purpose=purpose, approval=approval)
                if budget_ceiling_reached(elapsed_seconds=elapsed_seconds, request=request):
                    last_progress = publish_budget_stop(
                        root=root,
                        request=request,
                        dataset=cell,
                        rows_done=done,
                        rows_total=total,
                        elapsed_seconds=elapsed_seconds,
                        prior_completed_cell=prior_cell,
                        prior_progress=last_progress,
                    )
                    msg = "benchmark budget stop reached inside cell"
                    raise BudgetStop(msg)
                last_progress = publish_progress(
                    root=root,
                    request=request,
                    dataset=cell,
                    rows_done=done,
                    rows_total=total,
                )

            if identity_validated_reuse and purpose in {
                STEP60_PARTIAL_16OF17_PURPOSE,
                CHECKPOINT_PARTIAL_16OF17_PURPOSE,
            }:
                # reason: aggregate_only makes this loop iterate `()`, so adapter is never None inside it.
                adapter.on_progress = progress  # ty: ignore[invalid-assignment]
                result = run_shard(
                    adapter=adapter,  # ty: ignore[invalid-argument-type]
                    rows=rows,
                    output_root=root,
                    spec=ShardSpec(model=model, dataset=dataset, shard="full"),
                    evaluation_contract=evaluation,
                    volume=benchmark_volume,
                    force=False,
                )
                if not result.skipped:
                    msg = "identity-validated resume shard unexpectedly required inference"
                    raise RuntimeError(msg)
                skipped = True
            elif identity_validated_reuse:
                skipped = True
            else:
                # reason: aggregate_only makes this loop iterate `()`, so adapter is never None inside it.
                adapter.on_progress = progress  # ty: ignore[invalid-assignment]
                result = run_shard(
                    adapter=adapter,  # ty: ignore[invalid-argument-type]
                    rows=rows,
                    output_root=root,
                    spec=ShardSpec(model=model, dataset=dataset, shard="full"),
                    evaluation_contract=evaluation,
                    volume=benchmark_volume,
                    force=False,
                )
                skipped = result.skipped
            if dataset == "v2-eval":
                v2_eval_reused = skipped
            cell_elapsed = time.monotonic() - cell_started
            if skipped and dataset == BOOTSTRAP_DATASET and not identity_validated_reuse:
                observed_seconds_per_row.append(BOOTSTRAP_CELL_SECONDS / EVAL_EXPECTED_ROWS[BOOTSTRAP_DATASET])
            elif not skipped and cell_elapsed > 0:
                observed_seconds_per_row.append(cell_elapsed / len(rows))
            record = {
                "dataset": dataset,
                "rows": len(rows),
                "elapsed_seconds": cell_elapsed,
                "predicted_next_cell_seconds": predicted_next_cell_seconds,
                "estimated_all_in_cost_usd": (time.monotonic() - started) * A10G_ALL_IN_RATE_USD_PER_SECOND,
                "evaluation_generation_digest": evaluation.digest,
                "skipped": skipped,
            }
            cost_ledger.append(record)
            last_completed_cell = dataset
            _atomic_json(
                root / "progress" / f"{dataset}.json",
                {"status": "cell_terminal", **record},
            )
            benchmark_volume.commit()
        identities = scheduled_identities or expected_matrix_shard_identities(evaluation, scheduled_datasets)
        matrix = read_matrix_results(
            root,
            model,
            scheduled_datasets,
            require_done=True,
            expected_rows={dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in scheduled_datasets},
            expected_evaluation_contract=evaluation,
            expected_dataset_shard_identities=identities,
        )
        if matrix.missing_datasets:
            matrix_name = (
                "checkpoint partial 16/17 matrix"
                if purpose
                in {
                    STEP60_PARTIAL_16OF17_PURPOSE,
                    CHECKPOINT_PARTIAL_16OF17_PURPOSE,
                }
                else "complete matrix"
            )
            msg = f"{matrix_name} cannot aggregate missing cells: {list(matrix.missing_datasets)}"
            raise RuntimeError(msg)
        report = aggregate_results(
            matrix.rows_by_dataset,
            supported_labels=(adapter.supported_labels if adapter is not None else frozenset(PII_LABEL_SET)),
            expected_evaluation_contract=evaluation,
            expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
            result_sha256_by_config=matrix.result_sha256_by_dataset,
            timing_by_config=matrix.timing_by_dataset,
        )
        if purpose in {"full_benchmark", STEP60_BASELINE_PURPOSE}:
            assert_frozen_fixture(report)
            if (
                int(report["overall"]["rows"]) != FULL_ROWS
                or int(report["fixture"]["full9_gold_spans"]) != FULL_GOLD_SPANS
            ):
                msg = "complete matrix aggregate rows/gold spans drifted"
                raise RuntimeError(msg)
        elif int(report["overall"]["rows"]) != STEP60_PARTIAL_16OF17_ROWS:
            msg = "checkpoint partial 16/17 aggregate rows drifted"
            raise RuntimeError(msg)
        elapsed = time.monotonic() - started
        payload = {
            "status": (
                "aggregate_terminal"
                if purpose == "full_benchmark"
                else (
                    "step60_baseline_aggregate_terminal"
                    if purpose == STEP60_BASELINE_PURPOSE
                    else (
                        "step60_partial_16of17_aggregate_terminal"
                        if purpose == STEP60_PARTIAL_16OF17_PURPOSE
                        else "checkpoint_partial_16of17_aggregate_terminal"
                    )
                )
            ),
            "contract": contract,
            "checkpoint_digest": request.checkpoint_digest,
            "trajectory_digest": verified.trajectory_digest,
            "evaluation_generation_digest": evaluation.digest,
            "dataset_revisions": {
                "v2": V2_DATASET_REVISION,
                "external": EXTERNAL_DATASET_REVISION,
            },
            "result_sha256_by_cell": matrix.result_sha256_by_dataset,
            "cost_ledger": cost_ledger,
            "estimated_all_in_cost_usd": elapsed * A10G_ALL_IN_RATE_USD_PER_SECOND,
            "profile": request.profile,
            "modal_app_id": os.environ.get("MODAL_APP_ID"),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "truncated_documents": (adapter.truncated_documents if adapter is not None else 0),
            "throughput": report["overall"].get("rows_per_second"),
            "report": report,
        }
        if purpose == STEP60_BASELINE_PURPOSE:
            payload["coverage"] = step60_baseline_coverage_report(v2_eval_reused=v2_eval_reused)
        elif purpose in {
            STEP60_PARTIAL_16OF17_PURPOSE,
            CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        }:
            payload["coverage"] = step60_partial_16of17_coverage_report()
        if resume_state is not None:
            payload["resume"] = dict(resume_state)
        _atomic_json(root / "aggregate.json", payload)
        _atomic_json(
            root / "state.json",
            {
                "status": payload["status"],
                "checkpoint_digest": request.checkpoint_digest,
                "evaluation_generation_digest": evaluation.digest,
            },
        )
        benchmark_volume.commit()
        print(
            (
                "PII350_FULL_BENCHMARK_AGGREGATE::"
                if purpose == "full_benchmark"
                else (
                    "PII350_STEP60_BASELINE_AGGREGATE::"
                    if purpose == STEP60_BASELINE_PURPOSE
                    else (
                        "PII350_STEP60_PARTIAL_16OF17_AGGREGATE::"
                        if purpose == STEP60_PARTIAL_16OF17_PURPOSE
                        else "PII350_CHECKPOINT_PARTIAL_16OF17_AGGREGATE::"
                    )
                )
            )
            + json.dumps(
                {
                    "checkpoint_digest": request.checkpoint_digest,
                    "rows": contract["total_rows"],
                    "cost": payload["estimated_all_in_cost_usd"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # reason: this is the success path of a try whose excepts RE-RAISE budget and nonce control flow
        # reason: rather than swallow it. An `else` block would move the successful publish away from
        # reason: the code that produced the payload, on a spend-evidence path.
        return payload  # ruff: ignore[try-consider-else]
    except (BudgetStop, DuplicateLaunchNonce):
        raise
    except Exception as error:
        elapsed_seconds = time.monotonic() - started
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "checkpoint_digest": request.checkpoint_digest,
            "trajectory_digest": request.trajectory_digest,
            "evaluation_generation_digest": evaluation_generation_digest_from_root(root),
            "elapsed_seconds": elapsed_seconds,
            "estimated_all_in_cost_usd": elapsed_seconds * A10G_ALL_IN_RATE_USD_PER_SECOND,
            "projected_all_in_cost_usd": (elapsed_seconds + request.reserve_seconds_per_cell)
            * A10G_ALL_IN_RATE_USD_PER_SECOND,
            "last_completed_cell": last_completed_cell,
            "last_progress": dict(last_progress) if last_progress is not None else None,
            "resume": dict(resume_state) if resume_state is not None else None,
        }
        _atomic_json(root / "state.json", failure)
        _atomic_json(root / "failure.json", failure)
        benchmark_volume.commit()
        print(
            (
                "PII350_FULL_BENCHMARK_FAILED::"
                if purpose == "full_benchmark"
                else (
                    "PII350_STEP60_BASELINE_FAILED::"
                    if purpose == STEP60_BASELINE_PURPOSE
                    else (
                        "PII350_STEP60_PARTIAL_16OF17_FAILED::"
                        if purpose == STEP60_PARTIAL_16OF17_PURPOSE
                        else "PII350_CHECKPOINT_PARTIAL_16OF17_FAILED::"
                    )
                )
            )
            + json.dumps(failure, sort_keys=True),
            flush=True,
        )
        raise


@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def run_full_benchmark(
    request_payload: dict[str, Any],
    expected_checkpoint_digest: str,
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    return _run_complete_matrix(
        request_payload,
        expected_checkpoint_digest,
        purpose="full_benchmark",
        approval_payload=approval_payload,
    )


@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def run_step60_baseline(
    request_payload: dict[str, Any],
    expected_checkpoint_digest: str,
    expected_step60_reuse_receipt_digest: str,
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    return _run_complete_matrix(
        request_payload,
        expected_checkpoint_digest,
        purpose=STEP60_BASELINE_PURPOSE,
        approval_payload=approval_payload,
        expected_step60_reuse_receipt_digest=expected_step60_reuse_receipt_digest,
    )


@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def run_step60_partial_16of17(
    request_payload: dict[str, Any],
    expected_checkpoint_digest: str,
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    return _run_complete_matrix(
        request_payload,
        expected_checkpoint_digest,
        purpose=STEP60_PARTIAL_16OF17_PURPOSE,
        approval_payload=approval_payload,
    )


# reason: run fill missing combines verify and verify; splitting would split cleanup from writes.
@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def run_fill_missing_cells(  # ruff: ignore[complex-structure,too-many-locals,too-many-statements]
    request_payload: dict[str, Any],
    expected_checkpoint_digest: str,
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    """Complete the approved cells of one generation and never rerun a done cell.

    This is a deliberately separate executor from the complete-matrix path. That
    path may reuse an identity-validated existing cell; a fill run must refuse
    one, so the two cannot share a state machine without one weakening the other.

    The signature covers this list, so the approval is the schedule.

    """
    started = time.monotonic()
    request = BenchmarkRequest(**request_payload)
    if request.checkpoint_digest != expected_checkpoint_digest:
        msg = "caller checkpoint digest does not match benchmark request"
        raise RuntimeError(msg)
    purpose = FILL_MISSING_CELLS_PURPOSE
    failure_root = preflight_failure_root(request, purpose=purpose, approval=approval_payload)
    last_completed_cell: str | None = None
    last_progress: Mapping[str, Any] | None = None
    # reason: run fill missing's try keeps verify with verify; splitting would split cleanup from writes.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        approval = verify_launch_approval(approval_payload, request=request, purpose=purpose)
        require_active_launch_lease(request, purpose=purpose, approval=approval)
        claim_launch_nonce(approval)
        enable_offline_runtime(request)
        require_triton_c_compiler()
        import torch

        from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
            CheckpointArtifact,
            Pii350CheckpointAdapter,
            verify_checkpoint_artifact,
        )
        from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
        from meddies_pii.eval_baseline.baseline.run import (
            ShardSpec,
            expected_matrix_shard_identities,
            read_matrix_results,
            run_shard,
        )

        scheduled_datasets = approved_fill_cells(approval)
        contract = benchmark_contract(request, purpose=purpose, cells=scheduled_datasets)
        require_fill_missing_cells_matrix(contract, cells=scheduled_datasets)
        verified = verify_checkpoint_artifact(
            _checkpoint_local_root(request),
            CheckpointArtifact(
                request.repo_id,
                request.revision,
                request.artifact_path,
                request.checkpoint_digest,
            ),
        )
        require_expected_trajectory(request, verified.trajectory_digest)
        require_checkpoint_partial_checkpoint(request, verified)
        evaluation = _evaluation_contract(request, verified.trajectory_digest)
        root = evaluation_generation_root(request, evaluation.digest)
        model = checkpoint_model_name(request)
        _atomic_json(
            root / "state.json",
            {
                "status": "startup",
                "contract": contract,
                "trajectory_digest": verified.trajectory_digest,
                "evaluation_generation_digest": evaluation.digest,
            },
        )
        identities = expected_matrix_shard_identities(evaluation, scheduled_datasets)
        existing = read_matrix_results(
            root,
            model,
            scheduled_datasets,
            require_done=True,
            expected_rows={dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in scheduled_datasets},
            expected_evaluation_contract=evaluation,
            expected_dataset_shard_identities=identities,
        )
        require_fillable_cells(scheduled_datasets, existing.missing_datasets)
        adapter = Pii350CheckpointAdapter(verified)
        adapter.load()
        _atomic_json(
            root / "state.json",
            {
                "status": "model_loaded",
                "contract_digest": contract["contract_digest"],
                "checkpoint_digest": request.checkpoint_digest,
                "evaluation_generation_digest": evaluation.digest,
            },
        )
        cost_ledger: list[dict[str, Any]] = []
        for dataset in fill_execution_order(scheduled_datasets):
            elapsed_before_dispatch = time.monotonic() - started
            try:
                predicted_next_cell_seconds = fill_budget_allows_next_cell(
                    elapsed_seconds=elapsed_before_dispatch,
                    next_rows=EVAL_EXPECTED_ROWS[dataset],
                    request=request,
                )
            except BudgetRefusal as error:
                last_progress = publish_budget_stop(
                    root=root,
                    request=request,
                    dataset=dataset,
                    rows_done=0,
                    rows_total=EVAL_EXPECTED_ROWS[dataset],
                    elapsed_seconds=elapsed_before_dispatch,
                    predicted_next_cell_seconds=predict_fill_cell_seconds(next_rows=EVAL_EXPECTED_ROWS[dataset]),
                    prior_completed_cell=last_completed_cell,
                    prior_progress=last_progress,
                )
                msg = "benchmark budget stop before dispatch"
                raise BudgetStop(msg) from error
            rows = load_eval_cell(
                dataset,
                v2_limit=None,
                external_limit=None,
                v2_revision=V2_DATASET_REVISION,
                external_revision=EXTERNAL_DATASET_REVISION,
            )
            if len(rows) != EVAL_EXPECTED_ROWS[dataset]:
                msg = f"{dataset} row count drift: expected {EVAL_EXPECTED_ROWS[dataset]}, got {len(rows)}"
                raise RuntimeError(msg)
            cell_started = time.monotonic()

            # reason: `cell` is already captured by default argument; `prior_cell` closes the same
            # reason: hazard for the accumulator beside it. The write `last_completed_cell = dataset`
            # reason: sits at the END of this loop body, so the captured value equals the late-bound
            # reason: one at every current call — but a reordering of that write would make this
            # reason: callback report the CURRENT cell as the PRIOR one inside budget-stop evidence.
            def progress(
                done: int,
                total: int,
                *,
                cell: str = dataset,
                prior_cell: str | None = last_completed_cell,
            ) -> None:
                nonlocal last_progress
                if done % PROGRESS_EVERY != 0 and done != total:
                    return
                elapsed_seconds = time.monotonic() - started
                require_active_launch_lease(request, purpose=purpose, approval=approval)
                if budget_ceiling_reached(elapsed_seconds=elapsed_seconds, request=request):
                    last_progress = publish_budget_stop(
                        root=root,
                        request=request,
                        dataset=cell,
                        rows_done=done,
                        rows_total=total,
                        elapsed_seconds=elapsed_seconds,
                        prior_completed_cell=prior_cell,
                        prior_progress=last_progress,
                    )
                    msg = "benchmark budget stop reached inside cell"
                    raise BudgetStop(msg)
                last_progress = publish_progress(
                    root=root,
                    request=request,
                    dataset=cell,
                    rows_done=done,
                    rows_total=total,
                )

            adapter.on_progress = progress
            result = run_shard(
                adapter=adapter,
                rows=rows,
                output_root=root,
                spec=ShardSpec(model=model, dataset=dataset, shard="full"),
                evaluation_contract=evaluation,
                volume=benchmark_volume,
                force=False,
            )
            if result.skipped:
                msg = (
                    f"fill_missing_cells found an existing result for {dataset} "
                    "after admission; refusing an unaudited reuse"
                )
                raise RuntimeError(msg)
            cell_elapsed = time.monotonic() - cell_started
            record = {
                "dataset": dataset,
                "rows": len(rows),
                "elapsed_seconds": cell_elapsed,
                "observed_seconds_per_row": cell_elapsed / len(rows),
                "predicted_next_cell_seconds": predicted_next_cell_seconds,
                "estimated_all_in_cost_usd": (time.monotonic() - started) * A10G_ALL_IN_RATE_USD_PER_SECOND,
                "evaluation_generation_digest": evaluation.digest,
                "skipped": False,
            }
            cost_ledger.append(record)
            last_completed_cell = dataset
            _atomic_json(
                root / "progress" / f"{dataset}.json",
                {"status": "cell_terminal", **record},
            )
            benchmark_volume.commit()
        filled = read_matrix_results(
            root,
            model,
            scheduled_datasets,
            require_done=True,
            expected_rows={dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in scheduled_datasets},
            expected_evaluation_contract=evaluation,
            expected_dataset_shard_identities=identities,
        )
        if filled.missing_datasets:
            msg = f"fill_missing_cells did not persist every approved cell: {list(filled.missing_datasets)}"
            raise RuntimeError(msg)
        elapsed = time.monotonic() - started
        payload = {
            "status": "fill_terminal",
            "contract": contract,
            "checkpoint_digest": request.checkpoint_digest,
            "trajectory_digest": verified.trajectory_digest,
            "optimizer_step": verified.optimizer_step,
            "evaluation_generation_digest": evaluation.digest,
            "filled_cells": list(scheduled_datasets),
            "rows_by_cell": {dataset: EVAL_EXPECTED_ROWS[dataset] for dataset in scheduled_datasets},
            "result_sha256_by_cell": dict(filled.result_sha256_by_dataset),
            "cost_ledger": cost_ledger,
            "elapsed_seconds": elapsed,
            "estimated_all_in_cost_usd": elapsed * A10G_ALL_IN_RATE_USD_PER_SECOND,
            "profile": request.profile,
            "modal_app_id": os.environ.get("MODAL_APP_ID"),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "truncated_documents": adapter.truncated_documents,
            "coverage": fill_missing_cells_coverage_report(
                filled_cells=scheduled_datasets,
                generation_digest=evaluation.digest,
            ),
            "dataset_revisions": {
                "v2": V2_DATASET_REVISION,
                "external": EXTERNAL_DATASET_REVISION,
            },
        }
        _atomic_json(root / "fill.json", payload)
        _atomic_json(
            root / "state.json",
            {
                "status": "fill_terminal",
                "checkpoint_digest": request.checkpoint_digest,
                "evaluation_generation_digest": evaluation.digest,
            },
        )
        benchmark_volume.commit()
        print(
            "PII350_FILL_MISSING_CELLS::" + json.dumps(payload, sort_keys=True),
            flush=True,
        )
        # reason: this is the success path of a try whose first except RE-RAISES nonce control flow
        # reason: rather than swallow it. An `else` block would move the successful publish away from
        # reason: the code that produced the payload, on a spend-evidence path.
        return payload  # ruff: ignore[try-consider-else]
    except DuplicateLaunchNonce:
        raise
    except Exception as error:
        failure = {
            "status": "failed",
            "purpose": purpose,
            "error_type": type(error).__name__,
            "error": str(error),
            "checkpoint_digest": request.checkpoint_digest,
            "trajectory_digest": request.trajectory_digest,
            "elapsed_seconds": time.monotonic() - started,
            "last_completed_cell": last_completed_cell,
            "last_progress": dict(last_progress) if last_progress else None,
        }
        _atomic_json(failure_root / "failure.json", failure)
        benchmark_volume.commit()
        print(
            "PII350_FILL_MISSING_CELLS_FAILED::" + json.dumps(failure, sort_keys=True),
            flush=True,
        )
        raise


@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def run_checkpoint_partial_16of17(
    request_payload: dict[str, Any],
    expected_checkpoint_digest: str,
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the strict 16/17 matrix for a verified cadence checkpoint."""
    return _run_complete_matrix(
        request_payload,
        expected_checkpoint_digest,
        purpose=CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        approval_payload=approval_payload,
    )


# reason: run internal keeps read history beside verify; splitting would split cleanup from writes.
@app.function(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        BENCHMARK_MOUNT: benchmark_volume,
    },
)
def run_internal_control(  # ruff: ignore[too-many-locals,too-many-statements]
    request_payload: dict[str, Any],
    expected_checkpoint_digest: str,
    prior_history: list[dict[str, Any]],
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the 1,700-row loader-parity control without labeling it a full benchmark."""
    started = time.monotonic()
    request = BenchmarkRequest(**request_payload)
    if request.checkpoint_digest != expected_checkpoint_digest:
        msg = "caller checkpoint digest does not match internal control"
        raise RuntimeError(msg)
    approval = verify_launch_approval(approval_payload, request=request, purpose="internal_control")
    require_active_launch_lease(request, purpose="internal_control", approval=approval)
    claim_launch_nonce(approval)
    enable_offline_runtime(request)
    require_triton_c_compiler()
    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        CheckpointArtifact,
        Pii350CheckpointAdapter,
        verify_checkpoint_artifact,
    )
    from meddies_pii.eval_baseline.baseline.aggregate import aggregate_results
    from meddies_pii.eval_baseline.baseline.datasets import load_eval_cell
    from meddies_pii.eval_baseline.baseline.run import (
        ShardSpec,
        read_matrix_results,
        run_shard,
    )
    from meddies_pii.evaluation.identity import dataset_shard_identity

    contract = benchmark_contract(request, purpose="internal_control")
    verified = verify_checkpoint_artifact(
        _checkpoint_local_root(request),
        CheckpointArtifact(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
        ),
    )
    require_expected_trajectory(request, verified.trajectory_digest)
    rows = load_eval_cell(
        "v2-eval",
        v2_limit=None,
        external_limit=None,
        v2_revision=V2_DATASET_REVISION,
        external_revision=EXTERNAL_DATASET_REVISION,
    )
    expected_rows = EVAL_EXPECTED_ROWS["v2-eval"]
    if len(rows) != expected_rows:
        msg = f"internal_control row count drift: expected {expected_rows}, got {len(rows)}"
        raise RuntimeError(msg)
    evaluation = _evaluation_contract(request, verified.trajectory_digest)
    root = evaluation_generation_root(request, evaluation.digest) / "internal_control"
    _atomic_json(
        root / "state.json",
        {
            "status": "startup",
            "contract": contract,
            "checkpoint_digest": request.checkpoint_digest,
            "evaluation_generation_digest": evaluation.digest,
        },
    )
    elapsed_before_model_load = time.monotonic() - started
    try:
        predicted_control_seconds = internal_control_start_budget_seconds(
            elapsed_seconds=elapsed_before_model_load,
            request=request,
        )
    except BudgetRefusal as error:
        publish_budget_stop(
            root=root,
            request=request,
            dataset="v2-eval",
            rows_done=0,
            rows_total=expected_rows,
            elapsed_seconds=elapsed_before_model_load,
            predicted_next_cell_seconds=BOOTSTRAP_CELL_SECONDS,
        )
        msg = "internal control budget stop before model load"
        raise BudgetStop(msg) from error
    spec = ShardSpec(
        model=f"{checkpoint_model_name(request)}-internal-control",
        dataset="v2-eval",
        shard="full",
    )
    expected_identities = {
        "v2-eval": dataset_shard_identity(
            evaluation,
            dataset="v2-eval",
            shard="full",
            rows=rows,
        ),
    }
    stable_comparison_digest = comparison_contract_digest(
        evaluation,
        fixture_digest=expected_identities["v2-eval"].fixture_identity.digest,
    )
    durable_history = _read_durable_cadence_history(verified.trajectory_digest, stable_comparison_digest)
    validated_prior_history = _require_prior_cadence_history(
        prior_history,
        durable_history=durable_history,
        trajectory_digest=verified.trajectory_digest,
        stable_comparison_digest=stable_comparison_digest,
        current_step=verified.optimizer_step,
    )
    adapter = Pii350CheckpointAdapter(verified)
    adapter.load()

    def internal_control_progress(done: int, total: int) -> None:
        """Release the lease, which is reachable only after CPU admission.

        Recheck from the remote Volume at each bounded batch interval before more model work continues.

        """
        if done % INTERNAL_CONTROL_PROGRESS_EVERY != 0 and done != total:
            return
        elapsed_seconds = time.monotonic() - started
        enforce_internal_control_progress_budget(
            root=root,
            request=request,
            approval=approval,
            done=done,
            total=total,
            elapsed_seconds=elapsed_seconds,
        )

    adapter.on_progress = internal_control_progress
    elapsed_before_dispatch = time.monotonic() - started
    if budget_ceiling_reached(elapsed_seconds=elapsed_before_dispatch, request=request):
        publish_budget_stop(
            root=root,
            request=request,
            dataset="v2-eval",
            rows_done=0,
            rows_total=expected_rows,
            elapsed_seconds=elapsed_before_dispatch,
            predicted_next_cell_seconds=predicted_control_seconds,
        )
        msg = "internal control budget stop before inference"
        raise BudgetStop(msg)
    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=root,
        spec=spec,
        evaluation_contract=evaluation,
        volume=benchmark_volume,
        force=False,
    )
    matrix = read_matrix_results(
        root,
        spec.model,
        ("v2-eval",),
        require_done=True,
        expected_rows={"v2-eval": expected_rows},
        expected_evaluation_contract=evaluation,
        expected_dataset_shard_identities=expected_identities,
    )
    if matrix.missing_datasets:
        msg = f"internal control result failed immutable shard verification: {list(matrix.missing_datasets)}"
        raise RuntimeError(msg)
    report = aggregate_results(
        matrix.rows_by_dataset,
        supported_labels=adapter.supported_labels,
        expected_evaluation_contract=evaluation,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
        timing_by_config=matrix.timing_by_dataset,
    )
    summary = report["overall"]
    fixed_nine_exact_typed = {
        "precision": float(summary["full9_exact_precision"]),
        "recall": float(summary["full9_exact_recall"]),
        "f1": float(summary["full9_exact_f1"]),
    }
    current_cadence_result = {
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": verified.trajectory_digest,
        "evaluation_contract_digest": evaluation.digest,
        "comparison_contract_digest": stable_comparison_digest,
        "optimizer_step": verified.optimizer_step,
        "fixed_nine_exact_typed": fixed_nine_exact_typed,
    }
    history = [*validated_prior_history, current_cadence_result]
    _require_complete_cadence_history(
        history,
        trajectory_digest=verified.trajectory_digest,
        stable_comparison_digest=stable_comparison_digest,
        expected_last_step=verified.optimizer_step,
    )
    decision_receipt = build_cadence_decision_receipt(history)
    _persist_cadence_decision(
        trajectory_digest=verified.trajectory_digest,
        stable_comparison_digest=stable_comparison_digest,
        receipt=decision_receipt,
    )
    payload = {
        "status": decision_receipt["status"],
        "control_status": "internal_control_terminal",
        "contract": contract,
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": verified.trajectory_digest,
        "optimizer_step": verified.optimizer_step,
        "evaluation_contract_digest": evaluation.digest,
        "comparison_contract_digest": stable_comparison_digest,
        "dataset_shard_identity_digest": expected_identities["v2-eval"].digest,
        "result_sha256": matrix.result_sha256_by_dataset["v2-eval"],
        "rows": int(summary["rows"]),
        "spans_out": result.spans_out,
        "skipped": result.skipped,
        "result_path": str(result.output_path),
        "fixed_nine_exact_typed": fixed_nine_exact_typed,
        "decision_receipt": decision_receipt,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(root / "decision-receipt.json", decision_receipt)
    _atomic_json(root / "internal_control.json", payload)
    benchmark_volume.commit()
    print("PII350_INTERNAL_CONTROL::" + json.dumps(payload, sort_keys=True), flush=True)
    return payload


# reason: CLI parsing, approval checks, matrix dispatch, and exit codes share one operator-visible lifecycle.
@app.local_entrypoint()
def main(  # ruff: ignore[complex-structure,too-many-return-statements,too-many-branches,too-many-arguments,too-many-statements,too-many-positional-arguments]
    action: str = "",
    request_json: str = "",
    profile: str = "",
    prior_history_json: str = "",
    approval_json: str = "",
    approval_purpose: str = "",
    launch_nonce: str = "",
    ledger_path: str = "",
    approval_digest: str = "",
    settled_modeled_cost_usd: str = "",
    generation_digest: str = "",
    transfer_receipt_json: str = "",
    transfer_revision: str = "",
    cells: str = "",
) -> None:
    """Admit the run on CPU before Modal dispatches the A10G function.

    It is deliberately part of the normal launch action rather than a manual prerequisite, so an absent/mismatched
    challenge result costs no GPU time.

    The approval carries the schedule. A --cells argument is only ever a cross-check against it, never a second source of
    truth.

    CPU lease admission completes before the A10G function is sent.

    """
    if action not in {
        "cache_preflight",
        "triton_compiler_preflight",
        "render_contract",
        "receipt",
        "full_benchmark",
        "step60_reuse_receipt",
        "step60_baseline",
        STEP60_PARTIAL_16OF17_PURPOSE,
        CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        "internal_control",
        "issue_approval",
        "inspect_approval_ledger",
        "settle_approval",
        "release_approval",
        "export_cell_results",
        "import_cell_results",
        FILL_MISSING_CELLS_PURPOSE,
        "cell_score_summary",
    }:
        msg = (
            "action must be cache_preflight, triton_compiler_preflight, render_contract, receipt, full_benchmark, "
            "step60_reuse_receipt, step60_baseline, step60_partial_16of17, checkpoint_partial_16of17, internal_control, "
            "issue_approval, inspect_approval_ledger, settle_approval, release_approval, export_cell_results, "
            "or import_cell_results"
        )
        raise RuntimeError(msg)
    if action == "cache_preflight":
        require_local_profile_name(profile)
        print(json.dumps(cache_preflight.remote(profile), sort_keys=True))
        return
    if action == "triton_compiler_preflight":
        require_local_profile_name(profile)
        print(json.dumps(triton_compiler_preflight.remote(), sort_keys=True))
        return
    if action == "inspect_approval_ledger":
        if not ledger_path:
            msg = "ledger_path is required for inspect_approval_ledger"
            raise RuntimeError(msg)
        print(json.dumps(inspect_approval_ledger(ledger_path), sort_keys=True))
        return
    if action == "settle_approval":
        if not ledger_path or not approval_digest or not settled_modeled_cost_usd:
            msg = "ledger_path, approval_digest, and settled_modeled_cost_usd are required for settle_approval"
            raise RuntimeError(
                msg,
            )
        print(
            json.dumps(
                settle_launch_approval(
                    ledger_path,
                    approval_digest=approval_digest,
                    modeled_cost_usd=settled_modeled_cost_usd,
                ),
                sort_keys=True,
            ),
        )
        return
    if action == "release_approval":
        if not ledger_path or not approval_digest:
            msg = "ledger_path and approval_digest are required for release_approval"
            raise RuntimeError(msg)
        print(
            json.dumps(
                release_launch_approval(ledger_path, approval_digest=approval_digest),
                sort_keys=True,
            ),
        )
        return
    if not request_json:
        msg = "request_json is required"
        raise RuntimeError(msg)
    try:
        payload = json.loads(request_json)
    except json.JSONDecodeError as error:
        msg = "request_json must be valid JSON"
        raise RuntimeError(msg) from error
    if not isinstance(payload, dict):
        msg = "request_json must be an object"
        raise RuntimeError(msg)
    request = BenchmarkRequest(**payload)
    if action == "issue_approval":
        if not ledger_path or not approval_purpose or not launch_nonce:
            msg = "ledger_path, approval_purpose, and launch_nonce are required for issue_approval"
            raise RuntimeError(msg)
        print(
            json.dumps(
                issue_launch_approval(
                    request,
                    purpose=approval_purpose,
                    launch_nonce=launch_nonce,
                    ledger_path=ledger_path,
                    cells=(_parse_cells_argument(cells) if approval_purpose == FILL_MISSING_CELLS_PURPOSE else None),
                ),
                sort_keys=True,
            ),
        )
        return
    if action != "render_contract":
        require_local_profile(request)
    paid_purposes = {
        "full_benchmark": "full_benchmark",
        "step60_baseline": STEP60_BASELINE_PURPOSE,
        STEP60_PARTIAL_16OF17_PURPOSE: STEP60_PARTIAL_16OF17_PURPOSE,
        CHECKPOINT_PARTIAL_16OF17_PURPOSE: CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        FILL_MISSING_CELLS_PURPOSE: FILL_MISSING_CELLS_PURPOSE,
        "internal_control": "internal_control",
    }
    approval: dict[str, Any] | None = None
    if action in paid_purposes:
        if not approval_json:
            msg = "approval_json is required for paid launch actions"
            raise RuntimeError(msg)
        try:
            parsed_approval = json.loads(approval_json)
        except json.JSONDecodeError as error:
            msg = "approval_json must be valid JSON"
            raise RuntimeError(msg) from error
        approval = verify_launch_approval(parsed_approval, request=request, purpose=paid_purposes[action])

    def activate_paid_lease() -> None:
        """Write the live lease from a separate CPU admission, before the GPU function is submitted.

        The GPU function rechecks the same lease itself.

        """
        if approval is None or action not in paid_purposes:
            msg = "paid launch lease activation is invalid"
            raise RuntimeError(msg)
        if not ledger_path:
            msg = "ledger_path is required for paid launch actions"
            raise RuntimeError(msg)
        activate_paid_lease_from_ledger(
            ledger_path=ledger_path,
            request=request,
            purpose=paid_purposes[action],
            approval=approval,
        )

    if action == "render_contract":
        print(
            json.dumps(
                {
                    "contract": benchmark_contract(request, purpose="full_benchmark"),
                    "step60_baseline_contract": benchmark_contract(request, purpose=STEP60_BASELINE_PURPOSE),
                    "step60_partial_16of17_contract": benchmark_contract(request, purpose=STEP60_PARTIAL_16OF17_PURPOSE),
                    "checkpoint_partial_16of17_contract": benchmark_contract(
                        request,
                        purpose=CHECKPOINT_PARTIAL_16OF17_PURPOSE,
                    ),
                    "cache_preflight_command": render_cache_preflight_command(request.profile),
                    "triton_compiler_preflight_command": render_triton_compiler_preflight_command(request.profile),
                    "cpu_receipt_command": render_cpu_receipt_command(request),
                    "step60_reuse_receipt_command": render_step60_reuse_receipt_command(request),
                    "approval_json_required": True,
                    "paid_launch_commands": "call the matching renderer only with an issued Ed25519 approval JSON",
                    "issue_approval_command_template": (
                        "set PII350_BENCHMARK_APPROVAL_KEY_FILE, "
                        "then call --action issue_approval with "
                        "request_json, approval_purpose, "
                        "launch_nonce, and an external "
                        "ledger_path"
                    ),
                    "inspect_approval_ledger_command_template": "--action inspect_approval_ledger "
                    "--ledger-path <external-ledger-path>",
                    "settle_approval_command_template": (
                        "--action settle_approval --ledger-path "
                        "<external-ledger-path> --approval-digest "
                        "<approval-digest> --settled-modeled-cost-usd <usd>"
                    ),
                    "release_approval_command_template": (
                        "--action release_approval "
                        "--ledger-path <external-ledger-path> "
                        "--approval-digest "
                        "<approval-digest>"
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return
    if action == "receipt":
        print(json.dumps(checkpoint_receipt.remote(asdict(request)), sort_keys=True))
        return
    if action == "step60_reuse_receipt":
        print(json.dumps(step60_reuse_receipt.remote(asdict(request)), sort_keys=True))
        return
    if action == "cell_score_summary":
        if not generation_digest:
            msg = "generation_digest is required for cell_score_summary"
            raise RuntimeError(msg)
        print(
            json.dumps(
                cell_score_summary.remote(asdict(request), generation_digest),
                sort_keys=True,
            ),
        )
        return
    if action == "export_cell_results":
        if not generation_digest:
            msg = "generation_digest is required for export_cell_results"
            raise RuntimeError(msg)
        print(
            json.dumps(
                export_cell_results.remote(asdict(request), generation_digest),
                sort_keys=True,
            ),
        )
        return
    if action == "import_cell_results":
        if not transfer_receipt_json or not transfer_revision:
            msg = "transfer_receipt_json and transfer_revision are required for import_cell_results"
            raise RuntimeError(msg)
        try:
            transfer_receipt = json.loads(transfer_receipt_json)
        except json.JSONDecodeError as error:
            msg = "transfer_receipt_json must be valid JSON"
            raise RuntimeError(msg) from error
        if not isinstance(transfer_receipt, dict):
            msg = "transfer_receipt_json must be an object"
            raise RuntimeError(msg)
        print(
            json.dumps(
                import_cell_results.remote(asdict(request), transfer_receipt, transfer_revision),
                sort_keys=True,
            ),
        )
        return
    if action == "internal_control":
        if not prior_history_json:
            msg = "prior_history_json is required for internal_control"
            raise RuntimeError(msg)
        try:
            prior_history = json.loads(prior_history_json)
        except json.JSONDecodeError as error:
            msg = "prior_history_json must be valid JSON"
            raise RuntimeError(msg) from error
        prior_history = _cadence_history(prior_history, role="prior cadence history")
        activate_paid_lease()
        print(
            json.dumps(
                run_internal_control.remote(
                    asdict(request),
                    request.checkpoint_digest,
                    prior_history,
                    cast("dict[str, Any]", approval),
                ),
                sort_keys=True,
            ),
        )
        return
    if action == "step60_baseline":
        reuse_receipt = step60_reuse_receipt.remote(asdict(request))
        activate_paid_lease()
        print(
            json.dumps(
                run_step60_baseline.remote(
                    asdict(request),
                    request.checkpoint_digest,
                    reuse_receipt["receipt_digest"],
                    cast("dict[str, Any]", approval),
                ),
                sort_keys=True,
            ),
        )
        return
    if action == FILL_MISSING_CELLS_PURPOSE:
        approved = approved_fill_cells(cast("dict[str, Any]", approval))
        if cells and _parse_cells_argument(cells) != approved:
            msg = f"cells argument does not match the approved cell list: approved {list(approved)}"
            raise RuntimeError(msg)
        activate_paid_lease()
        print(
            json.dumps(
                run_fill_missing_cells.remote(
                    asdict(request),
                    request.checkpoint_digest,
                    cast("dict[str, Any]", approval),
                ),
                sort_keys=True,
            ),
        )
        return
    if action == STEP60_PARTIAL_16OF17_PURPOSE:
        activate_paid_lease()
        print(
            json.dumps(
                run_step60_partial_16of17.remote(
                    asdict(request),
                    request.checkpoint_digest,
                    cast("dict[str, Any]", approval),
                ),
                sort_keys=True,
            ),
        )
        return
    if action == CHECKPOINT_PARTIAL_16OF17_PURPOSE:
        activate_paid_lease()
        print(
            json.dumps(
                run_checkpoint_partial_16of17.remote(
                    asdict(request),
                    request.checkpoint_digest,
                    cast("dict[str, Any]", approval),
                ),
                sort_keys=True,
            ),
        )
        return
    activate_paid_lease()
    print(
        json.dumps(
            run_full_benchmark.remote(
                asdict(request),
                request.checkpoint_digest,
                cast("dict[str, Any]", approval),
            ),
            sort_keys=True,
        ),
    )
