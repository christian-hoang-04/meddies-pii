"""Modal CPU release-gate experiment for model-core versus regex postprocessing.

Render the frozen contract and commands first. No experiment action accepts a
different seed, data revision, artifact, runtime, resource envelope, or verdict.

    MODAL_PROFILE=huyhoang041100 uv run --no-sync modal run \
      scripts/ops/run_regex_release_gate.py::render

The contract needs a source commit, so it cannot be built at module scope where the image and the decorators are
defined. These mirror the frozen envelope in ReleaseGateContract.default() instead, and
test_modal_functions_request_the_contract_compute_envelope holds them to it. OMP_NUM_THREADS belongs here too: ONNX
Runtime reads it for intra-op parallelism, so a stale value caps decode below the cores we pay for.

Deterministic local paths for the derived ignore-list artifact + its provenance sidecar. `build_ignore_list` writes
both; `render` needs the printed digest to pin a contract; the Modal image bakes the directory in so `_hydrate` can
read the file without a separate upload step.

add_local_dir below needs this directory to exist at import time — a fresh clone has no scripts/ops/artifacts/ until
build_ignore_list runs once, so the mount must not depend on that having happened yet.
"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this Modal command is invoked and tested by file path; scripts/ops is not a distributable package API.
# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt remains explicit for this operational command; the shipped package owns the API contract.
# ruff: file-ignore[import-outside-top-level]
# reason: `_hydrate` runs in the Modal image and owns the dataset and model-download stack; local
# reason: render and approval commands must remain importable without loading those remote dependencies.
# ruff: file-ignore[print]
# reason: these Modal CLI entrypoints emit the contract, digest, commands, and remote receipts as their product.
import json
import os

# reason: the local render gate invokes only Git with list-form argv to bind the contract to a clean source commit.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from dataclasses import asdict
from pathlib import Path
from typing import cast

import modal

from meddies_pii.eval_baseline.regex_release.regex_ignore_list import (
    RegexIgnoreList,
    ignore_list_from_audit_proposal,
    load_ignore_list_jsonl,
    pinned_document_uids,
    write_ignore_list_jsonl,
    write_ignore_list_provenance,
)
from meddies_pii.eval_baseline.regex_release.regex_release_contract import (
    ReleaseGateContract,
    render_launch_commands,
    require_approval_receipt,
    require_paid_run_receipts,
    stage_approval_receipt,
    write_hydration_receipt,
    write_terminal_receipt,
)
from meddies_pii.eval_baseline.regex_release.regex_release_gate import (
    BudgetProjectionError,
    ChildRegexEvidence,
    aggregate_exposed_children,
)
from meddies_pii.eval_baseline.regex_release.regex_release_runtime import (
    child_from_payload,
    child_to_payload,
    evaluation_contracts,
    execute_child,
)
from meddies_pii.json_types import is_str_mapping
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
    use_pinned_debian_snapshot,
)

APP_NAME = "meddies-pii-regex-release-gate"
GATE_CPU_CORES = 32
GATE_MEMORY_MIB = 64 * 1024
GATE_TIMEOUT_SECONDS = 3 * 60 * 60
OUTPUT_MOUNT = "/regex-eval"
CACHE_MOUNT = "/cache"
ARTIFACTS_MOUNT = "/artifacts"

IGNORE_LIST_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
IGNORE_LIST_ARTIFACT_PATH = IGNORE_LIST_ARTIFACT_DIR / "ignore-list-tierA.jsonl"
IGNORE_LIST_PROVENANCE_PATH = IGNORE_LIST_ARTIFACT_DIR / "ignore-list-tierA.provenance.json"
IGNORE_LIST_MOUNT_PATH = f"{ARTIFACTS_MOUNT}/ignore-list-tierA.jsonl"

GATE_RUNTIME_ENV = {
    "HF_HOME": f"{CACHE_MOUNT}/huggingface",
    "HF_DATASETS_CACHE": f"{CACHE_MOUNT}/datasets",
    "OMP_NUM_THREADS": str(GATE_CPU_CORES),
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

IGNORE_LIST_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

output_volume = modal.Volume.from_name("meddies-pii-regex-release-gate", create_if_missing=True)
cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

runtime_image = (
    add_source_pythonpath(
        use_pinned_debian_snapshot(
            modal.Image.from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"),
        )
        .pip_install(
            "datasets==4.5.0",
            "huggingface_hub==1.19.0",
            "numpy==2.4.4",
            "onnxruntime==1.27.0",
            "pyarrow==23.0.1",
            "transformers==5.5.0",
        )
        .env(GATE_RUNTIME_ENV),
    )
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
    .add_local_dir(str(IGNORE_LIST_ARTIFACT_DIR), remote_path=ARTIFACTS_MOUNT)
)
"""add_local_dir MUST be the last image step (Modal forbids build steps after local mounts), so the neutral PYTHONPATH
wrapper is applied before the source mount. The artifacts mount bakes in the derived ignore-list JSONL so `_hydrate`
can read it without a separate network upload; `build_ignore_list` must run (and the artifact must exist on disk)
before any `modal run` of this file.
"""

app = modal.App(APP_NAME)


@app.function(
    image=runtime_image,
    cpu=float(GATE_CPU_CORES),
    memory=GATE_MEMORY_MIB,
    timeout=GATE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
)
def _stage_approval(
    source_commit: str,
    approved_contract_sha256: str,
    ignore_list_sha256: str | None = None,
) -> str:
    contract = ReleaseGateContract.default(source_commit=source_commit, ignore_list_sha256=ignore_list_sha256)
    stage_approval_receipt(
        contract.output_root,
        contract,
        approved_contract_sha256=approved_contract_sha256,
    )
    output_volume.commit()
    return contract.sha256


@app.function(
    image=runtime_image,
    cpu=float(GATE_CPU_CORES),
    memory=GATE_MEMORY_MIB,
    timeout=GATE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
    secrets=[hf_secret],
)
def _hydrate(
    source_commit: str,
    contract_sha256: str,
    ignore_list_sha256: str | None = None,
) -> dict[str, object]:
    from meddies_pii.bioes_inference.artifacts import (
        PUBLIC_Q8_ARTIFACT_SPEC,
        hydrate_artifact,
    )
    from meddies_pii.eval_baseline.baseline.datasets import load_v2_eval_rows
    from meddies_pii.eval_baseline.baseline.views import MODEL_CORE_VIEW
    from meddies_pii.evaluation.identity import dataset_shard_identity

    contract = _checked_contract(source_commit, contract_sha256, ignore_list_sha256)
    _require_approval(contract)
    artifact = hydrate_artifact(PUBLIC_Q8_ARTIFACT_SPEC, Path(CACHE_MOUNT) / "regex-release-artifacts")
    rows_by_config = {
        config: load_v2_eval_rows(config, revision=contract.dataset_revision) for config in contract.dataset_rows
    }
    row_counts = {config: len(rows) for config, rows in rows_by_config.items()}
    core_contract = evaluation_contracts(contract)[MODEL_CORE_VIEW]
    fixture_sha256s = {
        config: dataset_shard_identity(
            core_contract,
            dataset=contract.dataset_repo,
            shard=config,
            rows=rows,
        ).fixture_identity.digest
        for config, rows in rows_by_config.items()
    }
    ignore_list_sha256 = _stage_and_verify_ignore_list(
        contract.output_root,
        contract,
        local_artifact_path=IGNORE_LIST_MOUNT_PATH,
    )
    write_hydration_receipt(
        contract.output_root,
        contract,
        model_artifact_sha256s=tuple(file.sha256 for file in artifact.files),
        dataset_row_counts=row_counts,
        dataset_fixture_sha256s=fixture_sha256s,
        ignore_list_sha256=ignore_list_sha256,
    )
    cache_volume.commit()
    output_volume.commit()
    return {
        "status": "hydrated",
        "contract_sha256": contract.sha256,
        "dataset_row_counts": row_counts,
        "dataset_fixture_sha256s": fixture_sha256s,
        "artifact_count": len(artifact.files),
        "ignore_list_sha256": ignore_list_sha256,
    }


@app.function(
    image=runtime_image,
    cpu=float(GATE_CPU_CORES),
    memory=GATE_MEMORY_MIB,
    timeout=GATE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
)
def _run_child(
    source_commit: str,
    contract_sha256: str,
    config: str,
    ignore_list_sha256: str | None = None,
) -> dict[str, object]:
    contract = _checked_contract(source_commit, contract_sha256, ignore_list_sha256)
    require_paid_run_receipts(contract.output_root, contract)
    if config not in contract.dataset_rows:
        msg = "config is outside the approved contract"
        raise ValueError(msg)
    os.environ.update({"HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    ignore_list = _load_staged_ignore_list(contract.output_root, contract)
    try:
        child = execute_child(contract, config, volume=output_volume, ignore_list=ignore_list)
        child_path = _child_path(contract, config)
        _write_json(child_path, child_to_payload(child))
        write_terminal_receipt(
            contract.output_root,
            contract,
            stage=config,
            status="completed",
            aggregate={
                "documents": len(child.documents),
                "label_count": len(child.label_counts),
                "evidence_sha256": child.evidence_sha256,
            },
        )
        output_volume.commit()
        return {
            "status": "completed",
            "config": config,
            "documents": len(child.documents),
            "evidence_sha256": child.evidence_sha256,
        }
    except BudgetProjectionError as error:
        projection = error.projection
        write_terminal_receipt(
            contract.output_root,
            contract,
            stage=config,
            status="budget_stop",
            aggregate={
                "sample_rows": projection.sample_rows,
                "sample_seconds": projection.sample_seconds,
                "projected_total_seconds": projection.projected_total_seconds,
                "projected_cost_usd": projection.projected_cost_usd,
            },
        )
        output_volume.commit()
        raise
    except Exception as error:
        write_terminal_receipt(
            contract.output_root,
            contract,
            stage=config,
            status="failed",
            exception_type=type(error).__name__,
        )
        output_volume.commit()
        raise


@app.function(
    image=runtime_image,
    cpu=float(GATE_CPU_CORES),
    memory=GATE_MEMORY_MIB,
    timeout=GATE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
)
def _aggregate(
    source_commit: str,
    contract_sha256: str,
    ignore_list_sha256: str | None = None,
) -> dict[str, int | float | str | bool]:
    contract = _checked_contract(source_commit, contract_sha256, ignore_list_sha256)
    hydration = require_paid_run_receipts(contract.output_root, contract)
    try:
        return _execute_aggregate(contract, hydration)
    except Exception as error:
        write_terminal_receipt(
            contract.output_root,
            contract,
            stage="aggregate-5100",
            status="failed",
            exception_type=type(error).__name__,
        )
        output_volume.commit()
        raise


def _execute_aggregate(contract: ReleaseGateContract, hydration: dict[str, object]) -> dict[str, int | float | str | bool]:
    """Redaction-safety distribution, report-only: never a gate condition.

    Over-redaction distribution, report-only: never a gate condition. Empty whenever any row in a scope predates the
    document_chars field.
    """
    children = cast(
        "tuple[ChildRegexEvidence, ChildRegexEvidence]",
        tuple(child_from_payload(_read_json(_child_path(contract, config))) for config in ("eval", "eval-challenge")),
    )
    report = aggregate_exposed_children(
        contract=contract,
        children=children,
        hydrated_fixture_sha256s=_hydrated_fixture_sha256s(hydration),
    )
    payload = asdict(report)
    _write_json(
        Path(contract.output_root) / contract.comparison_id / "aggregate" / "exposed-5100.json",
        payload,
    )
    summary: dict[str, int | float | str | bool] = {
        "challenge_documents": report.challenge.documents,
        "challenge_model_f1": report.challenge.model_f1,
        "challenge_regex_f1": report.challenge.regex_f1,
        "challenge_model_precision": report.challenge.model_precision,
        "challenge_regex_precision": report.challenge.regex_precision,
        "challenge_f1_delta_lower_95": report.challenge.bootstrap.lower_95,
        "challenge_f1_delta_upper_95": report.challenge.bootstrap.upper_95,
        "challenge_sample_adequate": report.challenge.sample_adequate_for_inference,
        "quality_gate_passed": report.quality_gate_passed,
        "challenge_model_fp_ignored": report.challenge.model_fp_ignored,
        "challenge_regex_fp_ignored": report.challenge.regex_fp_ignored,
        "descriptive_pooled_documents": report.documents,
        "descriptive_pooled_model_f1": report.model_f1,
        "descriptive_pooled_regex_f1": report.regex_f1,
        "descriptive_pooled_f1_delta_lower_95": (report.descriptive_pooled_bootstrap.lower_95),
        "descriptive_pooled_f1_delta_upper_95": (report.descriptive_pooled_bootstrap.upper_95),
        "pooled_model_fp_ignored": report.model_fp_ignored,
        "pooled_regex_fp_ignored": report.regex_fp_ignored,
        "development_release_gate_passed": False,
        "shipping_confirmation_passed": False,
        "evidence_sha256": report.evidence_sha256,
    }
    for item in report.child_ignored_false_positives:
        summary[f"{item.config}_model_fp_ignored".replace("-", "_")] = item.model_fp_ignored
        summary[f"{item.config}_regex_fp_ignored".replace("-", "_")] = item.regex_fp_ignored
    for leak in report.descriptive_leak_metrics:
        prefix = f"leak_{leak.scope}_{leak.view}".replace("-", "_")
        summary[f"{prefix}_rows_with_gold"] = leak.rows_with_gold
        summary[f"{prefix}_any_leak_rows"] = leak.any_leak_rows
        summary[f"{prefix}_any_leak_row_rate"] = leak.any_leak_row_rate
        summary[f"{prefix}_p95_row_leak_fraction"] = leak.p95_row_leak_fraction
        summary[f"{prefix}_max_row_leak_fraction"] = leak.max_row_leak_fraction
        summary[f"{prefix}_char_weighted_leak_fraction"] = leak.char_weighted_leak_fraction
    for overredaction in report.descriptive_overredaction_metrics:
        prefix = f"overredaction_{overredaction.scope}_{overredaction.view}".replace("-", "_")
        summary[f"{prefix}_rows_with_text"] = overredaction.rows_with_text
        summary[f"{prefix}_any_overredaction_rows"] = overredaction.any_overredaction_rows
        summary[f"{prefix}_any_overredaction_row_rate"] = overredaction.any_overredaction_row_rate
        summary[f"{prefix}_p95_row_overredaction_fraction"] = overredaction.p95_row_overredaction_fraction
        summary[f"{prefix}_max_row_overredaction_fraction"] = overredaction.max_row_overredaction_fraction
        summary[f"{prefix}_char_weighted_overredaction_fraction"] = overredaction.char_weighted_overredaction_fraction
    write_terminal_receipt(
        contract.output_root,
        contract,
        stage="aggregate-5100",
        status="completed",
        aggregate=summary,
    )
    output_volume.commit()
    return summary


@app.local_entrypoint()
def build_ignore_list(proposal_path: str, tiers: str = "A") -> None:
    """Derive the ignore-list artifact and provenance sidecar locally.

    This makes no Modal call. Run this before `render`;
    the digest it prints is what `render --ignore-list-sha256` pins.
    """
    rows = _read_jsonl(Path(proposal_path))
    tier_set = frozenset(tier.strip() for tier in tiers.split(","))
    ignore_list = ignore_list_from_audit_proposal(rows, tiers=tier_set, document_uids=pinned_document_uids())
    write_ignore_list_jsonl(IGNORE_LIST_ARTIFACT_PATH, ignore_list)
    write_ignore_list_provenance(
        IGNORE_LIST_PROVENANCE_PATH,
        ignore_list=ignore_list,
        source_proposal_path=proposal_path,
        tiers=tier_set,
    )
    print(
        json.dumps(
            {
                "ignore_list_sha256": ignore_list.sha256,
                "entry_count": len(ignore_list.entries),
                "artifact_path": str(IGNORE_LIST_ARTIFACT_PATH),
                "provenance_path": str(IGNORE_LIST_PROVENANCE_PATH),
            },
            sort_keys=True,
            indent=2,
        ),
    )


@app.local_entrypoint()
def render(ignore_list_sha256: str | None = None) -> None:
    contract = ReleaseGateContract.default(source_commit=_clean_source_commit(), ignore_list_sha256=ignore_list_sha256)
    print(
        json.dumps(
            {
                "contract": contract.to_payload(),
                "contract_sha256": contract.sha256,
                "commands": render_launch_commands(contract),
            },
            sort_keys=True,
            indent=2,
        ),
    )


@app.local_entrypoint()
def stage_approval(approved_contract_sha256: str, ignore_list_sha256: str | None = None) -> None:
    source_commit = _clean_source_commit()
    print(_stage_approval.remote(source_commit, approved_contract_sha256, ignore_list_sha256))


@app.local_entrypoint()
def hydrate(contract_sha256: str, ignore_list_sha256: str | None = None) -> None:
    print(
        json.dumps(
            _hydrate.remote(_clean_source_commit(), contract_sha256, ignore_list_sha256),
            sort_keys=True,
        ),
    )


@app.local_entrypoint()
def run_eval(contract_sha256: str, ignore_list_sha256: str | None = None) -> None:
    print(
        json.dumps(
            _run_child.remote(_clean_source_commit(), contract_sha256, "eval", ignore_list_sha256),
            sort_keys=True,
        ),
    )


@app.local_entrypoint()
def run_eval_challenge(contract_sha256: str, ignore_list_sha256: str | None = None) -> None:
    print(
        json.dumps(
            _run_child.remote(
                _clean_source_commit(),
                contract_sha256,
                "eval-challenge",
                ignore_list_sha256,
            ),
            sort_keys=True,
        ),
    )


@app.local_entrypoint()
def aggregate_5100(contract_sha256: str, ignore_list_sha256: str | None = None) -> None:
    print(
        json.dumps(
            _aggregate.remote(_clean_source_commit(), contract_sha256, ignore_list_sha256),
            sort_keys=True,
        ),
    )


def _hydrated_fixture_sha256s(hydration: dict[str, object]) -> dict[str, str]:
    digests = hydration.get("dataset_fixture_sha256s")
    if not isinstance(digests, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in digests.items()
    ):
        msg = "hydration receipt fixture digests are malformed"
        raise ValueError(msg)
    return cast("dict[str, str]", digests)


def _checked_contract(
    source_commit: str,
    expected_sha256: str,
    ignore_list_sha256: str | None = None,
) -> ReleaseGateContract:
    contract = ReleaseGateContract.default(source_commit=source_commit, ignore_list_sha256=ignore_list_sha256)
    if contract.sha256 != expected_sha256:
        msg = "contract SHA-256 does not match frozen inputs"
        raise ValueError(msg)
    return contract


def _require_approval(contract: ReleaseGateContract) -> None:
    require_approval_receipt(contract.output_root, contract)


def _child_path(contract: ReleaseGateContract, config: str) -> Path:
    return Path(contract.output_root) / contract.comparison_id / "children" / f"{config}.json"


def _ignore_list_path(output_root: str | Path, contract: ReleaseGateContract) -> Path:
    return Path(output_root) / contract.comparison_id / "ignore-list.jsonl"


def _stage_and_verify_ignore_list(
    output_root: str | Path,
    contract: ReleaseGateContract,
    *,
    local_artifact_path: str | Path,
) -> str | None:
    """Verify and persist the image-mounted ignore-list.

    Verify it against the contract pin, then persist a run-scoped copy under
    output_root for later stages to load.

    Returns the verified digest, or None when the contract declares no
    ignore-list — a pure no-op, so a run without one is unaffected.
    """
    if contract.ignore_list_sha256 is None:
        return None
    ignore_list = load_ignore_list_jsonl(local_artifact_path)
    if ignore_list.sha256 != contract.ignore_list_sha256:
        msg = "mounted ignore-list digest does not match the contract pin"
        raise RuntimeError(msg)
    target = _ignore_list_path(output_root, contract)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(local_artifact_path).read_bytes())
    return contract.ignore_list_sha256


def _load_staged_ignore_list(output_root: str | Path, contract: ReleaseGateContract) -> RegexIgnoreList | None:
    if contract.ignore_list_sha256 is None:
        return None
    return load_ignore_list_jsonl(_ignore_list_path(output_root, contract))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not is_str_mapping(payload):
        msg = "JSON evidence must be an object"
        raise ValueError(msg)
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [cast("dict[str, object]", json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _clean_source_commit() -> str:
    status = subprocess.run(
        # reason: PATH intentionally selects the operator's Git; fixed argv and no shell leave nothing to parse.
        ["git", "status", "--porcelain", "--untracked-files=all"],  # ruff: ignore[start-process-with-partial-path]
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        msg = "commit the release-gate runner before rendering or launching"
        raise RuntimeError(msg)
    # reason: PATH intentionally selects the operator's Git; fixed argv and no shell leave nothing to parse.
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # ruff: ignore[start-process-with-partial-path]
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
