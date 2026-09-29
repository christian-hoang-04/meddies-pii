"""Frozen content-addressed contract and receipts for the regex release gate."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from meddies_pii.eval_baseline.baseline.datasets import V2_DATASET_REVISION, V2_REPO_ID
from meddies_pii.eval_baseline.regex_release.regex_corpus_registry import (
    TRUSTED_CORPUS_REGISTRY_SHA256,
)
from meddies_pii.eval_baseline.regex_release.regex_fixtures import fixture_manifest
from meddies_pii.evaluation.identity import canonical_sha256, is_sha256
from meddies_pii.json_types import is_str_mapping
from meddies_pii.regex_runtime import regex_manifest

REGEX_RELEASE_SEED = 20260807
BOOTSTRAP_REPLICATES = 10_000
FULL_GIT_COMMIT_HEX_LENGTH = 40


@dataclass(frozen=True, slots=True)
class ArtifactPin:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or self.size_bytes <= 0 or not is_sha256(self.sha256):
            msg = "artifact pin requires a path, positive size, and SHA-256"
            raise ValueError(msg)

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReleaseGateContract:
    source_commit: str
    comparison_id: str
    dataset_repo: str
    dataset_revision: str
    dataset_configs: tuple[tuple[str, int], ...]
    model_repo: str
    model_revision: str
    model_artifacts: tuple[ArtifactPin, ...]
    runtime_packages: tuple[str, ...]
    image_digest: str
    seed: int
    bootstrap_replicates: int
    cpu_cores: int
    memory_mib: int
    gpu: bool
    timeout_seconds: int
    projected_cost_ceiling_usd: float
    projection_sample_rows: int
    output_volume: str
    output_root: str
    cache_volume: str
    regex_manifest_sha256: str
    fixture_manifest_sha256: str
    registry_sha256: str
    exposed_evidence_only: bool
    development_release_allowed: bool
    shipping_allowed: bool
    ignore_list_sha256: str | None = None
    """Digest of the score-time ignore-list of audited unlabeled-gold spans this run excludes from false-positive
    counting. None means no exclusion is active and scoring behaves exactly as it did before the ignore-list existed.
    """

    def __post_init__(self) -> None:
        if len(self.source_commit) != FULL_GIT_COMMIT_HEX_LENGTH:
            msg = "source_commit must be a full Git commit"
            raise ValueError(msg)
        if dict(self.dataset_configs) != {"eval": 1_700, "eval-challenge": 3_400}:
            msg = "release gate requires exact eval and eval-challenge rows"
            raise ValueError(msg)
        if self.seed != REGEX_RELEASE_SEED or self.bootstrap_replicates != BOOTSTRAP_REPLICATES:
            msg = "release gate seed and bootstrap count are frozen"
            raise ValueError(msg)
        if (self.cpu_cores, self.memory_mib, self.gpu, self.timeout_seconds) != (
            32,
            64 * 1024,
            False,
            3 * 60 * 60,
        ):
            msg = "release gate resource envelope is frozen"
            raise ValueError(msg)
        if not self.exposed_evidence_only:
            msg = "both evaluation shards are historically exposed"
            raise ValueError(msg)
        if self.development_release_allowed or self.shipping_allowed:
            msg = "exposed aggregate cannot authorize release or shipping"
            raise ValueError(msg)
        digests = (
            self.regex_manifest_sha256,
            self.fixture_manifest_sha256,
            self.registry_sha256,
            self.image_digest,
        )
        if any(not is_sha256(digest) for digest in digests):
            msg = "release gate identities must be SHA-256 digests"
            raise ValueError(msg)
        if self.ignore_list_sha256 is not None and not is_sha256(self.ignore_list_sha256):
            msg = "ignore-list digest must be a SHA-256"
            raise ValueError(msg)

    @classmethod
    def default(cls, *, source_commit: str, ignore_list_sha256: str | None = None) -> ReleaseGateContract:
        """Create the preregistered default release-gate contract.

        Single-row CPU decode of the 350M model measured ~0.1 rows/s on 8 cores (live projection 2026-08-09:
        16,399.921s / $2.301237 for the 1,700-row eval child against a 7,200s / $1.010304 envelope). CPU decode keeps
        span-for-span parity with the release reference path; GPU risks near-tie flips (11/200 precedent), so the
        answer is more cores, not a different device. The ceiling covers both children with headroom.

        Returns:
            The frozen default regex release-gate contract.

        """
        rules = regex_manifest()
        fixtures = fixture_manifest()
        return cls(
            source_commit=source_commit,
            comparison_id="q8-c5c2f3a1-s20260807",
            dataset_repo=V2_REPO_ID,
            dataset_revision=V2_DATASET_REVISION,
            dataset_configs=(("eval", 1_700), ("eval-challenge", 3_400)),
            model_repo="Meddies/meddies-pii-v2-onnx",
            model_revision="47b041b16ec3ecdf75871d42e38951f7f6d65a31",
            model_artifacts=(
                ArtifactPin(
                    "onnx/model.q8.onnx",
                    592_471_465,
                    "896b7cc8a9621c9b11e54a7893071091acf05c0055a2e7df7e35220587c73fe3",
                ),
                ArtifactPin(
                    "tokenizer.json",
                    4_733_016,
                    "4905ab82b2cfc25e0c88adc8f4eeffe759c57c5626312b30b0aaeaf8ad3379bc",
                ),
                ArtifactPin(
                    "tokenizer_config.json",
                    526,
                    "1c02b0dd850ea012fa8824ae9facdf4cfb367463d3b33134e567b9f434cc9241",
                ),
            ),
            runtime_packages=(
                "datasets==4.5.0",
                "huggingface_hub==1.19.0",
                "numpy==2.4.4",
                "onnxruntime==1.27.0",
                "pyarrow==23.0.1",
                "transformers==5.5.0",
            ),
            image_digest=("72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"),
            seed=REGEX_RELEASE_SEED,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            cpu_cores=32,
            memory_mib=64 * 1024,
            gpu=False,
            timeout_seconds=3 * 60 * 60,
            projected_cost_ceiling_usd=6.50,
            projection_sample_rows=100,
            output_volume="meddies-pii-regex-release-gate",
            output_root="/regex-eval/runs",
            cache_volume="hf-cache",
            regex_manifest_sha256=str(rules["sha256"]),
            fixture_manifest_sha256=str(fixtures["sha256"]),
            registry_sha256=TRUSTED_CORPUS_REGISTRY_SHA256,
            exposed_evidence_only=True,
            development_release_allowed=False,
            shipping_allowed=False,
            ignore_list_sha256=ignore_list_sha256,
        )

    @property
    def dataset_rows(self) -> dict[str, int]:
        return dict(self.dataset_configs)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_commit": self.source_commit,
            "comparison_id": self.comparison_id,
            "dataset": {
                "repo": self.dataset_repo,
                "revision": self.dataset_revision,
                "configs": [{"name": name, "rows": rows} for name, rows in self.dataset_configs],
            },
            "model": {
                "repo": self.model_repo,
                "revision": self.model_revision,
                "artifacts": [artifact.to_payload() for artifact in self.model_artifacts],
            },
            "runtime_packages": list(self.runtime_packages),
            "image_digest": self.image_digest,
            "seed": self.seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "resources": {
                "cpu_cores": self.cpu_cores,
                "memory_mib": self.memory_mib,
                "gpu": self.gpu,
                "timeout_seconds": self.timeout_seconds,
                "projected_cost_ceiling_usd": self.projected_cost_ceiling_usd,
                "projection_sample_rows": self.projection_sample_rows,
            },
            "storage": {
                "output_volume": self.output_volume,
                "output_root": self.output_root,
                "cache_volume": self.cache_volume,
                "row_results": (
                    f"{self.output_root}/results/<model>/regex_ablation/"
                    f"{self.comparison_id}/<regex-sha256>/"
                    "<view>/<dataset>/<config>.jsonl"
                ),
                "child_evidence": (f"{self.output_root}/{self.comparison_id}/children/<config>.json"),
                "aggregate": (f"{self.output_root}/{self.comparison_id}/aggregate/exposed-5100.json"),
                "receipts": (f"{self.output_root}/{self.comparison_id}/receipts/<stage>-<status>.json"),
            },
            "evidence": {
                "regex_manifest_sha256": self.regex_manifest_sha256,
                "fixture_manifest_sha256": self.fixture_manifest_sha256,
                "registry_sha256": self.registry_sha256,
                "exposed_evidence_only": self.exposed_evidence_only,
                "development_release_allowed": self.development_release_allowed,
                "shipping_allowed": self.shipping_allowed,
                "ignore_list_sha256": self.ignore_list_sha256,
            },
        }


ReceiptStatus = Literal["approved", "hydrated", "completed", "failed", "budget_stop"]


def stage_approval_receipt(
    output_root: str | Path,
    contract: ReleaseGateContract,
    *,
    approved_contract_sha256: str,
) -> Path:
    """Persist explicit approval for one exact experiment contract.

    Returns:
        The path of the persisted approval receipt.

    Raises:
        ValueError: If the approved contract digest does not match the rendered contract.

    """
    if approved_contract_sha256 != contract.sha256:
        msg = "approved contract SHA-256 does not match rendered contract"
        raise ValueError(msg)
    return _write_receipt(
        _run_root(output_root, contract) / "approval.json",
        {
            "schema_version": 1,
            "status": "approved",
            "contract_sha256": contract.sha256,
            "contract": contract.to_payload(),
        },
    )


# reason: this keyword-only boundary mirrors the persisted hydration schema; each evidence field is independently pinned.
def write_hydration_receipt(  # ruff: ignore[too-many-arguments]
    output_root: str | Path,
    contract: ReleaseGateContract,
    *,
    model_artifact_sha256s: tuple[str, ...],
    dataset_row_counts: dict[str, int],
    dataset_fixture_sha256s: dict[str, str],
    ignore_list_sha256: str | None = None,
) -> Path:
    """Record that every pinned artifact is cached before offline inference.

    Returns:
        The path of the persisted hydration receipt.

    Raises:
        ValueError: If hydrated model, dataset, fixture, or ignore-list identities differ from the contract.

    """
    expected_artifacts = tuple(artifact.sha256 for artifact in contract.model_artifacts)
    if model_artifact_sha256s != expected_artifacts:
        msg = "hydrated model artifacts do not match contract pins"
        raise ValueError(msg)
    if dataset_row_counts != contract.dataset_rows:
        msg = "hydrated dataset rows do not match contract pins"
        raise ValueError(msg)
    if set(dataset_fixture_sha256s) != set(contract.dataset_rows) or any(
        not is_sha256(digest) for digest in dataset_fixture_sha256s.values()
    ):
        msg = "hydrated dataset fixtures require one SHA-256 per config"
        raise ValueError(msg)
    if ignore_list_sha256 != contract.ignore_list_sha256:
        msg = "hydrated ignore-list digest does not match contract pin"
        raise ValueError(msg)
    _validate_receipt(
        _run_root(output_root, contract) / "approval.json",
        contract,
        expected_status="approved",
        missing_message="approval receipt is missing",
    )
    return _write_receipt(
        _run_root(output_root, contract) / "hydration.json",
        {
            "schema_version": 1,
            "status": "hydrated",
            "contract_sha256": contract.sha256,
            "model_artifact_sha256s": list(model_artifact_sha256s),
            "dataset_row_counts": dataset_row_counts,
            "dataset_fixture_sha256s": dataset_fixture_sha256s,
            "ignore_list_sha256": ignore_list_sha256,
            "offline_ready": True,
        },
    )


def require_paid_run_receipts(output_root: str | Path, contract: ReleaseGateContract) -> dict[str, object]:
    run_root = _run_root(output_root, contract)
    _validate_receipt(
        run_root / "approval.json",
        contract,
        expected_status="approved",
        missing_message="approval receipt is missing",
    )
    hydration = _validate_receipt(
        run_root / "hydration.json",
        contract,
        expected_status="hydrated",
        missing_message="hydration receipt is missing",
    )
    if hydration.get("offline_ready") is not True:
        msg = "hydration receipt is not offline-ready"
        raise RuntimeError(msg)
    return hydration


def require_approval_receipt(output_root: str | Path, contract: ReleaseGateContract) -> None:
    _validate_receipt(
        _run_root(output_root, contract) / "approval.json",
        contract,
        expected_status="approved",
        missing_message="approval receipt is missing",
    )


def render_launch_commands(contract: ReleaseGateContract) -> dict[str, str]:
    """Render launch commands that reconstruct the same contract pin.

    Every stage reconstructs the contract from this same pin (or its absence); omitting the flag on a pinned contract
    sends a copy-paster straight back into the local/remote digest mismatch the pin exists to prevent.

    Returns:
        A mapping from each release-gate action to its digest-pinned launch command.

    """
    prefix = "MODAL_PROFILE=huyhoang041100 uv run --no-sync modal run scripts/ops/run_regex_release_gate.py"
    digest = contract.sha256
    pin_flag = f" --ignore-list-sha256 {contract.ignore_list_sha256}" if contract.ignore_list_sha256 is not None else ""
    return {
        "stage_approval": (f"{prefix}::stage_approval --approved-contract-sha256 {digest}{pin_flag}"),
        "hydrate": f"{prefix}::hydrate --contract-sha256 {digest}{pin_flag}",
        "run_eval": f"{prefix}::run_eval --contract-sha256 {digest}{pin_flag}",
        "run_eval_challenge": (f"{prefix}::run_eval_challenge --contract-sha256 {digest}{pin_flag}"),
        "aggregate_5100": (f"{prefix}::aggregate_5100 --contract-sha256 {digest}{pin_flag}"),
    }


# reason: this keyword-only boundary mirrors the terminal receipt schema; status-specific evidence stays explicit.
def write_terminal_receipt(  # ruff: ignore[too-many-arguments]
    output_root: str | Path,
    contract: ReleaseGateContract,
    *,
    stage: str,
    status: Literal["completed", "failed", "budget_stop"],
    aggregate: dict[str, int | float | str | bool] | None = None,
    exception_type: str | None = None,
) -> Path:
    """Write terminal evidence with the fields required by its status.

    A budget stop carries its projection the way a completion carries its counts; only a failure is metric-free,
    because its payload is the exception type and nothing derived from row content.

    Returns:
        The path of the persisted terminal receipt.

    Raises:
        ValueError: If stage, status, aggregate, or exception evidence is inconsistent.

    """
    carries_metrics = status in {"completed", "budget_stop"}
    if not stage or carries_metrics != (aggregate is not None):
        msg = "only completed and budget-stop receipts carry aggregates"
        raise ValueError(msg)
    if status == "failed" and not exception_type:
        msg = "failed receipts require a safe exception type"
        raise ValueError(msg)
    body: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "stage": stage,
        "contract_sha256": contract.sha256,
    }
    if aggregate is not None:
        body["aggregate"] = aggregate
    if exception_type is not None:
        body["exception_type"] = exception_type
    return _write_receipt(
        _run_root(output_root, contract) / "receipts" / f"{stage}-{status}.json",
        body,
    )


def _run_root(output_root: str | Path, contract: ReleaseGateContract) -> Path:
    return Path(output_root) / contract.comparison_id


def _write_receipt(path: Path, body: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**body, "receipt_sha256": canonical_sha256(body)}
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(temporary).replace(path)
    return path


def _validate_receipt(
    path: Path,
    contract: ReleaseGateContract,
    *,
    expected_status: ReceiptStatus,
    missing_message: str,
) -> dict[str, object]:
    receipt_label = "approval" if expected_status == "approved" else expected_status
    if not path.is_file():
        raise RuntimeError(missing_message)
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not is_str_mapping(value):
        msg = f"{receipt_label} receipt is malformed"
        raise RuntimeError(msg)
    raw = dict(value)
    receipt_sha256 = raw.pop("receipt_sha256", None)
    if (
        raw.get("status") != expected_status
        or raw.get("contract_sha256") != contract.sha256
        or receipt_sha256 != canonical_sha256(raw)
        or (expected_status == "approved" and raw.get("contract") != contract.to_payload())
    ):
        msg = f"{receipt_label} receipt does not match contract"
        raise RuntimeError(msg)
    return raw
