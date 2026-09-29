"""One-container PII350 continuation from the verified step-60 receipt."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack (torch, transformers, unsloth, datasets) is installed in the Modal image, so each
# reason: function body imports it inside the container.
# ruff: file-ignore[print]
# reason: this module runs inside a Modal container; its standard output is the operator's streamed run log.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
# ruff: file-ignore[implicit-namespace-package]
# reason: `modal/` is the only subpackage of `bioes/` without an `__init__.py` — assembly, data, eval,
# reason: reports and trainers all have one — so the asymmetry reads as an oversight, and adding the
# reason: file is likely inert under hatchling's src-layout discovery.
# reason: Deliberately deferred rather than fixed here: this directory holds every spend-authorization
# reason: gate, and its Modal-remote import paths have only fake-mediated local coverage, so adding
# reason: `__init__.py` is a post-merge change whose proof is a real GPU smoke run — owner ledger item.
# ruff: file-ignore[import-private-name]
# reason: the Modal orchestration modules share `full_run` and `full_run_runtime` privates —
# reason: `_validated_receipt_inventory`, `_asset_receipt_contract`, `_preflight_candidate_assets`,
# reason: `_receipt_path`, `_iter_packed_rows`, `_packed_batch` and siblings — so every launch path
# reason: computes its receipts and batches from ONE implementation. A second copy on a spend gate is
# reason: the failure mode this avoids. The underscore is the defect, not the import; promoting them
# reason: to a real seam is a public-API change on the spend surface and sits on the owner ledger.
import json
import os

# reason: the Modal parent launches the fixed torchrun worker with list-form argv inside its pinned image.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from itertools import islice
from operator import itemgetter
from pathlib import Path, PurePosixPath
from statistics import median
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeGuard, cast

import modal

from meddies_pii.modal_runtime import add_source_pythonpath
from meddies_pii.training.bioes.modal.probe_scaffold import (
    ARTIFACT_MOUNT,
    TrainableModule,
    VolumeMounts,
    append_event,
    artifact_volume,
    hf_cache_environment,
    hf_cache_volume,
    hf_secret,
    print_stderr_sink,
    print_stdout_sink,
    stream_child_output,
    volume_mounts,
)
from meddies_pii.training.bioes.trainers.full_run import (
    ENCODER_FULL_RUN_IMAGE_PACKAGES as CONTRACT_ENCODER_FULL_RUN_IMAGE_PACKAGES,
)
from meddies_pii.training.bioes.trainers.pii350_ddp_continuation import (
    CHECKPOINT_REQUIRED_FILES,
    CPU_AUXILIARY_RATE_USD_PER_SECOND,
    EPOCH_CURSOR,
    EPOCH_TERMINAL_STEP,
    EVAL_ROWS,
    EXECUTION_TOPOLOGIES,
    FINAL_GLOBAL_BATCH_SIZE,
    GLOBAL_BATCH_SIZE,
    SHUTDOWN_RESERVE_SECONDS,
    SOURCE_CURSOR,
    SOURCE_STEP,
    WORLD_SIZE,
    build_manifest,
    checkpoint_digest,
    checkpoint_directory_name,
    checkpoint_metadata_v2,
    effective_wave_carried_cost,
    launch_attempt_digest,
    launch_digest,
    preflight_digest,
    rank_windows,
    render_segment_contract,
    require_slice_admission,
    select_slice_timeout_seconds,
    validate_early_stop_receipt,
    verify_checkpoint_manifest,
)
from meddies_pii.training.bioes.trainers.pii350_ddp_runtime import (
    FourRankPlan,
    ResumeState,
    begin_four_rank_resume,
    commit_four_rank_update,
    projected_budget_stop,
    require_a100_40gb,
    require_runtime_rank_rng_records,
)
from meddies_pii.training.bioes.trainers.pins import PACKED_UNIT_COUNT

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

    from meddies_pii.training.bioes.modal.probe_scaffold import ChildProcess


class _ReloadableVolume(Protocol):
    def reload(self) -> None: ...

    def commit(self) -> None: ...


class _CheckpointUploadApi(Protocol):
    def create_repo(self, *, repo_id: str, private: bool, exist_ok: bool) -> object: ...

    def upload_file(
        self,
        *,
        path_or_fileobj: str | BytesIO,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
    ) -> object: ...


class _CheckpointBackbone(Protocol):
    def save_pretrained(self, destination: Path) -> object: ...


class _CheckpointClassifier(Protocol):
    def state_dict(self) -> Mapping[str, Tensor]: ...


class _CheckpointTaggerModule(Protocol):
    @property
    def backbone(self) -> _CheckpointBackbone: ...

    @property
    def classifier(self) -> _CheckpointClassifier: ...


class _DistributedCheckpointTagger(Protocol):
    @property
    def module(self) -> _CheckpointTaggerModule: ...


class _TrainOptions(TypedDict):
    gpu: str | None
    cpu: float
    memory: int
    timeout: int
    max_containers: int


def train_options(profile: str) -> _TrainOptions:
    topology = EXECUTION_TOPOLOGIES[profile]
    gpu, cpu, memory, timeout = (
        topology["gpu"],
        topology["cpu"],
        topology["memory_mib"],
        topology["timeout_seconds"],
    )
    if (
        not isinstance(gpu, str)
        or not isinstance(cpu, (int, float))
        or not isinstance(memory, int)
        or not isinstance(timeout, int)
    ):
        msg = "continuation topology is invalid"
        raise RuntimeError(msg)
    return {
        "gpu": gpu,
        "cpu": float(cpu),
        "memory": memory,
        "timeout": timeout,
        "max_containers": 1,
    }


SHA256_HEX_LENGTH = 64
LABEL_IGNORE_INDEX = -100
TIMING_WINDOW_SIZE = 3

TRAIN_OPTIONS = train_options("meddiesresearch")
CPU_OPTIONS: _TrainOptions = {
    "gpu": None,
    "cpu": 2.0,
    "memory": 8 * 1024,
    "timeout": 1_800,
    "max_containers": 1,
}
ENCODER_FULL_RUN_IMAGE_PACKAGES: tuple[str, ...] = (
    "torch==2.10.0",
    "transformers==5.2.0",
    "peft==0.19.1",
    "pyarrow==23.0.0",
    "datasets==4.3.0",
    "huggingface_hub==1.11.0",
    "nvidia-ml-py==13.590.44",
    "unsloth==2026.7.4",
    "unsloth_zoo==2026.7.4",
)
"""The dependency inventory gate audits pins by reading this file's literals.

So the pins are spelled out here and checked against the renderer's contract below. A starred import would leave the
shipped image unaudited.

"""
if ENCODER_FULL_RUN_IMAGE_PACKAGES != CONTRACT_ENCODER_FULL_RUN_IMAGE_PACKAGES:
    msg = "Modal continuation image pins drifted from the rendered run contract"
    raise RuntimeError(msg)
CHECKPOINT_REPOSITORY = "Meddies/pii350-trajectories-private"
SOURCE_CHECKPOINT_ROOT = Path("/artifacts/full-runs/pii350/2e679221fd7c4ec6925e13f27cf96769/checkpoints/step-00000060")
ARTIFACT_ROOT = Path("/artifacts/pii350-ddp-continuation")
if PACKED_UNIT_COUNT != EPOCH_CURSOR:
    msg = "continuation epoch cursor must match the immutable packed artifact"
    raise RuntimeError(msg)
CACHE_ENVIRONMENT = hf_cache_environment()
HUB_TRANSPORT_ENVIRONMENT = {
    **CACHE_ENVIRONMENT,
    "HF_HUB_OFFLINE": "0",
    "HF_DATASETS_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
}
"""huggingface_hub reads these flags during import in each remote container."""
base_image = add_source_pythonpath(
    modal.Image.from_registry(
        "python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941",
    ).pip_install(*ENCODER_FULL_RUN_IMAGE_PACKAGES),
)
image = base_image.env(CACHE_ENVIRONMENT).add_local_dir("src", remote_path="/root/src")
"""Every branch performs its environment build step before adding local source.

Modal rejects further build steps after ``add_local_dir``.

"""
transport_image = base_image.env(HUB_TRANSPORT_ENVIRONMENT).add_local_dir("src", remote_path="/root/src")
app = modal.App("meddies-pii350-ddp-continuation")
cache = hf_cache_volume()
artifacts = artifact_volume()
secret = hf_secret()
MOUNTS: VolumeMounts = volume_mounts(cache, artifacts)


@dataclass(frozen=True, slots=True)
class ChildSpec:
    contract: dict[str, Any]
    source_receipt: dict[str, Any]
    source_root: str
    carried_cost_usd: float
    shard_paths: tuple[str, ...]
    encoder_checkpoint_attestation: dict[str, Any]
    artifact_dir: str
    started_monotonic: float
    slice_timeout_seconds: int
    world_size: int
    all_in_rate_usd_per_second: float


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


# reason: claim launch exposes artifact/volume as its Modal schema; bundling would break callers.
def _claim_launch_attempt(  # ruff: ignore[too-many-arguments]
    artifact_root: Path,
    *,
    attempt_digest: str,
    launch_attempt_nonce: str,
    profile: str,
    source_launch_digest: str,
    volume: _ReloadableVolume | None = None,
) -> Path:
    """Atomically admit one manually authorized attempt on the shared Volume.

    Returns:
        The artifact directory this attempt owns. The path is derived from ``attempt_digest``
        rather than allocated, so the claim and the directory it hands back cannot disagree.

    Raises:
        RuntimeError: If the claim directory already exists. ``mkdir`` without ``exist_ok`` is
            the admission itself: the filesystem decides which caller wins, so a second launch
            of the same attempt digest is refused rather than allowed to share the directory.

    """
    volume = artifacts if volume is None else volume
    volume.reload()
    claim_root = artifact_root / "launch-claims" / attempt_digest
    claim_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_root.mkdir()
    except FileExistsError as error:
        msg = "continuation launch attempt was already claimed"
        raise RuntimeError(msg) from error
    _write_json(
        claim_root / "claim.json",
        {
            "attempt_digest": attempt_digest,
            "launch_attempt_nonce": launch_attempt_nonce,
            "profile": profile,
            "source_launch_digest": source_launch_digest,
        },
    )
    volume.commit()
    return artifact_root / "launch-attempts" / attempt_digest


def _cpu_cost_fields(started_monotonic: float, *, now_monotonic: float | None = None) -> dict[str, float]:
    elapsed = (time.monotonic() if now_monotonic is None else now_monotonic) - started_monotonic
    if elapsed < 0:
        msg = "CPU receipt clock moved backwards"
        raise RuntimeError(msg)
    return {
        "cpu_elapsed_seconds": elapsed,
        "estimated_cpu_compute_cost_usd": elapsed * CPU_AUXILIARY_RATE_USD_PER_SECOND,
        "cpu_all_in_rate_usd_per_second": CPU_AUXILIARY_RATE_USD_PER_SECOND,
    }


def _with_cpu_cost(receipt: Mapping[str, Any], started_monotonic: float) -> dict[str, Any]:
    return {**receipt, "cpu_cost": _cpu_cost_fields(started_monotonic)}


# reason: ensure pii350 exposes baseline/preflight as its Modal schema; bundling would break callers.
def _ensure_pii350_baseline_assets(  # ruff: ignore[too-many-arguments]
    baseline_contract: Mapping[str, Any],
    *,
    receipt_path: Path,
    preflight_candidate_assets: Callable[..., object],
    persist_preflight_receipt: Callable[..., object],
    validated_receipt_inventory: Callable[..., tuple[tuple[str, ...], Mapping[str, Any] | None]],
    preflight_inputs: Mapping[str, Any],
) -> tuple[tuple[str, ...], Mapping[str, Any] | None, bool]:
    """Reuse a validated baseline receipt or build it within this CPU preflight.

    Returns:
        The verified shard paths, the encoder attestation, and whether this call built the
        assets rather than reusing them. The build flag is not diagnostic: the caller commits
        the Volume on it, so a reused receipt costs no commit and a fresh build always does.

    Raises:
        RuntimeError: If the preflight returns no verification receipt, or persists one to a
            path other than the contract's own. Both refuse the same substitution — accepting
            a receipt that was never written where the inventory will later read it.

    """
    built = False
    if not receipt_path.exists():
        result = preflight_candidate_assets("pii350", contract=baseline_contract, **preflight_inputs)
        receipt = result.get("receipt") if isinstance(result, Mapping) else None
        if not isinstance(receipt, Mapping):
            msg = "PII350 asset preflight produced no verification receipt"
            raise RuntimeError(msg)
        persisted_path = persist_preflight_receipt(baseline_contract, receipt)
        if persisted_path != receipt_path:
            msg = "PII350 asset preflight persisted to an unexpected receipt path"
            raise RuntimeError(msg)
        built = True
    paths, attestation = validated_receipt_inventory(baseline_contract)
    return paths, attestation, built


def _append_event(root: Path, event: Mapping[str, Any]) -> None:
    append_event(root, event)


def _manifest_with_digest(root: Path) -> dict[str, Any]:
    files = {
        str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    if not set(CHECKPOINT_REQUIRED_FILES) <= set(files):
        msg = "source checkpoint lacks canonical continuation payloads"
        raise RuntimeError(msg)
    manifest_body = build_manifest(files)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        msg = "checkpoint metadata is not an object"
        raise RuntimeError(msg)
    return {
        **manifest_body,
        "checkpoint_digest": checkpoint_digest(metadata, manifest_body),
    }


def verify_downloaded_checkpoint(root: Path, manifest: Mapping[str, Any], *, purpose: str) -> dict[str, Any]:
    if purpose not in {"training", "evaluation"}:
        msg = "checkpoint purpose must be training or evaluation"
        raise ValueError(msg)
    digest = verify_checkpoint_manifest(root, manifest)
    names = [str(item["path"]) for item in manifest["files"]]
    if not set(CHECKPOINT_REQUIRED_FILES) <= set(names):
        msg = "checkpoint receipt lacks files required for its load purpose"
        raise RuntimeError(msg)
    return {
        "purpose": purpose,
        "verified_files": names,
        "checkpoint_digest": digest,
        "manifest_sha256": sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }


def inventory_source_checkpoint(root: Path = SOURCE_CHECKPOINT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        msg = "continuation source checkpoint root is absent"
        raise RuntimeError(msg)
    files = {
        str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    manifest_body = build_manifest(files)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        msg = "source checkpoint metadata is not an object"
        raise RuntimeError(msg)
    step, cursor = metadata.get("optimizer_step"), metadata.get("packed_cursor")
    world_size = metadata.get("world_size", 1)
    partial_terminal = step == EPOCH_TERMINAL_STEP and cursor == EPOCH_CURSOR and metadata.get("epoch_complete") is True
    if type(step) is not int or type(cursor) is not int or (cursor != step * GLOBAL_BATCH_SIZE and not partial_terminal):
        msg = "source checkpoint has an invalid step/cursor lineage"
        raise RuntimeError(msg)
    if world_size not in {1, 2, 4}:
        msg = "source checkpoint has an unsupported world size"
        raise RuntimeError(msg)
    if step < SOURCE_STEP or cursor < SOURCE_CURSOR:
        msg = "source checkpoint predates the verified step-60 lineage"
        raise RuntimeError(msg)
    if metadata.get("checkpoint_digest") is not None:
        msg = "source checkpoint metadata contains a cyclic digest"
        raise RuntimeError(msg)
    manifest = {
        **manifest_body,
        "checkpoint_digest": checkpoint_digest(metadata, manifest_body),
    }
    verify_checkpoint_manifest(root, manifest)
    return {
        "hash_verified": True,
        "checkpoint_digest": manifest["checkpoint_digest"],
        "optimizer_step": step,
        "packed_cursor": cursor,
        "world_size": world_size,
        "epoch_complete": metadata.get("epoch_complete") is True,
        "wave_profile": metadata.get("wave_profile"),
        "wave_cumulative_all_in_cost_usd": metadata.get("wave_cumulative_all_in_cost_usd", 0.0),
        "manifest": manifest,
        "source_root": str(root),
    }


def _manifest_matches(root: Path, manifest: Mapping[str, Any]) -> None:
    """Re-inventory all payload bytes against one already validated manifest.

    Raises:
        RuntimeError: If the re-derived manifest differs from the validated one in any field,
            including the digest. The caller runs this before AND after uploading, so the
            check is what makes a mid-upload byte change fail rather than ship.

    """
    observed = _manifest_with_digest(root)
    if observed != manifest:
        msg = "checkpoint payload changed since validation"
        raise RuntimeError(msg)
    verify_checkpoint_manifest(root, manifest)


def upload_manifest_last(
    hf_api: _CheckpointUploadApi,
    *,
    root: Path,
    repo_id: str,
    remote_prefix: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Upload validated payloads, then commit their unchanged manifest last.

    Returns:
        The manifest with the commit SHA of its own upload appended. Because the manifest is
        written last, that SHA is the marker a downloader can pin: a revision carrying the
        manifest is a revision where every payload it lists already landed.

    Raises:
        RuntimeError: If the payload bytes stop matching the manifest before or after the
            payload uploads, or if the manifest upload returns no commit SHA. Without a SHA
            there is nothing immutable to pin, so a downloader could be handed a moving tag.

    """
    _manifest_matches(root, manifest)
    hf_api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    for item in manifest["files"]:
        path = root / str(item["path"])
        hf_api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{remote_prefix}/{item['path']}",
            repo_id=repo_id,
            repo_type="model",
        )
    _manifest_matches(root, manifest)
    commit = hf_api.upload_file(
        path_or_fileobj=BytesIO(json.dumps(manifest, sort_keys=True).encode()),
        path_in_repo=f"{remote_prefix}/manifest.json",
        repo_id=repo_id,
        repo_type="model",
    )
    commit_sha = getattr(commit, "oid", None)
    if not isinstance(commit_sha, str) or not commit_sha:
        msg = "continuation manifest upload did not return a commit SHA"
        raise RuntimeError(msg)
    return {**manifest, "manifest_commit_sha": commit_sha}


# reason: verify and render share validated's state; extraction would fragment diagnostics.
def _validated_manifest_for_upload(  # ruff: ignore[complex-structure]
    root: Path,
    *,
    profile: str = "meddiesresearch",
    source_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Use a persisted manifest or derive only the exact legacy source manifest.

    Returns:
        The manifest to upload and whether it was derived in memory rather than read from
        disk. Deriving is the narrow path: it is admitted only for the one verified step-60
        source that predates persisted manifests, and the flag records that it happened.

    Raises:
        RuntimeError: If the persisted manifest is not valid JSON or not an object; if a
            manifest is absent on any checkpoint other than that exact step-60 source; if the
            legacy path is taken without the prior hash-verified CPU receipt, or against a
            receipt whose digest or manifest disagrees with the freshly inventoried bytes; or
            if the checkpoint's world size does not match the selected profile. The last is
            waived only for the exact legacy source, which was written at world size 1.

    """
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            msg = "continuation checkpoint manifest is invalid JSON"
            raise RuntimeError(msg) from error
        if not isinstance(persisted, Mapping):
            msg = "continuation checkpoint manifest is not an object"
            raise RuntimeError(msg)
        verify_checkpoint_manifest(root, persisted)
        manifest, legacy_source = dict(persisted), False
    else:
        receipt = inventory_source_checkpoint(root)
        if receipt["optimizer_step"] != SOURCE_STEP or receipt["packed_cursor"] != SOURCE_CURSOR:
            msg = "only the verified legacy step-60 source may omit manifest.json"
            raise RuntimeError(msg)
        if not isinstance(source_receipt, Mapping):
            msg = "legacy source upload requires its prior hash-verified CPU receipt"
            raise RuntimeError(msg)
        if (
            source_receipt.get("hash_verified") is not True
            or source_receipt.get("checkpoint_digest") != receipt["checkpoint_digest"]
            or source_receipt.get("manifest") != receipt["manifest"]
        ):
            msg = "legacy source bytes do not match the prior hash-verified CPU receipt"
            raise RuntimeError(msg)
        legacy_manifest = receipt["manifest"]
        if not isinstance(legacy_manifest, Mapping):
            msg = "legacy source inventory produced no manifest"
            raise RuntimeError(msg)
        manifest, legacy_source = dict(legacy_manifest), True
    if isinstance(source_receipt, Mapping) and (
        source_receipt.get("checkpoint_digest") != manifest["checkpoint_digest"]
        or source_receipt.get("manifest") != manifest
    ):
        msg = "checkpoint bytes do not match the prior hash-verified CPU receipt"
        raise RuntimeError(msg)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        msg = "checkpoint metadata is not an object"
        raise RuntimeError(msg)
    observed_world_size = metadata.get("world_size", 1)
    expected_world_size = render_segment_contract(profile)["execution"]["world_size"]
    exact_legacy_source = (
        legacy_source
        and metadata.get("optimizer_step") == SOURCE_STEP
        and metadata.get("packed_cursor") == SOURCE_CURSOR
        and observed_world_size == 1
    )
    if observed_world_size != expected_world_size and not exact_legacy_source:
        msg = "checkpoint world size does not match the selected profile"
        raise RuntimeError(msg)
    return manifest, legacy_source


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _requested_artifact_root(value: object) -> str:
    if not isinstance(value, str):
        msg = "upload requested root is not a string"
        raise RuntimeError(msg)
    path = PurePosixPath(value)
    artifact_root = PurePosixPath(ARTIFACT_MOUNT)
    if str(path) != value or not path.is_absolute() or not path.is_relative_to(artifact_root):
        msg = "upload requested root is not an exact artifact-volume path"
        raise RuntimeError(msg)
    return value


def _resolved_artifact_root() -> str:
    return str(Path(ARTIFACT_MOUNT).resolve())


def _is_modal_volume_root(value: object) -> TypeGuard[str]:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return (
        str(path) == value
        and path.is_absolute()
        and len(path.parts) >= 4  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        and path.parts[:3] == ("/", "__modal", "volumes")
    )


def _source_root_binding(
    launch_digest_value: str,
    requested_root: str,
    resolved_artifact_root: str,
    resolved_root: str,
) -> str:
    return sha256(
        json.dumps(
            {
                "launch_digest": launch_digest_value,
                "requested_source_root": requested_root,
                "resolved_artifact_root": resolved_artifact_root,
                "resolved_source_root": resolved_root,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
    ).hexdigest()


# reason: source receipt coordinates render with requested; extra seams would detach artifact evidence.
def _source_receipt_from_cpu_receipt(encoded: str, *, profile: str, checkpoint_root: str) -> dict[str, Any]:  # ruff: ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
    """Validate one CPU receipt before dispatching the checkpoint transport.

    Returns:
        The source checkpoint receipt carried inside the CPU receipt, as a plain dict, once
        every field it will be trusted for has been re-derived rather than read. The launch
        digest is recomputed from that returned value, so the receipt identifies itself.

    Raises:
        RuntimeError: If the encoded receipt is not valid JSON or not an object; if it lacks
            ``source_checkpoint_receipt``; if its profile or execution contract digest differs
            from the freshly rendered contract; if the requested root does not match
            ``checkpoint_root``, does not resolve to a Modal volume path, or is not mapped
            from the artifact mount; if the root binding does not recompute; if the lineage,
            world size, or manifest is invalid, non-canonical, or missing the canonical
            payloads; or if the launch digest does not recompute over the returned receipt.
            Every branch refuses the same class of substitution — a receipt that describes a
            different checkpoint than the bytes the caller is about to transport.

    """
    try:
        cpu_receipt = json.loads(encoded)
    except json.JSONDecodeError as error:
        msg = "upload source receipt JSON is invalid"
        raise RuntimeError(msg) from error
    if not isinstance(cpu_receipt, Mapping):
        msg = "upload source receipt JSON is not an object"
        raise RuntimeError(msg)
    source = cpu_receipt.get("source_checkpoint_receipt")
    if not isinstance(source, Mapping):
        msg = "upload CPU receipt lacks source_checkpoint_receipt"
        raise RuntimeError(msg)
    contract = render_segment_contract(profile)
    if (
        cpu_receipt.get("profile") != profile
        or cpu_receipt.get("execution_contract_digest") != contract["execution_contract_digest"]
    ):
        msg = "upload CPU receipt does not match the requested profile contract"
        raise RuntimeError(msg)
    expected_root = _requested_artifact_root(checkpoint_root)
    if cpu_receipt.get("requested_source_root") != expected_root:
        msg = "upload requested root does not match checkpoint_root"
        raise RuntimeError(msg)
    required = (
        "hash_verified",
        "checkpoint_digest",
        "optimizer_step",
        "packed_cursor",
        "world_size",
        "epoch_complete",
        "manifest",
        "source_root",
    )
    if any(key not in source for key in required):
        msg = "upload source checkpoint receipt is incomplete"
        raise RuntimeError(msg)
    resolved_root = source.get("source_root")
    resolved_artifact_root = cpu_receipt.get("resolved_artifact_root")
    if not _is_modal_volume_root(resolved_root) or not _is_modal_volume_root(resolved_artifact_root):
        msg = "upload resolved root is not a Modal volume path"
        raise RuntimeError(msg)
    relative_root = PurePosixPath(expected_root).relative_to(PurePosixPath(ARTIFACT_MOUNT))
    if str(PurePosixPath(resolved_artifact_root) / relative_root) != resolved_root:
        msg = "upload resolved root is not mapped from checkpoint_root"
        raise RuntimeError(msg)
    launch = cpu_receipt.get("launch_digest")
    if not isinstance(launch, str) or cpu_receipt.get("source_root_binding") != _source_root_binding(
        launch,
        expected_root,
        resolved_artifact_root,
        resolved_root,
    ):
        msg = "upload CPU receipt root binding is invalid"
        raise RuntimeError(msg)
    step = source.get("optimizer_step")
    cursor = source.get("packed_cursor")
    partial_terminal = step == EPOCH_TERMINAL_STEP and cursor == EPOCH_CURSOR and source.get("epoch_complete") is True
    world_size = source.get("world_size")
    # reason: source receipt keeps hash/checkpoint in one gate; helper predicates would scatter the rule.
    if (
        source.get("hash_verified") is not True  # ruff: ignore[too-many-boolean-expressions]
        or not _is_sha256(source.get("checkpoint_digest"))
        or type(step) is not int
        or type(cursor) is not int
        or (cursor != step * GLOBAL_BATCH_SIZE and not partial_terminal)
        or step < SOURCE_STEP
        or cursor < SOURCE_CURSOR
        or type(world_size) is not int
        or world_size not in {1, 2, 4}
        or type(source.get("epoch_complete")) is not bool
    ):
        msg = "upload source checkpoint receipt has invalid lineage"
        raise RuntimeError(msg)
    exact_legacy_source = (
        step == SOURCE_STEP and cursor == SOURCE_CURSOR and world_size == 1 and source.get("epoch_complete") is False
    )
    expected_world_size = contract["execution"]["world_size"]
    if world_size != expected_world_size and not exact_legacy_source:
        msg = "upload source receipt world size does not match the selected profile"
        raise RuntimeError(msg)
    manifest = source.get("manifest")
    if not isinstance(manifest, Mapping):
        msg = "upload source checkpoint receipt lacks a manifest"
        raise RuntimeError(msg)
    expected_manifest_fields = {
        "schema_version",
        "committed",
        "files",
        "checkpoint_digest",
    }
    if set(manifest) != expected_manifest_fields:
        msg = "upload source receipt manifest is non-canonical"
        raise RuntimeError(msg)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("committed") is not True
        or manifest.get("checkpoint_digest") != source["checkpoint_digest"]
    ):
        msg = "upload source receipt digest does not match manifest"
        raise RuntimeError(msg)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        msg = "upload source receipt manifest has no files"
        raise RuntimeError(msg)
    paths: list[str] = []
    for item in files:
        if not isinstance(item, Mapping):
            msg = "upload source receipt manifest file is invalid"
            raise RuntimeError(msg)
        name, byte_count, digest = (
            item.get("path"),
            item.get("bytes"),
            item.get("sha256"),
        )
        # reason: source receipt keeps is/boundary in one gate; helper predicates would scatter the rule.
        if (
            not isinstance(name, str)  # ruff: ignore[too-many-boolean-expressions]
            or str(PurePosixPath(name)) != name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or type(byte_count) is not int
            or byte_count < 0
            or not _is_sha256(digest)
        ):
            msg = "upload source receipt manifest file is invalid"
            raise RuntimeError(msg)
        paths.append(name)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        msg = "upload source receipt manifest file ordering is invalid"
        raise RuntimeError(msg)
    if not set(CHECKPOINT_REQUIRED_FILES) <= set(paths):
        msg = "upload source receipt lacks canonical continuation payloads"
        raise RuntimeError(msg)
    source_receipt = dict(source)
    if launch != launch_digest(contract["execution_contract_digest"], source_receipt):
        msg = "upload CPU receipt launch digest is invalid"
        raise RuntimeError(msg)
    return source_receipt


def torchrun_command(spec_path: str, result_path: str, *, world_size: int = WORLD_SIZE) -> tuple[str, ...]:
    return (
        "torchrun",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "-m",
        "meddies_pii.training.bioes.modal.pii350_ddp_continuation",
        "--child-spec",
        spec_path,
        "--child-result",
        result_path,
    )


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _gather_objects(distributed: Any, value: Mapping[str, Any]) -> list[dict[str, Any]]:  # ruff: ignore[any-type]
    gathered: list[object] = [None] * int(distributed.get_world_size())
    distributed.all_gather_object(gathered, dict(value))
    if not all(isinstance(item, Mapping) for item in gathered):
        msg = "continuation rank gather returned an invalid record"
        raise RuntimeError(msg)
    return [dict(item) for item in gathered if isinstance(item, Mapping)]


def _rank_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    cursor: int,
    rank: int,
    world_size: int = WORLD_SIZE,
) -> list[Mapping[str, Any]]:
    if len(rows) not in {GLOBAL_BATCH_SIZE, FINAL_GLOBAL_BATCH_SIZE}:
        msg = "continuation requires either the full global batch or exact terminal partial batch"
        raise RuntimeError(msg)
    windows = rank_windows(cursor=cursor, global_batch_size=len(rows), world_size=world_size)
    start, end = windows[rank]
    local_rows = list(rows[start - cursor : end - cursor])
    expected_local_batch_size = len(rows) // world_size
    exact_terminal = cursor == EPOCH_CURSOR - FINAL_GLOBAL_BATCH_SIZE and len(rows) == FINAL_GLOBAL_BATCH_SIZE
    if len(local_rows) != expected_local_batch_size or (
        expected_local_batch_size != GLOBAL_BATCH_SIZE // world_size and not exact_terminal
    ):
        msg = "continuation rank received a short local packed batch"
        raise RuntimeError(msg)
    return local_rows


def _next_plan_or_terminal(next_state: ResumeState, *, execution_world_size: int = WORLD_SIZE) -> FourRankPlan | None:
    """Proceed from the epoch terminal to checkpointing, not resume validation.

    Returns:
        The next resume plan, or ``None`` once the cursor reaches the epoch terminal. ``None``
        is the terminal signal rather than an error: the caller keeps its existing plan and
        falls through to checkpointing, because validating a resume at the epoch boundary
        would reject a run that in fact finished.

    """
    return (
        None
        if next_state.cursor == EPOCH_CURSOR
        else begin_four_rank_resume(next_state, execution_world_size=execution_world_size)
    )


def _active_labels(labels: Tensor) -> Tensor:
    return (labels != LABEL_IGNORE_INDEX).sum()


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _require_rank_consensus_bool(torch: Any, distributed: Any, *, device: int, local_value: bool, gate: str) -> None:  # ruff: ignore[any-type]
    """Turn a local safety check into one runtime-wide failure before an update.

    Raises:
        RuntimeError: If any rank's local value was false. The MIN all-reduce is what makes
            this collective rather than local — one dissenting rank drives the reduced value
            to zero, so every rank raises together instead of one aborting into a hang.

    """
    consensus = torch.tensor(int(local_value), device=device, dtype=torch.int32)
    distributed.all_reduce(consensus, op=distributed.ReduceOp.MIN)
    if int(consensus.item()) != 1:
        msg = f"continuation {gate} failed runtime-rank consensus"
        raise RuntimeError(msg)


def _canonical_trainable_named_parameters(tagger: TrainableModule) -> list[tuple[str, Any]]:
    """Canonicalize diagnostic/reduction order without changing optimizer ordering.

    Returns:
        The trainable ``(name, parameter)`` pairs sorted by name. Sorting is what makes the
        order canonical ACROSS ranks, which is the precondition for comparing inventories and
        for bucketing gradients identically everywhere. The optimizer keeps its own order.

    Raises:
        RuntimeError: If no parameter requires a gradient, or if two trainable parameters
            share a name. Either would make the cross-rank comparison meaningless — an empty
            inventory matches anything, and a duplicate name cannot be aligned to one tensor.

    """
    module = getattr(tagger, "module", tagger)
    pairs = sorted(
        ((name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad),
        key=itemgetter(0),
    )
    names = [name for name, _parameter in pairs]
    if not pairs or len(names) != len(set(names)):
        msg = "continuation trainable parameter inventory is invalid"
        raise RuntimeError(msg)
    return pairs


def _parameter_metadata(name: str, parameter: Tensor, *, gradients: bool) -> dict[str, Any]:
    tensor = parameter.grad if gradients else parameter
    return {
        "name": name,
        "shape": list(parameter.shape),
        "requires_grad": bool(parameter.requires_grad),
        "tensor_present": tensor is not None,
        "selected_tensor_dtype": str(tensor.dtype) if tensor is not None else None,
        "selected_tensor_device_type": (str(tensor.device.type) if tensor is not None else None),
        "selected_tensor_layout": str(tensor.layout) if tensor is not None else None,
        "selected_tensor_numel": int(tensor.numel()) if tensor is not None else None,
    }


# reason: ordered and reference share require matching's state; extraction would fragment diagnostics.
def _require_matching_parameter_inventory(inventory_records: Sequence[Mapping[str, Any]], *, stage: str) -> list[str]:  # ruff: ignore[complex-structure]
    """Reject an inventory mismatch before any per-tensor NCCL collective.

    Returns:
        The agreed trainable parameter names in canonical order, taken from rank 0 only after
        every other rank's record was proven equal to it. The caller compares that list back
        against its own pairs, so agreement and ordering are checked separately.

    Raises:
        RuntimeError: If the record count is not the contract world size; if the rank labels
            are not exactly ``0..n-1``; if rank 0's parameter list is missing or empty; if any
            rank's list differs from it; if a metadata entry is not a mapping, carries an
            invalid name, or is not trainable; if two names collide; or if any rank reports a
            missing tensor. Running before the collectives is the point — a shape or ordering
            mismatch discovered inside an all-reduce hangs the runtime instead of raising.

    """
    if len(inventory_records) not in {2, 4}:
        msg = "continuation parameter inventory lacks a rank record"
        raise RuntimeError(msg)
    ordered = sorted(inventory_records, key=lambda record: int(record["rank"]))
    if [record.get("rank") for record in ordered] != list(range(len(ordered))):
        msg = "continuation parameter inventory rank records are invalid"
        raise RuntimeError(msg)
    reference = ordered[0].get("parameters")
    if not isinstance(reference, list) or not reference:
        msg = "continuation parameter inventory is empty"
        raise RuntimeError(msg)
    if any(record.get("parameters") != reference for record in ordered[1:]):
        msg = f"continuation parameter name/order/shape/dtype inventory differs across ranks at {stage}"
        raise RuntimeError(msg)
    names: list[str] = []
    missing: list[str] = []
    for metadata in reference:
        if not isinstance(metadata, Mapping):
            msg = "continuation parameter inventory contains invalid metadata"
            raise RuntimeError(msg)
        name = metadata.get("name")
        if not isinstance(name, str) or not name:
            msg = "continuation parameter inventory name is invalid"
            raise RuntimeError(msg)
        if metadata.get("requires_grad") is not True:
            msg = "continuation parameter inventory includes a non-trainable parameter"
            raise RuntimeError(msg)
        names.append(name)
        if metadata.get("tensor_present") is not True:
            missing.append(name)
    if len(names) != len(set(names)):
        msg = "continuation parameter inventory has duplicate trainable names"
        raise RuntimeError(msg)
    if missing:
        msg = f"continuation diagnostic tensors are missing at {stage}: {', '.join(missing)}"
        raise RuntimeError(msg)
    return names


def _persist_inventory_mismatch(
    *,
    root: Path,
    rank: int,
    stage: str,
    inventory_records: Sequence[Mapping[str, Any]],
) -> None:
    if rank == 0:
        _append_event(
            root,
            {
                "event": "parameter_inventory_mismatch",
                "stage": stage,
                "rank_inventory": sorted(
                    (dict(record) for record in inventory_records),
                    key=lambda record: int(record["rank"]),
                ),
            },
        )
        artifacts.commit()


# reason: verified exposes distributed/gradients as its Modal schema; bundling would break callers.
# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _verified_trainable_pairs(  # ruff: ignore[too-many-arguments]
    distributed: Any,  # ruff: ignore[any-type]
    tagger: TrainableModule,
    *,
    root: Path,
    rank: int,
    stage: str,
    gradients: bool,
) -> list[tuple[str, Any]]:
    pairs = _canonical_trainable_named_parameters(tagger)
    inventory_records = _gather_objects(
        distributed,
        {
            "rank": rank,
            "parameters": [_parameter_metadata(name, parameter, gradients=gradients) for name, parameter in pairs],
        },
    )
    try:
        names = _require_matching_parameter_inventory(inventory_records, stage=stage)
    except RuntimeError:
        _persist_inventory_mismatch(
            root=root,
            rank=rank,
            stage=stage,
            inventory_records=inventory_records,
        )
        raise
    if names != [name for name, _parameter in pairs]:
        msg = "continuation canonical parameter order changed during inventory"
        raise RuntimeError(msg)
    return pairs


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _stage_progress(distributed: Any, *, root: Path, rank: int, stage: str, boundary: str) -> None:  # ruff: ignore[any-type]
    records = _gather_objects(
        distributed,
        {"rank": rank, "stage": stage, "boundary": boundary},
    )
    if rank == 0:
        _append_event(
            root,
            {
                "event": "stage_progress",
                "stage": stage,
                "boundary": boundary,
                "rank_records": sorted(records, key=lambda record: int(record["rank"])),
            },
        )


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _mean_gradients(
    torch: Any,  # ruff: ignore[any-type]
    distributed: Any,  # ruff: ignore[any-type]
    tagger: TrainableModule,
    *,
    root: Path,
    rank: int,
) -> None:
    """Use canonical SUM/runtime-world-size buckets after inventory consensus.

    Raises:
        RuntimeError: If a trainable parameter has no gradient, or, from the inventory
            consensus this calls first, if the ranks disagree on parameter names, order,
            shapes or dtypes. Dividing by the RUNTIME world size rather than the contract one
            is deliberate: the mean must match the ranks that actually contributed to the SUM.

    """
    pairs = _verified_trainable_pairs(
        distributed,
        tagger,
        root=root,
        rank=rank,
        stage="pre_gradient_mean",
        gradients=True,
    )
    grouped: dict[tuple[str, str], list[Any]] = {}
    for _name, parameter in pairs:
        if parameter.grad is None:
            msg = "continuation has a missing trainable gradient"
            raise RuntimeError(msg)
        grouped.setdefault((str(parameter.grad.device.type), str(parameter.grad.dtype)), []).append(parameter.grad)
    for _bucket, gradients in sorted(grouped.items()):
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        distributed.all_reduce(flat, op=distributed.ReduceOp.SUM)
        flat.div_(distributed.get_world_size())
        offset = 0
        for gradient in gradients:
            next_offset = offset + gradient.numel()
            gradient.copy_(flat[offset:next_offset].reshape_as(gradient))
            offset = next_offset


# reason: max trainable exposes torch/gradients as its Modal schema; bundling would break callers.
# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _max_trainable_drift(  # ruff: ignore[too-many-arguments]
    torch: Any,  # ruff: ignore[any-type]
    distributed: Any,  # ruff: ignore[any-type]
    tagger: TrainableModule,
    *,
    root: Path,
    rank: int,
    stage: str,
    gradients: bool,
) -> float:
    pairs = _verified_trainable_pairs(
        distributed,
        tagger,
        root=root,
        rank=rank,
        stage=stage,
        gradients=gradients,
    )
    maximum = 0.0
    device: Any | None = None
    for _name, parameter in pairs:
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            msg = "continuation diagnostic tensor is absent"
            raise RuntimeError(msg)
        device = tensor.device
        reference = tensor.detach().clone()
        distributed.broadcast(reference, src=0)
        maximum = max(
            maximum,
            float((tensor.detach().float() - reference.float()).abs().max().item()),
        )
    if device is None:
        msg = "continuation has no trainable tensors to diagnose"
        raise RuntimeError(msg)
    maximum_tensor = torch.tensor(maximum, device=device, dtype=torch.float64)
    distributed.all_reduce(maximum_tensor, op=distributed.ReduceOp.MAX)
    return float(maximum_tensor.item())


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _require_collective_ready(torch: Any, distributed: Any, *, rank: int, local_rank: int) -> dict[str, Any]:  # ruff: ignore[any-type]
    world_size = int(distributed.get_world_size())
    peers = [peer for peer in range(world_size) if peer != rank]
    peer_access = {peer: bool(torch.cuda.can_device_access_peer(local_rank, peer)) for peer in peers}
    records = _gather_objects(distributed, {"rank": rank, "peers": peer_access})
    if len(records) != world_size or any(not all(bool(value) for value in record["peers"].values()) for record in records):
        msg = "continuation requires CUDA peer access between every rank"
        raise RuntimeError(msg)
    probe = torch.tensor(rank + 1, device=local_rank, dtype=torch.int64)
    distributed.all_reduce(probe, op=distributed.ReduceOp.SUM)
    if int(probe.item()) != sum(range(1, world_size + 1)):
        msg = "continuation NCCL all-reduce readiness probe failed"
        raise RuntimeError(msg)
    return {"peer_access": records, "all_reduce_sum": int(probe.item())}


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _rank_rng_policy(
    torch: Any,  # ruff: ignore[any-type]
    *,
    receipt: Mapping[str, Any],
    rank: int,
    execution_world_size: int,
) -> Callable[[Mapping[str, Any]], None]:
    def apply(rng: Mapping[str, Any]) -> None:
        source_world_size = int(receipt.get("world_size", 0))
        if source_world_size != execution_world_size:
            from meddies_pii.training.bioes.trainers.pii350_ddp_continuation import (
                derived_rank_seed,
            )

            seed = derived_rank_seed(
                str(receipt["checkpoint_digest"]),
                rank,
                world_size=execution_world_size,
            )
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            return
        states = rng.get("rank_rng_states")
        if not isinstance(states, list):
            msg = "same-topology continuation checkpoint has no rank RNG records"
            raise RuntimeError(msg)
        require_runtime_rank_rng_records(states, world_size=execution_world_size)
        record = states[rank]
        cpu = torch.tensor(list(bytes.fromhex(str(record["cpu_rng"]))), dtype=torch.uint8)
        cuda = torch.tensor(list(bytes.fromhex(str(record["cuda_rng"]))), dtype=torch.uint8)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda, device=rank)

    return apply


# reason: save checkpoint exposes root/source world as its Modal schema; bundling would break callers.
# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _save_checkpoint(  # ruff: ignore[too-many-arguments]
    *,
    root: Path,
    tagger: _DistributedCheckpointTagger,
    optimizer: Optimizer,
    contract: Mapping[str, Any],
    parent_checkpoint_digest: str,
    step: int,
    cursor: int,
    terminal: bool,
    wave_profile: str,
    wave_cumulative_all_in_cost_usd: float,
    distributed: Any,  # ruff: ignore[any-type]
    torch: Any,  # ruff: ignore[any-type]
    rank: int,
    execution_world_size: int,
    source_world_size: int,
) -> dict[str, Any] | None:
    local_rng = {
        "rank": rank,
        "cpu_rng": bytes(torch.get_rng_state().cpu().tolist()).hex(),
        "cuda_rng": bytes(torch.cuda.get_rng_state(rank).cpu().tolist()).hex(),
    }
    records = sorted(_gather_objects(distributed, local_rng), key=lambda value: int(value["rank"]))
    require_runtime_rank_rng_records(records, world_size=execution_world_size)
    distributed.barrier()
    result: dict[str, Any] | None = None
    if rank == 0:
        checkpoint_name = checkpoint_directory_name(step=step, terminal=terminal)
        final_root = root / "checkpoints" / checkpoint_name
        temporary = final_root.with_name(final_root.name + ".tmp")
        if temporary.exists() or final_root.exists():
            msg = "continuation checkpoint destination already exists"
            raise RuntimeError(msg)
        temporary.mkdir(parents=True)
        # reason: save checkpoint's try keeps write json with verify; splitting would split cleanup from writes.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            tagger.module.backbone.save_pretrained(temporary / "adapter")
            torch.save(tagger.module.classifier.state_dict(), temporary / "classifier.pt")
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            torch.save({"rank_rng_states": records}, temporary / "rng.pt")
            metadata = checkpoint_metadata_v2(
                contract,
                parent_checkpoint_digest=parent_checkpoint_digest,
                step=step,
                cursor=cursor,
                lifecycle_state="terminal" if terminal else "update_committed",
                rank_rng_states=records,
                execution_world_size=execution_world_size,
                source_world_size=source_world_size,
                wave_profile=wave_profile,
                wave_cumulative_all_in_cost_usd=wave_cumulative_all_in_cost_usd,
            )
            _write_json(temporary / "metadata.json", metadata)
            manifest = _manifest_with_digest(temporary)
            _write_json(temporary / "manifest.json", manifest)
            verify_checkpoint_manifest(temporary, manifest)
            temporary.replace(final_root)
            artifacts.commit()
            result = {
                "path": str(final_root),
                "manifest": manifest,
                "checkpoint_digest": manifest["checkpoint_digest"],
                "step": step,
                "cursor": cursor,
                "terminal": terminal,
            }
        except Exception:
            if temporary.exists():
                import shutil

                shutil.rmtree(temporary)
            raise
    result_records = _gather_objects(distributed, {"rank": rank, "checkpoint": result})
    distributed.barrier()
    checkpoints = [record["checkpoint"] for record in result_records if record["checkpoint"] is not None]
    if len(checkpoints) != 1 or not isinstance(checkpoints[0], Mapping):
        msg = "continuation checkpoint consensus failed"
        raise RuntimeError(msg)
    return dict(checkpoints[0])


# reason: run worker owns validate and write json together; splitting would fragment diagnostics.
def _run_worker(spec: ChildSpec) -> dict[str, Any]:  # ruff: ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
    """``_build_tagger`` uses ``.cuda()``.

    Select this sibling process's device before construction so runtime ranks never pile onto cuda:0.

    Restore model/head/optimizer bytes before selecting rank-local RNG streams for the execution topology.

    The terminal checkpoint exists even if the budget prevents another step.

    Returns:
        This rank's run outcome: ``status`` ``"ok"``, the final step and cursor, the
        ``terminal_reason`` that ended the loop, every checkpoint written, the observed
        topology, collective-readiness and runtime attestations, the document-isolation proof,
        and the elapsed/estimated/wave-cumulative cost fields. The cost fields are the record
        the next wave's carried budget is computed from, so they are outcome rather than
        telemetry. ``terminal_reason`` distinguishes finishing the packed epoch from stopping
        against the budget ceiling, the deadline reserve, or an exhausted packed stream — all
        four return normally, and only the first means the epoch is done.

    Raises:
        RuntimeError: If torchrun did not supply ranks matching the contract world size, if
            the spec carries no CPU-verified packed shards, if the source receipt lineage is
            invalid, if a rank-consensus or parameter-inventory check fails, if drift is
            non-zero after the optimizer, or if a checkpoint was not returned after rank
            consensus. Every one of these raises on ALL ranks by construction — the collective
            checks reduce before they decide, so a single-rank abort into a hang is not a
            reachable outcome. The process group is destroyed in ``finally`` regardless.

    """
    import torch
    from torch import distributed
    from torch.nn.parallel import DistributedDataParallel

    from meddies_pii.training.bioes.modal.pii350_ddp_smoke import (
        _require_document_isolation,
    )
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _iter_packed_rows,
        _packed_batch,
        _validate_runtime_label_vocabulary,
        load_verified_pii_resume_state,
    )
    from meddies_pii.training.bioes.trainers.pii350_ddp_runtime import (
        load_pii350_composed_runtime,
    )

    rank, local_rank = (
        int(os.environ.get("RANK", "-1")),
        int(os.environ.get("LOCAL_RANK", "-1")),
    )
    execution_world_size = int(spec.world_size)
    if (
        execution_world_size not in {2, 4}
        or rank not in range(execution_world_size)
        or local_rank not in range(execution_world_size)
    ):
        msg = "torchrun did not provide the contract local ranks"
        raise RuntimeError(msg)
    if not spec.shard_paths:
        msg = "continuation requires CPU-verified packed shards"
        raise RuntimeError(msg)
    partial_terminal = (
        spec.source_receipt.get("optimizer_step") == EPOCH_TERMINAL_STEP
        and spec.source_receipt.get("packed_cursor") == EPOCH_CURSOR
        and spec.source_receipt.get("epoch_complete") is True
    )
    if (
        type(spec.source_receipt.get("optimizer_step")) is not int
        or type(spec.source_receipt.get("packed_cursor")) is not int
        or (
            spec.source_receipt["packed_cursor"] != spec.source_receipt["optimizer_step"] * GLOBAL_BATCH_SIZE
            and not partial_terminal
        )
        or spec.source_receipt.get("world_size") not in {1, 2, 4}
    ):
        msg = "continuation child received an invalid source receipt"
        raise RuntimeError(msg)
    profile = str(spec.contract["execution"]["profile"])
    root = Path(spec.artifact_dir)
    carried_cost_usd = spec.carried_cost_usd
    if carried_cost_usd < 0:
        msg = "continuation child received a negative carried cost"
        raise RuntimeError(msg)
    torch.cuda.set_device(local_rank)
    topology = require_a100_40gb(torch, world_size=execution_world_size)
    # reason: torch declares the torch.distributed members behind an availability gate, so no static
    # reason: reader can prove they are present; this path runs only inside the multi-rank container.
    distributed.init_process_group("nccl")  # ty: ignore[possibly-missing-attribute]
    try:
        # reason: torch declares the torch.distributed members behind an availability gate, so no static
        # reason: reader can prove they are present; this path runs only inside the multi-rank container.
        if (
            distributed.get_world_size() != execution_world_size  # ty: ignore[possibly-missing-attribute]
            or distributed.get_rank() != rank  # ty: ignore[possibly-missing-attribute]
        ):
            msg = "continuation initialized the wrong distributed topology"
            raise RuntimeError(msg)
        collective = _require_collective_ready(torch, distributed, rank=rank, local_rank=local_rank)
        _stage_progress(distributed, root=root, rank=rank, stage="model_load", boundary="before")
        _validate_runtime_label_vocabulary(spec.contract["trajectory"]["baseline_full_run_contract"])
        tagger, _tokenizer, runtime_attestation = load_pii350_composed_runtime(
            spec.contract["trajectory"],
            checkpoint_attestation=spec.encoder_checkpoint_attestation,
        )
        _stage_progress(distributed, root=root, rank=rank, stage="model_load", boundary="after")
        if not tagger.packed_segment_isolation:
            msg = "continuation tagger lost packed document isolation"
            raise RuntimeError(msg)
        _stage_progress(
            distributed,
            root=root,
            rank=rank,
            stage="document_isolation",
            boundary="before",
        )
        isolation = _require_document_isolation(tagger, spec.shard_paths, device=f"cuda:{local_rank}")
        _stage_progress(
            distributed,
            root=root,
            rank=rank,
            stage="document_isolation",
            boundary="after",
        )
        _stage_progress(distributed, root=root, rank=rank, stage="ddp_wrap", boundary="before")
        tagger = DistributedDataParallel(
            tagger,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        _stage_progress(distributed, root=root, rank=rank, stage="ddp_wrap", boundary="after")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in tagger.parameters() if parameter.requires_grad],
            lr=4e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            fused=False,
        )
        _stage_progress(distributed, root=root, rank=rank, stage="source_restore", boundary="before")
        restored = load_verified_pii_resume_state(
            spec.source_root,
            tagger=tagger.module,
            optimizer=optimizer,
            verified_metadata=spec.source_receipt,
            rank_local_rng_policy=_rank_rng_policy(
                torch,
                receipt=spec.source_receipt,
                rank=rank,
                execution_world_size=execution_world_size,
            ),
        )
        _stage_progress(distributed, root=root, rank=rank, stage="source_restore", boundary="after")
        plan = begin_four_rank_resume(
            ResumeState(
                step=int(restored["optimizer_step"]),
                cursor=int(restored["packed_cursor"]),
                parent_checkpoint_digest=str(spec.source_receipt["checkpoint_digest"]),
                source_world_size=int(spec.source_receipt["world_size"]),
            ),
            execution_world_size=execution_world_size,
        )
        if plan.windows != tuple(rank_windows(cursor=plan.cursor, world_size=execution_world_size)):
            msg = "continuation ranks do not cover the source cursor's next global window"
            raise RuntimeError(msg)
        _stage_progress(
            distributed,
            root=root,
            rank=rank,
            stage="post_restore_inventory",
            boundary="before",
        )
        # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
        # reason: equality, so any tolerance here would let a real drift or contamination pass the gate.
        if (
            _max_trainable_drift(  # ruff: ignore[float-equality-comparison]
                torch,
                distributed,
                tagger,
                root=root,
                rank=rank,
                stage="post_restore_parameters",
                gradients=False,
            )
            != 0.0
        ):
            msg = "continuation replicas drifted after source restore"
            raise RuntimeError(msg)
        _stage_progress(
            distributed,
            root=root,
            rank=rank,
            stage="post_restore_inventory",
            boundary="after",
        )
        iterator = islice(_iter_packed_rows(spec.shard_paths), plan.cursor, None)
        current_step, current_cursor = plan.step, plan.cursor
        durations: list[float] = []
        checkpoints: list[dict[str, Any]] = []
        parent_checkpoint_digest = str(spec.source_receipt["checkpoint_digest"])
        checkpoint_parent_world_size = int(spec.source_receipt["world_size"])
        terminal_reason = "packed_epoch_complete"
        while current_cursor < EPOCH_CURSOR:
            elapsed = time.monotonic() - spec.started_monotonic
            predicted = float(median(durations[-3:])) if len(durations) >= TIMING_WINDOW_SIZE else 0.0
            ceiling = float(spec.contract["execution"]["all_in_ceiling_usd"])
            if carried_cost_usd >= ceiling:
                terminal_reason = "wave_cumulative_budget_ceiling"
                break
            if elapsed >= spec.slice_timeout_seconds - SHUTDOWN_RESERVE_SECONDS or projected_budget_stop(
                worker_elapsed_seconds=elapsed,
                predicted_step_seconds=predicted,
                all_in_rate_usd_per_second=spec.all_in_rate_usd_per_second,
                ceiling_usd=ceiling - carried_cost_usd,
                persistence_reserve_seconds=SHUTDOWN_RESERVE_SECONDS,
            ):
                terminal_reason = "budget_or_deadline_reserve"
                break
            consumed_units = min(GLOBAL_BATCH_SIZE, EPOCH_CURSOR - current_cursor)
            rows = list(islice(iterator, consumed_units))
            if len(rows) != consumed_units:
                terminal_reason = "packed_stream_exhausted"
                break
            local_rows = _rank_batch(rows, cursor=current_cursor, rank=rank, world_size=execution_world_size)
            batch, local_tokens = _packed_batch(local_rows, f"cuda:{local_rank}")
            local_active = _active_labels(batch["labels"])
            global_active = local_active.clone()
            # reason: torch declares the torch.distributed members behind an availability gate, so no static
            # reason: reader can prove they are present; this path runs only inside the multi-rank container.
            distributed.all_reduce(global_active, op=distributed.ReduceOp.SUM)  # ty: ignore[possibly-missing-attribute]
            if int(local_active.item()) <= 0 or int(global_active.item()) <= 0:
                msg = "continuation received an inactive-label local/global batch"
                raise RuntimeError(msg)
            first_update = not durations
            if first_update:
                _stage_progress(
                    distributed,
                    root=root,
                    rank=rank,
                    stage="first_update",
                    boundary="before",
                )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(local_rank)
            started = time.monotonic()
            with tagger.no_sync():
                output = tagger(**batch)
                loss = output.get("loss")
                _require_rank_consensus_bool(
                    torch,
                    distributed,
                    device=local_rank,
                    local_value=loss is not None and bool(torch.isfinite(loss).all()),
                    gate="finite-loss",
                )
                if loss is None:
                    msg = "continuation model returned no loss"
                    raise RuntimeError(msg)
                weighted_loss = loss * (execution_world_size * local_active.to(loss.dtype) / global_active.to(loss.dtype))
                _require_rank_consensus_bool(
                    torch,
                    distributed,
                    device=local_rank,
                    local_value=bool(torch.isfinite(weighted_loss).all()),
                    gate="finite-weighted-loss",
                )
                weighted_loss.backward()
            local_gradients = [parameter.grad for parameter in tagger.module.parameters() if parameter.requires_grad]
            _require_rank_consensus_bool(
                torch,
                distributed,
                device=local_rank,
                local_value=bool(local_gradients)
                and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in local_gradients),
                gate="finite-gradients",
            )
            _mean_gradients(torch, distributed, tagger, root=root, rank=rank)
            gradient_drift = _max_trainable_drift(
                torch,
                distributed,
                tagger,
                root=root,
                rank=rank,
                stage="post_gradient_mean",
                gradients=True,
            )
            # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
            # reason: equality, so any tolerance here would let a real drift or contamination pass the gate.
            if gradient_drift != 0.0:  # ruff: ignore[float-equality-comparison]
                msg = "continuation has non-zero gradient drift after explicit SUM/runtime-world-size"
                raise RuntimeError(msg)
            optimizer.step()
            parameter_drift = _max_trainable_drift(
                torch,
                distributed,
                tagger,
                root=root,
                rank=rank,
                stage="post_optimizer_step",
                gradients=False,
            )
            # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
            # reason: equality, so any tolerance here would let a real drift or contamination pass the gate.
            if parameter_drift != 0.0:  # ruff: ignore[float-equality-comparison]
                msg = "continuation has non-zero parameter drift after optimizer"
                raise RuntimeError(msg)
            if first_update:
                _stage_progress(
                    distributed,
                    root=root,
                    rank=rank,
                    stage="first_update",
                    boundary="after",
                )
            torch.cuda.synchronize(local_rank)
            duration = time.monotonic() - started
            durations.append(duration)
            consensus = _gather_objects(
                distributed,
                {"rank": rank, "ok": True, "loss": float(loss.detach().float().cpu())},
            )
            next_state = commit_four_rank_update(
                plan,
                finite_loss=True,
                finite_gradients=True,
                gradient_drift=gradient_drift,
                parameter_drift=parameter_drift,
                rank_successes=tuple(
                    bool(record["ok"]) for record in sorted(consensus, key=lambda record: int(record["rank"]))
                ),
                consumed_units=consumed_units,
            )
            current_step, current_cursor = next_state.step, next_state.cursor
            next_plan = _next_plan_or_terminal(next_state, execution_world_size=execution_world_size)
            if next_plan is not None:
                plan = next_plan
            elapsed = time.monotonic() - spec.started_monotonic
            rank_telemetry = _gather_objects(
                distributed,
                {
                    "rank": rank,
                    "tokens": local_tokens,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
                    "step_seconds": duration,
                },
            )
            synchronized_duration = max(float(record["step_seconds"]) for record in rank_telemetry)
            real_tokens_per_second = sum(int(record["tokens"]) for record in rank_telemetry) / synchronized_duration
            terminal = current_cursor == EPOCH_CURSOR
            checkpoint: dict[str, Any] | None = None
            if current_step % 50 == 0 or terminal:
                checkpoint = _save_checkpoint(
                    root=Path(spec.artifact_dir),
                    tagger=tagger,
                    optimizer=optimizer,
                    contract=spec.contract,
                    parent_checkpoint_digest=parent_checkpoint_digest,
                    step=current_step,
                    cursor=current_cursor,
                    terminal=terminal,
                    wave_profile=profile,
                    wave_cumulative_all_in_cost_usd=carried_cost_usd + elapsed * spec.all_in_rate_usd_per_second,
                    distributed=distributed,
                    torch=torch,
                    rank=rank,
                    execution_world_size=execution_world_size,
                    source_world_size=checkpoint_parent_world_size,
                )
                if checkpoint is None:
                    msg = "continuation checkpoint was not returned after rank consensus"
                    raise RuntimeError(msg)
                checkpoints.append(checkpoint)
                parent_checkpoint_digest = str(checkpoint["checkpoint_digest"])
                checkpoint_parent_world_size = execution_world_size
                if rank == 0:
                    _write_json(
                        Path(spec.artifact_dir) / "last-progress.json",
                        {
                            "step": current_step,
                            "cursor": current_cursor,
                            "checkpoint": checkpoint,
                            "elapsed_seconds": elapsed,
                            "projected_cost_usd": elapsed * spec.all_in_rate_usd_per_second,
                        },
                    )
            if rank == 0:
                _append_event(
                    Path(spec.artifact_dir),
                    {
                        "event": "optimizer_step",
                        "step": current_step,
                        "cursor": current_cursor,
                        "rank_windows": rank_windows(
                            cursor=current_cursor - consumed_units,
                            global_batch_size=consumed_units,
                            world_size=execution_world_size,
                        ),
                        "loss": float(loss.detach().float().cpu()),
                        "real_tokens_per_second": real_tokens_per_second,
                        "step_seconds": synchronized_duration,
                        "rank_telemetry": rank_telemetry,
                        "elapsed_seconds": elapsed,
                        "projected_cost_usd": elapsed * spec.all_in_rate_usd_per_second,
                        "wave_cumulative_cost_usd": carried_cost_usd + elapsed * spec.all_in_rate_usd_per_second,
                        "remaining_budget_usd": ceiling - carried_cost_usd - elapsed * spec.all_in_rate_usd_per_second,
                        "gradient_drift": gradient_drift,
                        "parameter_drift": parameter_drift,
                        "checkpoint": checkpoint,
                    },
                )
        if not checkpoints or checkpoints[-1]["step"] != current_step or not checkpoints[-1]["terminal"]:
            checkpoint = _save_checkpoint(
                root=Path(spec.artifact_dir),
                tagger=tagger,
                optimizer=optimizer,
                contract=spec.contract,
                parent_checkpoint_digest=parent_checkpoint_digest,
                step=current_step,
                cursor=current_cursor,
                terminal=True,
                wave_profile=profile,
                wave_cumulative_all_in_cost_usd=carried_cost_usd
                + (time.monotonic() - spec.started_monotonic) * spec.all_in_rate_usd_per_second,
                distributed=distributed,
                torch=torch,
                rank=rank,
                execution_world_size=execution_world_size,
                source_world_size=checkpoint_parent_world_size,
            )
            if checkpoint is None:
                msg = "terminal continuation checkpoint was not returned after rank consensus"
                raise RuntimeError(msg)
            checkpoints.append(checkpoint)
            parent_checkpoint_digest = str(checkpoint["checkpoint_digest"])
            if rank == 0:
                _write_json(
                    Path(spec.artifact_dir) / "last-progress.json",
                    {
                        "step": current_step,
                        "cursor": current_cursor,
                        "checkpoint": checkpoint,
                        "elapsed_seconds": time.monotonic() - spec.started_monotonic,
                        "projected_cost_usd": (time.monotonic() - spec.started_monotonic)
                        * spec.all_in_rate_usd_per_second,
                    },
                )
        if rank == 0:
            _append_event(
                Path(spec.artifact_dir),
                {
                    "event": "terminal",
                    "reason": terminal_reason,
                    "step": current_step,
                    "cursor": current_cursor,
                },
            )
        # reason: torch declares the torch.distributed members behind an availability gate, so no static
        # reason: reader can prove they are present; this path runs only inside the multi-rank container.
        distributed.barrier()  # ty: ignore[possibly-missing-attribute]
        return {
            "status": "ok",
            "step": current_step,
            "cursor": current_cursor,
            "terminal_reason": terminal_reason,
            "checkpoints": checkpoints,
            "topology": topology,
            "collective_readiness": collective,
            "runtime_attestation": runtime_attestation,
            "document_isolation_proof": isolation,
            "elapsed_seconds": time.monotonic() - spec.started_monotonic,
            "estimated_all_in_cost_usd": (time.monotonic() - spec.started_monotonic) * spec.all_in_rate_usd_per_second,
            "wave_cumulative_all_in_cost_usd": carried_cost_usd
            + (time.monotonic() - spec.started_monotonic) * spec.all_in_rate_usd_per_second,
        }
    finally:
        # reason: torch declares the torch.distributed members behind an availability gate, so no static
        # reason: reader can prove they are present; this path runs only inside the multi-rank container.
        distributed.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def _child_main(spec_path: str, result_path: str) -> int:
    spec = ChildSpec(**json.loads(Path(spec_path).read_text(encoding="utf-8")))
    try:
        result = _run_worker(spec)
    except Exception as error:
        if os.environ.get("RANK", "0") == "0":
            _write_json(
                Path(result_path),
                {
                    "status": "failed",
                    "error": repr(error),
                    "artifact_root": spec.artifact_dir,
                },
            )
        raise
    if os.environ.get("RANK", "0") == "0":
        _write_json(Path(result_path), result)
    return 0


def _stream_child_output(process: ChildProcess) -> tuple[int, str, str]:
    return stream_child_output(
        process,
        stdout_sink=print_stdout_sink,
        stderr_sink=print_stderr_sink,
        capture_stdout=True,
    )


def _run_isolated_torchrun(spec: ChildSpec) -> dict[str, Any]:
    root = Path(spec.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec_path, result_path = root / "child.spec.json", root / "result.json"
    _write_json(spec_path, asdict(spec))
    command = torchrun_command(str(spec_path), str(result_path), world_size=spec.world_size)
    _append_event(root, {"event": "torchrun_starting", "command": list(command)})
    try:
        # reason: torchrun_command builds fixed flags; spec/result paths are single operands and no shell is involved.
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        msg = f"continuation torchrun process could not start: {error!r}"
        raise RuntimeError(msg) from error
    returncode, stdout, stderr = _stream_child_output(process)
    if not result_path.is_file():
        _write_json(
            result_path,
            {
                "status": "failed",
                "error": f"torchrun exited {returncode}",
                "artifact_root": str(root),
            },
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        msg = "continuation torchrun result was not a JSON object"
        raise RuntimeError(msg)
    outcome = {
        **result,
        "torchrun_returncode": returncode,
        "torchrun_stdout_tail": stdout[-4_000:],
        "torchrun_stderr_tail": stderr[-4_000:],
    }
    _write_json(result_path, outcome)
    _append_event(
        root,
        {
            "event": "torchrun_completed" if returncode == 0 and outcome.get("status") == "ok" else "torchrun_failed",
            **outcome,
        },
    )
    artifacts.commit()
    if returncode != 0 or outcome.get("status") != "ok":
        msg = f"continuation torchrun failed: {outcome}"
        raise RuntimeError(msg)
    return outcome


@app.function(image=image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)
def cpu_preflight(profile: str) -> dict[str, Any]:
    """Run now on either profile.  It has no dependency on a future parent checkpoint.

    Returns:
        The CPU preflight receipt: the profile, its execution contract digest, the eval row
        count, the verified packed shards with the digest over their paths, the encoder
        checkpoint attestation, the ``preflight_digest`` binding all of it together, and the
        CPU cost fields. The paid launch re-derives that digest from freshly inventoried
        assets and refuses if it differs, so this receipt is the thing being bound, not a log.

    Raises:
        RuntimeError: If the baseline assets yield no encoder attestation, or, from the asset
            preflight this calls, if the verification receipt is missing or persisted to an
            unexpected path. Running on CPU is the point — every refusal here happens before
            any GPU container is billed.

    """
    started_monotonic = time.monotonic()
    contract = render_segment_contract(profile)
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForTokenClassification

    from meddies_pii.training.bioes.modal.full_run import (
        _asset_receipt_contract,
        _persist_preflight_receipt,
        _preflight_candidate_assets,
        _receipt_path,
        _validated_receipt_inventory,
    )

    baseline_contract = contract["trajectory"]["baseline_full_run_contract"]
    receipt_path = _receipt_path(_asset_receipt_contract(baseline_contract))
    shard_paths, encoder_attestation, built_assets = _ensure_pii350_baseline_assets(
        baseline_contract,
        receipt_path=receipt_path,
        preflight_candidate_assets=_preflight_candidate_assets,
        persist_preflight_receipt=_persist_preflight_receipt,
        validated_receipt_inventory=_validated_receipt_inventory,
        preflight_inputs={
            "snapshot_download": snapshot_download,
            "load_dataset": load_dataset,
            "auto_model_for_token_classification": AutoModelForTokenClassification,
        },
    )
    if built_assets:
        artifacts.commit()
    if not encoder_attestation:
        msg = "continuation CPU receipt requires a PII350 encoder attestation"
        raise RuntimeError(msg)
    packed_paths_sha256 = sha256("\n".join(shard_paths).encode()).hexdigest()
    return _with_cpu_cost(
        {
            "profile": profile,
            "execution_contract_digest": contract["execution_contract_digest"],
            "eval_rows": EVAL_ROWS,
            "offline_training_assets": True,
            "packed_shards": {
                "count": len(shard_paths),
                "paths": list(shard_paths),
                "paths_sha256": packed_paths_sha256,
            },
            "encoder_checkpoint_attestation": encoder_attestation,
            "preflight_digest": preflight_digest(
                contract["execution_contract_digest"],
                packed_paths_sha256=packed_paths_sha256,
                shard_count=len(shard_paths),
                encoder_checkpoint_attestation=encoder_attestation,
            ),
        },
        started_monotonic,
    )


@app.function(image=image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)
def cpu_receipt(profile: str, source_root: str = str(SOURCE_CHECKPOINT_ROOT)) -> dict[str, Any]:
    """Hash one explicit parent root; stage two calls this only after transfer.

    Returns:
        The hash-verified source receipt: the profile and its execution contract digest, the
        transport repository, the full source checkpoint inventory, the ``launch_digest``
        binding contract to checkpoint, the requested and resolved roots, and the
        ``source_root_binding`` over all three paths. The paid launch recomputes both digests,
        so a receipt naming a different root or different bytes cannot authorize a run.

    Raises:
        RuntimeError: If the resolved source root or artifact root is not a Modal volume path,
            or if the resolved root is not the requested root mapped through the artifact
            mount. Both refuse a receipt whose path relationship was not proven rather than
            asserted, and the inventory this calls raises on any invalid checkpoint lineage.

    """
    started_monotonic = time.monotonic()
    contract = render_segment_contract(profile)
    requested_root = _requested_artifact_root(source_root)
    source = inventory_source_checkpoint(Path(requested_root))
    resolved_root = source.get("source_root")
    resolved_artifact_root = _resolved_artifact_root()
    if not _is_modal_volume_root(resolved_root) or not _is_modal_volume_root(resolved_artifact_root):
        msg = "CPU receipt resolved root is not a Modal volume path"
        raise RuntimeError(msg)
    relative_root = PurePosixPath(requested_root).relative_to(PurePosixPath(ARTIFACT_MOUNT))
    if str(PurePosixPath(resolved_artifact_root) / relative_root) != resolved_root:
        msg = "CPU receipt resolved root is not mapped from its requested root"
        raise RuntimeError(msg)
    launch = launch_digest(contract["execution_contract_digest"], source)
    return _with_cpu_cost(
        {
            "profile": profile,
            "execution_contract_digest": contract["execution_contract_digest"],
            "transport_repository": CHECKPOINT_REPOSITORY,
            "source_checkpoint_receipt": source,
            "launch_digest": launch,
            "requested_source_root": requested_root,
            "resolved_artifact_root": resolved_artifact_root,
            "source_root_binding": _source_root_binding(launch, requested_root, resolved_artifact_root, resolved_root),
        },
        started_monotonic,
    )


@app.function(image=transport_image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)
def upload_checkpoint(
    profile: str,
    checkpoint_root: str,
    remote_prefix: str,
    source_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU-only transport.  The manifest is uploaded last as the commit marker.

    Returns:
        The transport receipt: the repository and remote prefix, the uploaded manifest with
        its commit SHA, that SHA again as the immutable revision a downloader must pin, and
        whether the manifest was derived in memory for the one legacy source. Plus CPU cost.

    Raises:
        RuntimeError: If the checkpoint root is outside the artifact volume, or if the prior
            CPU source receipt is absent. The manifest validation and the upload this calls
            raise on a receipt/bytes mismatch, a mid-upload byte change, or a missing commit
            SHA. Refusing an off-volume root matters because the receipt binds a volume path.

    """
    started_monotonic = time.monotonic()
    render_segment_contract(profile)
    from huggingface_hub import HfApi

    root = Path(checkpoint_root)
    if not root.is_relative_to(ARTIFACT_MOUNT):
        msg = "continuation transport only uploads checkpoints from the artifact volume"
        raise RuntimeError(msg)
    if not isinstance(source_receipt, Mapping):
        msg = "continuation transport requires the prior CPU source receipt"
        raise RuntimeError(msg)
    manifest, legacy_source = _validated_manifest_for_upload(root, profile=profile, source_receipt=source_receipt)
    uploaded = upload_manifest_last(
        HfApi(),
        root=root,
        repo_id=CHECKPOINT_REPOSITORY,
        remote_prefix=remote_prefix,
        manifest=manifest,
    )
    return _with_cpu_cost(
        {
            "repository": CHECKPOINT_REPOSITORY,
            "remote_prefix": remote_prefix,
            "manifest": uploaded,
            "manifest_commit_sha": uploaded["manifest_commit_sha"],
            "legacy_source_manifest_derived_in_memory": legacy_source,
        },
        started_monotonic,
    )


@app.function(image=transport_image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)
def download_checkpoint(
    profile: str,
    remote_prefix: str,
    destination: str,
    *,
    revision: str,
    purpose: str,
) -> dict[str, Any]:
    """Download into a temporary volume directory, verify, then publish atomically.

    Returns:
        The verified download receipt: the purpose-scoped verification result, ``hash_verified``,
        the published root, the checkpoint's step, cursor and world size read from its own
        metadata, the pinned immutable revision, and CPU cost. Publishing is a rename out of
        the temporary directory AFTER verification, so the destination path never exists in a
        half-downloaded or unverified state.

    Raises:
        RuntimeError: If the destination is outside the artifact volume, already exists, or no
            revision was supplied; if the downloaded manifest is not an object or holds an
            invalid file entry; or, from the verification this calls, if the bytes do not
            match the manifest or the files required for the load purpose are absent.
            Requiring an explicit revision is what makes the download reproducible: a moving
            tag could serve different bytes to two callers that both look verified. The
            temporary directory is removed on every failure path.

    """
    started_monotonic = time.monotonic()
    render_segment_contract(profile)
    import shutil

    from huggingface_hub import hf_hub_download

    root = Path(destination)
    if not root.is_relative_to(ARTIFACT_MOUNT):
        msg = "continuation transport only downloads checkpoints into the artifact volume"
        raise RuntimeError(msg)
    if root.exists():
        msg = "continuation download destination already exists"
        raise RuntimeError(msg)
    if not revision:
        msg = "continuation download requires the uploader's immutable manifest commit SHA"
        raise RuntimeError(msg)
    temporary = root.with_name(root.name + f".tmp-{time.time_ns()}")
    # reason: download's try keeps verify with read text; splitting would split cleanup from writes.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        manifest_path = Path(
            hf_hub_download(
                repo_id=CHECKPOINT_REPOSITORY,
                filename=f"{remote_prefix}/manifest.json",
                repo_type="model",
                revision=revision,
                local_dir=str(temporary),
            ),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            msg = "downloaded checkpoint manifest is not an object"
            # reason: invalid remote evidence must raise inside the boundary that deletes the temporary download tree.
            raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
        for item in manifest.get("files", []):
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                msg = "downloaded checkpoint manifest contains an invalid file entry"
                # reason: invalid remote evidence must raise inside the boundary that deletes the temporary download tree.
                raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
            hf_hub_download(
                repo_id=CHECKPOINT_REPOSITORY,
                filename=f"{remote_prefix}/{item['path']}",
                repo_type="model",
                revision=revision,
                local_dir=str(temporary),
            )
        downloaded_root = temporary / remote_prefix
        receipt = verify_downloaded_checkpoint(downloaded_root, manifest, purpose=purpose)
        metadata = json.loads((downloaded_root / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            msg = "downloaded checkpoint metadata is not an object"
            # reason: invalid remote evidence must raise inside the boundary that deletes the temporary download tree.
            raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
        downloaded_root.replace(root)
        shutil.rmtree(temporary, ignore_errors=True)
        artifacts.commit()
        return _with_cpu_cost(
            {
                **receipt,
                "hash_verified": True,
                "source_root": str(root),
                "optimizer_step": metadata["optimizer_step"],
                "packed_cursor": metadata["packed_cursor"],
                "world_size": metadata["world_size"],
                "immutable_hf_revision": revision,
            },
            started_monotonic,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


# reason: validated and run torchrun share train segment's state; extraction would fragment diagnostics.
def _train_segment_impl(  # ruff: ignore[complex-structure,too-many-branches,too-many-arguments,too-many-statements,too-many-positional-arguments]
    profile: str,
    execution_contract_digest: str,
    source_receipt: Mapping[str, Any] | None = None,
    source_launch_digest: str = "",
    preflight_receipt: Mapping[str, Any] | None = None,
    early_stop_receipt: Mapping[str, Any] | None = None,
    interruption_cost_receipt: Mapping[str, Any] | None = None,
    launch_attempt_nonce: str = "",
    slice_timeout_seconds: int = 0,
    *,
    execute: bool = False,
    confirmation: str = "",
    primary_action: str = "",
) -> dict[str, Any]:
    """Re-read source bytes inside the paid container.

    A received receipt is an identity claim; it cannot substitute for byte verification at load time.

    Returns:
        The torchrun outcome for this paid segment, from the isolated child run. Everything
        before that dispatch is refusal: by the time this returns, the launch was authorized,
        the source bytes were re-inventoried, the budget was admitted and the attempt was
        claimed.

    Raises:
        RuntimeError: This is a SPEND-AUTHORIZATION GATE, and the first check is the one that
            stands between a typo and a billed multi-GPU run. It refuses unless ``execute`` is
            true AND ``confirmation`` and ``primary_action`` are byte-equal to the values on
            the freshly rendered profile contract, so an authorization approved for one
            profile topology cannot launch another. It then refuses if the execution contract
            digest is not the rendered one; if the source receipt is absent or not
            hash-verified; if the launch digest does not bind this exact source checkpoint; if
            the preflight receipt is missing, incomplete, carries the wrong contract, or has
            an invalid preflight digest; if the receipt's source root is missing or no longer
            matches the freshly inventoried checkpoint; if the wave's cumulative budget is
            already exhausted before dispatch; or if the assets no longer match the CPU
            preflight receipt. Re-inventorying the source rather than trusting the receipt is
            the load-bearing choice: the receipt was produced by an earlier container, and a
            claim about bytes is not the bytes. Early-stop validation, slice admission and the
            attempt claim raise on their own terms, the last refusing a duplicate launch of
            one attempt digest.

    """
    started_monotonic = time.monotonic()
    contract = render_segment_contract(profile)
    manual_launch = contract["manual_launch"]
    if not execute or confirmation != manual_launch["confirmation"] or primary_action != manual_launch["primary_action"]:
        msg = "continuation training requires the exact profile topology authorization"
        raise RuntimeError(msg)
    if execution_contract_digest != contract["execution_contract_digest"]:
        msg = "continuation training requires the exact execution contract digest"
        raise RuntimeError(msg)
    if not isinstance(source_receipt, Mapping) or source_receipt.get("hash_verified") is not True:
        msg = "continuation training requires the CPU hash-verified source checkpoint receipt"
        raise RuntimeError(msg)
    if source_launch_digest != launch_digest(contract["execution_contract_digest"], source_receipt):
        msg = "continuation training launch digest does not bind this exact source checkpoint"
        raise RuntimeError(msg)
    attempt_digest = launch_attempt_digest(
        contract["execution_contract_digest"],
        source_launch_digest,
        launch_attempt_nonce,
    )
    if not isinstance(preflight_receipt, Mapping):
        msg = "continuation training requires a CPU cache/data/model preflight receipt"
        raise RuntimeError(msg)
    packed = preflight_receipt.get("packed_shards")
    attestation = preflight_receipt.get("encoder_checkpoint_attestation")
    if not isinstance(packed, Mapping) or not isinstance(attestation, Mapping):
        msg = "continuation training preflight receipt is incomplete"
        raise RuntimeError(msg)
    if preflight_receipt.get("execution_contract_digest") != contract["execution_contract_digest"]:
        msg = "continuation training preflight has the wrong execution contract"
        raise RuntimeError(msg)
    if preflight_receipt.get("preflight_digest") != preflight_digest(
        contract["execution_contract_digest"],
        packed_paths_sha256=str(packed.get("paths_sha256", "")),
        shard_count=int(packed.get("count", 0)),
        encoder_checkpoint_attestation=attestation,
    ):
        msg = "continuation training preflight digest is invalid"
        raise RuntimeError(msg)
    source_root = source_receipt.get("source_root")
    if not isinstance(source_root, str):
        msg = "continuation source receipt lacks its source root"
        raise RuntimeError(msg)
    observed_source = inventory_source_checkpoint(Path(source_root))
    for key in ("checkpoint_digest", "optimizer_step", "packed_cursor", "world_size"):
        if source_receipt.get(key) != observed_source.get(key):
            msg = "continuation source receipt no longer matches the verified checkpoint"
            raise RuntimeError(msg)
    validate_early_stop_receipt(
        contract,
        source_step=int(observed_source["optimizer_step"]),
        receipt=early_stop_receipt,
    )
    carried_cost_usd = effective_wave_carried_cost(
        profile,
        observed_source,
        interruption_cost_receipt=interruption_cost_receipt,
    )
    if carried_cost_usd >= float(contract["execution"]["all_in_ceiling_usd"]):
        msg = "continuation wave cumulative budget is exhausted before GPU dispatch"
        raise RuntimeError(msg)
    require_slice_admission(
        profile,
        carried_cost_usd=carried_cost_usd,
        slice_timeout_seconds=slice_timeout_seconds,
    )
    from meddies_pii.training.bioes.modal.full_run import _validated_receipt_inventory

    baseline = contract["trajectory"]["baseline_full_run_contract"]
    shard_paths, checkpoint_attestation = _validated_receipt_inventory(baseline)
    if not checkpoint_attestation:
        msg = "continuation requires the CPU-verified PII350 encoder attestation"
        raise RuntimeError(msg)
    fresh_paths_digest = sha256("\n".join(shard_paths).encode()).hexdigest()
    if (
        fresh_paths_digest != packed.get("paths_sha256")
        or len(shard_paths) != packed.get("count")
        or dict(checkpoint_attestation) != dict(attestation)
    ):
        msg = "continuation paid launch assets no longer match its CPU preflight receipt"
        raise RuntimeError(msg)
    root = _claim_launch_attempt(
        ARTIFACT_ROOT,
        attempt_digest=attempt_digest,
        launch_attempt_nonce=launch_attempt_nonce,
        profile=profile,
        source_launch_digest=source_launch_digest,
    )
    return _run_isolated_torchrun(
        ChildSpec(
            contract=contract,
            source_receipt=observed_source,
            source_root=str(Path(source_root)),
            carried_cost_usd=carried_cost_usd,
            shard_paths=tuple(shard_paths),
            encoder_checkpoint_attestation=dict(checkpoint_attestation or {}),
            artifact_dir=str(root),
            started_monotonic=started_monotonic,
            slice_timeout_seconds=slice_timeout_seconds,
            world_size=int(contract["execution"]["world_size"]),
            all_in_rate_usd_per_second=float(contract["execution"]["all_in_rate_usd_per_second"]),
        ),
    )


def _train_segment_for_timeout(
    slice_timeout_seconds: int,
    *args: object,
    **kwargs: object,
) -> dict[str, Any]:
    payload = {**kwargs, "slice_timeout_seconds": slice_timeout_seconds}
    train_segment = cast("Callable[..., dict[str, Any]]", _train_segment_impl)
    return train_segment(*args, **payload)


@app.function(
    image=image,
    volumes=MOUNTS,
    secrets=[secret],
    gpu="A100-40GB:2",
    cpu=4.0,
    memory=64 * 1024,
    timeout=7_900,
    max_containers=1,
)
def train_segment_7900_world2(*args: object, **kwargs: object) -> dict[str, Any]:
    return _train_segment_for_timeout(7_900, *args, **kwargs)


@app.function(
    image=image,
    volumes=MOUNTS,
    secrets=[secret],
    gpu="A100-40GB:4",
    cpu=8.0,
    memory=128 * 1024,
    timeout=10_800,
    max_containers=1,
)
def train_segment_10800(*args: object, **kwargs: object) -> dict[str, Any]:
    return _train_segment_for_timeout(10_800, *args, **kwargs)


@app.function(
    image=image,
    volumes=MOUNTS,
    secrets=[secret],
    gpu="A100-40GB:4",
    cpu=8.0,
    memory=128 * 1024,
    timeout=7_000,
    max_containers=1,
)
def train_segment_7000(*args: object, **kwargs: object) -> dict[str, Any]:
    return _train_segment_for_timeout(7_000, *args, **kwargs)


@app.function(
    image=image,
    volumes=MOUNTS,
    secrets=[secret],
    gpu="A100-40GB:4",
    cpu=8.0,
    memory=128 * 1024,
    timeout=6_000,
    max_containers=1,
)
def train_segment_6000(*args: object, **kwargs: object) -> dict[str, Any]:
    return _train_segment_for_timeout(6_000, *args, **kwargs)


@app.function(
    image=image,
    volumes=MOUNTS,
    secrets=[secret],
    gpu="A100-40GB:4",
    cpu=8.0,
    memory=128 * 1024,
    timeout=4_000,
    max_containers=1,
)
def train_segment_4000(*args: object, **kwargs: object) -> dict[str, Any]:
    return _train_segment_for_timeout(4_000, *args, **kwargs)


# reason: pii350 ddp coordinates render with stop receipt; extra seams would fragment diagnostics.
@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(  # ruff: ignore[complex-structure,too-many-branches,too-many-arguments,too-many-statements,too-many-positional-arguments]
    profile: str = "",
    preflight: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    cpu_receipt: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    upload: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    download: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    train: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    execute: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    confirmation: str = "",
    primary_action: str = "",
    execution_contract_digest: str = "",
    source_root: str = "",
    source_receipt_json: str = "",
    source_launch_digest: str = "",
    preflight_receipt_json: str = "",
    early_stop_receipt_json: str = "",
    interruption_cost_receipt_json: str = "",
    launch_attempt_nonce: str = "",
    checkpoint_root: str = "",
    remote_prefix: str = "",
    destination: str = "",
    revision: str = "",
    purpose: str = "",
) -> None:
    if sum((preflight, cpu_receipt, upload, download, train)) != 1 or not profile:
        msg = "choose exactly one preflight, CPU source receipt, upload, download, or train action with a profile"
        raise RuntimeError(
            msg,
        )
    if os.environ.get("MODAL_PROFILE") != profile:
        msg = "continuation local dispatch requires MODAL_PROFILE to equal the requested profile"
        raise RuntimeError(msg)
    if preflight:
        print(json.dumps(globals()["cpu_preflight"].remote(profile), sort_keys=True))
        return
    if cpu_receipt:
        print(
            json.dumps(
                globals()["cpu_receipt"].remote(profile, source_root or str(SOURCE_CHECKPOINT_ROOT)),
                sort_keys=True,
            ),
        )
        return
    if upload:
        if not checkpoint_root or not remote_prefix:
            msg = "upload requires checkpoint_root and remote_prefix"
            raise RuntimeError(msg)
        if not source_receipt_json:
            msg = "upload requires the prior CPU source receipt JSON"
            raise RuntimeError(msg)
        source_receipt = _source_receipt_from_cpu_receipt(
            source_receipt_json,
            profile=profile,
            checkpoint_root=checkpoint_root,
        )
        print(
            json.dumps(
                upload_checkpoint.remote(
                    profile,
                    checkpoint_root,
                    remote_prefix,
                    source_receipt,
                ),
                sort_keys=True,
            ),
        )
        return
    if download:
        if not remote_prefix or not destination or not revision or purpose not in {"training", "evaluation"}:
            msg = "download requires remote_prefix, destination, immutable revision, and training/evaluation purpose"
            raise RuntimeError(
                msg,
            )
        print(
            json.dumps(
                download_checkpoint.remote(
                    profile,
                    remote_prefix,
                    destination,
                    revision=revision,
                    purpose=purpose,
                ),
                sort_keys=True,
            ),
        )
        return
    if not source_receipt_json or not source_launch_digest or not preflight_receipt_json:
        msg = "train requires explicit CPU source and preflight receipts plus its launch digest"
        raise RuntimeError(msg)
    try:
        source_receipt = json.loads(source_receipt_json)
    except json.JSONDecodeError as error:
        msg = "train source receipt JSON is invalid"
        raise RuntimeError(msg) from error
    if not isinstance(source_receipt, Mapping):
        msg = "train source receipt JSON is not an object"
        raise RuntimeError(msg)
    try:
        preflight_receipt = json.loads(preflight_receipt_json)
    except json.JSONDecodeError as error:
        msg = "train preflight receipt JSON is invalid"
        raise RuntimeError(msg) from error
    if not isinstance(preflight_receipt, Mapping):
        msg = "train preflight receipt JSON is not an object"
        raise RuntimeError(msg)
    try:
        early_stop_receipt = json.loads(early_stop_receipt_json) if early_stop_receipt_json else None
    except json.JSONDecodeError as error:
        msg = "train early-stop receipt JSON is invalid"
        raise RuntimeError(msg) from error
    try:
        interruption_cost_receipt = json.loads(interruption_cost_receipt_json) if interruption_cost_receipt_json else None
    except json.JSONDecodeError as error:
        msg = "train interruption cost receipt JSON is invalid"
        raise RuntimeError(msg) from error
    if interruption_cost_receipt is not None and not isinstance(interruption_cost_receipt, Mapping):
        msg = "train interruption cost receipt JSON is not an object"
        raise RuntimeError(msg)
    contract = render_segment_contract(profile)
    validate_early_stop_receipt(
        contract,
        source_step=int(source_receipt["optimizer_step"]),
        receipt=early_stop_receipt,
    )
    launch_attempt_digest(
        contract["execution_contract_digest"],
        source_launch_digest,
        launch_attempt_nonce,
    )
    carried = effective_wave_carried_cost(
        profile,
        source_receipt,
        interruption_cost_receipt=interruption_cost_receipt,
    )
    timeout = select_slice_timeout_seconds(profile, carried_cost_usd=carried)
    require_slice_admission(profile, carried_cost_usd=carried, slice_timeout_seconds=timeout)
    endpoint = {
        (2, 7_900): train_segment_7900_world2,
        (4, 10_800): train_segment_10800,
        (4, 7_000): train_segment_7000,
        (4, 6_000): train_segment_6000,
        (4, 4_000): train_segment_4000,
    }[int(contract["execution"]["world_size"]), timeout]
    print(
        json.dumps(
            endpoint.remote(
                profile,
                execution_contract_digest,
                source_receipt,
                source_launch_digest,
                preflight_receipt,
                early_stop_receipt,
                interruption_cost_receipt,
                launch_attempt_nonce,
                execute=execute,
                confirmation=confirmation,
                primary_action=primary_action,
            ),
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec")
    parser.add_argument("--child-result")
    arguments = parser.parse_args()
    if not arguments.child_spec or not arguments.child_result:
        msg = "continuation worker requires --child-spec and --child-result"
        raise RuntimeError(msg)
    raise SystemExit(_child_main(arguments.child_spec, arguments.child_result))
