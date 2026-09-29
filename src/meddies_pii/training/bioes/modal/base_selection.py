"""Isolated H100 batch-size probes for the LFM2.5 BIOES base-selection gate.

Local configuration rendering deliberately lives in ``trainers.base_selection``:
``python -m meddies_pii.training.bioes.trainers.base_selection``.  Importing
this file constructs a Modal application, so it is never the render command.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack (torch, transformers, unsloth, datasets) is installed in the Modal image, so each
# reason: function body imports it inside the container.
# ruff: file-ignore[print]
# reason: this module runs inside a Modal container; its standard output is the operator's streamed run log.
# ruff: file-ignore[invalid-function-name]
# reason: the NVML protocol mirrors pynvml's upstream `nvml*` function names exactly for structural typing.
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
import json
import os

# reason: the Modal parent launches this same pinned module through sys.executable and list-form argv.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast
from uuid import uuid4

import modal

if TYPE_CHECKING:
    from torch import nn

    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger

from meddies_pii.modal_runtime import MODAL_PYTHONPATH
from meddies_pii.training.bioes.modal.probe_scaffold import (
    artifact_volume,
    child_argv_paths,
    hf_cache_environment,
    hf_cache_volume,
    hf_secret,
    source_mounted_image,
    volume_mounts,
    write_child_spec,
)
from meddies_pii.training.bioes.modal.probe_scaffold import (
    require_modal_profile as require_profile,
)
from meddies_pii.training.bioes.trainers.base_selection import (
    CANDIDATES,
    GATE_SEED,
    H100_GPU,
    MODAL_H100_USD_PER_SECOND,
    PROBE_BUDGET_USD,
    LoraEvidence,
    MemoryContract,
    adamw_kwargs,
    apply_gate_lora,
    enable_native_gradient_checkpointing,
    load_encoder_body,
    load_pii350_body,
    load_standard_body,
    probe_batch_plan,
    render_gate_config,
    require_estimated_cost_within_budget,
    require_execute_confirmation,
)
from meddies_pii.training.bioes.trainers.packing import (
    PackedRowRange,
    PackedTrainingUnit,
)

DISCOVERY_PROBE_STEPS = 2
FULL_PROBE_STEPS = 10
MIN_PACKED_SEGMENTS = 2
TIMING_WARMUP_STEPS = 3

DATASET_ID = "Meddies/meddies-pii-mixed"
PACKED_CONFIG = "packed"
CACHE_ROOT = "/cache/hf"
PACKED_DATASET_REVISION = "11fd43ec9ebb187e1d0f94fe77bcf1090a2a18ee"
PACKED_MANIFEST_SHA256 = "7cadda8e81ef4b2a1111f37a8b508492158a983e90f54fecafdc24764f792969"
PACKED_SHARD_COUNT = 469
PACKED_CURSOR = 1_000_000
ARTIFACT_ROOT = "/artifacts/base-selection"
ARTIFACT_VOLUME_NAME = "meddies-pii-bioes-artifacts"
ARTIFACT_VOLUME_MOUNT = "/artifacts"
H100_TIMEOUT_SECONDS = 55 * 60
SHUTDOWN_RESERVE_SECONDS = 120.0
EXPECTED_CHILD_SECONDS = 180.0
EXPECTED_DISCOVERY_CHILD_SECONDS = 60.0
CANDIDATE_PROBE_BATCH_SIZES = {
    "base230": (224, 240, 256),
    "encoder230": (224, 240, 256),
    "encoder350": (200,),
}
CANDIDATE_PROBE_STEP_COUNTS = {
    "base230": (2, 10),
    "encoder230": (2, 10),
    "encoder350": (10,),
}
WARNING_FRACTIONS = (0.50, 0.75, 0.90)
H100_CACHE_ENVIRONMENT = hf_cache_environment(
    root=CACHE_ROOT,
    hub_cache=f"{CACHE_ROOT}/hub",
    disable_unsloth_statistics=False,
)


class _LaneOptions(TypedDict):
    gpu: str
    cpu: float
    memory: int
    timeout: int
    max_containers: int
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    secrets: list[modal.Secret]


image = source_mounted_image(
    modal.Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install(
        "torch==2.10.0",
        "transformers==5.2.0",
        "peft==0.19.1",
        "pyarrow==23.0.0",
        "huggingface_hub==1.11.0",
        "nvidia-ml-py==13.590.44",
    )
    .env(H100_CACHE_ENVIRONMENT),
)
cache = hf_cache_volume()
artifacts = artifact_volume(ARTIFACT_VOLUME_NAME)
secret = hf_secret()
app = modal.App("meddies-lfm25-four-base-selection", image=image)
LANE_OPTIONS: _LaneOptions = {
    "gpu": H100_GPU,
    "cpu": 4.0,
    "memory": 64 * 1024,
    "timeout": H100_TIMEOUT_SECONDS,
    "max_containers": 1,
    "volumes": volume_mounts(cache, artifacts),
    "secrets": [secret],
}


@dataclass(frozen=True, slots=True)
class VerifiedPackedArtifact:
    revision: str
    manifest: Mapping[str, Any]
    manifest_path: str
    shard_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChildSpec:
    lane: str
    candidate_key: str
    batch_size: int
    fused_adamw: bool
    dataset_revision: str
    manifest_path: str
    shard_paths: tuple[str, ...]
    artifact_dir: str
    deadline_monotonic: float
    role: str
    step_count: int = 10


class _NvmlUtilization(Protocol):
    gpu: int
    memory: int


class _NvmlMemoryInfo(Protocol):
    used: int
    total: int


class _NvmlModule(Protocol):
    def nvmlInit(self) -> None: ...

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> _NvmlUtilization: ...

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _NvmlMemoryInfo: ...

    def nvmlDeviceGetPowerUsage(self, handle: object) -> int: ...


class _BudgetWarning(TypedDict):
    event: str
    fraction: float
    elapsed_seconds: float


class ArtifactWriter:
    """Append-only evidence writer.  Every append is committed by the caller."""

    def __init__(
        self,
        root: str | Path,
        lane: str,
        commit: Callable[[], None] | None,
        *,
        execution_id: str | None = None,
    ) -> None:
        self.execution_id = execution_id or uuid4().hex
        self.root = Path(root) / lane / self.execution_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.events_path = self.root / "events.jsonl"
        self.commit = commit

    def append(self, event: Mapping[str, Any]) -> None:
        encoded = json.dumps({**event, "execution_id": self.execution_id}, sort_keys=True)
        print(encoded, flush=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.commit is not None:
            self.commit()

    def finalize(self, result: Mapping[str, Any]) -> None:
        temporary = self.root / "result.json.tmp"
        temporary.write_text(json.dumps({**result, "execution_id": self.execution_id}, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.root / "result.json")
        if self.commit is not None:
            self.commit()


def render_dry_run() -> str:
    """Compatibility wrapper.  Prefer the pure trainers module render command.

    Returns:
        The gate config as sorted, indented JSON with a trailing newline -- byte-identical
        to what the trainers module renders, which is what makes this wrapper safe to drop.

    """
    return json.dumps(render_gate_config(), indent=2, sort_keys=True) + "\n"


def require_modal_profile() -> None:
    require_profile("hahuyhoang411", "H100 selection gate")


def _probe_lane_guard(lane: str, *, execute: bool, confirmation: str, estimated_cost_usd: float) -> dict[str, object]:
    if lane not in {"230", "350"}:
        msg = "lane must be 230 or 350"
        raise ValueError(msg)
    require_execute_confirmation(execute=execute, confirmation=confirmation)
    require_estimated_cost_within_budget(estimated_cost_usd, PROBE_BUDGET_USD)
    lanes = render_gate_config()["probe_lanes"]
    if not isinstance(lanes, dict) or not isinstance(lanes.get(lane), dict):
        msg = "rendered probe lane is invalid"
        raise RuntimeError(msg)
    return dict(lanes[lane])


def require_candidate_probe(lane: str, candidate_key: str, batch_size: int, step_count: int) -> None:
    candidate = CANDIDATES.get(candidate_key)
    if candidate is None or candidate.lane != lane:
        msg = "candidate does not belong to the requested lane"
        raise ValueError(msg)
    allowed_batch_sizes = CANDIDATE_PROBE_BATCH_SIZES.get(candidate_key)
    allowed_step_counts = CANDIDATE_PROBE_STEP_COUNTS.get(candidate_key)
    if allowed_batch_sizes is None or allowed_step_counts is None:
        msg = "candidate probe supports only explicit candidates"
        raise ValueError(msg)
    if batch_size not in allowed_batch_sizes:
        msg = "candidate probe batch_size is outside the allowlist"
        raise ValueError(msg)
    if step_count not in allowed_step_counts:
        msg = "candidate probe step_count is outside the allowlist"
        raise ValueError(msg)


def require_probe_mode(lane: str, mode: str, candidate_key: str, batch_size: int, step_count: int) -> None:
    if mode == "full_lane":
        if candidate_key or batch_size or step_count:
            msg = "full_lane mode does not accept candidate probe parameters"
            raise ValueError(msg)
        return
    if mode == "candidate_probe":
        require_candidate_probe(lane, candidate_key, batch_size, step_count)
        return
    msg = "mode must be full_lane or candidate_probe"
    raise ValueError(msg)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# reason: verify packed coordinates read text with sha file; extra seams would fragment diagnostics.
def verify_packed_artifact(  # ruff: ignore[complex-structure]
    *,
    revision: str,
    cache_dir: str = CACHE_ROOT,
    download: Callable[..., str] | None = None,
    expected_manifest_sha256: str = PACKED_MANIFEST_SHA256,
    expected_shard_count: int = PACKED_SHARD_COUNT,
) -> VerifiedPackedArtifact:
    """Pins and verifies every Parquet blob before a model is constructed.

    Returns:
        The verified artifact: the revision, the manifest, the manifest path, and every
        shard path in manifest order. A caller holding one of these has the bytes checked,
        not merely located.

    Raises:
        ValueError: If the revision is not the published packed commit. The pin is exact;
            there is no newer-is-fine case.
        RuntimeError: If the manifest names a different packed config, is not marked
            complete, carries a different resume cursor, or does not list exactly the
            expected shard count; if the manifest's own SHA-256 differs from the launch pin;
            if a shard entry is not a mapping, or its path is not a string under
            ``packed/data/`` -- a path outside that prefix is refused rather than
            downloaded; if an entry lacks its size or digest; if a downloaded shard's size
            or SHA-256 disagrees with the entry; or if the manifest yields no shards at all.

    """
    if revision != PACKED_DATASET_REVISION:
        msg = "MEDDIES_PACKED_REVISION must match the published packed commit"
        raise ValueError(msg)
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    manifest_path = Path(
        download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            filename=f"{PACKED_CONFIG}/manifest.json",
            cache_dir=cache_dir,
            local_files_only=True,
        ),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("packed_config") != PACKED_CONFIG
        or manifest.get("complete") is not True
        or manifest.get("resume_cursor") != PACKED_CURSOR
        or not isinstance(manifest.get("shards"), list)
        or len(manifest["shards"]) != expected_shard_count
    ):
        msg = "packed config is absent or its manifest is incomplete"
        raise RuntimeError(msg)
    if _sha_file(manifest_path) != expected_manifest_sha256:
        msg = "packed manifest SHA-256 does not match the launch pin"
        raise RuntimeError(msg)
    paths: list[str] = []
    for shard in manifest["shards"]:
        if not isinstance(shard, Mapping):
            msg = "packed manifest shard is invalid"
            raise RuntimeError(msg)
        name, byte_count, digest = (
            shard.get("path"),
            shard.get("bytes"),
            shard.get("sha256"),
        )
        if not isinstance(name, str) or not name.startswith("packed/data/"):
            msg = "packed manifest contains an unsafe shard path"
            raise RuntimeError(msg)
        if not isinstance(byte_count, int) or not isinstance(digest, str):
            msg = "packed manifest shard lacks size or sha256"
            raise RuntimeError(msg)
        path = Path(
            download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                revision=revision,
                filename=name,
                cache_dir=cache_dir,
                local_files_only=True,
            ),
        )
        if path.stat().st_size != byte_count or _sha_file(path) != digest:
            msg = f"packed shard verification failed for {name}"
            raise RuntimeError(msg)
        paths.append(str(path))
    if not paths:
        msg = "packed manifest has no training shards"
        raise RuntimeError(msg)
    return VerifiedPackedArtifact(revision, manifest, str(manifest_path), tuple(paths))


def _budget_seconds() -> float:
    return PROBE_BUDGET_USD / MODAL_H100_USD_PER_SECOND


def _cost_progress(*, started: float, now: float) -> dict[str, float]:
    elapsed = max(0.0, now - started)
    return {
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": elapsed * MODAL_H100_USD_PER_SECOND,
        "budget_usd": PROBE_BUDGET_USD,
        "budget_seconds": _budget_seconds(),
    }


def _can_start(*, now: float, started: float, expected_seconds: float, budget_seconds: float) -> bool:
    return now - started + expected_seconds + SHUTDOWN_RESERVE_SECONDS <= budget_seconds


def _warning_events(elapsed_seconds: float, budget_seconds: float) -> list[_BudgetWarning]:
    fraction = elapsed_seconds / budget_seconds
    return [
        {
            "event": "budget_warning",
            "fraction": threshold,
            "elapsed_seconds": elapsed_seconds,
        }
        for threshold in WARNING_FRACTIONS
        if fraction >= threshold
    ]


# reason: work spec exposes lane/step count as its Modal schema; bundling would break callers.
def _work_spec(  # ruff: ignore[too-many-arguments]
    *,
    lane: str,
    candidate_key: str,
    batch_size: int,
    fused_adamw: bool,
    artifact: VerifiedPackedArtifact,
    artifact_dir: str,
    deadline_monotonic: float,
    role: str,
    step_count: int,
) -> ChildSpec:
    return ChildSpec(
        lane=lane,
        candidate_key=candidate_key,
        batch_size=batch_size,
        fused_adamw=fused_adamw,
        dataset_revision=artifact.revision,
        manifest_path=artifact.manifest_path,
        shard_paths=artifact.shard_paths,
        artifact_dir=artifact_dir,
        deadline_monotonic=deadline_monotonic,
        role=role,
        step_count=step_count,
    )


def _progress_child_spec(spec: ChildSpec) -> dict[str, object]:
    shard_paths = spec.shard_paths
    return {
        "lane": spec.lane,
        "candidate_key": spec.candidate_key,
        "batch_size": spec.batch_size,
        "fused_adamw": spec.fused_adamw,
        "dataset_revision": spec.dataset_revision,
        "manifest_path": spec.manifest_path,
        "artifact_dir": spec.artifact_dir,
        "deadline_monotonic": spec.deadline_monotonic,
        "role": spec.role,
        "step_count": spec.step_count,
        "shard_count": len(shard_paths),
        "shard_paths_sha256": sha256("\n".join(shard_paths).encode()).hexdigest(),
        "shard_path_sample": {"first": shard_paths[0], "last": shard_paths[-1]},
    }


# reason: run lane plan owns run and gate config together; splitting would fragment diagnostics.
def run_lane_plan(  # ruff: ignore[complex-structure]
    *,
    lane: str,
    artifact: VerifiedPackedArtifact,
    child_runner: Callable[[ChildSpec], Mapping[str, Any]],
    writer: ArtifactWriter,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Schedule children sequentially. The child is the CUDA/process boundary.

    Returns:
        The lane record: lane, execution id, dataset revision, budget, elapsed seconds,
        estimated cost, and one child result per probe run in the order they ran.

    Raises:
        RuntimeError: If the remaining budget cannot finish the next probe before the
            shutdown reserve; if batch discovery fails for a candidate with anything other
            than an out-of-memory result; if every fallback batch size runs out of memory,
            which stops the lane; or if the selected batch does not complete its ten-step
            stability run.

    """
    models = tuple(render_gate_config()["probe_lanes"][lane]["models"])
    plan = probe_batch_plan()
    started, budget_seconds = clock(), _budget_seconds()
    writer.append({
        "event": "lane_plan_started",
        "lane": lane,
        "models": models,
        **_cost_progress(started=started, now=started),
        "deadline_monotonic": started + budget_seconds - SHUTDOWN_RESERVE_SECONDS,
    })
    results: list[dict[str, Any]] = []
    warned: set[float] = set()

    def run_one(
        candidate_key: str,
        batch_size: int,
        role: str,
        *,
        step_count: int,
        expected_seconds: float,
    ) -> Mapping[str, Any]:
        now = clock()
        if not _can_start(
            now=now,
            started=started,
            expected_seconds=expected_seconds,
            budget_seconds=budget_seconds,
        ):
            event = {
                "event": "budget_blocked",
                "candidate": candidate_key,
                "batch_size": batch_size,
                "role": role,
                **_cost_progress(started=started, now=now),
            }
            writer.append(event)
            msg = "budget cannot finish the next probe before shutdown reserve"
            raise RuntimeError(msg)
        spec = _work_spec(
            lane=lane,
            candidate_key=candidate_key,
            batch_size=batch_size,
            fused_adamw=False,
            artifact=artifact,
            artifact_dir=str(writer.root),
            deadline_monotonic=started + budget_seconds - SHUTDOWN_RESERVE_SECONDS,
            role=role,
            step_count=step_count,
        )
        writer.append({
            "event": "child_started",
            **_progress_child_spec(spec),
            **_cost_progress(started=started, now=now),
        })
        try:
            result = dict(child_runner(spec))
        except Exception as error:
            writer.append({
                "event": "child_failed",
                "candidate": candidate_key,
                "batch_size": batch_size,
                "error": repr(error),
                **_cost_progress(started=started, now=clock()),
            })
            raise
        completed_at = clock()
        writer.append({
            "event": "child_finished",
            **result,
            **_cost_progress(started=started, now=completed_at),
        })
        results.append(result)
        elapsed = completed_at - started
        for warning in _warning_events(elapsed, budget_seconds):
            fraction = float(warning["fraction"])
            if fraction not in warned:
                warned.add(fraction)
                writer.append({**warning, **_cost_progress(started=started, now=completed_at)})
        return result

    for candidate_key in models:
        discovery_batches = plan.discovery_batches + plan.oom_fallback_batches
        selected_batch: int | None = None
        for index, batch_size in enumerate(discovery_batches):
            role = f"discovery_b{batch_size}" if index == 0 else f"oom_fallback_b{batch_size}"
            discovery = run_one(
                candidate_key,
                batch_size,
                role,
                step_count=plan.discovery_steps,
                expected_seconds=EXPECTED_DISCOVERY_CHILD_SECONDS,
            )
            if discovery.get("status") == "ok":
                selected_batch = batch_size
                break
            if discovery.get("status") != "oom":
                msg = f"batch discovery failed for {candidate_key}: {discovery}"
                raise RuntimeError(msg)
        if selected_batch is None:
            msg = f"batch {plan.oom_fallback_batches[-1]} OOM for {candidate_key}; lane stops"
            raise RuntimeError(msg)
        chosen = run_one(
            candidate_key,
            selected_batch,
            "stability_10",
            step_count=sum(plan.steps.values()),
            expected_seconds=EXPECTED_CHILD_SECONDS,
        )
        if chosen.get("status") != "ok":
            msg = f"batch {selected_batch} did not complete 10-step stability for {candidate_key}"
            raise RuntimeError(msg)
        writer.append({
            "event": "candidate_selected",
            "candidate": candidate_key,
            "batch_size": chosen["batch_size"],
            "median_tokens_per_second": chosen["median_tokens_per_second"],
            "fused_adamw": False,
            **_cost_progress(started=started, now=clock()),
        })
    return {
        "lane": lane,
        "execution_id": writer.execution_id,
        "dataset_revision": artifact.revision,
        "budget_usd": PROBE_BUDGET_USD,
        "elapsed_seconds": clock() - started,
        "estimated_cost_usd": (clock() - started) * MODAL_H100_USD_PER_SECOND,
        "children": results,
    }


# reason: run candidate exposes lane/clock as its Modal schema; bundling would break callers.
def run_candidate_probe_plan(  # ruff: ignore[too-many-arguments]
    *,
    lane: str,
    candidate_key: str,
    batch_size: int,
    step_count: int,
    artifact: VerifiedPackedArtifact,
    child_runner: Callable[[ChildSpec], Mapping[str, Any]],
    writer: ArtifactWriter,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one explicit candidate probe; it never schedules fallback or stability.

    Returns:
        The probe record, including a ``next_action`` naming what a human should do next --
        review the ten-step result, run the ten-step probe at this batch, or review the
        out-of-memory artifacts before trying any lower batch. This entry point decides
        nothing on its own.

    Raises:
        RuntimeError: If the remaining budget cannot finish the probe before the shutdown
            reserve, plus whatever ``require_candidate_probe`` refuses about the requested
            lane, candidate, batch size and step count.

    """
    require_candidate_probe(lane, candidate_key, batch_size, step_count)
    started, budget_seconds = clock(), _budget_seconds()
    deadline = started + budget_seconds - SHUTDOWN_RESERVE_SECONDS
    expected_seconds = EXPECTED_DISCOVERY_CHILD_SECONDS if step_count == DISCOVERY_PROBE_STEPS else EXPECTED_CHILD_SECONDS
    writer.append({
        "event": "candidate_probe_started",
        "mode": "candidate_probe",
        "lane": lane,
        "candidate": candidate_key,
        "batch_size": batch_size,
        "step_count": step_count,
        "expected_seconds": expected_seconds,
        **_cost_progress(started=started, now=started),
        "deadline_monotonic": deadline,
    })
    now = clock()
    if not _can_start(
        now=now,
        started=started,
        expected_seconds=expected_seconds,
        budget_seconds=budget_seconds,
    ):
        writer.append({
            "event": "budget_blocked",
            "candidate": candidate_key,
            "batch_size": batch_size,
            "step_count": step_count,
            **_cost_progress(started=started, now=now),
        })
        msg = "budget cannot finish the next probe before shutdown reserve"
        raise RuntimeError(msg)
    spec = _work_spec(
        lane=lane,
        candidate_key=candidate_key,
        batch_size=batch_size,
        fused_adamw=False,
        artifact=artifact,
        artifact_dir=str(writer.root),
        deadline_monotonic=deadline,
        role=f"explicit_b{batch_size}_s{step_count}",
        step_count=step_count,
    )
    writer.append({
        "event": "child_started",
        **_progress_child_spec(spec),
        **_cost_progress(started=started, now=now),
    })
    try:
        child_result = dict(child_runner(spec))
    except Exception as error:
        writer.append({
            "event": "child_failed",
            "candidate": candidate_key,
            "batch_size": batch_size,
            "error": repr(error),
            **_cost_progress(started=started, now=clock()),
        })
        raise
    completed_at = clock()
    writer.append({
        "event": "child_finished",
        **child_result,
        **_cost_progress(started=started, now=completed_at),
    })
    for warning in _warning_events(completed_at - started, budget_seconds):
        writer.append({**warning, **_cost_progress(started=started, now=completed_at)})
    status = str(child_result.get("status"))
    if status not in {"ok", "oom"}:
        msg = f"candidate probe failed for {candidate_key}: {child_result}"
        raise RuntimeError(msg)
    next_action = (
        "review_10_step_result"
        if status == "ok" and step_count == FULL_PROBE_STEPS
        else f"run_explicit_10_step_probe_at_{batch_size}"
        if status == "ok"
        else "review_oom_artifacts_before_any_lower_batch"
    )
    result = {
        "status": status,
        "mode": "candidate_probe",
        "lane": lane,
        "candidate": candidate_key,
        "execution_id": writer.execution_id,
        "dataset_revision": artifact.revision,
        "batch_size": batch_size,
        "step_count": step_count,
        "next_action": next_action,
        "budget_usd": PROBE_BUDGET_USD,
        "elapsed_seconds": clock() - started,
        "estimated_cost_usd": (clock() - started) * MODAL_H100_USD_PER_SECOND,
        "children": [child_result],
    }
    writer.append({
        "event": "candidate_probe_complete",
        **result,
    })
    return result


def _child_environment() -> dict[str, str]:
    """Keep child imports on the mounted source tree even outside shell startup.

    Returns:
        The parent environment with the mounted source root prepended to PYTHONPATH, so a
        child resolves the repo before anything installed in the image.

    """
    inherited = os.environ.get("PYTHONPATH", "")
    pythonpath = MODAL_PYTHONPATH if not inherited else f"{MODAL_PYTHONPATH}:{inherited}"
    return {**os.environ, "PYTHONPATH": pythonpath}


def _subprocess_child(spec: ChildSpec) -> Mapping[str, Any]:
    """Isolate remote-code patches and CUDA allocator state in a fresh interpreter.

    Returns:
        The child's parsed result object, read from the file the child wrote rather than
        from its standard output, so a crash mid-print cannot be mistaken for a result.

    Raises:
        RuntimeError: If the child wrote no result file, whatever its exit status; if the
            result file is not valid JSON; if it parses to something other than an object;
            or if the child exited non-zero, in which case the child's own recorded error is
            reported rather than the exit code alone.

    """
    root = Path(spec.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    token = f"{spec.candidate_key}-b{spec.batch_size}-{spec.role}"
    spec_path, result_path = root / f"{token}.spec.json", root / f"{token}.result.json"
    write_child_spec(spec_path, asdict(spec))
    # reason: sys.executable and the module/flags are fixed; derived artifact paths remain single argv operands.
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.modal.base_selection",
            "--child-spec",
            str(spec_path),
            "--child-result",
            str(result_path),
        ],
        text=True,
        check=False,
        env=_child_environment(),
    )
    if not result_path.exists():
        msg = f"probe child exited {completed.returncode} and did not write a result: {result_path}"
        raise RuntimeError(msg)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = f"probe child wrote invalid JSON result: {result_path}"
        raise RuntimeError(msg) from error
    if not isinstance(result, dict):
        msg = f"probe child result is not an object: {result_path}"
        raise RuntimeError(msg)
    if completed.returncode != 0:
        msg = f"probe child failed: {result.get('error', result)}"
        raise RuntimeError(msg)
    return result


def _append_child_event(spec: ChildSpec, event: Mapping[str, Any]) -> None:
    """Durable per-step evidence, independent of parent subprocess capture."""
    path = Path(spec.artifact_dir) / f"{spec.candidate_key}-b{spec.batch_size}-{spec.role}.events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(event), sort_keys=True)
    print(encoded, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class _NvmlSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        pynvml = cast("_NvmlModule", import_module("pynvml"))

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def sample_once() -> None:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.samples.append({
                "at_monotonic": time.monotonic(),
                "gpu_sm_utilization": float(utilization.gpu),
                "gpu_memory_controller_utilization": float(utilization.memory),
                "gpu_power_watts": float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
                "device_memory_used_bytes": float(memory.used),
                "device_memory_total_bytes": float(memory.total),
            })

        def sample() -> None:
            while not self._stop.wait(0.5):
                sample_once()

        sample_once()
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.samples

    def latest(self) -> dict[str, float]:
        return dict(self.samples[-1]) if self.samples else {}


def _iter_parquet_rows(paths: Sequence[str]) -> Iterator[Mapping[str, Any]]:
    pq = import_module("pyarrow.parquet")

    while True:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for record_batch in parquet.iter_batches(batch_size=8):
                yield from record_batch.to_pylist()


def _batch_from_rows(rows: Sequence[Mapping[str, Any]], device: str) -> tuple[dict[str, Any], int]:
    import torch

    input_ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"] for row in rows], dtype=torch.long, device=device)
    positions = torch.tensor([row["position_ids"] for row in rows], dtype=torch.long, device=device)
    lengths = torch.tensor(
        [length for row in rows for length in row["seq_lengths"] if length > 0],
        dtype=torch.int32,
        device=device,
    )
    if int(lengths.sum().item()) != int(input_ids.numel()):
        msg = "packed segment lengths do not cover the physical batch"
        raise RuntimeError(msg)
    real_tokens = sum(int(row["real_token_count"]) for row in rows)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": positions,
        "packed_seq_lengths": lengths,
    }, real_tokens


def _packed_row_ranges(row: Mapping[str, object]) -> tuple[PackedRowRange, ...]:
    raw_ranges = row.get("row_ranges")
    if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
        msg = "packed row ranges are invalid"
        raise RuntimeError(msg)
    ranges: list[PackedRowRange] = []
    for item in raw_ranges:
        if not isinstance(item, Mapping):
            msg = "packed row range is invalid"
            raise RuntimeError(msg)
        uid, start, end = item.get("uid"), item.get("start"), item.get("end")
        if not isinstance(uid, str) or not isinstance(start, int) or not isinstance(end, int):
            msg = "packed row range lacks uid, start, or end"
            raise RuntimeError(msg)
        ranges.append(PackedRowRange(uid, start, end))
    return tuple(ranges)


def _packed_ints(row: Mapping[str, object], key: str) -> tuple[int, ...]:
    values = row.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        msg = f"packed row {key} is invalid"
        raise RuntimeError(msg)
    narrowed = tuple(value for value in values if isinstance(value, int))
    if len(narrowed) != len(values):
        msg = f"packed row {key} must contain integers"
        raise RuntimeError(msg)
    return narrowed


def _packed_strings(row: Mapping[str, object], key: str) -> tuple[str, ...]:
    values = row.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        msg = f"packed row {key} is invalid"
        raise RuntimeError(msg)
    narrowed = tuple(value for value in values if isinstance(value, str))
    if len(narrowed) != len(values):
        msg = f"packed row {key} must contain strings"
        raise RuntimeError(msg)
    return narrowed


def _packed_training_unit_from_row(
    row: Mapping[str, object],
    *,
    input_ids: tuple[int, ...] | None = None,
) -> tuple[PackedTrainingUnit, tuple[PackedRowRange, ...]]:
    ranges = _packed_row_ranges(row)
    row_input_ids = _packed_ints(row, "input_ids") if input_ids is None else input_ids
    real_token_count, padded_token_count = (
        row.get("real_token_count"),
        row.get("padded_token_count"),
    )
    if not isinstance(real_token_count, int) or not isinstance(padded_token_count, int):
        msg = "packed row token counts are invalid"
        raise RuntimeError(msg)
    return (
        PackedTrainingUnit(
            input_ids=row_input_ids,
            labels=_packed_ints(row, "labels"),
            seq_lengths=_packed_ints(row, "seq_lengths"),
            position_ids=_packed_ints(row, "position_ids"),
            row_uids=_packed_strings(row, "row_uids"),
            row_ranges=ranges,
            real_token_count=real_token_count,
            padded_token_count=padded_token_count,
        ),
        ranges,
    )


def _load_tagger(
    candidate_key: str,
    *,
    attn_implementation: str,
) -> tuple[HiddenStateTokenTagger, LoraEvidence, MemoryContract]:
    """Tagger implements universal segment-by-segment isolation.

    It works for both causal and bidirectional bodies without trusting undocumented masks.

    Returns:
        The constructed tagger, the LoRA evidence read back off the built model, and the
        memory contract measured for it -- evidence rather than the configuration that was
        requested, so a silently-ignored setting shows up.

    Raises:
        RuntimeError: If the loaded backbone is not a torch module and therefore cannot be
            adapted with LoRA.

    """
    # reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
    # reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
    # reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
    import peft
    import torch
    from transformers import (
        AutoModel,  # ty: ignore[possibly-missing-import]
        AutoModelForMaskedLM,
        AutoModelForTokenClassification,
    )

    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger

    candidate = CANDIDATES[candidate_key]
    if candidate.loader == "token_classifier_body":
        loaded = load_pii350_body(AutoModelForTokenClassification, attn_implementation=attn_implementation)
    elif candidate.loader == "masked_lm_body":
        loaded = load_encoder_body(AutoModelForMaskedLM, candidate, attn_implementation=attn_implementation)
    else:
        loaded = load_standard_body(AutoModel, candidate, attn_implementation=attn_implementation)
    memory_contract = enable_native_gradient_checkpointing(loaded.body)
    body = loaded.body.cuda().train()
    hidden_size = int(getattr(body.config, "hidden_size", 1024))
    torch.manual_seed(GATE_SEED)
    tagger = (
        HiddenStateTokenTagger(
            body,
            hidden_size,
            37,
            request_hidden_states=memory_contract.output_hidden_states,
            classifier_dtype=torch.bfloat16,
            packed_segment_isolation=True,
        )
        .cuda()
        .train()
    )
    if not isinstance(tagger.backbone, torch.nn.Module):
        msg = "BIOES tagger backbone is not a torch module"
        raise RuntimeError(msg)
    adapted, evidence = apply_gate_lora(tagger.backbone, tagger.classifier, peft_module=peft)
    tagger.backbone = adapted
    return tagger, evidence, memory_contract


def _require_child_spec(spec: ChildSpec) -> None:
    if spec.candidate_key not in CANDIDATES:
        msg = "unknown candidate"
        raise ValueError(msg)
    if spec.batch_size not in {64, 96, 128, 160, 192, 200, 224, 240, 256}:
        msg = "batch size is outside the probe contract"
        raise ValueError(msg)
    if spec.fused_adamw:
        msg = "base-selection gate requires unfused AdamW"
        raise ValueError(msg)
    if spec.step_count not in {2, 10}:
        msg = "probe step count is outside the contract"
        raise ValueError(msg)


# reason: run child owns load tagger and validate together; splitting would fragment diagnostics.
def _run_child(spec: ChildSpec) -> dict[str, Any]:  # ruff: ignore[complex-structure,too-many-locals,too-many-statements]
    """Real CUDA work. This is called only by the fresh subprocess entrypoint.

    A final, sparsely filled shard unit can contain one document. Find a genuine boundary rather than treating that as
    evidence of isolation.

    Returns:
        The child record. ``status`` is ``"ok"`` with the per-step timings and their median,
        ``"oom"`` when CUDA ran out of memory, or ``"failed"`` with the repr of whatever
        else was raised. Every one of the three is returned, not raised, so the parent gets
        a record instead of a traceback.

    Raises:
        RuntimeError: If CUDA is absent. That check sits above the try block, so it is the one
            refusal that escapes to ``_child_main``. The later raises -- no packed unit with
            two segments, a contamination report, the budget deadline arriving before the next
            optimizer step, and the BIOES head returning no loss -- are inside the try and are
            caught, becoming the ``"failed"`` record above rather than propagating.

    """
    import torch

    from meddies_pii.training.bioes.trainers.contamination import (
        run_packed_attention_contamination_probe,
    )

    _require_child_spec(spec)
    if not torch.cuda.is_available():
        msg = "H100 probe child requires CUDA"
        raise RuntimeError(msg)
    iterator = _iter_parquet_rows(spec.shard_paths)
    sampler = _NvmlSampler()
    events: list[dict[str, Any]] = []
    started = time.monotonic()

    def child_progress(event: Mapping[str, Any]) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - started)
        return {
            **event,
            "run_elapsed_seconds": elapsed,
            "estimated_cost_usd": elapsed * MODAL_H100_USD_PER_SECOND,
            "budget_usd": PROBE_BUDGET_USD,
            "deadline_monotonic": spec.deadline_monotonic,
        }

    started_event = child_progress({
        "event": "child_process_started",
        "candidate": spec.candidate_key,
        "batch_size": spec.batch_size,
        "role": spec.role,
    })
    events.append(started_event)
    _append_child_event(spec, started_event)
    # reason: run child's try keeps load tagger with validate; splitting would fragment diagnostics.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        first = next(iterator)
        for _ in range(32):
            if len(_packed_row_ranges(first)) >= MIN_PACKED_SEGMENTS:
                break
            first = next(iterator)
        clean, ranges = _packed_training_unit_from_row(first)
        if len(ranges) >= MIN_PACKED_SEGMENTS:
            mutated_ids = list(first["input_ids"])
            mutated_ids[ranges[0].start] = (mutated_ids[ranges[0].start] + 1) % 100
            mutated, _ = _packed_training_unit_from_row(first, input_ids=tuple(mutated_ids))
        else:
            msg = "packed contamination probe needs a unit with two segments"
            # reason: this refusal must become the enclosing paid-child failure record rather than escape unpersisted.
            raise RuntimeError(msg)  # ruff: ignore[raise-within-try]

        def validate_contamination(tagger: HiddenStateTokenTagger) -> None:
            contamination = run_packed_attention_contamination_probe(
                cast("nn.Module", tagger),
                clean_unit=clean,
                mutated_unit=mutated,
                target_row_uid=ranges[1].uid,
                device="cuda",
                atol=0.0,
                rtol=0.0,
            )
            # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
            # reason: equality, so any tolerance here would let a real drift or contamination pass the gate.
            if not contamination.passed or contamination.max_abs_diff != 0.0:  # ruff: ignore[float-equality-comparison]
                msg = f"packed contamination probe failed: {contamination}"
                # reason: this nested validation runs inside the paid-child boundary that persists the failed proof.
                raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
            contamination_event = child_progress({
                "event": "contamination_passed",
                **contamination.to_dict(),
            })
            events.append(contamination_event)
            _append_child_event(spec, contamination_event)

        attention = "sdpa"
        tagger, lora, memory_contract = _load_tagger(spec.candidate_key, attn_implementation=attention)
        memory_event = child_progress({
            "event": "memory_contract_enabled",
            "memory_contract": asdict(memory_contract),
            "attention": attention,
            "fused_adamw": False,
        })
        events.append(memory_event)
        _append_child_event(spec, memory_event)
        validate_contamination(tagger)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in tagger.parameters() if parameter.requires_grad],
            **adamw_kwargs(fused=spec.fused_adamw),
        )
        static_bytes = int(torch.cuda.memory_allocated())
        sampler.start()
        timings: list[float] = []
        step_durations: list[float] = []
        for step in range(spec.step_count):
            expected_step_seconds = max(
                5.0,
                sum(step_durations) / len(step_durations) if step_durations else 5.0,
            )
            if time.monotonic() + (spec.step_count - step) * expected_step_seconds >= spec.deadline_monotonic:
                msg = "budget deadline reached before next optimizer step"
                # reason: this stop must be caught and persisted by the enclosing paid-child failure boundary.
                raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
            rows = [next(iterator) for _ in range(spec.batch_size)]
            batch, tokens = _batch_from_rows(rows, "cuda")
            torch.cuda.synchronize()
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            outputs = tagger(**batch)
            loss = outputs["loss"]
            if loss is None:
                msg = "BIOES head returned no loss"
                # reason: this model-contract failure must be caught and persisted with the partial paid-run evidence.
                raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
            torch.autograd.backward(loss)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - before
            step_durations.append(elapsed)
            event = child_progress({
                "event": "step",
                "step": step,
                "elapsed_seconds": elapsed,
                "real_tokens": tokens,
                "real_bioes_tokens_per_second": tokens / elapsed,
                "loss": float(loss.detach().float().cpu()),
                "vram_allocated_bytes": int(torch.cuda.memory_allocated()),
                "vram_reserved_bytes": int(torch.cuda.memory_reserved()),
                "vram_peak_bytes": int(torch.cuda.max_memory_allocated()),
                **sampler.latest(),
                "fused_adamw": spec.fused_adamw,
                "candidate": spec.candidate_key,
                "revision": CANDIDATES[spec.candidate_key].revision,
                "batch_size": spec.batch_size,
                "data_cursor": step * spec.batch_size,
            })
            events.append(event)
            _append_child_event(spec, event)
            if spec.step_count == FULL_PROBE_STEPS and step >= TIMING_WARMUP_STEPS:
                timings.append(float(event["real_bioes_tokens_per_second"]))
        median = sorted(timings)[len(timings) // 2] if timings else None
        return {
            "status": "ok",
            "candidate": spec.candidate_key,
            "batch_size": spec.batch_size,
            "fused_adamw": spec.fused_adamw,
            "attention": attention,
            "median_tokens_per_second": median,
            "memory_contract": asdict(memory_contract),
            "static_bytes": static_bytes,
            "peak_bytes": int(torch.cuda.max_memory_allocated()),
            "total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "lora": asdict(lora),
            "events": events,
            "nvml": sampler.stop(),
        }
    except torch.cuda.OutOfMemoryError as error:
        result = {
            "status": "oom",
            "candidate": spec.candidate_key,
            "batch_size": spec.batch_size,
            "error": repr(error),
            "events": events,
            "nvml": sampler.stop(),
        }
        _append_child_event(spec, child_progress({"event": "oom", **result}))
        return result
    # reason: candidate execution persists every CUDA, model, or data failure as evidence before returning.
    except Exception as error:  # ruff: ignore[blind-except]
        result = {
            "status": "failed",
            "candidate": spec.candidate_key,
            "batch_size": spec.batch_size,
            "error": repr(error),
            "events": events,
            "nvml": sampler.stop(),
        }
        _append_child_event(spec, child_progress({"event": "failed", **result}))
        return result


def _child_main(spec_path: str, result_path: str) -> int:
    """Run with subprocess-owned CUDA state and a shared evidence volume.

    Commit here so handled OOM/failure and all preceding steps survive parent exit.

    Returns:
        The process exit code: 0 only when the child ran and did not report failure, 1 both
        when it reported a failed status and when it raised. The result file and the volume
        commit happen on every path, including the raising one, which is the point of
        catching here rather than letting the exception end the process.

    """
    spec = ChildSpec(**json.loads(Path(spec_path).read_text(encoding="utf-8")))
    result: dict[str, Any]
    try:
        result = _run_child(spec)
        exit_code = 1 if result.get("status") == "failed" else 0
    # reason: this subprocess terminal boundary writes and commits a failure result for every otherwise-unhandled error.
    except Exception as error:  # ruff: ignore[blind-except]
        result = {
            "status": "failed",
            "candidate": spec.candidate_key,
            "batch_size": spec.batch_size,
            "error": repr(error),
        }
        exit_code = 1
    Path(result_path).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.commit()
    return exit_code


# reason: run probe lane exposes lane/step count as its Modal schema; bundling would break callers.
@app.function(**LANE_OPTIONS)
def run_probe_lane(  # ruff: ignore[too-many-arguments]
    lane: str,
    *,
    execute: bool = False,
    confirmation: str = "",
    estimated_cost_usd: float = PROBE_BUDGET_USD,
    dataset_revision: str = "",
    mode: str = "full_lane",
    candidate_key: str = "",
    batch_size: int = 0,
    step_count: int = 0,
) -> dict[str, Any]:
    _probe_lane_guard(
        lane,
        execute=execute,
        confirmation=confirmation,
        estimated_cost_usd=estimated_cost_usd,
    )
    require_probe_mode(lane, mode, candidate_key, batch_size, step_count)
    if not dataset_revision:
        msg = "MEDDIES_PACKED_REVISION is required; packed is not assumed published"
        raise RuntimeError(msg)
    writer = ArtifactWriter(ARTIFACT_ROOT, lane, artifacts.commit)
    lifecycle_started = time.monotonic()
    writer.append({
        "event": "lane_started",
        "lane": lane,
        "mode": mode,
        "candidate": candidate_key or None,
        "batch_size": batch_size or None,
        "step_count": step_count or None,
        "dataset_revision": dataset_revision,
        "status": "running",
        **_cost_progress(started=lifecycle_started, now=lifecycle_started),
    })
    # reason: run probe lane's try keeps verify with run probe; splitting would fragment diagnostics.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        artifact = verify_packed_artifact(revision=dataset_revision)
        writer.append({
            "event": "packed_verified",
            "revision": artifact.revision,
            "manifest_path": artifact.manifest_path,
            "shards": len(artifact.shard_paths),
            "manifest_sha256": PACKED_MANIFEST_SHA256,
            **_cost_progress(started=lifecycle_started, now=time.monotonic()),
        })
        if mode == "candidate_probe":
            result = run_candidate_probe_plan(
                lane=lane,
                candidate_key=candidate_key,
                batch_size=batch_size,
                step_count=step_count,
                artifact=artifact,
                child_runner=_subprocess_child,
                writer=writer,
            )
        else:
            result = run_lane_plan(
                lane=lane,
                artifact=artifact,
                child_runner=_subprocess_child,
                writer=writer,
            )
        writer.append({
            "event": "lane_complete",
            "lane": lane,
            "mode": mode,
            "candidate": candidate_key or None,
            "batch_size": batch_size or None,
            "step_count": step_count or None,
            "status": result.get("status", "ok"),
            **_cost_progress(started=lifecycle_started, now=time.monotonic()),
        })
        final_result = {
            "status": "ok",
            **result,
            "execution_id": writer.execution_id,
            "runtime_backend": "transformers_peft" if mode == "candidate_probe" else None,
            "optimizer_steps": step_count if mode == "candidate_probe" else None,
            "artifact_root": str(writer.root),
        }
        writer.finalize(final_result)
    except Exception as error:
        failure = {"status": "failed", "lane": lane, "error": repr(error)}
        writer.append({
            "event": "lane_failed",
            **failure,
            **_cost_progress(started=lifecycle_started, now=time.monotonic()),
        })
        writer.finalize(failure)
        raise
    else:
        return final_result


# reason: base selection exposes lane/step count as its Modal schema; bundling would break callers.
@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    lane: str = "",
    execute: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    confirmation: str = "",
    dataset_revision: str = "",
    mode: str = "full_lane",
    candidate_key: str = "",
    batch_size: int = 0,
    step_count: int = 0,
) -> None:
    if not lane:
        msg = "choose exactly one lane: --lane 230 or --lane 350"
        raise RuntimeError(msg)
    require_modal_profile()
    _probe_lane_guard(
        lane,
        execute=execute,
        confirmation=confirmation,
        estimated_cost_usd=PROBE_BUDGET_USD,
    )
    require_probe_mode(lane, mode, candidate_key, batch_size, step_count)
    if not dataset_revision:
        dataset_revision = os.environ.get("MEDDIES_PACKED_REVISION", "")
    result = run_probe_lane.remote(
        lane,
        execute=execute,
        confirmation=confirmation,
        dataset_revision=dataset_revision,
        mode=mode,
        candidate_key=candidate_key,
        batch_size=batch_size,
        step_count=step_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    child_paths = child_argv_paths(sys.argv)
    if child_paths is not None:
        raise SystemExit(_child_main(*child_paths))
