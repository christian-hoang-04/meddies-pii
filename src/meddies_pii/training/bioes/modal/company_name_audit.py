"""CPU-only Modal audit for company_name support in packed BIOES training data."""

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
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import modal

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.modal.probe_scaffold import (
    CACHE_ROOT,
    VolumeMounts,
    artifact_volume,
    hf_cache_environment,
    hf_cache_volume,
    hf_secret,
    source_mounted_image,
    volume_mounts,
)
from meddies_pii.training.bioes.trainers.company_name_audit import (
    PREFIX_PACKED_UNITS,
    CompanyNameCounter,
    PackedCompanyNameAudit,
    audit_eval_records,
    authoritative_label_maps,
    merge_packed_company_name_audits,
    render_modal_command,
    require_execution,
)
from meddies_pii.training.bioes.trainers.pins import (
    EVAL_CONFIG,
    EVAL_DATASET_ID,
    EVAL_DATASET_REVISION,
    EVAL_ROWS,
    EVAL_SPLIT,
    PACKED_DATASET_ID,
    PACKED_DATASET_REVISION,
    PACKED_MANIFEST_SHA256,
    PACKED_SHARD_COUNT,
    PACKED_UNIT_COUNT,
)

EXPECTED_MANIFEST_ROWS = 1_000_000
SHA256_HEX_LENGTH = 64
MIN_AUDIT_WORKERS = 2

ARTIFACT_ROOT = "/artifacts/company-name-audit"
STOP_COMMAND = "MODAL_PROFILE=meddies-pii uv run modal app stop meddies-bioes-company-name-audit"


class _CpuOptions(TypedDict):
    cpu: float
    memory: tuple[int, int]
    timeout: int
    max_containers: int


CPU_OPTIONS: _CpuOptions = {
    "cpu": 12.0,
    "memory": (32 * 1024, 64 * 1024),
    "timeout": 7_200,
    "max_containers": 1,
}
FULL_AUDIT_WORKERS = int(CPU_OPTIONS["cpu"])
IMAGE_ENVIRONMENT = hf_cache_environment(
    offline=None,
    disable_hub_telemetry=True,
    disable_unsloth_statistics=False,
)
image = source_mounted_image(
    modal.Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install("pyarrow==23.0.0", "datasets==4.3.0", "huggingface_hub==1.11.0")
    .env(IMAGE_ENVIRONMENT),
)
cache = hf_cache_volume()
artifacts = artifact_volume()
secret = hf_secret()
app = modal.App("meddies-bioes-company-name-audit", image=image)
MOUNTS: VolumeMounts = volume_mounts(cache, artifacts)


class ArtifactWriter:
    def __init__(self, root: str | Path, commit: Callable[[], None] | None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.commit = commit

    def event(self, payload: Mapping[str, object]) -> None:
        encoded = json.dumps(dict(payload), sort_keys=True)
        print(encoded, flush=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.commit is not None:
            self.commit()

    def result(self, payload: Mapping[str, object]) -> Path:
        path = self.root / "result.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        if self.commit is not None:
            self.commit()
        return path


def _remote_guard(confirmation: str) -> None:
    require_execution(confirmation)


def render_dry_run(profile: str) -> str:
    return render_modal_command(profile)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest_shards(manifest_path: Path) -> tuple[Mapping[str, object], ...]:
    if _sha_file(manifest_path) != PACKED_MANIFEST_SHA256:
        msg = "packed manifest SHA-256 does not match the full-run pin"
        raise RuntimeError(msg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        msg = "packed manifest is not an object"
        raise RuntimeError(msg)
    shards = manifest.get("shards")
    if (
        manifest.get("packed_config") != "packed"
        or manifest.get("complete") is not True
        or manifest.get("resume_cursor") != EXPECTED_MANIFEST_ROWS
        or not isinstance(shards, list)
        or len(shards) != PACKED_SHARD_COUNT
    ):
        msg = "packed manifest cannot support the requested audit"
        raise RuntimeError(msg)
    records: list[Mapping[str, object]] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            msg = "packed manifest has an invalid shard"
            raise RuntimeError(msg)
        path, size, digest = shard.get("path"), shard.get("bytes"), shard.get("sha256")
        # reason: manifest shards keeps startswith/packed in one gate; helper predicates would scatter the rule.
        if (
            not isinstance(path, str)  # ruff: ignore[too-many-boolean-expressions]
            or not path.startswith("packed/data/")
            or not isinstance(size, int)
            or size < 1
            or not isinstance(digest, str)
            or len(digest) != SHA256_HEX_LENGTH
        ):
            msg = "packed manifest shard lacks an exact path, size, or SHA-256"
            raise RuntimeError(msg)
        records.append(shard)
    return tuple(records)


def _iter_parquet_rows(path: str) -> Iterator[Mapping[str, object]]:
    from importlib import import_module

    pq = import_module("pyarrow.parquet")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            if not is_str_mapping(row):
                msg = "packed parquet record is not an object"
                raise RuntimeError(msg)
            yield dict(row)


def _audit_local_shard(path: str) -> PackedCompanyNameAudit:
    """Parse one verified shard in a worker process and retain only its receipt.

    Returns:
        The shard's finished `PackedCompanyNameAudit` receipt. Only the receipt crosses the
        process boundary; the parsed rows stay in the worker.

    """
    label2id, id2label = authoritative_label_maps()
    counter = CompanyNameCounter(label2id, id2label)
    for row in _iter_parquet_rows(path):
        counter.consume(row)
    return counter.finish()


def _download_verified_shard(
    shard: Mapping[str, object],
    download: Callable[[Mapping[str, object]], Path],
) -> tuple[str, int, Path]:
    filename, expected_bytes, expected_digest = (
        shard.get("path"),
        shard.get("bytes"),
        shard.get("sha256"),
    )
    if not isinstance(filename, str) or not isinstance(expected_bytes, int) or not isinstance(expected_digest, str):
        msg = "validated packed manifest changed shape"
        raise RuntimeError(msg)
    local_path = download(shard)
    if local_path.stat().st_size != expected_bytes or _sha_file(local_path) != expected_digest:
        msg = f"packed shard integrity check failed: {filename}"
        raise RuntimeError(msg)
    return filename, expected_bytes, local_path


def _audit_ordered_prefix(
    shards: Sequence[Mapping[str, object]],
    *,
    download: Callable[[Mapping[str, object]], Path],
    row_iter: Callable[[str], Iterator[Mapping[str, object]]],
    emit: Callable[[Mapping[str, object]], None],
    prefix_units: int,
) -> PackedCompanyNameAudit:
    """Read only the ordered shard prefix required for the exact 7,680-unit receipt.

    Returns:
        The receipt for exactly `prefix_units` packed units, finished with that count as its
        required-unit pin so a short or over-long prefix cannot be reported as complete.

    """
    label2id, id2label = authoritative_label_maps()
    counter = CompanyNameCounter(label2id, id2label)
    started = time.monotonic()
    for shard_index, shard in enumerate(shards, start=1):
        _filename, _expected_bytes, local_path = _download_verified_shard(shard, download)
        for row in row_iter(str(local_path)):
            if counter.processed_packed_units == prefix_units:
                break
            counter.consume(row)
        emit({
            "event": "prefix_shard_complete",
            "shard_index": shard_index,
            "shard_count": len(shards),
            "prefix_packed_units_processed": counter.processed_packed_units,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stop_command": STOP_COMMAND,
            "message": f"Completed prefix shard {shard_index}/{len(shards)}",
        })
        if counter.processed_packed_units == prefix_units:
            return counter.finish(required_units=prefix_units)
    return counter.finish(required_units=prefix_units)


def _parallel_full_shard_audit(
    shards: Sequence[Mapping[str, object]],
    *,
    download: Callable[[Mapping[str, object]], Path],
    emit: Callable[[Mapping[str, object]], None],
    full_units: int,
    worker_count: int = FULL_AUDIT_WORKERS,
) -> PackedCompanyNameAudit:
    """Bound parsing to physical CPUs while retaining only per-shard audit receipts.

    Returns:
        The merged receipt over every shard, once its processed-unit count equals `full_units`.

    Raises:
        ValueError: if `worker_count` is below the two-process minimum.
        RuntimeError: if the merged receipt is short of `full_units`, which means the corpus
            geometry did not match the pin rather than that a shard merely failed.

    """
    if worker_count < MIN_AUDIT_WORKERS:
        msg = "full packed audit requires at least two worker processes"
        raise ValueError(msg)
    started = time.monotonic()
    pending: dict[Future[PackedCompanyNameAudit], tuple[int, int]] = {}
    reports: list[PackedCompanyNameAudit] = []
    completed_bytes = 0
    completed_units = 0

    def collect_completed() -> None:
        nonlocal completed_bytes, completed_units
        done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
        for future in sorted(done, key=lambda item: pending[item][0]):
            shard_index, expected_bytes = pending.pop(future)
            report = future.result()
            reports.append(report)
            completed_bytes += expected_bytes
            completed_units += report.processed_packed_units
            emit({
                "event": "shard_complete",
                "shard_index": shard_index,
                "shard_count": len(shards),
                "packed_units_processed": completed_units,
                "bytes_verified": completed_bytes,
                "worker_count": worker_count,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "stop_command": STOP_COMMAND,
                "message": f"Completed shard {shard_index}/{len(shards)}",
            })

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for shard_index, shard in enumerate(shards, start=1):
            _filename, expected_bytes, local_path = _download_verified_shard(shard, download)
            future = executor.submit(_audit_local_shard, str(local_path))
            pending[future] = (shard_index, expected_bytes)
            if len(pending) >= worker_count:
                collect_completed()
        while pending:
            collect_completed()
    report = merge_packed_company_name_audits(reports)
    if report.processed_packed_units != full_units:
        msg = f"packed full-audit geometry is short: expected {full_units}, got {report.processed_packed_units}"
        raise RuntimeError(
            msg,
        )
    return report


# reason: stream packed exposes shards/full units as its Modal schema; bundling would break callers.
def _stream_packed_audit(  # ruff: ignore[too-many-arguments]
    shards: Sequence[Mapping[str, object]],
    *,
    download: Callable[[Mapping[str, object]], Path],
    row_iter: Callable[[str], Iterator[Mapping[str, object]]],
    emit: Callable[[Mapping[str, object]], None],
    prefix_units: int = PREFIX_PACKED_UNITS,
    full_units: int = PACKED_UNIT_COUNT,
) -> tuple[PackedCompanyNameAudit, PackedCompanyNameAudit]:
    """Keep the prefix deterministic, then parse the full corpus on 12 workers.

    Returns:
        The prefix receipt and the full-corpus receipt, in that order. Both are produced here so
        the deterministic prefix can be compared against the parallel full pass.

    """
    prefix_report = _audit_ordered_prefix(
        shards,
        download=download,
        row_iter=row_iter,
        emit=emit,
        prefix_units=prefix_units,
    )
    full_report = _parallel_full_shard_audit(shards, download=download, emit=emit, full_units=full_units)
    return prefix_report, full_report


@app.function(image=image, volumes=MOUNTS, secrets=[secret], **CPU_OPTIONS)
def run_company_name_audit(confirmation: str) -> dict[str, object]:
    """Download and stream the pinned corpus only inside the CPU Modal worker.

    Returns:
        The audit result mapping written to the artifact volume for this execution id.

    """
    _remote_guard(confirmation)
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    execution_id = uuid.uuid4().hex
    writer = ArtifactWriter(Path(ARTIFACT_ROOT) / execution_id, artifacts.commit)
    started = time.monotonic()
    writer.event({
        "event": "audit_started",
        "execution_id": execution_id,
        "packed_revision": PACKED_DATASET_REVISION,
        "eval_revision": EVAL_DATASET_REVISION,
        "prefix_packed_units": PREFIX_PACKED_UNITS,
        "full_packed_units": PACKED_UNIT_COUNT,
        "full_audit_workers": FULL_AUDIT_WORKERS,
        "memory_mib": CPU_OPTIONS["memory"],
        "stop_command": STOP_COMMAND,
        "message": "Starting pinned company_name audit",
    })
    manifest_path = Path(
        hf_hub_download(
            repo_id=PACKED_DATASET_ID,
            repo_type="dataset",
            revision=PACKED_DATASET_REVISION,
            filename="packed/manifest.json",
            cache_dir=CACHE_ROOT,
            local_files_only=False,
        ),
    )
    shards = _manifest_shards(manifest_path)

    def download_shard(shard: Mapping[str, object]) -> Path:
        filename = shard.get("path")
        if not isinstance(filename, str):
            msg = "validated packed manifest changed shape"
            raise RuntimeError(msg)
        return Path(
            hf_hub_download(
                repo_id=PACKED_DATASET_ID,
                repo_type="dataset",
                revision=PACKED_DATASET_REVISION,
                filename=filename,
                cache_dir=CACHE_ROOT,
                local_files_only=False,
            ),
        )

    prefix_report, full_report = _stream_packed_audit(
        shards,
        download=download_shard,
        row_iter=_iter_parquet_rows,
        emit=writer.event,
    )
    evaluation = load_dataset(
        EVAL_DATASET_ID,
        EVAL_CONFIG,
        split=EVAL_SPLIT,
        revision=EVAL_DATASET_REVISION,
        cache_dir=CACHE_ROOT,
    )
    eval_report = audit_eval_records((dict(row) for row in evaluation), expected_rows=EVAL_ROWS)
    result: dict[str, object] = {
        "status": "ok",
        "execution_id": execution_id,
        "packed": {
            "id": PACKED_DATASET_ID,
            "revision": PACKED_DATASET_REVISION,
            "manifest_sha256": PACKED_MANIFEST_SHA256,
            "shard_count": len(shards),
            "full_audit_workers": FULL_AUDIT_WORKERS,
            "memory_mib": CPU_OPTIONS["memory"],
            "prefix_7680": prefix_report.as_dict(),
            "full": full_report.as_dict(),
        },
        "evaluation": {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            **eval_report.as_dict(),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stop_command": STOP_COMMAND,
    }
    result_path = writer.result(result)
    writer.event({
        "event": "audit_complete",
        "status": "ok",
        "result_path": str(result_path),
        "stop_command": STOP_COMMAND,
        "message": "Pinned company_name audit completed",
    })
    return result


@app.local_entrypoint()
def main(confirmation: str = "") -> None:
    _remote_guard(confirmation)
    result = run_company_name_audit.remote(confirmation)
    print(json.dumps(result, indent=2, sort_keys=True))
