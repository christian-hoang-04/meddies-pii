"""Modal-only, fail-closed two-A100 DDP smoke for the PII350 constant-3e-4 recipe."""

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
import re

# reason: the Modal parent launches the fixed torchrun worker with list-form argv inside its pinned image.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, TypedDict

import modal

from meddies_pii.training.bioes.modal.full_run import (
    COMMON_MOUNTS,
    CPU_PREFLIGHT_OPTIONS,
    PREFLIGHT_RECEIPT_SCHEMA_VERSION,
    _asset_receipt_contract,
    _preflight_candidate_assets,
    _prewarm_candidate_assets,
    _receipt_path,
    _validated_receipt_inventory,
    artifacts,
    cache,
    secret,
)
from meddies_pii.training.bioes.modal.probe_scaffold import (
    TrainableModule,
    append_event,
    child_argv_paths,
    hf_cache_environment,
    print_stderr_sink,
    print_stdout_sink,
    require_modal_profile,
    source_mounted_image,
    stream_child_output,
)
from meddies_pii.training.bioes.trainers.pii350_ddp_smoke import (
    ALL_IN_RATE_USD_PER_SECOND,
    BF16_PARAMETER_DRIFT_TOLERANCE,
    DDP_GRADIENT_REDUCTION,
    DDP_SMOKE_CONFIRMATION,
    DDP_SMOKE_PRIMARY_ACTION,
    GPU_SPEC,
    HARD_TIMEOUT_SECONDS,
    LOCAL_BATCH_SIZE,
    MODAL_PROFILE,
    OPTIMIZER_STEPS,
    TRAINING_DEADLINE_SECONDS,
    WORLD_SIZE,
    absolute_training_deadline,
    aggregate_real_tokens_per_second,
    matched_constant3_contract,
    ordered_identity_digest,
    packed_unit_identities,
    rank_window,
    render_ddp_smoke_contract,
    require_ddp_smoke_execute,
    scale_local_token_mean_loss,
    training_deadline_reached,
    validate_collective_readiness,
    validate_rank_identity_coverage,
    validate_rank_windows,
)

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from meddies_pii.training.bioes.modal.probe_scaffold import ChildProcess


class _DdpOptions(TypedDict):
    gpu: str
    cpu: float
    memory: int
    timeout: int
    max_containers: int


class _CpuPrewarmOptions(TypedDict):
    gpu: None
    cpu: float
    memory: int
    timeout: int
    max_containers: int


DEADLINE_ABS_TOLERANCE = 1e-6
LABEL_IGNORE_INDEX = -100
MIN_PACKED_SEGMENTS = 2
PERSISTENCE_RESERVE_SECONDS = 300

ARTIFACT_ROOT = "/artifacts/pii350-ddp-smoke"
STEP_ONE_DIAGNOSTIC_PHASES = (
    "post_ddp_wrap_parameters",
    "pre_explicit_reduction_gradients",
    "post_explicit_reduction_gradients",
    "post_optimizer_parameters",
)
CACHE_ENVIRONMENT = hf_cache_environment()
ONLINE_CACHE_ENVIRONMENT = hf_cache_environment(
    offline=False,
    disable_hub_telemetry=True,
)
"""Prewarm must never inherit the training image's offline-mode contract."""
DDP_OPTIONS: _DdpOptions = {
    "gpu": GPU_SPEC,
    "cpu": 4.0,
    "memory": 64 * 1024,
    "timeout": HARD_TIMEOUT_SECONDS,
    "max_containers": 1,
}
CPU_PREWARM_OPTIONS: _CpuPrewarmOptions = {
    "gpu": None,
    "cpu": 12.0,
    "memory": 64 * 1024,
    "timeout": 3_600,
    "max_containers": 1,
}
IMAGE_PACKAGES = (
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
package_image = modal.Image.from_registry(
    "python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941",
).pip_install(*IMAGE_PACKAGES)
"""The two images share the expensive package-install layer.

Only the final environment layer differs: training/receipt is offline; prewarm is online.

"""
image = source_mounted_image(package_image.env(CACHE_ENVIRONMENT))
prewarm_image = source_mounted_image(package_image.env(ONLINE_CACHE_ENVIRONMENT))
app = modal.App("meddies-pii350-ddp-a100-smoke")


@dataclass(frozen=True, slots=True)
class ChildSpec:
    contract: dict[str, Any]
    matched_constant3_contract: dict[str, Any]
    shard_paths: tuple[str, ...]
    expected_encoder_checkpoint_attestation: dict[str, Any]
    artifact_dir: str
    outer_modal_started_monotonic: float
    """torchrun ranks are sibling processes in this one Modal Linux container.

    So Python's monotonic clock has the same origin for both parent and child.

    """
    absolute_training_deadline_monotonic: float
    pre_torchrun_interconnect: dict[str, Any]
    """Captured by the single Modal parent before torchrun.

    Never ask rank zero to run a topology command while rank one waits in a collective.

    """


def validate_child_spec(spec: ChildSpec) -> None:
    require_ddp_smoke_execute(
        spec.contract,
        execute=True,
        confirmation=DDP_SMOKE_CONFIRMATION,
        primary_action=DDP_SMOKE_PRIMARY_ACTION,
    )
    if spec.matched_constant3_contract != matched_constant3_contract():
        msg = "DDP smoke must use the exact matched constant-3e-4 scout contract"
        raise RuntimeError(msg)
    if tuple(spec.contract.get("runtime_packages", ())) != IMAGE_PACKAGES:
        msg = "DDP smoke image packages disagree with the immutable contract"
        raise RuntimeError(msg)
    if not spec.shard_paths:
        msg = "DDP smoke requires CPU-verified packed shards"
        raise RuntimeError(msg)
    if not spec.expected_encoder_checkpoint_attestation:
        msg = "DDP smoke requires CPU-verified encoder checkpoint attestation"
        raise RuntimeError(msg)
    if not Path(spec.artifact_dir).is_absolute():
        msg = "DDP smoke artifact directory must be absolute"
        raise RuntimeError(msg)
    expected_deadline = absolute_training_deadline(outer_started_monotonic=spec.outer_modal_started_monotonic)
    if abs(spec.absolute_training_deadline_monotonic - expected_deadline) > DEADLINE_ABS_TOLERANCE:
        msg = "DDP smoke child deadline is not the parent absolute deadline"
        raise RuntimeError(msg)
    validate_pre_torchrun_interconnect_evidence(spec.pre_torchrun_interconnect)


def torchrun_command(spec_path: str, result_path: str) -> tuple[str, ...]:
    return (
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "meddies_pii.training.bioes.modal.pii350_ddp_smoke",
        "--child-spec",
        spec_path,
        "--child-result",
        result_path,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_event(root: Path, event: Mapping[str, Any]) -> None:
    append_event(root, event)


def _emit_stage_progress(rank: int, stage: str, **details: object) -> None:
    """Emit one rank-zero stdout event per bounded lifecycle stage, never a collective.

    Telemetry failure must not let rank zero strand rank one before its next
    collective. The parent still persists the child outcome and captured output.
    """
    if rank == 0:
        try:
            print(
                json.dumps({"event": "ddp_stage", "stage": stage, **details}, sort_keys=True),
                flush=True,
            )
        except (BrokenPipeError, OSError, TypeError, ValueError):
            return


@app.function(image=image, volumes=COMMON_MOUNTS, secrets=[secret], **CPU_PREFLIGHT_OPTIONS)
def preflight_ddp_assets() -> dict[str, Any]:
    """Create the exact baseline receipt from this app, never cross-app remote-call it.

    Returns:
        The baseline receipt mapping, extended with the path it was written to and its schema
        version.

    Raises:
        RuntimeError: if a receipt already exists at that path and differs from the one just
            computed. The receipt is immutable by contract, so a difference means the CPU
            verification is no longer reproducible and the run must not proceed on it.

    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification

    contract = matched_constant3_contract()
    result = _preflight_candidate_assets(
        "pii350",
        snapshot_download=snapshot_download,
        load_dataset=load_dataset,
        auto_model_for_token_classification=AutoModelForTokenClassification,
        auto_model_for_masked_lm=AutoModelForMaskedLM,
        contract=contract,
    )
    receipt = result.pop("receipt")
    path = _receipt_path(_asset_receipt_contract(contract))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        msg = "immutable DDP CPU verification receipt already differs"
        raise RuntimeError(msg)
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    artifacts.commit()
    return {
        **result,
        "receipt_path": str(path),
        "receipt_schema_version": PREFLIGHT_RECEIPT_SCHEMA_VERSION,
    }


@app.function(
    image=prewarm_image,
    volumes={"/cache": cache},
    secrets=[secret],
    **CPU_PREWARM_OPTIONS,
)
def prewarm_ddp_assets() -> dict[str, Any]:
    """CPU-only online hydration; the later offline receipt remains the proof.

    Returns:
        The hydration result mapping. This path is CPU-only and deliberately proves nothing on its
        own; the later offline receipt remains the proof.

    """
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    print(
        json.dumps({"event": "ddp_asset_prewarm_started", "candidate": "pii350"}),
        flush=True,
    )
    try:
        result = _prewarm_candidate_assets("pii350", snapshot_download=snapshot_download, load_dataset=load_dataset)
    finally:
        cache.commit()
    print(
        json.dumps({"event": "ddp_asset_prewarm_complete", **result}, sort_keys=True),
        flush=True,
    )
    return result


def _require_exact_runtime_packages() -> dict[str, str]:
    expected: dict[str, str] = {package.split("==", 1)[0]: package.split("==", 1)[1] for package in IMAGE_PACKAGES}
    observed = {name: version(name) for name in expected}
    if observed != expected:
        msg = f"DDP smoke runtime package attestation mismatch: {observed}"
        raise RuntimeError(msg)
    return observed


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _require_exact_a100_topology(torch: Any, local_rank: int) -> dict[str, Any]:  # ruff: ignore[any-type]
    """Modal's A100-40GB request is exact.

    NVIDIA exposes usable HBM below the marketing 40 GiB, so the attestation accepts only the documented 39-40 GiB band.

    Returns:
        The topology record: this process's local rank, the visible device count, the per-device
        inventory and the memory reading each device reported.

    Raises:
        RuntimeError: if CUDA is unavailable; if the visible device count is not exactly two; or if
            any visible device is not an A100-40GB by name and reported byte capacity. The check is
            exact rather than a minimum, because the run is priced against that topology.

    """
    if not torch.cuda.is_available():
        msg = "DDP smoke requires CUDA"
        raise RuntimeError(msg)
    if torch.cuda.device_count() != WORLD_SIZE:
        msg = "DDP smoke requires exactly two visible CUDA devices"
        raise RuntimeError(msg)
    torch.cuda.set_device(local_rank)
    inventory: list[dict[str, Any]] = []
    minimum_bytes = 39 * 1024**3
    maximum_bytes = 40 * 1024**3
    for index in range(WORLD_SIZE):
        properties = torch.cuda.get_device_properties(index)
        name = str(properties.name)
        total_memory = int(properties.total_memory)
        if "A100" not in name or not minimum_bytes <= total_memory <= maximum_bytes:
            msg = f"DDP smoke requires A100-40GB exactly; device={index} name={name!r} bytes={total_memory}"
            raise RuntimeError(msg)
        inventory.append({"local_index": index, "name": name, "total_memory_bytes": total_memory})
    return {
        "local_rank": local_rank,
        "visible_device_count": WORLD_SIZE,
        "inventory": inventory,
        "memory_attestation_bytes": {
            "minimum": minimum_bytes,
            "maximum": maximum_bytes,
        },
    }


TOPOLOGY_COMMAND = ("nvidia-smi", "topo", "-m")
UUID_INVENTORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,name,uuid,memory.total",
    "--format=csv,noheader,nounits",
)
NVLINK_STATUS_COMMAND = ("nvidia-smi", "nvlink", "--status")
MINIMUM_A100_40GB_MEMORY_MIB = 39 * 1024
MAXIMUM_A100_40GB_MEMORY_MIB = 40 * 1024
GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")


def _run_nvidia_command(
    command: tuple[str, ...],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    try:
        return (
            run(list(command), check=False, text=True, capture_output=True, timeout=10),
            None,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{type(error).__name__}: {error}"


class PreTorchrunInterconnectError(RuntimeError):
    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def _validate_uuid_devices(
    devices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(devices) != WORLD_SIZE:
        msg = "UUID inventory requires exactly two non-empty rows"
        raise RuntimeError(msg)
    normalized: list[dict[str, Any]] = []
    for device in devices:
        index = device.get("index")
        name = device.get("name")
        uuid = device.get("uuid")
        memory_total_mib = device.get("memory_total_mib")
        if not isinstance(index, int):
            msg = "UUID inventory GPU index must be an integer"
            raise RuntimeError(msg)
        if not isinstance(name, str) or "A100" not in name or "40GB" not in name:
            msg = "UUID inventory must prove two A100-40GB devices"
            raise RuntimeError(msg)
        if not isinstance(uuid, str) or not GPU_UUID_PATTERN.fullmatch(uuid):
            msg = "UUID inventory GPU UUID is malformed"
            raise RuntimeError(msg)
        if not isinstance(memory_total_mib, int) or not (
            MINIMUM_A100_40GB_MEMORY_MIB <= memory_total_mib <= MAXIMUM_A100_40GB_MEMORY_MIB
        ):
            msg = "UUID inventory A100-40GB memory must be within 39-40 GiB"
            raise RuntimeError(msg)
        normalized.append({
            "index": index,
            "name": name,
            "uuid": uuid,
            "memory_total_mib": memory_total_mib,
        })
    if {device["index"] for device in normalized} != set(range(WORLD_SIZE)):
        msg = "UUID inventory GPU indices must be exactly {0, 1}"
        raise RuntimeError(msg)
    if len({str(device["uuid"]) for device in normalized}) != WORLD_SIZE:
        msg = "UUID inventory GPU UUIDs must be unique"
        raise RuntimeError(msg)
    return sorted(normalized, key=lambda device: int(device["index"]))


def _parse_uuid_inventory(rows: Sequence[str]) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 4:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            msg = "UUID inventory rows must have exactly four comma-separated fields"
            raise RuntimeError(msg)
        try:
            index = int(fields[0])
            memory_total_mib = int(fields[3])
        except ValueError as error:
            msg = "UUID inventory GPU index and memory must be integers"
            raise RuntimeError(msg) from error
        devices.append({
            "index": index,
            "name": fields[1],
            "uuid": fields[2],
            "memory_total_mib": memory_total_mib,
        })
    return _validate_uuid_devices(devices)


# reason: validate pre coordinates parse uuid with raw rows; extra seams would fragment diagnostics.
def validate_pre_torchrun_interconnect_evidence(evidence: Mapping[str, Any]) -> None:  # ruff: ignore[complex-structure]
    if evidence.get("availability") != "validated":
        msg = "DDP smoke requires pre-torchrun interconnect evidence"
        raise RuntimeError(msg)
    if evidence.get("topology_command") != list(TOPOLOGY_COMMAND):
        msg = "DDP smoke topology command is invalid"
        raise RuntimeError(msg)
    if evidence.get("topology_returncode") != 0 and not isinstance(evidence.get("topology_unavailable_reason"), str):
        msg = "DDP smoke topology-unavailable reason is missing"
        raise RuntimeError(msg)
    if evidence.get("uuid_inventory_command") != list(UUID_INVENTORY_COMMAND):
        msg = "DDP smoke UUID inventory command is invalid"
        raise RuntimeError(msg)
    if evidence.get("uuid_inventory_returncode") != 0:
        msg = "DDP smoke UUID inventory must return zero"
        raise RuntimeError(msg)
    raw_rows = evidence.get("uuid_inventory_raw")
    if not isinstance(raw_rows, list) or not all(isinstance(row, str) and row for row in raw_rows):
        msg = "DDP smoke UUID inventory rows are missing"
        raise RuntimeError(msg)
    parsed_devices = _parse_uuid_inventory(raw_rows)
    recorded_devices = evidence.get("uuid_devices")
    if recorded_devices != parsed_devices:
        msg = "DDP smoke UUID devices do not match raw provenance"
        raise RuntimeError(msg)
    if evidence.get("nvlink_status_command") != list(NVLINK_STATUS_COMMAND):
        msg = "DDP smoke NVLink status command is invalid"
        raise RuntimeError(msg)
    if evidence.get("nvlink_status_returncode") != 0:
        msg = "DDP smoke NVLink status must return zero"
        raise RuntimeError(msg)
    if not isinstance(evidence.get("nvlink_status_raw"), str) or not evidence["nvlink_status_raw"].strip():
        msg = "DDP smoke NVLink status output is empty"
        raise RuntimeError(msg)


def capture_pre_torchrun_interconnect(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Capture raw topology/NVLink and exact GPU UUID inventory before torchrun.

    Returns:
        The interconnect evidence mapping — the raw NVLink status text and the exact GPU UUID
        inventory — captured before torchrun starts so it describes the machine the run will use.

    Raises:
        PreTorchrunInterconnectError: if the UUID inventory command fails to execute or exits
            non-zero; if the NVLink status command fails to execute, exits non-zero, or returns
            empty output; or if the captured UUID inventory does not parse. Note the type: this
            function raises its own error class, not `RuntimeError`, so a caller catching only
            `RuntimeError` will not see these.

    """
    topology, topology_error = _run_nvidia_command(TOPOLOGY_COMMAND, run=run)
    if topology is None:
        topology_reason = f"nvidia-smi topo -m failed before torchrun: {topology_error}"
        topology_returncode: int | None = None
        topology_stdout = ""
        topology_stderr = ""
    else:
        topology_reason = "" if topology.returncode == 0 else f"nvidia-smi topo -m returned {topology.returncode}"
        topology_returncode = int(topology.returncode)
        topology_stdout = str(topology.stdout or "")
        topology_stderr = str(topology.stderr or "")

    evidence: dict[str, Any] = {
        "availability": "validated",
        "topology_command": list(TOPOLOGY_COMMAND),
        "topology_returncode": topology_returncode,
        "topology_stdout": topology_stdout,
        "topology_stderr": topology_stderr,
        "topology_unavailable_reason": topology_reason or None,
        "uuid_inventory_command": list(UUID_INVENTORY_COMMAND),
        "nvlink_status_command": list(NVLINK_STATUS_COMMAND),
    }
    uuid_inventory, uuid_error = _run_nvidia_command(UUID_INVENTORY_COMMAND, run=run)
    if uuid_inventory is None:
        evidence.update({
            "uuid_inventory_returncode": None,
            "uuid_inventory_raw": [],
            "uuid_inventory_stderr": uuid_error or "",
        })
        msg = f"UUID inventory failed before torchrun: {uuid_error}"
        raise PreTorchrunInterconnectError(msg, evidence=evidence)
    uuid_rows = [line for line in str(uuid_inventory.stdout or "").splitlines() if line.strip()]
    evidence.update({
        "uuid_inventory_returncode": int(uuid_inventory.returncode),
        "uuid_inventory_raw": uuid_rows,
        "uuid_inventory_stderr": str(uuid_inventory.stderr or ""),
    })
    if uuid_inventory.returncode != 0:
        msg = f"UUID inventory returned {uuid_inventory.returncode}: {uuid_inventory.stderr}"
        raise PreTorchrunInterconnectError(
            msg,
            evidence=evidence,
        )
    try:
        evidence["uuid_devices"] = _parse_uuid_inventory(uuid_rows)
    except RuntimeError as error:
        msg = f"UUID inventory evidence is invalid: {error}"
        raise PreTorchrunInterconnectError(msg, evidence=evidence) from error

    nvlink_status, nvlink_error = _run_nvidia_command(NVLINK_STATUS_COMMAND, run=run)
    if nvlink_status is None:
        evidence.update({
            "nvlink_status_returncode": None,
            "nvlink_status_raw": "",
            "nvlink_status_stderr": nvlink_error or "",
        })
        msg = f"NVLink status failed before torchrun: {nvlink_error}"
        raise PreTorchrunInterconnectError(msg, evidence=evidence)
    evidence.update({
        "nvlink_status_returncode": int(nvlink_status.returncode),
        "nvlink_status_raw": str(nvlink_status.stdout or ""),
        "nvlink_status_stderr": str(nvlink_status.stderr or ""),
    })
    if nvlink_status.returncode != 0:
        msg = f"NVLink status returned {nvlink_status.returncode}: {nvlink_status.stderr}"
        raise PreTorchrunInterconnectError(
            msg,
            evidence=evidence,
        )
    if not evidence["nvlink_status_raw"].strip():
        msg = "NVLink status output is empty"
        raise PreTorchrunInterconnectError(msg, evidence=evidence)
    validate_pre_torchrun_interconnect_evidence(evidence)
    return evidence


def _active_labels(labels: Tensor) -> Tensor:
    return (labels != LABEL_IGNORE_INDEX).sum()


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _require_collective_ready(torch: Any, distributed: Any, *, local_rank: int, rank: int) -> dict[str, Any]:  # ruff: ignore[any-type]
    """Prove bidirectional CUDA peer access and one NCCL reduction before model load.

    Both ranks still gather a false record and reject the same readiness state; no peer should enter a collective while
    another throws locally.

    Returns:
        The readiness record produced by `validate_collective_readiness` from the observed peer
        access matrix and the all-reduce result.

    Raises:
        RuntimeError: if the ranks disagree on the NCCL all-reduce output, which means the
            collective is not usable and no model should be loaded against it.

    """
    peer_rank = WORLD_SIZE - 1 - rank
    peer_access_error: str | None = None
    try:
        can_access_peer = bool(torch.cuda.can_device_access_peer(local_rank, peer_rank))
    # reason: CUDA peer-access probing is diagnostic; driver-specific failures are gathered as evidence, not fatal here.
    except Exception as error:  # ruff: ignore[blind-except]
        can_access_peer = False
        peer_access_error = repr(error)
    peer_access_records = _gather_objects(
        distributed,
        {
            "rank": rank,
            "peer_rank": peer_rank,
            "can_access_peer": can_access_peer,
            "peer_access_error": peer_access_error,
        },
    )
    readiness_probe = torch.tensor(rank + 1, device=local_rank, dtype=torch.int64)
    distributed.all_reduce(readiness_probe, op=distributed.ReduceOp.SUM)
    probe_records = _gather_objects(
        distributed,
        {"rank": rank, "observed_all_reduce_sum": int(readiness_probe.item())},
    )
    observed_values = {int(record["observed_all_reduce_sum"]) for record in probe_records}
    if len(observed_values) != 1:
        msg = "DDP readiness probe ranks disagree on NCCL all-reduce output"
        raise RuntimeError(msg)
    return validate_collective_readiness(
        peer_access_records=peer_access_records,
        observed_all_reduce_sum=observed_values.pop(),
    )


def _trainable_named_parameters(tagger: TrainableModule) -> list[tuple[str, Any]]:
    """Return the exact optimizer/DDP parameter order, rejecting silent aliases.

    Returns:
        The trainable parameters as `(name, tensor)` pairs in the exact order the optimizer and DDP
        will see them.

    Raises:
        RuntimeError: if the inventory is invalid — the same tensor reachable under two names would
            make the order ambiguous, which is what this refuses.

    """
    module = getattr(tagger, "module", tagger)
    pairs = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    names = [name for name, _parameter in pairs]
    if not pairs or len(names) != len(set(names)):
        msg = "DDP smoke trainable parameter inventory is invalid"
        raise RuntimeError(msg)
    return pairs


def _parameter_metadata(name: str, parameter: Tensor, *, stage: str) -> dict[str, Any]:
    """Verify the reduced tensor before deciding a bucket order.

    Parameter dtype alone is not enough to prove that two ranks will issue identical NCCL collectives.

    CUDA index is rank-local by design (rank zero owns cuda:0, rank one cuda:1). The device *type* is the collective
    compatibility contract.

    Returns:
        The per-parameter record: name, shape, `requires_grad`, dtype and device, which is what the
        cross-rank comparison is performed on.

    """
    tensor = parameter.grad if stage == "gradients" else parameter
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


# reason: require matching orders ordered before reference; helper seams would fragment diagnostics.
def _require_matching_parameter_inventory(inventory_records: Sequence[Mapping[str, Any]], *, stage: str) -> list[str]:  # ruff: ignore[complex-structure]
    """Fail every rank before tensor collectives if names/order/grad presence differ.

    Returns:
        The agreed trainable parameter names, in rank order.

    Raises:
        RuntimeError: if any rank contributed no record; if the inventory is empty; if names,
            order, shapes or dtypes differ across ranks at this stage; if a trainable name appears
            twice; if a trainable gradient is missing at this stage; if any metadata entry or name
            is malformed; or if a non-trainable parameter appears in it. Every one fails ALL ranks
            before any tensor collective runs, which is the point — a mismatched inventory would
            otherwise deadlock or silently reduce the wrong tensors.

    """
    if len(inventory_records) != WORLD_SIZE:
        msg = "DDP parameter inventory lacks a rank record"
        raise RuntimeError(msg)
    ordered = sorted(inventory_records, key=lambda record: int(record["rank"]))
    reference = ordered[0].get("parameters")
    if not isinstance(reference, list) or not reference:
        msg = "DDP parameter inventory is empty"
        raise RuntimeError(msg)
    if any(record.get("parameters") != reference for record in ordered[1:]):
        msg = f"DDP parameter name/order/shape/dtype inventory differs across ranks at {stage}"
        raise RuntimeError(msg)
    names: list[str] = []
    missing: list[str] = []
    for metadata in reference:
        if not isinstance(metadata, Mapping):
            msg = "DDP parameter inventory contains invalid metadata"
            raise RuntimeError(msg)
        name = metadata.get("name")
        if not isinstance(name, str) or not name:
            msg = "DDP parameter inventory name is invalid"
            raise RuntimeError(msg)
        if metadata.get("requires_grad") is not True:
            msg = "DDP parameter inventory includes a non-trainable parameter"
            raise RuntimeError(msg)
        names.append(name)
        if stage == "gradients" and metadata.get("tensor_present") is not True:
            missing.append(name)
    if len(names) != len(set(names)):
        msg = "DDP parameter inventory has duplicate trainable names"
        raise RuntimeError(msg)
    if missing:
        msg = f"DDP trainable gradients are missing at {stage}: {', '.join(missing)}"
        raise RuntimeError(msg)
    return names


# reason: distributed arrives as an injected module so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _parameter_stage_diagnostics(
    distributed: Any,  # ruff: ignore[any-type]
    tagger: TrainableModule,
    *,
    rank: int,
    stage: str,
) -> dict[str, Any]:
    """Compare every named trainable tensor to rank zero without changing it.

    This makes the first failed stage observable: post-wrap parameters,
    pre-reduction gradients, post-reduction gradients, or post-update params.
    A broadcast is used only on a detached clone, never to repair a replica.

    guarded above; narrows the tensor type for runtime.

    Returns:
        The cross-rank diagnostic record for this stage, tagged
        `trainable_tensor_cross_rank_diagnostic_v1`. It compares only; no tensor is modified.

    Raises:
        ValueError: if `stage` is neither `parameters` nor `gradients`.
        RuntimeError: if the diagnostic order differs across ranks, or if a named tensor
            disappeared while the stage was being read.

    """
    if stage not in {"parameters", "gradients"}:
        msg = "DDP diagnostic stage must be parameters or gradients"
        raise ValueError(msg)
    pairs = _trainable_named_parameters(tagger)
    inventory_records = _gather_objects(
        distributed,
        {
            "rank": rank,
            "parameters": [_parameter_metadata(name, parameter, stage=stage) for name, parameter in pairs],
        },
    )
    expected_names = _require_matching_parameter_inventory(inventory_records, stage=stage)
    local_differences: list[dict[str, Any]] = []
    for name, parameter in pairs:
        tensor = parameter.grad if stage == "gradients" else parameter
        if tensor is None:
            msg = f"DDP tensor disappeared during {stage}: {name}"
            raise RuntimeError(msg)
        reference = tensor.detach().clone()
        distributed.broadcast(reference, src=0)
        difference = float((tensor.detach().float() - reference.float()).abs().max().item())
        local_differences.append({
            "name": name,
            "max_abs_difference_from_rank0": difference,
            "max_abs_value": float(tensor.detach().float().abs().max().item()),
            "numel": int(tensor.numel()),
        })
    difference_records = _gather_objects(
        distributed,
        {"rank": rank, "differences": local_differences},
    )
    by_rank = sorted(difference_records, key=lambda record: int(record["rank"]))
    if any([item.get("name") for item in record.get("differences", [])] != expected_names for record in by_rank):
        msg = "DDP parameter diagnostic order differs across ranks"
        raise RuntimeError(msg)
    differences: list[dict[str, Any]] = []
    for index, name in enumerate(expected_names):
        records = [record["differences"][index] for record in by_rank]
        maximum = max(float(record["max_abs_difference_from_rank0"]) for record in records)
        differences.append({
            "name": name,
            "numel": int(records[0]["numel"]),
            "max_abs_difference_from_rank0_by_rank": [
                float(record["max_abs_difference_from_rank0"]) for record in records
            ],
            "max_abs_value_by_rank": [float(record["max_abs_value"]) for record in records],
            "max_abs_difference": maximum,
        })
    drifting = [item for item in differences if item["max_abs_difference"] > 0.0]
    return {
        "kind": "trainable_tensor_cross_rank_diagnostic_v1",
        "stage": stage,
        "parameter_count": len(expected_names),
        "max_abs_difference": max((float(item["max_abs_difference"]) for item in differences), default=0.0),
        "drifting_parameters": drifting,
        "parameters": differences,
        "rank_inventory": sorted(inventory_records, key=lambda record: int(record["rank"])),
    }


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _explicit_mean_trainable_gradients(
    torch: Any,  # ruff: ignore[any-type]
    distributed: Any,  # ruff: ignore[any-type]
    tagger: TrainableModule,
    *,
    rank: int,
) -> list[dict[str, Any]]:
    """Mean every local trainable gradient after a no-sync packed backward.

    Stock DDP cannot safely reduce an encoder re-entered once per packed segment
    under reentrant checkpointing. This is mathematically the same world-size
    mean DDP would apply, but its communication is explicit and occurs only
    after the entire variable-segment graph has completed.

    Ordered dtype/device buckets make the collective sequence identical on both ranks while keeping the smoke's throughput
    measurement honest.

    Returns:
        The per-bucket reduction records for every local trainable gradient.

    Raises:
        RuntimeError: if the trainable gradient inventory changed between capture and reduction, or
            if a named gradient disappeared before it could be reduced. Both mean the reduction
            would not be over the tensor set that was agreed, so it fails rather than proceeds.

    """
    pairs = _trainable_named_parameters(tagger)
    names = _require_matching_parameter_inventory(
        _gather_objects(
            distributed,
            {
                "rank": rank,
                "parameters": [_parameter_metadata(name, parameter, stage="gradients") for name, parameter in pairs],
            },
        ),
        stage="gradients",
    )
    if names != [name for name, _parameter in pairs]:
        msg = "DDP trainable gradient inventory changed before reduction"
        raise RuntimeError(msg)
    grouped: dict[tuple[str, str], list[tuple[str, Any]]] = {}
    for name, parameter in pairs:
        if parameter.grad is None:
            msg = f"DDP trainable gradient disappeared before reduction: {name}"
            raise RuntimeError(msg)
        key = (str(parameter.grad.device.type), str(parameter.grad.dtype))
        grouped.setdefault(key, []).append((name, parameter.grad))
    buckets: list[dict[str, Any]] = []
    for (device_type, dtype), gradients in sorted(grouped.items()):
        flat = torch.cat([gradient.reshape(-1) for _name, gradient in gradients])
        distributed.all_reduce(flat, op=distributed.ReduceOp.SUM)
        flat.div_(WORLD_SIZE)
        offset = 0
        for _name, gradient in gradients:
            next_offset = offset + int(gradient.numel())
            gradient.copy_(flat[offset:next_offset].reshape_as(gradient))
            offset = next_offset
        buckets.append({
            "device_type": device_type,
            "dtype": dtype,
            "parameter_count": len(gradients),
            "numel": int(flat.numel()),
            "parameter_names": [name for name, _gradient in gradients],
        })
    return buckets


def _require_zero_parameter_drift(diagnostics: Mapping[str, Any], *, phase: str, step: int) -> float:
    maximum = float(diagnostics["max_abs_difference"])
    if maximum > BF16_PARAMETER_DRIFT_TOLERANCE:
        names = [str(item["name"]) for item in diagnostics.get("drifting_parameters", []) if isinstance(item, Mapping)]
        msg = f"DDP {phase} divergence at step {step}: {maximum} > {BF16_PARAMETER_DRIFT_TOLERANCE}; parameters={names}"
        raise RuntimeError(
            msg,
        )
    return maximum


def _persist_step_one_diagnostics(spec: ChildSpec, *, rank: int, phase: str, diagnostics: Mapping[str, Any]) -> None:
    """Write the complete two-rank evidence before a later gate can abort torchrun.

    These are the only child-created nested artifacts. Parent owns the root, but a fresh run has no diagnostics directory
    when rank zero reaches the post-wrap gate.

    """
    if rank != 0:
        return
    path = Path(spec.artifact_dir) / "diagnostics" / f"step-001-{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "config_digest": spec.contract["config_digest"],
            "step": 1,
            "phase": phase,
            "gradient_reduction": DDP_GRADIENT_REDUCTION,
            "diagnostics": dict(diagnostics),
        },
    )


def _step_one_diagnostic_paths() -> list[str]:
    return [f"diagnostics/step-001-{phase}.json" for phase in STEP_ONE_DIAGNOSTIC_PHASES]


def _nvml_sample(local_rank: int) -> dict[str, int | None]:
    try:
        pynvml = import_module("pynvml")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "device_vram_bytes": int(memory.total),
            "sampled_device_vram_used_bytes": int(memory.used),
        }
    # reason: NVML sampling is best-effort telemetry and must not abort the paid DDP correctness probe.
    except Exception:  # ruff: ignore[blind-except]
        return {"device_vram_bytes": None, "sampled_device_vram_used_bytes": None}


# reason: torch and distributed arrive as injected modules so a test can pass a fake in place of the real one.
# reason: Measured: ModuleType rejects the SimpleNamespace fake, and a Protocol rejects the real module.
def _gather_objects(distributed: Any, value: Mapping[str, Any]) -> list[dict[str, Any]]:  # ruff: ignore[any-type]
    gathered: list[object] = [None for _ in range(WORLD_SIZE)]
    distributed.all_gather_object(gathered, dict(value))
    if not all(isinstance(item, dict) for item in gathered):
        msg = "DDP telemetry gather returned an invalid rank record"
        raise RuntimeError(msg)
    return [dict(item) for item in gathered if isinstance(item, dict)]


def _rank_batch(rows: Sequence[Mapping[str, Any]], *, step: int, rank: int) -> Sequence[Mapping[str, Any]]:
    expected_start, expected_end = rank_window(global_step=step, rank=rank)
    global_start = (step - 1) * WORLD_SIZE * LOCAL_BATCH_SIZE
    if len(rows) != WORLD_SIZE * LOCAL_BATCH_SIZE:
        msg = "DDP smoke lacks one full global packed batch"
        raise RuntimeError(msg)
    local_rows = rows[expected_start - global_start : expected_end - global_start]
    if len(local_rows) != LOCAL_BATCH_SIZE:
        msg = "DDP smoke rank batch is short"
        raise RuntimeError(msg)
    return local_rows


def _require_document_isolation(tagger: Module, paths: Sequence[str], *, device: str) -> dict[str, Any]:
    """Exercise the packed attention boundary before timing any DDP update.

    Returns:
        The contamination proof as a mapping, recorded before any DDP update is timed.

    Raises:
        RuntimeError: if the packed unit holds only one document, so isolation cannot be tested; or
            if the proof shows attention crossing a document boundary.

    """
    from meddies_pii.training.bioes.trainers.contamination import (
        run_packed_attention_contamination_probe,
    )
    from meddies_pii.training.bioes.trainers.packing import (
        PackedRowRange,
        PackedTrainingUnit,
    )

    parquet = import_module("pyarrow.parquet")
    candidate: Mapping[str, Any] | None = None
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                ranges = row.get("row_ranges")
                if isinstance(ranges, list) and len(ranges) >= MIN_PACKED_SEGMENTS:
                    candidate = row
                    break
            if candidate is not None:
                break
        if candidate is not None:
            break
    if candidate is None:
        msg = "DDP smoke packed isolation proof requires a multi-document unit"
        raise RuntimeError(msg)
    ranges = tuple(
        PackedRowRange(str(item["uid"]), int(item["start"]), int(item["end"])) for item in candidate["row_ranges"]
    )
    ids = list(candidate["input_ids"])
    altered = list(ids)
    altered[ranges[0].start] = (altered[ranges[0].start] + 1) % 100
    common = (
        tuple(candidate["labels"]),
        tuple(candidate["seq_lengths"]),
        tuple(candidate["position_ids"]),
        tuple(candidate["row_uids"]),
        ranges,
        int(candidate["real_token_count"]),
        int(candidate["padded_token_count"]),
        int(candidate["boundary_token_count"]),
    )
    clean = PackedTrainingUnit(tuple(ids), *common)
    mutated = PackedTrainingUnit(tuple(altered), *common)
    proof = run_packed_attention_contamination_probe(
        tagger,
        clean_unit=clean,
        mutated_unit=mutated,
        target_row_uid=ranges[1].uid,
        device=device,
        atol=0.0,
        rtol=0.0,
    )
    # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
    # reason: equality, so any tolerance here would let a real drift or contamination pass the gate.
    if not proof.passed or proof.max_abs_diff != 0.0:  # ruff: ignore[float-equality-comparison]
        msg = f"DDP smoke packed isolation proof failed: {proof}"
        raise RuntimeError(msg)
    return proof.to_dict()


# reason: run worker owns validate and build tagger together; splitting would fragment diagnostics.
def _run_worker(spec: ChildSpec) -> dict[str, Any]:  # ruff: ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
    """Re-enter the same Unsloth checkpointed encoder for each document segment.

    Its count is intentionally data/rank dependent, so do not ask stock DDP to reduce reentrant hooks. DDP still broadcasts
    the wrapped replica at construction; the complete gradient set is explicitly averaged below.

    Returns:
        The worker result mapping for this rank.

    Raises:
        RuntimeError: on any of these, each of which stops the paid run rather than degrading it —
            the absolute training deadline being reached before worker setup, before expensive
            setup, or before ten updates; the 300-second persistence reserve not being retained;
            torchrun supplying an invalid two-rank local topology; DDP initialising with the wrong
            world size or rank; the tagger losing packed segment isolation; the isolation proof not
            being gathered from every rank or the ranks disagreeing on it; the packed stream ending
            before ten global batches; the ranks disagreeing on the ordered global packed-unit
            digest; a global batch carrying no active labels; a local or weighted loss that is
            absent or non-finite; and step-one parameter diagnostics not being captured.

    """
    if training_deadline_reached(
        absolute_deadline_monotonic=spec.absolute_training_deadline_monotonic,
        now_monotonic=time.monotonic(),
    ):
        msg = "DDP smoke absolute training deadline reached before worker setup"
        raise RuntimeError(msg)
    import torch
    from torch import distributed
    from torch.nn.parallel import DistributedDataParallel

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _build_tagger,
        _iter_packed_rows,
        _packed_batch,
        _validate_runtime_label_vocabulary,
    )

    validate_child_spec(spec)
    if training_deadline_reached(
        absolute_deadline_monotonic=spec.absolute_training_deadline_monotonic,
        now_monotonic=time.monotonic(),
    ):
        msg = "DDP smoke absolute training deadline reached before expensive setup"
        raise RuntimeError(msg)
    if HARD_TIMEOUT_SECONDS - TRAINING_DEADLINE_SECONDS != PERSISTENCE_RESERVE_SECONDS:
        msg = "DDP smoke must retain the full 300-second persistence reserve"
        raise RuntimeError(msg)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    if local_rank not in range(WORLD_SIZE) or rank not in range(WORLD_SIZE):
        msg = "torchrun did not provide a valid two-rank local topology"
        raise RuntimeError(msg)
    topology = _require_exact_a100_topology(torch, local_rank)
    _emit_stage_progress(rank, "rank_setup", topology=topology)
    # reason: torch declares the torch.distributed members behind an availability gate, so no static
    # reason: reader can prove they are present; this path runs only inside the multi-rank container.
    distributed.init_process_group(backend="nccl")  # ty: ignore[possibly-missing-attribute]
    try:
        # reason: torch declares the torch.distributed members behind an availability gate, so no static
        # reason: reader can prove they are present; this path runs only inside the multi-rank container.
        if (
            distributed.get_world_size() != WORLD_SIZE  # ty: ignore[possibly-missing-attribute]
            or distributed.get_rank() != rank  # ty: ignore[possibly-missing-attribute]
        ):
            msg = "DDP initialized with the wrong world size or rank"
            raise RuntimeError(msg)
        collective_readiness = _require_collective_ready(torch, distributed, local_rank=local_rank, rank=rank)
        _emit_stage_progress(rank, "collective_ready", **collective_readiness)
        packages = _require_exact_runtime_packages()
        _validate_runtime_label_vocabulary(spec.matched_constant3_contract)
        torch.manual_seed(int(spec.contract["seed"]))
        tagger, _tokenizer, runtime_attestation = _build_tagger(
            "pii350",
            contract=spec.matched_constant3_contract,
            expected_encoder_checkpoint_attestation=spec.expected_encoder_checkpoint_attestation,
        )
        if not tagger.packed_segment_isolation:
            msg = "DDP tagger lost packed segment isolation"
            raise RuntimeError(msg)
        _emit_stage_progress(rank, "model_ready", runtime_attestation=runtime_attestation)
        local_document_isolation_proof = _require_document_isolation(tagger, spec.shard_paths, device=f"cuda:{local_rank}")
        proof_records = _gather_objects(
            distributed,
            {"rank": rank, "document_isolation_proof": local_document_isolation_proof},
        )
        proofs = [record["document_isolation_proof"] for record in proof_records]
        if not all(isinstance(proof, dict) for proof in proofs):
            msg = "DDP smoke document-isolation proof was not gathered from every rank"
            raise RuntimeError(msg)
        document_isolation_proof = dict(proofs[0])
        if any(dict(proof) != document_isolation_proof for proof in proofs if isinstance(proof, dict)):
            msg = "DDP smoke ranks disagree on document-isolation proof"
            raise RuntimeError(msg)
        _emit_stage_progress(
            rank,
            "document_isolation_proven",
            document_isolation_proof=document_isolation_proof,
        )
        tagger = DistributedDataParallel(
            tagger,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        initial_parameter_diagnostics = _parameter_stage_diagnostics(
            distributed,
            tagger,
            rank=rank,
            stage="parameters",
        )
        _persist_step_one_diagnostics(
            spec,
            rank=rank,
            phase="post_ddp_wrap_parameters",
            diagnostics=initial_parameter_diagnostics,
        )
        _require_zero_parameter_drift(initial_parameter_diagnostics, phase="post-DDP-wrap parameter", step=0)
        dropout_seed = int(spec.contract["seed"]) + rank
        torch.manual_seed(dropout_seed)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in tagger.parameters() if parameter.requires_grad],
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            fused=False,
        )
        _emit_stage_progress(
            rank,
            "ddp_ready",
            packages=packages,
            pre_torchrun_interconnect=spec.pre_torchrun_interconnect,
            dropout_seed_policy="seed_plus_rank_after_ddp_initialization",
        )
        iterator = _iter_packed_rows(spec.shard_paths)
        durations: list[float] = []
        aggregate_tokens_per_second: list[float] = []
        per_rank_tokens: list[int] = [0 for _ in range(WORLD_SIZE)]
        per_rank_peak: list[int] = [0 for _ in range(WORLD_SIZE)]
        per_rank_sampled: list[int | None] = [None for _ in range(WORLD_SIZE)]
        timed_training_started = time.monotonic()
        post_update_parameter_diagnostics: dict[str, Any] | None = None
        for step in range(1, OPTIMIZER_STEPS + 1):
            if training_deadline_reached(
                absolute_deadline_monotonic=spec.absolute_training_deadline_monotonic,
                now_monotonic=time.monotonic(),
            ):
                msg = "DDP smoke training deadline reached before ten updates"
                raise RuntimeError(msg)
            rows = [next(iterator, None) for _ in range(WORLD_SIZE * LOCAL_BATCH_SIZE)]
            if any(row is None for row in rows):
                msg = "DDP smoke packed stream ended before ten global batches"
                raise RuntimeError(msg)
            global_rows = [row for row in rows if row is not None]
            local_rows = _rank_batch(global_rows, step=step, rank=rank)
            global_identities = packed_unit_identities(global_rows)
            local_identities = packed_unit_identities(local_rows)
            global_digest = ordered_identity_digest(global_identities)
            identity_records = _gather_objects(
                distributed,
                {
                    "rank": rank,
                    "global_digest": global_digest,
                    "local_identities": local_identities,
                },
            )
            if {str(record["global_digest"]) for record in identity_records} != {global_digest}:
                msg = "DDP ranks disagree on the ordered global packed-unit digest"
                raise RuntimeError(msg)
            ordered_identity_records = sorted(identity_records, key=lambda item: int(item["rank"]))
            identity_coverage_digest = validate_rank_identity_coverage(
                global_identities=global_identities,
                rank_identities=[list(record["local_identities"]) for record in ordered_identity_records],
            )
            windows = _gather_objects(
                distributed,
                {"rank": rank, "window": rank_window(global_step=step, rank=rank)},
            )
            ordered_windows = [tuple(record["window"]) for record in sorted(windows, key=lambda item: int(item["rank"]))]
            coverage = validate_rank_windows(global_step=step, windows=ordered_windows)
            batch, local_real_tokens = _packed_batch(local_rows, f"cuda:{local_rank}")
            local_active = _active_labels(batch["labels"])
            global_active = local_active.clone()
            # reason: torch declares the torch.distributed members behind an availability gate, so no static
            # reason: reader can prove they are present; this path runs only inside the multi-rank container.
            distributed.all_reduce(global_active, op=distributed.ReduceOp.SUM)  # ty: ignore[possibly-missing-attribute]
            if int(global_active.item()) <= 0:
                msg = "DDP smoke global batch has no active labels"
                raise RuntimeError(msg)
            global_real_tokens = torch.tensor(local_real_tokens, device=local_rank, dtype=torch.int64)
            # reason: torch declares the torch.distributed members behind an availability gate, so no static
            # reason: reader can prove they are present; this path runs only inside the multi-rank container.
            distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
                global_real_tokens,
                op=distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
            )
            torch.cuda.synchronize(local_rank)
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with tagger.no_sync():
                output = tagger(**batch)
                loss = output["loss"]
                if loss is None or not torch.isfinite(loss):
                    msg = "DDP smoke produced an absent or non-finite local loss"
                    raise RuntimeError(msg)
                weighted_loss = scale_local_token_mean_loss(loss, local_active=local_active, global_active=global_active)
                if not torch.isfinite(weighted_loss):
                    msg = "DDP smoke produced a non-finite weighted loss"
                    raise RuntimeError(msg)
                torch.autograd.backward(weighted_loss)
            if step == 1:
                pre_reduction_gradient_diagnostics = _parameter_stage_diagnostics(
                    distributed,
                    tagger,
                    rank=rank,
                    stage="gradients",
                )
                _persist_step_one_diagnostics(
                    spec,
                    rank=rank,
                    phase="pre_explicit_reduction_gradients",
                    diagnostics=pre_reduction_gradient_diagnostics,
                )
            gradient_reduction_buckets = _explicit_mean_trainable_gradients(torch, distributed, tagger, rank=rank)
            if step == 1:
                post_reduction_gradient_diagnostics = _parameter_stage_diagnostics(
                    distributed,
                    tagger,
                    rank=rank,
                    stage="gradients",
                )
                post_reduction_gradient_diagnostics["reduction_buckets"] = gradient_reduction_buckets
                _persist_step_one_diagnostics(
                    spec,
                    rank=rank,
                    phase="post_explicit_reduction_gradients",
                    diagnostics=post_reduction_gradient_diagnostics,
                )
                _require_zero_parameter_drift(
                    post_reduction_gradient_diagnostics,
                    phase="post-explicit-reduction gradient",
                    step=step,
                )
            optimizer.step()
            if step == 1:
                post_update_parameter_diagnostics = _parameter_stage_diagnostics(
                    distributed,
                    tagger,
                    rank=rank,
                    stage="parameters",
                )
                _persist_step_one_diagnostics(
                    spec,
                    rank=rank,
                    phase="post_optimizer_parameters",
                    diagnostics=post_update_parameter_diagnostics,
                )
                _require_zero_parameter_drift(
                    post_update_parameter_diagnostics,
                    phase="post-optimizer parameter",
                    step=step,
                )
            torch.cuda.synchronize(local_rank)
            duration = time.monotonic() - before
            durations.append(duration)
            duration_tensor = torch.tensor(duration, device=local_rank, dtype=torch.float64)
            # reason: torch declares the torch.distributed members behind an availability gate, so no static
            # reason: reader can prove they are present; this path runs only inside the multi-rank container.
            distributed.all_reduce(duration_tensor, op=distributed.ReduceOp.MAX)  # ty: ignore[possibly-missing-attribute]
            synchronized_duration = float(duration_tensor.item())
            rank_durations = _gather_objects(distributed, {"rank": rank, "step_wall_seconds": duration})
            aggregate_rate = aggregate_real_tokens_per_second(
                global_real_tokens=int(global_real_tokens.item()),
                rank_step_seconds=[float(record["step_wall_seconds"]) for record in rank_durations],
            )
            aggregate_tokens_per_second.append(aggregate_rate)
            sample = _nvml_sample(local_rank)
            telemetry = {
                "rank": rank,
                "local_real_tokens": local_real_tokens,
                "local_active_labels": int(local_active.item()),
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(local_rank)),
                **sample,
            }
            ranks = _gather_objects(distributed, telemetry)
            for record in ranks:
                record_rank = int(record["rank"])
                per_rank_tokens[record_rank] += int(record["local_real_tokens"])
                per_rank_peak[record_rank] = max(per_rank_peak[record_rank], int(record["peak_allocated_vram_bytes"]))
                sampled = record["sampled_device_vram_used_bytes"]
                if isinstance(sampled, int):
                    prior = per_rank_sampled[record_rank] or 0
                    per_rank_sampled[record_rank] = max(prior, sampled)
            if step == 1:
                if post_update_parameter_diagnostics is None:
                    msg = "DDP smoke step-one parameter diagnostics were not captured"
                    raise RuntimeError(msg)
                drift = _require_zero_parameter_drift(
                    post_update_parameter_diagnostics,
                    phase="post-optimizer parameter",
                    step=step,
                )
            elif step == OPTIMIZER_STEPS:
                drift = _require_zero_parameter_drift(
                    _parameter_stage_diagnostics(distributed, tagger, rank=rank, stage="parameters"),
                    phase="post-optimizer parameter",
                    step=step,
                )
            else:
                drift = None
            _emit_stage_progress(
                rank,
                "optimizer_step",
                step=step,
                rank_window_coverage=coverage,
                ordered_global_packed_unit_sha256=identity_coverage_digest,
                global_real_tokens=int(global_real_tokens.item()),
                aggregate_real_tokens_per_second=aggregate_rate,
                step_wall_seconds=synchronized_duration,
                per_rank_step_wall_seconds=rank_durations,
                weighted_loss_rank0=float(weighted_loss.detach().float().cpu()),
                rank_telemetry=ranks,
                max_trainable_parameter_drift=drift,
                gradient_reduction=DDP_GRADIENT_REDUCTION,
                gradient_reduction_buckets=(gradient_reduction_buckets if step == 1 else None),
            )
        median_warm = float(median(aggregate_tokens_per_second[2:]))
        outer_elapsed_through_child_result = time.monotonic() - spec.outer_modal_started_monotonic
        result = {
            "status": "ok",
            "config_digest": spec.contract["config_digest"],
            "optimizer_steps": OPTIMIZER_STEPS,
            "evaluation": False,
            "checkpoint": False,
            "topology": topology,
            "pre_torchrun_interconnect": spec.pre_torchrun_interconnect,
            "collective_readiness": collective_readiness,
            "packages": packages,
            "runtime_attestation": runtime_attestation,
            "packed_segment_isolation": True,
            "document_isolation_proof": document_isolation_proof,
            "step_one_diagnostic_paths": _step_one_diagnostic_paths(),
            "dropout_seed_policy": "seed_plus_rank_after_ddp_initialization",
            "per_rank_real_tokens": per_rank_tokens,
            "aggregate_real_tokens": sum(per_rank_tokens),
            "step_wall_seconds": durations,
            "median_aggregate_real_tokens_per_second_steps_3_to_10": median_warm,
            "per_rank_peak_allocated_vram_bytes": per_rank_peak,
            "per_rank_sampled_device_vram_used_bytes": per_rank_sampled,
            "training_elapsed_seconds": time.monotonic() - timed_training_started,
            "outer_modal_elapsed_seconds_through_child_result": outer_elapsed_through_child_result,
            "estimated_all_in_cost_usd_through_child_result": (
                outer_elapsed_through_child_result * ALL_IN_RATE_USD_PER_SECOND
            ),
            "all_in_rate_usd_per_second": ALL_IN_RATE_USD_PER_SECOND,
            "cost_scope": (
                "outer Modal function entry through child result write; excludes "
                "parent post-child result read and final Volume commit time"
            ),
            "artifact_root": spec.artifact_dir,
        }
        # reason: torch declares the torch.distributed members behind an availability gate, so no static
        # reason: reader can prove they are present; this path runs only inside the multi-rank container.
        distributed.barrier()  # ty: ignore[possibly-missing-attribute]
        return result
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
                    "step_one_diagnostic_paths": _step_one_diagnostic_paths(),
                },
            )
        raise
    if os.environ.get("RANK", "0") == "0" and not Path(result_path).is_file():
        _write_json(Path(result_path), result)
    return 0


def _stream_child_output(process: ChildProcess) -> tuple[int, str, str]:
    return stream_child_output(
        process,
        stdout_sink=print_stdout_sink,
        stderr_sink=print_stderr_sink,
        capture_stdout=True,
    )


# reason: record torchrun exposes root/commit as its Modal schema; bundling would break callers.
def record_torchrun_outcome(  # ruff: ignore[too-many-arguments]
    *,
    root: Path,
    result_path: Path,
    result: Mapping[str, Any],
    returncode: int,
    stdout: str,
    stderr: str,
    commit: Callable[[], object],
) -> dict[str, Any]:
    """Persist the complete child outcome before the parent may re-raise it.

    Returns:
        The persisted outcome mapping. It is written before the parent is allowed to re-raise, so
        a failing child still leaves its complete record behind.

    """
    outcome = dict(result)
    outcome["torchrun_returncode"] = returncode
    outcome["torchrun_stdout_tail"] = stdout[-4_000:]
    outcome["torchrun_stderr_tail"] = stderr[-4_000:]
    successful = returncode == 0 and outcome.get("status") == "ok"
    event = "torchrun_completed" if successful else "torchrun_failed"
    _write_json(result_path, outcome)
    _append_event(root, {"event": event, **outcome})
    commit()
    return outcome


def _run_isolated_torchrun(spec: ChildSpec) -> dict[str, Any]:
    root = Path(spec.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec_path, result_path = root / "child.spec.json", root / "result.json"
    _write_json(spec_path, asdict(spec))
    command = torchrun_command(str(spec_path), str(result_path))
    _append_event(
        root,
        {
            "event": "torchrun_starting",
            "command": list(command),
            "pre_torchrun_interconnect": spec.pre_torchrun_interconnect,
        },
    )
    try:
        # reason: torchrun_command builds fixed flags; spec/result paths are single operands and no shell is involved.
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
    except OSError as error:
        outcome = record_torchrun_outcome(
            root=root,
            result_path=result_path,
            result={
                "status": "failed",
                "error": f"torchrun process could not start: {error!r}",
                "artifact_root": str(root),
            },
            returncode=127,
            stdout="",
            stderr="",
            commit=artifacts.commit,
        )
        msg = f"PII350 DDP smoke torchrun failed: {outcome}"
        raise RuntimeError(msg) from error
    _append_event(
        root,
        {
            "event": "torchrun_started",
            "child_pid": process.pid,
            "command": list(command),
            "outer_modal_started_monotonic": spec.outer_modal_started_monotonic,
            "absolute_training_deadline_monotonic": spec.absolute_training_deadline_monotonic,
            "outer_elapsed_seconds_at_torchrun_spawn": (time.monotonic() - spec.outer_modal_started_monotonic),
            "pre_torchrun_interconnect": spec.pre_torchrun_interconnect,
        },
    )
    returncode, stdout, stderr = _stream_child_output(process)
    if not result_path.is_file():
        _write_json(
            result_path,
            {
                "status": "failed",
                "error": f"torchrun exited {returncode}; stderr={stderr[-4_000:]}",
                "artifact_root": str(root),
            },
        )
    parsed_result: object
    try:
        parsed_result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parsed_result = {
            "status": "failed",
            "error": f"torchrun result could not be read: {error!r}",
            "artifact_root": str(root),
        }
    if not isinstance(parsed_result, Mapping):
        parsed_result = {
            "status": "failed",
            "error": "torchrun result was not a JSON object",
            "artifact_root": str(root),
        }
    result: dict[str, Any] = dict(parsed_result)
    if returncode == 0 and result.get("status") == "ok":
        outer_elapsed_before_parent_commit = time.monotonic() - spec.outer_modal_started_monotonic
        result["outer_modal_elapsed_seconds_before_parent_commit"] = outer_elapsed_before_parent_commit
        result["estimated_all_in_cost_usd_before_parent_commit"] = (
            outer_elapsed_before_parent_commit * ALL_IN_RATE_USD_PER_SECOND
        )
        result["post_child_parent_commit_time_excluded"] = True
    outcome = record_torchrun_outcome(
        root=root,
        result_path=result_path,
        result=result,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        commit=artifacts.commit,
    )
    if returncode != 0 or outcome.get("status") != "ok":
        msg = f"PII350 DDP smoke torchrun failed: {outcome}"
        raise RuntimeError(msg)
    return outcome


@app.function(image=image, volumes=COMMON_MOUNTS, secrets=[secret], **DDP_OPTIONS)
def run_pii350_ddp_smoke(
    contract: Mapping[str, Any],
    *,
    execute: bool = False,
    confirmation: str = "",
    primary_action: str = "",
) -> dict[str, Any]:
    outer_modal_started_monotonic = time.monotonic()
    absolute_deadline_monotonic = absolute_training_deadline(outer_started_monotonic=outer_modal_started_monotonic)
    require_ddp_smoke_execute(
        contract,
        execute=execute,
        confirmation=confirmation,
        primary_action=primary_action,
    )
    baseline = matched_constant3_contract()
    shard_paths, checkpoint_attestation = _validated_receipt_inventory(baseline)
    if checkpoint_attestation is None:
        msg = "DDP smoke requires the PII350 CPU receipt checkpoint attestation"
        raise RuntimeError(msg)
    root = Path(ARTIFACT_ROOT) / f"{contract['config_digest']}-{time.time_ns()}"
    try:
        pre_torchrun_interconnect = capture_pre_torchrun_interconnect()
    except PreTorchrunInterconnectError as error:
        failure = {
            "status": "failed",
            "error": f"pre-torchrun interconnect evidence rejected: {error}",
            "artifact_root": str(root),
            "pre_torchrun_interconnect": error.evidence,
        }
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "result.json", failure)
        _append_event(root, {"event": "pre_torchrun_interconnect_failed", **failure})
        artifacts.commit()
        raise
    return _run_isolated_torchrun(
        ChildSpec(
            dict(contract),
            baseline,
            shard_paths,
            dict(checkpoint_attestation),
            str(root),
            outer_modal_started_monotonic,
            absolute_deadline_monotonic,
            pre_torchrun_interconnect,
        ),
    )


def _require_profile() -> None:
    require_modal_profile(MODAL_PROFILE, "PII350 DDP smoke")


def require_ddp_prewarm_only(
    *,
    preflight_assets: bool,
    execute: bool,
    confirmation: str,
    primary_action: str,
    config_digest: str,
) -> None:
    """Reject every receipt or GPU-launch flag from the cache-hydration surface.

    Raises:
        RuntimeError: if any receipt argument or paid GPU-launch flag is present. This surface
            hydrates caches only, so accepting a launch flag here would let a typo start a billed
            run from the cheap path.

    """
    if preflight_assets or execute or confirmation or primary_action or config_digest:
        msg = "DDP asset prewarm accepts only --prewarm-assets; no receipt or paid launch flags"
        raise RuntimeError(msg)


# reason: pii350 ddp smoke exposes preflight/config as its Modal schema; bundling would break callers.
@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    preflight_assets: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    prewarm_assets: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    execute: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    confirmation: str = "",
    primary_action: str = "",
    config_digest: str = "",
) -> None:
    if prewarm_assets:
        require_ddp_prewarm_only(
            preflight_assets=preflight_assets,
            execute=execute,
            confirmation=confirmation,
            primary_action=primary_action,
            config_digest=config_digest,
        )
        _require_profile()
        print(json.dumps(prewarm_ddp_assets.remote(), indent=2, sort_keys=True))
        return
    if preflight_assets:
        if execute or confirmation or primary_action or config_digest:
            msg = "DDP smoke preflight cannot include paid launch flags"
            raise RuntimeError(msg)
        _require_profile()
        print(json.dumps(preflight_ddp_assets.remote(), indent=2, sort_keys=True))
        return
    contract = render_ddp_smoke_contract()
    if config_digest != contract["config_digest"]:
        msg = "DDP smoke requires the exact --config-digest"
        raise RuntimeError(msg)
    _require_profile()
    print(
        json.dumps(
            run_pii350_ddp_smoke.remote(
                contract,
                execute=execute,
                confirmation=confirmation,
                primary_action=primary_action,
            ),
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    child_paths = child_argv_paths(sys.argv)
    if child_paths is not None:
        raise SystemExit(_child_main(*child_paths))
