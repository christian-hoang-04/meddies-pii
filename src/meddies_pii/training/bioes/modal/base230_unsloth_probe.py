"""Isolated, candidate-only Base230 Unsloth throughput probes."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack (torch, transformers, unsloth, datasets) is installed in the Modal image, so each
# reason: function body imports it inside the container.
# ruff: file-ignore[print]
# reason: this module runs inside a Modal container; its standard output is the operator's streamed run log.
# ruff: file-ignore[implicit-namespace-package]
# reason: `modal/` is the only subpackage of `bioes/` without an `__init__.py` — assembly, data, eval,
# reason: reports and trainers all have one — so the asymmetry reads as an oversight, and adding the
# reason: file is likely inert under hatchling's src-layout discovery.
# reason: Deliberately deferred rather than fixed here: this directory holds every spend-authorization
# reason: gate, and its Modal-remote import paths have only fake-mediated local coverage, so adding
# reason: `__init__.py` is a post-merge change whose proof is a real GPU smoke run — owner ledger item.
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import modal

from meddies_pii.json_types import as_json_object
from meddies_pii.training.bioes.data.artifacts import load_pretrained
from meddies_pii.training.bioes.modal.base_selection import (
    require_modal_profile,
    verify_packed_artifact,
)
from meddies_pii.training.bioes.modal.probe_scaffold import (
    CACHE_MOUNT,
    VolumeMounts,
    append_event,
    artifact_volume,
    child_argv_paths,
    hf_cache_environment,
    hf_cache_volume,
    hf_secret,
    launch_child_process,
    source_mounted_image,
    stream_child_output,
    volume_mounts,
    write_child_spec,
)
from meddies_pii.training.bioes.trainers.base230_unsloth_probe import (
    PROBE_CANDIDATE,
    PROBE_CHILD_DEADLINE_SECONDS,
    PROBE_CONFIRMATION,
    PROBE_HARD_TIMEOUT_SECONDS,
    PROBE_MAX_LIVE_ESTIMATE_USD,
    PROBE_RATE_USD_PER_SECOND,
    PROBE_STEPS,
    build_unsloth_base230,
    render_probe,
    require_probe_execute,
    resolve_base230_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from meddies_pii.training.bioes.modal.probe_scaffold import ChildProcess

MIN_PACKED_SEGMENTS = 2

ARTIFACT_ROOT = "/artifacts/base230-unsloth-probe"
H100_CACHE_ENVIRONMENT = hf_cache_environment(
    disable_unsloth_statistics=False,
)
"""CPU prewarm uses ``snapshot_download(cache_dir=/cache/hf)``.

That places ``models--...`` directly under this root, not under ``hub/``.

"""
IMAGE_PACKAGES = (
    "torch==2.10.0",
    "transformers==5.2.0",
    "peft==0.19.1",
    "pyarrow==23.0.0",
    "huggingface_hub==1.11.0",
    "nvidia-ml-py==13.590.44",
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
)


class _H100Options(TypedDict):
    gpu: str
    cpu: float
    memory: int
    timeout: int
    max_containers: int
    volumes: VolumeMounts
    secrets: list[modal.Secret]


class _CpuPreflightOptions(TypedDict):
    gpu: None
    cpu: float
    memory: int
    timeout: int
    max_containers: int
    volumes: VolumeMounts


def _build_probe_image(base_image: modal.Image) -> modal.Image:
    """Install runtime dependencies before setting the offline cache contract.

    Returns:
        The image with the pinned packages installed and the offline cache environment set,
        in that order -- the environment last, so the install itself may still reach the
        network while everything after it cannot.

    """
    return base_image.pip_install(*IMAGE_PACKAGES).env(H100_CACHE_ENVIRONMENT)


image = source_mounted_image(
    _build_probe_image(
        modal.Image.from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941"),
    ),
)
cache = hf_cache_volume()
artifacts = artifact_volume()
secret = hf_secret()
app = modal.App("meddies-base230-unsloth-bioes-probe")
FUNCTION_OPTIONS: _H100Options = {
    "gpu": "H100!",
    "cpu": 4.0,
    "memory": 64 * 1024,
    "timeout": PROBE_HARD_TIMEOUT_SECONDS,
    "max_containers": 1,
    "volumes": volume_mounts(cache, artifacts),
    "secrets": [secret],
}
CPU_PREFLIGHT_OPTIONS: _CpuPreflightOptions = {
    "gpu": None,
    "cpu": 1.0,
    "memory": 2 * 1024,
    "timeout": 300,
    "max_containers": 1,
    "volumes": {CACHE_MOUNT: cache},
}


@dataclass(frozen=True, slots=True)
class ChildSpec:
    contract: dict[str, Any]
    manifest_path: str
    shard_paths: tuple[str, ...]
    artifact_dir: str
    deadline_monotonic: float


def _validate_child_spec(spec: ChildSpec) -> None:
    require_probe_execute(spec.contract, execute=True, confirmation=PROBE_CONFIRMATION)
    if not spec.shard_paths or not spec.manifest_path:
        msg = "Base230 Unsloth child requires verified packed shards"
        raise ValueError(msg)
    if not Path(spec.artifact_dir).is_absolute():
        msg = "Base230 Unsloth child artifact path must be absolute"
        raise ValueError(msg)


def _preflight_base230_tokenizer_from_snapshot(
    *,
    snapshot_download: Callable[..., str],
    auto_config: object,
    auto_processor: object,
    auto_tokenizer: object,
) -> dict[str, Any]:
    """Offline proof of the exact local snapshot Unsloth passes as tokenizer_name.

    Returns:
        The snapshot path, model type, processor and tokenizer class names, vocabulary size
        and the files checked -- read back off objects loaded with ``local_files_only``, so
        the record proves the cache resolves rather than that a download would.

    Raises:
        RuntimeError: If the snapshot lacks ``config.json`` or ``tokenizer_config.json``, if
            the offline config resolves no non-empty ``model_type``, or if the tokenizer
            reports no positive ``vocab_size``. Each names what was missing rather than
            failing later inside Unsloth.

    """
    snapshot = Path(resolve_base230_snapshot(snapshot_download))
    required_files = ("config.json", "tokenizer_config.json")
    missing = [name for name in required_files if not (snapshot / name).is_file()]
    if missing:
        msg = f"Base230 local snapshot lacks required files: {missing}"
        raise RuntimeError(msg)
    loader_kwargs = {"local_files_only": True}
    config = load_pretrained(auto_config, str(snapshot), loader_kwargs)
    processor = load_pretrained(auto_processor, str(snapshot), loader_kwargs)
    tokenizer = load_pretrained(auto_tokenizer, str(snapshot), loader_kwargs)
    model_type = getattr(config, "model_type", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(model_type, str) or not model_type:
        msg = "offline Base230 config resolution returned no model_type"
        raise RuntimeError(msg)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        msg = "offline Base230 tokenizer returned no positive vocab_size"
        raise RuntimeError(msg)
    return {
        "snapshot_path": str(snapshot),
        "model_type": model_type,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": vocab_size,
        "required_files": list(required_files),
    }


def _iter_rows(paths: Sequence[str]) -> Iterator[Mapping[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    while True:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                yield from batch.to_pylist()


def _batch_rows(rows: Sequence[Mapping[str, Any]], device: str) -> tuple[dict[str, Any], int]:
    import torch

    ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"] for row in rows], dtype=torch.long, device=device)
    positions = torch.tensor([row["position_ids"] for row in rows], dtype=torch.long, device=device)
    lengths = torch.tensor(
        [length for row in rows for length in row["seq_lengths"] if length > 0],
        dtype=torch.int32,
        device=device,
    )
    if int(lengths.sum().item()) != int(ids.numel()):
        msg = "packed lengths do not cover the physical batch"
        raise RuntimeError(msg)
    return {
        "input_ids": ids,
        "labels": labels,
        "position_ids": positions,
        "packed_seq_lengths": lengths,
    }, sum(int(row["real_token_count"]) for row in rows)


def _append_event(
    root: Path,
    event: Mapping[str, Any],
    *,
    commit: Callable[[], None] | None = None,
) -> None:
    append_event(root, event, commit=commit, fsync=True)


class _NvmlSampler:
    """Samples during work; synchronized step-end snapshots are idle-only."""

    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # reason: pynvml ships with the NVIDIA driver container and is absent here; every use sits inside try/except.
        pynvml = import_module("pynvml")

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
                "device_vram_bytes": float(memory.total),
                "device_vram_used_bytes": float(memory.used),
            })

        def sample() -> None:
            while not self._stop.wait(0.5):
                sample_once()

        sample_once()
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def latest(self) -> dict[str, float]:
        return dict(self.samples[-1]) if self.samples else {}

    def stop(self) -> dict[str, float | int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "nvml_sample_count": len(self.samples),
            "peak_device_vram_used_bytes": max(
                (sample["device_vram_used_bytes"] for sample in self.samples),
                default=None,
            ),
        }


# reason: run child orders child spec before resolve; helper seams would split cleanup from writes.
def _run_child(spec: ChildSpec) -> dict[str, Any]:  # ruff: ignore[too-many-locals,too-many-statements]
    """Run the real Unsloth-only body in a fresh interpreter.

    Returns:
        The probe record: ``"ok"`` with the candidate, backend, execution id, batch size and
        step counts once every step lands, or ``"budget_exhausted"`` naming the step it
        stopped at and the events path, so an exhausted run is a reported outcome rather
        than a failure.

    Raises:
        RuntimeError: If CUDA is absent; if the Unsloth backbone loads in something other
            than a bfloat16 floating runtime dtype, which would silently change what the
            probe measures; if no packed unit carries two segments, leaving the
            contamination probe no genuine boundary; if that probe reports contamination; or
            if the BIOES classifier produces no loss.

    """
    import torch

    fast_language_model = import_module("unsloth").FastLanguageModel

    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger
    from meddies_pii.training.bioes.trainers.contamination import (
        run_packed_attention_contamination_probe,
    )
    from meddies_pii.training.bioes.trainers.packing import (
        PackedRowRange,
        PackedTrainingUnit,
    )

    _validate_child_spec(spec)
    contract = spec.contract
    if not torch.cuda.is_available():
        msg = "Base230 Unsloth probe requires CUDA"
        raise RuntimeError(msg)
    root = Path(spec.artifact_dir)
    started = time.monotonic()
    torch.manual_seed(int(contract["seed"]))
    from huggingface_hub import snapshot_download

    tokenizer_name = resolve_base230_snapshot(snapshot_download)
    model, _tokenizer = build_unsloth_base230(
        fast_language_model,
        dtype=torch.bfloat16,
        tokenizer_name=tokenizer_name,
    )
    floating_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()})
    if "torch.bfloat16" not in floating_dtypes:
        msg = "Base230 Unsloth backbone did not load a bfloat16 floating runtime dtype"
        raise RuntimeError(msg)
    dtype = torch.bfloat16
    tagger = (
        HiddenStateTokenTagger(
            backbone=model,
            hidden_size=int(model.config.hidden_size),
            num_labels=37,
            request_hidden_states=True,
            classifier_dtype=dtype,
            dropout=0.1,
            packed_segment_isolation=True,
        )
        .cuda()
        .train()
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in tagger.parameters() if parameter.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=False,
    )
    _append_event(
        root,
        {
            "event": "bf16_backbone_verified",
            "backbone_floating_dtypes": floating_dtypes,
            "classifier_dtype": str(tagger.classifier.weight.dtype),
        },
        commit=artifacts.commit,
    )
    iterator = _iter_rows(spec.shard_paths)
    first = next(iterator)
    for _ in range(32):
        if len(first["row_ranges"]) >= MIN_PACKED_SEGMENTS:
            break
        first = next(iterator)
    ranges = tuple(PackedRowRange(str(item["uid"]), int(item["start"]), int(item["end"])) for item in first["row_ranges"])
    if len(ranges) < MIN_PACKED_SEGMENTS:
        msg = "packed contamination probe needs two segments"
        raise RuntimeError(msg)
    mutated_ids = list(first["input_ids"])
    mutated_ids[ranges[0].start] = (mutated_ids[ranges[0].start] + 1) % 100
    clean = PackedTrainingUnit(
        tuple(first["input_ids"]),
        tuple(first["labels"]),
        tuple(first["seq_lengths"]),
        tuple(first["position_ids"]),
        tuple(first["row_uids"]),
        ranges,
        int(first["real_token_count"]),
        int(first["padded_token_count"]),
        int(first["boundary_token_count"]),
    )
    mutated = PackedTrainingUnit(
        tuple(mutated_ids),
        tuple(first["labels"]),
        tuple(first["seq_lengths"]),
        tuple(first["position_ids"]),
        tuple(first["row_uids"]),
        ranges,
        int(first["real_token_count"]),
        int(first["padded_token_count"]),
        int(first["boundary_token_count"]),
    )
    contamination = run_packed_attention_contamination_probe(
        tagger,
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
        raise RuntimeError(msg)
    _append_event(
        root,
        {"event": "contamination_passed", **contamination.to_dict()},
        commit=artifacts.commit,
    )
    sampler = _NvmlSampler()
    sampler.start()
    timings: list[float] = []
    throughputs: list[float] = []
    batch_size = int(contract["batch_size"])
    try:
        for step in range(PROBE_STEPS):
            next_estimate = max(timings[-1] if timings else 5.0, 0.001)
            estimated_live = (time.monotonic() - started) * PROBE_RATE_USD_PER_SECOND
            if (
                time.monotonic() + next_estimate >= spec.deadline_monotonic
                or estimated_live + next_estimate * PROBE_RATE_USD_PER_SECOND > PROBE_MAX_LIVE_ESTIMATE_USD
            ):
                _append_event(
                    root,
                    {
                        "event": "budget_exhausted_before_next_step",
                        "step": step,
                        "estimated_live_cost_usd": estimated_live,
                        "maximum_live_estimate_usd": PROBE_MAX_LIVE_ESTIMATE_USD,
                    },
                    commit=artifacts.commit,
                )
                return {
                    "status": "budget_exhausted",
                    "step": step,
                    "events_path": str(root / "events.jsonl"),
                    **sampler.stop(),
                }
            rows = [next(iterator) for _ in range(batch_size)]
            batch, real_tokens = _batch_rows(rows, "cuda")
            torch.cuda.synchronize()
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            output = tagger(**batch)
            loss = output["loss"]
            if loss is None:
                msg = "BIOES classifier produced no loss"
                raise RuntimeError(msg)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - before
            timings.append(elapsed)
            tokens_per_second = real_tokens / elapsed
            throughputs.append(tokens_per_second)
            _append_event(
                root,
                {
                    "event": "optimizer_step",
                    "step": step,
                    "batch_size": batch_size,
                    "real_tokens": real_tokens,
                    "real_bioes_tokens_per_second": tokens_per_second,
                    "loss": float(loss.detach().float().cpu()),
                    "lr": 1e-4,
                    "packed_cursor": (step + 1) * batch_size,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    "estimated_live_cost_usd": (time.monotonic() - started) * PROBE_RATE_USD_PER_SECOND,
                    "estimated_billed_cost_usd": (time.monotonic() - started) * PROBE_RATE_USD_PER_SECOND * 1.20,
                    **sampler.latest(),
                },
                commit=artifacts.commit,
            )
        summary = sampler.stop()
        _append_event(root, {"event": "probe_complete", **summary}, commit=artifacts.commit)
        return {
            "status": "ok",
            "candidate": PROBE_CANDIDATE,
            "runtime_backend": "unsloth",
            "execution_id": root.name,
            "batch_size": batch_size,
            "optimizer_steps": PROBE_STEPS,
            "steps": PROBE_STEPS,
            "median_real_bioes_tokens_per_second": sorted(throughputs)[len(throughputs) // 2],
            "backbone_floating_dtypes": floating_dtypes,
            "events_path": str(root / "events.jsonl"),
            "artifact_root": str(root),
            **summary,
        }
    finally:
        sampler.stop()


def _child_main(spec_path: str, result_path: str) -> int:
    spec = ChildSpec(**json.loads(Path(spec_path).read_text(encoding="utf-8")))
    try:
        _validate_child_spec(spec)
        result = _run_child(spec)
        code = 0 if result["status"] == "ok" else 1
    # reason: this child-process terminal boundary must persist every failure as evidence before the process exits.
    except Exception as error:  # ruff: ignore[blind-except]
        result, code = {"status": "failed", "error": repr(error)}, 1
    Path(result_path).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.commit()
    return code


def _stream_child_output(
    process: ChildProcess,
    *,
    stdout_sink: Callable[[str], None],
    stderr_sink: Callable[[str], None],
) -> tuple[int, str]:
    returncode, _stdout, stderr = stream_child_output(
        process,
        stdout_sink=stdout_sink,
        stderr_sink=stderr_sink,
    )
    return returncode, stderr


def _run_isolated_child(spec: ChildSpec) -> dict[str, Any]:
    """Preserve JSONL exactly so Modal's live log boundary exposes each progress record before the child exits.

    Returns:
        The child's parsed result, read from the file it wrote rather than from its streamed
        output, which is forwarded live only so the operator can watch a paid GPU run.

    Raises:
        RuntimeError: If the child exits non-zero, reporting its pid and the last 4000
            characters of stderr; or if it exits cleanly having written a result whose
            status is not success, reporting that result.

    """
    root = Path(spec.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec_path, result_path = root / "child.spec.json", root / "child.result.json"
    write_child_spec(spec_path, asdict(spec))
    process = launch_child_process(
        "meddies_pii.training.bioes.modal.base230_unsloth_probe",
        spec_path,
        result_path,
    )
    _append_event(
        root,
        {"event": "child_process_started", "child_pid": process.pid},
        commit=artifacts.commit,
    )
    returncode, stderr = _stream_child_output(
        process,
        stdout_sink=lambda line: print(line, end="", flush=True),
        stderr_sink=lambda line: print(line, end="", file=sys.stderr, flush=True),
    )
    if not result_path.exists():
        msg = f"Base230 child pid={process.pid} exited {returncode}; stderr={stderr[-4000:]}"
        raise RuntimeError(msg)
    result = as_json_object(json.loads(result_path.read_text(encoding="utf-8")))
    if result is None:
        msg = "Base230 child result must be a JSON object"
        raise RuntimeError(msg)
    if returncode:
        msg = f"Base230 child failed: {result}"
        raise RuntimeError(msg)
    (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result


@app.function(image=image, **CPU_PREFLIGHT_OPTIONS)
def preflight_base230_cache() -> dict[str, Any]:
    """CPU-only cache smoke check. It never constructs Unsloth or allocates GPU.

    Returns:
        The same snapshot record the GPU path proves, obtained the same offline way, so a
        cache problem surfaces here at CPU rates instead of on an H100.

    """
    # reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
    # reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
    # reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer  # ty: ignore[possibly-missing-import]

    return _preflight_base230_tokenizer_from_snapshot(
        snapshot_download=snapshot_download,
        auto_config=AutoConfig,
        auto_processor=AutoProcessor,
        auto_tokenizer=AutoTokenizer,
    )


@app.function(image=image, **FUNCTION_OPTIONS)
def run_base230_unsloth_probe(
    contract: Mapping[str, Any],
    *,
    execute: bool = False,
    confirmation: str = "",
) -> dict[str, Any]:
    require_probe_execute(contract, execute=execute, confirmation=confirmation)
    artifact = verify_packed_artifact(revision=str(contract["packed_dataset"]["revision"]))
    root = Path(ARTIFACT_ROOT) / f"b{contract['batch_size']}-{time.time_ns()}"
    spec = ChildSpec(
        dict(contract),
        artifact.manifest_path,
        artifact.shard_paths,
        str(root),
        time.monotonic() + PROBE_CHILD_DEADLINE_SECONDS,
    )
    try:
        return _run_isolated_child(spec)
    finally:
        artifacts.commit()


def _local_remote_call(contract: Mapping[str, Any], *, execute: bool, confirmation: str) -> dict[str, Any]:
    """Keep the account guard adjacent to the only ``.remote`` call.

    Returns:
        Whatever the remote probe returns, unchanged. The guard runs first and in the same
        function, so no future edit can move a ``.remote`` call out from behind it.

    """
    require_modal_profile()
    return run_base230_unsloth_probe.remote(contract, execute=execute, confirmation=confirmation)


def _local_cpu_preflight_call() -> dict[str, Any]:
    require_modal_profile()
    return preflight_base230_cache.remote()


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(
    batch_size: int = 0,
    execute: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    confirmation: str = "",
    preflight: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
) -> None:
    if preflight:
        print(json.dumps(_local_cpu_preflight_call(), indent=2, sort_keys=True))
        return
    contract = render_probe(batch_size)
    require_probe_execute(contract, execute=execute, confirmation=confirmation)
    print(
        json.dumps(
            _local_remote_call(contract, execute=execute, confirmation=confirmation),
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    child_paths = child_argv_paths(sys.argv)
    if child_paths is not None:
        raise SystemExit(_child_main(*child_paths))
