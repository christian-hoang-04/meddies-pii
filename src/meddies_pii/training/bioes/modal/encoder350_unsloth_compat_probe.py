"""One paid, one-step Encoder350 Unsloth 2026.7.4 compatibility qualification."""

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
# ruff: file-ignore[import-private-name]
# reason: the Modal orchestration modules share `full_run` and `full_run_runtime` privates —
# reason: `_validated_receipt_inventory`, `_asset_receipt_contract`, `_preflight_candidate_assets`,
# reason: `_receipt_path`, `_iter_packed_rows`, `_packed_batch` and siblings — so every launch path
# reason: computes its receipts and batches from ONE implementation. A second copy on a spend gate is
# reason: the failure mode this avoids. The underscore is the defect, not the import; promoting them
# reason: to a real seam is a public-API change on the spend surface and sits on the owner ledger.
import json
import sys
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import modal

from meddies_pii.json_types import as_json_object
from meddies_pii.training.bioes.modal.full_run import _validated_receipt_inventory
from meddies_pii.training.bioes.modal.probe_scaffold import (
    VolumeMounts,
    append_event,
    artifact_volume,
    child_argv_paths,
    hf_cache_environment,
    hf_cache_volume,
    hf_secret,
    launch_child_process,
    print_stderr_sink,
    print_stdout_sink,
    require_modal_profile,
    source_mounted_image,
    stream_child_output,
    volume_mounts,
    write_child_spec,
)
from meddies_pii.training.bioes.trainers.config import DEFAULT_UNSLOTH_MECHANICS, LORA_TARGET_MODULES
from meddies_pii.training.bioes.trainers.encoder350_unsloth_compat_probe import (
    PROBE_CANDIDATE,
    PROBE_CONFIRMATION,
    PROBE_HARD_TIMEOUT_SECONDS,
    PROBE_RUNTIME_PACKAGES,
    PROBE_STEPS,
    render_probe,
    require_probe_execute,
    require_runtime_package_tuple,
    resolve_encoder350_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch.nn import Module

    from meddies_pii.training.bioes.modal.probe_scaffold import ChildProcess

MIN_PACKED_SEGMENTS = 2

ARTIFACT_ROOT = "/artifacts/encoder350-unsloth-2026-7-4-compat"
CACHE_ENVIRONMENT = hf_cache_environment()
IMAGE_PACKAGES = (
    "torch==2.10.0",
    "transformers==5.2.0",
    "peft==0.19.1",
    "pyarrow==23.0.0",
    "huggingface_hub==1.11.0",
    "nvidia-ml-py==13.590.44",
    "unsloth==2026.7.4",
    "unsloth_zoo==2026.7.4",
)


class _H100Options(TypedDict):
    gpu: str
    cpu: float
    memory: int
    timeout: int
    max_containers: int


cache = hf_cache_volume()
artifacts = artifact_volume()
secret = hf_secret()
image = source_mounted_image(
    modal.Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install(*IMAGE_PACKAGES)
    .env(CACHE_ENVIRONMENT),
)
app = modal.App("meddies-encoder350-unsloth-2026-7-4-compat")
H100_OPTIONS: _H100Options = {
    "gpu": "H100!",
    "cpu": 4.0,
    "memory": 64 * 1024,
    "timeout": PROBE_HARD_TIMEOUT_SECONDS,
    "max_containers": 1,
}
MOUNTS: VolumeMounts = volume_mounts(cache, artifacts)


def _lora_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    training = contract["training"]
    lora = {
        "rank": int(training["lora_rank"]),
        "alpha": int(training["lora_alpha"]),
        "dropout": float(training["lora_dropout"]),
        "target_modules": list(LORA_TARGET_MODULES),
    }
    mechanics = {
        **DEFAULT_UNSLOTH_MECHANICS,
        "gradient_checkpointing": training["gradient_checkpointing"],
        "lora": {
            **DEFAULT_UNSLOTH_MECHANICS["lora"],
            **lora,
            "random_state": int(contract["seed"]),
        },
    }
    return {"training": {"lora": lora, "mechanics": mechanics}}


@dataclass(frozen=True, slots=True)
class ChildSpec:
    contract: dict[str, Any]
    shard_paths: tuple[str, ...]
    expected_encoder_checkpoint_attestation: dict[str, Any]
    artifact_dir: str


def _validate_child_spec(spec: ChildSpec) -> None:
    require_probe_execute(spec.contract, execute=True, confirmation=PROBE_CONFIRMATION)
    if not spec.shard_paths:
        msg = "Encoder350 compatibility probe requires verified packed shards"
        raise ValueError(msg)
    if not spec.expected_encoder_checkpoint_attestation:
        msg = "Encoder350 compatibility probe requires CPU body attestation"
        raise ValueError(msg)
    if not Path(spec.artifact_dir).is_absolute():
        msg = "Encoder350 compatibility probe artifact path must be absolute"
        raise ValueError(msg)


def _append_event(root: Path, event: Mapping[str, Any]) -> None:
    append_event(root, event, commit=artifacts.commit)


def _first_packed_batch(paths: Sequence[str], batch_size: int) -> dict[str, Any]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
    import torch

    rows: list[Mapping[str, Any]] = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            rows.extend(batch.to_pylist())
            if len(rows) >= batch_size:
                break
        if len(rows) >= batch_size:
            break
    if len(rows) < batch_size:
        msg = "verified packed shards do not contain one physical batch"
        raise RuntimeError(msg)
    rows = rows[:batch_size]
    lengths = [length for row in rows for length in row["seq_lengths"] if length > 0]
    inputs = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device="cuda")
    if sum(lengths) != inputs.numel():
        msg = "packed lengths do not cover the physical batch"
        raise RuntimeError(msg)
    return {
        "input_ids": inputs,
        "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long, device="cuda"),
        "position_ids": torch.tensor([row["position_ids"] for row in rows], dtype=torch.long, device="cuda"),
        "packed_seq_lengths": torch.tensor(lengths, dtype=torch.int32, device="cuda"),
    }


def _require_document_isolation(tagger: Module, paths: Sequence[str]) -> dict[str, Any]:
    """Use two packed documents to prove no cross-document attention before step one.

    Returns:
        The contamination proof as a mapping, recorded in the probe receipt.

    Raises:
        RuntimeError: if the packed batch holds only one document, so isolation is untestable;
            or if the proof shows attention crossing a document boundary.

    """
    import pyarrow.parquet as pq

    from meddies_pii.training.bioes.trainers.contamination import (
        run_packed_attention_contamination_probe,
    )
    from meddies_pii.training.bioes.trainers.packing import (
        PackedRowRange,
        PackedTrainingUnit,
    )

    candidate: Mapping[str, Any] | None = None
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                if len(row["row_ranges"]) >= MIN_PACKED_SEGMENTS:
                    candidate = row
                    break
            if candidate is not None:
                break
        if candidate is not None:
            break
    if candidate is None:
        msg = "packed isolation proof requires a multi-document unit"
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
        device="cuda",
        atol=0.0,
        rtol=0.0,
    )
    # reason: the equivalence probe runs at atol=0.0/rtol=0.0 and the claim it proves is bit-exact
    # reason: equality, so any tolerance here would let a real drift or contamination pass the gate.
    if not proof.passed or proof.max_abs_diff != 0.0:  # ruff: ignore[float-equality-comparison]
        msg = f"document-isolated packing contamination proof failed: {proof}"
        raise RuntimeError(msg)
    return proof.to_dict()


# reason: run child combines child spec and check state; splitting would split cleanup from writes.
def _run_child(spec: ChildSpec) -> dict[str, Any]:  # ruff: ignore[too-many-locals,too-many-statements]
    """Fresh interpreter: exact packages, strict FastModel path, exactly one optimizer step.

    Unsloth must patch before PEFT/Transformers imports.

    Returns:
        The probe receipt for this child run, including its loss and isolation proof.

    Raises:
        RuntimeError: if CUDA is unavailable; if the Unsloth wrapper exposes no `.lfm2` body or
            the adapter model no config; if document-isolated packing is lost; or if the single
            step produces a loss that is absent or non-finite. Nothing here is caught locally —
            this function's `try` has no matching handler, so every one escapes to the caller.

    """
    import torch

    # reason: unsloth publishes no macOS wheel and installs in the training container, so this import is env-only.
    # reason: unsloth patches transformers and peft at import time, so it must load first.
    fast_model = import_module("unsloth").FastModel

    from huggingface_hub import snapshot_download
    from peft import TaskType
    from transformers import AutoModelForMaskedLM

    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _attach_unsloth_feature_extraction_lora,
        _propagate_unsloth_remote_code_provenance,
        _validate_observed_training_state,
        _validate_trainable_lora_parameters,
        require_lfm2_checkpoint_body_state_attestation,
    )

    _validate_child_spec(spec)
    if not torch.cuda.is_available():
        msg = "Encoder350 Unsloth compatibility probe requires CUDA"
        raise RuntimeError(msg)
    observed_packages = {name: version(name) for name in PROBE_RUNTIME_PACKAGES}
    require_runtime_package_tuple(observed_packages)
    contract, root = spec.contract, Path(spec.artifact_dir)
    torch.manual_seed(int(contract["seed"]))
    snapshot = resolve_encoder350_snapshot(snapshot_download)
    body_hash = require_lfm2_checkpoint_body_state_attestation(spec.expected_encoder_checkpoint_attestation, snapshot)
    wrapper, _tokenizer = fast_model.from_pretrained(
        model_name=str(contract["model_id"]),
        revision=str(contract["model_revision"]),
        tokenizer_name=snapshot,
        max_seq_length=int(contract["packed_dataset"]["max_length"]),
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        dtype=torch.bfloat16,
        full_finetuning=False,
        fast_inference=False,
        trust_remote_code=True,
        auto_model=AutoModelForMaskedLM,
    )
    body = getattr(wrapper, "lfm2", None)
    if body is None:
        msg = "Unsloth Encoder350 wrapper has no .lfm2 body"
        raise RuntimeError(msg)
    _propagate_unsloth_remote_code_provenance(wrapper, body)
    model = _attach_unsloth_feature_extraction_lora(
        fast_model,
        body,
        task_type=TaskType.FEATURE_EXTRACTION,
        contract=_lora_contract(contract),
    )
    if getattr(model, "config", None) is None:
        msg = "Unsloth Encoder350 adapter model has no config"
        raise RuntimeError(msg)
    model.config.use_cache = False
    training_state = _validate_observed_training_state(model)
    lora = _validate_trainable_lora_parameters(model, LORA_TARGET_MODULES)
    tagger = (
        HiddenStateTokenTagger(
            model,
            int(model.config.hidden_size),
            37,
            request_hidden_states=False,
            classifier_dtype=torch.bfloat16,
            dropout=0.1,
            packed_segment_isolation=True,
        )
        .cuda()
        .train()
    )
    if not tagger.packed_segment_isolation:
        msg = "Encoder350 compatibility probe lost document-isolated packing"
        raise RuntimeError(msg)
    contamination = _require_document_isolation(tagger, spec.shard_paths)
    _append_event(
        root,
        {
            "event": "runtime_attested",
            "runtime_backend": "unsloth",
            "installed_package_versions": observed_packages,
            "loader": "FastModel.from_pretrained(auto_model=AutoModelForMaskedLM)",
            "body_checkpoint_attestation": body_hash,
            "packed_segment_isolation": True,
            "document_isolation_proof": contamination,
            **training_state,
            **lora,
        },
    )
    optimizer = torch.optim.AdamW(
        [p for p in tagger.parameters() if p.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=False,
    )
    batch = _first_packed_batch(spec.shard_paths, int(contract["batch_size"]))
    torch.cuda.synchronize()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    output = tagger(**batch)
    loss = output["loss"]
    if loss is None or not torch.isfinite(loss):
        msg = "Encoder350 compatibility probe loss is absent or non-finite"
        raise RuntimeError(msg)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    _append_event(
        root,
        {
            "event": "optimizer_step",
            "step": 1,
            "loss": float(loss.detach().float().cpu()),
            "elapsed_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "finite_loss": True,
        },
    )
    result = {
        "status": "ok",
        "candidate": PROBE_CANDIDATE,
        "runtime_backend": "unsloth",
        "optimizer_steps": PROBE_STEPS,
        "artifact_root": str(root),
        "events_path": str(root / "events.jsonl"),
        "installed_package_versions": observed_packages,
    }
    (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.commit()
    return result


def _child_main(spec_path: str, result_path: str) -> int:
    spec = ChildSpec(**json.loads(Path(spec_path).read_text(encoding="utf-8")))
    try:
        result = _run_child(spec)
    # reason: this child-process terminal boundary must persist every compatibility failure before exiting.
    except Exception as error:  # ruff: ignore[blind-except]
        result = {"status": "failed", "error": repr(error)}
    Path(result_path).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    artifacts.commit()
    return 0 if result["status"] == "ok" else 1


def _stream_child_output(process: ChildProcess) -> tuple[int, str]:
    returncode, _stdout, stderr = stream_child_output(
        process,
        stdout_sink=print_stdout_sink,
        stderr_sink=print_stderr_sink,
    )
    return returncode, stderr


def _run_isolated_child(spec: ChildSpec) -> dict[str, Any]:
    root = Path(spec.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec_path, result_path = root / "child.spec.json", root / "child.result.json"
    write_child_spec(spec_path, asdict(spec))
    process = launch_child_process(
        "meddies_pii.training.bioes.modal.encoder350_unsloth_compat_probe",
        spec_path,
        result_path,
    )
    _append_event(root, {"event": "child_process_started", "child_pid": process.pid})
    returncode, stderr = _stream_child_output(process)
    if not result_path.is_file():
        msg = f"Encoder350 compatibility child pid={process.pid} exited {returncode}; stderr={stderr[-4000:]}"
        raise RuntimeError(
            msg,
        )
    result = as_json_object(json.loads(result_path.read_text(encoding="utf-8")))
    if result is None:
        msg = "Encoder350 compatibility child result must be a JSON object"
        raise RuntimeError(msg)
    if returncode or result.get("status") != "ok":
        msg = f"Encoder350 compatibility child failed: {result}"
        raise RuntimeError(msg)
    return result


@app.function(image=image, volumes=MOUNTS, secrets=[secret], **H100_OPTIONS)
def run_encoder350_unsloth_compat_probe(
    contract: Mapping[str, Any],
    *,
    execute: bool = False,
    confirmation: str = "",
) -> dict[str, Any]:
    require_probe_execute(contract, execute=execute, confirmation=confirmation)
    from meddies_pii.training.bioes.trainers.full_run import render_full_run

    shards, attestation = _validated_receipt_inventory(render_full_run(PROBE_CANDIDATE))
    if attestation is None:
        msg = "Encoder350 compatibility probe requires CPU receipt body attestation"
        raise RuntimeError(msg)
    root = Path(ARTIFACT_ROOT) / f"{contract['config_digest']}-{time.time_ns()}"
    return _run_isolated_child(ChildSpec(dict(contract), shards, dict(attestation), str(root)))


def _local_remote_call(contract: Mapping[str, Any], *, execute: bool, confirmation: str) -> dict[str, Any]:
    require_modal_profile("huyhoang041100", "Encoder350 compatibility probe")
    return run_encoder350_unsloth_compat_probe.remote(contract, execute=execute, confirmation=confirmation)


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(execute: bool = False, confirmation: str = "") -> None:  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    contract = render_probe()
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
