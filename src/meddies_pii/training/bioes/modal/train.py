from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra; importing it at module load would make the package unimportable without
# reason: it.
# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
# ruff: file-ignore[implicit-namespace-package]
# reason: `modal/` is the only subpackage of `bioes/` without an `__init__.py` — assembly, data, eval,
# reason: reports and trainers all have one — so the asymmetry reads as an oversight, and adding the
# reason: file is likely inert under hatchling's src-layout discovery.
# reason: Deliberately deferred rather than fixed here: this directory holds every spend-authorization
# reason: gate, and its Modal-remote import paths have only fake-mediated local coverage, so adding
# reason: `__init__.py` is a post-merge change whose proof is a real GPU smoke run — owner ledger item.
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NotRequired, TypedDict, Unpack, cast

from modal.app import App
from modal.image import Image
from modal.secret import Secret
from modal.volume import Volume

from meddies_pii.json_types import JsonObject, is_json_value

if TYPE_CHECKING:
    from meddies_pii.training.bioes.modal.modal_types import TypedModalApp
    from meddies_pii.training.bioes.trainers.config import SmokeTrainingPayload

DEFAULT_MAX_LENGTH = 512
DEFAULT_TRAIN_LIMIT = 32
DEFAULT_EVAL_LIMIT = 16
DEFAULT_BATCH_SIZE = 2
DEFAULT_LORA_RANK = 4

DEFAULT_GPU = "A10"
H100_GPU = "H100"
ARTIFACT_VOLUME_NAME = "meddies-pii-bioes-artifacts"
ARTIFACT_VOLUME_MOUNT = "/artifacts"
DATA_VOLUME_NAME = "meddies-bioes-v2"
"""The bioes-v2 corpus volume (assembled by data_pipeline.assemble_mix).

Mounting it read-only here lets a retrain read the deduped mix directly off the volume: --dataset-id
/data/mix/train.<historical-label-corpus>.jsonl _load_rows (trainer.py) takes the local-file branch when dataset_id is a
path.

"""
DATA_VOLUME_MOUNT = "/data"
FULL_TRAINING_TIMEOUT_SECONDS = 24 * 60 * 60
PINNED_PACKAGES = (
    "torch==2.10.0",
    "transformers==5.5.0",
    "datasets==4.3.0",
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
    "python-dotenv==1.2.2",
    "pytest==9.1.1",
    "ruff==0.14.9",
)

image = (
    Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install(*PINNED_PACKAGES)
    .add_local_dir("src", remote_path="/root/src")
)

artifact_volume = Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
data_volume = Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
app = cast(
    "TypedModalApp",
    App("meddies-lfm25-unsloth-bioes-smoke-train", image=image),
)


class SmokeRunOptions(TypedDict, total=False):
    backend: str
    steps: int | None
    epochs: int | None
    logging_steps: int
    checkpoint_every_steps: int
    resume_from_checkpoint: str | None
    max_length: int
    train_limit: int
    eval_limit: int
    batch_size: int
    gradient_accumulation_steps: int
    lora_rank: int
    lora_alpha: int | None
    packing: bool
    length_bucketing: bool
    viterbi_calibration_path: str | None
    target_eval_slice: str | None
    target_eval_limit: int
    target_eval_scan_multiplier: int
    dataset_id: str
    dataset_split: str
    eval_dataset_split: str | None
    eval_dataset_id: str | None
    eval_dataset_revision: str | None
    dataset_revision: str | None
    model_id: str
    model_revision: str | None
    tokenizer_id: str | None
    tokenizer_revision: str | None
    train_config: str
    eval_config: str
    trust_remote_code: bool
    allow_same_local_eval: bool
    artifact_root: str | None


class TrainingArtifactProvenance(TypedDict):
    modal_app: str
    modal_profile: str | None
    modal_url: str | None
    command_kwargs: JsonObject
    out: str | None
    raw_log: str | None
    artifact_root: NotRequired[str]
    artifact_volume: NotRequired[str]
    artifact_persisted: NotRequired[bool]


class TrainingArtifactPayload(TypedDict):
    artifact_type: str
    schema_version: int
    provenance: TrainingArtifactProvenance
    result: SmokeTrainingPayload


def _uses_artifact_volume(artifact_root: str | None) -> bool:
    if artifact_root is None:
        return False
    root = str(PurePosixPath(artifact_root))
    return root == ARTIFACT_VOLUME_MOUNT or root.startswith(f"{ARTIFACT_VOLUME_MOUNT}/")


def _default_raw_log_path(out: str | None) -> str | None:
    if out is None:
        return None
    destination = Path(out)
    return str(destination.parent / "raw_logs" / f"{destination.stem}.log")


def _command_kwargs(gpu: str, options: SmokeRunOptions) -> JsonObject:
    command_kwargs: JsonObject = {"gpu": gpu}
    for name, value in options.items():
        if not is_json_value(value):
            msg = f"unsupported Modal command value: {type(value).__name__}"
            raise TypeError(msg)
        command_kwargs[name] = value
    return command_kwargs


# reason: artifact payload exposes result/artifact as its Modal schema; bundling would break callers.
def _artifact_payload(  # ruff: ignore[too-many-arguments]
    *,
    result: SmokeTrainingPayload,
    command_kwargs: JsonObject,
    out: str | None,
    raw_log: str | None,
    modal_url: str | None,
    modal_profile: str | None,
    artifact_root: str | None = None,
    artifact_volume: str | None = None,
    artifact_persisted: bool | None = None,
) -> TrainingArtifactPayload:
    provenance: TrainingArtifactProvenance = {
        "modal_app": "meddies-lfm25-unsloth-bioes-smoke-train",
        "modal_profile": modal_profile,
        "modal_url": modal_url,
        "command_kwargs": command_kwargs,
        "out": out,
        "raw_log": raw_log or _default_raw_log_path(out),
    }
    if artifact_root is not None:
        provenance["artifact_root"] = artifact_root
    if artifact_volume is not None:
        provenance["artifact_volume"] = artifact_volume
    if artifact_persisted is not None:
        provenance["artifact_persisted"] = artifact_persisted
    return {
        "artifact_type": "modal_smoke_train_result",
        "schema_version": 1,
        "provenance": provenance,
        "result": result,
    }


# reason: run smoke impl exposes backend/artifact as its Modal schema; bundling would break callers.
def _run_smoke_impl(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    backend: str = "unsloth",
    steps: int | None = None,
    epochs: int | None = None,
    logging_steps: int = 1,
    checkpoint_every_steps: int = 0,
    resume_from_checkpoint: str | None = None,
    max_length: int = 512,
    train_limit: int = 32,
    eval_limit: int = 16,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    lora_rank: int = 4,
    lora_alpha: int | None = None,
    *,
    packing: bool = False,
    length_bucketing: bool = False,
    viterbi_calibration_path: str | None = None,
    target_eval_slice: str | None = None,
    target_eval_limit: int = 4,
    target_eval_scan_multiplier: int = 32,
    dataset_id: str = "Meddies/meddies-pii",
    dataset_split: str = "train",
    eval_dataset_split: str | None = None,
    eval_dataset_id: str | None = None,
    eval_dataset_revision: str | None = None,
    dataset_revision: str | None = None,
    model_id: str = "LiquidAI/LFM2.5-350M-Base",
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    train_config: str = "train",
    eval_config: str = "test",
    trust_remote_code: bool = False,
    allow_same_local_eval: bool = False,
    artifact_root: str | None = None,
) -> SmokeTrainingPayload:

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")

    from meddies_pii.training.bioes.trainers import SmokeTrainingConfig
    from meddies_pii.training.bioes.trainers.trainer import run_smoke_training

    config = SmokeTrainingConfig(
        backend=backend,
        steps=10 if steps is None and epochs is None else steps,
        epochs=epochs,
        logging_steps=logging_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        max_length=max_length,
        train_limit=train_limit,
        eval_limit=eval_limit,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        packing=packing,
        length_bucketing=length_bucketing,
        viterbi_calibration_path=viterbi_calibration_path,
        target_eval_slice=target_eval_slice,
        target_eval_limit=target_eval_limit,
        target_eval_scan_multiplier=target_eval_scan_multiplier,
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        eval_dataset_split=eval_dataset_split,
        eval_dataset_id=eval_dataset_id,
        eval_dataset_revision=eval_dataset_revision,
        dataset_revision=dataset_revision,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        train_config=train_config,
        eval_config=eval_config,
        trust_remote_code=trust_remote_code,
        allow_same_local_eval=allow_same_local_eval,
    )
    if artifact_root is None:
        return run_smoke_training(config)
    result = run_smoke_training(
        config,
        artifact_root=artifact_root,
        artifact_commit=(artifact_volume.commit if _uses_artifact_volume(artifact_root) else None),
    )
    result["artifact_root"] = artifact_root
    if _uses_artifact_volume(artifact_root):
        artifact_volume.commit()
        result["artifact_volume"] = ARTIFACT_VOLUME_NAME
        result["artifact_persisted"] = True
    else:
        result["artifact_persisted"] = False
    return result


@app.function(
    gpu=DEFAULT_GPU,
    timeout=FULL_TRAINING_TIMEOUT_SECONDS,
    secrets=[Secret.from_name("huggingface-secret")],
    volumes={
        ARTIFACT_VOLUME_MOUNT: artifact_volume,
        DATA_VOLUME_MOUNT: data_volume,
    },
)
def run_smoke(
    **kwargs: Unpack[SmokeRunOptions],
) -> SmokeTrainingPayload:
    return _run_smoke_impl(**kwargs)


@app.function(
    gpu=H100_GPU,
    timeout=FULL_TRAINING_TIMEOUT_SECONDS,
    secrets=[Secret.from_name("huggingface-secret")],
    volumes={
        ARTIFACT_VOLUME_MOUNT: artifact_volume,
        DATA_VOLUME_MOUNT: data_volume,
    },
)
def run_smoke_h100(
    **kwargs: Unpack[SmokeRunOptions],
) -> SmokeTrainingPayload:
    return _run_smoke_impl(**kwargs)


# reason: write text and command share train main's state; extraction would misattribute row errors.
@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(  # ruff: ignore[complex-structure,too-many-branches,too-many-arguments,too-many-statements,too-many-positional-arguments]
    gpu: str = DEFAULT_GPU,
    backend: str = "unsloth",
    steps: int | None = None,
    epochs: int | None = None,
    logging_steps: int = 1,
    checkpoint_every_steps: int = 0,
    resume_from_checkpoint: str | None = None,
    max_length: int = 512,
    train_limit: int = 32,
    eval_limit: int = 16,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    lora_rank: int = 4,
    lora_alpha: int | None = None,
    packing: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    length_bucketing: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    viterbi_calibration_path: str | None = None,
    target_eval_slice: str | None = None,
    target_eval_limit: int = 4,
    target_eval_scan_multiplier: int = 32,
    dataset_id: str = "Meddies/meddies-pii",
    dataset_split: str = "train",
    eval_dataset_split: str | None = None,
    eval_dataset_id: str | None = None,
    eval_dataset_revision: str | None = None,
    dataset_revision: str | None = None,
    model_id: str = "LiquidAI/LFM2.5-350M-Base",
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    train_config: str = "train",
    eval_config: str = "test",
    trust_remote_code: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    allow_same_local_eval: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    artifact_root: str | None = None,
    out: str | None = None,
    raw_log: str | None = None,
    modal_url: str | None = None,
) -> None:
    kwargs: SmokeRunOptions = {"backend": backend}
    selected_gpu = gpu.upper()
    if steps is not None:
        kwargs["steps"] = steps
    if epochs is not None:
        kwargs["epochs"] = epochs
    if logging_steps != 1:
        kwargs["logging_steps"] = logging_steps
    if checkpoint_every_steps != 0:
        kwargs["checkpoint_every_steps"] = checkpoint_every_steps
    if resume_from_checkpoint is not None:
        kwargs["resume_from_checkpoint"] = resume_from_checkpoint
    if max_length != DEFAULT_MAX_LENGTH:
        kwargs["max_length"] = max_length
    if train_limit != DEFAULT_TRAIN_LIMIT:
        kwargs["train_limit"] = train_limit
    if eval_limit != DEFAULT_EVAL_LIMIT:
        kwargs["eval_limit"] = eval_limit
    if batch_size != DEFAULT_BATCH_SIZE:
        kwargs["batch_size"] = batch_size
    if gradient_accumulation_steps != 1:
        kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps
    if lora_rank != DEFAULT_LORA_RANK:
        kwargs["lora_rank"] = lora_rank
    if lora_alpha is not None:
        kwargs["lora_alpha"] = lora_alpha
    if packing:
        kwargs["packing"] = packing
    if length_bucketing:
        kwargs["length_bucketing"] = length_bucketing
    if viterbi_calibration_path is not None:
        kwargs["viterbi_calibration_path"] = viterbi_calibration_path
    if target_eval_slice is not None:
        kwargs["target_eval_slice"] = target_eval_slice
        kwargs["target_eval_limit"] = target_eval_limit
        kwargs["target_eval_scan_multiplier"] = target_eval_scan_multiplier
    if dataset_id != "Meddies/meddies-pii":
        kwargs["dataset_id"] = dataset_id
    if dataset_split != "train":
        kwargs["dataset_split"] = dataset_split
    if eval_dataset_split is not None:
        kwargs["eval_dataset_split"] = eval_dataset_split
    if eval_dataset_id is not None:
        kwargs["eval_dataset_id"] = eval_dataset_id
    if eval_dataset_revision is not None:
        kwargs["eval_dataset_revision"] = eval_dataset_revision
    if dataset_revision is not None:
        kwargs["dataset_revision"] = dataset_revision
    if model_id != "LiquidAI/LFM2.5-350M-Base":
        kwargs["model_id"] = model_id
    if model_revision is not None:
        kwargs["model_revision"] = model_revision
    if tokenizer_id is not None:
        kwargs["tokenizer_id"] = tokenizer_id
    if tokenizer_revision is not None:
        kwargs["tokenizer_revision"] = tokenizer_revision
    if train_config != "train":
        kwargs["train_config"] = train_config
    if eval_config != "test":
        kwargs["eval_config"] = eval_config
    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
    if allow_same_local_eval:
        kwargs["allow_same_local_eval"] = allow_same_local_eval
    if artifact_root is not None:
        kwargs["artifact_root"] = artifact_root
    if selected_gpu in {"H100", "H100-80GB"}:
        remote = run_smoke_h100
    elif selected_gpu in {"A10", "A10G"}:
        remote = run_smoke
    else:
        msg = f"Unsupported GPU {gpu!r}; expected A10 or H100"
        raise ValueError(msg)
    result = remote.remote(**kwargs)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if out is not None:
        command_kwargs = _command_kwargs(selected_gpu, kwargs)
        artifact = _artifact_payload(
            result=result,
            command_kwargs=command_kwargs,
            out=out,
            raw_log=raw_log,
            modal_url=modal_url,
            modal_profile=os.environ.get("MODAL_PROFILE"),
            artifact_root=artifact_root,
            artifact_volume=(ARTIFACT_VOLUME_NAME if _uses_artifact_volume(artifact_root) else None),
            artifact_persisted=(bool(result.get("artifact_persisted")) if artifact_root is not None else None),
        )
        destination = Path(out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(rendered, end="")
