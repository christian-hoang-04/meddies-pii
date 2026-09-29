"""Training command and configuration for the setup preview."""

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.training.bioes.reports.json_narrowing import int_like, map_at
from meddies_pii.training.bioes.trainers.config import LORA_TARGET_MODULES

if TYPE_CHECKING:
    from collections.abc import Mapping

DATASET_REVISION = "04be20f2c42d3f92b022edefbfb4d343fef78b2c"
PLANNED_STEPS = 120
CHECKPOINT_EVERY_STEPS = 10
RUN_SLUG = "20260604_h100_8192_r128a256_pack_bs128_120step_ckpt10"
LOCAL_RUN_SLUG = "modal_openmedical_h100_bioes_8192_r128a256_pack_bs128_120step_ckpt10_20260604"
REPORT_SLUG = "lfm25-bioes-full-8192-r128-a256-pack-bs128"
ARTIFACT_VOLUME = "meddies-pii-bioes-artifacts"
TRAIN_ROWS = 173_652
VALIDATION_ROWS = 500

DEFAULT_DATA_ROOT = Path("data/bioes-v2/base")
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_ROOT / f"all_lang_unique_augmented-20260530.train.{LEGACY_ARTIFACT_TOKEN}.jsonl"
DEFAULT_VALIDATION_JSONL = (
    DEFAULT_DATA_ROOT / f"all_lang_unique_augmented-20260530.validation.{LEGACY_ARTIFACT_TOKEN}.jsonl"
)
DEFAULT_SPLIT_SUMMARY_JSON = DEFAULT_DATA_ROOT / "pii-bioes.split.summary.json"
DEFAULT_DATASET_SUMMARY_JSON = DEFAULT_DATA_ROOT / "pii-bioes.summary.json"
DEFAULT_SMOKE_RESULT_JSON = Path(
    "reports/bioes-training/smoke/modal_openmedical_h100_bioes_4096_r32a64_pack_bs128_5step_20260602.json",
)
DEFAULT_OUTPUT_DIR = Path("reports/training-setups") / REPORT_SLUG


def _default_hyperparameters() -> dict[str, Any]:
    return {
        "status": "Not launched yet",
        "hardware": {
            "modal_profile": "openmedical",
            "gpu": "H100",
            "num_gpus": 1,
        },
        "model": {
            "model_id": "LiquidAI/LFM2.5-350M-Base",
            "backend": "unsloth",
            "trust_remote_code": False,
        },
        "data": {
            "dataset_id": "Meddies/meddies-pii",
            "dataset_revision": DATASET_REVISION,
            "train_config": "pii-bioes",
            "eval_config": "pii-bioes",
            "dataset_split": "train",
            "eval_dataset_split": "validation",
            "train_limit": TRAIN_ROWS,
            "train_limit_semantics": "upper_bound_before_quality_filtering",
            "eval_limit": VALIDATION_ROWS,
            "published_total_rows": TRAIN_ROWS + VALIDATION_ROWS,
        },
        "training": {
            "epochs": None,
            "steps": PLANNED_STEPS,
            "logging_steps": 1,
            "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
            "max_length": 8192,
            "batch_size": 128,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 128,
            "learning_rate": 1e-4,
            "packing": True,
            "length_bucketing": False,
            "fused_adamw": False,
        },
        "lora": {
            "rank": 128,
            "alpha": 256,
            "dropout": 0,
            "bias": "none",
            "target_modules": list(LORA_TARGET_MODULES),
            "target_groups": {
                "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "mlp": ["in_proj", "w1", "w2", "w3"],
            },
        },
        "artifacts": {
            "artifact_volume": ARTIFACT_VOLUME,
            "artifact_root": f"/artifacts/bioes/{RUN_SLUG}",
            "out": f"reports/bioes-training/runs/{LOCAL_RUN_SLUG}.json",
            "raw_log": f"reports/bioes-training/raw_logs/{LOCAL_RUN_SLUG}.log",
        },
        "runtime": {
            "seed": 3407,
            "detached": True,
            "resume_from_checkpoint": None,
        },
    }


def _flag(name: str, value: object | None = None) -> list[str]:
    flag = "--" + name.replace("_", "-")
    if value is None:
        return [flag]
    return [flag, str(value)]


def training_command(hparams: Mapping[str, Any]) -> str:
    hardware = hparams["hardware"]
    model = hparams["model"]
    data = hparams["data"]
    training = hparams["training"]
    lora = hparams["lora"]
    artifacts = hparams["artifacts"]
    raw_log = str(artifacts["raw_log"])
    preview_dir = str(Path(raw_log).parent)
    args: list[str] = [
        "env",
        f"MODAL_PROFILE={hardware['modal_profile']}",
        "uv",
        "run",
        "modal",
        "run",
        "--detach",
        "--timestamps",
        "src/meddies_pii/training/bioes/modal/train.py",
        *_flag("gpu", hardware["gpu"]),
        *_flag("backend", model["backend"]),
        *_flag("steps", training["steps"]),
        *_flag("logging_steps", training["logging_steps"]),
        *_flag("checkpoint_every_steps", training["checkpoint_every_steps"]),
        *_flag("max_length", training["max_length"]),
        *_flag("train_limit", data["train_limit"]),
        *_flag("eval_limit", data["eval_limit"]),
        *_flag("batch_size", training["batch_size"]),
        *_flag("gradient_accumulation_steps", training["gradient_accumulation_steps"]),
        *_flag("lora_rank", lora["rank"]),
        *_flag("lora_alpha", lora["alpha"]),
        *_flag("dataset_revision", data["dataset_revision"]),
        *_flag("train_config", data["train_config"]),
        *_flag("eval_config", data["eval_config"]),
        *_flag("eval_dataset_split", data["eval_dataset_split"]),
        *_flag("artifact_root", artifacts["artifact_root"]),
        *_flag("out", artifacts["out"]),
        *_flag("raw_log", raw_log),
    ]
    if training.get("epochs") is not None:
        args.extend(_flag("epochs", training["epochs"]))
    if hparams.get("runtime", {}).get("resume_from_checkpoint") is not None:
        args.extend(
            _flag(
                "resume_from_checkpoint",
                hparams["runtime"]["resume_from_checkpoint"],
            ),
        )
    if training["packing"]:
        args.extend(_flag("packing"))
    if model["trust_remote_code"]:
        args.extend(_flag("trust_remote_code"))
    shell_command = " ".join(shlex.quote(arg) for arg in args)
    return f"mkdir -p {shlex.quote(preview_dir)} && set -o pipefail; {shell_command} 2>&1 | tee {shlex.quote(raw_log)}"


def _packing_estimate(hparams: Mapping[str, Any], smoke: Mapping[str, Any]) -> dict[str, Any]:
    result = map_at(smoke, "result") or smoke
    config = map_at(result, "config")
    train_examples = int_like(result.get("train_examples") or 0)
    packed_examples = int_like(result.get("train_packed_examples") or 0)
    smoke_length = int_like(config.get("max_length") or 4096)
    target_length = int(hparams["training"]["max_length"])
    batch_size = int(hparams["training"]["batch_size"])
    train_limit = int(hparams["data"]["train_limit"])
    if not train_examples or not packed_examples or not smoke_length or not target_length:
        return {}
    target_packed = math.ceil(train_limit * (packed_examples / train_examples) * (smoke_length / target_length))
    steps_per_epoch = math.ceil(target_packed / batch_size)
    return {
        "smoke_train_examples": train_examples,
        "smoke_packed_examples": packed_examples,
        "smoke_max_length": smoke_length,
        "target_estimated_packed_units": target_packed,
        "target_estimated_steps_per_epoch": steps_per_epoch,
        "planned_steps_this_chunk": hparams["training"]["steps"],
        "checkpoint_every_steps": hparams["training"]["checkpoint_every_steps"],
        "smoke_h100_max_reserved_gb": map_at(result, "cuda_memory").get("max_reserved_gb"),
        "smoke_artifact_persisted": result.get("artifact_persisted"),
    }
