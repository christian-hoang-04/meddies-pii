from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
# ruff: file-ignore[try-except-in-loop]
# reason: a per-package version inventory: a package that is absent records None and the sweep
# reason: continues. Hoisting the try would abort the whole inventory on the first missing package.
import json
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    normalize_viterbi_transition_biases,
)
from meddies_pii.training.bioes.data.artifacts import ProbeArtifacts, build_hf_artifacts, build_unsloth_artifacts
from meddies_pii.training.bioes.data.manifest import row_hash_from_parts

if TYPE_CHECKING:
    from meddies_pii.training.bioes.data.preparation import PreparedRow

    from .config import SmokeTrainingConfig


def _load_viterbi_calibration_biases(path: str) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = "Viterbi calibration artifact must be a JSON object"
        raise ValueError(msg)
    if "operating_points" in payload:
        operating_points = payload["operating_points"]
        if not isinstance(operating_points, dict):
            msg = "Viterbi calibration operating_points must be an object"
            raise ValueError(msg)
        default_point = operating_points.get("default")
        if not isinstance(default_point, dict):
            msg = "Viterbi calibration must contain operating_points.default"
            raise ValueError(msg)
        biases = default_point.get("biases")
    else:
        biases = payload.get("biases", payload)
    if not isinstance(biases, dict):
        msg = "Viterbi calibration biases must be an object"
        raise ValueError(msg)
    return normalize_viterbi_transition_biases(biases)


def _resolve_viterbi_transition_biases(
    *,
    transition_biases: dict[str, float] | None,
    calibration_path: str | None,
) -> dict[str, float] | None:
    if transition_biases is not None and calibration_path is not None:
        msg = "Specify either viterbi_transition_biases or viterbi_calibration_path, not both"
        raise ValueError(msg)
    if calibration_path is not None:
        return _load_viterbi_calibration_biases(calibration_path)
    if transition_biases is not None:
        return normalize_viterbi_transition_biases(transition_biases)
    return None


def _build_artifacts(config: SmokeTrainingConfig) -> ProbeArtifacts:
    tokenizer_id = getattr(config, "tokenizer_id", None)
    tokenizer_revision = getattr(config, "tokenizer_revision", None)
    model_revision = getattr(config, "model_revision", None)
    trust_remote_code = bool(getattr(config, "trust_remote_code", False))
    entity_labels = ENTITY_LABELS
    if config.backend == "hf":
        return build_hf_artifacts(
            config.model_id,
            entity_labels,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )
    if config.backend == "unsloth":
        return build_unsloth_artifacts(
            config.model_id,
            entity_labels,
            max_seq_length=config.max_length,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )
    msg = f"Unsupported backend: {config.backend}"
    raise ValueError(msg)


def _package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "transformers",
        "datasets",
        "unsloth",
        "unsloth_zoo",
        "liger-kernel",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    versions["torch_cuda"] = getattr(torch.version, "cuda", None)
    return versions


def _cuda_memory_report(device: str) -> dict[str, float | str | None]:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {
            "device": str(device),
            "available": False,
            "allocated_gb": None,
            "reserved_gb": None,
            "max_allocated_gb": None,
            "max_reserved_gb": None,
        }
    return {
        "device": torch.cuda.get_device_name(torch.device(device)),
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def _row_hash(row: PreparedRow) -> str:
    return row_hash_from_parts(uid=row.uid, raw=row.raw, spans=row.parsed.spans)


def _optimizer_for_config(
    model: torch.nn.Module,
    config: SmokeTrainingConfig,
    *,
    device: torch.device,
) -> torch.optim.Optimizer:
    if config.fused_adamw and device.type != "cuda":
        msg = "fused_adamw=True requires a CUDA device"
        raise ValueError(msg)
    kwargs: dict[str, Any] = {}
    if config.fused_adamw:
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        **kwargs,
    )
