"""One-step, fail-closed Unsloth compatibility contract for Encoder350."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base_selection import CANDIDATES, GATE_SEED
from .config import LORA_TARGET_MODULES
from .full_run import PACKED_MAX_LENGTH
from .pins import (
    PACKED_DATASET_ID,
    PACKED_DATASET_REVISION,
    PACKED_MANIFEST_SHA256,
    PACKED_SHARD_COUNT,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

PROBE_CONFIRMATION = "LAUNCH_H100_ENCODER350_UNSLOTH_2026_7_4_COMPAT"
PROBE_CANDIDATE = "encoder350"
PROBE_BATCH_SIZE = 192
PROBE_STEPS = 1
PROBE_HARD_TIMEOUT_SECONDS = 900
PROBE_RUNTIME_PACKAGES = {
    "torch": "2.10.0",
    "transformers": "5.2.0",
    "peft": "0.19.1",
    "unsloth": "2026.7.4",
    "unsloth_zoo": "2026.7.4",
}
HF_CACHE_ROOT = "/cache/hf"


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render_probe() -> dict[str, Any]:
    candidate = CANDIDATES[PROBE_CANDIDATE]
    body: dict[str, Any] = {
        "candidate_key": PROBE_CANDIDATE,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "backend": "unsloth",
        "runtime_packages": dict(PROBE_RUNTIME_PACKAGES),
        "batch_size": PROBE_BATCH_SIZE,
        "optimizer_steps": PROBE_STEPS,
        "seed": GATE_SEED,
        "loader": "FastModel.from_pretrained(auto_model=AutoModelForMaskedLM)",
        "packed_dataset": {
            "id": PACKED_DATASET_ID,
            "revision": PACKED_DATASET_REVISION,
            "config": "packed",
            "manifest_sha256": PACKED_MANIFEST_SHA256,
            "shard_count": PACKED_SHARD_COUNT,
            "max_length": PACKED_MAX_LENGTH,
            "segment_isolation": True,
        },
        "training": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": list(LORA_TARGET_MODULES),
            "task_type": "FEATURE_EXTRACTION",
            "gradient_checkpointing": "unsloth",
            "bf16": True,
            "optimizer": "adamw",
            "fused_adamw": False,
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "hard_timeout_seconds": PROBE_HARD_TIMEOUT_SECONDS,
    }
    return {**body, "config_digest": _digest(body)}


def validate_probe_contract(contract: Mapping[str, Any]) -> None:
    expected = render_probe()
    for key, value in expected.items():
        if contract.get(key) != value:
            msg = f"Encoder350 Unsloth compatibility probe mismatch: {key}"
            raise ValueError(msg)


def require_probe_execute(contract: Mapping[str, Any], *, execute: bool, confirmation: str) -> None:
    validate_probe_contract(contract)
    if not execute or confirmation != PROBE_CONFIRMATION:
        msg = "Encoder350 Unsloth compatibility probe requires --execute and exact confirmation"
        raise RuntimeError(msg)


def require_runtime_package_tuple(observed: Mapping[str, str]) -> dict[str, str]:
    expected = dict(PROBE_RUNTIME_PACKAGES)
    if dict(observed) != expected:
        msg = f"Unsloth compatibility package tuple mismatch: expected={expected}, observed={dict(observed)}"
        raise RuntimeError(msg)
    return expected


def resolve_encoder350_snapshot(snapshot_download: Callable[..., str]) -> str:
    candidate = CANDIDATES[PROBE_CANDIDATE]
    snapshot = snapshot_download(
        repo_id=candidate.model_id,
        revision=candidate.revision,
        cache_dir=HF_CACHE_ROOT,
        local_files_only=True,
    )
    if not isinstance(snapshot, str) or not Path(snapshot).is_absolute():
        msg = "Encoder350 snapshot resolver returned no absolute local path"
        raise RuntimeError(msg)
    return snapshot


def render_command() -> str:
    return (
        "MODAL_PROFILE=huyhoang041100 uv run modal run --timestamps -m "
        "meddies_pii.training.bioes.modal.encoder350_unsloth_compat_probe "
        f"--execute --confirmation {PROBE_CONFIRMATION}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-config", action="store_true")
    args = parser.parse_args()
    if args.render_config:
        print(json.dumps(render_probe(), indent=2, sort_keys=True))
    else:
        parser.error("use --render-config")


if __name__ == "__main__":
    _main()
