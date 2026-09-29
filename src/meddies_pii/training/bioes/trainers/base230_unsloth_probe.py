"""Pure contract for the two fair Base230 Unsloth throughput cells."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from math import floor
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

    from meddies_pii.training.bioes.data.artifacts import UnslothModelApi

PROBE_CONFIRMATION = "LAUNCH_H100_BASE230_UNSLOTH_PROBE"
PROBE_CANDIDATE = "base230"
PROBE_BATCHES = (224, 240)
PROBE_STEPS = 10
PROBE_RATE_USD_PER_SECOND = 0.001097
PROBE_TARGET_ACTUAL_USD = 3.0
PROBE_BILLING_OVERHEAD_RESERVE = 1.20
PROBE_MAX_LIVE_ESTIMATE_USD = 2.50
PROBE_HARD_TIMEOUT_SECONDS = floor(PROBE_MAX_LIVE_ESTIMATE_USD / PROBE_RATE_USD_PER_SECOND)
PROBE_SHUTDOWN_RESERVE_SECONDS = 120
PROBE_CHILD_DEADLINE_SECONDS = PROBE_HARD_TIMEOUT_SECONDS - PROBE_SHUTDOWN_RESERVE_SECONDS
HF_CACHE_ROOT = "/cache/hf"


@dataclass(frozen=True, slots=True)
class Base230UnslothProbe:
    candidate_key: str
    model_id: str
    model_revision: str
    backend: str
    load_in_16bit: bool
    use_gradient_checkpointing: str
    seed: int
    batch_size: int
    optimizer_steps: int
    packed_dataset: Mapping[str, Any]
    training: Mapping[str, Any]
    cost: Mapping[str, Any]
    config_digest: str


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render_probe(batch_size: int) -> dict[str, Any]:
    if batch_size not in PROBE_BATCHES:
        msg = "Base230 Unsloth probe batch must be 224 or 240"
        raise ValueError(msg)
    candidate = CANDIDATES[PROBE_CANDIDATE]
    body: dict[str, Any] = {
        "candidate_key": PROBE_CANDIDATE,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "backend": "unsloth",
        "load_in_16bit": True,
        "torch_dtype": "bfloat16",
        "use_gradient_checkpointing": "unsloth",
        "seed": GATE_SEED,
        "batch_size": batch_size,
        "optimizer_steps": PROBE_STEPS,
        "packed_dataset": {
            "id": PACKED_DATASET_ID,
            "revision": PACKED_DATASET_REVISION,
            "config": "packed",
            "manifest_sha256": PACKED_MANIFEST_SHA256,
            "shard_count": PACKED_SHARD_COUNT,
            "max_length": PACKED_MAX_LENGTH,
        },
        "training": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": list(LORA_TARGET_MODULES),
            "bf16": True,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw",
            "fused_adamw": False,
            "lr": 1e-4,
            "schedule": "constant",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "torch_compile": False,
            "contamination_check": "exact_zero",
        },
        "cost": {
            "rate_usd_per_second": PROBE_RATE_USD_PER_SECOND,
            "target_actual_usd": PROBE_TARGET_ACTUAL_USD,
            "billing_overhead_reserve": PROBE_BILLING_OVERHEAD_RESERVE,
            "maximum_live_estimate_usd": PROBE_MAX_LIVE_ESTIMATE_USD,
            "hard_timeout_seconds": PROBE_HARD_TIMEOUT_SECONDS,
            "shutdown_commit_reserve_seconds": PROBE_SHUTDOWN_RESERVE_SECONDS,
            "child_deadline_seconds": PROBE_CHILD_DEADLINE_SECONDS,
            "stop_before_next_step": True,
        },
    }
    return {**body, "config_digest": _digest(body)}


def render_all_probes() -> dict[str, dict[str, Any]]:
    return {str(batch): render_probe(batch) for batch in PROBE_BATCHES}


def validate_probe_contract(contract: Mapping[str, Any]) -> None:
    batch_size = contract.get("batch_size")
    if not isinstance(batch_size, int) or batch_size not in PROBE_BATCHES:
        msg = "Base230 Unsloth probe has an invalid batch"
        raise ValueError(msg)
    expected = render_probe(batch_size)
    for key in expected:
        if contract.get(key) != expected[key]:
            msg = f"Base230 Unsloth probe mismatch: {key}"
            raise ValueError(msg)
    if contract["candidate_key"] != PROBE_CANDIDATE:
        msg = "Base230 probe cannot select another candidate"
        raise ValueError(msg)
    if contract["backend"] != "unsloth":
        msg = "Base230 probe cannot use a native PEFT fallback"
        raise ValueError(msg)
    if contract["optimizer_steps"] != PROBE_STEPS:
        msg = "Base230 probe requires exactly ten optimizer steps"
        raise ValueError(msg)


def require_probe_execute(contract: Mapping[str, Any], *, execute: bool, confirmation: str) -> None:
    validate_probe_contract(contract)
    if not execute or confirmation != PROBE_CONFIRMATION:
        msg = "Base230 Unsloth probe requires --execute and exact confirmation"
        raise RuntimeError(msg)


def resolve_candidate_snapshot(candidate_key: str, snapshot_download: Callable[..., str]) -> str:
    """Resolve one exact prewarmed model snapshot without consulting uncached ``main``.

    Returns:
        The absolute local path of the pinned snapshot for `candidate`.

    Raises:
        ValueError: if `candidate` is not one of the known probe candidates.
        RuntimeError: if the resolver returns a path that is not absolute and local, which would
            mean the snapshot was not already prewarmed in this image.

    """
    if candidate_key not in CANDIDATES:
        msg = "snapshot resolver requires one known candidate"
        raise ValueError(msg)
    candidate = CANDIDATES[candidate_key]
    snapshot = snapshot_download(
        repo_id=candidate.model_id,
        revision=candidate.revision,
        cache_dir=HF_CACHE_ROOT,
        local_files_only=True,
    )
    if not isinstance(snapshot, str) or not Path(snapshot).is_absolute():
        msg = "candidate snapshot resolver returned no absolute local path"
        raise RuntimeError(msg)
    return snapshot


def resolve_base230_snapshot(snapshot_download: Callable[..., str]) -> str:
    """Backward-compatible Base230 specialization of the exact snapshot resolver.

    Returns:
        The absolute local path of the pinned Base230 snapshot. Refusals are the resolver's own;
        this wrapper adds none.

    """
    return resolve_candidate_snapshot(PROBE_CANDIDATE, snapshot_download)


def build_unsloth_base230(
    fast_language_model: UnslothModelApi,
    *,
    dtype: object,
    tokenizer_name: str,
) -> tuple[Any, Any]:
    """Construct Base230 only through the allowed Unsloth seam.

    Unsloth 2026.7.4 otherwise injects ``finetune_vision_layers=False`` on this new-model route, which scopes an explicit
    target list and drops LFM2's convolutional ``in_proj``. Base230 has no vision tower, but True disables that scoping
    branch; the other three flags keep every explicit attention and MLP target eligible for LoRA attachment.

    Returns:
        The constructed Base230 model and its tokenizer, from the Unsloth route only.

    """
    candidate = CANDIDATES[PROBE_CANDIDATE]
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=candidate.model_id,
        revision=candidate.revision,
        tokenizer_name=tokenizer_name,
        max_seq_length=PACKED_MAX_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        dtype=dtype,
        full_finetuning=False,
        fast_inference=False,
        trust_remote_code=candidate.trust_remote_code,
    )
    model = fast_language_model.get_peft_model(
        model,
        r=64,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        use_dora=False,
        random_state=GATE_SEED,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )
    return model, tokenizer


def render_command(batch_size: int) -> str:
    render_probe(batch_size)
    return f"python -m meddies_pii.training.bioes.trainers.base230_unsloth_probe --render-config --batch-size {batch_size}"


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-config", action="store_true")
    parser.add_argument("--render-all", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.render_all:
        print(json.dumps(render_all_probes(), indent=2, sort_keys=True))
    elif args.render_config and args.batch_size is not None:
        print(json.dumps(render_probe(args.batch_size), indent=2, sort_keys=True))
    else:
        parser.error("use --render-all or --render-config --batch-size {224,240}")


if __name__ == "__main__":
    _main()
