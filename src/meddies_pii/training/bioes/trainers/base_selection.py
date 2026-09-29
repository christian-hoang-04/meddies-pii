"""Fail-closed preparation for the four-base LFM2.5 selection gate.

This module deliberately renders and validates a launch contract.  It does not
load weights at import time and does not allocate a GPU.  Remote execution is
owned by the Modal entrypoint and must pass its two explicit launch gates.
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TypedDict

import torch
from torch import nn

from meddies_pii.training.bioes.data.artifacts import load_pretrained

from .config import LORA_TARGET_MODULES

GATE_SEED = 3407
NUM_LABELS = 37
HEAD_HIDDEN_SIZE = 1024
H100_GPU = "H100!"
PROBE_BUDGET_USD = 3.0
FULL_RUN_BUDGET_USD = 10.0
MODAL_H100_USD_PER_SECOND = 0.001097
"""The quoted lane rate is the all-in allocation.

One H100 plus the configured four physical CPUs and 64 GiB memory.  The historical name is retained for compatibility with
the gate config.

"""
PACKED_CONFIGS: dict[str, dict[str, str]] = {
    "packed": {
        "path": "packed/data/train-*",
        "causal_tokenizer_sha256": "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd",
        "encoder_tokenizer_sha256": "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711",
        "parity": "required_zero_mismatches",
    },
}


@dataclass(frozen=True, slots=True)
class BaseCandidate:
    key: str
    model_id: str
    revision: str
    lane: str
    tokenizer_config: str
    trust_remote_code: bool
    loader: str


CANDIDATES: dict[str, BaseCandidate] = {
    "base230": BaseCandidate(
        key="base230",
        model_id="LiquidAI/LFM2.5-230M-Base",
        revision="9d2be5519834990d30996f878b6771cccbd24f2c",
        lane="230",
        tokenizer_config="packed",
        trust_remote_code=False,
        loader="auto_model",
    ),
    "encoder230": BaseCandidate(
        key="encoder230",
        model_id="LiquidAI/LFM2.5-Encoder-230M",
        revision="0b649ad0c684378b03d4d8304f7577a662ab89bc",
        lane="230",
        tokenizer_config="packed",
        trust_remote_code=True,
        loader="masked_lm_body",
    ),
    "encoder350": BaseCandidate(
        key="encoder350",
        model_id="LiquidAI/LFM2.5-Encoder-350M",
        revision="b886781f7c6f10ca9b7096e21b83e30a073c2f39",
        lane="350",
        tokenizer_config="packed",
        trust_remote_code=True,
        loader="masked_lm_body",
    ),
    "pii350": BaseCandidate(
        key="pii350",
        model_id="LiquidAI/LFM2.5-Encoder-350M-PII-Detector",
        revision="b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2",
        lane="350",
        tokenizer_config="packed",
        trust_remote_code=True,
        loader="token_classifier_body",
    ),
}


@dataclass(frozen=True, slots=True)
class LoadedBody:
    candidate: BaseCandidate
    body: nn.Module
    loading_info: Mapping[str, Sequence[str]]


@dataclass(frozen=True, slots=True)
class LoraEvidence:
    logical_targets: tuple[str, ...]
    resolved_tensor_names: tuple[str, ...]
    resolved_module_count: int
    trainable_parameter_count: int
    parameter_dtypes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryContract:
    output_hidden_states: bool
    gradient_checkpointing: bool
    use_cache: bool
    checkpointing_kwargs: dict[str, bool]


@dataclass(frozen=True, slots=True)
class BatchProbePlan:
    discovery_batches: tuple[int, ...]
    oom_fallback_batches: tuple[int, ...]
    discovery_steps: int
    steps: dict[str, int]


class CandidateConfig(TypedDict):
    key: str
    model_id: str
    revision: str
    lane: str
    tokenizer_config: str
    trust_remote_code: bool
    loader: str
    load: dict[str, object]


class FrameworkConfig(TypedDict):
    backend: str
    dtype: str
    torch_compile: bool
    flash_attention_2: str
    unsloth: bool
    attention_implementation: str
    fused_adamw: bool
    gradient_checkpointing: str
    output_hidden_states: bool
    use_cache: bool
    runtime_precision: str
    packing_max_length: int
    gradient_accumulation_steps: int
    june_baseline_deviations: list[str]


class LoraConfig(TypedDict):
    rank: int
    alpha: int
    alpha_to_rank: float
    dropout: float
    task_type: str
    target_modules: list[str]


class ProbeLaneConfig(TypedDict):
    gpu: str
    models: list[str]
    max_estimated_cost_usd: float
    modal_profile: str


class FullRunConfig(TypedDict):
    gpu: str
    max_estimated_cost_usd: float
    eval_rows: int
    telemetry: str


class OptimizerConfig(TypedDict):
    name: str
    lr: float
    fused: bool
    gradient_accumulation_steps: int


class ProbeProtocol(TypedDict):
    cold_steps: int
    warmup_steps: int
    measured_steps: int
    anchor_batch: int
    oom_fallback: list[int]
    discovery_steps: int
    discovery_batches: list[int]
    metric: str
    vrm_occupancy: str
    telemetry: list[str]


class AdamWKwargs(TypedDict):
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    fused: bool


class GateConfig(TypedDict):
    schema_version: int
    seed: int
    models: dict[str, CandidateConfig]
    packed_configs: dict[str, dict[str, str]]
    framework: FrameworkConfig
    lora: LoraConfig
    optimizer: OptimizerConfig
    probe_lanes: dict[str, ProbeLaneConfig]
    full_runs: dict[str, FullRunConfig]
    probe_protocol: ProbeProtocol


@dataclass(slots=True)
class CostLedger:
    budget_usd: float
    rate_usd_per_second: float
    elapsed_seconds: float = 0.0

    @property
    def estimated_cost_usd(self) -> float:
        return self.elapsed_seconds * self.rate_usd_per_second

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.estimated_cost_usd

    def add_elapsed(self, seconds: float) -> None:
        if seconds < 0:
            msg = "elapsed seconds must be non-negative"
            raise ValueError(msg)
        self.elapsed_seconds += seconds
        require_estimated_cost_within_budget(self.estimated_cost_usd, self.budget_usd)


def build_meddies_head(*, seed: int = GATE_SEED) -> nn.Sequential:
    """Reset the global torch seed immediately before identical head creation.

    Returns:
        A freshly seeded dropout-and-linear head in bfloat16. Reseeding here is what makes two
        calls with the same seed produce identical initial weights.

    """
    torch.manual_seed(seed)
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(HEAD_HIDDEN_SIZE, NUM_LABELS)).to(dtype=torch.bfloat16)


def _normalized_loading_info(value: object) -> Mapping[str, Sequence[str]]:
    if not isinstance(value, Mapping):
        msg = "model loader did not return loading information"
        raise RuntimeError(msg)
    result: dict[str, Sequence[str]] = {}
    for key in ("missing_keys", "mismatched_keys", "unexpected_keys"):
        entries = value.get(key, ())
        if not isinstance(entries, Collection) or isinstance(entries, (Mapping, str, bytes)):
            msg = f"model loading info {key} is invalid"
            raise RuntimeError(msg)
        names: list[str] = []
        for entry in entries:
            if key != "mismatched_keys" and isinstance(entry, str):
                names.append(entry)
            elif key == "mismatched_keys" and isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[0], str):  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
                names.append(entry[0])
            else:
                msg = f"model loading info {key} is invalid"
                raise RuntimeError(msg)
        result[key] = tuple(sorted(names))
    return result


def _assert_complete_body(info: Mapping[str, Sequence[str]], body_prefix: str = "") -> None:
    failed = [
        name
        for key in ("missing_keys", "mismatched_keys")
        for name in info.get(key, ())
        if not body_prefix or name.startswith(body_prefix)
    ]
    if failed:
        msg = f"model body tensors were not loaded exactly: {sorted(failed)}"
        raise RuntimeError(msg)


def load_standard_body(
    auto_model: object,
    candidate: BaseCandidate,
    *,
    attn_implementation: str | None = None,
    cache_dir: str | None = None,
) -> LoadedBody:
    if candidate.loader != "auto_model":
        msg = f"{candidate.key} does not use AutoModel"
        raise ValueError(msg)
    kwargs: dict[str, object] = {
        "revision": candidate.revision,
        "trust_remote_code": candidate.trust_remote_code,
        "torch_dtype": torch.bfloat16,
        "output_loading_info": True,
        "cache_dir": cache_dir or os.environ.get("HF_HOME", "/cache/hf"),
        "local_files_only": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    loaded = load_pretrained(auto_model, candidate.model_id, kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[0], nn.Module):  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "AutoModel must return (model, loading_info)"
        raise RuntimeError(msg)
    info = _normalized_loading_info(loaded[1])
    _assert_complete_body(info)
    return LoadedBody(candidate, loaded[0], info)


def load_encoder_body(
    auto_model_for_masked_lm: object,
    candidate: BaseCandidate,
    *,
    attn_implementation: str | None = None,
    cache_dir: str | None = None,
) -> LoadedBody:
    if candidate.loader != "masked_lm_body":
        msg = f"{candidate.key} does not use AutoModelForMaskedLM"
        raise ValueError(msg)
    kwargs: dict[str, object] = {
        "revision": candidate.revision,
        "trust_remote_code": candidate.trust_remote_code,
        "torch_dtype": torch.bfloat16,
        "output_loading_info": True,
        "cache_dir": cache_dir or os.environ.get("HF_HOME", "/cache/hf"),
        "local_files_only": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    loaded = load_pretrained(auto_model_for_masked_lm, candidate.model_id, kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[0], nn.Module):  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "AutoModelForMaskedLM must return (model, loading_info)"
        raise RuntimeError(msg)
    wrapper, raw_info = loaded
    body = getattr(wrapper, "lfm2", None)
    if not isinstance(body, nn.Module):
        msg = "masked-LM wrapper has no .lfm2 encoder body"
        raise RuntimeError(msg)
    info = _normalized_loading_info(raw_info)
    _assert_complete_body(info, body_prefix="lfm2.")
    return LoadedBody(candidate, body, info)


def load_pii350_body(
    auto_model_for_token_classification: object,
    *,
    attn_implementation: str | None = None,
    cache_dir: str | None = None,
) -> LoadedBody:
    candidate = CANDIDATES["pii350"]
    kwargs: dict[str, object] = {
        "revision": candidate.revision,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "output_loading_info": True,
        "cache_dir": cache_dir or os.environ.get("HF_HOME", "/cache/hf"),
        "local_files_only": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    loaded = load_pretrained(auto_model_for_token_classification, candidate.model_id, kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[0], nn.Module):  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "PII token-classification loader must return (model, loading_info)"
        raise RuntimeError(msg)
    wrapper, raw_info = loaded
    body = getattr(wrapper, "lfm2", None)
    if not isinstance(body, nn.Module):
        msg = "PII wrapper has no .lfm2 encoder body"
        raise RuntimeError(msg)
    info = _normalized_loading_info(raw_info)
    _assert_complete_body(info, body_prefix="lfm2.")
    return LoadedBody(candidate, body, info)


def candidate_load_plan(candidate_key: str) -> dict[str, object]:
    candidate = CANDIDATES[candidate_key]
    return {
        **asdict(candidate),
        "torch_dtype": "bfloat16",
        "isolated_child_process": True,
        "allow_co_load_with_remote_code": False,
        "output_hidden_states": False,
        "use_cache": False,
        "gradient_checkpointing": "native_non_reentrant_required",
    }


def _resolved_lora_modules(body: nn.Module) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {target: [] for target in LORA_TARGET_MODULES}
    for name, module in body.named_modules():
        if not name:
            continue
        logical_name = name.rsplit(".", 1)[-1]
        if logical_name in resolved:
            resolved[logical_name].extend(f"{name}.{parameter}" for parameter, _ in module.named_parameters(recurse=False))
    missing = [target for target, names in resolved.items() if not names]
    if missing:
        msg = f"LoRA logical targets are missing: {missing}"
        raise RuntimeError(msg)
    return resolved


def apply_gate_lora(body: nn.Module, head: nn.Module, *, peft_module: object) -> tuple[nn.Module, LoraEvidence]:
    resolved = _resolved_lora_modules(body)
    task_type = getattr(getattr(peft_module, "TaskType", None), "FEATURE_EXTRACTION", None)
    lora_config = getattr(peft_module, "LoraConfig", None)
    get_peft_model = getattr(peft_module, "get_peft_model", None)
    if task_type is None or not callable(lora_config) or not callable(get_peft_model):
        msg = "PEFT module does not expose the required LoRA construction API"
        raise TypeError(msg)
    config = lora_config(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        task_type=task_type,
        target_modules=list(LORA_TARGET_MODULES),
        use_rslora=False,
        use_dora=False,
    )
    adapted = get_peft_model(body, config)
    if not isinstance(adapted, nn.Module):
        msg = "PEFT returned a non-module adapted model"
        raise TypeError(msg)
    for parameter in head.parameters():
        parameter.requires_grad = True
    parameters = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
    parameters.extend(parameter for parameter in head.parameters() if parameter.requires_grad)
    tensors = tuple(sorted(name for names in resolved.values() for name in names))
    return adapted, LoraEvidence(
        logical_targets=LORA_TARGET_MODULES,
        resolved_tensor_names=tensors,
        resolved_module_count=sum(len(names) for names in resolved.values()),
        trainable_parameter_count=sum(parameter.numel() for parameter in parameters),
        parameter_dtypes=tuple(sorted({str(parameter.dtype) for parameter in parameters})),
    )


def adamw_kwargs(*, fused: bool) -> AdamWKwargs:
    return {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "fused": fused,
    }


def enable_native_gradient_checkpointing(body: nn.Module) -> MemoryContract:
    config = getattr(body, "config", None)
    enable = getattr(body, "gradient_checkpointing_enable", None)
    if config is None or not hasattr(config, "use_cache"):
        msg = "model body has no configurable use_cache setting"
        raise RuntimeError(msg)
    if not callable(enable):
        msg = "model body does not expose native gradient checkpointing"
        raise RuntimeError(msg)
    config.use_cache = False
    checkpointing_kwargs = {"use_reentrant": False}
    try:
        enable(gradient_checkpointing_kwargs=checkpointing_kwargs)
    except Exception as error:
        msg = "native gradient checkpointing could not be enabled"
        raise RuntimeError(msg) from error
    if getattr(body, "is_gradient_checkpointing", False) is not True:
        msg = "native gradient checkpointing did not become active"
        raise RuntimeError(msg)
    if getattr(config, "use_cache", None) is not False:
        msg = "native gradient checkpointing did not disable use_cache"
        raise RuntimeError(msg)
    return MemoryContract(
        output_hidden_states=False,
        gradient_checkpointing=True,
        use_cache=False,
        checkpointing_kwargs=checkpointing_kwargs,
    )


def probe_batch_plan() -> BatchProbePlan:
    return BatchProbePlan(
        discovery_batches=(224,),
        oom_fallback_batches=(192, 160, 128, 96, 64),
        discovery_steps=2,
        steps={"cold": 1, "warmup": 2, "measured": 7},
    )


def select_fused_adamw(unfused_tokens_per_second: float, fused_tokens_per_second: float) -> bool:
    """Decide whether a measured fused AdamW run is fast enough to adopt.

    Call this with the two throughput numbers a base-selection probe measured. Fused wins only
    at a 3% margin, so measurement noise does not flip the optimizer between runs.

    Returns:
        True when the fused measurement clears the margin.

    Raises:
        ValueError: when either throughput is not positive.

    """
    if unfused_tokens_per_second <= 0 or fused_tokens_per_second <= 0:
        msg = "throughput measurements must be positive"
        raise ValueError(msg)
    return fused_tokens_per_second >= unfused_tokens_per_second * 1.03


def require_estimated_cost_within_budget(estimated_cost_usd: float, budget_usd: float) -> None:
    if estimated_cost_usd < 0 or budget_usd <= 0:
        msg = "cost values are invalid"
        raise ValueError(msg)
    if estimated_cost_usd > budget_usd:
        msg = f"estimated cost ${estimated_cost_usd:.2f} exceeds budget ${budget_usd:.2f}"
        raise RuntimeError(msg)


def require_execute_confirmation(*, execute: bool, confirmation: str) -> None:
    if not execute or confirmation != "LAUNCH_H100_BASE_SELECTION":
        msg = "H100 allocation requires --execute and confirmation LAUNCH_H100_BASE_SELECTION"
        raise RuntimeError(msg)


def finalization_reserve_seconds(*, final_eval_seconds: float, save_seconds: float) -> float:
    """Size the wall-clock reserve a run must keep for its final evaluation and save.

    Call this when planning a budget so the run is not stopped between its last optimizer step
    and its persisted checkpoint. The reserve carries headroom over both measurements and never
    falls below ten minutes.

    Returns:
        Seconds to hold back from the training budget.

    Raises:
        ValueError: when either measurement is negative.

    """
    if final_eval_seconds < 0 or save_seconds < 0:
        msg = "finalization measurements must be non-negative"
        raise ValueError(msg)
    return max(600.0, 1.25 * final_eval_seconds + 1.5 * save_seconds + 120.0)


def checkpoint_payload(
    *,
    manifest_digests: Mapping[str, str],
    data_cursor: int,
    rng_state: Mapping[str, object],
) -> dict[str, object]:
    """Build the resumable state a base-selection checkpoint has to carry.

    Call this at every save point. The three fields together are what a resume needs: where the
    data cursor stopped, which manifests the run was reading, and the sampler state to restore.

    Returns:
        The checkpoint body, with both mappings copied so later mutation cannot reach it.

    Raises:
        ValueError: when the cursor is negative or no manifest digest is given.

    """
    if data_cursor < 0 or not manifest_digests:
        msg = "checkpoint needs a cursor and manifest digests"
        raise ValueError(msg)
    return {
        "data_cursor": data_cursor,
        "manifest_digests": dict(manifest_digests),
        "rng_state": dict(rng_state),
    }


def render_gate_config() -> GateConfig:
    models: dict[str, CandidateConfig] = {
        key: {
            "key": candidate.key,
            "model_id": candidate.model_id,
            "revision": candidate.revision,
            "lane": candidate.lane,
            "tokenizer_config": candidate.tokenizer_config,
            "trust_remote_code": candidate.trust_remote_code,
            "loader": candidate.loader,
            "load": candidate_load_plan(key),
        }
        for key, candidate in CANDIDATES.items()
    }
    full_runs: dict[str, FullRunConfig] = {
        key: {
            "gpu": H100_GPU,
            "max_estimated_cost_usd": FULL_RUN_BUDGET_USD,
            "eval_rows": 1700,
            "telemetry": "jsonl_per_step",
        }
        for key in CANDIDATES
    }
    return {
        "schema_version": 1,
        "seed": GATE_SEED,
        "models": models,
        "packed_configs": PACKED_CONFIGS,
        "framework": {
            "backend": "transformers_peft",
            "dtype": "bfloat16",
            "torch_compile": False,
            "flash_attention_2": "disabled_for_base_selection",
            "unsloth": False,
            "attention_implementation": "sdpa",
            "fused_adamw": False,
            "gradient_checkpointing": "native_non_reentrant_required",
            "output_hidden_states": False,
            "use_cache": False,
            "runtime_precision": "bf16_16bit",
            "packing_max_length": 8192,
            "gradient_accumulation_steps": 1,
            "june_baseline_deviations": ["transformers_peft_native_checkpointing_replaces_unsloth"],
        },
        "lora": {
            "rank": 64,
            "alpha": 128,
            "alpha_to_rank": 2.0,
            "dropout": 0.0,
            "task_type": "FEATURE_EXTRACTION",
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-4,
            "fused": False,
            "gradient_accumulation_steps": 1,
        },
        "probe_lanes": {
            "230": {
                "gpu": H100_GPU,
                "models": ["base230", "encoder230"],
                "max_estimated_cost_usd": PROBE_BUDGET_USD,
                "modal_profile": "hahuyhoang411",
            },
            "350": {
                "gpu": H100_GPU,
                "models": ["encoder350", "pii350"],
                "max_estimated_cost_usd": PROBE_BUDGET_USD,
                "modal_profile": "hahuyhoang411",
            },
        },
        "full_runs": full_runs,
        "probe_protocol": {
            "cold_steps": 1,
            "warmup_steps": 2,
            "measured_steps": 7,
            "anchor_batch": 224,
            "oom_fallback": [192, 160, 128, 96, 64],
            "discovery_steps": 2,
            "discovery_batches": [224],
            "metric": "median_real_bioes_tokens_per_second",
            "vrm_occupancy": "stable_95_to_100_percent_allowed_not_selected",
            "telemetry": [
                "gpu_sm_utilization",
                "gpu_memory_controller_utilization",
                "gpu_power_watts",
                "vram_allocated_bytes",
                "vram_reserved_bytes",
                "vram_total_bytes",
                "real_bioes_tokens_per_second",
            ],
        },
    }


def render_gate_json() -> str:
    return json.dumps(render_gate_config(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    print(render_gate_json(), end="")
