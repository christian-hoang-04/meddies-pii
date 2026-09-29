"""Exact, fail-closed contracts for four independent full BIOES runs.

The module is intentionally pure Python.  ``python -m ...full_run --render-config``
never imports Modal, CUDA, Transformers, PEFT, or Unsloth.
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: the render subcommands write their result to standard output; that output is this module's product.
import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from .base_selection import CANDIDATES, GATE_SEED, H100_GPU
from .config import DEFAULT_UNSLOTH_MECHANICS, LORA_TARGET_MODULES
from .pins import (
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

if TYPE_CHECKING:
    from collections.abc import Mapping

Backend = Literal["unsloth"]
QUALIFICATION_OPTIMIZER_STEPS = 10

FULL_RUN_CONFIRMATION = "LAUNCH_H100_BIOES_FULL_RUN"
FULL_RUN_PYTHON_VERSION = "3.11"
FULL_RUN_IMAGE_PACKAGES: tuple[str, ...] = (
    "torch==2.10.0",
    "transformers==5.2.0",
    "peft==0.19.1",
    "pyarrow==23.0.0",
    "datasets==4.3.0",
    "huggingface_hub==1.11.0",
    "nvidia-ml-py==13.590.44",
    "unsloth==2026.5.2",
    "unsloth_zoo==2026.5.1",
)
ENCODER_FULL_RUN_IMAGE_PACKAGES: tuple[str, ...] = (
    *FULL_RUN_IMAGE_PACKAGES[:-2],
    "unsloth==2026.7.4",
    "unsloth_zoo==2026.7.4",
)
FULL_RUN_TARGET_USD = 10.0
GPU_ONLY_BUDGET_USD = FULL_RUN_TARGET_USD
FULL_RUN_RATE_USD_PER_SECOND = 0.001097
FULL_RUN_HARD_TIMEOUT_SECONDS = 9_000
FULL_RUN_SHUTDOWN_RESERVE_SECONDS = 600
FULL_RUN_TRAINING_DEADLINE_SECONDS = FULL_RUN_HARD_TIMEOUT_SECONDS - FULL_RUN_SHUTDOWN_RESERVE_SECONDS
FULL_RUN_CPU_RATE_USD_PER_SECOND = 4 * 0.0000131
FULL_RUN_MEMORY_RATE_USD_PER_SECOND = 64 * 0.00000222
FULL_RUN_ALL_IN_RATE_USD_PER_SECOND = (
    FULL_RUN_RATE_USD_PER_SECOND + FULL_RUN_CPU_RATE_USD_PER_SECOND + FULL_RUN_MEMORY_RATE_USD_PER_SECOND
)
PACKED_MAX_LENGTH = 8192
RAW_SOURCE_POPULATION = 1_000_000
ACCEPTED_ROW_COUNT = 836_000
FULL_RUN_LABEL_VOCABULARY: tuple[str, ...] = (
    "O",
    "B-address",
    "I-address",
    "E-address",
    "S-address",
    "B-company_name",
    "I-company_name",
    "E-company_name",
    "S-company_name",
    "B-date",
    "I-date",
    "E-date",
    "S-date",
    "B-email_address",
    "I-email_address",
    "E-email_address",
    "S-email_address",
    "B-human_name",
    "I-human_name",
    "E-human_name",
    "S-human_name",
    "B-id_number",
    "I-id_number",
    "E-id_number",
    "S-id_number",
    "B-phone_number",
    "I-phone_number",
    "E-phone_number",
    "S-phone_number",
    "B-private_url",
    "I-private_url",
    "E-private_url",
    "S-private_url",
    "B-secret",
    "I-secret",
    "E-secret",
    "S-secret",
)
EXPECTED_FULL_BATCHES = {
    "base230": 416,
    "encoder230": 240,
    "encoder350": 192,
    "pii350": 192,
}
BASE230_OOM_FALLBACK_BATCH_SIZE = 384
BASE230_OOM_FALLBACK_PRIMARY_ACTION = "HA_AUTHORIZE_BASE230_B384_AFTER_B416_OOM"
BASE230_B416_OOM_EVIDENCE: dict[str, str | int] = {
    "candidate": "base230",
    "batch_size": 416,
    "status": "failed_oom",
    "failed_optimizer_step": 2,
    "app_id": "ap-3oMP3pkzSyphMm539IxTzl",
}
PII350_UTILIZATION_BATCH_SIZE = 256
PII350_UTILIZATION_PROFILE = "retraction"
PII350_UTILIZATION_PRIMARY_ACTION = "HA_AUTHORIZE_PII350_BATCH256_FULL_BUDGET"
PII350_CAPACITY_BATCH_SIZE = 256
PII350_CAPACITY_LORA_RANK = 128
PII350_CAPACITY_LORA_ALPHA = 256
PII350_CAPACITY_OOM_FALLBACK_BATCH_SIZE = 160
PII350_CAPACITY_PROFILE = "retraction"
PII350_CAPACITY_PRIMARY_ACTION = "HA_AUTHORIZE_PII350_R128A256_BATCH256_FULL_BUDGET"
PII350_MILESTONE_PACKED_CURSOR = 7_680
PII350_MILESTONE_EVALUATION_ROWS = 1_700
PII350_MILESTONE_ALL_IN_CREDIT_CEILING_USD = 9.80
PII350_MILESTONE_H100_BUDGET_USD = 8.20
PII350_MILESTONE_OBSERVED_ALL_IN_RUNTIME_MULTIPLIER = 1.177284
PII350_MILESTONE_HARD_TIMEOUT_SECONDS = 7_474
PII350_MILESTONE_SHUTDOWN_RESERVE_SECONDS = 600
PII350_MILESTONE_TRAINING_DEADLINE_SECONDS = (
    PII350_MILESTONE_HARD_TIMEOUT_SECONDS - PII350_MILESTONE_SHUTDOWN_RESERVE_SECONDS
)
PII350_MILESTONE_RUNS: dict[str, dict[str, str | int]] = {
    "meddies-pii": {
        "batch_size": 192,
        "optimizer_step": 40,
        "primary_action": "HA_AUTHORIZE_PII350_R128A256_B192_MILESTONE7680_FULL_BUDGET",
    },
    "anhthunguyenump": {
        "batch_size": 160,
        "optimizer_step": 48,
        "primary_action": "HA_AUTHORIZE_PII350_R128A256_B160_MILESTONE7680_FULL_BUDGET",
    },
}
PII350_SCOUT_ARMS: dict[str, dict[str, str | float]] = {
    "lr1e-4": {
        "lr": 1e-4,
        "primary_action": "HA_AUTHORIZE_PII350_R128A256_B128_LR1E4_SCOUT",
        "modal_profile": "hocdatacamp-hoangthu",
    },
    "lr2e-4": {
        "lr": 2e-4,
        "primary_action": "HA_AUTHORIZE_PII350_R128A256_B128_LR2E4_SCOUT",
        "modal_profile": "hocdatacamp-hoangthu",
    },
    "lr3e-4": {
        "lr": 3e-4,
        "primary_action": "HA_AUTHORIZE_PII350_R128A256_B128_LR3E4_SCOUT",
        "modal_profile": "hocdatacamp-hoangthu",
    },
    "lr4e-4": {
        "lr": 4e-4,
        "primary_action": "HA_AUTHORIZE_PII350_R128A256_B128_LR4E4_60STEP_SCOUT",
        "modal_profile": "meddiesresearch",
    },
    "wsd3e-4": {
        "lr": 3e-4,
        "primary_action": "HA_AUTHORIZE_PII350_R128A256_B128_WSD3E4_SCOUT",
        "modal_profile": "meddiesresearch",
        "schedule": "wsd",
        "total_steps": 60,
        "warmup_steps": 6,
        "stable_steps": 48,
        "decay_steps": 6,
        "warmup_type": "linear",
        "decay_type": "cosine",
        "min_lr_ratio": 0.0,
    },
}
M230_COMPARISON_CANDIDATES = ("base230", "encoder230")
M230_COMPARISON_BATCH_SIZES = {
    "base230": 320,
    "encoder230": 288,
}
M230_COMPARISON_PROFILES = {
    "base230": "meddies-pii",
    "encoder230": "hahuyhoang041100",
}
M230_COMPARISON_PRIMARY_ACTIONS = {
    "base230": "HA_AUTHORIZE_BASE230_B320_R64A128_FULL_BUDGET",
    "encoder230": "HA_AUTHORIZE_ENCODER230_B288_R64A128_FULL_BUDGET",
}
PII350_B192_COMPLETE_EVIDENCE: dict[str, str | int] = {
    "candidate": "pii350",
    "batch_size": 192,
    "status": "completed_ok_deadline_reached",
    "optimizer_steps": 59,
    "execution_id": "65a28933214247d483391c432f8d2847",
    "app_id": "ap-kEb6qDE1CReGVggcahetGo",
    "maximum_peak_allocated_vram_bytes": 49_283_107_328,
    "maximum_device_vram_used_bytes": 58_263_797_760,
}
FULL_RUN_MODAL_PROFILES = {
    "base230": "hahuyhoang411",
    "encoder230": "huyhoangha0411",
    "encoder350": "huyhoang041100",
    "pii350": "huyhoang041100",
}


@dataclass(frozen=True, slots=True)
class Qualification:
    status: Literal["qualified", "unqualified", "accepted_transfer", "accepted_live"]
    artifact_id: str | None
    execution_id: str | None
    selected_batch_size: int | None
    median_real_tokens_per_second: float | None
    rationale: str


QUALIFICATIONS: dict[str, Qualification] = {
    "base230": Qualification(
        "accepted_live",
        "ha-accepted-unsloth-base230-b416-fresh-full-budget",
        None,
        416,
        None,
        (
            "Ha stopped the Base230 batch 240 run after optimizer step 10 and explicitly accepted a "
            "fresh strict-Unsloth batch 416 full-budget attempt; it has no resume checkpoint or step "
            "cap, and batch 384 is manual-only after a real 416 OOM"
        ),
    ),
    "encoder230": Qualification(
        "accepted_transfer",
        "ha-accepted-native-base230-b240-transfer",
        None,
        240,
        None,
        (
            "Ha explicitly accepted transfer from the native Base230 240 cell; first full-run "
            "optimizer step is the in-budget stability check, with OOM fail-closed and no lower-batch "
            "fallback"
        ),
    ),
    "encoder350": Qualification(
        "accepted_transfer",
        "ha-accepted-native-encoder350-b192-transfer",
        "ded3bd6ec1c04ee486500308f4f05035",
        192,
        11248.277676680911,
        (
            "Ha accepted transfer from the native Transformers+PEFT Encoder350 batch 192 "
            "qualification (execution ded3bd6ec1c04ee486500308f4f05035); strict-Unsloth first "
            "full-run optimizer step is the in-budget compatibility check, and native evidence "
            "retained median 11248.277676680911 versus 6905.214375535371 at batch 200"
        ),
    ),
    "pii350": Qualification(
        "accepted_transfer",
        "ha-accepted-native-encoder350-b192-transfer",
        None,
        192,
        None,
        (
            "Ha explicitly accepted transfer from the native Encoder350 192 cell; first full-run "
            "optimizer step is the in-budget stability check, with OOM fail-closed and no lower-batch "
            "fallback"
        ),
    ),
}

BASE230_LIVE_ACCEPTANCE: dict[str, str | int] = {
    "candidate": "base230",
    "runtime_backend": "unsloth",
    "batch_size": 416,
    "decision": "fresh_full_budget_attempt_after_b240_step10",
    "acceptance_id": "ha-accepted-unsloth-base230-b416-fresh-full-budget",
    "accepted_by": "Ha",
}

MANUAL_TRANSFER_ACCEPTANCES: dict[str, Mapping[str, str | int]] = {
    "encoder230": {
        "acceptance_id": "ha-accepted-native-base230-b240-transfer",
        "qualification_evidence_candidate": "base230",
        "qualification_evidence_backend": "unsloth",
        "qualification_evidence_batch_size": 240,
        "accepted_by": "Ha",
    },
    "encoder350": {
        "acceptance_id": "ha-accepted-native-encoder350-b192-transfer",
        "qualification_evidence_candidate": "encoder350",
        "qualification_evidence_backend": "transformers_peft",
        "qualification_evidence_execution_id": "ded3bd6ec1c04ee486500308f4f05035",
        "qualification_evidence_batch_size": 192,
        "accepted_by": "Ha",
    },
    "pii350": {
        "acceptance_id": "ha-accepted-native-encoder350-b192-transfer",
        "qualification_evidence_candidate": "encoder350",
        "qualification_evidence_backend": "transformers_peft",
        "qualification_evidence_batch_size": 192,
        "accepted_by": "Ha",
    },
}
"""These are candidate-specific approvals, not a rule that any model may borrow another model's batch size.

They are deliberately stable identifiers carried through the rendered contract and required at launch.

"""

ENCODER350_UNSLOTH_STEP1_QUALIFICATION: dict[str, str | int] = {
    "app_id": "ap-Yuc2x2y2LN1VEtKe6KM9er",
    "candidate": "encoder350",
    "runtime_backend": "unsloth",
    "unsloth": "2026.7.4",
    "unsloth_zoo": "2026.7.4",
    "batch_size": 192,
    "optimizer_steps": 1,
    "status": "ok",
}


@dataclass(frozen=True, slots=True)
class BackendContract:
    backend: Backend
    loader: str
    image: str
    bf16: bool
    gradient_checkpointing: str
    attention_implementation: str
    output_hidden_states: bool
    packed_segment_isolation: bool
    use_cache: bool
    trust_remote_code: bool


BACKENDS: dict[str, BackendContract] = {
    "base230": BackendContract(
        "unsloth",
        "FastLanguageModel.from_pretrained",
        "bioes-full-run-unsloth-isolated",
        bf16=True,
        gradient_checkpointing="use_gradient_checkpointing='unsloth'",
        attention_implementation="unsloth_managed",
        output_hidden_states=True,
        packed_segment_isolation=True,
        use_cache=False,
        trust_remote_code=False,
    ),
    "encoder230": BackendContract(
        "unsloth",
        "FastModel.from_pretrained(auto_model=AutoModelForMaskedLM)",
        "bioes-full-run-unsloth-2026-7-4-encoder-isolated",
        bf16=True,
        gradient_checkpointing="use_gradient_checkpointing='unsloth'",
        attention_implementation="unsloth_managed",
        output_hidden_states=False,
        packed_segment_isolation=True,
        use_cache=False,
        trust_remote_code=True,
    ),
    "encoder350": BackendContract(
        "unsloth",
        "FastModel.from_pretrained(auto_model=AutoModelForMaskedLM)",
        "bioes-full-run-unsloth-2026-7-4-encoder-isolated",
        bf16=True,
        gradient_checkpointing="use_gradient_checkpointing='unsloth'",
        attention_implementation="unsloth_managed",
        output_hidden_states=False,
        packed_segment_isolation=True,
        use_cache=False,
        trust_remote_code=True,
    ),
    "pii350": BackendContract(
        "unsloth",
        "FastModel.from_pretrained(auto_model=AutoModelForTokenClassification)",
        "bioes-full-run-unsloth-2026-7-4-encoder-isolated",
        bf16=True,
        gradient_checkpointing="use_gradient_checkpointing='unsloth'",
        attention_implementation="unsloth_managed",
        output_hidden_states=False,
        packed_segment_isolation=True,
        use_cache=False,
        trust_remote_code=True,
    ),
}


@dataclass(frozen=True, slots=True)
class FullRunContract:
    candidate_key: str
    model_id: str
    model_revision: str
    gpu: str
    backend: BackendContract
    qualification: Qualification
    seed: int
    packed_dataset: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    training: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    budget: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    config_digest: str


@dataclass(frozen=True, slots=True)
class ResumeState:
    candidate_key: str
    config_digest: str
    packed_revision: str
    packed_manifest_sha256: str
    eval_revision: str
    packed_cursor: int
    optimizer_step: int


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def backend_for(candidate_key: str) -> BackendContract:
    if candidate_key not in CANDIDATES or candidate_key not in BACKENDS:
        msg = "full run requires exactly one known candidate"
        raise ValueError(msg)
    return BACKENDS[candidate_key]


def qualification_for(candidate_key: str) -> Qualification:
    backend_for(candidate_key)
    return QUALIFICATIONS[candidate_key]


def runtime_packages_for(candidate_key: str) -> tuple[str, ...]:
    backend_for(candidate_key)
    return FULL_RUN_IMAGE_PACKAGES if candidate_key == "base230" else ENCODER_FULL_RUN_IMAGE_PACKAGES


def _full_epoch_geometry(batch_size: int) -> dict[str, int | bool]:
    """Report the fixed packed epoch shape; these values do not cap runtime.

    Returns:
        The packed-unit count, the optimizer steps a full epoch would take, the size of the
        final short batch, the per-step token ceiling, and a ``descriptive_only`` marker.
        That marker is the point: nothing here caps the run. The deadline and the optimizer
        step cap end training, and a reader who took ``optimizer_steps`` for a limit would
        misread every run that stops early.

    """
    full_batches, remainder = divmod(PACKED_UNIT_COUNT, batch_size)
    return {
        "packed_unit_count": PACKED_UNIT_COUNT,
        "optimizer_steps": full_batches + int(remainder > 0),
        "final_physical_batch_size": remainder or batch_size,
        "max_physical_tokens_per_step": batch_size * PACKED_MAX_LENGTH,
        "descriptive_only": True,
    }


def _contract_body(candidate_key: str) -> dict[str, Any]:
    backend = backend_for(candidate_key)
    candidate = CANDIDATES[candidate_key]
    qualification = qualification_for(candidate_key)
    return {
        "candidate_key": candidate_key,
        "model_id": candidate.model_id,
        "model_revision": candidate.revision,
        "gpu": H100_GPU,
        "backend": asdict(backend),
        "qualification": {
            **asdict(qualification),
            "manual_transfer_acceptance": deepcopy(MANUAL_TRANSFER_ACCEPTANCES.get(candidate_key)),
        },
        "seed": GATE_SEED,
        "runtime": {
            "python_version": FULL_RUN_PYTHON_VERSION,
            "packages": list(runtime_packages_for(candidate_key)),
        },
        "label_vocabulary": list(FULL_RUN_LABEL_VOCABULARY),
        "packed_dataset": {
            "id": PACKED_DATASET_ID,
            "revision": PACKED_DATASET_REVISION,
            "config": "packed",
            "manifest_sha256": PACKED_MANIFEST_SHA256,
            "shard_count": PACKED_SHARD_COUNT,
            "raw_source_population": RAW_SOURCE_POPULATION,
            "raw_source_revision": "f5d88d6ea0fbc6775d19097ceb406ceeaef19fdf",
            "accepted_row_count": ACCEPTED_ROW_COUNT,
            "rejected_row_count": 164_000,
            "packed_unit_count": PACKED_UNIT_COUNT,
            "permutation_seed": GATE_SEED,
            "permutation_digest": "f12ac0ba125c17b0d64058a0e2cd943f4f76eb3d272745e77a9b792054fa4c21",
            "real_token_count": 850_154_043,
            "physical_token_count": 983_334_912,
            "padding_token_count": 133_180_869,
            "packing_utilization": 0.864562045570899,
            "tokenizer_digests": {
                "causal": "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd",
                "encoder": "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711",
            },
            "compatibility_decision": "accepted",
            "boundary_contract": "lfm2_segment_isolation_v1",
            "boundary_token_count": 0,
            "max_length": PACKED_MAX_LENGTH,
            "full_seeded_artifact": True,
            "row_limit": None,
            "train_limit": None,
            "first_n": None,
        },
        "evaluation": {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "expected_rows": EVAL_ROWS,
            "timing": "final_only",
            "batching": "length_aware",
            "metrics": [
                "exact_span_typed",
                "exact_span_untyped",
                "containment_span_typed",
                "containment_span_untyped",
                "per_language_recall",
                "per_label_recall",
            ],
        },
        "training": {
            "epochs": 1,
            "batch_size": EXPECTED_FULL_BATCHES[candidate_key],
            "gradient_accumulation_steps": 1,
            "lora": {
                "rank": 64,
                "alpha": 128,
                "dropout": 0.0,
                "target_modules": list(LORA_TARGET_MODULES),
            },
            "torch_compile": False,
            "mechanics": deepcopy(DEFAULT_UNSLOTH_MECHANICS),
            "checkpoint_every_optimizer_steps": 10,
            "full_epoch_geometry": _full_epoch_geometry(EXPECTED_FULL_BATCHES[candidate_key]),
            **(
                {
                    "fresh_run": True,
                    "resume_from_checkpoint": None,
                    "optimizer_step_cap": None,
                }
                if candidate_key == "base230"
                else {}
            ),
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-4,
            "schedule": "constant",
            "fused": False,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
        },
        "budget": {
            "gpu_only_target_usd": FULL_RUN_TARGET_USD,
            "gpu_only_rate_usd_per_second": FULL_RUN_RATE_USD_PER_SECOND,
            "all_in_rate_usd_per_second": FULL_RUN_ALL_IN_RATE_USD_PER_SECOND,
            "gpu_only_budget_usd": GPU_ONLY_BUDGET_USD,
            "hard_timeout_seconds": FULL_RUN_HARD_TIMEOUT_SECONDS,
            "shutdown_reserve_seconds": FULL_RUN_SHUTDOWN_RESERVE_SECONDS,
            "training_deadline_seconds": FULL_RUN_TRAINING_DEADLINE_SECONDS,
            "gpu_only_timeout_estimate_usd": round(FULL_RUN_HARD_TIMEOUT_SECONDS * FULL_RUN_RATE_USD_PER_SECOND, 6),
            "all_in_timeout_estimate_usd": round(FULL_RUN_HARD_TIMEOUT_SECONDS * FULL_RUN_ALL_IN_RATE_USD_PER_SECOND, 6),
            "budget_enforcement": "Modal hard timeout only; no early runtime budget stop",
            "container_estimate_only": True,
            "reconciliation_fields": [
                "estimated_gpu_only_cost_usd",
                "estimated_all_in_cost_usd",
                "platform_actual_cost_usd",
            ],
        },
        "artifacts": {
            "jsonl_each_optimizer_step": True,
            "checkpoint_contents": [
                "adapter",
                "classifier",
                "optimizer",
                "packed_cursor",
                "rng_states",
                "data_pins",
                "model_pins",
                "eval_pins",
                "config_digest",
            ],
            "final_checkpoint": True,
            "save_final_eval_on_deadline": True,
            "run_root_files": ["label_vocabulary.json"],
        },
    }


def render_full_run(candidate_key: str) -> dict[str, Any]:
    """Return one deterministic candidate contract. No generic backend exists.

    Returns:
        The candidate's contract body with its ``config_digest`` appended. The digest is
        computed over the body alone, so it identifies the configuration rather than the
        rendering: two calls for one candidate give the same digest, and any edit to the body
        gives a different one. Every launch gate compares against this exact rendered value.

    """
    body = _contract_body(candidate_key)
    digest = _canonical_digest(body)
    return {**body, "config_digest": digest}


def render_base230_oom_fallback_contract() -> dict[str, Any]:
    """Render, but never select, Base230's manual lower-batch OOM fallback.

    Returns:
        The fallback contract at the lower batch size, carrying a ``fallback_launch`` block
        that records ``automatic_launch: False``, the exact primary OOM evidence required,
        and the primary action required. Rendering is deliberately separate from selecting:
        this can be called freely, and nothing here launches or authorizes anything.

    """
    body = _contract_body("base230")
    body["training"] = {
        **body["training"],
        "batch_size": BASE230_OOM_FALLBACK_BATCH_SIZE,
        "full_epoch_geometry": _full_epoch_geometry(BASE230_OOM_FALLBACK_BATCH_SIZE),
    }
    body["fallback_launch"] = {
        "automatic_launch": False,
        "required_primary_oom_evidence": BASE230_B416_OOM_EVIDENCE,
        "requires_primary_action": BASE230_OOM_FALLBACK_PRIMARY_ACTION,
    }
    return {**body, "config_digest": _canonical_digest(body)}


def require_base230_oom_fallback_execute(
    fallback_contract: Mapping[str, Any],
    *,
    primary_oom_evidence: Mapping[str, Any] | None,
    primary_action: str,
) -> None:
    """Gate the lower batch behind a real 416 OOM and explicit primary action.

    Raises:
        RuntimeError: If the supplied contract is not byte-equal to the freshly rendered
            fallback contract, if the primary OOM evidence is absent or not exactly the
            recorded batch-416 evidence, or if the primary action is not the exact required
            string. All three are equality checks against rendered constants rather than
            shape checks, so a contract that merely looks similar is refused. This authorizes
            a paid H100 run, and the cost of a false pass is a wrong run that bills.

    """
    if dict(fallback_contract) != render_base230_oom_fallback_contract():
        msg = "Base230 fallback contract is not the exact rendered contract"
        raise RuntimeError(msg)
    if primary_oom_evidence is None or dict(primary_oom_evidence) != BASE230_B416_OOM_EVIDENCE:
        msg = "Base230 fallback requires exact primary 416 OOM evidence"
        raise RuntimeError(msg)
    if primary_action != BASE230_OOM_FALLBACK_PRIMARY_ACTION:
        msg = "Base230 fallback requires explicit primary action"
        raise RuntimeError(msg)


def render_base230_oom_fallback_command() -> str:
    """Render the exact, explicitly OOM-gated Base230 batch-384 invocation.

    Returns:
        The full command line, with the confirmation token, the OOM evidence and the primary
        action already embedded. It renders the contract first purely to fail here if that
        contract cannot be built, so a command string is never produced for a configuration
        that would be refused at launch.

    """
    render_base230_oom_fallback_contract()
    inventory = json.dumps(
        {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "rows": EVAL_ROWS,
        },
        separators=(",", ":"),
    )
    evidence = json.dumps(BASE230_B416_OOM_EVIDENCE, separators=(",", ":"))
    return (
        f"MODAL_PROFILE={FULL_RUN_MODAL_PROFILES['base230']} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        "--candidate base230 --base230-oom-fallback --execute "
        f"--confirmation {FULL_RUN_CONFIRMATION} "
        f"--primary-oom-evidence-json '{evidence}' "
        f"--primary-action {BASE230_OOM_FALLBACK_PRIMARY_ACTION} "
        f"--eval-inventory-json '{inventory}'"
    )


def render_pii350_utilization_run_contract() -> dict[str, Any]:
    """Render the separately approved PII350 batch-256 utilization run.

    Returns:
        The utilization contract: batch 256, a fresh run with no resume checkpoint and no
        step cap, plus a ``manual_launch`` block naming the required source-run evidence, the
        required primary action, the Modal profile and the 85-90 percent device-memory target
        this run exists to reach. ``automatic_launch`` is ``False`` -- the batch size is the
        thing being tested, so it is never selected on the run's own behalf.

    """
    body = _contract_body("pii350")
    body["training"] = {
        **body["training"],
        "batch_size": PII350_UTILIZATION_BATCH_SIZE,
        "full_epoch_geometry": _full_epoch_geometry(PII350_UTILIZATION_BATCH_SIZE),
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "optimizer_step_cap": None,
    }
    body["manual_launch"] = {
        "purpose": "maximize_stable_h100_utilization",
        "automatic_launch": False,
        "required_source_run_evidence": PII350_B192_COMPLETE_EVIDENCE,
        "requires_primary_action": PII350_UTILIZATION_PRIMARY_ACTION,
        "modal_profile": PII350_UTILIZATION_PROFILE,
        "target_total_device_memory_percent": {"lower": 85, "upper": 90},
    }
    return {**body, "config_digest": _canonical_digest(body)}


def require_pii350_utilization_run_execute(
    utilization_contract: Mapping[str, Any],
    *,
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
) -> None:
    """Gate batch 256 behind the completed batch-192 run and Ha's approval.

    Raises:
        RuntimeError: If the supplied contract is not byte-equal to the rendered utilization
            contract, if the source-run evidence is absent or not exactly the recorded
            batch-192 completion evidence, or if the primary action is not the exact required
            string. The evidence check is what makes this an escalation rather than a choice:
            batch 256 is only reachable once batch 192 has actually completed.

    """
    if dict(utilization_contract) != render_pii350_utilization_run_contract():
        msg = "PII350 utilization contract is not the exact rendered contract"
        raise RuntimeError(msg)
    if source_run_evidence is None or dict(source_run_evidence) != PII350_B192_COMPLETE_EVIDENCE:
        msg = "PII350 utilization run requires exact source run evidence"
        raise RuntimeError(msg)
    if primary_action != PII350_UTILIZATION_PRIMARY_ACTION:
        msg = "PII350 utilization run requires explicit primary action"
        raise RuntimeError(msg)


def render_pii350_utilization_run_command() -> str:
    """Render the exact fresh PII350 batch-256 full-budget invocation.

    Returns:
        The full command line with the confirmation token, source-run evidence and primary
        action embedded. Rendering the contract first means an unbuildable configuration
        fails here rather than producing a command that would be refused at launch.

    """
    render_pii350_utilization_run_contract()
    inventory = json.dumps(
        {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "rows": EVAL_ROWS,
        },
        separators=(",", ":"),
    )
    evidence = json.dumps(PII350_B192_COMPLETE_EVIDENCE, separators=(",", ":"))
    return (
        f"MODAL_PROFILE={PII350_UTILIZATION_PROFILE} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        "--candidate pii350 --pii350-utilization-run --execute "
        f"--confirmation {FULL_RUN_CONFIRMATION} "
        f"--source-run-evidence-json '{evidence}' "
        f"--primary-action {PII350_UTILIZATION_PRIMARY_ACTION} "
        f"--eval-inventory-json '{inventory}'"
    )


def render_pii350_utilization_prewarm_command() -> str:
    """Render the CPU-only online cache hydration for the utilization run.

    Returns:
        The prewarm command. It runs on CPU and only populates caches, so it costs no GPU
        time and needs no evidence or primary action -- which is why it carries neither.

    """
    return (
        f"MODAL_PROFILE={PII350_UTILIZATION_PROFILE} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        "--candidate pii350 --pii350-utilization-run --prewarm-assets"
    )


def render_pii350_utilization_preflight_command() -> str:
    """Render the CPU-only offline receipt for the utilization run.

    Returns:
        The preflight command. Offline and CPU-only: it proves the assets are already present
        rather than fetching them, which is the prerequisite the paid run's launch checks.

    """
    return (
        f"MODAL_PROFILE={PII350_UTILIZATION_PROFILE} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        "--candidate pii350 --pii350-utilization-run --preflight-assets"
    )


def render_pii350_capacity_run_contract() -> dict[str, Any]:
    """Render the separately approved r128/alpha256 PII350 batch-256 capacity run.

    Returns:
        The capacity contract at batch 256 with the larger LoRA rank and alpha applied to BOTH
        the public ``training.lora`` block and the Unsloth ``mechanics.lora`` copy. The runtime
        cross-checks those two against each other, so writing one without the other would be
        refused at launch rather than silently training a different adapter than the contract
        advertises.

    """
    body = _contract_body("pii350")
    mechanics = body["training"]["mechanics"]
    body["training"] = {
        **body["training"],
        "batch_size": PII350_CAPACITY_BATCH_SIZE,
        "lora": {
            **body["training"]["lora"],
            "rank": PII350_CAPACITY_LORA_RANK,
            "alpha": PII350_CAPACITY_LORA_ALPHA,
        },
        "mechanics": {
            **mechanics,
            "lora": {
                **mechanics["lora"],
                "rank": PII350_CAPACITY_LORA_RANK,
                "alpha": PII350_CAPACITY_LORA_ALPHA,
            },
        },
        "full_epoch_geometry": _full_epoch_geometry(PII350_CAPACITY_BATCH_SIZE),
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "optimizer_step_cap": None,
    }
    body["manual_launch"] = {
        "purpose": "measure_lora_capacity_at_fixed_h100_budget",
        "automatic_launch": False,
        "required_source_run_evidence": PII350_B192_COMPLETE_EVIDENCE,
        "requires_primary_action": PII350_CAPACITY_PRIMARY_ACTION,
        "modal_profile": PII350_CAPACITY_PROFILE,
        "oom_policy": {
            "automatic_fallback": False,
            "manual_fallback_batch_size": PII350_CAPACITY_OOM_FALLBACK_BATCH_SIZE,
            "requires_persisted_primary_oom": True,
        },
    }
    return {**body, "config_digest": _canonical_digest(body)}


def require_pii350_capacity_run_execute(
    capacity_contract: Mapping[str, Any],
    *,
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
) -> None:
    """Gate the higher-capacity run behind the completed baseline and Ha's approval.

    Raises:
        RuntimeError: If the supplied contract is not byte-equal to the rendered capacity
            contract, if the source-run evidence is absent or not exactly the recorded batch-192
            completion evidence, or if the primary action is not the exact required string. This
            gate and the utilization gate take the SAME source evidence but different primary
            actions, so completing batch 192 unlocks each of them only through its own action --
            one approval cannot be reused to launch the other run.

    """
    if dict(capacity_contract) != render_pii350_capacity_run_contract():
        msg = "PII350 capacity contract is not the exact rendered contract"
        raise RuntimeError(msg)
    if source_run_evidence is None or dict(source_run_evidence) != PII350_B192_COMPLETE_EVIDENCE:
        msg = "PII350 capacity run requires exact source run evidence"
        raise RuntimeError(msg)
    if primary_action != PII350_CAPACITY_PRIMARY_ACTION:
        msg = "PII350 capacity run requires explicit primary action"
        raise RuntimeError(msg)


def render_pii350_capacity_run_command() -> str:
    """Render the exact fresh PII350 r128/alpha256 invocation.

    Returns:
        The full command line for the capacity run, with confirmation, evidence and primary
        action embedded, after rendering the contract to fail early on an unbuildable one.

    """
    render_pii350_capacity_run_contract()
    inventory = json.dumps(
        {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "rows": EVAL_ROWS,
        },
        separators=(",", ":"),
    )
    evidence = json.dumps(PII350_B192_COMPLETE_EVIDENCE, separators=(",", ":"))
    return (
        f"MODAL_PROFILE={PII350_CAPACITY_PROFILE} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        "--candidate pii350 --pii350-capacity-run --execute "
        f"--confirmation {FULL_RUN_CONFIRMATION} "
        f"--source-run-evidence-json '{evidence}' "
        f"--primary-action {PII350_CAPACITY_PRIMARY_ACTION} "
        f"--eval-inventory-json '{inventory}'"
    )


def _pii350_milestone_spec(profile: str) -> Mapping[str, str | int]:
    try:
        return PII350_MILESTONE_RUNS[profile]
    except KeyError as error:
        msg = "PII350 milestone run requires an approved profile"
        raise ValueError(msg) from error


def _pii350_milestone_qualification(profile: str, *, selected_batch_size: int) -> dict[str, Any]:
    """Describe the actual strict-Unsloth source evidence for each decision arm.

    Returns:
        The qualification block for the profile: its acceptance status, artifact and execution
        ids, the batch size the evidence was gathered at, and who accepted it. It reports the
        evidence that actually exists rather than the evidence the arm would prefer, so an arm
        whose source run differs from the selected batch size still names its real provenance.

    """
    source_run = {
        "candidate": "pii350",
        "runtime_backend": "unsloth",
        "batch_size": int(PII350_B192_COMPLETE_EVIDENCE["batch_size"]),
        "execution_id": PII350_B192_COMPLETE_EVIDENCE["execution_id"],
        "status": PII350_B192_COMPLETE_EVIDENCE["status"],
    }
    if profile == "meddies-pii":
        return {
            "status": "accepted_live",
            "artifact_id": "pii350-strict-unsloth-b192-complete",
            "execution_id": PII350_B192_COMPLETE_EVIDENCE["execution_id"],
            "selected_batch_size": selected_batch_size,
            "median_real_tokens_per_second": None,
            "rationale": (
                "PII350 strict-Unsloth batch 192 completed to the training deadline and is live evidence "
                "for this batch-192 decision arm"
            ),
            "source_run": source_run,
            "transfer_basis": None,
            "manual_transfer_acceptance": None,
        }
    return {
        "status": "accepted_transfer",
        "artifact_id": "pii350-strict-unsloth-b192-complete",
        "execution_id": PII350_B192_COMPLETE_EVIDENCE["execution_id"],
        "selected_batch_size": selected_batch_size,
        "median_real_tokens_per_second": None,
        "rationale": (
            "PII350 batch 160 is an accepted lower-batch transfer from the completed strict-Unsloth "
            "batch-192 source run; it does not claim batch-160 live throughput evidence"
        ),
        "source_run": source_run,
        "transfer_basis": "accepted_lower_batch_transfer_from_strict_unsloth_pii350_b192",
        "manual_transfer_acceptance": {
            "acceptance_id": "ha-authorized-pii350-b160-transfer-from-strict-unsloth-b192",
            "qualification_evidence_candidate": "pii350",
            "qualification_evidence_backend": "unsloth",
            "qualification_evidence_execution_id": PII350_B192_COMPLETE_EVIDENCE["execution_id"],
            "qualification_evidence_batch_size": 192,
            "accepted_by": "Ha",
        },
    }


def render_pii350_milestone_run_contract(profile: str) -> dict[str, Any]:
    """Render one equal-cursor PII350 milestone decision run.

    Each profile encodes a distinct scientific estimand. The 7,680 packed-unit
    milestone is a checkpoint-and-evaluate event, never a training stop cap.

    Returns:
        The milestone contract for that profile, with evaluation timing set to
        ``milestone_and_final`` so the run scores at the milestone AND at the end. The
        milestone is a checkpoint-and-evaluate event rather than a stop cap, which is why the
        contract carries no ``optimizer_step_cap`` alongside it -- the runtime refuses a
        contract that sets both, because they would disagree about when training ends.

    Raises:
        RuntimeError: If the profile's batch size times its optimizer step does not reach the
            fixed milestone cursor. Every arm must hit the SAME packed cursor for the arms to
            be comparable; an arm that reached a different cursor would answer a different
            question while looking like the same experiment.

    """
    spec = _pii350_milestone_spec(profile)
    body = _contract_body("pii350")
    body["evaluation"] = {
        **body["evaluation"],
        "timing": "milestone_and_final",
    }
    mechanics = body["training"]["mechanics"]
    batch_size = int(spec["batch_size"])
    optimizer_step = int(spec["optimizer_step"])
    if batch_size * optimizer_step != PII350_MILESTONE_PACKED_CURSOR:
        msg = "PII350 milestone batch and step do not reach cursor 7,680"
        raise RuntimeError(msg)
    body["qualification"] = _pii350_milestone_qualification(profile, selected_batch_size=batch_size)
    body["training"] = {
        **body["training"],
        "batch_size": batch_size,
        "lora": {
            **body["training"]["lora"],
            "rank": PII350_CAPACITY_LORA_RANK,
            "alpha": PII350_CAPACITY_LORA_ALPHA,
        },
        "mechanics": {
            **mechanics,
            "lora": {
                **mechanics["lora"],
                "rank": PII350_CAPACITY_LORA_RANK,
                "alpha": PII350_CAPACITY_LORA_ALPHA,
            },
        },
        "full_epoch_geometry": _full_epoch_geometry(batch_size),
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "optimizer_step_cap": None,
        "milestone": {
            "optimizer_step": optimizer_step,
            "packed_cursor": PII350_MILESTONE_PACKED_CURSOR,
            "evaluation_rows": PII350_MILESTONE_EVALUATION_ROWS,
            "checkpoint": True,
            "evaluate": True,
            "continues_training": True,
        },
    }
    body["budget"] = {
        "all_in_credit_ceiling_usd": PII350_MILESTONE_ALL_IN_CREDIT_CEILING_USD,
        "all_in_observed_runtime_multiplier": (PII350_MILESTONE_OBSERVED_ALL_IN_RUNTIME_MULTIPLIER),
        "all_in_projected_timeout_estimate_usd": round(
            PII350_MILESTONE_HARD_TIMEOUT_SECONDS
            * FULL_RUN_RATE_USD_PER_SECOND
            * PII350_MILESTONE_OBSERVED_ALL_IN_RUNTIME_MULTIPLIER,
            4,
        ),
        "budget_rationale": (
            "USD 9.80 is an all-in account credit; equal H100 wall time uses a conservative USD 8.20 H100 ceiling"
        ),
        "gpu_only_target_usd": PII350_MILESTONE_H100_BUDGET_USD,
        "gpu_only_rate_usd_per_second": FULL_RUN_RATE_USD_PER_SECOND,
        "hard_timeout_seconds": PII350_MILESTONE_HARD_TIMEOUT_SECONDS,
        "shutdown_reserve_seconds": PII350_MILESTONE_SHUTDOWN_RESERVE_SECONDS,
        "training_deadline_seconds": PII350_MILESTONE_TRAINING_DEADLINE_SECONDS,
        "gpu_only_timeout_estimate_usd": round(PII350_MILESTONE_HARD_TIMEOUT_SECONDS * FULL_RUN_RATE_USD_PER_SECOND, 6),
        "budget_enforcement": "Modal hard timeout only; no early runtime budget stop",
    }
    body["manual_launch"] = {
        "purpose": "compare_pii350_r128a256_at_equal_packed_cursor_and_fixed_h100_dollars",
        "automatic_launch": False,
        "required_source_run_evidence": PII350_B192_COMPLETE_EVIDENCE,
        "requires_primary_action": spec["primary_action"],
        "modal_profile": profile,
        "oom_policy": {"automatic_fallback": False},
    }
    return {**body, "config_digest": _canonical_digest(body)}


def require_pii350_milestone_run_execute(
    milestone_contract: Mapping[str, Any],
    *,
    profile: str,
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
) -> None:
    """Fail closed unless the selected profile, source evidence, and action match.

    Raises:
        RuntimeError: If the supplied contract is not byte-equal to the one rendered for that
            profile, if the source-run evidence is absent or not exactly the recorded
            batch-192 completion evidence, or if the primary action is not the one that
            profile's own contract requires. The action is read back off the rendered
            contract rather than from a constant, so each profile demands its own action and
            an action approved for one arm cannot launch another.

    """
    expected = render_pii350_milestone_run_contract(profile)
    if dict(milestone_contract) != expected:
        msg = "PII350 milestone run requires the exact rendered contract"
        raise RuntimeError(msg)
    if source_run_evidence is None or dict(source_run_evidence) != PII350_B192_COMPLETE_EVIDENCE:
        msg = "PII350 milestone run requires exact source run evidence"
        raise RuntimeError(msg)
    if primary_action != expected["manual_launch"]["requires_primary_action"]:
        msg = "PII350 milestone run requires explicit primary action"
        raise RuntimeError(msg)


def render_pii350_milestone_run_command(profile: str) -> str:
    """Render one explicit H100 dispatch command without launching it.

    Returns:
        The dispatch command for that profile, carrying the confirmation token, the source-run
        evidence and the profile's own required primary action, which is read back off the
        rendered contract. Rendering never launches; the command still has to be run.

    """
    contract = render_pii350_milestone_run_contract(profile)
    inventory = json.dumps(
        {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "rows": EVAL_ROWS,
        },
        separators=(",", ":"),
    )
    evidence = json.dumps(PII350_B192_COMPLETE_EVIDENCE, separators=(",", ":"))
    return (
        f"MODAL_PROFILE={profile} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate pii350 --pii350-milestone-run {profile} --execute "
        f"--confirmation {FULL_RUN_CONFIRMATION} "
        f"--source-run-evidence-json '{evidence}' "
        f"--primary-action {contract['manual_launch']['requires_primary_action']} "
        f"--eval-inventory-json '{inventory}'"
    )


def render_pii350_milestone_run_preflight_command(profile: str) -> str:
    """Render the CPU-only offline receipt prerequisite for one decision run.

    Returns:
        The preflight command for that profile. The contract is rendered first so an invalid
        profile fails here rather than at dispatch.

    """
    render_pii350_milestone_run_contract(profile)
    return (
        f"MODAL_PROFILE={profile} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate pii350 --pii350-milestone-run {profile} --preflight-assets"
    )


def render_pii350_scout_contract(arm: str) -> dict[str, Any]:
    """Render one immutable batch-128 PII350 anchor without selecting an arm.

    Returns:
        The scout contract for that arm at the fixed batch-128 anchor. The batch size is held
        constant across arms on purpose: the arms differ in schedule and learning rate, so
        holding the batch fixed is what makes their results comparable.

    Raises:
        ValueError: If the arm is not one of the approved arms.
        RuntimeError: If the arm's recorded Modal profile is not a string, and for the
            remaining contract guards in this function. The module declares why these report
            ``RuntimeError`` rather than ``TypeError``: they describe a malformed launch
            contract, and the same function raises this type from non-isinstance guards too.

    """
    try:
        spec = PII350_SCOUT_ARMS[arm]
    except KeyError as error:
        msg = "PII350 scout requires one approved arm"
        raise ValueError(msg) from error
    profile = spec["modal_profile"]
    if not isinstance(profile, str):
        msg = "PII350 scout profile must be a string"
        # reason: every guard here reports an environment or contract failure - a missing asset, an unverified
        # reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
        # reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
        # reason: make one failure class signal two exception types.
        raise RuntimeError(msg)  # ruff: ignore[type-check-without-type-error]
    schedule = spec.get("schedule", "constant")
    body = _contract_body("pii350")
    mechanics = body["training"]["mechanics"]
    if schedule == "wsd":
        total_steps = spec.get("total_steps")
        warmup_steps = spec.get("warmup_steps")
        stable_steps = spec.get("stable_steps")
        decay_steps = spec.get("decay_steps")
        warmup_type = spec.get("warmup_type")
        decay_type = spec.get("decay_type")
        min_lr_ratio = spec.get("min_lr_ratio")
        peak_lr = spec["lr"]
        if (
            not all(isinstance(value, int) for value in (total_steps, warmup_steps, stable_steps, decay_steps))
            or not isinstance(warmup_type, str)
            or not isinstance(decay_type, str)
            or not isinstance(min_lr_ratio, (int, float))
            or not isinstance(peak_lr, (int, float))
        ):
            msg = "PII350 WSD scout schedule is invalid"
            raise RuntimeError(msg)
        # reason: inert narrowing - the guard above already raised RuntimeError on any non-int, so stripping these
        # reason: under -O changes no reachable behavior; they exist so the type checker sees the int.
        assert isinstance(total_steps, int)  # ruff: ignore[assert]
        assert isinstance(warmup_steps, int)  # ruff: ignore[assert]
        assert isinstance(stable_steps, int)  # ruff: ignore[assert]
        assert isinstance(decay_steps, int)  # ruff: ignore[assert]
        if warmup_steps + stable_steps + decay_steps != total_steps:
            msg = "PII350 WSD scout phases must sum to the step cap"
            raise RuntimeError(msg)
        schedule_fields = {
            "peak_lr": float(peak_lr),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "stable_steps": stable_steps,
            "decay_steps": decay_steps,
            "warmup_type": warmup_type,
            "decay_type": decay_type,
            "min_lr_ratio": float(min_lr_ratio),
        }
        body["evaluation"] = {**body["evaluation"], "timing": "final_only"}
        body["training"] = {
            **body["training"],
            "batch_size": 128,
            "lora": {**body["training"]["lora"], "rank": 128, "alpha": 256},
            "mechanics": {
                **mechanics,
                "lora": {**mechanics["lora"], "rank": 128, "alpha": 256},
                "adamw": {**mechanics["adamw"], "lr": float(peak_lr)},
                "scheduler": {"name": "wsd", **schedule_fields},
            },
            "full_epoch_geometry": _full_epoch_geometry(128),
            "fresh_run": True,
            "resume_from_checkpoint": None,
            "optimizer_step_cap": total_steps,
            "milestone": None,
        }
        body["optimizer"] = {
            **body["optimizer"],
            "lr": float(peak_lr),
            "schedule": "wsd",
            **schedule_fields,
        }
        body["budget"] = {
            "all_in_credit_ceiling_usd": PII350_MILESTONE_ALL_IN_CREDIT_CEILING_USD,
            "all_in_observed_runtime_multiplier": PII350_MILESTONE_OBSERVED_ALL_IN_RUNTIME_MULTIPLIER,
            "all_in_projected_timeout_estimate_usd": 9.652526,
            "gpu_only_target_usd": PII350_MILESTONE_H100_BUDGET_USD,
            "gpu_only_rate_usd_per_second": FULL_RUN_RATE_USD_PER_SECOND,
            "hard_timeout_seconds": PII350_MILESTONE_HARD_TIMEOUT_SECONDS,
            "shutdown_reserve_seconds": PII350_MILESTONE_SHUTDOWN_RESERVE_SECONDS,
            "training_deadline_seconds": PII350_MILESTONE_TRAINING_DEADLINE_SECONDS,
            "gpu_only_timeout_estimate_usd": 8.198978,
            "budget_enforcement": "Modal hard timeout only; no early runtime budget stop",
        }
        body["manual_launch"] = {
            "purpose": "anchor_pii350_r128a256_batch128_wsd_learning_rate",
            "automatic_launch": False,
            "required_source_run_evidence": PII350_B192_COMPLETE_EVIDENCE,
            "requires_primary_action": spec["primary_action"],
            "modal_profile": profile,
            "oom_policy": {"automatic_fallback": False},
        }
        return {**body, "config_digest": _canonical_digest(body)}
    if schedule != "constant":
        msg = "PII350 scout requires a supported optimizer schedule"
        raise RuntimeError(msg)
    body["evaluation"] = {**body["evaluation"], "timing": "milestone_and_final"}
    body["training"] = {
        **body["training"],
        "batch_size": 128,
        "lora": {**body["training"]["lora"], "rank": 128, "alpha": 256},
        "mechanics": {
            **mechanics,
            "lora": {**mechanics["lora"], "rank": 128, "alpha": 256},
            "adamw": {**mechanics["adamw"], "lr": spec["lr"]},
        },
        "full_epoch_geometry": _full_epoch_geometry(128),
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "optimizer_step_cap": None,
        "milestone": {
            "optimizer_step": 60,
            "packed_cursor": 7680,
            "evaluation_rows": 1700,
            "checkpoint": True,
            "evaluate": True,
            "continues_training": True,
        },
    }
    body["optimizer"] = {**body["optimizer"], "lr": spec["lr"]}
    body["budget"] = {
        "all_in_credit_ceiling_usd": PII350_MILESTONE_ALL_IN_CREDIT_CEILING_USD,
        "all_in_observed_runtime_multiplier": PII350_MILESTONE_OBSERVED_ALL_IN_RUNTIME_MULTIPLIER,
        "all_in_projected_timeout_estimate_usd": 9.652526,
        "gpu_only_target_usd": PII350_MILESTONE_H100_BUDGET_USD,
        "gpu_only_rate_usd_per_second": FULL_RUN_RATE_USD_PER_SECOND,
        "hard_timeout_seconds": PII350_MILESTONE_HARD_TIMEOUT_SECONDS,
        "shutdown_reserve_seconds": PII350_MILESTONE_SHUTDOWN_RESERVE_SECONDS,
        "training_deadline_seconds": PII350_MILESTONE_TRAINING_DEADLINE_SECONDS,
        "gpu_only_timeout_estimate_usd": 8.198978,
        "budget_enforcement": "Modal hard timeout only; no early runtime budget stop",
    }
    body["manual_launch"] = {
        "purpose": "anchor_pii350_r128a256_batch128_learning_rate",
        "automatic_launch": False,
        "required_source_run_evidence": PII350_B192_COMPLETE_EVIDENCE,
        "requires_primary_action": spec["primary_action"],
        "modal_profile": profile,
        "oom_policy": {"automatic_fallback": False},
    }
    if arm == "lr4e-4":
        body["evaluation"] = {**body["evaluation"], "timing": "final_only"}
        body["training"] = {
            **body["training"],
            "optimizer_step_cap": 60,
            "milestone": None,
        }
    return {**body, "config_digest": _canonical_digest(body)}


def require_pii350_scout_execute(
    scout_contract: Mapping[str, Any],
    *,
    arm: str,
    source_run_evidence: Mapping[str, Any] | None,
    primary_action: str,
) -> None:
    expected = render_pii350_scout_contract(arm)
    if dict(scout_contract) != expected:
        msg = "PII350 scout requires the exact rendered contract"
        raise RuntimeError(msg)
    if source_run_evidence is None or dict(source_run_evidence) != PII350_B192_COMPLETE_EVIDENCE:
        msg = "PII350 scout requires exact source run evidence"
        raise RuntimeError(msg)
    if primary_action != expected["manual_launch"]["requires_primary_action"]:
        msg = "PII350 scout requires explicit primary action"
        raise RuntimeError(msg)


def render_pii350_scout_command(arm: str) -> str:
    contract = render_pii350_scout_contract(arm)
    inventory = json.dumps(
        {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "rows": EVAL_ROWS,
        },
        separators=(",", ":"),
    )
    evidence = json.dumps(PII350_B192_COMPLETE_EVIDENCE, separators=(",", ":"))
    return (
        f"MODAL_PROFILE={contract['manual_launch']['modal_profile']} uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate pii350 --pii350-scout-run {arm} --execute "
        f"--confirmation {FULL_RUN_CONFIRMATION} --source-run-evidence-json '{evidence}' "
        f"--primary-action {contract['manual_launch']['requires_primary_action']} "
        f"--eval-inventory-json '{inventory}'"
    )


def render_pii350_scout_preflight_command(arm: str) -> str:
    """Render the CPU-only receipt prerequisite bound to one scout arm.

    Returns:
        The preflight command, with the Modal profile taken from the arm's own rendered
        contract rather than from a caller argument, so the receipt is always produced under
        the profile that will run it.

    """
    contract = render_pii350_scout_contract(arm)
    return (
        f"MODAL_PROFILE={contract['manual_launch']['modal_profile']} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate pii350 --pii350-scout-run {arm} --preflight-assets"
    )


def render_m230_comparison_run_contract(candidate_key: str) -> dict[str, Any]:
    """Render the approved Base230-320 or Encoder230-288 comparison contract.

    Returns:
        The comparison contract for that candidate, pinned to the shared encoder image and
        package set. Both candidates are rendered against the SAME image, which is what makes
        the comparison a measurement of the models rather than of their runtimes.

    Raises:
        ValueError: If the candidate is neither of the two the comparison is defined over.

    """
    if candidate_key not in M230_COMPARISON_CANDIDATES:
        msg = "230M comparison requires base230 or encoder230"
        raise ValueError(msg)
    body = _contract_body(candidate_key)
    body["runtime"] = {
        **body["runtime"],
        "packages": list(ENCODER_FULL_RUN_IMAGE_PACKAGES),
    }
    body["backend"] = {
        **body["backend"],
        "image": "bioes-full-run-unsloth-2026-7-4-shared-isolated",
    }
    mechanics = body["training"]["mechanics"]
    body["training"] = {
        **body["training"],
        "batch_size": M230_COMPARISON_BATCH_SIZES[candidate_key],
        "lora": {
            **body["training"]["lora"],
            "rank": 64,
            "alpha": 128,
        },
        "mechanics": {
            **mechanics,
            "lora": {
                **mechanics["lora"],
                "rank": 64,
                "alpha": 128,
            },
        },
        "full_epoch_geometry": _full_epoch_geometry(M230_COMPARISON_BATCH_SIZES[candidate_key]),
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "optimizer_step_cap": None,
    }
    body["manual_launch"] = {
        "purpose": "compare_230m_base_and_encoder_at_fixed_budget",
        "automatic_launch": False,
        "requires_primary_action": M230_COMPARISON_PRIMARY_ACTIONS[candidate_key],
        "modal_profile": M230_COMPARISON_PROFILES[candidate_key],
        "oom_policy": {"automatic_fallback": False},
    }
    return {**body, "config_digest": _canonical_digest(body)}


def require_m230_comparison_run_execute(
    comparison_contract: Mapping[str, Any],
    *,
    candidate_key: str,
    primary_action: str,
) -> None:
    """Gate one 230M comparison behind its exact contract and Ha's action.

    Raises:
        RuntimeError: If the supplied contract is not byte-equal to the rendered comparison
            contract for that candidate, or if the primary action is not the one recorded for
            that candidate. The action is keyed per candidate, so an approval for the Base230
            arm cannot launch the Encoder230 arm. Unlike the utilization and milestone gates
            this one takes no source-run evidence, because the comparison has no prerequisite
            run -- the two arms ARE the experiment.

    """
    if dict(comparison_contract) != render_m230_comparison_run_contract(candidate_key):
        msg = "230M comparison requires the exact rendered contract"
        raise RuntimeError(msg)
    if primary_action != M230_COMPARISON_PRIMARY_ACTIONS[candidate_key]:
        msg = "230M comparison requires explicit primary action"
        raise RuntimeError(msg)


def render_m230_comparison_run_command(candidate_key: str) -> str:
    """Render one fresh candidate-specific 230M H100 invocation.

    Returns:
        The dispatch command for that candidate, with its own Modal profile, confirmation
        token and candidate-keyed primary action embedded.

    """
    render_m230_comparison_run_contract(candidate_key)
    inventory = json.dumps(
        {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "rows": EVAL_ROWS,
        },
        separators=(",", ":"),
    )
    return (
        f"MODAL_PROFILE={M230_COMPARISON_PROFILES[candidate_key]} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate {candidate_key} --m230-comparison-run --execute "
        f"--confirmation {FULL_RUN_CONFIRMATION} "
        f"--primary-action {M230_COMPARISON_PRIMARY_ACTIONS[candidate_key]} "
        f"--eval-inventory-json '{inventory}'"
    )


def render_m230_comparison_prewarm_command(candidate_key: str) -> str:
    """Render one CPU-only online cache hydration for a 230M candidate.

    Returns:
        The prewarm command for that candidate. CPU-only and cache-populating, so it carries
        no action and costs no GPU time.

    """
    render_m230_comparison_run_contract(candidate_key)
    return (
        f"MODAL_PROFILE={M230_COMPARISON_PROFILES[candidate_key]} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate {candidate_key} --m230-comparison-run --prewarm-assets"
    )


def render_m230_comparison_preflight_command(candidate_key: str) -> str:
    """Render one CPU-only offline receipt for a 230M candidate.

    Returns:
        The preflight command for that candidate, offline and CPU-only, proving the assets are
        already cached rather than fetching them.

    """
    render_m230_comparison_run_contract(candidate_key)
    return (
        f"MODAL_PROFILE={M230_COMPARISON_PROFILES[candidate_key]} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate {candidate_key} --m230-comparison-run --preflight-assets"
    )


def render_all_full_runs() -> dict[str, dict[str, Any]]:
    return {key: render_full_run(key) for key in CANDIDATES}


# reason: Qualification report, backend choice, candidate identity, and failures form one launch verdict.
def require_qualified_full_run(contract: Mapping[str, Any], *, qualification_artifact: Mapping[str, Any] | None) -> None:  # ruff: ignore[complex-structure]
    candidate_key = contract.get("candidate_key")
    if not isinstance(candidate_key, str):
        msg = "full-run contract lacks candidate"
        # reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
        # reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
        # reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    qualification = qualification_for(candidate_key)
    if qualification_artifact is None:
        msg = "a qualification artifact is required for full launch"
        raise RuntimeError(msg)
    if qualification.status == "accepted_live":
        if dict(qualification_artifact) != BASE230_LIVE_ACCEPTANCE:
            msg = "base230 requires Ha's exact accepted three-step Unsloth live evidence"
            raise RuntimeError(msg)
        return
    if qualification.status == "accepted_transfer":
        accepted = MANUAL_TRANSFER_ACCEPTANCES[candidate_key]
        required = {
            "candidate": candidate_key,
            "batch_size": EXPECTED_FULL_BATCHES[candidate_key],
            **accepted,
        }
        if any(qualification_artifact.get(key) != value for key, value in required.items()):
            msg = f"{candidate_key} requires its exact recorded manual transfer acceptance"
            raise RuntimeError(msg)
        return
    if qualification.status == "unqualified":
        expected_backend = backend_for(candidate_key).backend
        # reason: require qualified keeps ok/candidate in one gate; helper predicates would scatter the rule.
        if (
            qualification_artifact.get("candidate") != candidate_key  # ruff: ignore[too-many-boolean-expressions]
            or qualification_artifact.get("runtime_backend") != expected_backend
            or qualification_artifact.get("batch_size") != EXPECTED_FULL_BATCHES[candidate_key]
            or qualification_artifact.get("optimizer_steps") != QUALIFICATION_OPTIMIZER_STEPS
            or qualification_artifact.get("status") != "ok"
            or not isinstance(qualification_artifact.get("execution_id"), str)
        ):
            msg = f"{candidate_key} is UNQUALIFIED; attach its successful candidate-specific runtime artifact"
            raise RuntimeError(
                msg,
            )
        return
    if qualification_artifact.get("execution_id") != qualification.execution_id:
        msg = "qualification artifact execution does not match the selected evidence"
        raise RuntimeError(msg)
    if qualification_artifact.get("candidate") != candidate_key:
        msg = "qualification artifact candidate does not match full run"
        raise RuntimeError(msg)
    if qualification_artifact.get("batch_size") != qualification.selected_batch_size:
        msg = "qualification artifact batch does not match selected evidence"
        raise RuntimeError(msg)


def validate_full_run_contract(contract: Mapping[str, Any]) -> None:
    candidate_key = contract.get("candidate_key")
    if not isinstance(candidate_key, str):
        msg = "full run requires exactly one candidate"
        # reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
        # reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
        # reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    expected = render_full_run(candidate_key)
    actual = dict(contract)
    if actual != expected:
        differing = sorted(set(actual) ^ set(expected))
        if not differing:
            differing = sorted(key for key in expected if actual.get(key) != expected[key])
        msg = f"full-run contract mismatch: {', '.join(differing)}"
        raise ValueError(msg)
    if contract["gpu"] != H100_GPU:
        msg = "full run requires H100!"
        raise ValueError(msg)
    if contract["training"]["epochs"] != 1:
        msg = "full run has a one-epoch hard cap"
        raise ValueError(msg)
    if any(contract["packed_dataset"].get(key) is not None for key in ("row_limit", "train_limit", "first_n")):
        msg = "full run forbids row limits and first-N sampling"
        raise ValueError(msg)
    # reason: the right side is a pinned contract constant the producer writes verbatim, so an
    # reason: approximate match would admit a checkpoint or budget that is not the allowlisted one.
    if float(contract["budget"]["gpu_only_budget_usd"]) != FULL_RUN_TARGET_USD:  # ruff: ignore[float-equality-comparison]
        msg = "full run GPU-only budget must equal USD 10.00"
        raise ValueError(msg)
    if contract["budget"]["hard_timeout_seconds"] != FULL_RUN_HARD_TIMEOUT_SECONDS:
        msg = "full run hard timeout mismatch"
        raise ValueError(msg)


def require_full_run_execute(
    contract: Mapping[str, Any],
    *,
    execute: bool,
    confirmation: str,
    qualification_artifact: Mapping[str, Any] | None,
) -> None:
    validate_full_run_contract(contract)
    if not execute or confirmation != FULL_RUN_CONFIRMATION:
        msg = "full H100 run requires --execute and exact confirmation"
        raise RuntimeError(msg)
    require_qualified_full_run(contract, qualification_artifact=qualification_artifact)


def validate_pinned_eval_inventory(inventory: Mapping[str, Any]) -> None:
    """Gate encoding on the pinned ``eval/train`` population, not a row cap.

    Raises:
        RuntimeError: If the inventory is not exactly the pinned dataset id, revision, config,
            split and row count. The comparison is whole-dict equality, so an inventory that
            names the right dataset with a different revision, or the right revision with a
            different row count, is refused. A row count alone would not catch a revision that
            moved while keeping its size, which is the case this guards.

    """
    expected = {
        "id": EVAL_DATASET_ID,
        "revision": EVAL_DATASET_REVISION,
        "config": EVAL_CONFIG,
        "split": EVAL_SPLIT,
        "rows": EVAL_ROWS,
    }
    if dict(inventory) != expected:
        msg = "pinned eval inventory must be eval/train with exactly 1,700 rows"
        raise RuntimeError(msg)


def validate_resume_state(contract: Mapping[str, Any], state: ResumeState) -> None:
    validate_full_run_contract(contract)
    if state.candidate_key != contract["candidate_key"]:
        msg = "resume candidate does not match launch contract"
        raise RuntimeError(msg)
    if state.config_digest != contract["config_digest"]:
        msg = "resume config digest does not match launch contract"
        raise RuntimeError(msg)
    if state.packed_revision != PACKED_DATASET_REVISION or state.packed_manifest_sha256 != PACKED_MANIFEST_SHA256:
        msg = "resume packed pin does not match launch contract"
        raise RuntimeError(msg)
    if state.eval_revision != EVAL_DATASET_REVISION:
        msg = "resume eval pin does not match launch contract"
        raise RuntimeError(msg)
    if not 0 <= state.packed_cursor <= PACKED_UNIT_COUNT:
        msg = "resume packed cursor is outside the full artifact"
        raise RuntimeError(msg)
    if state.optimizer_step < 0:
        msg = "resume optimizer step is invalid"
        raise RuntimeError(msg)


def render_full_run_command(candidate_key: str) -> str:
    """Render, but never execute, the exact one-candidate Modal invocation.

    The rendered profile is candidate-specific.  PII350 shares a profile with
    Encoder350 but has an independent run root and budget.

    Returns:
        The full dispatch command for that candidate. The contract is rendered first and
        discarded, purely so an unbuildable candidate fails here rather than producing a
        command that would be refused at launch.

    """
    render_full_run(candidate_key)
    inventory = json.dumps(
        {
            "id": EVAL_DATASET_ID,
            "revision": EVAL_DATASET_REVISION,
            "config": EVAL_CONFIG,
            "split": EVAL_SPLIT,
            "rows": EVAL_ROWS,
        },
        separators=(",", ":"),
    )
    qualification = qualification_for(candidate_key)
    placeholder = (
        json.dumps(
            {
                "candidate": candidate_key,
                "batch_size": EXPECTED_FULL_BATCHES[candidate_key],
                **MANUAL_TRANSFER_ACCEPTANCES[candidate_key],
            },
            separators=(",", ":"),
        )
        if qualification.status == "accepted_transfer"
        else json.dumps(BASE230_LIVE_ACCEPTANCE, separators=(",", ":"))
        if qualification.status == "accepted_live"
        else json.dumps(
            {
                "candidate": candidate_key,
                "batch_size": qualification.selected_batch_size,
                "execution_id": qualification.execution_id,
            },
            separators=(",", ":"),
        )
    )
    return (
        f"MODAL_PROFILE={FULL_RUN_MODAL_PROFILES[candidate_key]} uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate {candidate_key} --execute "
        f"--confirmation {FULL_RUN_CONFIRMATION} "
        f"--qualification-artifact-json '{placeholder}' "
        f"--eval-inventory-json '{inventory}'"
    )


def render_full_run_preflight_command(candidate_key: str) -> str:
    """Render the candidate-specific CPU asset-receipt prerequisite only.

    Returns:
        The preflight command for that candidate. It carries no ``--execute`` and no
        confirmation token, which is what makes it safe to run unattended.

    """
    render_full_run(candidate_key)
    return (
        f"MODAL_PROFILE={FULL_RUN_MODAL_PROFILES[candidate_key]} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate {candidate_key} --preflight-assets"
    )


# reason: main coordinates parse args with full run; extra seams would duplicate totals or escaping.
def _main() -> None:  # ruff: ignore[complex-structure,too-many-branches]
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-config", choices=tuple(CANDIDATES))
    parser.add_argument("--render-all", action="store_true")
    parser.add_argument("--render-command", choices=tuple(CANDIDATES))
    parser.add_argument("--render-preflight-command", choices=tuple(CANDIDATES))
    parser.add_argument("--render-base230-oom-fallback", action="store_true")
    parser.add_argument("--render-base230-oom-fallback-command", action="store_true")
    parser.add_argument("--render-pii350-utilization-run", action="store_true")
    parser.add_argument("--render-pii350-utilization-run-command", action="store_true")
    parser.add_argument("--render-pii350-milestone-run", choices=tuple(PII350_MILESTONE_RUNS))
    parser.add_argument(
        "--render-pii350-milestone-run-command",
        choices=tuple(PII350_MILESTONE_RUNS),
    )
    parser.add_argument(
        "--render-pii350-milestone-run-preflight-command",
        choices=tuple(PII350_MILESTONE_RUNS),
    )
    parser.add_argument("--render-pii350-scout-command", choices=tuple(PII350_SCOUT_ARMS))
    parser.add_argument(
        "--render-pii350-scout-preflight-command",
        choices=tuple(PII350_SCOUT_ARMS),
    )
    args = parser.parse_args()
    if args.render_config:
        print(json.dumps(render_full_run(args.render_config), indent=2, sort_keys=True))
    elif args.render_all:
        print(json.dumps(render_all_full_runs(), indent=2, sort_keys=True))
    elif args.render_command:
        print(render_full_run_command(args.render_command))
    elif args.render_preflight_command:
        print(render_full_run_preflight_command(args.render_preflight_command))
    elif args.render_base230_oom_fallback:
        print(json.dumps(render_base230_oom_fallback_contract(), indent=2, sort_keys=True))
    elif args.render_base230_oom_fallback_command:
        print(render_base230_oom_fallback_command())
    elif args.render_pii350_utilization_run:
        print(json.dumps(render_pii350_utilization_run_contract(), indent=2, sort_keys=True))
    elif args.render_pii350_utilization_run_command:
        print(render_pii350_utilization_run_command())
    elif args.render_pii350_milestone_run:
        print(
            json.dumps(
                render_pii350_milestone_run_contract(args.render_pii350_milestone_run),
                indent=2,
                sort_keys=True,
            ),
        )
    elif args.render_pii350_milestone_run_command:
        print(render_pii350_milestone_run_command(args.render_pii350_milestone_run_command))
    elif args.render_pii350_milestone_run_preflight_command:
        print(render_pii350_milestone_run_preflight_command(args.render_pii350_milestone_run_preflight_command))
    elif args.render_pii350_scout_command:
        print(render_pii350_scout_command(args.render_pii350_scout_command))
    elif args.render_pii350_scout_preflight_command:
        print(render_pii350_scout_preflight_command(args.render_pii350_scout_preflight_command))
    else:
        parser.error(
            "choose --render-config, --render-all, --render-command, or "
            "--render-preflight-command, --render-base230-oom-fallback, or "
            "--render-base230-oom-fallback-command, "
            "--render-pii350-utilization-run, or "
            "--render-pii350-utilization-run-command, "
            "--render-pii350-milestone-run, or "
            "--render-pii350-milestone-run-command, or "
            "--render-pii350-milestone-run-preflight-command, "
            "--render-pii350-scout-command, or "
            "--render-pii350-scout-preflight-command",
        )


if __name__ == "__main__":
    _main()
