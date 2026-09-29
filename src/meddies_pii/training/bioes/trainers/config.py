from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.json_types import JsonObject

if TYPE_CHECKING:
    from meddies_pii.evaluation.span_metrics import (
        ContainmentSpanMetricBlock,
        ExactSpanMetricBlock,
        MetricDelta,
    )

LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "in_proj",
    "w1",
    "w2",
    "w3",
)
DEFAULT_BACKEND = "unsloth"
DEFAULT_UNSLOTH_MECHANICS: dict[str, Any] = {
    "backend": DEFAULT_BACKEND,
    "load_in_16bit": True,
    "load_in_8bit": False,
    "load_in_4bit": False,
    "full_finetuning": False,
    "fast_inference": False,
    "gradient_checkpointing": "unsloth",
    "head": "Dropout(0.1)+Linear(1024,37,bfloat16)",
    "seed_before_head": 3407,
    "cross_entropy": {
        "logits": "fp32",
        "ignore_index": -100,
        "label_smoothing": 0.0,
        "class_weights": None,
    },
    "lora": {
        "rank": 64,
        "alpha": 128,
        "dropout": 0.0,
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "random_state": 3407,
        "target_modules": list(LORA_TARGET_MODULES),
    },
    "adamw": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "fused": False,
    },
    "scheduler": None,
    "max_length": 8192,
    "gradient_accumulation_steps": 1,
    "length_bucketing": False,
    "UNSLOTH_DISABLE_STATISTICS": "1",
}

type SmokeTrainingPayload = JsonObject


@dataclass(slots=True)
class SmokeTrainingConfig:
    backend: str = DEFAULT_BACKEND
    dataset_id: str = "Meddies/meddies-pii"
    dataset_split: str = "train"
    dataset_revision: str | None = None
    model_id: str = "LiquidAI/LFM2.5-350M-Base"
    model_revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    train_config: str = "train"
    eval_config: str = "test"
    eval_dataset_split: str | None = None
    eval_dataset_id: str | None = None
    """Eval may live in a DIFFERENT HF repo than train (e.g.

    train on meddies-pii-mixed, eval on meddies-pii-v2). When ``eval_dataset_id`` is None the eval loads from
    ``dataset_id`` exactly as before (back-compat).

    """
    eval_dataset_revision: str | None = None
    train_limit: int = 32
    eval_limit: int = 16
    max_length: int = 512
    steps: int | None = 10
    epochs: int | None = None
    logging_steps: int = 1
    checkpoint_every_steps: int = 0
    resume_from_checkpoint: str | None = None
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    fused_adamw: bool = False
    lora_rank: int = 4
    lora_alpha: int | None = None
    packing: bool = False
    length_bucketing: bool = False
    viterbi_transition_biases: dict[str, float] | None = None
    viterbi_calibration_path: str | None = None
    target_eval_slice: str | None = None
    target_eval_limit: int = 4
    target_eval_scan_multiplier: int = 32
    target_eval_require_support: bool = True
    allow_label_repairs: bool = False
    require_label_json: bool = True
    allow_same_local_eval: bool = False

    def resolved_eval_dataset_id(self) -> str:
        """Repo the eval split loads from — ``eval_dataset_id`` if set, else train's.

        Returns:
            The eval repo id, falling back to the training repo so the common single-repo
            case needs no eval-specific configuration.

        """
        return self.eval_dataset_id or self.dataset_id

    def resolved_eval_revision(self) -> str | None:
        """Revision for the eval repo. Independent of train when eval is a separate repo.

        Returns:
            The eval revision when a separate ``eval_dataset_id`` is configured, otherwise
            the TRAIN revision. The pairing is deliberate: pinning one repo to two different
            revisions in the same run would evaluate against data the training pin excludes,
            so the revision follows whichever repo id was resolved.

        """
        if self.eval_dataset_id:
            return self.eval_dataset_revision
        return self.dataset_revision


@dataclass(slots=True)
class SmokeTrainingResult:
    backend: str
    steps: int
    epochs: int | None
    steps_per_epoch: int | None
    fused_adamw: bool
    config: JsonObject
    viterbi_transition_biases: dict[str, float] | None
    mean_train_loss: float
    first_train_loss: float
    final_train_loss: float
    train_loss_delta: float
    train_loss_values: list[float]
    initial_eval_exact_span_f1: float
    initial_eval_exact_span_precision: float
    initial_eval_exact_span_recall: float
    initial_eval_exact_span_untyped_f1: float
    initial_eval_exact_span_untyped_precision: float
    initial_eval_exact_span_untyped_recall: float
    initial_eval_containment_span_f1: float
    initial_eval_containment_span_precision: float
    initial_eval_containment_span_recall: float
    initial_eval_containment_span_untyped_f1: float
    initial_eval_containment_span_untyped_precision: float
    initial_eval_containment_span_untyped_recall: float
    eval_exact_span_f1: float
    eval_exact_span_precision: float
    eval_exact_span_recall: float
    eval_exact_span_untyped_f1: float
    eval_exact_span_untyped_precision: float
    eval_exact_span_untyped_recall: float
    eval_containment_span_f1: float
    eval_containment_span_precision: float
    eval_containment_span_recall: float
    eval_containment_span_untyped_f1: float
    eval_containment_span_untyped_precision: float
    eval_containment_span_untyped_recall: float
    eval_exact_span_f1_delta: float
    eval_containment_span_f1_delta: float
    eval_exact_span_typed: ExactSpanMetricBlock
    eval_exact_span_untyped: ExactSpanMetricBlock
    eval_exact_span_untyped_minus_typed: MetricDelta
    eval_containment_span_typed: ContainmentSpanMetricBlock
    eval_containment_span_untyped: ContainmentSpanMetricBlock
    eval_containment_span_untyped_minus_typed: MetricDelta
    eval_adversarial_slices: JsonObject
    eval_containment_adversarial_slices: JsonObject
    targeted_eval: JsonObject | None
    train_slice_filter_report: JsonObject
    eval_slice_filter_report: JsonObject
    train_examples: int
    eval_examples: int
    train_packed_examples: int | None
    train_packing_utilization: float | None
    train_packing_boundary_token_count: int | None
    packing_attention_probe: JsonObject | None
    train_row_uids: list[str]
    eval_row_uids: list[str]
    train_row_hashes: list[str]
    eval_row_hashes: list[str]
    package_versions: dict[str, str | None]
    cuda_memory: dict[str, float | str | None]
    model_artifact: str | None
    classifier_artifact: str
    latest_checkpoint: str | None
    resume_from_checkpoint: str | None
    completed_steps: int
    notes: list[str]


# reason: Smoke path, file checks, shape limits, and error text form one configuration verdict.
def validate_smoke_training_config(config: SmokeTrainingConfig) -> None:  # ruff: ignore[complex-structure,too-many-branches]
    if config.backend not in {DEFAULT_BACKEND, "hf"}:
        msg = f"Unsupported backend: {config.backend}"
        raise ValueError(msg)
    if config.steps is None and config.epochs is None:
        msg = "either steps or epochs must be set"
        raise ValueError(msg)
    if config.steps is not None and config.steps <= 0:
        msg = "steps must be positive when set"
        raise ValueError(msg)
    if config.epochs is not None and config.epochs <= 0:
        msg = "epochs must be positive when set"
        raise ValueError(msg)
    if config.batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if config.gradient_accumulation_steps <= 0:
        msg = "gradient_accumulation_steps must be positive"
        raise ValueError(msg)
    if config.lora_rank <= 0:
        msg = "lora_rank must be positive"
        raise ValueError(msg)
    if config.lora_alpha is not None and config.lora_alpha <= 0:
        msg = "lora_alpha must be positive when set"
        raise ValueError(msg)
    if config.logging_steps < 0:
        msg = "logging_steps must be >= 0"
        raise ValueError(msg)
    if config.checkpoint_every_steps < 0:
        msg = "checkpoint_every_steps must be >= 0"
        raise ValueError(msg)
    if config.packing and config.backend != "unsloth":
        msg = "packing=True requires backend='unsloth'"
        raise ValueError(msg)
    if config.length_bucketing:
        msg = (
            "length_bucketing=True is disabled until it uses an upstream "
            "trainer/sampler implementation rather than custom local packing."
        )
        raise NotImplementedError(
            msg,
        )
    if Path(config.dataset_id).is_file() and not config.allow_same_local_eval:
        msg = (
            "same local JSONL would be used for train and eval; pass allow_same_local_eval=True only for debug smoke tests"
        )
        raise ValueError(
            msg,
        )
