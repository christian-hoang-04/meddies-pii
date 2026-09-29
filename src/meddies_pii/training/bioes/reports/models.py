from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from meddies_pii.annotations.bioes import ENTITY_LABELS

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MODAL_PROFILE = "openmedical"


def _planned_training_config(max_length: object = 4096) -> dict[str, object]:
    """Return the intended full-run config when no Modal smoke JSON is attached.

    Returns:
        The planned config, tagged ``config_source =
        "planned_default_no_modal_result_attached"`` so a reader can tell a PLANNED report
        from one backed by a real run. Every other field is a hard-coded intent rather than
        an observation, so nothing here is evidence that a run used these values.

    """
    return {
        "config_source": "planned_default_no_modal_result_attached",
        "model_id": "LiquidAI/LFM2.5-350M-Base",
        "backend": "unsloth",
        "dataset_id": "Meddies/meddies-pii",
        "dataset_split": "train",
        "train_config": "pii-bioes",
        "eval_config": "pii-bioes",
        "eval_dataset_split": "validation",
        "max_length": max_length,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "packing": True,
        "length_bucketing": False,
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "fused_adamw": False,
        "trust_remote_code": False,
        "allow_label_repairs": False,
        "require_label_json": True,
    }


@dataclass(frozen=True, slots=True)
class TrainingReportOptions:
    """Configuration for a static training-readiness report."""

    train_jsonl: Path
    output_html: Path
    validation_jsonl: Path | None = None
    summary_json: Path | None = None
    split_summary_json: Path | None = None
    dataset_summary_json: Path | None = None
    audit_jsonl: Path | None = None
    adversarial_jsonl: Path | None = None
    modal_result_json: Path | None = None
    todo_md: Path | None = None
    progress_md: Path | None = None
    title: str = "Meddies Labels BIOES Training Readiness Report"
    rows_per_label: int = 3
    audit_samples_per_rule: int = 3
    row_table_limit: int = 0
    include_all_rows: bool = False
    snippet_context: int = 180


@dataclass(slots=True)
class LabelExample:
    row_index: int
    example_id: str
    label: str
    span_text: str
    snippet_html: str


@dataclass(slots=True)
class RowPreview:
    row_index: int
    example_id: str
    labels: tuple[str, ...]
    text_len: int
    span_count: int
    snippet_html: str


@dataclass(slots=True)
class TrainingScan:
    rows_seen: int = 0
    span_count: int = 0
    invalid_rows: list[str] = field(default_factory=list)
    label_counts: Counter[str] = field(default_factory=Counter)
    label_examples: dict[str, list[LabelExample]] = field(default_factory=lambda: {label: [] for label in ENTITY_LABELS})
    row_previews: list[RowPreview] = field(default_factory=list)
