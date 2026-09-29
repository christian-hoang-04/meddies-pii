"""Persist the BIOES training-setup preview and public configuration API."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .json_narrowing import load_optional_json_object
from .training_setup_preview_config import (
    ARTIFACT_VOLUME,
    CHECKPOINT_EVERY_STEPS,
    DATASET_REVISION,
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASET_SUMMARY_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMOKE_RESULT_JSON,
    DEFAULT_SPLIT_SUMMARY_JSON,
    DEFAULT_TRAIN_JSONL,
    DEFAULT_VALIDATION_JSONL,
    LOCAL_RUN_SLUG,
    PLANNED_STEPS,
    REPORT_SLUG,
    RUN_SLUG,
    TRAIN_ROWS,
    VALIDATION_ROWS,
    _default_hyperparameters,
    _packing_estimate,
    training_command,
)
from .training_setup_preview_html import render_training_setup_preview_html
from .training_setup_preview_style import SCRIPT, STYLE

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ARTIFACT_VOLUME",
    "CHECKPOINT_EVERY_STEPS",
    "DATASET_REVISION",
    "DEFAULT_DATASET_SUMMARY_JSON",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SMOKE_RESULT_JSON",
    "DEFAULT_SPLIT_SUMMARY_JSON",
    "DEFAULT_TRAIN_JSONL",
    "DEFAULT_VALIDATION_JSONL",
    "LOCAL_RUN_SLUG",
    "PLANNED_STEPS",
    "REPORT_SLUG",
    "RUN_SLUG",
    "SCRIPT",
    "STYLE",
    "TRAIN_ROWS",
    "VALIDATION_ROWS",
    "render_html",
    "training_command",
]


# reason: render html exposes train jsonl/sample count as its public contract; bundling would break callers.
def render_html(  # ruff: ignore[too-many-arguments]
    *,
    train_jsonl: Path,
    validation_jsonl: Path,
    split_summary_json: Path,
    dataset_summary_json: Path,
    smoke_result_json: Path | None,
    output_dir: Path,
    sample_count: int = 8,
) -> Path:
    """Write the setup preview and machine-readable launch artifacts.

    Returns:
        The path of the written HTML preview. Every JSON input is loaded through
        ``load_optional_json_object``, so an absent split summary, dataset summary or smoke
        result yields an empty section rather than a failure -- a preview built before the
        smoke run exists is a supported case, not a degraded one. What comes back describes
        the PLANNED launch: the hyperparameters and command are defaults, so nothing here is
        evidence about a run that happened.

    """
    hparams = _default_hyperparameters()
    command = training_command(hparams)
    split_summary = load_optional_json_object(split_summary_json)
    dataset_summary = load_optional_json_object(dataset_summary_json)
    smoke = load_optional_json_object(smoke_result_json)
    estimate = _packing_estimate(hparams, smoke)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = render_training_setup_preview_html(
        train_jsonl=train_jsonl,
        validation_jsonl=validation_jsonl,
        hparams=hparams,
        command=command,
        split_summary=split_summary,
        dataset_summary=dataset_summary,
        estimate=estimate,
        generated_at=generated_at,
        sample_count=sample_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hyperparameters.json").write_text(json.dumps(hparams, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training-command.sh").write_text(command + "\n", encoding="utf-8")
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
