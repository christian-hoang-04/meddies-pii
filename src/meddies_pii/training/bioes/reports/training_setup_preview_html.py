"""Render the training-setup preview HTML document."""

from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape
from typing import TYPE_CHECKING, Any

from meddies_pii.annotations.bioes import ENTITY_LABELS
from meddies_pii.annotations.span_records import BIOES_LABELS
from meddies_pii.training.bioes.reports.json_narrowing import int_map, map_at
from meddies_pii.training.bioes.reports.metric_card import metric_card

from .training_setup_preview_config import ARTIFACT_VOLUME, DATASET_REVISION
from .training_setup_preview_data import (
    _label_css,
    _sample_cards,
    _top_count_rows,
)
from .training_setup_preview_style import SCRIPT, STYLE

if TYPE_CHECKING:
    from pathlib import Path


def _estimate_html(estimate: Mapping[str, Any]) -> str:
    if not estimate:
        return '<p class="subtitle">No smoke result attached for packing estimate.</p>'
    rows = "".join(
        f'<tr><td>{escape(str(key))}</td><td class="num">{escape(str(value))}</td></tr>' for key, value in estimate.items()
    )
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Signal</th><th>Value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        '<p class="subtitle">Estimate scales the 4096-context packed smoke to 8192. '
        "Actual packed units are logged after Modal prepares the capped train split.</p>"
    )


def _label_counts_html(dataset_summary: Mapping[str, Any]) -> str:
    all_rows = map_at(dataset_summary, "all_rows_summary")
    if not all_rows:
        return '<p class="subtitle">No dataset summary attached.</p>'
    counts = int_map(all_rows.get("label_span_counts"))
    if not counts:
        return '<p class="subtitle">Dataset summary has no label_span_counts.</p>'
    rows = [
        "<tr>"
        f'<td><span class="badge label-{_label_css(label)}">{escape(label)}</span></td>'
        f'<td class="num">{counts.get(label, 0):,}</td>'
        "</tr>"
        for label in ENTITY_LABELS
    ]
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Label</th><th>Span count</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _split_summary_html(split_summary: Mapping[str, Any]) -> str:
    train = map_at(split_summary, "train")
    validation = map_at(split_summary, "validation")
    checks = map_at(split_summary, "split_validation")
    payload = {
        "train_rows": train.get("rows"),
        "validation_rows": validation.get("rows"),
        "train_validation_id_overlap": checks.get("train_validation_id_overlap"),
        "train_validation_text_overlap": checks.get("train_validation_text_overlap"),
        "prior_max_length_validation": checks.get("max_length"),
        "train_rows_over_prior_max_length": checks.get("train_rows_over_max_length"),
        "validation_rows_over_prior_max_length": checks.get("validation_rows_over_max_length"),
    }
    rows = "".join(
        f'<tr><td>{escape(str(key))}</td><td class="num">{escape(str(value))}</td></tr>' for key, value in payload.items()
    )
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Check</th><th>Value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


# reason: render training exposes train jsonl/sample count as its public contract; bundling would break callers.
def render_training_setup_preview_html(  # ruff: ignore[too-many-arguments]
    *,
    train_jsonl: Path,
    validation_jsonl: Path,
    hparams: Mapping[str, Any],
    command: str,
    split_summary: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    estimate: Mapping[str, Any],
    generated_at: str,
    sample_count: int = 8,
) -> str:
    all_rows = dataset_summary.get("all_rows_summary", {})
    domain_counts = all_rows.get("domain_bucket_counts", {}) if isinstance(all_rows, Mapping) else {}
    source_counts = all_rows.get("source_counts", {}) if isinstance(all_rows, Mapping) else {}

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meddies PII BIOES full training setup</title>
  <style>{STYLE}</style>
</head>
<body>
<main>
<header>
  <h1>Meddies PII — BIOES full training setup</h1>
  <p class="subtitle">Generated {escape(generated_at)}. Status: \
<strong>{escape(hparams["status"])}</strong>. This is the pre-launch command/config artifact for \
the next H100 run.</p>
  <div class="metrics">
    {metric_card("Context", hparams["training"]["max_length"])}
    {metric_card("Batch", hparams["training"]["batch_size"])}
    {metric_card("LoRA", "r128 / alpha256")}
    {metric_card("Train cap", f"{hparams['data']['train_limit']:,}")}
    {metric_card("Validation rows", f"{hparams['data']['eval_limit']:,}")}
    {metric_card("Steps", hparams["training"]["steps"])}
    {metric_card("BIOES classes", len(BIOES_LABELS))}
  </div>
</header>

<section class="setup">
  <div class="setup-grid">
    <div class="setup-panel">
      <h2>Training command</h2>
      <div class="command"><pre>{escape(command)}</pre></div>
    </div>
    <div class="setup-panel">
      <h2>Hyperparameters</h2>
      <div class="textbox"><pre>{escape(json.dumps(hparams, ensure_ascii=False, indent=2))}</pre></div>
    </div>
  </div>
</section>

<section class="setup">
  <div class="setup-grid">
    <div class="setup-panel">
      <h2>Launch rationale</h2>
      <div class="textbox"><pre>{
        escape(
            "Budget-bounded chunk: 8192 context, physical batch "
            "128, LoRA r128/alpha256, --steps 120, "
            "--logging-steps 1, --checkpoint-every-steps 10. Each "
            "checkpoint is committed to the "
            "Modal Volume so another account can resume from "
            "/artifacts/.../checkpoints/step-XXXXX."
        )
    }</pre></div>
    </div>
    <div class="setup-panel">
      <h2>Packing / step estimate</h2>
      {_estimate_html(estimate)}
    </div>
  </div>
</section>

<section class="setup">
  <div class="setup-grid">
    <div class="setup-panel">
      <h2>Dataset split checks</h2>
      {_split_summary_html(split_summary)}
    </div>
    <div class="setup-panel">
      <h2>Label span counts</h2>
      {_label_counts_html(dataset_summary)}
    </div>
  </div>
</section>

<section class="setup">
  <div class="setup-grid">
    <div class="setup-panel">
      <h2>Domain mix</h2>
      {_top_count_rows(domain_counts)}
    </div>
    <div class="setup-panel">
      <h2>Top source datasets</h2>
      {_top_count_rows(source_counts)}
    </div>
  </div>
</section>

<section class="setup">
  <div class="setup-panel">
    <h2>Sample browser</h2>
    <div class="command"><input id="search" type="search" placeholder="Search sample text, label, \
or id…" style="width:100%;padding:10px;border-radius:12px;border:1px solid \
#24405f;background:#050914;color:#e5edf8;"><p id="visible-count" class="subtitle"></p></div>
  </div>
</section>

<h3>Training samples</h3>
{_sample_cards(train_jsonl, sample_count=sample_count, split="train")}

<h3>Validation samples</h3>
{_sample_cards(validation_jsonl, sample_count=min(sample_count, 4), split="validation")}

<footer>
  Contract: train the pinned <code>Meddies/meddies-pii</code> / <code>pii-bioes</code> revision \
<code>{DATASET_REVISION}</code>; persist adapter + classifier to Modal Volume \
<code>{ARTIFACT_VOLUME}</code>; capture local raw log and JSON result.
</footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""
