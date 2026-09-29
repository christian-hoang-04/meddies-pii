"""Compose the static BIOES training-readiness HTML document."""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: the multiplication sign is typography meaning "by" in an operator-facing report label
# reason: (source x language x bucket, 2xA100); the ASCII letter x would misrender the heading.
from datetime import UTC, datetime
from html import escape
from typing import TYPE_CHECKING, Any

from meddies_pii.annotations.bioes import ENTITY_LABELS
from meddies_pii.annotations.span_records import BIOES_LABELS

from .html_evidence import (
    _audit_samples_html,
    _decision_count_table,
    _invalid_rows_html,
    _label_examples_html,
    _modal_html,
    _modal_summary,
    _row_preview_table,
)
from .html_overview import (
    _label_count_table,
    _metadata_count_table,
    _metric_card,
    _readiness_items,
    _readiness_li,
    _summary_counts,
    _summary_decision_counts,
    _summary_label_counts,
    _target_mix_table,
    _validation_rows_note,
    _validation_rows_value,
    _validation_spans_value,
)
from .html_splits import (
    _external_language_policy_html,
    _split_count_table,
    _split_label_doc_table,
    _split_overview_html,
)
from .html_style import _css
from .html_training import (
    _adapter_config_html,
    _complete_config_json_html,
    _hyperparameter_table,
    _launch_decision_config_html,
    _modal_command_html,
    _training_stack_html,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .models import RowPreview, TrainingReportOptions, TrainingScan


# reason: render training exposes options/progress as its public contract; bundling would break callers.
def render_training_report_html(  # ruff: ignore[too-many-arguments]
    *,
    options: TrainingReportOptions,
    scan: TrainingScan,
    validation_scan: TrainingScan | None,
    summary: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    audit_samples: Mapping[str, Sequence[Mapping[str, object]]],
    adversarial: Sequence[RowPreview],
    modal: Mapping[str, Any],
    todo_text: str,
    progress_text: str,
) -> str:
    """Render the report as a complete HTML document.

    Returns:
        The whole document as one string, self-contained with no external asset references,
        so the caller can write it anywhere and it renders. The generation timestamp is
        stamped here at second resolution, so two renders of identical inputs differ.

    """
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    modal_summary = _modal_summary(modal)
    readiness = _readiness_items(
        scan=scan,
        validation_scan=validation_scan,
        summary=summary,
        modal_summary=modal_summary,
    )
    label_counts = _summary_label_counts(summary, scan)
    decision_counts = _summary_decision_counts(summary)
    if options.include_all_rows:
        row_preview_note = (
            f"showing all {len(scan.row_previews):,} rows as compact highlighted snippets. "
            "Use browser find for labels, ids, or text."
        )
    elif options.row_table_limit <= 0:
        row_preview_note = "row browser disabled for compact report; use the per-label examples above."
    else:
        row_preview_note = (
            f"showing first {len(scan.row_previews):,} rows as compact highlighted snippets. "
            "Use browser find for labels, ids, or text."
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(options.title)}</title>
  <style>{_css()}</style>
</head>
<body>
<header>
  <p class="eyebrow">pre-launch review artifact</p>
  <h1>{escape(options.title)}</h1>
  <p class="sub">Generated {escape(generated_at)} from <code>{escape(str(options.train_jsonl))}</code>.</p>
</header>
<main>
  <section class="grid cards">
    {_metric_card("Train rows", f"{scan.rows_seen:,}", f"invalid parsed rows: {len(scan.invalid_rows):,}")}
    {_metric_card("Validation rows", _validation_rows_value(validation_scan), _validation_rows_note(validation_scan))}
    {
        _metric_card(
            "Train spans",
            f"{scan.span_count:,}",
            f"{len(ENTITY_LABELS)} labels / \
{len(BIOES_LABELS)} BIOES classes",
        )
    }
    {_metric_card("Validation spans", _validation_spans_value(validation_scan), "held-out label-list spans")}
  </section>

  <section>
    <h2>Readiness gates</h2>
    <ul class="checks">
      {"".join(_readiness_li(item) for item in readiness)}
    </ul>
  </section>

  <section>
    <h2>Training stack and hyperparameters</h2>
    {_training_stack_html(modal, modal_summary)}
    <h3>Launch decision config</h3>
    {_launch_decision_config_html(modal, dataset_summary=dataset_summary)}
    <h3>Adapter / LoRA config</h3>
    {_adapter_config_html(modal)}
    <h3>Hyperparameters from attached Modal smoke</h3>
    {_hyperparameter_table(modal)}
    <h3>Complete attached smoke config JSON</h3>
    {_complete_config_json_html(modal)}
    <h3>Modal command preview</h3>
    {_modal_command_html(modal)}
  </section>

  <section>
    <h2>Dataset split breakdown</h2>
    {_split_overview_html(split_summary, train_scan=scan, validation_scan=validation_scan)}
    <h3>Train vs validation — domain bucket</h3>
    {_split_count_table(split_summary, "domain_bucket_counts")}
    <h3>Train vs validation — language bucket</h3>
    {_split_count_table(split_summary, "language_bucket_counts")}
    <h3>Train vs validation — source dataset</h3>
    {_split_count_table(split_summary, "source_counts")}
    <h3>Train vs validation — document support per label</h3>
    {_split_label_doc_table(split_summary)}
    <h3>External language policy</h3>
    {_external_language_policy_html(dataset_summary)}
  </section>

  <section>
    <h2>Label distribution</h2>
    {_label_count_table(label_counts)}
  </section>

  <section>
    <h2>Language and source mix</h2>
    {_target_mix_table(summary)}
    <h3>Domain bucket</h3>
    {_metadata_count_table(_summary_counts(summary, "domain_bucket_counts"), total=scan.rows_seen)}
    <h3>Language bucket</h3>
    {_metadata_count_table(_summary_counts(summary, "language_bucket_counts"), total=scan.rows_seen)}
    <h3>Source dataset / field</h3>
    {
        _metadata_count_table(
            _summary_counts(summary, "source_counts") or _summary_counts(summary, "kept_source_counts"),
            total=scan.rows_seen,
        )
    }
    <h3>Raw language field</h3>
    {
        _metadata_count_table(
            _summary_counts(summary, "language_counts") or _summary_counts(summary, "kept_language_counts"),
            total=scan.rows_seen,
        )
    }
    <h3><code>other</code> bucket raw-language breakdown</h3>
    {
        _metadata_count_table(
            _summary_counts(summary, "other_language_counts"),
            total=max(1, _summary_counts(summary, "language_bucket_counts").get("other", 0)),
        )
    }
    <h3>Source × language × bucket</h3>
    {_metadata_count_table(_summary_counts(summary, "source_language_counts"), total=scan.rows_seen)}
  </section>

  <section>
    <h2>Migration decision summary</h2>
    {_decision_count_table(decision_counts)}
  </section>

  <section>
    <h2>Real training examples by label</h2>
    <p class="hint">Up to {options.rows_per_label} examples per label, scanned from the real training JSONL.</p>
    {_label_examples_html(scan.label_examples)}
  </section>

  <section>
    <h2>Held-out validation examples by label</h2>
    <p class="hint">Up to {options.rows_per_label} examples per label, scanned from the held-out validation JSONL.</p>
    {
        _label_examples_html(validation_scan.label_examples)
        if validation_scan
        else '<p class="empty">No validation JSONL attached.</p>'
    }
  </section>

  <section>
    <h2>Adversarial / edge-case examples</h2>
    <p class="hint">Synthetic OPF-style edge cases that should be mixed into validation/training deliberately.</p>
    {_row_preview_table(adversarial)}
  </section>

  <section>
    <h2>Audit samples by migration rule</h2>
    <p class="hint">Bounded samples from <code>{escape(str(options.audit_jsonl or "none"))}</code>; \
counts above come from summary.</p>
    {_audit_samples_html(audit_samples)}
  </section>

  <section>
    <h2>Modal smoke evidence</h2>
    {_modal_html(modal_summary)}
  </section>

  <section>
    <h2>Training row browser</h2>
    <p class="hint">{escape(row_preview_note)}</p>
    {_row_preview_table(scan.row_previews)}
  </section>

  <section>
    <h2>Open handoff notes</h2>
    <details open><summary>TODO</summary><pre>{escape(todo_text)}</pre></details>
    <details><summary>Progress</summary><pre>{escape(progress_text)}</pre></details>
  </section>

  <section>
    <h2>Invalid row parse errors</h2>
    {_invalid_rows_html(scan.invalid_rows)}
  </section>
</main>
</body>
</html>
"""
