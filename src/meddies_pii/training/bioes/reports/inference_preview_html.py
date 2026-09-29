"""Render the interactive BIOES inference preview document."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from html import escape
from itertools import starmap
from typing import Any

from meddies_pii.training.bioes.reports.span_records import (
    report_span_from_mapping,
    report_span_mappings,
)

from .inference_preview_metrics import compare_spans, preview_metrics
from .inference_preview_style import SCRIPT, STYLE
from .metric_card import metric_card


def _format_float(value: object, *, digits: int = 3) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_rate(value: object) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{float(value):.3f}"


def _label_css(label: str) -> str:
    normalized = label.replace("_", "-").lower()
    safe = "".join(char if char.isalnum() or char == "-" else "-" for char in normalized)
    return safe.strip("-") or "unknown"


def _highlight_spans(
    text: str,
    spans: Sequence[Mapping[str, object]],
    *,
    role: str,
) -> str:
    report_spans = [report_span_from_mapping(span) for span in spans]
    pieces: list[str] = []
    cursor = 0
    for span in sorted(
        report_spans,
        key=lambda item: (
            item.start,
            item.end,
            item.label,
        ),
    ):
        start = max(0, min(span.start, len(text)))
        end = max(start, min(span.end, len(text)))
        if start < cursor or start == end:
            continue
        label = span.label
        pieces.extend((
            escape(text[cursor:start], quote=False),
            (
                f'<mark class="pii {escape(role)} label-{escape(_label_css(label))}" '
                f'title="{escape(label)} {start}:{end}">'
                f"{escape(text[start:end], quote=False)}"
                f'<span class="tag">{escape(label)}</span>'
                "</mark>"
            ),
        ))
        cursor = end
    pieces.append(escape(text[cursor:], quote=False))
    return "".join(pieces)


def _span_table(spans: Sequence[Mapping[str, object]], *, empty: str) -> str:
    if not spans:
        return f'<p class="subtitle">{escape(empty)}</p>'
    rows = []
    for span in [report_span_from_mapping(span) for span in spans]:
        label = span.label
        rows.append(
            "<tr>"
            f'<td><span class="badge label-{escape(_label_css(label))}">{escape(label)}</span></td>'
            f'<td class="num">{span.start:,}</td>'
            f'<td class="num">{span.end:,}</td>'
            f'<td class="mono">{escape(span.text)}</td>'
            "</tr>",
        )
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Label</th><th>Start</th><th>End</th><th>Text</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _summary_panel(result: Mapping[str, Any], preview: Mapping[str, Any]) -> str:
    config = result.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    preview_meta = preview.get("provenance", {})
    if not isinstance(preview_meta, Mapping):
        preview_meta = {}
    setup = {
        "checkpoint": preview_meta.get("checkpoint"),
        "dataset": {
            "id": config.get("dataset_id"),
            "revision": config.get("dataset_revision"),
            "eval_config": config.get("eval_config"),
            "eval_split": config.get("eval_dataset_split"),
        },
        "model": {
            "model_id": config.get("model_id"),
            "backend": config.get("backend"),
            "max_length": config.get("max_length"),
            "lora_rank": config.get("lora_rank"),
            "lora_alpha": config.get("lora_alpha"),
        },
        "preview": {
            "gpu": preview_meta.get("gpu"),
            "prepared_rows": preview_meta.get("prepared_rows"),
            "source_candidates": preview_meta.get("source_candidates"),
            "prepared_candidates": preview_meta.get("prepared_candidates"),
            "generated_at": preview_meta.get("generated_at"),
        },
    }
    return (
        '<section class="panel"><div class="panel-body">'
        "<h2>Run + preview provenance</h2>"
        f"<pre>{escape(json.dumps(setup, ensure_ascii=False, indent=2, sort_keys=True))}</pre>"
        "</div></section>"
    )


def _record_with_computed_exact(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    gold_spans = report_span_mappings(record.get("gold_spans", []))
    predicted_spans = report_span_mappings(record.get("predicted_spans", []))
    normalized["gold_spans"] = gold_spans
    normalized["predicted_spans"] = predicted_spans
    normalized["exact"] = compare_spans(
        gold_spans,
        predicted_spans,
    )
    return normalized


def _card(record: Mapping[str, Any], index: int) -> str:
    text = str(record["text"])
    gold_spans = report_span_mappings(record.get("gold_spans", []))
    predicted_spans = report_span_mappings(record.get("predicted_spans", []))
    exact = record.get("exact", {})
    if not isinstance(exact, Mapping):
        exact = {}
    uid = str(record.get("uid", f"row-{index}"))
    labels = sorted({
        str(span["label"]) for span in [*gold_spans, *predicted_spans] if isinstance(span, Mapping) and "label" in span
    })
    label_badges = "".join(
        f'<span class="badge label-{escape(_label_css(label))}">{escape(label)}</span>' for label in labels
    )
    status_class = (
        "ok" if int(exact.get("false_positive", 0)) == 0 and int(exact.get("false_negative", 0)) == 0 else "warn"
    )
    status_label = "exact match" if status_class == "ok" else "needs review"
    search_text = " ".join([
        uid,
        text,
        " ".join(labels),
        json.dumps(gold_spans, ensure_ascii=False),
        json.dumps(predicted_spans, ensure_ascii=False),
    ]).lower()
    return f"""
<section class="card" data-search="{escape(search_text, quote=True)}">
  <div class="card-header">
    <div class="title-line">
      <span class="rank">#{index + 1}</span>
      <span class="path">{escape(uid)}</span>
      <span class="badge {status_class}">{escape(status_label)}</span>
      {label_badges}
    </div>
    <div class="meta">
      <span>chars: <strong>{len(text):,}</strong></span>
      <span>gold: <strong>{len(gold_spans):,}</strong></span>
      <span>pred: <strong>{len(predicted_spans):,}</strong></span>
      <span>TP/FP/FN: \
<strong>{exact.get("true_positive", 0)}/{exact.get("false_positive", 0)}/{exact.get("false_negative", 0)}</strong></span>
      <span>row F1: <strong>{_format_rate(exact.get("f1", 0.0))}</strong></span>
    </div>
  </div>
  <div class="card-body">
    <div class="grid-2">
      <div>
        <h2>Gold spans</h2>
        <div class="textbox"><div class="snippet">{_highlight_spans(text, gold_spans, role="gold")}</div></div>
      </div>
      <div>
        <h2>Predicted spans</h2>
        <div class="textbox"><div class="snippet">{_highlight_spans(text, predicted_spans, role="pred")}</div></div>
      </div>
    </div>
    <div class="grid-2" style="margin-top: 14px;">
      <div>
        <h2>False positives</h2>
        {_span_table(list(exact.get("fp_spans", [])), empty="No false positives.")}
      </div>
      <div>
        <h2>False negatives</h2>
        {_span_table(list(exact.get("fn_spans", [])), empty="No false negatives.")}
      </div>
    </div>
  </div>
</section>
"""


def render_inference_preview_html(
    *,
    preview: Mapping[str, Any],
    result: Mapping[str, Any],
    preview_json_label: str,
    generated_at: str | None = None,
) -> str:
    records = preview.get("records", [])
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        msg = "Inference preview payload is invalid: records must be a sequence"
        raise ValueError(msg)
    normalized_records = [_record_with_computed_exact(record) for record in records if isinstance(record, Mapping)]
    preview_metric = preview_metrics(normalized_records)
    full_eval_metrics = {
        "F1": _format_rate(result.get("eval_exact_span_f1")),
        "precision": _format_rate(result.get("eval_exact_span_precision")),
        "recall": _format_rate(result.get("eval_exact_span_recall")),
        "steps": _format_float(result.get("completed_steps")),
        "loss": _format_rate(result.get("final_train_loss")),
        "preview F1": _format_rate(preview_metric["f1"]),
        "preview rows": f"{len(records):,}",
    }
    metrics_html = "".join(starmap(metric_card, full_eval_metrics.items()))
    cards = "\n".join(_card(record, index) for index, record in enumerate(normalized_records))
    generated_at = generated_at or datetime.now(UTC).isoformat()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meddies BIOES inference preview — step 150</title>
  <style>{STYLE}</style>
</head>
<body>
<main>
  <header>
    <h1>Meddies BIOES inference preview — LFM2.5 step 150</h1>
    <p class="subtitle">Gold vs predicted spans from the persisted Modal checkpoint. Full metrics \
are from the completed validation run; preview metrics are only for the rendered subset.</p>
    <div class="metrics">{metrics_html}</div>
  </header>
  {_summary_panel(result, preview)}
  <div class="toolbar">
    <input id="search" type="search" placeholder="Filter by UID, label, or text">
    <div id="visible-count" class="subtitle"></div>
  </div>
  {cards}
  <footer>
    Generated at {escape(generated_at)}. Source JSON: <span class="mono">{escape(preview_json_label)}</span>
  </footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""
