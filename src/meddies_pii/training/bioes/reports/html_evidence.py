"""Render examples, audit evidence, smoke results, and parse errors."""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING, Any

from meddies_pii.annotations.bioes import ENTITY_LABELS

from .json_narrowing import map_at

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .models import LabelExample, RowPreview


INVALID_ROW_PREVIEW_LIMIT = 200


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)] + "…"


def _css_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")


def _decision_count_table(counts: Mapping[str, int]) -> str:
    if not counts:
        return '<p class="empty">No decision summary attached.</p>'
    rows = [
        "<tr><th>decision:rule</th><th>count</th></tr>",
        *[f'<tr><td><code>{escape(key)}</code></td><td class="num">{value:,}</td></tr>' for key, value in counts.items()],
    ]
    return '<table class="dense">' + "".join(rows) + "</table>"


def _label_examples_html(
    examples_by_label: Mapping[str, Sequence[LabelExample]],
) -> str:
    chunks: list[str] = []
    for label in ENTITY_LABELS:
        examples = examples_by_label.get(label, [])
        if not examples:
            chunks.append(
                f'<details class="label-section"><summary><code>{escape(label)}</code> — no examples</summary></details>',
            )
            continue
        rows = [
            "<tr><th>row</th><th>id</th><th>span</th><th>context</th></tr>",
            *[
                "<tr>"
                f'<td class="num">{example.row_index:,}</td>'
                f"<td><code>{escape(example.example_id)}</code></td>"
                f"<td><code>{escape(_truncate_text(example.span_text, 120))}</code></td>"
                f'<td class="snippet">{example.snippet_html}</td>'
                "</tr>"
                for example in examples
            ],
        ]
        chunks.append(
            f'<details class="label-section" open><summary><code>{escape(label)}</code> — \
{len(examples)} examples</summary>'
            f'<table class="examples">{"".join(rows)}</table></details>',
        )
    return "".join(chunks)


def _row_preview_table(rows: Sequence[RowPreview]) -> str:
    if not rows:
        return '<p class="empty">No rows attached.</p>'
    html_rows = ["<tr><th>row</th><th>id</th><th>labels</th><th>chars</th><th>spans</th><th>snippet</th></tr>"]
    for row in rows:
        labels = " ".join(f'<span class="pill label-{_css_label(label)}">{escape(label)}</span>' for label in row.labels)
        html_rows.append(
            "<tr>"
            f'<td class="num">{row.row_index:,}</td>'
            f"<td><code>{escape(row.example_id)}</code></td>"
            f"<td>{labels}</td>"
            f'<td class="num">{row.text_len:,}</td>'
            f'<td class="num">{row.span_count:,}</td>'
            f'<td class="snippet">{row.snippet_html}</td>'
            "</tr>",
        )
    return '<table class="examples">' + "".join(html_rows) + "</table>"


def _audit_samples_html(
    audit_samples: Mapping[str, Sequence[Mapping[str, object]]],
) -> str:
    if not audit_samples:
        return '<p class="empty">No audit samples attached.</p>'
    sections: list[str] = []
    for rule, samples in audit_samples.items():
        rows = ["<tr><th>uid</th><th>status</th><th>old</th><th>new</th><th>text</th></tr>"]
        rows.extend(
            "<tr>"
            f"<td><code>{escape(str(sample.get('uid', '')))}</code></td>"
            f"<td>{escape(str(sample.get('status', '')))}</td>"
            f"<td><code>{escape(str(sample.get('old_label', '')))}</code></td>"
            f"<td><code>{escape(str(sample.get('new_label', '')))}</code></td>"
            f"<td><code>{escape(_truncate_text(str(sample.get('text', '')), 180))}</code></td>"
            "</tr>"
            for sample in samples
        )
        sections.append(
            f"<details><summary><code>{escape(rule)}</code> — {len(samples)} samples</summary>"
            f'<table class="dense">{"".join(rows)}</table></details>',
        )
    return "".join(sections)


def _modal_summary(modal: Mapping[str, Any]) -> dict[str, object]:
    result = map_at(modal, "result")
    provenance = map_at(modal, "provenance")
    config = map_at(result, "config")
    return {
        "model_id": config.get("model_id"),
        "dataset_id": config.get("dataset_id"),
        "dataset_split": config.get("dataset_split"),
        "train_config": config.get("train_config"),
        "eval_config": config.get("eval_config"),
        "eval_dataset_split": config.get("eval_dataset_split"),
        "backend": result.get("backend") or config.get("backend"),
        "steps": result.get("steps") or config.get("steps"),
        "max_length": config.get("max_length"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "train_examples": result.get("train_examples"),
        "eval_examples": result.get("eval_examples"),
        "first_train_loss": result.get("first_train_loss"),
        "final_train_loss": result.get("final_train_loss"),
        "train_loss_delta": result.get("train_loss_delta"),
        "train_loss_values": result.get("train_loss_values"),
        "train_row_uids": result.get("train_row_uids"),
        "eval_row_uids": result.get("eval_row_uids"),
        "modal_url": provenance.get("modal_url"),
    }


def _modal_html(summary: Mapping[str, object]) -> str:
    if not summary or all(value is None for value in summary.values()):
        return '<p class="empty">No Modal smoke result attached.</p>'
    rows = []
    for key, value in summary.items():
        display = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value)
        if value is None:
            display = "n/a"
        if key == "modal_url" and value:
            display = f'<a href="{escape(str(value))}">{escape(str(value))}</a>'
        else:
            display = escape(display)
        rows.append(f"<tr><td><code>{escape(key)}</code></td><td>{display}</td></tr>")
    return '<table class="dense">' + "".join(rows) + "</table>"


def _invalid_rows_html(invalid_rows: Sequence[str]) -> str:
    if not invalid_rows:
        return '<p class="empty good">No invalid rows.</p>'
    items = "".join(f"<li><code>{escape(item)}</code></li>" for item in invalid_rows[:200])
    extra = "" if len(invalid_rows) <= INVALID_ROW_PREVIEW_LIMIT else f"<p>... {len(invalid_rows) - 200} more</p>"
    return f"<ul>{items}</ul>{extra}"
