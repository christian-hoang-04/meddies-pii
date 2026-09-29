"""Render train/validation split evidence and language-policy details."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

from meddies_pii.annotations.bioes import ENTITY_LABELS

from .json_narrowing import int_like, int_map, map_at

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .models import TrainingScan


def _split_overview_html(
    split_summary: Mapping[str, Any],
    *,
    train_scan: TrainingScan,
    validation_scan: TrainingScan | None,
) -> str:
    train = map_at(split_summary, "train")
    validation = map_at(split_summary, "validation")
    split_validation = map_at(split_summary, "split_validation")
    train_rows = int_like(train.get("rows") or train_scan.rows_seen)
    validation_rows = int_like(validation.get("rows") or (validation_scan.rows_seen if validation_scan is not None else 0))
    total_rows = int_like(split_summary.get("input_rows") or train_rows + validation_rows)
    rows = [
        ("total rows before split", total_rows),
        ("train rows", train_rows),
        ("validation rows", validation_rows),
        (
            "train/validation id overlap",
            int_like(split_validation.get("train_validation_id_overlap") or 0),
        ),
        (
            "train/validation text overlap",
            int_like(split_validation.get("train_validation_text_overlap") or 0),
        ),
    ]
    return (
        '<table class="dense">'
        "<tr><th>item</th><th>value</th></tr>"
        + "".join(f'<tr><td>{escape(key)}</td><td class="num">{value:,}</td></tr>' for key, value in rows)
        + "</table>"
    )


def _split_count_table(split_summary: Mapping[str, Any], key: str) -> str:
    train = map_at(split_summary, "train")
    validation = map_at(split_summary, "validation")
    train_counts = int_map(train.get(key))
    validation_counts = int_map(validation.get(key))
    if not train_counts and not validation_counts:
        return '<p class="empty">No split count summary attached.</p>'
    all_keys = sorted(set(map(str, train_counts)) | set(map(str, validation_counts)))
    train_total = sum(train_counts.values()) or 1
    validation_total = sum(validation_counts.values()) or 1
    rows = ["<tr><th>value</th><th>train rows</th><th>train %</th><th>validation rows</th><th>validation %</th></tr>"]
    for item in all_keys:
        train_value = train_counts.get(item, 0)
        validation_value = validation_counts.get(item, 0)
        rows.append(
            "<tr>"
            f"<td><code>{escape(item)}</code></td>"
            f'<td class="num">{train_value:,}</td>'
            f'<td class="num">{train_value / train_total * 100:.2f}%</td>'
            f'<td class="num">{validation_value:,}</td>'
            f'<td class="num">{validation_value / validation_total * 100:.2f}%</td>'
            "</tr>",
        )
    return '<table class="dense">' + "".join(rows) + "</table>"


def _split_label_doc_table(split_summary: Mapping[str, Any]) -> str:
    train = map_at(split_summary, "train")
    validation = map_at(split_summary, "validation")
    train_counts = int_map(train.get("label_doc_counts"))
    validation_counts = int_map(validation.get("label_doc_counts"))
    if not train_counts and not validation_counts:
        return '<p class="empty">No label document support summary attached.</p>'
    rows = ["<tr><th>label</th><th>train docs</th><th>validation docs</th></tr>"]
    rows.extend(
        "<tr>"
        f"<td><code>{escape(label)}</code></td>"
        f'<td class="num">{train_counts.get(label, 0):,}</td>'
        f'<td class="num">{validation_counts.get(label, 0):,}</td>'
        "</tr>"
        for label in ENTITY_LABELS
    )
    return '<table class="dense">' + "".join(rows) + "</table>"


def _external_language_policy_html(dataset_summary: Mapping[str, Any]) -> str:
    all_rows_summary = map_at(dataset_summary, "all_rows_summary")
    summary: Mapping[str, object] = all_rows_summary or dataset_summary
    dropped = int_map(map_at(summary, "dropped_external_spans"))
    unsupported = {key: value for key, value in dropped.items() if ":unsupported_language:" in str(key)}
    policy = str(
        dataset_summary.get("language_policy") or "External rows are restricted to Meddies-supported language codes.",
    )
    if not unsupported:
        return f'<p>{escape(policy)}</p><p class="empty">No unsupported-language skips recorded.</p>'
    rows = ["<tr><th>skipped external language bucket</th><th>rows skipped during scan</th></tr>"]
    for key, value in sorted(unsupported.items(), key=lambda item: (-item[1], item[0])):
        rows.append(f'<tr><td><code>{escape(str(key))}</code></td><td class="num">{value:,}</td></tr>')
    return f"<p>{escape(policy)}</p>" + '<table class="dense">' + "".join(rows) + "</table>"
