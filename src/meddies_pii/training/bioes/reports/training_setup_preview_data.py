"""Read and render source-data fragments for the training setup preview."""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.annotations.span_rendering import snippet_html
from meddies_pii.training.bioes.reports.json_narrowing import int_like

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _label_css(label: str) -> str:
    return label.replace("_", "-")


def _top_count_rows(counts: Mapping[str, object], *, limit: int = 12) -> str:
    rows = []
    normalized_counts = {key: int_like(value) for key, value in counts.items()}
    total = sum(normalized_counts.values())
    for key, value in sorted(normalized_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        share = (value / total * 100) if total else 0.0
        rows.append(f'<tr><td>{escape(key)}</td><td class="num">{value:,}</td><td class="num">{share:.1f}%</td></tr>')
    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Bucket</th><th>Count</th><th>Share</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _sample_cards(jsonl_path: Path, *, sample_count: int, split: str) -> str:
    cards: list[str] = []
    if not jsonl_path.exists():
        return '<p class="subtitle">No local JSONL attached for sample preview.</p>'
    with jsonl_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if len(cards) >= sample_count:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            example_id, text, spans = parse_labeled_record(record, default_id=f"{split}-{row_index}")
            labels = tuple(sorted({span.label for span in spans}))
            label_badges = "".join(
                f'<span class="badge label-{_label_css(label)}">{escape(label)}</span>' for label in labels
            )
            rendered = snippet_html(
                text,
                spans,
                focus_span=spans[0] if spans else None,
                context=320,
            )
            search = " ".join([split, example_id, text, " ".join(labels)]).lower()
            cards.append(
                f"""
<section class="card" data-search="{escape(search, quote=True)}">
  <div class="card-header">
    <div class="title-line">
      <span class="rank">#{len(cards) + 1}</span>
      <span class="path">{escape(split)} / {escape(example_id)}</span>
      {label_badges}
    </div>
    <div class="meta">
      <span>row_index: <strong>{row_index}</strong></span>
      <span>chars: <strong>{len(text):,}</strong></span>
      <span>spans: <strong>{len(spans):,}</strong></span>
    </div>
  </div>
  <div class="card-body">
    <h2>Highlighted BIOES span preview</h2>
    <div class="textbox"><div class="snippet">{rendered}</div></div>
  </div>
</section>
""",
            )
    return "\n".join(cards)
