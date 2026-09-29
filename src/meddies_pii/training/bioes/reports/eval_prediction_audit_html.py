"""Render the evaluation-prediction audit HTML document."""

from __future__ import annotations

import json
from collections import Counter
from html import escape
from itertools import starmap
from typing import TYPE_CHECKING, Any

from meddies_pii.training.bioes.eval.audit import EvalAuditIssue, summarize_issues

from .eval_prediction_metrics import metric_scenarios
from .metric_card import metric_card

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

STYLE = """
:root {
  color-scheme: dark;
  --bg: #070a12;
  --panel: #0d1220;
  --line: #213047;
  --line-soft: #172136;
  --text: #e7edf8;
  --muted: #98a6bd;
  --blue: #7dd3fc;
  --green: #86efac;
  --red: #fca5a5;
  --amber: #fbbf24;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 12% -10%, rgba(56, 189, 248, 0.18), transparent 34rem),
    radial-gradient(circle at 92% 0%, rgba(251, 191, 36, 0.10), transparent 30rem),
    var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
main { max-width: 1680px; margin: 0 auto; padding: 32px; }
header {
  position: sticky;
  top: 0;
  z-index: 5;
  margin: -32px -32px 24px;
  padding: 28px 32px 22px;
  border-bottom: 1px solid var(--line);
  background: rgba(7, 10, 18, 0.92);
  backdrop-filter: blur(18px);
}
h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: -0.045em; }
h2 {
  margin: 0 0 12px;
  color: var(--muted);
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.subtitle { margin: 0; color: var(--muted); font-size: 14px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
  margin-top: 20px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(13, 18, 32, 0.8);
}
.metric .label {
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.metric .value { margin-top: 4px; font-size: 18px; font-weight: 780; }
.panel {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: rgba(13, 18, 32, 0.8);
  margin: 20px 0;
  overflow: hidden;
}
.panel-body { padding: 18px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.table-wrap {
  overflow: auto;
  border: 1px solid var(--line-soft);
  border-radius: 16px;
  background: #050914;
}
table { width: 100%; border-collapse: collapse; }
th, td { border-top: 1px solid var(--line-soft); padding: 8px 10px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 11px; letter-spacing: .09em; text-transform: uppercase; }
td { font-size: 13px; }
.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin: 20px 0;
}
input[type="search"] {
  width: min(560px, 100%);
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #070b14;
  color: var(--text);
  padding: 11px 15px;
  outline: none;
}
.card {
  border: 1px solid var(--line);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(8, 13, 25, 0.96));
  box-shadow: 0 16px 70px rgba(0, 0, 0, 0.28);
  margin: 18px 0;
  overflow: hidden;
}
.card-header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 16px 18px;
  border-bottom: 1px solid var(--line);
}
.title-line { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 38px;
  height: 32px;
  border-radius: 999px;
  background: #172554;
  color: #bfdbfe;
  font-weight: 850;
}
.path { color: #dbeafe; font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.badge {
  display: inline-flex;
  align-items: center;
  border: 1px solid #24405f;
  border-radius: 999px;
  background: #111c33;
  color: #c7d2fe;
  padding: 5px 9px;
  font-size: 12px;
  font-weight: 740;
}
.badge.high { color: var(--red); border-color: #6f2a2a; background: #271111; }
.badge.medium { color: var(--amber); border-color: #7c5f1d; background: #281f0d; }
.badge.low { color: var(--green); border-color: #1f5f3f; background: #0f2419; }
.meta { color: var(--muted); font-size: 12px; text-align: right; }
.card-body { padding: 18px; }
.context {
  border: 1px solid var(--line-soft);
  border-radius: 16px;
  background: #050914;
  padding: 14px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: #dce8f8;
  font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
mark.issue {
  background: #fbbf24;
  color: #0f172a;
  padding: 1px 4px;
  border-radius: 6px;
  box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.22);
}
pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 460px;
  overflow: auto;
  color: #dbeafe;
  font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
footer { color: var(--muted); padding: 30px 0 12px; }
@media (max-width: 1280px) {
  .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .grid-2 { grid-template-columns: 1fr; }
  .toolbar { align-items: stretch; flex-direction: column; }
}
"""

SCRIPT = """
const input = document.querySelector('#search');
const cards = Array.from(document.querySelectorAll('.card[data-search]'));
const visibleCount = document.querySelector('#visible-count');
function update() {
  const query = input.value.trim().toLowerCase();
  let visible = 0;
  for (const card of cards) {
    const show = !query || card.dataset.search.includes(query);
    card.style.display = show ? '' : 'none';
    if (show) visible += 1;
  }
  visibleCount.textContent = `${visible} / ${cards.length} issues shown`;
}
input.addEventListener('input', update);
update();
"""


def _count_table(title: str, counts: Mapping[str, int]) -> str:
    rows = []
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(f'<tr><td>{escape(key)}</td><td class="num">{value:,}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="2">No issues.</td></tr>')
    return (
        "<section>"
        f"<h2>{escape(title)}</h2>"
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Bucket</th><th>Count</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</section>"
    )


def _context_html(record: Mapping[str, object], issue: EvalAuditIssue) -> str:
    text = str(record.get("text", ""))
    start = max(0, min(issue.start, len(text)))
    end = max(start, min(issue.end, len(text)))
    window_start = max(0, start - 220)
    window_end = min(len(text), end + 220)
    pieces = []
    if window_start > 0:
        pieces.append("…")
    pieces.extend((
        escape(text[window_start:start], quote=False),
        f'<mark class="issue">{escape(text[start:end], quote=False)}</mark>',
        escape(text[end:window_end], quote=False),
    ))
    if window_end < len(text):
        pieces.append("…")
    return "".join(pieces)


def _issue_card(
    issue: EvalAuditIssue,
    *,
    index: int,
    record: Mapping[str, object],
) -> str:
    issue_dict = issue.to_dict()
    search = " ".join([
        issue.uid,
        issue.issue_type,
        issue.recommended_action,
        issue.label,
        issue.text,
        issue.reason,
        str(issue.evidence),
    ]).lower()
    return f"""
<section class="card" data-search="{escape(search, quote=True)}">
  <div class="card-header">
    <div class="title-line">
      <span class="rank">#{index + 1}</span>
      <span class="path">{escape(issue.uid)}</span>
      <span class="badge {escape(issue.severity)}">{escape(issue.severity)}</span>
      <span class="badge">{escape(issue.issue_type)}</span>
      <span class="badge">{escape(issue.label)}</span>
    </div>
    <div class="meta">
      <div>{escape(issue.recommended_action)}</div>
      <div>{issue.start:,}:{issue.end:,}</div>
    </div>
  </div>
  <div class="card-body">
    <h2>Context</h2>
    <div class="context">{_context_html(record, issue)}</div>
    <div class="grid-2" style="margin-top: 14px;">
      <div>
        <h2>Reason</h2>
        <pre>{escape(issue.reason)}</pre>
      </div>
      <div>
        <h2>Evidence</h2>
        <pre>{escape(json.dumps(issue_dict["evidence"], ensure_ascii=False, indent=2, sort_keys=True))}</pre>
      </div>
    </div>
  </div>
</section>
"""


def render_eval_prediction_audit_html(
    *,
    records: Sequence[Mapping[str, Any]],
    issues: Sequence[EvalAuditIssue],
    predictions_json: Path,
    max_issues: int,
    generated_at: str,
) -> str:
    summary = summarize_issues(issues)
    scenarios = metric_scenarios(records, issues)
    record_by_uid = {str(record.get("uid", "unknown")): record for record in records}
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    rendered_issues = sorted(
        issues,
        key=lambda issue: (
            severity_rank[issue.severity],
            issue.recommended_action,
            issue.uid,
            issue.start,
        ),
    )[:max_issues]
    metrics = {
        "records": f"{len(records):,}",
        "issues": f"{len(issues):,}",
        "high": f"{summary['by_severity'].get('high', 0):,}",
        "raw F1": f"{float(scenarios['raw_exact']['f1']):.3f}",
        "adj F1": (f"{float(scenarios['if_accept_prediction_gold_adds_and_suspicious_gold_removals']['f1']):.3f}"),
        "pred add": f"{summary['by_action'].get('review_gold_add_predicted_span', 0):,}",
        "regex cand": f"{summary['by_action'].get('review_regex_candidate', 0):,}",
        "gold remove": f"{summary['by_action'].get('review_gold_remove_or_relabel', 0):,}",
        "rendered": f"{len(rendered_issues):,}",
    }
    metrics_html = "".join(starmap(metric_card, metrics.items()))
    by_type = Counter(issue.issue_type for issue in issues)
    by_action = Counter(issue.recommended_action for issue in issues)
    by_label = Counter(issue.label for issue in issues)
    cards = "\n".join(
        _issue_card(
            issue,
            index=index,
            record=record_by_uid.get(issue.uid, {}),
        )
        for index, issue in enumerate(rendered_issues)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Meddies BIOES validation audit</title>
  <style>{STYLE}</style>
</head>
<body>
<main>
  <header>
    <h1>Meddies BIOES validation audit — checkpoint step 150</h1>
    <p class="subtitle">Heuristic triage queue. This does not auto-correct gold; it tells us where \
human adjudication will change the metric.</p>
    <div class="metrics">{metrics_html}</div>
  </header>
  <section class="panel">
    <div class="panel-body grid-2">
      {_count_table("Issue type", by_type)}
      {_count_table("Recommended action", by_action)}
      {_count_table("Label", by_label)}
      <section>
        <h2>Metric scenario</h2>
        <pre>{escape(json.dumps(scenarios, ensure_ascii=False, indent=2, sort_keys=True))}</pre>
      </section>
    </div>
  </section>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Filter by UID, label, action, reason, or text">
    <div id="visible-count" class="subtitle"></div>
  </div>
  {cards}
      <footer>
    Full queue is in <span class="path">review_queue.jsonl</span>; summary is in <span \
class="path">summary.json</span>. Generated at {escape(generated_at)} from <span \
class="path">{escape(str(predictions_json))}</span>.
  </footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""
