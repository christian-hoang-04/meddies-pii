from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from meddies_pii.json_types import is_str_mapping
from meddies_pii.taxonomy import PII_LABELS
from meddies_pii.training.bioes.reports.json_narrowing import (
    map_at,
    parse_json_object_line,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

DEFAULT_SAMPLE_LIMIT = 4_000
INLINE_TAG_RE = re.compile(r"\]<([a-z_]+)>")


@dataclass(frozen=True, slots=True)
class TrainingDataSourceConfig:
    name: str
    role: str
    note: str
    paths: tuple[Path, ...]
    recommend: str


@dataclass(slots=True)
class TrainingDataSourceStats:
    name: str
    role: str
    note: str
    paths: tuple[Path, ...]
    recommend: str
    rows: int = 0
    existing_paths: int = 0
    languages: Counter[str] = field(default_factory=Counter)
    labels: Counter[str] = field(default_factory=Counter)
    invalid_labels: Counter[str] = field(default_factory=Counter)
    fmt: str = "?"
    sample_text: str = ""

    @classmethod
    def from_config(cls, config: TrainingDataSourceConfig) -> TrainingDataSourceStats:
        return cls(
            name=config.name,
            role=config.role,
            note=config.note,
            paths=config.paths,
            recommend=config.recommend,
        )


def row_labels(record: Mapping[str, object]) -> tuple[list[str], str]:
    spans = record.get("label")
    if isinstance(spans, list) and spans:
        first = spans[0]
        if is_str_mapping(first):
            return [
                str(span.get("category") or span.get("label") or "") for span in spans if is_str_mapping(span)
            ], "label-dicts"
        if isinstance(first, (list, tuple)) and len(first) >= 3:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            return [str(span[2]) for span in spans if isinstance(span, (list, tuple)) and len(span) >= 3], "label-triples"  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
    text = record.get("text")
    if isinstance(text, str) and INLINE_TAG_RE.search(text):
        return [match.group(1) for match in INLINE_TAG_RE.finditer(text)], "inline-tags"
    return [], "no-labels"


def row_language(record: Mapping[str, object]) -> str:
    info = map_at(record, "info")
    raw = info.get("language") or info.get("lang") or record.get("language") or record.get("lang") or "?"
    return str(raw).strip().lower()


def scan_training_data_source(
    source: TrainingDataSourceConfig,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> TrainingDataSourceStats:
    stats = TrainingDataSourceStats.from_config(source)
    formats: Counter[str] = Counter()
    for path in source.paths:
        if not path.exists():
            continue
        stats.existing_paths += 1
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_index, line in enumerate(handle):
                stats.rows += 1
                if line_index >= sample_limit:
                    continue
                record = parse_json_object_line(line)
                if record is None:
                    continue
                labels, label_format = row_labels(record)
                formats[label_format] += 1
                stats.languages[row_language(record)] += 1
                _count_labels(stats, labels)
                text = record.get("text")
                if not stats.sample_text and isinstance(text, str):
                    stats.sample_text = text[:220]
    stats.fmt = ", ".join(key for key, _ in formats.most_common(2)) or "?"
    return stats


def render_training_data_breakdown_html(
    sources: Sequence[TrainingDataSourceStats],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> str:
    train_rows = sum(source.rows for source in sources if source.role not in {"eval", "misc"} and source.existing_paths)
    local_sources = len([source for source in sources if source.existing_paths and source.role not in {"eval", "misc"}])
    max_label = (
        max(
            (max(source.labels.values()) if source.labels else 0 for source in sources),
            default=0,
        )
        or 1
    )
    cards = "".join(_source_card(source, max_label) for source in sources)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meddies bioes-v2 — training-data breakdown</title>
<style>{_css()}</style></head><body><div class="wrap">
<h1>Meddies bioes-v2 — training-data breakdown</h1>
<p class="sub">Locally-scanned sources for the corpus inclusion decision · sampled {sample_limit:,} \
rows/file, counts exact</p>
<div class="summary">
  <div class="stat"><b>{train_rows:,}</b><span>local training-eligible rows (scanned)</span></div>
  <div class="stat"><b>{local_sources}</b><span>local sources</span></div>
  <div class="stat"><b>{len([source for source in sources if not source.paths])}</b><span>sources not scanned</span></div>
  <div class="stat"><b>{len(PII_LABELS)} / 17</b><span>labels / languages target</span></div>
</div>
<div class="legend"><b>How to read the recommendation chip:</b>
<b style="color:#16a34a">include</b> = clean PII labels, ready ·
<b style="color:#d97706">clean-first</b> = filter to 9 labels + normalize langs ·
<b style="color:#d97706">convert-first</b> = raw, needs legacy conversion ·
<b style="color:#64748b">eval-only</b> = keep disjoint from training ·
<b style="color:#dc2626">exclude</b> = wrong schema / not usable.
The bioes-v2 gate (label-filter + 17-lang allowlist + audit) is what makes <i>clean-first</i> safe.</div>
<h2 style="margin-top:28px;font-size:18px">Sources</h2>
{cards}
<p class="sub" style="margin-top:24px">Honest gaps: source catalogs can lag the current assembly \
plan. PR5 owns the canonical source manifest; this report only renders local scan evidence.</p>
</div></body></html>"""


def write_training_data_breakdown_report(
    output_path: Path,
    sources: Sequence[TrainingDataSourceStats],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_training_data_breakdown_html(sources, sample_limit=sample_limit),
        encoding="utf-8",
    )
    return output_path


def _count_labels(stats: TrainingDataSourceStats, labels: Sequence[str]) -> None:
    for label in labels:
        if label in PII_LABELS:
            stats.labels[label] += 1
        elif label:
            stats.invalid_labels[label] += 1


def _source_card(source: TrainingDataSourceStats, max_label: int) -> str:
    palette = {
        "internal-augmented": "#2563eb",
        "internal-synthetic": "#0891b2",
        "internal-grpo": "#7c3aed",
        "external-raw": "#d97706",
        "eval": "#64748b",
        "misc": "#94a3b8",
    }
    recommendation_colors = {
        "include": "#16a34a",
        "clean-first": "#d97706",
        "convert-first": "#d97706",
        "eval-only": "#64748b",
        "exclude": "#dc2626",
    }
    languages = ", ".join(f"{key} ({value})" for key, value in source.languages.most_common(6)) or "—"
    invalid = _invalid_label_warning(source.invalid_labels)
    scanned = _scanned_status(source)
    role_color = palette.get(source.role, "#64748b")
    recommendation_color = recommendation_colors.get(source.recommend, "#64748b")
    return f"""
        <div class="card">
          <div class="chd">
            <span class="role" style="background:{role_color}">{html.escape(source.role)}</span>
            <h3>{html.escape(source.name)}</h3>
            <span class="rec" \
style="color:{recommendation_color};border-color:{recommendation_color}">{html.escape(source.recommend)}</span>
          </div>
          <div class="meta"><b>{source.rows:,}</b> rows · {scanned} · format: <code>{html.escape(source.fmt)}</code></div>
          <p class="note">{html.escape(source.note)}</p>
          <div class="cols">
            <div><div class="lbl">PII label coverage (sampled)</div>{_bars(source.labels, PII_LABELS, max_label)}</div>
            <div><div class="lbl">Languages (sampled)</div><div class="langs">{html.escape(languages)}</div>
                 {invalid}
                 <div class="lbl" style="margin-top:10px">Sample</div>
                 <div class="sample">{html.escape(source.sample_text) or "—"}</div></div>
          </div>
        </div>"""


def _invalid_label_warning(invalid_labels: Counter[str]) -> str:
    if not invalid_labels:
        return ""
    top = ", ".join(html.escape(key) for key, _ in invalid_labels.most_common(4))
    return (
        f'<div class="warn">⚠ {len(invalid_labels)} non-PII tag types in sample '
        f"(top: {top}) — document markup / field names, filtered at build</div>"
    )


def _scanned_status(source: TrainingDataSourceStats) -> str:
    if not source.paths:
        return "<b>NOT scanned</b> (not locally staged)"
    if source.existing_paths == 0:
        return "<b>NOT scanned</b> (local paths missing)"
    return "scanned"


def _bars(counter: Counter[str], keys: Sequence[str], maximum: int) -> str:
    rows = []
    for key in keys:
        value = counter.get(key, 0)
        width = int(100 * value / maximum) if maximum else 0
        color = "#2563eb" if value else "#e2e8f0"
        rows.append(
            f'<div class="bar"><span class="bl">{html.escape(key)}</span>'
            f'<span class="bt"><span class="bf" style="width:{width}%;background:{color}"></span></span>'
            f'<span class="bv">{value or "·"}</span></div>',
        )
    return "".join(rows)


def _css() -> str:
    return """
*{box-sizing:border-box}
body{font-family:-apple-system,Inter,Segoe UI,Roboto,sans-serif;color:#0f172a;background:#f8fafc;margin:0;line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px}
h1{font-size:26px;margin:0 0 4px} .sub{color:#64748b;margin:0 0 24px}
.summary{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px}
.stat{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;min-width:150px}
.stat b{font-size:26px;display:block} .stat span{color:#64748b;font-size:13px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:20px;margin-bottom:18px}
.chd{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.chd h3{margin:0;font-size:17px;flex:1}
.role{color:#fff;font-size:11px;font-weight:600;padding:3px \
9px;border-radius:20px;text-transform:uppercase;letter-spacing:.3px}
.rec{font-size:12px;font-weight:700;border:1.5px solid;border-radius:20px;padding:3px 11px;text-transform:uppercase}
.meta{color:#475569;font-size:13px;margin:10px 0 4px} code{background:#f1f5f9;padding:1px \
6px;border-radius:5px;font-size:12px}
.note{color:#475569;font-size:13.5px;margin:6px 0 14px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.lbl{font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:8px}
.bar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
.bl{width:108px;color:#334155;flex-shrink:0} .bt{flex:1;background:#f1f5f9;border-radius:4px;height:11px;overflow:hidden}
.bf{display:block;height:100%} .bv{width:42px;text-align:right;color:#64748b;flex-shrink:0}
.langs{font-size:13px;color:#334155} \
.sample{font-size:11.5px;color:#64748b;background:#f8fafc;border:1px solid \
#f1f5f9;border-radius:8px;padding:8px;font-family:ui-monospace,monospace;white-space:pre-wrap;max-height:90px;overflow:hidden}
.warn{background:#fffbeb;border:1px solid \
#fde68a;color:#92400e;font-size:12px;border-radius:8px;padding:8px;margin-top:10px}
.legend{font-size:12.5px;color:#475569;background:#fff;border:1px solid \
#e2e8f0;border-radius:12px;padding:14px 18px;margin-top:8px}
.legend b{color:#0f172a}
@media(max-width:720px){.cols{grid-template-columns:1fr}}
"""
