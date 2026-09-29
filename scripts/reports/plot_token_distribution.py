#!/usr/bin/env python
"""Draw a token-length distribution diagram for a Meddies dataset as a self-contained HTML.

Reads the pre-computed ``token_count_lfm25`` column (no re-tokenizing, no GPU), so it is
fast and reusable: point it at full ``meddies-pii-mixed`` now, or at any ABLATION SUBSET
later (subsets keep the column) to confirm what a data drop removed length-wise.

Renders, in one theme-aware HTML file (inline SVG, no external libraries):
  * overall histogram of token lengths, with the training cap marked
  * percentile summary (p50 / p95 / p99 / max, % over cap)
  * per-group breakdown table (by language / source / domain) sorted by p95

``--compare <other>`` overlays a second dataset (e.g. before vs after a drop) as grouped
bars, so an ablation's effect on the length profile is visible at a glance.

Colors are the validated data-viz reference palette (blue magnitude, green second series,
critical-red cap line). Reads only; writes one HTML file; never touches the hub.
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import html
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

if TYPE_CHECKING:
    from collections.abc import Sequence

OVER_CAP_WARNING_PERCENT = 5

FINE_BINS = 64
"""Fine linear histogram.

FINE_BINS even bins over [0, CLIP], plus one overflow bar for the long tail (> CLIP) so the pathological rows don't flatten
the whole curve.

"""
CLIP = 10240
DEFAULT_COLUMN = "token_count_lfm25"
DEFAULT_CAP = 8192


def load_frame(spec: str, column: str, groupby: str) -> tuple[np.ndarray, list[str]]:
    """Return (lengths, group_labels) for an HF repo id or a local saved-dataset path.

    spec may be "repo" or "repo:config"; default config/split for mixed.

    """
    if Path(spec).exists():
        ds = load_from_disk(spec)
    else:
        repo, _, config = spec.partition(":")
        ds = load_dataset(repo, config or "default", split="train")
    # reason: load_from_disk and a split= load both widen to include DatasetDict, whose column_names
    # reason: is a per-split mapping; every path here loads one split, so the flatten result is a Dataset.
    ds = cast("Dataset", ds.flatten())
    if column not in ds.column_names:
        msg = f"column {column!r} not in {spec} (have: {ds.column_names[:8]}...)"
        raise SystemExit(msg)
    lengths = np.asarray(ds[column], dtype=np.int64)
    key = f"info.{groupby}"
    groups = [str(g) for g in ds[key]] if key in ds.column_names else ["all"] * len(lengths)
    return lengths, groups


def pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if len(arr) else 0.0


def histogram_counts(lengths: np.ndarray) -> list[int]:
    """FINE_BINS even bins over [0, CLIP], plus a final overflow bin for > CLIP."""
    edges = np.linspace(0, CLIP, FINE_BINS + 1)
    counts, _ = np.histogram(lengths[lengths < CLIP], bins=edges)
    overflow = int(np.sum(lengths >= CLIP))
    return [int(c) for c in counts] + [overflow]


# reason: svg histogram combines x at and linspace; splitting would duplicate totals or escaping.
def svg_histogram(series: list[tuple[str, list[int], str]], cap: int) -> str:  # ruff: ignore[too-many-locals]
    """Fine linear histogram. Single series = filled bars; compare = overlaid + translucent.

    Normalize each series to its OWN total (density, % of rows) so distributions of different sizes (e.g. full vs a 10%
    subset) overlay on the same scale.

    bars (overlaid across series so the distribution shape is comparable).

    """
    edges = np.linspace(0, CLIP, FINE_BINS + 1)
    total_bars = FINE_BINS + 1
    w, h = 720, 300
    pad_l, pad_b, pad_t, pad_r = 52, 44, 24, 16
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_b - pad_t
    bar_w = plot_w / total_bars
    totals = [max(sum(c), 1) for _, c, _ in series]
    max_frac = max((c / t for (_, cs, _), t in zip(series, totals, strict=False) for c in cs), default=1) or 1
    fill_op = 0.9 if len(series) == 1 else 0.55

    def x_at(tok: float) -> float:
        return pad_l + (tok / CLIP) * (FINE_BINS * bar_w)

    parts: list[str] = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="token length histogram" class="hist">']
    for i in range(5):
        val = max_frac * i / 4 * 100
        y = pad_t + plot_h - plot_h * i / 4
        parts.extend((
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="ytick">{val:.0f}%</text>',
        ))
    for (name, counts, color), total in zip(series, totals, strict=False):
        for bi in range(total_bars):
            c = counts[bi]
            bh = plot_h * (c / total) / max_frac
            x = pad_l + bi * bar_w
            y = pad_t + plot_h - bh
            if bi < FINE_BINS:
                lo, hi = int(edges[bi]), int(edges[bi + 1])
                # reason: an EN DASH is the correct typography for a numeric range in a rendered chart
                # reason: label, and a hyphen would read as a minus sign between two thousands-separated
                # reason: numbers. This string is drawn, never parsed.
                blabel = f"{lo:,}–{hi:,}"  # ruff: ignore[ambiguous-unicode-character-string]
            else:
                blabel = f"{CLIP:,}+"
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.9:.2f}" height="{bh:.2f}" '
                f'fill="{color}" fill-opacity="{fill_op}"><title>{html.escape(name)} | '
                f"{blabel} tok\n{c:,} rows ({100 * c / total:.2f}%)</title></rect>",
            )
    parts.extend(
        f'<text x="{x_at(tok):.1f}" y="{h - pad_b + 16}" class="xtick">'
        f"{tok // 1000 if tok else 0}{'k' if tok else ''}</text>"
        for tok in (0, 2000, 4000, 6000, 8000, 10000)
    )
    parts.extend((
        (
            f'<text x="{pad_l + (FINE_BINS + 0.5) * bar_w:.1f}" y="{h - pad_b + 16}" '
            f'class="xtick">{CLIP // 1024 * 1024 // 1000}k+</text>'
        ),
        f'<text x="{(pad_l + w - pad_r) / 2:.0f}" y="{h - 6}" class="axtitle">tokens per row</text>',
    ))
    cx = x_at(cap)
    parts.extend((
        f'<line x1="{cx:.1f}" y1="{pad_t}" x2="{cx:.1f}" y2="{pad_t + plot_h}" class="cap"/>',
        f'<text x="{cx + 4:.1f}" y="{pad_t + 10}" class="cap-label">cap {cap:,}</text>',
        "</svg>",
    ))
    return "\n".join(parts)


def group_table(lengths: np.ndarray, groups: list[str], cap: int) -> str:
    garr = np.asarray(groups)
    rows = []
    for g in sorted(set(groups)):
        v = lengths[garr == g]
        rows.append((
            g,
            len(v),
            pct(v, 50),
            pct(v, 95),
            pct(v, 99),
            int(v.max()) if len(v) else 0,
            100 * float(np.sum(v > cap)) / max(len(v), 1),
        ))
    rows.sort(key=lambda r: -r[3])
    max_p95 = max((r[3] for r in rows), default=1) or 1
    out = [
        (
            '<table class="grp"><thead><tr>'
            "<th>group</th><th>rows</th><th>p50</th><th>p95</th><th>p99</th>"
            "<th>max</th><th>% &gt; cap</th><th></th></tr></thead><tbody>"
        ),
    ]
    for g, n, p50, p95, p99, mx, over in rows:
        bar = 100 * p95 / max_p95
        flag = ' class="over"' if over >= OVER_CAP_WARNING_PERCENT else ""
        out.append(
            f'<tr><td class="g">{html.escape(g)}</td><td>{n:,}</td>'
            f"<td>{p50:.0f}</td><td>{p95:.0f}</td><td>{p99:.0f}</td><td>{mx:,}</td>"
            f"<td{flag}>{over:.2f}%</td>"
            f'<td class="barcell"><span class="pbar" style="width:{bar:.1f}%"></span></td></tr>',
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def stat_tiles(lengths: np.ndarray, cap: int) -> str:
    over = 100 * float(np.sum(lengths > cap)) / max(len(lengths), 1)
    tiles = [
        ("rows", f"{len(lengths):,}"),
        ("p50", f"{pct(lengths, 50):.0f}"),
        ("p95", f"{pct(lengths, 95):.0f}"),
        ("p99", f"{pct(lengths, 99):.0f}"),
        ("max", f"{int(lengths.max()) if len(lengths) else 0:,}"),
        (f"% &gt; {cap:,}", f"{over:.3f}%"),
    ]
    cells = "".join(f'<div class="tile"><div class="tval">{v}</div><div class="tlabel">{k}</div></div>' for k, v in tiles)
    return f'<div class="tiles">{cells}</div>'


PAGE_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f0efec;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#8a897f;
--series-1:#2a78d6;--series-2:#008300;--cap:#d03b3b;--border:#e2e1db;
font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--surface-1);color:var(--text-primary);padding:24px;max-width:860px;margin:0 auto;}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
color-scheme:dark;--surface-1:#1a1a19;--surface-2:#26261f;--text-primary:#fff;
--text-secondary:#c3c2b7;--text-muted:#8a897f;--series-1:#3987e5;--series-2:#008300;
--cap:#e66767;--border:#33332c;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#26261f;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8a897f;--series-1:#3987e5;
--series-2:#008300;--cap:#e66767;--border:#33332c;}
.viz-root h1{font-size:19px;margin:0 0 2px;}
.viz-root .sub{color:var(--text-secondary);font-size:13px;margin:0 0 18px;}
.viz-root h2{font-size:14px;margin:22px 0 8px;color:var(--text-secondary);
text-transform:uppercase;letter-spacing:.04em;}
.tiles{display:flex;flex-wrap:wrap;gap:8px;}
.tile{background:var(--surface-2);border:1px solid var(--border);border-radius:8px;
padding:10px 14px;min-width:92px;}
.tval{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;}
.tlabel{font-size:11px;color:var(--text-secondary);margin-top:2px;}
.hist{width:100%;height:auto;overflow:visible;}
.hist .grid{stroke:var(--border);stroke-width:1;}
.hist .ytick{fill:var(--text-muted);font-size:10px;text-anchor:end;font-variant-numeric:tabular-nums;}
.hist .xtick{fill:var(--text-secondary);font-size:10px;text-anchor:middle;font-variant-numeric:tabular-nums;}
.hist .axtitle{fill:var(--text-muted);font-size:10px;text-anchor:middle;text-transform:uppercase;letter-spacing:.05em;}
.hist .cap{stroke:var(--cap);stroke-width:2;stroke-dasharray:4 3;}
.hist .cap-label{fill:var(--cap);font-size:10px;font-weight:600;}
.hist rect{transition:opacity .1s;}.hist rect:hover{opacity:.78;}
.legend{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);margin:4px 0 0;}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle;}
table.grp{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;}
table.grp th{text-align:right;color:var(--text-muted);font-weight:500;padding:4px 8px;
border-bottom:1px solid var(--border);}
table.grp th:first-child{text-align:left;}
table.grp td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--border);}
table.grp td.g{text-align:left;color:var(--text-primary);}
table.grp td.over{color:var(--cap);font-weight:600;}
.barcell{width:120px;}
.pbar{display:inline-block;height:9px;border-radius:3px;background:var(--series-1);vertical-align:middle;}
"""


# reason: build html exposes title/legend as its public contract; bundling would break callers.
def build_html(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    title: str,
    subtitle: str,
    primary: np.ndarray,
    groups: list[str],
    cap: int,
    groupby: str,
    series: list[tuple[str, list[int], str]],
    legend: str | None,
) -> str:
    return f"""<style>{PAGE_CSS}</style>
<div class="viz-root">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(subtitle)}</p>
<h2>Summary</h2>
{stat_tiles(primary, cap)}
<h2>Token-length histogram</h2>
{svg_histogram(series, cap)}
{legend or ""}
<h2>By {html.escape(groupby)} (sorted by p95)</h2>
{group_table(primary, groups, cap)}
</div>"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="Meddies/meddies-pii-mixed",
        help="HF repo id (optionally repo:config) or a local saved-dataset path.",
    )
    p.add_argument(
        "--compare",
        default=None,
        help="Second dataset to overlay (before/after an ablation drop).",
    )
    p.add_argument("--column", default=DEFAULT_COLUMN)
    p.add_argument(
        "--groupby",
        default="language",
        choices=("language", "source_dataset", "domain_bucket"),
    )
    p.add_argument("--cap", type=int, default=DEFAULT_CAP)
    p.add_argument("--out", default="token_distribution.html")
    p.add_argument("--title", default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"loading {args.dataset} ...", flush=True)
    lengths, groups = load_frame(args.dataset, args.column, args.groupby)

    series = [(args.dataset, histogram_counts(lengths), "var(--series-1)")]
    legend = None
    if args.compare:
        print(f"loading compare {args.compare} ...", flush=True)
        lengths2, _ = load_frame(args.compare, args.column, args.groupby)
        series.append((args.compare, histogram_counts(lengths2), "var(--series-2)"))
        legend = (
            '<div class="legend">'
            f'<span><span class="sw" style="background:var(--series-1)"></span>'
            f"{html.escape(args.dataset)}</span>"
            f'<span><span class="sw" style="background:var(--series-2)"></span>'
            f"{html.escape(args.compare)}</span></div>"
        )

    title = args.title or f"Token-length distribution — {args.dataset}"
    subtitle = f"{len(lengths):,} rows · tokenizer column {args.column} · cap {args.cap:,}" + (
        f" · vs {args.compare}" if args.compare else ""
    )
    page = build_html(title, subtitle, lengths, groups, args.cap, args.groupby, series, legend)

    out = Path(args.out)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({len(lengths):,} rows, grouped by {args.groupby})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
