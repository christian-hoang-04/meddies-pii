"""Build compact HTML preview for Meddies Labels MiMo generated/repair artifacts."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: these are command-line entry points; the printed table and report path are the product.
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.annotations.span_rendering import snippet_html
from meddies_pii.generation.label_corpus.synthetic import (
    DEFAULT_TARGETED_GENERATION_DIR,
)
from meddies_pii.generation.label_corpus.validate import (
    _LEFTOVER_LABEL_RE,
    _MALFORMED_LEFTOVER_LABEL_RE,
)
from meddies_pii.json_types import as_json_object
from meddies_pii.jsonl import read_jsonl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from meddies_pii.spans import CharSpan


@dataclass(slots=True)
class LabeledScan:
    path: Path
    rows: int = 0
    spans: int = 0
    invalid: list[str] = field(default_factory=list)
    bad: list[str] = field(default_factory=list)
    duplicates: int = 0
    labels: Counter[str] = field(default_factory=Counter)
    scenarios: Counter[str] = field(default_factory=Counter)
    examples_by_label: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    examples_by_scenario: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def record_info(record: Mapping[str, object]) -> dict[str, object]:
    info = record.get("info")
    return {str(key): value for key, value in info.items()} if isinstance(info, dict) else {}


# reason: scan labeled keeps read jsonl beside parse record; splitting would desync retries and counters.
def _scan_labeled(path: Path, *, per_label: int, per_scenario: int) -> LabeledScan:  # ruff: ignore[complex-structure]
    scan = LabeledScan(path=path)
    seen_texts: set[str] = set()
    if not path.exists():
        scan.invalid.append(f"missing: {path}")
        return scan
    for row_index, record in enumerate(read_jsonl(path), start=1):
        scan.rows += 1
        try:
            example_id, text, spans = parse_labeled_record(record, default_id=f"row-{row_index}")
        except ValueError as exc:
            scan.invalid.append(f"line {row_index}: {exc}")
            continue
        if text in seen_texts:
            scan.duplicates += 1
        seen_texts.add(text)
        cats = {span.label for span in spans}
        if not {"private_url", "secret"} <= cats:
            scan.bad.append(f"line {row_index}: missing private_url/secret")
        if "]<" in text or _LEFTOVER_LABEL_RE.search(text) or _MALFORMED_LEFTOVER_LABEL_RE.search(text):
            scan.bad.append(f"line {row_index}: leftover marker")
        for span in spans:
            if text[span.start : span.end] != span.text:
                scan.bad.append(f"line {row_index}: bad offset for {span.label}")
                break
        scan.spans += len(spans)
        scan.labels.update(span.label for span in spans)
        info = record_info(record)
        scenario = str(info.get("scenario") or "unknown")
        scan.scenarios.update([scenario])
        for span in spans:
            bucket = scan.examples_by_label[span.label]
            if len(bucket) < per_label:
                bucket.append(_example_block(example_id, text, spans, span, info))
        scenario_bucket = scan.examples_by_scenario[scenario]
        if len(scenario_bucket) < per_scenario:
            focus = spans[0] if spans else None
            scenario_bucket.append(_example_block(example_id, text, spans, focus, info))
    return scan


def _example_block(
    example_id: str,
    text: str,
    spans: tuple[CharSpan, ...],
    focus: CharSpan | None,
    info: dict[str, object],
) -> str:
    label_line = ", ".join(sorted({span.label for span in spans}))
    return (
        f'<div class="example"><div class="meta"><b>{escape(example_id)}</b> '
        f"· scenario={escape(str(info.get('scenario', 'unknown')))} "
        f"· spans={len(spans)} · labels={escape(label_line)}</div>"
        f"<pre>{snippet_html(text, spans, focus_span=focus, context=180)}</pre></div>"
    )


def _scan_unrepaired(path: Path, *, per_reason: int = 2) -> tuple[Counter[str], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return counts, examples
    for record in read_jsonl(path):
        errors = record.get("repair_errors") or record.get("errors") or []
        if not isinstance(errors, list):
            continue
        original_errors = record.get("errors")
        if not isinstance(original_errors, list):
            original_errors = []
        counts.update(str(error) for error in errors)
        for error in errors:
            key = str(error)
            if len(examples[key]) >= per_reason:
                continue
            original = str(record.get("content") or "")
            repaired = str(record.get("repaired_content") or "")
            examples[key].append(
                '<div class="example"><div class="meta">'
                f"original={escape(','.join(map(str, original_errors)))} "
                f"· repair={escape(key)}</div>"
                f"<details><summary>original</summary><pre>{escape(_short(original), quote=False)}</pre></details>"
                f"<details><summary>after attempted \
repair</summary><pre>{escape(_short(repaired), quote=False)}</pre></details>"
                "</div>",
            )
    return counts, examples


def _short(text: str, limit: int = 900) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in read_jsonl(path))


def _bar(value: int, max_value: int) -> str:
    width = 0 if max_value <= 0 else max(1, int(100 * value / max_value))
    return f'<span class="bar"><i style="width:{width}%"></i></span>'


def _table_counter(title: str, counter: Counter[str]) -> str:
    max_value = max(counter.values(), default=0)
    rows = "".join(
        f"<tr><td>{escape(key)}</td><td>{value:,}</td><td>{_bar(value, max_value)}</td></tr>"
        for key, value in sorted(counter.items())
    )
    return f"<section><h3>{escape(title)}</h3><table>{rows}</table></section>"


def _read_json(path: Path) -> dict[str, object]:
    """Decode one object-shaped repair summary at its JSON boundary.

    Returns:
        The decoded summary, or an empty dict when the file does not exist. A missing summary
        is an ordinary state -- the preview renders whatever repair runs have happened -- so it
        is distinguished from a summary that exists but is the wrong shape.

    Raises:
        ValueError: If the file decodes to anything other than a JSON object. A list or a bare
            scalar would flow into the preview as an empty mapping and render a page that
            silently omits the repair section, so it refuses at the boundary instead.

    """
    if not path.exists():
        return {}
    decoded = as_json_object(json.loads(path.read_text(encoding="utf-8")))
    if decoded is None:
        msg = f"repair summary {path} must be a JSON object"
        raise ValueError(msg)
    return dict(decoded)


def _select_repaired_prefix(output_dir: Path, requested_prefix: str | None) -> str:
    if requested_prefix:
        return requested_prefix
    clean_vi = output_dir / "repaired_candidates.clean.vi.jsonl"
    clean_en = output_dir / "repaired_candidates.clean.en.jsonl"
    if clean_vi.exists() and clean_en.exists():
        return "repaired_candidates.clean"
    return "repaired_candidates"


def build_preview(
    output_dir: Path,
    output_html: Path,
    *,
    per_label: int,
    per_scenario: int,
    repaired_prefix: str | None = None,
) -> Path:
    selected_repaired_prefix = _select_repaired_prefix(output_dir, repaired_prefix)
    scans: dict[str, dict[str, LabeledScan]] = {}
    repair_summaries: dict[str, dict[str, object]] = {}
    unrepaired: dict[str, tuple[Counter[str], dict[str, list[str]]]] = {}
    for code, _language in (("vi", "Vietnamese"), ("en", "English")):
        scans[code] = {
            "accepted": _scan_labeled(
                output_dir / f"accepted.{code}.jsonl",
                per_label=per_label,
                per_scenario=per_scenario,
            ),
            "repaired": _scan_labeled(
                output_dir / f"{selected_repaired_prefix}.{code}.jsonl",
                per_label=per_label,
                per_scenario=per_scenario,
            ),
        }
        # reason: the preview does not render these, but `_read_json` refuses a summary that exists and
        # reason: is the wrong shape, so this read stays as that check. Deleting it drops the validation.
        repair_summaries[code] = _read_json(output_dir / f"repair_summary.{code}.json")
        unrepaired[code] = _scan_unrepaired(output_dir / f"unrepaired_content_rejected.{code}.jsonl")

    html = _render_html(
        output_dir,
        scans,
        unrepaired,
        per_label,
        per_scenario,
        selected_repaired_prefix,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


# reason: render html keeps output dir/repaired at its adapter seam; bundling would hide required inputs.
def _render_html(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    output_dir: Path,
    scans: dict[str, dict[str, LabeledScan]],
    unrepaired: dict[str, tuple[Counter[str], dict[str, list[str]]]],
    per_label: int,
    per_scenario: int,
    repaired_prefix: str,
) -> str:
    cards: list[str] = []
    for code in ("vi", "en"):
        accepted = scans[code]["accepted"]
        repaired = scans[code]["repaired"]
        raw_count = _count_jsonl(output_dir / f"raw.{code}.jsonl")
        rejected_count = _count_jsonl(output_dir / f"rejected.{code}.jsonl")
        api_count = _count_jsonl(output_dir / f"api_errors.{code}.jsonl")
        content_count = _count_jsonl(output_dir / f"content_rejected.{code}.jsonl")
        flagged_count = _count_jsonl(output_dir / f"repaired_candidates.flagged.{code}.jsonl")
        cards.append(
            f'<div class="card"><h2>{code.upper()}</h2>'
            f"<p><b>accepted</b> {accepted.rows:,} · <b>{escape(repaired_prefix)}</b> {repaired.rows:,} "
            f"· <b>combined clean candidates</b> {accepted.rows + repaired.rows:,}</p>"
            f"<p><b>raw</b> {raw_count:,} · <b>original rejected</b> {rejected_count:,} "
            f"· <b>api errors</b> {api_count:,} · <b>content rejects</b> {content_count:,}</p>"
            f"<p><b>flagged/quarantined repaired rows</b> {flagged_count:,}</p>"
            f"<p><b>accepted bad probe</b> {len(accepted.bad)} · <b>repaired bad probe</b> {len(repaired.bad)} "
            f"· <b>accepted duplicates</b> {accepted.duplicates} · <b>repaired duplicates</b> \
{repaired.duplicates}</p></div>",
        )

    sections: list[str] = []
    for code in ("vi", "en"):
        accepted = scans[code]["accepted"]
        repaired = scans[code]["repaired"]
        unrepaired_counts, unrepaired_examples = unrepaired[code]
        sections.extend((
            f'<h2 id="{code}">{code.upper()} detail</h2>',
            '<div class="grid two">',
            _table_counter(f"{code.upper()} accepted label distribution", accepted.labels),
            _table_counter(f"{code.upper()} repaired label distribution", repaired.labels),
            _table_counter(f"{code.upper()} accepted scenarios", accepted.scenarios),
            _table_counter(f"{code.upper()} repaired scenarios", repaired.scenarios),
            "</div>",
            _sample_section(f"{code.upper()} repaired examples by label ({per_label}/label)", repaired.examples_by_label),
            _sample_section(
                f"{code.upper()} repaired examples by scenario ({per_scenario}/scenario)",
                repaired.examples_by_scenario,
            ),
            _table_counter(
                f"{code.upper()} unrepaired reasons after heuristic",
                Counter(dict(unrepaired_counts.most_common(12))),
            ),
            _sample_section(f"{code.upper()} unrepaired examples", dict(list(unrepaired_examples.items())[:8])),
        ))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Meddies Labels MiMo Generation Repair Preview</title>
<style>
:root {{ --bg:#f7f5ef; --paper:#fffefa; --ink:#181713; --muted:#6b675c; --line:#ddd8ca; \
--blue:#1846d2; --bad:#9f1239; --good:#0f766e; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 \
-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px; }}
h1 {{ font-size:30px; margin:0 0 8px; }}
h2 {{ margin-top:30px; border-top:1px solid var(--line); padding-top:22px; }}
h3 {{ margin:0 0 10px; }}
.note {{ color:var(--muted); margin-bottom:20px; }}
.grid {{ display:grid; gap:14px; }} .grid.cards {{ \
grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }} .grid.two {{ \
grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }}
.card, section, .example {{ background:var(--paper); border:1px solid var(--line); \
border-radius:14px; padding:14px; box-shadow:0 1px 2px #00000008; }}
table {{ width:100%; border-collapse:collapse; }} td, th {{ border-top:1px solid var(--line); \
padding:7px 8px; text-align:left; vertical-align:top; }} td:nth-child(2) {{ text-align:right; \
font-variant-numeric:tabular-nums; }}
.bar {{ display:block; width:100%; height:9px; background:#e6e3db; border-radius:99px; \
overflow:hidden; }} .bar i {{ display:block; height:100%; background:var(--blue); }}
.examples {{ display:grid; gap:10px; margin:10px 0 18px; }}
.meta {{ color:var(--muted); font-size:12px; margin-bottom:8px; }}
pre {{ white-space:pre-wrap; word-break:break-word; margin:0; font:12px/1.45 \
ui-monospace,SFMono-Regular,Menlo,monospace; }}
details {{ margin-top:8px; }} summary {{ cursor:pointer; color:var(--blue); }}
mark.pii {{ background:#fff3bf; border:1px solid #f2c94c; border-radius:4px; padding:0 2px; }} mark \
.tag {{ color:var(--blue); font-size:10px; margin-left:3px; }}
nav a {{ margin-right:12px; color:var(--blue); }}
</style>
</head>
<body><main>
<h1>Meddies Labels MiMo Generation Repair Preview</h1>
<p class="note">Compact review of accepted rows, separated API/content rejects, and conservative \
heuristic repair candidates. This preview uses repaired prefix \
<code>{escape(repaired_prefix)}</code>; flagged rows are quarantined outside the clean preview. \
Raw/original files are preserved separately. Generated examples are capped at {per_label} per label \
and {per_scenario} per scenario.</p>
<nav><a href="#vi">Vietnamese</a><a href="#en">English</a></nav>
<div class="grid cards">{"".join(cards)}</div>
{"".join(sections)}
</main></body></html>"""


def _sample_section(title: str, grouped_html: dict[str, list[str]]) -> str:
    parts = [f"<section><h3>{escape(title)}</h3>"]
    if not grouped_html:
        parts.append('<p class="note">No examples.</p>')
    for key, examples in sorted(grouped_html.items()):
        if not examples:
            continue
        parts.append(f'<h4>{escape(str(key))}</h4><div class="examples">{"".join(examples)}</div>')
    parts.append("</section>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TARGETED_GENERATION_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TARGETED_GENERATION_DIR / "repair_preview.html",
    )
    parser.add_argument("--per-label", type=int, default=2)
    parser.add_argument("--per-scenario", type=int, default=2)
    parser.add_argument(
        "--repaired-prefix",
        default=None,
        help="Repaired JSONL prefix to preview; defaults to repaired_candidates.clean when present.",
    )
    args = parser.parse_args()
    print(
        build_preview(
            args.output_dir,
            args.output,
            per_label=args.per_label,
            per_scenario=args.per_scenario,
            repaired_prefix=args.repaired_prefix,
        ),
    )


if __name__ == "__main__":
    main()
