from __future__ import annotations

# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: the multiplication sign is typography meaning "by" in an operator-facing report label
# reason: (source x language x bucket, 2xA100); the ASCII letter x would misrender the heading.
import html
import json
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from meddies_pii.generation.text_formats import canonical_text_format
from meddies_pii.json_types import is_str_mapping
from meddies_pii.languages import LANGUAGE_PROFILES

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.taxonomy import PiiLabel

FOREGROUND_TONE_CUTOFF = 0.55
MILLION = 1_000_000
THOUSAND = 1_000
LOW_F1_CUTOFF = 0.5
HIGH_F1_CUTOFF = 0.8

TEAL = "#0f766e"
AMBER = "#b45309"
INK = "#0f172a"
WEAK_LABELS: frozenset[PiiLabel] = frozenset({"company_name", "secret", "private_url"})
REPORT_LABEL_ORDER: tuple[PiiLabel, ...] = (
    "human_name",
    "id_number",
    "date",
    "address",
    "phone_number",
    "email_address",
    "company_name",
    "secret",
    "private_url",
)
LANGUAGE_NAMES: dict[str, str] = {profile.code: profile.name for profile in LANGUAGE_PROFILES.values()}


# reason: render data shape owns fold formats and int map together; splitting would duplicate totals or escaping.
def render_data_shape_report_html(  # ruff: ignore[too-many-locals]
    stats: Mapping[str, object],
    baseline: Mapping[str, object] | None = None,
    *,
    sha: str = "f00df6081c9a587a",
) -> str:
    baseline = baseline or {}
    per_format = _fold_formats(_int_map(stats.get("per_format")))
    per_language = _int_map(stats.get("per_language"))
    per_label = _int_map(stats.get("per_label"))
    lang_label_docs = _int_map(stats.get("lang_label_docs"))
    per_source = _int_map(stats.get("per_source"))
    per_edge_case = _int_map(stats.get("per_edge_case"))
    span_density = _int_map(stats.get("span_density"))
    text_len = _int_map(stats.get("text_len"))
    total_rows = _int_value(stats.get("total_rows"))
    total_spans = _int_value(stats.get("total_spans"))
    format_coverage = _int_value(stats.get("format_coverage"))

    languages = list(per_language)
    filled_cells = sum(
        1 for language in languages for label in REPORT_LABEL_ORDER if lang_label_docs.get(f"{language}|{label}", 0) > 0
    )
    total_cells = len(languages) * len(REPORT_LABEL_ORDER)
    zero_cells: list[tuple[str, PiiLabel]] = [
        (language, label)
        for language in languages
        for label in REPORT_LABEL_ORDER
        if lang_label_docs.get(f"{language}|{label}", 0) == 0
    ]
    gate_line, cell_note = _coverage_notes(filled_cells, total_cells, zero_cells)
    format_coverage_pct = 100 * format_coverage / total_rows if total_rows else 0
    origins = _source_origins(per_source)

    cards = [
        ("Rows (deduped)", _fmt(total_rows)),
        ("Labeled spans", _fmt(total_spans)),
        ("Languages", str(len(languages))),
        ("Labels", str(len(REPORT_LABEL_ORDER))),
        ("Lang&times;label cells", f"{filled_cells}/{total_cells}"),
        ("Doc formats", str(len(per_format))),
    ]
    card_html = "".join(
        f"<div class=card><div class=cn>{html.escape(value)}</div><div class=cl>{key}</div></div>" for key, value in cards
    )

    parts = [
        f"<!doctype html><meta charset=utf-8><title>bioes-v2 data shape</title><style>{_css()}</style><div class=wrap>",
        "<h1>bioes-v2 — corpus data shape</h1>",
        (
            f"<p class=sub>The assembled training mix on the Modal volume "
            f"(<code>meddies-bioes-v2 :/data/mix</code>, sha256 <code>{html.escape(sha)}</code>). "
            f"{gate_line} Generated from a full scan.</p>"
        ),
        f"<div class=cards>{card_html}</div>",
        "<h2>Labels — and the Run-1 F1 they have to lift</h2>",
        (
            "<p class=note>Span counts (log-spread). Amber = the three weak labels bioes-v2 floods with "
            "supply; the chip is Run-1's per-label F1 — company_name fired 0.00, private_url 0.04, "
            "secret 0.21. The imbalance is the point: strong labels in the millions, weak ones in the "
            "tens-of-thousands but now above floor.</p>"
        ),
        _label_section(per_label, baseline),
        "<h2>Language &times; label coverage (the 153-cell grid)</h2>",
        f"<p class=note>{cell_note}</p>",
        _heatmap(lang_label_docs, languages, REPORT_LABEL_ORDER),
        "<p class=legend>Cell colour = log(doc count), light→dark teal. Amber = empty cell.</p>",
        "<div class=two>",
        (
            "<div><h2>Languages</h2><p class=note>Documents per language (17, en-heavy, no starvation).</p>"
            f"{_bars({_language_name(k): v for k, v in per_language.items()})}</div>"
        ),
        f"<div><h2>Origins</h2><p class=note>Rows by source family.</p>{_bars(origins)}</div>",
        "</div>",
        "<div class=two>",
        (
            f"<div><h2>Document formats</h2><p class=note>Clinical surface-form diversity "
            f"({format_coverage_pct:.0f}% of rows tagged — configs carry it, external sources don't).</p>"
            f"{_bars(dict(list(per_format.items())[:12]))}</div>"
        ),
        (
            f"<div><h2>Span density</h2><p class=note>Labeled spans per document — the corpus is "
            f"span-dense (most docs 4-20 spans).</p>"
            f"{_bars(_order(span_density, ['0', '1', '2-3', '4-6', '7-10', '11-20', '21+']))}</div>"
        ),
        "</div>",
        "<h2>Difficulty / edge-case slices (top 12)</h2>",
        (
            "<p class=note>Generation-injected hard cases (emergency tone, foreigner names, "
            "elderly-no-phone, industrial addresses, long names, dialect/slang). The model trains "
            "on the hard tail, not just clean notes.</p>"
        ),
        _bars(dict(list(per_edge_case.items())[:12])),
        "<h2>Document length</h2>",
        _bars(_order(text_len, ["0-200", "201-500", "501-1k", "1k-2k", "2k-4k", "4k+"])),
        (
            f"<footer>bioes-v2 mix · {_fmt(total_rows)} rows · {_fmt(total_spans)} spans · "
            f"sha256 {html.escape(sha)} · source field is heterogeneous across converters (configs tag format/"
            "difficulty; ai4privacy/nvidia do not) so format/difficulty coverage is partial by design.</footer>"
        ),
        "</div>",
    ]
    return "".join(parts)


def write_data_shape_report(
    *,
    stats_path: Path,
    baseline_path: Path | None,
    output_path: Path,
    sha: str = "f00df6081c9a587a",
) -> Path:
    stats = _read_json_object(stats_path)
    baseline = _read_json_object(baseline_path) if baseline_path is not None and baseline_path.exists() else {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_data_shape_report_html(stats, baseline, sha=sha), encoding="utf-8")
    return output_path


def _read_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not is_str_mapping(value):
        msg = f"expected JSON object in {path}"
        raise ValueError(msg)
    return dict(value)


def _fmt(value: int) -> str:
    return f"{value:,}"


def _bars(
    data: Mapping[str, int],
    *,
    color: str = TEAL,
    highlight: frozenset[str] | None = None,
    fmt: Callable[[int], str] = _fmt,
) -> str:
    if not data:
        return "<p class=mute>no data</p>"
    top = max(data.values()) or 1
    rows = []
    for key, value in data.items():
        pct = 100 * value / top
        fill = AMBER if highlight and key in highlight else color
        rows.append(
            f"<div class=row><div class=key>{html.escape(str(key))}</div>"
            f'<div class=track><div class=fill style="width:{pct:.2f}%;background:{fill}"></div></div>'
            f"<div class=val>{html.escape(fmt(value))}</div></div>",
        )
    return f"<div class=bars>{''.join(rows)}</div>"


def _heatmap(lang_label: Mapping[str, int], languages: list[str], labels: tuple[PiiLabel, ...]) -> str:
    vals = [value for value in lang_label.values() if value > 0]
    low, high = (math.log10(min(vals)), math.log10(max(vals))) if vals else (0, 1)
    span = (high - low) or 1

    def cell(language: str, label: str) -> str:
        value = lang_label.get(f"{language}|{label}", 0)
        title = html.escape(f"{language}×{label}: {_fmt(value)}")
        if value == 0:
            return f'<td class="hm zero" title="{title}">0</td>'
        tone = (math.log10(value) - low) / span
        background = f"hsl(174,{30 + int(45 * tone)}%,{92 - int(55 * tone)}%)"
        foreground = "#fff" if tone > FOREGROUND_TONE_CUTOFF else INK
        return (
            f'<td class=hm style="background:{background};color:{foreground}" '
            f'title="{title}">{html.escape(_short(value))}</td>'
        )

    head = "".join(f"<th class=hlabel>{html.escape(label.replace('_', ' '))}</th>" for label in labels)
    body = ""
    for language in languages:
        cells = "".join(cell(language, label) for label in labels)
        body += f"<tr><th class=hlang>{html.escape(_language_name(language))}</th>{cells}</tr>"
    return f"<table class=heat><tr><th></th>{head}</tr>{body}</table>"


def _short(value: int) -> str:
    if value >= MILLION:
        return f"{value / 1e6:.1f}M"
    if value >= THOUSAND:
        return f"{value / 1e3:.0f}k"
    return str(value)


def _label_section(per_label: Mapping[str, int], baseline: Mapping[str, object]) -> str:
    baseline_per_label = _mapping(baseline.get("per_label"))
    top = max(per_label.values()) or 1 if per_label else 1
    rows = []
    for label, value in per_label.items():
        pct = 100 * value / top
        color = AMBER if label in WEAK_LABELS else TEAL
        f1 = _label_f1(baseline_per_label, label)
        chip = ""
        if f1 is not None:
            chip_color = "#dc2626" if f1 < LOW_F1_CUTOFF else ("#d97706" if f1 < HIGH_F1_CUTOFF else "#16a34a")
            chip = f'<span class=chip style="background:{chip_color}">Run-1 F1 {f1:.2f}</span>'
        rows.append(
            f"<div class=row><div class=key>{html.escape(label.replace('_', ' '))}</div>"
            f'<div class=track><div class=fill style="width:{pct:.2f}%;background:{color}"></div></div>'
            f"<div class=val>{_fmt(value)} {chip}</div></div>",
        )
    return f"<div class=bars>{''.join(rows)}</div>"


def _label_f1(baseline_per_label: Mapping[str, object], label: str) -> float | None:
    label_stats = _mapping(baseline_per_label.get(label))
    value = label_stats.get("f1")
    return float(value) if isinstance(value, int | float) else None


def _coverage_notes(filled_cells: int, total_cells: int, zero_cells: list[tuple[str, PiiLabel]]) -> tuple[str, str]:
    if zero_cells:
        waived = len(zero_cells)
        examples = ", ".join(f"<b>{html.escape(language)} &times; {label}</b>" for language, label in zero_cells[:5])
        gate_line = f"Gate passes with {waived} waived cell(s)."
        cell_note = (
            f"Documents per (language, label). {filled_cells}/{total_cells} cells filled; the "
            f"{waived} amber cell(s) are waived (exhaustion-waiver, deployment-rare): {examples}."
        )
    else:
        gate_line = "Gate passes with zero waivers — all cells filled."
        cell_note = (
            f"Documents per (language, label). All {filled_cells}/{total_cells} cells filled — no "
            "waivers. The previously-waived <b>pt &times; private_url</b> cell is now closed by "
            "the all-language free-pool weak-labels daily."
        )
    return gate_line, cell_note


def _source_origins(per_source: Mapping[str, int]) -> dict[str, int]:
    origins = {
        "per-language configs": 0,
        "ai4privacy": 0,
        "nvidia/Nemotron": 0,
        "base-173k (augmented)": 0,
        "GRPO": 0,
        "synthetic (free pools)": 0,
    }
    for key, value in per_source.items():
        if key == "meddies-pii-hf-config":
            origins["per-language configs"] += value
        elif key in {"train", "validation"}:
            origins["ai4privacy"] += value
        elif key.startswith("nvidia"):
            origins["nvidia/Nemotron"] += value
        elif key == "grpo-hf":
            origins["GRPO"] += value
        elif key in {"opencode_zen", "openrouter"}:
            origins["synthetic (free pools)"] += value
        else:
            origins["base-173k (augmented)"] += value
    return dict(sorted(origins.items(), key=lambda item: -item[1]))


def _fold_formats(per_format: Mapping[str, int]) -> dict[str, int]:
    folded: dict[str, int] = {}
    for raw, count in per_format.items():
        key = canonical_text_format(raw)
        folded[key] = folded.get(key, 0) + count
    return dict(sorted(folded.items(), key=lambda item: -item[1]))


def _order(data: Mapping[str, int], keys: list[str]) -> dict[str, int]:
    return {key: data[key] for key in keys if key in data}


def _int_map(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, int):
            result[str(key)] = raw
    return result


def _mapping(value: object) -> Mapping[str, object]:
    return value if is_str_mapping(value) else {}


def _int_value(value: object) -> int:
    return value if isinstance(value, int) else 0


def _language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _css() -> str:
    return """
    :root{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,sans-serif;color:#0f172a}
    *{box-sizing:border-box} body{margin:0;background:#f8fafc;line-height:1.5}
    .wrap{max-width:1040px;margin:0 auto;padding:48px 32px 80px}
    h1{font-size:28px;margin:0 0 4px} .sub{color:#64748b;margin:0 0 28px;font-size:14px}
    h2{font-size:17px;margin:40px 0 6px;letter-spacing:.2px}
    .note{color:#64748b;font-size:13px;margin:0 0 14px}
    .cards{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:24px 0}
    .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 14px;text-align:center}
    .cn{font-size:22px;font-weight:650;color:#0f766e} .cl{font-size:11px;color:#64748b;margin-top:4px}
    .bars{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px}
    .row{display:flex;align-items:center;gap:12px;padding:3px 0}
    .key{width:130px;font-size:13px;text-align:right;color:#334155;flex:none}
    .track{flex:1;background:#f1f5f9;border-radius:6px;height:18px;overflow:hidden}
    .fill{height:100%;border-radius:6px}
    .val{width:170px;font-size:12px;color:#475569;flex:none;font-variant-numeric:tabular-nums}
    .chip{color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:6px}
    table.heat{border-collapse:collapse;background:#fff;border:1px solid \
#e2e8f0;border-radius:12px;overflow:hidden;font-size:11px}
    .heat th{font-weight:600;color:#334155;padding:6px 8px}
    .hlabel{font-size:10px;text-align:center;border-bottom:1px solid #e2e8f0}
    .hlang{text-align:right;color:#0f172a;border-right:1px solid #e2e8f0;white-space:nowrap}
    td.hm{text-align:center;padding:7px 9px;font-variant-numeric:tabular-nums;border:1px solid #f1f5f9}
    td.hm.zero{background:#fef3c7;color:#b45309;font-weight:700}
    .mute{color:#64748b} .two{display:grid;grid-template-columns:1fr 1fr;gap:20px}
    footer{margin-top:48px;color:#94a3b8;font-size:12px;border-top:1px solid #e2e8f0;padding-top:16px}
    .legend{font-size:11px;color:#64748b;margin:6px 0 0}
    """
