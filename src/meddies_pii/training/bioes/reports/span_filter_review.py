from __future__ import annotations

import glob
import html
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data.span_quality import bad_span_reason
from meddies_pii.training.bioes.reports.json_narrowing import parse_json_object_line

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from meddies_pii.taxonomy import PiiLabel

EXAMPLES_PER_LABEL_LIMIT = 400
LONG_SPAN_MIN_CHARACTERS = 25
REMOVED_EXAMPLES_PER_REASON = 4
REMOVED_EXAMPLES_TOTAL = 14
VERY_LONG_SPAN_CHARACTERS = 70

SPAN_FILTER_LABEL_ORDER: tuple[PiiLabel, ...] = (
    "id_number",
    "human_name",
    "company_name",
    "address",
    "date",
    "phone_number",
    "email_address",
    "private_url",
    "secret",
)
REASON_COLORS: dict[str, str] = {
    "V1_struct_prefix": "#b91c1c",
    "V2_struct_body": "#c2410c",
    "V3_phone_too_long": "#a16207",
    "V5_email_no_at": "#7c3aed",
    "V7_no_digit": "#0e7490",
    "V7_name_paragraph": "#be185d",
}
REASON_DESCRIPTIONS: dict[str, str] = {
    "V1_struct_prefix": "span starts with a JSON/XML/DICOM token",
    "V2_struct_body": "≥2 FHIR/HL7/DICOM markers inside",
    "V3_phone_too_long": "phone > 30 chars",
    "V5_email_no_at": "long 'email' with no @ / at / dot",
    "V7_no_digit": "id/phone with no digit",
    "V7_name_paragraph": ">60-char name across ≥2 sentences",
}
DEFAULT_SPAN_FILTER_PATTERNS: tuple[str, ...] = (
    f"data/bioes-v2/base/all_lang_unique_augmented-20260530.train.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    f"data/bioes-v2/internal/grpo/grpo-gemini.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    "data/bioes-v2/synthetic/**/accepted*.jsonl",
)


@dataclass(slots=True)
class SpanFilterCounts:
    total: int = 0
    removed: int = 0


@dataclass(frozen=True, slots=True)
class RemovedSpanExample:
    reason: str
    span_text: str
    context_html: str
    source: str


@dataclass(frozen=True, slots=True)
class KeptSpanExample:
    span_length: int
    span_text: str
    context_html: str
    source: str


@dataclass(slots=True)
class SpanFilterReview:
    counts: defaultdict[str, SpanFilterCounts] = field(default_factory=lambda: defaultdict(SpanFilterCounts))
    reason_counts: Counter[str] = field(default_factory=Counter)
    removed: defaultdict[str, list[RemovedSpanExample]] = field(default_factory=lambda: defaultdict(list))
    kept: defaultdict[str, list[KeptSpanExample]] = field(default_factory=lambda: defaultdict(list))
    max_base: int = 80_000

    @property
    def total(self) -> int:
        return sum(count.total for count in self.counts.values())

    @property
    def total_removed(self) -> int:
        return sum(count.removed for count in self.counts.values())


def collect_span_filter_review(
    patterns: Sequence[str] = DEFAULT_SPAN_FILTER_PATTERNS,
    *,
    max_base: int = 80_000,
) -> SpanFilterReview:
    review = SpanFilterReview(max_base=max_base)
    for path in _matched_paths(patterns):
        source = source_label_for_path(path)
        try:
            with path.open(encoding="utf-8") as handle:
                _collect_file(handle, source, review)
        except OSError:
            continue
    return review


def source_label_for_path(path: Path) -> str:
    parts = path.parts
    if "bioes-v2" in parts:
        index = parts.index("bioes-v2")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name or str(path)


def span_context_html(
    document: str,
    start: int,
    end: int,
    span_text: str,
    *,
    context_chars: int = 55,
) -> str:
    window_start = max(0, start - context_chars)
    window_end = min(len(document), end + context_chars)
    pre = html.escape(document[window_start:start])
    hit = html.escape(document[start:end] or span_text)
    post = html.escape(document[end:window_end])
    lead = "…" if window_start > 0 else ""
    tail = "…" if window_end < len(document) else ""
    return f"{lead}{pre}<mark>{hit}</mark>{post}{tail}"


def render_span_filter_review_html(review: SpanFilterReview) -> str:
    total = review.total
    total_removed = review.total_removed
    css = _css()
    parts = [
        f"<!doctype html><meta charset=utf-8><title>span filter review</title><style>{css}</style><div class=wrap>",
        "<h1>Span-quality filter — removed vs kept</h1>",
        (
            f"<p class=sub>Every accepted span run through <code>bad_span_reason</code>. "
            f"<b>{total_removed:,}</b> of <b>{total:,}</b> ({_percent(total_removed, total)}) removed. "
            "Left = REMOVED (why highlighted); right = KEPT (longest shown first — the borderline ones to sanity-check). "
            "Judge: anything on the LEFT that should stay? anything on the RIGHT that should go?</p>"
        ),
        _legend(review.reason_counts),
    ]

    for label in _labels_to_render(review):
        counts = review.counts[label]
        if counts.total == 0:
            continue
        kept_count = counts.total - counts.removed
        parts.extend((
            (
                f"<h2><span class=labchip>{label}</span> "
                f"<span class=pill>{counts.removed:,} removed · {kept_count:,} kept · "
                f"{_percent(counts.removed, counts.total)} removed</span></h2>"
            ),
            "<div class=cols>",
            _removed_examples(review.removed[label]),
            _kept_examples(review.kept[label]),
            "</div>",
        ))

    parts.append(
        f"<footer>Local corpus sample (base capped at {review.max_base:,} file lines + grpo + synthetic). "
        "Rules: span_quality.bad_span_reason (V1-V3, V5, V7). Re-run after tuning a validator.</footer></div>",
    )
    return "".join(parts)


def write_span_filter_review_report(
    review: SpanFilterReview,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_span_filter_review_html(review), encoding="utf-8")
    return output_path


def _matched_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        # reason: `patterns` is a public parameter typed Sequence[str], so a caller may pass a whole path
        # reason: pattern including its anchor. Path.glob cannot express an absolute pattern, so the swap
        # reason: would narrow what this function accepts while changing nothing for the defaults.
        paths.extend(Path(match) for match in sorted(glob.glob(pattern, recursive=True)))  # ruff: ignore[glob]
    return paths


def _collect_file(handle: TextIO, source: str, review: SpanFilterReview) -> None:
    for line_index, line in enumerate(handle):
        if source == "base" and line_index >= review.max_base:
            break
        record = parse_json_object_line(line)
        if record is None:
            continue
        document = record.get("text")
        spans = _span_list(record)
        if not isinstance(document, str) or spans is None:
            continue
        for span in spans:
            _collect_span(document, span, source, review)


def _span_list(record: Mapping[str, object]) -> list[Mapping[str, object]] | None:
    spans = record.get("label") if isinstance(record.get("label"), list) else record.get("spans")
    if not isinstance(spans, list):
        return None
    return [span for span in spans if is_str_mapping(span)]


def _collect_span(
    document: str,
    span: Mapping[str, object],
    source: str,
    review: SpanFilterReview,
) -> None:
    label = span.get("category") or span.get("label")
    span_text = span.get("text")
    if not (isinstance(label, str) and isinstance(span_text, str) and span_text):
        return
    counts = review.counts[label]
    counts.total += 1
    reason = bad_span_reason(label, span_text)
    start_raw = span.get("start")
    end_raw = span.get("end")
    start = start_raw if isinstance(start_raw, int) else 0
    end = end_raw if isinstance(end_raw, int) else 0
    context = span_context_html(document, start, end, span_text)
    if reason:
        counts.removed += 1
        review.reason_counts[reason] += 1
        if len(review.removed[label]) < EXAMPLES_PER_LABEL_LIMIT:
            review.removed[label].append(RemovedSpanExample(reason, span_text, context, source))
    elif len(span_text) > LONG_SPAN_MIN_CHARACTERS and len(review.kept[label]) < EXAMPLES_PER_LABEL_LIMIT:
        review.kept[label].append(KeptSpanExample(len(span_text), span_text, context, source))


def _legend(reason_counts: Counter[str]) -> str:
    parts = ["<p class=legend>"]
    for reason, color in REASON_COLORS.items():
        parts.append(
            f'<span><span class=dot style="background:{color}"></span>'
            f"{reason} — {REASON_DESCRIPTIONS[reason]} ({reason_counts.get(reason, 0):,})</span>",
        )
    parts.append("</p>")
    return "".join(parts)


def _labels_to_render(review: SpanFilterReview) -> list[str]:
    ordered = []
    for label in SPAN_FILTER_LABEL_ORDER:
        counts = review.counts.get(label)
        if counts is not None and counts.total:
            ordered.append(label)
    extras = sorted(label for label, counts in review.counts.items() if counts.total and label not in ordered)
    return [*ordered, *extras]


def _removed_examples(examples: Sequence[RemovedSpanExample]) -> str:
    parts = ['<div class="col rm"><h3>✗ Removed — and why</h3>']
    shown = 0
    seen_reason: defaultdict[str, int] = defaultdict(int)
    for example in sorted(examples, key=lambda item: (item.reason, -len(item.span_text))):
        if seen_reason[example.reason] >= REMOVED_EXAMPLES_PER_REASON or shown >= REMOVED_EXAMPLES_TOTAL:
            continue
        seen_reason[example.reason] += 1
        shown += 1
        color = REASON_COLORS.get(example.reason, "#b91c1c")
        why = REASON_DESCRIPTIONS.get(example.reason, example.reason)
        parts.append(
            "<div class=ex>"
            f'<div class=why style="color:{color}">✗ {html.escape(why)} '
            f"<span class=code>{html.escape(example.reason)}</span></div>"
            f'<div class=span>"{html.escape(_truncate(example.span_text, 110))}" '
            f"<span class=meta>· {len(example.span_text)} chars · {html.escape(example.source)}</span></div>"
            f"<div class=ctx>in context: {example.context_html}</div></div>",
        )
    if shown == 0:
        parts.append("<div class=ex><span class=meta>nothing removed for this label in the sample</span></div>")
    parts.append("</div>")
    return "".join(parts)


def _kept_examples(examples: Sequence[KeptSpanExample]) -> str:
    parts = ['<div class="col kp"><h3>✓ Kept — and why</h3>']
    for example in sorted(examples, key=lambda item: -item.span_length)[:14]:
        note = (
            "long but clean — no structure tokens, valid for the label"
            if example.span_length > VERY_LONG_SPAN_CHARACTERS
            else "valid entity, no structure tokens"
        )
        parts.append(
            "<div class=ex>"
            f'<div class=why style="color:#15803d">✓ {note}</div>'
            f'<div class=span>"{html.escape(_truncate(example.span_text, 110))}" '
            f"<span class=meta>· {example.span_length} chars · {html.escape(example.source)}</span></div>"
            f"<div class=ctx>in context: {example.context_html}</div></div>",
        )
    if not examples:
        parts.append("<div class=ex><span class=meta>no long kept spans sampled for this label</span></div>")
    parts.append("</div>")
    return "".join(parts)


def _truncate(value: str, limit: int) -> str:
    return value[:limit] + ("…" if len(value) > limit else "")


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{100 * numerator / denominator:.2f}%"


def _css() -> str:
    return """
    :root{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;color:#0f172a}
    *{box-sizing:border-box} body{margin:0;background:#f8fafc;line-height:1.5}
    .wrap{max-width:1180px;margin:0 auto;padding:40px 28px 80px}
    h1{font-size:25px;margin:0 0 4px} .sub{color:#64748b;margin:0 0 20px;font-size:14px}
    h2{font-size:18px;margin:34px 0 4px;display:flex;align-items:center;gap:10px}
    .pill{font-size:11px;color:#fff;background:#0f766e;border-radius:10px;padding:2px 9px;font-weight:600}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px}
    .col{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px}
    .col h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.5px}
    .col.rm h3{color:#b91c1c} .col.kp h3{color:#16a34a}
    .ex{border-top:1px solid #f1f5f9;padding:10px 0;font-size:12.5px}
    .ex:first-of-type{border-top:none}
    .why{font-weight:600;font-size:12.5px;margin-bottom:3px}
    .code{font-size:10px;color:#94a3b8;font-family:ui-monospace,Menlo,monospace;font-weight:400}
    .labchip{background:#0f172a;color:#fff;font-size:14px;padding:3px 12px;border-radius:8px}
    .span{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600;word-break:break-word;color:#1e293b}
    .ctx{color:#64748b;font-size:11.5px;margin-top:3px;font-family:ui-monospace,Menlo,monospace;word-break:break-word;max-height:74px;overflow:auto}
    mark{background:#fde68a;color:#7c2d12;padding:0 1px;border-radius:2px}
    .col.rm mark{background:#fecaca;color:#7f1d1d}
    .meta{color:#94a3b8;font-size:10.5px}
    .legend{font-size:11.5px;color:#475569;margin:10px 0 0}
    .legend span{display:inline-block;margin-right:14px}
    .dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:middle}
    footer{margin-top:44px;color:#94a3b8;font-size:12px;border-top:1px solid #e2e8f0;padding-top:14px}
    """
