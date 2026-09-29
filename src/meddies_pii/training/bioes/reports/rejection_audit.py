from __future__ import annotations

import glob
import html
import re
import statistics as st
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from meddies_pii.training.bioes.eval.rejection_policy import (
    RejectionClassification,
    RejectionVerdict,
    classify_rejection_reason,
)
from meddies_pii.training.bioes.reports.json_narrowing import (
    parse_json_object_line,
)

REASON_EXAMPLES_PER_CATEGORY = 4

DEFAULT_REJECTION_PATTERNS: tuple[str, ...] = (
    "data/bioes-v2/synthetic/**/rejected*.jsonl",
    "data/run2a/**/rejected*.jsonl",
)

ORDER: tuple[RejectionVerdict, ...] = tuple(RejectionVerdict)
VCOLOR: dict[RejectionVerdict, str] = {
    RejectionVerdict.OVER_FILTER: "#dc2626",
    RejectionVerdict.PARSER_BUG: "#d97706",
    RejectionVerdict.BORDERLINE: "#7c3aed",
    RejectionVerdict.CORRECT: "#16a34a",
    RejectionVerdict.UNCLASSIFIED: "#64748b",
}
PHONE_RE = re.compile(r"\[([^\[\]]{0,40})\]<phone_number>")


@dataclass(frozen=True, slots=True)
class RejectionAuditEvidence:
    dropped_docs: int
    reason_counts: Counter[str]
    label_counts_by_reason: Mapping[str, Sequence[int]]
    examples_by_reason: Mapping[str, Sequence[Mapping[str, object]]]
    classifications: Mapping[str, RejectionClassification]
    verdict_counts: Counter[RejectionVerdict]
    total_reasons: int
    recoverable_reasons: int


def load_rejection_records(
    patterns: Sequence[str] = DEFAULT_REJECTION_PATTERNS,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for pattern in patterns:
        # reason: `patterns` is a public parameter typed Sequence[str], so a caller may pass a whole path
        # reason: pattern including its anchor. Path.glob cannot express an absolute pattern, so the swap
        # reason: would narrow what this function accepts while changing nothing for the defaults.
        for file_name in glob.glob(pattern, recursive=True):  # ruff: ignore[glob]
            with Path(file_name).open(encoding="utf-8") as fh:
                records.extend(value for line in fh if (value := parse_json_object_line(line)) is not None)
    return records


def excerpt(content: str, errors: Sequence[str], limit: int = 420) -> str:
    """Show a window that includes the offending bit when we can locate it.

    Returns:
        An HTML-escaped excerpt. For a phone-span error whose number is locatable the window
        is CENTRED on the match and the match is highlighted, so the excerpt may start
        mid-document and is not the first ``limit`` characters. Otherwise it is the leading
        ``limit`` characters with an ellipsis when truncated. The return is always safe to
        embed; callers must not escape it again.

    """
    if "short_phone_span" in errors or "underbounded_phone_span" in errors:
        match = PHONE_RE.search(content)
        if match:
            start = max(0, match.start() - 80)
            fragment = content[start : match.end() + 120]
            return _highlight(fragment, match.group(0))
    return html.escape(content[:limit]) + ("…" if len(content) > limit else "")


def _highlight(fragment: str, needle: str) -> str:
    escaped_fragment = html.escape(fragment)
    escaped_needle = html.escape(needle)
    return escaped_fragment.replace(escaped_needle, f"<mark>{escaped_needle}</mark>")


def analyze_rejections(
    records: Sequence[Mapping[str, object]],
) -> RejectionAuditEvidence:
    reason_count: Counter[str] = Counter()
    reason_labels: dict[str, list[int]] = defaultdict(list)
    reason_examples: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        label_count = _int_value(record.get("parsed_label_count"))
        errors = record.get("errors")
        normalized_errors = (
            [str(error) for error in errors]
            if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes))
            else ["(none)"]
        )
        for error in normalized_errors:
            reason = error.split(":")[0]
            reason_count[reason] += 1
            reason_labels[reason].append(label_count)
            content = str(record.get("content") or "")
            if len(reason_examples[reason]) < REASON_EXAMPLES_PER_CATEGORY and content.strip():
                reason_examples[reason].append(record)

    classifications = {reason: classify_rejection_reason(reason) for reason in reason_count}
    verdict_counts: Counter[RejectionVerdict] = Counter()
    recoverable_reasons = 0
    for reason, count in reason_count.items():
        classification = classifications[reason]
        verdict_counts[classification.verdict] += count
        if classification.recoverable:
            recoverable_reasons += count
    return RejectionAuditEvidence(
        dropped_docs=len(records),
        reason_counts=reason_count,
        label_counts_by_reason=reason_labels,
        examples_by_reason=reason_examples,
        classifications=classifications,
        verdict_counts=verdict_counts,
        total_reasons=sum(reason_count.values()),
        recoverable_reasons=recoverable_reasons,
    )


def build_rejection_audit_html(records: Sequence[Mapping[str, object]]) -> str:
    return render_rejection_audit_html(analyze_rejections(records))


def render_rejection_audit_html(evidence: RejectionAuditEvidence) -> str:
    """Render already-classified rejection evidence without owning policy.

    Returns:
        The whole audit page as one self-contained HTML string with its CSS inlined. This
        renders a verdict it did not reach: classification lives in ``analyze_rejections``,
        so changing what counts as a recoverable rejection means editing that function, and
        editing this one only changes presentation.

    """
    css = _css()
    parts = [f"<!doctype html><meta charset=utf-8><title>rejection audit</title><style>{css}</style><div class=wrap>"]
    parts.extend((
        "<h1>bioes-v2 — synthetic rejection audit</h1>",
        (
            f"<p class=sub>{evidence.total_reasons} rejection-reasons across "
            f"{evidence.dropped_docs} dropped docs. "
            "The question for each: did we correctly drop bad data, or carelessly drop a good doc?</p>"
        ),
        _cards(evidence.dropped_docs, evidence.verdict_counts, evidence.recoverable_reasons, evidence.total_reasons),
        _verdict_bar(evidence.verdict_counts, evidence.recoverable_reasons, evidence.total_reasons),
        _reason_table(evidence.reason_counts, evidence.label_counts_by_reason, evidence.classifications),
        _examples(evidence.reason_counts, evidence.examples_by_reason, evidence.classifications),
        (
            f"<footer>Audit of {evidence.dropped_docs} rejected synthetic docs "
            "(local rejected*.jsonl). "
            "Reasons defined in <code>validate_tagged_document</code> (synthetic.py). "
            "Recoverable = over-filter + parser-bug.</footer></div>"
        ),
    ))
    return "".join(parts)


def write_rejection_audit_report(records: Sequence[Mapping[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_rejection_audit_html(records), encoding="utf-8")
    return output_path


def _int_value(value: object) -> int:
    return value if isinstance(value, int) else 0


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0%"
    return f"{100 * numerator / denominator:.0f}%"


def _cards(
    dropped_docs: int,
    verdict_total: Counter[RejectionVerdict],
    recoverable: int,
    total: int,
) -> str:
    cards = [
        ("Dropped docs", str(dropped_docs), "#0f172a"),
        (
            "Over-filtered (good)",
            str(verdict_total[RejectionVerdict.OVER_FILTER]),
            VCOLOR[RejectionVerdict.OVER_FILTER],
        ),
        (
            "Parser-bug (good)",
            str(verdict_total[RejectionVerdict.PARSER_BUG]),
            VCOLOR[RejectionVerdict.PARSER_BUG],
        ),
        ("Recoverable", _percent(recoverable, total), "#0f766e"),
    ]
    return (
        "<div class=cards>"
        + "".join(
            f'<div class=card><div class=cn style="color:{color}">{value}</div><div class=cl>{key}</div></div>'
            for key, value, color in cards
        )
        + "</div>"
    )


def _verdict_bar(verdict_total: Counter[RejectionVerdict], recoverable: int, total: int) -> str:
    parts = ["<div class=barbig>"]
    for verdict in ORDER:
        count = verdict_total.get(verdict, 0)
        if count and total:
            parts.append(
                f'<div style="width:{100 * count / total:.1f}%;background:{VCOLOR[verdict]}">{verdict} {count}</div>',
            )
    parts.extend((
        "</div>",
        (
            f"<p class=note><b>{recoverable} of {total}</b> rejections ({_percent(recoverable, total)}) are "
            "rich docs we can recover: fix the phone validator (drop the bad <i>span</i>, keep the <i>doc</i>) "
            "and the inline-tag parser (bracket handling in JSON/FHIR).</p>"
        ),
    ))
    return "".join(parts)


def _reason_table(
    reason_count: Counter[str],
    reason_labels: Mapping[str, Sequence[int]],
    classifications: Mapping[str, RejectionClassification],
) -> str:
    parts = ["<h2>By reason</h2>"]
    parts.append(
        "<table><tr><th>reason</th><th>count</th><th>median labels in dropped doc</th><th>verdict</th><th>why</th></tr>",
    )
    for reason, count in reason_count.most_common():
        classification = classifications[reason]
        verdict = classification.verdict
        why = classification.explanation
        median = int(st.median(reason_labels[reason])) if reason_labels[reason] else 0
        parts.append(
            f"<tr><td><code>{html.escape(reason)}</code></td><td class=num>{count}</td><td class=num>{median}</td>"
            f'<td><span class=badge style="background:{VCOLOR[verdict]}">{verdict}</span></td>'
            f'<td style="font-size:12px;color:#475569">{html.escape(why)}</td></tr>',
        )
    parts.extend((
        "</table>",
        (
            "<p class=note>“median labels in dropped doc”: a high number means we threw away a doc that was "
            "otherwise rich (many correct spans). Over-filter + parser-bug rows have ~19 — near-complete docs.</p>"
        ),
    ))
    return "".join(parts)


def _examples(
    reason_count: Counter[str],
    reason_examples: Mapping[str, Sequence[Mapping[str, object]]],
    classifications: Mapping[str, RejectionClassification],
) -> str:
    parts: list[str] = []
    for verdict in ORDER:
        reasons = [reason for reason in reason_count if classifications[reason].verdict == verdict]
        if not reasons:
            continue
        parts.append(f'<h2><span class=badge style="background:{VCOLOR[verdict]}">{verdict}</span> &nbsp;examples</h2>')
        for reason in sorted(reasons, key=lambda item: -reason_count[item]):
            parts.extend(_example_card(reason, record) for record in reason_examples.get(reason, [])[:3])
    return "".join(parts)


def _example_card(reason: str, record: Mapping[str, object]) -> str:
    label_count = _int_value(record.get("parsed_label_count"))
    errors = _string_list(record.get("errors"))
    return (
        "<div class=ex>"
        f"<div class=meta><code>{html.escape(reason)}</code> · {html.escape(str(record.get('language', '')))} · "
        f"{html.escape(str(record.get('scenario', '')))} · format “{html.escape(str(record.get('text_format', '')))}” · "
        f"<b>{label_count} labels parsed</b> · errors: {html.escape(', '.join(errors))}</div>"
        f"<pre>{excerpt(str(record.get('content', '')), errors)}</pre>"
        "</div>"
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _css() -> str:
    return """
    :root{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;color:#0f172a}
    *{box-sizing:border-box} body{margin:0;background:#f8fafc;line-height:1.5}
    .wrap{max-width:980px;margin:0 auto;padding:44px 30px 80px}
    h1{font-size:26px;margin:0 0 4px} .sub{color:#64748b;margin:0 0 24px;font-size:14px}
    h2{font-size:16px;margin:34px 0 4px} .note{color:#64748b;font-size:13px;margin:0 0 14px}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}
    .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px;text-align:center}
    .cn{font-size:22px;font-weight:650} .cl{font-size:11px;color:#64748b;margin-top:3px}
    table{border-collapse:collapse;width:100%;background:#fff;border:1px solid \
#e2e8f0;border-radius:12px;overflow:hidden;font-size:13px}
    th,td{padding:8px 11px;text-align:left;border-bottom:1px solid #f1f5f9}
    th{background:#f8fafc;font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.4px}
    td.num{text-align:right;font-variant-numeric:tabular-nums}
    .badge{color:#fff;font-size:10px;padding:2px 8px;border-radius:10px;font-weight:600;white-space:nowrap}
    .ex{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin:10px 0}
    .ex .meta{font-size:11px;color:#64748b;margin-bottom:7px}
    .ex pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:11.5px;\
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f8fafc;border-radius:7px;\
padding:9px 11px;max-height:230px;overflow:auto}
    mark{background:#fecaca;color:#991b1b;padding:0 2px;border-radius:3px}
    .barbig{display:flex;height:30px;border-radius:8px;overflow:hidden;margin:6px 0 2px;border:1px solid #e2e8f0}
    .barbig div{display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:600}
    footer{margin-top:44px;color:#94a3b8;font-size:12px;border-top:1px solid #e2e8f0;padding-top:14px}
    code{background:#f1f5f9;padding:1px 5px;border-radius:5px;font-size:12px}
    """
