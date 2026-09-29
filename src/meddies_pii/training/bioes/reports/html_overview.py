"""Render top-level report metrics, readiness gates, and dataset mix."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import TYPE_CHECKING, Any

from meddies_pii.annotations.bioes import ENTITY_LABELS

if TYPE_CHECKING:
    from .models import TrainingScan


PRIVATE_URL_READINESS_MINIMUM = 500


def _summary_label_counts(summary: Mapping[str, Any], scan: TrainingScan) -> dict[str, int]:
    counts = summary.get("label_counts")
    if isinstance(counts, Mapping):
        return {label: int(counts.get(label, 0) or 0) for label in ENTITY_LABELS}
    return {label: int(scan.label_counts.get(label, 0)) for label in ENTITY_LABELS}


def _as_int(value: object) -> int:
    """Convert one summary count exactly as `int()` does, rejecting the types it rejects.

    The summary is parsed JSON, so a count arrives as `object`. Listing the types `int()` accepts
    keeps every input that worked before working, and every input that raised still raising.

    Returns:
        The value as an ``int``, with the same truncation ``int()`` performs -- a float count
        rounds toward zero rather than to nearest. The isinstance gate lists exactly the
        types ``int()`` accepts, so this neither widens nor narrows the previous behaviour.

    Raises:
        TypeError: if the value is not a number or a numeric string.

    """
    if isinstance(value, (int, float, str)):
        return int(value)
    msg = f"summary count must be a number, got {type(value).__name__}"
    raise TypeError(msg)


def _summary_decision_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    counts = summary.get("decision_counts")
    if not isinstance(counts, Mapping):
        return {}
    return {str(key): _as_int(value) for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def _summary_counts(summary: Mapping[str, Any], key: str) -> dict[str, int]:
    counts = summary.get(key)
    if not isinstance(counts, Mapping):
        return {}
    return {
        str(item_key): _as_int(item_value)
        for item_key, item_value in sorted(counts.items(), key=lambda item: (-_as_int(item[1]), str(item[0])))
    }


def _metric_card(title: str, value: str, note: str) -> str:
    return (
        '<div class="card">'
        f'<div class="metric-title">{escape(title)}</div>'
        f'<div class="metric">{escape(value)}</div>'
        f'<div class="note">{escape(note)}</div>'
        "</div>"
    )


def _validation_rows_value(validation_scan: TrainingScan | None) -> str:
    return "n/a" if validation_scan is None else f"{validation_scan.rows_seen:,}"


def _validation_rows_note(validation_scan: TrainingScan | None) -> str:
    if validation_scan is None:
        return "no validation split attached"
    return f"invalid parsed rows: {len(validation_scan.invalid_rows):,}"


def _validation_spans_value(validation_scan: TrainingScan | None) -> str:
    return "n/a" if validation_scan is None else f"{validation_scan.span_count:,}"


def _readiness_items(
    *,
    scan: TrainingScan,
    validation_scan: TrainingScan | None,
    summary: Mapping[str, Any],
    modal_summary: Mapping[str, object],
) -> list[tuple[str, str, str]]:
    label_counts = _summary_label_counts(summary, scan)
    missing = [label for label, count in label_counts.items() if count <= 0]
    private_url_count = label_counts.get("private_url", 0)
    modal_delta = modal_summary.get("train_loss_delta")
    modal_pass = isinstance(modal_delta, int | float) and modal_delta < 0
    validation_label_counts = _summary_label_counts({}, validation_scan) if validation_scan is not None else {}
    validation_missing = [label for label, count in validation_label_counts.items() if count <= 0]
    eval_split = modal_summary.get("eval_dataset_split")
    train_config = modal_summary.get("train_config")
    eval_config = modal_summary.get("eval_config")
    pii_bioes_split_smoke = train_config == "pii-bioes" and eval_config == "pii-bioes" and eval_split == "validation"
    items: list[tuple[str, str, str]] = [
        (
            "pass" if scan.invalid_rows == [] else "fail",
            "Training JSONL parses",
            f"{len(scan.invalid_rows):,} invalid rows",
        ),
        (
            "pass" if not missing else "fail",
            "Every Meddies Labels label has support",
            "missing: " + ", ".join(missing) if missing else "all labels present",
        ),
        (
            "pass"
            if validation_scan is not None and validation_scan.invalid_rows == [] and not validation_missing
            else "warn",
            "Held-out validation split exists",
            (
                f"{validation_scan.rows_seen:,} rows; all labels present"
                if validation_scan is not None and not validation_missing
                else "missing validation split or labels: " + ", ".join(validation_missing)
            ),
        ),
        (
            "warn" if private_url_count < PRIVATE_URL_READINESS_MINIMUM else "pass",
            "private_url has enough positives",
            f"{private_url_count:,} positives; target more before final training",
        ),
        (
            "pass" if modal_pass else "warn",
            "Modal smoke loss drops",
            f"delta={modal_delta}" if modal_delta is not None else "no modal result attached",
        ),
        (
            "pass" if pii_bioes_split_smoke else "warn",
            "Published pii-bioes train/validation smoke",
            (
                "train_config=pii-bioes, eval split=validation"
                if pii_bioes_split_smoke
                else "need smoke using train split and validation split"
            ),
        ),
    ]
    return items


def _readiness_li(item: tuple[str, str, str]) -> str:
    status, title, note = item
    return f'<li class="{escape(status)}"><b>{escape(title)}</b><span>{escape(note)}</span></li>'


def _label_count_table(counts: Mapping[str, int]) -> str:
    max_count = max(counts.values(), default=1) or 1
    rows = []
    for label in ENTITY_LABELS:
        count = int(counts.get(label, 0))
        width = max(1, round((count / max_count) * 100))
        rows.append(
            "<tr>"
            f"<td><code>{escape(label)}</code></td>"
            f'<td class="num">{count:,}</td>'
            f'<td><div class="bar"><span style="width:{width}%"></span></div></td>'
            "</tr>",
        )
    return '<table class="counts"><tbody>' + "".join(rows) + "</tbody></table>"


def _target_mix_table(summary: Mapping[str, Any]) -> str:
    bucket_counts = _summary_counts(summary, "language_bucket_counts")
    kept_sources = _summary_counts(summary, "kept_source_counts")
    kept_languages = _summary_counts(summary, "kept_language_counts")
    total = int(
        summary.get("kept_rows") or summary.get("rows") or sum(bucket_counts.values()) or sum(kept_sources.values()) or 0,
    )
    if total <= 0:
        return '<p class="empty">No language/source summary attached.</p>'

    if bucket_counts:
        vietnamese = bucket_counts.get("vi", 0)
        english = bucket_counts.get("en", 0)
        other = bucket_counts.get("other", max(0, total - vietnamese - english))
        vi_rule = "language_bucket=vi"
        en_rule = "language_bucket=en"
        other_rule = "language_bucket=other"
    else:
        vietnamese = kept_sources.get("vietnamese", 0) + kept_sources.get("vietnamese-translated", 0)
        english = kept_languages.get("English", 0)
        other = max(0, total - vietnamese - english)
        vi_rule = "source=vietnamese + vietnamese-translated"
        en_rule = "language=English"
        other_rule = "remaining kept rows"
    rows = [
        ("Vietnamese target", vietnamese, "50%", vi_rule),
        ("English target", english, "30%", en_rule),
        ("Other target", other, "20%", other_rule),
    ]
    html_rows = ["<tr><th>bucket</th><th>rows</th><th>current</th><th>target</th><th>rule</th></tr>"]
    for name, count, target, rule in rows:
        percent = count / total * 100
        html_rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f'<td class="num">{count:,}</td>'
            f'<td class="num">{percent:.2f}%</td>'
            f'<td class="num">{escape(target)}</td>'
            f"<td><code>{escape(rule)}</code></td>"
            "</tr>",
        )
    return '<table class="dense">' + "".join(html_rows) + "</table>"


def _metadata_count_table(counts: Mapping[str, int], *, total: int) -> str:
    if not counts:
        return '<p class="empty">No metadata counts attached.</p>'
    rows = ["<tr><th>value</th><th>rows</th><th>%</th></tr>"]
    denominator = total or sum(counts.values()) or 1
    for key, value in counts.items():
        rows.append(
            "<tr>"
            f"<td><code>{escape(key)}</code></td>"
            f'<td class="num">{value:,}</td>'
            f'<td class="num">{value / denominator * 100:.2f}%</td>'
            "</tr>",
        )
    return '<table class="dense">' + "".join(rows) + "</table>"
