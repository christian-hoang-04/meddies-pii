"""Meddies Labels migration and audit helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from meddies_pii.languages import language_bucket
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PII_LABELS
from meddies_pii.training.bioes.data.legacy_decisions import (
    SpanDecision,
    decision_for_span,
    dedupe_and_validate_spans,
)
from meddies_pii.training.bioes.data.legacy_sources import (
    current_tagged_spans,
    original_spans,
    plain_text,
    row_uid,
)
from meddies_pii.training.bioes.data.record_io import write_jsonl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LegacyConversionStatus = Literal["kept", "quarantined"]


@dataclass(frozen=True, slots=True)
class ConvertedLegacyRow:
    uid: str
    text: str
    labels: tuple[CharSpan, ...]
    decisions: tuple[SpanDecision, ...]
    status: LegacyConversionStatus = "kept"
    quarantine_reason: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        source = str(self.source.get("source") or "UNKNOWN")
        language = str(self.source.get("language") or "UNKNOWN")
        return {
            "text": self.text,
            "label": [
                {
                    "category": span.label,
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                }
                for span in self.labels
            ],
            "info": {
                "id": self.uid,
                "source_dataset": "Meddies/meddies-pii",
                "source": source,
                "language": language,
                "domain_bucket": "medical",
                "language_bucket": language_bucket(language=language, source=source),
            },
        }

    def audit_record(self) -> dict[str, object]:
        return {
            "uid": self.uid,
            "status": self.status,
            "quarantine_reason": self.quarantine_reason,
            "decisions": [decision.asdict() for decision in self.decisions],
        }


def convert_legacy_row_to_pii_labels(row: Mapping[str, Any], *, default_uid: str = "row-0") -> ConvertedLegacyRow:
    """Convert one legacy/current row to a Meddies Labels training record.

    Returns:
        The converted row carrying both the record and the per-decision audit trail. Original
        spans are preferred; only when a row has none does it fall back to reparsing the
        current tagged text, and that fallback is recorded so the audit distinguishes a row
        whose spans were authoritative from one whose spans were recovered.

    """
    uid = row_uid(row, default_uid)
    raw = plain_text(row)
    source_spans = original_spans(row, raw)
    recover_collapsed = False
    if not source_spans:
        source_spans = current_tagged_spans(row, raw)
        recover_collapsed = True

    decisions: list[SpanDecision] = []
    labels: list[CharSpan] = []
    for span in source_spans:
        decision = decision_for_span(span, raw, recover_collapsed=recover_collapsed)
        decisions.append(decision)
        if decision.decision == "quarantine":
            return ConvertedLegacyRow(
                uid=uid,
                text=raw,
                labels=(),
                decisions=tuple(decisions),
                status="quarantined",
                quarantine_reason=decision.rule,
                source=row,
            )
        if decision.decision == "keep" and decision.new_label is not None:
            labels.append(
                CharSpan(
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    label=decision.new_label,
                ),
            )

    try:
        validated_labels = dedupe_and_validate_spans(labels)
    except ValueError:
        return ConvertedLegacyRow(
            uid=uid,
            text=raw,
            labels=(),
            decisions=tuple(decisions),
            status="quarantined",
            quarantine_reason="overlapping_spans",
            source=row,
        )
    return ConvertedLegacyRow(
        uid=uid,
        text=raw,
        labels=validated_labels,
        decisions=tuple(decisions),
        source=row,
    )


def convert_legacy_rows_to_pii_labels(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[ConvertedLegacyRow], dict[str, object]]:
    migrated = [convert_legacy_row_to_pii_labels(row, default_uid=f"row-{index}") for index, row in enumerate(rows)]
    return _kept_and_summary(migrated, source_row_count=len(rows))


def _kept_and_summary(
    migrated: Sequence[ConvertedLegacyRow],
    *,
    source_row_count: int,
) -> tuple[list[ConvertedLegacyRow], dict[str, object]]:
    kept = [row for row in migrated if row.status == "kept"]
    quarantined = [row for row in migrated if row.status == "quarantined"]
    label_counts: dict[str, int] = dict.fromkeys(PII_LABELS, 0)
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    kept_source_counts: Counter[str] = Counter()
    quarantined_source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    kept_language_counts: Counter[str] = Counter()
    quarantined_language_counts: Counter[str] = Counter()
    label_counts_by_language: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in migrated:
        source = str(row.source.get("source") or "UNKNOWN")
        language = str(row.source.get("language") or "UNKNOWN")
        source_counts[source] += 1
        language_counts[language] += 1
        if row.status == "kept":
            kept_source_counts[source] += 1
            kept_language_counts[language] += 1
        else:
            quarantined_source_counts[source] += 1
            quarantined_language_counts[language] += 1
        for span in row.labels:
            label_counts[span.label] += 1
            label_counts_by_language[language][span.label] += 1
        decision_counts.update(f"{decision.decision}:{decision.rule}" for decision in row.decisions)
    return kept, {
        "rows": source_row_count,
        "kept_rows": len(kept),
        "quarantined_rows": len(quarantined),
        "label_counts": label_counts,
        "decision_counts": dict(decision_counts),
        "language_counts": dict(sorted(language_counts.items())),
        "kept_language_counts": dict(sorted(kept_language_counts.items())),
        "quarantined_language_counts": dict(sorted(quarantined_language_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "kept_source_counts": dict(sorted(kept_source_counts.items())),
        "quarantined_source_counts": dict(sorted(quarantined_source_counts.items())),
        "label_counts_by_language": {
            language: {label: label_counts_by_language[language].get(label, 0) for label in PII_LABELS}
            for language in sorted(label_counts_by_language)
        },
    }


def write_legacy_conversion_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
    audit_path: str | Path,
    summary_path: str | Path,
) -> dict[str, object]:
    """Write kept Meddies Labels JSONL, per-row audit JSONL, and summary JSON.

    Returns:
        The summary, already written alongside the two JSONL files. The audit file carries
        every migrated row while the output carries only the kept ones, so a dropped row is
        still accounted for and can be traced to the decision that dropped it.

    """
    migrated = [convert_legacy_row_to_pii_labels(row, default_uid=f"row-{index}") for index, row in enumerate(rows)]
    kept, summary = _kept_and_summary(migrated, source_row_count=len(rows))
    write_jsonl(output_path, (row.as_record() for row in kept))
    write_jsonl(audit_path, (row.audit_record() for row in migrated))
    target = Path(summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
