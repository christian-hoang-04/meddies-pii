"""Canonical record construction and span recovery for external PII sources."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from meddies_pii.taxonomy import PII_LABELS
from meddies_pii.training.bioes.data.record_schema import NormalizedRecord, Record, record_from_object

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from meddies_pii.spans import CharSpan


# reason: external record exposes text/language as its public contract; bundling would break callers.
def external_record(  # ruff: ignore[too-many-arguments]
    *,
    text: str,
    spans: Sequence[CharSpan],
    uid: str,
    source_dataset: str,
    source: str,
    language: str,
    domain_bucket: str,
    language_bucket: str,
) -> NormalizedRecord:
    return {
        "text": text,
        "label": [
            {
                "category": span.label,
                "start": span.start,
                "end": span.end,
                "text": span.text,
            }
            for span in spans
        ],
        "info": {
            "id": uid,
            "source_dataset": source_dataset,
            "source": source,
            "language": language,
            "domain_bucket": domain_bucket,
            "language_bucket": language_bucket,
        },
    }


def locate_external_span(*, text: str, value: str, start: object, end: object) -> tuple[int, int] | None:
    if isinstance(start, int) and isinstance(end, int) and text[start:end] == value:
        return start, end
    if value:
        found = text.find(value)
        if found != -1:
            return found, found + len(value)
    return None


def dedupe_non_overlapping(spans: Sequence[CharSpan]) -> list[CharSpan]:
    deduped = sorted(
        {
            (span.start, span.end, span.label): span
            for span in spans
            if span.start < span.end and span.label in PII_LABELS
        }.values(),
        key=lambda span: (span.start, -(span.end - span.start), span.label),
    )
    kept: list[CharSpan] = []
    cursor = -1
    for span in deduped:
        if span.start >= cursor:
            kept.append(span)
            cursor = span.end
    return kept


def read_jsonl_records(path: Path) -> Iterable[Record]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield record_from_object(json.loads(line), source=f"{path}:{line_number}")
