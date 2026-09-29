from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meddies_pii.taxonomy import PII_LABELS

INLINE_TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]<([a-zA-Z][a-zA-Z0-9_-]*)>")
INLINE_TEXT_FIELDS = ("content", "text_tagged", "output")


@dataclass(frozen=True)
class InlineSpan:
    label: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class InlineRecord:
    record_id: str
    language: str | None
    tagged_text: str
    plain_text: str
    spans: tuple[InlineSpan, ...]


@dataclass(frozen=True)
class ValidationResult:
    records: tuple[InlineRecord, ...]
    errors: tuple[str, ...]

    @property
    def span_count(self) -> int:
        return sum(len(record.spans) for record in self.records)

    @property
    def labels(self) -> tuple[str, ...]:
        seen = {span.label for record in self.records for span in record.spans}
        return tuple(label for label in PII_LABELS if label in seen)


def validate_inline_jsonl(path: str | Path) -> ValidationResult:
    jsonl_path = Path(path)
    records: list[InlineRecord] = []
    errors: list[str] = []

    try:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return ValidationResult(records=(), errors=(f"{jsonl_path}: {exc}",))

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"{jsonl_path}:{line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{jsonl_path}:{line_number}: expected JSON object")
            continue
        record, record_errors = parse_inline_record(value, line_number=line_number)
        errors.extend(f"{jsonl_path}:{line_number}: {error}" for error in record_errors)
        if record_errors:
            continue
        if record is None:
            errors.append(f"{jsonl_path}:{line_number}: record parser returned no record")
            continue
        records.append(record)

    if not records and not errors:
        errors.append(f"{jsonl_path}: no JSONL records found")

    return ValidationResult(records=tuple(records), errors=tuple(errors))


def parse_inline_record(row: dict[str, Any], *, line_number: int) -> tuple[InlineRecord | None, tuple[str, ...]]:
    tagged_text = _inline_text(row)
    if tagged_text is None:
        return None, (f"missing one of: {', '.join(INLINE_TEXT_FIELDS)}",)

    errors: list[str] = []
    plain_parts: list[str] = []
    spans: list[InlineSpan] = []
    cursor = 0
    plain_length = 0

    for match in INLINE_TAG_PATTERN.finditer(tagged_text):
        value, raw_label = match.groups()
        label = raw_label.lower()
        prefix = tagged_text[cursor : match.start()]
        plain_parts.append(prefix)
        plain_length += len(prefix)
        start = plain_length
        plain_parts.append(value)
        end = start + len(value)
        plain_length = end
        if label not in PII_LABELS:
            errors.append(f"unknown PII label {raw_label!r}")
        else:
            spans.append(InlineSpan(label=label, start=start, end=end, text=value))
        cursor = match.end()

    plain_parts.append(tagged_text[cursor:])
    if not spans and not errors:
        errors.append("expected at least one inline PII span")

    if errors:
        return None, tuple(errors)

    record_id = str(row.get("id") or f"line-{line_number}")
    language = row.get("language")
    return (
        InlineRecord(
            record_id=record_id,
            language=str(language) if language is not None else None,
            tagged_text=tagged_text,
            plain_text="".join(plain_parts),
            spans=tuple(spans),
        ),
        (),
    )


def _inline_text(row: dict[str, Any]) -> str | None:
    for field in INLINE_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            return value
    return None
