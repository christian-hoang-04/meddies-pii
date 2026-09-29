"""Usage-record file persistence."""

from __future__ import annotations

import json
from pathlib import Path

from meddies_pii.generation.usage.records import (
    UsageRecord,
    decode_usage_record,
    merge_usage,
)


def write_usage_record(record: UsageRecord, usage_dir: str | Path) -> Path:
    """Persist one day, merging a valid same-date record when present.

    A malformed existing file is not overwritten via silent coercion: this run's
    validated record replaces it, preserving the previous operational fallback.

    Returns:
        The path written, ``<usage_dir>/<date>.json``. When a valid record for the same date is
        already there the two are merged, so several runs on one day accumulate rather than the
        last one winning. An existing file that will not validate is replaced outright, because
        merging into it would carry an unvalidated shape forward.

    """
    directory = Path(usage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['date']}.json"
    merged = record
    if path.exists():
        try:
            merged = merge_usage(
                decode_usage_record(json.loads(path.read_text(encoding="utf-8"))),
                record,
            )
        except (json.JSONDecodeError, OSError, ValueError):
            merged = record
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
