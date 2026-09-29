"""JSONL file primitives.

Four pure functions for reading, writing, appending, and counting JSONL records.
File I/O is UTF-8 throughout, with ensure_ascii=False on writes — the project's
multi-language data (Vietnamese, Chinese, Thai, Tamil, etc.) must round-trip
without escape sequences. Bad lines on read are logged and skipped; callers
that need bad lines surfaced as failures use their own line loops.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.json_types import JsonObject, is_json_object

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)


def read_jsonl(path: str | Path) -> Iterator[JsonObject]:
    """Yield JSON objects (dicts) from a JSONL file.

    Blank lines are skipped silently. Malformed lines and non-object
    JSON values (lists, strings, numbers, null) are logged and skipped —
    JSONL records in this project are always objects.

    Yields:
        Each line that decodes to a JSON object, in file order. Blank lines are skipped
        silently; a malformed line or a well-formed non-object value is logged with its line
        number and skipped. Nothing raises, so one bad line in a large corpus file costs that
        line rather than the read -- and the log is what keeps the loss visible.

    """
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL at %s:%s: %s", path, line_number, exc)
                continue
            if not is_json_object(value):
                logger.warning(
                    "Skipping non-object JSONL at %s:%s: got %s",
                    path,
                    line_number,
                    type(value).__name__,
                )
                continue
            yield value


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    return _dump(path, records, mode="w")


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    return _dump(path, records, "a")


def count_jsonl(path: str | Path) -> int:
    if not Path(path).exists():
        return 0
    with Path(path).open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _dump(path: str | Path, records: Iterable[dict[str, Any]], mode: str) -> int:
    """Crash-truncated previous writes can leave a record without a trailing newline.

    Gluing the next record onto it produces `}{` which count_jsonl undercounts and read_jsonl can't parse. (Append-mode
    files don't support seek/tell on every platform, hence the separate binary-read pass.).

    Returns:
        The number of records written by this call, not the file's total. In append mode the
        file's last byte is checked first and a newline added when a previous crash left one
        missing, so the first appended record starts on its own line instead of gluing onto the
        truncated one.

    """
    if mode == "a" and Path(path).exists() and Path(path).stat().st_size > 0:
        with Path(path).open("rb") as fr:
            fr.seek(-1, os.SEEK_END)
            if fr.read(1) != b"\n":
                with Path(path).open("ab") as fa:
                    fa.write(b"\n")
    count = 0
    with Path(path).open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
