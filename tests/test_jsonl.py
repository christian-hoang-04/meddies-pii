"""Behavior tests for meddies_pii.jsonl primitives.

All file I/O goes through tmp_path — no mock_open, no patching. Tests assert
on file contents and yielded records, not on which open() call happened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from meddies_pii.jsonl import (
    append_jsonl,
    count_jsonl,
    read_jsonl,
    write_jsonl,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    records = [
        {"id": 1, "text": "hello"},
        {"id": 2, "text": "world"},
        {"id": 3, "text": "vietnamese: xin chào"},
    ]
    path = tmp_path / "data.jsonl"
    written = write_jsonl(path, records)
    assert written == 3
    assert list(read_jsonl(path)) == records


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blanks.jsonl"
    path.write_text(
        '{"a": 1}\n\n   \n{"a": 2}\n',
        encoding="utf-8",
    )
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_read_logs_and_skips_malformed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"a": 1}\nnot json at all\n{"a": 2}\n',
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="meddies_pii.jsonl"):
        records = list(read_jsonl(path))

    assert records == [{"a": 1}, {"a": 2}]
    assert any("malformed JSONL" in m for m in caplog.messages)
    assert any(":2:" in m for m in caplog.messages)


def test_append_grows_file(tmp_path: Path) -> None:
    path = tmp_path / "grow.jsonl"
    write_jsonl(path, [{"i": 1}])
    appended = append_jsonl(path, [{"i": 2}, {"i": 3}])

    assert appended == 2
    assert list(read_jsonl(path)) == [{"i": 1}, {"i": 2}, {"i": 3}]


def test_count_jsonl_skips_parsing(tmp_path: Path) -> None:
    path = tmp_path / "count.jsonl"
    path.write_text(
        '{"a": 1}\n\n{"a": 2}\nnot json but still a non-blank line\n{"a": 3}\n',
        encoding="utf-8",
    )
    assert count_jsonl(path) == 4


def test_count_missing_file_returns_zero(tmp_path: Path) -> None:
    assert count_jsonl(tmp_path / "does-not-exist.jsonl") == 0


def test_utf8_ensure_ascii_false_preserved(tmp_path: Path) -> None:
    """Multi-language content survives roundtrip without escape sequences."""
    records = [
        {"vi": "Bệnh nhân"},
        {"zh": "病人"},
        {"th": "ผู้ป่วย"},
    ]
    path = tmp_path / "intl.jsonl"
    write_jsonl(path, records)
    raw = path.read_text(encoding="utf-8")

    assert "Bệnh nhân" in raw
    assert "病人" in raw
    assert "ผู้ป่วย" in raw
    assert "\\u" not in raw


def test_write_returns_zero_for_empty_iterable(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    assert write_jsonl(path, []) == 0
    assert path.stat().st_size == 0


def test_write_overwrites_existing(tmp_path: Path) -> None:
    path = tmp_path / "overwrite.jsonl"
    write_jsonl(path, [{"a": 1}, {"a": 2}, {"a": 3}])
    write_jsonl(path, [{"b": 1}])

    assert list(read_jsonl(path)) == [{"b": 1}]


def test_read_yields_lazily(tmp_path: Path) -> None:
    """read_jsonl should be a generator, not a materialized list."""
    path = tmp_path / "lazy.jsonl"
    write_jsonl(path, [{"i": i} for i in range(5)])

    gen = read_jsonl(path)
    first = next(gen)
    assert first == {"i": 0}

    remaining = list(gen)
    assert remaining == [{"i": i} for i in range(1, 5)]


def test_read_skips_non_object_json(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """JSONL records are always objects; lists/strings/null get logged and skipped."""
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"a": 1}\n[1, 2, 3]\n"just a string"\nnull\n{"a": 2}\n',
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="meddies_pii.jsonl"):
        records = list(read_jsonl(path))

    assert records == [{"a": 1}, {"a": 2}]
    assert any("non-object JSONL" in m for m in caplog.messages)


def test_append_repairs_crash_truncated_file(tmp_path: Path) -> None:
    """A previous crash-truncated write left no trailing newline.

    append_jsonl must not produce glued `}{` records — count_jsonl would
    undercount and read_jsonl couldn't parse the joined line. Regression
    test for the bug found by codex review.

    Simulate a write that lost its trailing newline (process killed mid-line).

    """
    path = tmp_path / "crash.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}', encoding="utf-8")
    assert not path.read_text(encoding="utf-8").endswith("\n")

    appended = append_jsonl(path, [{"a": 3}])

    assert appended == 1
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert count_jsonl(path) == 3


def test_append_to_nonexistent_file_does_not_seek(tmp_path: Path) -> None:
    """append_jsonl on a missing file creates it.

    It must not try to seek for a trailing newline that no file carries yet.
    """
    path = tmp_path / "fresh.jsonl"
    appended = append_jsonl(path, [{"a": 1}])

    assert appended == 1
    assert list(read_jsonl(path)) == [{"a": 1}]


def test_append_to_clean_file_does_not_double_newline(tmp_path: Path) -> None:
    """If the previous write ended cleanly, no extra blank line slips in."""
    path = tmp_path / "clean.jsonl"
    write_jsonl(path, [{"a": 1}])
    append_jsonl(path, [{"a": 2}])

    raw = path.read_text(encoding="utf-8")
    assert raw == '{"a": 1}\n{"a": 2}\n'
