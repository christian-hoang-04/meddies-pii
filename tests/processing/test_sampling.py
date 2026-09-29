from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from meddies_pii.jsonl import read_jsonl, write_jsonl
from meddies_pii.processing.sampling import sample_and_check

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from meddies_pii.json_types import JsonObject


def _read_records(path: Path) -> list[JsonObject]:
    return list(read_jsonl(path))


def test_sample_and_check_logs_missing_file_without_writing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"

    with caplog.at_level(logging.INFO, logger="meddies_pii.processing.sampling"):
        sample_and_check(str(missing_path))

    assert "File not found" in caplog.text
    assert not (tmp_path / "sample_missing.jsonl").exists()


def test_sample_and_check_reports_error_only_input_without_writing(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "errors.jsonl"
    write_jsonl(input_path, [{"error": "provider rejected row"}])

    with caplog.at_level(logging.INFO, logger="meddies_pii.processing.sampling"):
        sample_and_check(str(input_path), sample_size=10)

    assert 'Found 1 lines with "error"' in caplog.text
    assert "No valid records found." in caplog.text
    assert not (tmp_path / "sample_errors.jsonl").exists()


def test_sample_and_check_saves_bounded_sample_and_reports_parse_failure(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "records.jsonl"
    records = [
        {"id": "valid", "content": "Seen [Nguyen Van A]<human_name>."},
        {"id": "bad-content", "content": "No tag exists here."},
        {"id": "bad-output", "output": "Also missing an inline tag."},
        {"error": "excluded from quality check"},
    ]
    write_jsonl(input_path, records)
    with input_path.open("a", encoding="utf-8") as file:
        file.write("not valid json\n")

    random.seed(18)
    with caplog.at_level(logging.INFO):
        sample_and_check(str(input_path), sample_size=20)

    saved_path = tmp_path / "sample_records.jsonl"
    saved_records = _read_records(saved_path)
    assert len(saved_records) == 3
    assert {record["id"] for record in saved_records} == {
        "valid",
        "bad-content",
        "bad-output",
    }
    assert 'Found 1 lines with "error"' in caplog.text
    assert "Total valid records: 3" in caplog.text
    assert "Quality check on 3 samples" in caplog.text
    assert "Passed: 1/3 (33.3%)" in caplog.text
    assert "Failure examples:" in caplog.text
    assert "Sample 1: expected at least one inline PII span" in caplog.text
    assert "Sample 2: expected at least one inline PII span" in caplog.text
    assert "Preview: No tag exists here...." in caplog.text
    assert "Preview: Also missing an inline tag...." in caplog.text


def test_sample_and_check_is_seed_deterministic_for_subsamples(
    tmp_path: Path,
) -> None:
    records = [{"id": str(index), "content": f"[Person {index}]<human_name>"} for index in range(4)]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_jsonl(first_path, records)
    write_jsonl(second_path, records)

    random.seed(71)
    sample_and_check(str(first_path), sample_size=2)
    random.seed(71)
    sample_and_check(str(second_path), sample_size=2)

    assert _read_records(tmp_path / "sample_first.jsonl") == _read_records(tmp_path / "sample_second.jsonl")
