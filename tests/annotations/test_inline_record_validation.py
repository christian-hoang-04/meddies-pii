from __future__ import annotations

import json
from typing import TYPE_CHECKING

from meddies_pii.annotations.inline_records import validate_inline_jsonl

if TYPE_CHECKING:
    from pathlib import Path


def test_validator_reports_an_unreadable_input_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    result = validate_inline_jsonl(missing)

    assert result.records == ()
    assert len(result.errors) == 1
    assert str(missing) in result.errors[0]
    assert "No such file" in result.errors[0]


def test_validator_rejects_an_empty_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("\n \n", encoding="utf-8")

    result = validate_inline_jsonl(input_path)

    assert result.records == ()
    assert result.errors == (f"{input_path}: no JSONL records found",)


def test_validator_reports_each_malformed_record_without_losing_valid_rows(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "mixed.jsonl"
    input_path.write_text(
        "\n".join([
            "",
            "{not-json}",
            json.dumps(["not", "an", "object"]),
            json.dumps({}),
            json.dumps({"content": 42, "text_tagged": "plain clinical text"}),
            json.dumps({
                "content": "[Ada]<human_name>",
                "id": "valid",
                "language": "en",
            }),
        ]),
        encoding="utf-8",
    )

    result = validate_inline_jsonl(input_path)

    assert [record.record_id for record in result.records] == ["valid"]
    assert len(result.errors) == 4
    assert any("invalid JSON" in error for error in result.errors)
    assert any("expected JSON object" in error for error in result.errors)
    assert any("missing one of" in error for error in result.errors)
    assert any("expected at least one inline PII span" in error for error in result.errors)


def test_validator_accepts_the_last_supported_text_field_and_default_identity(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "output-field.jsonl"
    input_path.write_text(
        json.dumps({
            "content": None,
            "text_tagged": 17,
            "output": "Call [0901234567]<phone_number>.",
            "language": 7,
        })
        + "\n",
        encoding="utf-8",
    )

    result = validate_inline_jsonl(input_path)

    assert result.errors == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.record_id == "line-1"
    assert record.language == "7"
    assert record.plain_text == "Call 0901234567."
    assert record.spans[0].text == "0901234567"
