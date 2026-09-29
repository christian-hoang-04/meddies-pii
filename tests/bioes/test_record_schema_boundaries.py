from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.data.augmentation import validate_record
from meddies_pii.training.bioes.data.augmentation_artifacts import read_jsonl
from meddies_pii.training.bioes.data.augmentation_tokens import token_lengths
from meddies_pii.training.bioes.data.manifest import select_candidates_by_token_length
from meddies_pii.training.bioes.data.record_schema import record_from_object

if TYPE_CHECKING:
    from pathlib import Path


def test_record_from_object_preserves_valid_persisted_fields() -> None:
    record = record_from_object(
        {"text": "Ada", "info": {"id": "row-1"}, "label": []},
        source="fixture",
    )

    assert record == {"text": "Ada", "info": {"id": "row-1"}, "label": []}


@pytest.mark.parametrize("value", [[], "not a record", 4])
def test_record_from_object_rejects_non_object_rows(value: object) -> None:
    with pytest.raises(ValueError, match="fixture row must be an object"):
        record_from_object(value, source="fixture")


def test_jsonl_reader_rejects_non_object_line_with_its_source_location(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(["not", "a", "record"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"rows\.jsonl:1 row must be an object"):
        read_jsonl(source)


class _InvalidTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        padding: bool,
    ) -> dict[str, object]:
        del texts, add_special_tokens, truncation, padding
        return {"input_ids": [42]}


def test_token_length_boundary_rejects_non_sequence_token_rows() -> None:
    with pytest.raises(TypeError, match=r"input_ids.*invalid row 0"):
        token_lengths(
            ["Ada"],
            tokenizer=_InvalidTokenizer(),
            batch_size=1,
        )


def test_manifest_rejects_non_integer_token_length_before_sorting() -> None:
    with pytest.raises(ValueError, match="token_length must be an integer"):
        select_candidates_by_token_length(
            [
                {
                    "uid": "row-1",
                    "row_hash": "hash",
                    "token_length": "eight",
                    "accepted": True,
                },
            ],
            required=1,
        )


def test_augmentation_validation_reports_source_schema_errors() -> None:
    assert validate_record({"text": 42}, default_id="fixture") == ["parse_error:record text must be a string"]
