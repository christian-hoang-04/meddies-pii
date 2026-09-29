from __future__ import annotations

import json
from typing import TYPE_CHECKING

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data.augmentation import load_augmentation_rows
from meddies_pii.training.bioes.data.mixed_sources import load_local_external_rows

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class _CharacterizationTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        padding: bool,
    ) -> dict[str, list[list[int]]]:
        del add_special_tokens, truncation, padding
        return {"input_ids": [list(range(len(text))) for text in texts]}


def _info_id(row: Mapping[str, object]) -> object:
    """Return a loaded row's ``info.id``, narrowing the nested block where it is read.

    `load_local_external_rows` returns `list[dict[str, object]]`, so the `info` block
    arrives as `object` and cannot be subscripted until the shape is stated.

    Returns:
        The ``id`` field of the row's ``info`` block.

    """
    info = row["info"]
    assert is_str_mapping(info), f"info must be a mapping, got {type(info).__name__}"
    return info["id"]


def _record(record_id: str, text: str) -> dict[str, object]:
    return {
        "text": text,
        "label": [
            {
                "category": "human_name",
                "start": 0,
                "end": 3,
                "text": text[:3],
            },
        ],
        "info": {"id": record_id, "source": "fixture"},
    }


def test_augmentation_loader_preserves_file_and_row_order_after_rejections(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                _record("kept-first", "Ada"),
                _record("duplicate-id", "Bob"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    malformed = _record("bad-offset", "Cy")
    malformed["label"] = [{"category": "human_name", "start": 0, "end": 3, "text": "Cy"}]
    second.write_text(
        "\n".join(json.dumps(record) for record in (malformed, _record("kept-second", "Dee"))) + "\n",
        encoding="utf-8",
    )

    rows, summary = load_augmentation_rows(
        [first, second],
        tokenizer=_CharacterizationTokenizer(),
        max_length=10,
        token_batch_size=2,
        existing_ids={"duplicate-id"},
        existing_text_hashes=set(),
    )

    assert [row["info"]["id"] for row in rows] == ["kept-first", "kept-second"]
    assert summary["invalid_row_count"] == 1
    assert summary["duplicate_id_examples"] == ["duplicate-id"]


def test_local_external_rows_use_sorted_paths_and_report_unknown_stems(
    tmp_path: Path,
) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "gretel.jsonl").write_text(
        json.dumps({
            "uid": "gretel",
            "text": "Email gretel@example.com.",
            "entities": [{"entity": "gretel@example.com", "types": ["email"]}],
        })
        + "\n",
        encoding="utf-8",
    )
    (external_dir / "unknown.jsonl").write_text("{}\n", encoding="utf-8")

    rows, dropped = load_local_external_rows(external_dir)

    assert [_info_id(row) for row in rows] == ["gretelai/gretel-pii-masking-en-v1:gretel"]
    assert dropped == {"unknown_source:unknown": 1}
