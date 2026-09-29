from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.json_types import is_str_mapping
from meddies_pii.training.bioes.data.mixed_sources import (
    EXTERNAL_DATASETS,
    dataset_id_for_stem,
    load_local_external_rows,
    load_remote_external_rows,
    target_external_count,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _ai_row(uid: str = "ai") -> dict[str, object]:
    return {
        "uid": uid,
        "language": "en",
        "source_text": "Alice",
        "privacy_mask": [{"label": "GIVENNAME", "value": "Alice", "start": 0, "end": 5}],
    }


def test_local_source_stems_dispatch_converters_and_report_unknown_files(
    tmp_path: Path,
) -> None:
    _write_jsonl(tmp_path / "ai4privacy-1m-small.jsonl", [_ai_row()])
    _write_jsonl(
        tmp_path / "nemotron.jsonl",
        [
            {
                "uid": "n",
                "locale": "en",
                "text": "Bob",
                "spans": [{"label": "first_name", "text": "Bob", "start": 0, "end": 3}],
            },
        ],
    )
    _write_jsonl(tmp_path / "unmapped.jsonl", [_ai_row("ignored")])

    rows, dropped = load_local_external_rows(tmp_path)

    assert [_info_id(row) for row in rows] == [
        "ai4privacy/pii-masking-openpii-1m:ai",
        "nvidia/Nemotron-PII:n",
    ]
    assert dropped == {"unknown_source:unmapped": 1}
    assert dataset_id_for_stem("gretel-2026") == "gretelai/gretel-pii-masking-en-v1"
    assert dataset_id_for_stem("missing") is None


def test_remote_loading_honors_quotas_languages_scan_boundaries_and_shortfalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meddies_pii.training.bioes.data import mixed_sources

    rows_by_dataset: dict[str, list[dict[str, object]]] = {
        "nvidia/Nemotron-PII": [
            {
                "uid": "unsupported",
                "locale": "zz",
                "text": "No",
                "spans": [{"label": "first_name", "text": "No", "start": 0, "end": 2}],
            },
            {
                "uid": "n",
                "locale": "en",
                "text": "Bob",
                "spans": [{"label": "first_name", "text": "Bob", "start": 0, "end": 3}],
            },
        ],
        "ai4privacy/pii-masking-openpii-1m": [_ai_row()],
        "ai4privacy/open-pii-masking-500k-ai4privacy": [_ai_row("ai500")],
        "gretelai/gretel-pii-masking-en-v1": [
            {
                "uid": "g",
                "language": "en",
                "text": "cara@example.com",
                "entities": [{"entity": "cara@example.com", "types": ["email_address"]}],
            },
        ],
    }

    def fake_load_dataset(dataset_id: str, **_: object) -> list[dict[str, object]]:
        return rows_by_dataset[dataset_id]

    monkeypatch.setattr(mixed_sources, "load_dataset", fake_load_dataset)
    rows, dropped = load_remote_external_rows(
        target_external=4,
        split="train",
        max_scan_per_dataset=2,
        supported_languages_only=True,
    )

    assert len(rows) == len(EXTERNAL_DATASETS)
    assert dropped["nvidia/Nemotron-PII:unsupported_language:zz"] == 1


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1])
def test_target_external_count_rejects_invalid_ratios(ratio: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        target_external_count(10, ratio)


def test_target_external_count_rounds_to_requested_mix() -> None:
    assert target_external_count(6, 0.6) == 4
