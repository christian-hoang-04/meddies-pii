from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
import sys
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.data.augmentation import (
    load_augmentation_rows,
    load_current_splits,
    validate_splits,
)

if TYPE_CHECKING:
    from pathlib import Path


class LengthTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, object]:
        return {"input_ids": [[0] * len(text) for text in texts]}


def _row(row_id: str, text: str) -> dict[str, object]:
    return {
        "text": text,
        "label": [{"category": "human_name", "start": 0, "end": 5, "text": text[:5]}],
        "info": {"id": row_id, "language": "en", "source_dataset": "fixture"},
    }


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_augmentation_filters_invalid_long_duplicate_id_and_duplicate_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rows.jsonl"
    _write_jsonl(
        source,
        [
            _row("keep", "Alice"),
            _row("existing-id", "Bruno"),
            _row("same-text", "Alice"),
            _row("long", "Charlotte"),
            {"text": "bad", "label": []},
        ],
    )

    kept, summary = load_augmentation_rows(
        [source],
        tokenizer=LengthTokenizer(),
        max_length=6,
        token_batch_size=2,
        existing_ids={"existing-id"},
        existing_text_hashes=set(),
    )

    assert [row["info"]["id"] for row in kept] == ["keep"]
    assert summary["invalid_row_count"] == 1
    assert summary["dropped_over_max_length"] == 1
    assert summary["duplicate_id_dropped"] == 1
    assert summary["duplicate_text_dropped"] == 1


def test_load_current_splits_requires_named_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meddies_pii.training.bioes.data import augmentation

    monkeypatch.setattr(augmentation, "load_dataset", lambda *_: [])
    with pytest.raises(TypeError, match="named splits"):
        load_current_splits("fixture", "train")

    monkeypatch.setattr(
        augmentation,
        "load_dataset",
        lambda *_: {
            "train": [_row("train", "Alice")],
            "validation": [_row("val", "Bruno")],
        },
    )
    train, validation = load_current_splits("fixture", "train")
    assert [row["info"]["id"] for row in train] == ["train"]
    assert [row["info"]["id"] for row in validation] == ["val"]


def test_validate_splits_reports_overlap_invalid_rows_and_length() -> None:
    report = validate_splits(
        train_rows=[_row("same", "Alice"), _row("same", "Alice")],
        validation_rows=[
            _row("same", "Charlotte"),
            {"text": "plain", "label": [], "info": {"id": "empty"}},
        ],
        tokenizer=LengthTokenizer(),
        max_length=6,
        token_batch_size=4,
    )

    assert report["train_validation_id_overlap"] == 1
    assert report["duplicate_train_ids"] == 1
    assert report["duplicate_train_texts"] == 1
    assert report["empty_label_rows"] == 1
    assert report["validation_rows_over_max_length"] == 1


def test_augmentation_main_writes_merged_artifacts_from_local_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    from meddies_pii.training.bioes.data import augmentation

    augmentation_path = tmp_path / "augmentation.jsonl"
    _write_jsonl(augmentation_path, [_row("new", "Clara")])
    output_dir = tmp_path / "out"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> LengthTokenizer:
            return LengthTokenizer()

    monkeypatch.setattr(
        augmentation,
        "load_dataset",
        lambda *_: {
            "train": [_row("base", "Alice")],
            "validation": [_row("val", "Bruno")],
        },
    )
    monkeypatch.setattr(augmentation, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(augmentation, "write_parquet", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "augmentation",
            "--augmentation-file",
            str(augmentation_path),
            "--output-dir",
            str(output_dir),
            "--max-length",
            "10",
        ],
    )

    augmentation.main()

    rendered = json.loads(capsys.readouterr().out)
    assert rendered["total_rows"] == 3
    assert (output_dir / "pii-bioes.summary.json").exists()
    assert (output_dir / "hf_upload/reports/pii-bioes.summary.json").exists()
