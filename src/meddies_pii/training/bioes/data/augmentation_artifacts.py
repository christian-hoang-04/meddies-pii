"""Stable artifact serialization for BIOES augmentation builds."""

from __future__ import annotations

# ruff: file-ignore[useless-import-alias]
# reason: an explicit `X as X` re-export, which is this module's published surface: ruff's own
# reason: unsafe fix DELETED five of these once and broke every caller, so the alias stays.
import hashlib
import json
from typing import TYPE_CHECKING

from datasets import Dataset, Features, Value
from datasets.features import List

from meddies_pii.training.bioes.data.record_io import write_jsonl as write_jsonl
from meddies_pii.training.bioes.data.record_schema import Record, record_from_object

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

BIOES_FEATURES = Features({
    "info": {
        "domain_bucket": Value("string"),
        "domain_profile": Value("string"),
        "document_type": Value("string"),
        "generation_model": Value("string"),
        "id": Value("string"),
        "language": Value("string"),
        "language_bucket": Value("string"),
        "label_policy": Value("string"),
        "original_id": Value("string"),
        "scenario": Value("string"),
        "source": Value("string"),
        "source_dataset": Value("string"),
        "split_purpose": Value("string"),
        "text_format": Value("string"),
    },
    "label": List({
        "category": Value("string"),
        "end": Value("int64"),
        "start": Value("int64"),
        "text": Value("string"),
    }),
    "text": Value("string"),
})


def read_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as handle:
        return [
            record_from_object(json.loads(line), source=f"{path}:{line_number}")
            for line_number, line in enumerate(handle, start=1)
            if line.strip()
        ]


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([dict(row) for row in rows], features=BIOES_FEATURES).to_parquet(path)
