#!/usr/bin/env python
"""Build the gretel external Meddies Labels dataset (train + eval) locally.

Converts ``gretelai/gretel-pii-masking-en-v1`` into the Meddies Labels span schema via
the tested ``convert_gretel_row`` adapter, then assembles the two splits agreed
for ``Meddies/meddies-pii-external`` / config ``gretel``:

* ``train`` = gretel ``train`` + ``validation`` (merged)
* ``eval``  = gretel ``test``

Writes local JSONL + HF-ready parquet + a summary under ``--output-dir``. This
script NEVER publishes — pushing to the Hub is a separate, explicit step.
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset

from meddies_pii.historical_artifacts import legacy_jsonl_locator
from meddies_pii.training.bioes.data.augmentation import (
    text_hash,
    write_json,
    write_jsonl,
    write_parquet,
)
from meddies_pii.training.bioes.data.mixed import convert_gretel_row, summarize_records
from meddies_pii.training.bioes.data.splits import normalize_text

if TYPE_CHECKING:
    from meddies_pii.training.bioes.data.record_schema import NormalizedRecord

DATASET_ID = "gretelai/gretel-pii-masking-en-v1"


def convert_split(split: str, *, max_scan: int | None) -> tuple[list[NormalizedRecord], Counter[str]]:
    """Convert every row of one gretel split into Meddies Labels span records."""
    records: list[NormalizedRecord] = []
    dropped: Counter[str] = Counter()
    rows = load_dataset(DATASET_ID, split=split, streaming=True)
    for index, row in enumerate(rows):
        if max_scan is not None and index >= max_scan:
            break
        record, row_dropped = convert_gretel_row(row, default_uid=f"{split}:{index}")
        dropped.update(row_dropped)
        if record is not None:
            records.append(record)
    return records, dropped


def drop_eval_leakage(
    train_records: list[NormalizedRecord],
    eval_records: list[NormalizedRecord],
) -> tuple[list[NormalizedRecord], int]:
    """Drop eval rows whose normalized text also appears in train (no leakage)."""
    train_hashes = {text_hash(normalize_text(str(record.get("text") or ""))) for record in train_records}
    kept: list[NormalizedRecord] = []
    leaked = 0
    for record in eval_records:
        digest = text_hash(normalize_text(str(record.get("text") or "")))
        if digest in train_hashes:
            leaked += 1
            continue
        kept.append(record)
    return kept, leaked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bioes-v2/external/gretel"),
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        help="Limit rows per source split (smoke test). Default: all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_a, drop_train = convert_split("train", max_scan=args.max_scan)
    train_b, drop_validation = convert_split("validation", max_scan=args.max_scan)
    train_records = [*train_a, *train_b]
    eval_records, drop_test = convert_split("test", max_scan=args.max_scan)
    eval_records, leaked = drop_eval_leakage(train_records, eval_records)

    write_jsonl(args.output_dir / legacy_jsonl_locator("train"), train_records)
    write_jsonl(args.output_dir / legacy_jsonl_locator("eval"), eval_records)
    write_parquet(
        args.output_dir / "hf_upload/gretel/train-00000-of-00001.parquet",
        train_records,
    )
    write_parquet(
        args.output_dir / "hf_upload/gretel/eval-00000-of-00001.parquet",
        eval_records,
    )

    dropped = drop_train + drop_validation + drop_test
    summary = {
        "dataset_id": DATASET_ID,
        "repo_target": "Meddies/meddies-pii-external",
        "config": "gretel",
        "train_source": "gretel train + validation (merged)",
        "eval_source": "gretel test",
        "eval_leakage_rows_dropped": leaked,
        "splits": {
            "train": summarize_records(train_records).asdict(),
            "eval": summarize_records(eval_records).asdict(),
        },
        "dropped_external_labels": dict(dropped.most_common()),
    }
    write_json(args.output_dir / "gretel.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
