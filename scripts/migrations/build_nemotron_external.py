#!/usr/bin/env python
"""Build the Nemotron external Meddies Labels dataset (train + eval) locally.

Converts ``nvidia/Nemotron-PII`` into the Meddies Labels span schema via the tested
``convert_nemotron_row`` adapter, then assembles the two splits agreed for
``Meddies/meddies-pii-external`` / config ``nemotron``:

* ``train`` = nemotron ``train``
* ``eval``  = nemotron ``test``

Nemotron is English-only, so there is no language gate; ``info.source`` carries
each row's domain (one of ~30) for later filtering. Robustness mirrors the
ai4privacy builder: NON-streaming download (cached, resilient), train written
before touching test, progress logging. This script NEVER publishes.
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
from meddies_pii.training.bioes.data.mixed import (
    convert_nemotron_row,
    summarize_records,
)
from meddies_pii.training.bioes.data.splits import normalize_text

if TYPE_CHECKING:
    from collections.abc import Mapping

    from meddies_pii.training.bioes.data.record_schema import NormalizedRecord

DATASET_ID = "nvidia/Nemotron-PII"


def _row_hash(record: Mapping[str, object]) -> str:
    return text_hash(normalize_text(str(record.get("text") or "")))


def convert_split(
    split: str,
    *,
    max_scan: int | None,
    log_every: int = 50_000,
) -> tuple[list[NormalizedRecord], Counter[str]]:
    """Convert every row of one Nemotron split into Meddies Labels span records."""
    records: list[NormalizedRecord] = []
    dropped: Counter[str] = Counter()
    rows = load_dataset(DATASET_ID, split=split)
    total = len(rows)
    print(f"[{split}] loaded {total} rows, converting...", flush=True)
    for index, row in enumerate(rows):
        if max_scan is not None and index >= max_scan:
            break
        record, row_dropped = convert_nemotron_row(row, dataset_id=DATASET_ID, default_uid=f"{split}:{index}")
        dropped.update(row_dropped)
        if record is not None:
            records.append(record)
        if (index + 1) % log_every == 0:
            print(f"[{split}] {index + 1}/{total} | kept {len(records)}", flush=True)
    print(f"[{split}] done: {len(records)} records kept", flush=True)
    return records, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bioes-v2/external/nemotron"),
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        help="Limit rows per source split (smoke test). Default: all rows.",
    )
    return parser.parse_args()


def main() -> None:
    """--- train: convert + persist immediately (survives a later test-split failure).

    --- eval (test split): convert, drop leakage vs train, persist.

    """
    args = parse_args()

    train_records, drop_train = convert_split("train", max_scan=args.max_scan)
    write_jsonl(args.output_dir / legacy_jsonl_locator("train"), train_records)
    write_parquet(
        args.output_dir / "hf_upload/nemotron/train-00000-of-00001.parquet",
        train_records,
    )
    train_summary = summarize_records(train_records).asdict()
    train_hashes = {_row_hash(r) for r in train_records}
    del train_records
    print(f"[train] written ({len(train_hashes)} hashes for leakage guard)", flush=True)

    eval_records, drop_test = convert_split("test", max_scan=args.max_scan)
    kept: list[NormalizedRecord] = []
    leaked = 0
    for record in eval_records:
        if _row_hash(record) in train_hashes:
            leaked += 1
            continue
        kept.append(record)
    eval_records = kept
    write_jsonl(args.output_dir / legacy_jsonl_locator("eval"), eval_records)
    write_parquet(
        args.output_dir / "hf_upload/nemotron/eval-00000-of-00001.parquet",
        eval_records,
    )

    dropped = drop_train + drop_test
    summary = {
        "dataset_id": DATASET_ID,
        "repo_target": "Meddies/meddies-pii-external",
        "config": "nemotron",
        "train_source": "nemotron train",
        "eval_source": "nemotron test",
        "language_gate": "none (English-only source)",
        "eval_leakage_rows_dropped": leaked,
        "splits": {
            "train": train_summary,
            "eval": summarize_records(eval_records).asdict(),
        },
        "dropped_external_labels": dict(dropped.most_common()),
    }
    write_json(args.output_dir / "nemotron.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
