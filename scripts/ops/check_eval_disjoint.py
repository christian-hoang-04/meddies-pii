#!/usr/bin/env python
"""Assert the gold eval is text-disjoint from the training pool (brief leakage rule).

No eval document may appear in training. The "identical text" key matches the
generation pipeline's own dedup (synthetic.py `_existing_text_hashes`):
sha256 of the exact `text` field, no normalization — so this check agrees with
what the pipeline already considers a duplicate. Streams the training files so
the 5 GB mix never loads into memory at once. Exit 1 on any overlap (gate).
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.json_types import is_str_mapping
from meddies_pii.jsonl import read_jsonl

if TYPE_CHECKING:
    from collections.abc import Sequence

COLLISION_PREVIEW_LIMIT = 20

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_DIR = REPO / "data/run2a/eval_gold"
DEFAULT_TRAIN_FILES = (REPO / f"data/bioes-v2/mix/train.{LEGACY_ARTIFACT_TOKEN}.jsonl",)
PROGRESS_EVERY = 250_000


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_gold_hashes(gold_dir: Path) -> dict[str, dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    for path in sorted(gold_dir.glob("accepted.*.jsonl")):
        for record in read_jsonl(path):
            text = record.get("text")
            if isinstance(text, str):
                info = record.get("info")
                info = info if is_str_mapping(info) else {}
                by_hash[text_hash(text)] = {
                    "lang": info.get("language"),
                    "id": info.get("id"),
                    "preview": text[:90].replace("\n", " "),
                }
    return by_hash


def scan_training(train_files: Sequence[Path], gold: dict[str, dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    scanned = 0
    collisions: list[dict[str, Any]] = []
    for path in train_files:
        if not path.exists():
            print(f"  WARN: training file missing, skipped: {path}")
            continue
        print(f"  scanning {path.name} ...")
        for record in read_jsonl(path):
            text = record.get("text")
            if not isinstance(text, str):
                continue
            scanned += 1
            if scanned % PROGRESS_EVERY == 0:
                print(f"    {scanned:,} train rows scanned, {len(collisions)} hits")
            hit = gold.get(text_hash(text))
            if hit is not None:
                info = record.get("info")
                info = info if is_str_mapping(info) else {}
                collisions.append({**hit, "train_id": info.get("id")})
    return scanned, collisions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument(
        "--train-files",
        type=Path,
        nargs="+",
        default=list(DEFAULT_TRAIN_FILES),
        help="JSONL training files to scan (default: the built train mix).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gold = load_gold_hashes(args.gold_dir)
    if not gold:
        msg = f"No gold rows under {args.gold_dir}"
        raise SystemExit(msg)
    print(f"Gold: {len(gold)} unique-text docs from {args.gold_dir}")

    scanned, collisions = scan_training(args.train_files, gold)
    print(f"\nScanned {scanned:,} training rows across {len(args.train_files)} file(s).")

    if collisions:
        print(f"\n❌ LEAKAGE — {len(collisions)} training rows match a gold doc:")
        for hit in collisions[:20]:
            print(f"  [{hit['lang']}] gold={hit['id']} train={hit['train_id']} :: {hit['preview']}")
        if len(collisions) > COLLISION_PREVIEW_LIMIT:
            print(f"  ... and {len(collisions) - 20} more")
        return 1

    print("\n✓ DISJOINT — 0 gold documents appear in the training pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
