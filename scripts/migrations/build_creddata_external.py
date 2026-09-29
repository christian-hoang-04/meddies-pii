#!/usr/bin/env python
"""Build the CredData external Meddies Labels dataset (train + eval).

The input is an acquired CredData dataset directory, the output of CredData's ``download_data.py``.

Assumes acquisition is already done (the shallow per-commit clone + obfuscation).
Iterates ``meta/*.csv`` (one CSV per repo), converts each repo's files into
Meddies Labels windowed records via ``convert_creddata_file``, then splits BY REPO,
holding out ~15% of credentials for ``eval`` — stratified so ``private_url`` and
``id_number`` (the rare labels) appear in eval. Writes JSONL + HF-ready parquet +
a summary. This script NEVER publishes.

Splitting by whole repos (not rows) prevents near-duplicate files from the same
project leaking across train/eval.
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import binascii
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.historical_artifacts import legacy_jsonl_locator
from meddies_pii.training.bioes.data.augmentation import (
    text_hash,
    write_json,
    write_jsonl,
    write_parquet,
)
from meddies_pii.training.bioes.data.creddata import (
    CredMetaRow,
    convert_creddata_file,
)
from meddies_pii.training.bioes.data.mixed import summarize_records
from meddies_pii.training.bioes.data.splits import normalize_text

if TYPE_CHECKING:
    from meddies_pii.training.bioes.data.record_schema import NormalizedRecord


def _repo_url_map(creddata_dir: Path) -> dict[str, str]:
    """Map the 8-hex meta repo id (crc32 of the 64-hex key) → its GitHub URL."""
    snap_path = creddata_dir / "snapshot.json"
    if not snap_path.exists():
        return {}
    out: dict[str, str] = {}
    for key, url in json.loads(snap_path.read_text(encoding="utf-8")).items():
        try:
            out[f"{binascii.crc32(binascii.unhexlify(key)):08x}"] = str(url)
        # reason: a malformed key is skipped and the next one is read. Hoisting the `try` out of the loop would end the
        # reason: map at the first bad key instead of stepping over it.
        except (binascii.Error, ValueError):  # ruff: ignore[try-except-in-loop]
            continue
    return out


def _convert_repo(
    creddata_dir: Path,
    meta_csv: Path,
    source: str,
) -> tuple[list[NormalizedRecord], Counter[str], Counter[str]]:
    """Convert one repo's files (one meta CSV) into Meddies Labels records."""
    rows = list(csv.DictReader(meta_csv.open(encoding="utf-8")))
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_file[row["FilePath"]].append(row)

    records: list[NormalizedRecord] = []
    dropped: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for file_path, file_rows in by_file.items():
        path = creddata_dir / file_path
        if not path.exists():
            dropped["missing_file"] += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        recs, drp = convert_creddata_file(
            text,
            [CredMetaRow.from_csv(r) for r in file_rows],
            repo=source,
            file_id=Path(file_path).name,
        )
        dropped.update(drp)
        for rec in recs:
            for span in rec["label"]:
                label_counts[str(span["category"])] += 1
        records.extend(recs)
    return records, dropped, label_counts


def _select_eval_repos(
    repos: list[tuple[str, list[NormalizedRecord], Counter[str]]],
    eval_fraction: float,
) -> set[str]:
    """Hold out whole repos until ~``eval_fraction`` of credentials are in eval.

    Then ensure the rare labels (private_url, id_number) are represented.

    Fill eval from the smallest repos first (deterministic: cred-count then id) so we approach the target closely instead
    of overshooting on one large repo.

    """
    total = sum(sum(lc.values()) for _, _, lc in repos)
    target = total * eval_fraction
    chosen: set[str] = set()
    held = 0
    for repo_id, _recs, lc in sorted(repos, key=lambda r: (sum(r[2].values()), r[0])):
        if held >= target:
            break
        chosen.add(repo_id)
        held += sum(lc.values())

    for label in ("private_url", "id_number"):
        present = any(label in lc for _, _, lc in repos)
        covered = any(label in lc for rid, _, lc in repos if rid in chosen)
        if present and not covered:
            candidates = [r for r in repos if label in r[2] and r[0] not in chosen]
            smallest = min(candidates, key=lambda r: sum(r[2].values()))
            chosen.add(smallest[0])
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--creddata-dir",
        type=Path,
        required=True,
        help="Acquired CredData root (contains meta/ + the downloaded data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bioes-v2/external/creddata"),
    )
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    """No-leakage guard: drop eval records whose normalized text appears in train."""
    args = parse_args()
    url_map = _repo_url_map(args.creddata_dir)
    meta_dir = args.creddata_dir / "meta"

    dropped: Counter[str] = Counter()
    repos: list[tuple[str, list[NormalizedRecord], Counter[str]]] = []
    for meta_csv in sorted(meta_dir.glob("*.csv")):
        repo_id = meta_csv.stem
        records, repo_dropped, label_counts = _convert_repo(args.creddata_dir, meta_csv, url_map.get(repo_id, repo_id))
        dropped.update(repo_dropped)
        if records:
            repos.append((repo_id, records, label_counts))
    print(f"converted {len(repos)} repos with credentials", flush=True)

    eval_repos = _select_eval_repos(repos, args.eval_fraction)
    train_records: list[NormalizedRecord] = []
    eval_records: list[NormalizedRecord] = []
    for repo_id, records, _lc in repos:
        (eval_records if repo_id in eval_repos else train_records).extend(records)

    train_hashes = {text_hash(normalize_text(str(r.get("text") or ""))) for r in train_records}
    kept: list[NormalizedRecord] = []
    leaked = 0
    for record in eval_records:
        if text_hash(normalize_text(str(record.get("text") or ""))) in train_hashes:
            leaked += 1
            continue
        kept.append(record)
    eval_records = kept

    write_jsonl(args.output_dir / legacy_jsonl_locator("train"), train_records)
    write_jsonl(args.output_dir / legacy_jsonl_locator("eval"), eval_records)
    write_parquet(
        args.output_dir / "hf_upload/creddata/train-00000-of-00001.parquet",
        train_records,
    )
    write_parquet(
        args.output_dir / "hf_upload/creddata/eval-00000-of-00001.parquet",
        eval_records,
    )

    summary = {
        "dataset_id": "Samsung/CredData",
        "repo_target": "Meddies/meddies-pii-external",
        "config": "creddata",
        "split_policy": f"by-repo holdout, eval_fraction={args.eval_fraction}",
        "total_repos": len(repos),
        "eval_repos": len(eval_repos),
        "eval_leakage_rows_dropped": leaked,
        "splits": {
            "train": summarize_records(train_records).asdict(),
            "eval": summarize_records(eval_records).asdict(),
        },
        "dropped": dict(dropped.most_common()),
    }
    write_json(args.output_dir / "creddata.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
