#!/usr/bin/env python
"""Validate sampled rows from the pushed Meddies PII Hugging Face dataset."""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
import os
import random
from typing import TYPE_CHECKING, Any, Protocol, cast

from meddies_pii.tags import TAG_PATTERN
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _SampledDataset(Protocol):
    """The two members this validator needs from a loaded split.

    ``load_dataset`` returns a union wide enough that neither member resolves statically, and
    the validator only ever reads a column list and indexes single rows. Naming that surface
    states what the script depends on instead of silencing the reads one by one.
    """

    column_names: list[str]

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


DEFAULT_REPO_ID = "Meddies/meddies-pii"
LEGACY_REQUIRED_COLS = {"text", "raw", "label"}
MODERN_REQUIRED_COLS = {"text", "label", "info"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sampled rows from a pushed Meddies PII dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Rows to sample from each config.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for row sampling.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token.",
    )
    return parser.parse_args(argv)


def validate_config_rows(dataset: _SampledDataset, sample_indices: Sequence[int]) -> list[str]:
    """Dispatch on the row schema, which this repo carries in two shapes at once.

    Legacy configs carry a `raw` column and a JSON-string `label`; modern Meddies Labels configs
    (pii-bioes, eval) carry `info` and a span list. The validator must recognize each.
    """
    columns = set(dataset.column_names)
    if "raw" in columns:
        missing = LEGACY_REQUIRED_COLS - columns
        if missing:
            return [f"  FAIL: legacy config missing columns {missing}"]
        return _validate_legacy_rows(dataset, sample_indices)
    if columns >= MODERN_REQUIRED_COLS:
        return _validate_modern_rows(dataset, sample_indices)
    return [f"  FAIL: unrecognized schema, columns {sorted(columns)}"]


def _validate_legacy_rows(dataset: _SampledDataset, sample_indices: Sequence[int]) -> list[str]:
    issues: list[str] = []
    for index in sample_indices:
        row = dataset[index]

        if not row.get("text") or not row["text"].strip():
            issues.append(f"  row {index}: empty text")

        if not row.get("raw") or not row["raw"].strip():
            issues.append(f"  row {index}: empty raw")

        if row.get("raw") and TAG_PATTERN.search(row["raw"]):
            issues.append(f"  row {index}: tags leaked into raw")

        try:
            label = json.loads(row.get("label", "{}"))
            bad_keys = [key for key in label if key not in PII_LABEL_SET]
            if bad_keys:
                issues.append(f"  row {index}: invalid label keys {bad_keys}")
        except (json.JSONDecodeError, TypeError):
            issues.append(f"  row {index}: label not valid JSON")

        if row.get("text") and row.get("raw"):
            expected_raw = TAG_PATTERN.sub(r"\1", row["text"])
            if row["raw"] != expected_raw:
                issues.append(f"  row {index}: raw != strip(text)")

    return issues


def _validate_modern_rows(dataset: _SampledDataset, sample_indices: Sequence[int]) -> list[str]:
    issues: list[str] = []
    for index in sample_indices:
        row = dataset[index]
        text = row.get("text") or ""
        if not text.strip():
            issues.append(f"  row {index}: empty text")

        label = row.get("label")
        if not isinstance(label, list):
            issues.append(f"  row {index}: label is not a span list")
            continue

        for span in label:
            category = span.get("category")
            if category not in PII_LABEL_SET:
                issues.append(f"  row {index}: invalid category {category!r}")
            start, end, span_text = span.get("start"), span.get("end"), span.get("text")
            if isinstance(start, int) and isinstance(end, int) and text[start:end] != span_text:
                issues.append(f"  row {index}: span offset mismatch for {category!r}")

    return issues


def validate_pushed_dataset(repo_id: str, sample_size: int, seed: int) -> bool:
    from datasets import get_dataset_config_names, load_dataset

    # reason: the caller's `seed` selects which rows this validation spot-checks; a reported failure
    # reason: has to be reproducible from the seed alone, which a cryptographic source would prevent.
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    configs = get_dataset_config_names(repo_id)
    print(f"Total configs: {len(configs)}\n")

    all_ok = True
    for config in configs:
        dataset = load_dataset(repo_id, config, split="train")
        sample_indices = rng.sample(range(len(dataset)), min(sample_size, len(dataset)))

        print(f"=== {config} ({len(dataset)} rows, cols: {dataset.column_names}) ===")
        # reason: every config here is a row-indexed split; Dataset.__getitem__ is wider than the two
        # reason: members this validator reads, so the narrowing is stated once at the boundary.
        issues = validate_config_rows(cast("_SampledDataset", dataset), sample_indices)
        if issues:
            for issue in issues:
                print(issue)
            all_ok = False
        else:
            print("  OK")
        print()

    return all_ok


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv
    from huggingface_hub import login

    args = parse_args(argv)
    load_dotenv()
    token = os.environ.get(args.token_env)
    if not token:
        msg = f"Missing required environment variable: {args.token_env}"
        raise SystemExit(msg)

    login(token=token)
    all_ok = validate_pushed_dataset(args.repo_id, args.sample_size, args.seed)
    print("ALL CLEAN" if all_ok else "ISSUES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
