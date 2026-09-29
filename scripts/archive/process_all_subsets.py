from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this archived standalone command is loaded by file path; a package marker would imply a supported API.
# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt remains explicit for this archived command; generated return sections would add no API value.
# ruff: file-ignore[import-outside-top-level]
# reason: this archived command loads its optional datasets client only while processing and its
# reason: dotenv helper only in the CLI entrypoint; importing analysis helpers must not require either.
# ruff: file-ignore[print]
# reason: this archived command's progress, validation summary, and push status are its operator-facing output.
import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from meddies_pii.json_types import is_str_mapping
from meddies_pii.tags import process_row
from meddies_pii.taxonomy import PII_LABELS_BRACKETED

if TYPE_CHECKING:
    from collections.abc import Sequence

ALL_CONFIGS = [
    "burmese",
    "chinese",
    "english",
    "filipino",
    "french",
    "german",
    "indonesian",
    "japanese",
    "korean",
    "laos",
    "malay",
    "portuguese",
    "russian",
    "spanish",
    "tamil",
    "thai",
    "vietnamese",
    "nvidia-health",
    "nvidia-non-health",
]
MAX_INVALID_EXAMPLES_PER_LABEL = 3


class ProcessSuccess(TypedDict):
    config: str
    rows: int
    valid_pii: int
    invalid_labels: int
    invalid_examples: dict[str, list[str]]


class ProcessFailure(TypedDict):
    config: str
    error: str


ProcessResult = ProcessSuccess | ProcessFailure


def count_invalid_labels(results: list[dict[str, object]]) -> tuple[int, int, dict[str, list[str]]]:
    """Count valid and invalid PII labels in processed results."""
    valid_count = 0
    invalid_count = 0
    invalid_examples: dict[str, list[str]] = {}

    for row in results:
        text = row.get("text", "")
        if not isinstance(text, str) or not text:
            continue
        try:
            pii: object = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not is_str_mapping(pii):
            continue
        for label, entities in pii.items():
            if not isinstance(entities, list):
                continue
            full_label = f"<{label}>"
            if full_label in PII_LABELS_BRACKETED:
                valid_count += len(entities)
            else:
                invalid_count += len(entities)
                if full_label not in invalid_examples:
                    invalid_examples[full_label] = []
                if len(invalid_examples[full_label]) < MAX_INVALID_EXAMPLES_PER_LABEL:
                    invalid_examples[full_label].extend(
                        [f"[{e}]<{label}>" for e in entities[:MAX_INVALID_EXAMPLES_PER_LABEL]],
                    )

    return valid_count, invalid_count, invalid_examples


def process_config(
    config: str,
    token: str | None,
    repo_id: str,
    output_dir: Path,
    *,
    push: bool = False,
) -> ProcessSuccess:
    """Process a single config/subset."""
    from datasets import Dataset, load_dataset

    print(f"\n{'=' * 60}")
    print(f"Processing: {config}")
    print(f"{'=' * 60}")

    ds = load_dataset("Meddies/meddies-pii", config, token=token)
    all_rows: list[dict[str, object]] = []
    for split in ds:
        all_rows.extend(dict(row) for row in ds[split] if is_str_mapping(row))

    print(f"  Loaded {len(all_rows):,} rows")

    results = [process_row(row) for row in all_rows]

    valid_count, invalid_count, invalid_examples = count_invalid_labels(results)
    print(f"  Valid PII tags: {valid_count:,}")
    print(f"  Invalid labels: {invalid_count}")

    if invalid_examples:
        print("  Invalid label examples:")
        for label, examples in list(invalid_examples.items())[:5]:
            print(f"    {label}: {examples[:2]}")

    output_file = output_dir / f"{config}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    print(f"  Saved to: {output_file}")

    if push and invalid_count == 0:
        print(f"  Pushing to {repo_id} (config: {config})...")
        Dataset.from_list(results).push_to_hub(repo_id, config_name=config, token=token)
        print("  Pushed successfully!")
    elif push and invalid_count > 0:
        print(f"  SKIPPING push due to {invalid_count} invalid labels!")

    return {
        "config": config,
        "rows": len(results),
        "valid_pii": valid_count,
        "invalid_labels": invalid_count,
        "invalid_examples": invalid_examples,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--configs",
        nargs="+",
        default=ALL_CONFIGS,
        help="Configs to process (default: all)",
    )
    parser.add_argument("-o", "--output_dir", default="data/subsets")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo_id", default="Meddies/meddies-pii-cleaned")
    parser.add_argument("--skip", nargs="+", default=[], help="Configs to skip")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()

    token = os.getenv("HF_TOKEN")
    output_dir = Path(args.output_dir)

    configs_to_process = [c for c in args.configs if c not in args.skip]

    print(f"Processing {len(configs_to_process)} configs: {configs_to_process}")
    print(f"Push to hub: {args.push}")
    print(f"Output dir: {output_dir}")

    results: list[ProcessResult] = []
    for config in configs_to_process:
        # reason: each config is an independent batch job; one failure is recorded and must not stop later configs.
        try:
            result = process_config(config, token, args.repo_id, output_dir, push=args.push)
            results.append(result)
        # reason: this batch boundary records one config's failure and continues with the remaining independent configs.
        except Exception as e:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"ERROR processing {config}: {e}")
            results.append({"config": config, "error": str(e)})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_rows = 0
    total_valid = 0
    total_invalid = 0
    for r in results:
        if "error" in r:
            print(f"  {r['config']}: ERROR - {r['error']}")
        else:
            status = "OK" if r["invalid_labels"] == 0 else f"INVALID ({r['invalid_labels']})"
            print(f"  {r['config']}: {r['rows']:,} rows, {r['valid_pii']:,} valid PII - {status}")
            total_rows += r["rows"]
            total_valid += r["valid_pii"]
            total_invalid += r["invalid_labels"]

    print(f"\nTotal: {total_rows:,} rows, {total_valid:,} valid PII, {total_invalid} invalid labels")
    failures = [r for r in results if "error" in r]
    if failures and not args.allow_partial:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
