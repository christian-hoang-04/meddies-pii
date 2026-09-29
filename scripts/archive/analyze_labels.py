#!/usr/bin/env python
"""Analyze PII label distributions across all subsets of a HuggingFace repo."""

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.

import argparse
import json
import os
import pathlib
import re
from collections import defaultdict
from typing import TypedDict

from datasets import get_dataset_config_names, load_dataset
from dotenv import load_dotenv

from meddies_pii.taxonomy import PII_LABELS

load_dotenv()

TAG_PATTERN = re.compile(r"\[([^\]]*)\]\s*<([a-zA-Z][a-zA-Z0-9_-]*)>")
VALID_LABELS = PII_LABELS

CONTENT_FIELD_PRIORITY = ("text_tagged", "content", "output", "text")


class ConfigAnalysis(TypedDict, total=False):
    """Per-config analysis row.

    Only ``config`` is always present: a load failure carries ``error`` alone, an empty
    split carries ``total`` 0, and the counter fields appear only when a content field was
    found and the scan ran.
    """

    config: str
    error: str
    total: int
    avg_entities: float
    content_key: str
    label_total_count: dict[str, int]
    rows_with_label: dict[str, int]
    rows_with_multi_same: dict[str, int]
    rows_missing_label: dict[str, int]


def detect_content_field(sample: dict[str, object]) -> str | None:
    for field in CONTENT_FIELD_PRIORITY:
        if field in sample:
            return field
    return None


# reason: Dataset loading, label counting, and per-source summaries share one scan; splitting would fork counters.
def analyze_repo(repo: str, token: str | None) -> list[ConfigAnalysis]:  # ruff: ignore[complex-structure,too-many-locals]
    configs = get_dataset_config_names(repo, token=token)
    print(f"Found {len(configs)} configs: {configs}\n")

    all_results: list[ConfigAnalysis] = []

    for config in configs:
        print(f"Loading {config}...", end=" ", flush=True)
        try:
            ds = load_dataset(repo, config, token=token)
        # reason: a config that will not load raises whatever its loader raises — network, schema or auth — and every one
        # reason: means the same thing here: record this config as an error row and keep sweeping the others.
        except Exception as e:  # ruff: ignore[blind-except]
            print(f"ERROR: {e}")
            all_results.append({"config": config, "error": str(e)})
            continue

        rows = []
        for split_name in ds:
            rows.extend(dict(row) for row in ds[split_name])

        total = len(rows)
        if not rows:
            print("0 rows")
            all_results.append({"config": config, "total": 0})
            continue

        content_key = detect_content_field(rows[0])
        print(f"{total} rows (field: {content_key})")

        if not content_key:
            print(f"  Unknown columns: {list(rows[0].keys())}")
            all_results.append({"config": config, "total": total, "error": "no content field"})
            continue

        label_total_count = defaultdict(int)
        rows_with_label = defaultdict(int)
        rows_with_multi_same = defaultdict(int)
        rows_missing_label = defaultdict(int)
        total_entities_all = 0

        for row in rows:
            entities_by_label = defaultdict(list)
            text = row.get(content_key, "")
            if text:
                for value, label in TAG_PATTERN.findall(text):
                    entities_by_label[label.lower().strip()].append(value)

            row_entity_count = 0
            for lab in VALID_LABELS:
                count = len(entities_by_label.get(lab, []))
                label_total_count[lab] += count
                row_entity_count += count
                if count > 0:
                    rows_with_label[lab] += 1
                else:
                    rows_missing_label[lab] += 1
                if count > 1:
                    rows_with_multi_same[lab] += 1

            total_entities_all += row_entity_count

        avg_entities = total_entities_all / total if total > 0 else 0

        all_results.append({
            "config": config,
            "total": total,
            "avg_entities": round(avg_entities, 2),
            "content_key": content_key,
            "label_total_count": dict(label_total_count),
            "rows_with_label": dict(rows_with_label),
            "rows_with_multi_same": dict(rows_with_multi_same),
            "rows_missing_label": dict(rows_missing_label),
        })

    return all_results


# reason: print summary coordinates total rows with label totals; extra seams would leak shared intermediate state.
def print_summary(repo: str, all_results: list[ConfigAnalysis]) -> None:  # ruff: ignore[complex-structure]
    print("\n" + "=" * 150)
    print(f"LABEL DISTRIBUTION ANALYSIS — {repo}")
    print("=" * 150)

    print(f"\n{'Config':<20} {'Rows':>8} {'Avg Ent':>8} {'Field':>12}  ", end="")
    for lab in VALID_LABELS:
        print(f"{lab[:8]:>10}", end="")
    print()
    print("-" * 150)

    grand_total_rows = 0
    grand_label_totals = defaultdict(int)

    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<20} ERROR: {r['error']}")
            continue
        if r.get("total", 0) == 0:
            continue

        grand_total_rows += r["total"]
        ck = r.get("content_key", "?")
        print(
            f"{r['config']:<20} {r['total']:>8} {r['avg_entities']:>8.2f} {ck:>12}  ",
            end="",
        )
        for lab in VALID_LABELS:
            cnt = r["label_total_count"].get(lab, 0)
            grand_label_totals[lab] += cnt
            print(f"{cnt:>10,}", end="")
        print()

    print("-" * 150)
    print(f"{'TOTAL':<20} {grand_total_rows:>8} {'':>8} {'':>12}  ", end="")
    for lab in VALID_LABELS:
        print(f"{grand_label_totals[lab]:>10,}", end="")
    print()

    print(f"\n\n{'=' * 150}")
    print("ROWS WITH LABEL PRESENT (% of rows)")
    print("=" * 150)
    print(f"{'Config':<20} {'Rows':>8}  ", end="")
    for lab in VALID_LABELS:
        print(f"{lab[:8]:>10}", end="")
    print()
    print("-" * 150)

    for r in all_results:
        if "error" in r or r.get("total", 0) == 0:
            continue
        total = r["total"]
        print(f"{r['config']:<20} {total:>8}  ", end="")
        for lab in VALID_LABELS:
            cnt = r["rows_with_label"].get(lab, 0)
            pct = cnt / total * 100
            print(f"{pct:>9.1f}%", end="")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PII label distributions")
    parser.add_argument("--repo", default="Meddies/meddies-pii", help="HuggingFace repo to analyze")
    parser.add_argument("--output", help="Save raw results as JSON to this path")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    all_results = analyze_repo(args.repo, token)
    print_summary(args.repo, all_results)

    if args.output:
        with pathlib.Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
