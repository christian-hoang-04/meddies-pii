#!/usr/bin/env python
"""Merge Gemini label corrections into Gold labels for eval/test datasets.

Strategy: Gemini-first with Gold salvage.
- Gemini as base (cleaner, near-zero hallucination)
- Add back Gold entities that Gemini missed IF they pass validation
- Skip Gold entities that are subsumed by Gemini (substring match)

Gemini consistently misses IPv4/MAC — recover from raw text.

"""

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
import re
from collections import Counter
from collections.abc import Mapping

from datasets import Dataset, load_dataset

from meddies_pii.json_types import is_str_mapping
from meddies_pii.jsonl import read_jsonl
from meddies_pii.processing.label_cleanup import (
    apply_quality_filters,
    clean_bracket_artifacts,
)
from meddies_pii.taxonomy import PII_LABEL_SET


def parse_gemini_results(path: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for data in read_jsonl(path):
        key = data.get("key")
        if not isinstance(key, str):
            continue
        response = data.get("response")
        if not is_str_mapping(response):
            continue
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates or not is_str_mapping(candidates[0]):
            continue
        content = candidates[0].get("content")
        if not is_str_mapping(content):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not is_str_mapping(part) or "thought" in part:
                continue
            text = part.get("text")
            if isinstance(text, str):
                results[key] = text
                break
    return results


def clean_gemini_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


def _is_subsumed(value: str, other_values: list[str]) -> bool:
    return any(value in ov or ov in value for ov in other_values)


_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")


def _dataset_row_text(row: object, field: str) -> str:
    if not isinstance(row, Mapping):
        msg = f"dataset row must be a mapping, got {type(row).__name__}"
        raise TypeError(msg)
    value = row.get(field)
    if not isinstance(value, str):
        msg = f"dataset row field {field!r} must be a string"
        raise TypeError(msg)
    return value


def _recover_network_ids(merged: dict[str, list[str]], raw_text: str) -> dict[str, list[str]]:
    existing = set()
    for values in merged.values():
        existing.update(values)

    found = []
    for pattern in (_IPV4_RE, _MAC_RE):
        for m in pattern.finditer(raw_text):
            val = m.group()
            if val not in existing:
                found.append(val)

    if found:
        ids = merged.get("id_number", [])
        ids.extend(found)
        merged["id_number"] = ids

    return merged


def merge_labels(
    gold_json: str,
    gemini_text: str | None,
    raw_text: str,
) -> tuple[str, str]:
    """Pure Gemini output.

    Audit showed salvage adds back too much Gold noise Gemini has 0.07% value hallucination vs Gold's widespread label
    noise.

    Verify all values exist in raw text (remove Gemini hallucinations).

    Gemini has a blind spot for network identifiers — recover from raw text.

    """
    try:
        gold = json.loads(gold_json)
        if not isinstance(gold, dict):
            gold = {}
    except (json.JSONDecodeError, TypeError):
        gold = {}

    gold = clean_bracket_artifacts(gold)
    gold = apply_quality_filters(gold)
    gold = {k: v for k, v in gold.items() if k in PII_LABEL_SET}

    if gemini_text is None:
        return json.dumps(gold, ensure_ascii=False, sort_keys=True), "gold"

    text = clean_gemini_text(gemini_text)
    try:
        gemini = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return json.dumps(gold, ensure_ascii=False, sort_keys=True), "gold"

    if not isinstance(gemini, dict):
        return json.dumps(gold, ensure_ascii=False, sort_keys=True), "gold"

    gemini = clean_bracket_artifacts(gemini)
    gemini = apply_quality_filters(gemini)

    gemini = {k: v for k, v in gemini.items() if k in PII_LABEL_SET}

    merged: dict[str, list[str]] = {k: [str(item) for item in v] for k, v in gemini.items() if isinstance(v, list) and v}

    for label in list(merged):
        merged[label] = [v for v in merged[label] if v in raw_text]
        if not merged[label]:
            del merged[label]

    merged = _recover_network_ids(merged, raw_text)

    return json.dumps(merged, ensure_ascii=False, sort_keys=True), "gemini"


# reason: push is bound from the --push argparse flag at the only call site, so no unlabelled boolean reaches it.
def run_merge(
    config: str,
    predictions_path: str,
    push: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    repo_id: str = "Meddies/meddies-pii",
) -> Dataset:
    dataset = load_dataset(repo_id, config, split="train")
    gemini = parse_gemini_results(predictions_path)

    print(f"Loaded {len(dataset)} rows, {len(gemini)} Gemini results")

    stats: Counter[str] = Counter()
    new_labels: list[str] = []

    for i, row in enumerate(dataset):
        key = str(i)
        raw = _dataset_row_text(row, "raw")
        gold_label = _dataset_row_text(row, "label")

        merged_label, source = merge_labels(
            gold_label,
            gemini.get(key),
            raw,
        )
        new_labels.append(merged_label)
        stats[source] += 1

    parts = [f"{count} {source}" for source, count in sorted(stats.items())]
    print(f"Sources: {' / '.join(parts)}")

    updated: object = dataset.remove_columns(["label"])
    if not isinstance(updated, Dataset):
        msg = "datasets.remove_columns returned a non-Dataset value"
        raise TypeError(msg)
    updated = updated.add_column("label", new_labels)
    if not isinstance(updated, Dataset):
        msg = "datasets.add_column returned a non-Dataset value"
        raise TypeError(msg)
    dataset = updated

    if push:
        dataset.push_to_hub(repo_id, config_name=config)
        print(f"Pushed {config} ({len(dataset):,} rows) to {repo_id}")

    return dataset


def audit(dataset: Dataset, original_dataset: Dataset, _n: int = 30) -> None:
    from meddies_pii.tags import extract_entities

    added_total: Counter[str] = Counter()
    removed_total: Counter[str] = Counter()

    for i in range(len(dataset)):
        old_entities = extract_entities(original_dataset[i]["label"])
        new_entities = extract_entities(dataset[i]["label"])

        added = new_entities - old_entities
        removed = old_entities - new_entities

        for _, label in added:
            added_total[label] += 1
        for _, label in removed:
            removed_total[label] += 1

    all_labels = sorted(set(added_total) | set(removed_total))
    print(f"\n{'Label':<20} {'Added':>8} {'Removed':>8} {'Net':>8}")
    print(f"{'-' * 44}")
    ta = tr = 0
    for label in all_labels:
        a = added_total[label]
        r = removed_total[label]
        ta += a
        tr += r
        print(f"{label:<20} {a:>+8} {r:>8} {a - r:>+8}")
    print(f"{'-' * 44}")
    print(f"{'TOTAL':<20} {ta:>+8} {tr:>8} {ta - tr:>+8}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=["test", "eval"])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default="Meddies/meddies-pii")
    args = parser.parse_args()

    run_merge(args.config, args.predictions, args.push, args.repo_id)


if __name__ == "__main__":
    main()
