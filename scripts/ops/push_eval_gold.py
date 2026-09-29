#!/usr/bin/env python
"""Assemble the gold eval set and push it as a Meddies/meddies-pii config.

Reads the per-language `accepted.*.jsonl` gold (gpt-5.4-mini, heuristic-gated)
and pushes the concatenation as one config/split, overwriting the legacy broken
`eval` subset. Dry-run by default: prints label/language/source distributions
for review, then `--push` publishes. Reusable for any per-language gold dir via
the flags.
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import collections
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.jsonl import read_jsonl
from meddies_pii.publishing.huggingface import push_config
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO / "data/run2a/eval_gold"
DEFAULT_REPO_ID = "Meddies/meddies-pii"
DEFAULT_CONFIG = "eval"
DEFAULT_SPLIT = "train"


def load_rows(source_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("accepted.*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def print_preview(rows: list[dict[str, Any]]) -> int:
    per_lang: collections.Counter[str] = collections.Counter()
    per_source: collections.Counter[str] = collections.Counter()
    span_per_label: collections.Counter[str] = collections.Counter()
    docs_per_label: collections.Counter[str] = collections.Counter()
    contaminated: collections.Counter[str] = collections.Counter()
    key_shapes: set[tuple[str, ...]] = set()

    for row in rows:
        info = row.get("info", {})
        per_lang[info.get("language", "?")] += 1
        per_source[info.get("source", "?")] += 1
        key_shapes.add(tuple(sorted(row.keys())))
        seen: set[str] = set()
        for span in row.get("label", []):
            category = span.get("category", "?")
            span_per_label[category] += 1
            seen.add(category)
            if category not in PII_LABEL_SET:
                contaminated[category] += 1
        for category in seen:
            docs_per_label[category] += 1

    print(f"\nTOTAL DOCS: {len(rows)}")
    print(f"ROW SCHEMA(S): {sorted(key_shapes)}")
    print(f"SOURCE: {dict(per_source)}")

    print(f"\nPER-LANGUAGE ({len(per_lang)} langs):")
    for lang, count in sorted(per_lang.items()):
        print(f"  {lang:<14} {count}")

    print("\nPER-LABEL (docs containing it / total spans):")
    for label in sorted(PII_LABEL_SET):
        print(f"  {label:<16} {docs_per_label[label]:>5} docs / {span_per_label[label]:>6} spans")

    if contaminated:
        print(f"\n  ⚠️  CONTAMINATION — categories outside the 9 labels: {dict(contaminated)}")
        return 1
    print("\n  ✓ 0 contamination (every span is one of the 9 PII labels)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--push",
        action="store_true",
        help="Publish to HF (overwrites the config/split). Omit for dry-run preview.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.source_dir)
    if not rows:
        msg = f"No accepted.*.jsonl rows under {args.source_dir}"
        raise SystemExit(msg)

    contamination = print_preview(rows)
    target = f"{args.repo_id}/{args.config_name} (split={args.split})"

    if not args.push:
        print(f"\nDRY RUN — would push {len(rows)} rows -> {target}")
        print("Re-run with --push to publish.")
        return 0

    if contamination:
        msg = "Refusing to push: contamination detected above."
        raise SystemExit(msg)

    count = push_config(args.repo_id, args.config_name, rows, split=args.split)
    print(f"\nPUSHED {count} rows -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
