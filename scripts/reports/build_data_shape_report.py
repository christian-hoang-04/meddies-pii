#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from pathlib import Path

from meddies_pii.training.bioes.reports.data_shape import write_data_shape_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the bioes-v2 assembled-corpus data-shape HTML report.")
    # reason: both defaults name a file this script READS, so the rule's hazard — creating a temp
    # reason: file whose name an attacker can pre-empt — does not arise. A planted file can only
    # reason: corrupt the operator's own local render, and `--stats`/`--baseline` override it.
    parser.add_argument("--stats", default="/tmp/corpus_stats.json")  # ruff: ignore[hardcoded-temp-file]
    parser.add_argument("--baseline", default="/tmp/run1_baseline.json")  # ruff: ignore[hardcoded-temp-file]
    parser.add_argument("--sha", default="f00df6081c9a587a")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output_path = write_data_shape_report(
        stats_path=Path(args.stats),
        baseline_path=Path(args.baseline),
        output_path=Path(args.out),
        sha=args.sha,
    )
    print(f"wrote {output_path} ({output_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
