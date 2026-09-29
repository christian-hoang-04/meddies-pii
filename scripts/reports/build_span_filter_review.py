#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from pathlib import Path

from meddies_pii.training.bioes.reports.span_filter_review import (
    collect_span_filter_review,
    write_span_filter_review_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a side-by-side span-quality removed-vs-kept HTML review.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-base", type=int, default=80_000)
    args = parser.parse_args()

    review = collect_span_filter_review(max_base=args.max_base)
    output_path = write_span_filter_review_report(review, Path(args.out))
    print(
        f"wrote {output_path} ({output_path.stat().st_size // 1024} KB) — "
        f"{review.total_removed:,}/{review.total:,} removed",
    )


if __name__ == "__main__":
    main()
