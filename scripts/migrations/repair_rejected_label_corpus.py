#!/usr/bin/env python
"""Split API-error rejects and repair conservative Meddies Labels tag-format rejects."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.generation.label_corpus.repair import repair_rejected_artifacts
from meddies_pii.generation.label_corpus.synthetic import (
    DEFAULT_TARGETED_GENERATION_DIR,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TARGETED_GENERATION_DIR,
    )
    parser.add_argument("--model", default="mimo-v2.5-pro")
    parser.add_argument("--provider", default="mimo")
    parser.add_argument(
        "--language",
        action="append",
        choices=["Vietnamese", "English", "vi", "en"],
        help="Language to process; repeatable. Defaults to Vietnamese and English.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = args.language or ["Vietnamese", "English"]
    for language in languages:
        normalized = {"vi": "Vietnamese", "en": "English"}.get(language, language)
        summary = repair_rejected_artifacts(
            output_dir,
            normalized,
            model=args.model,
            provider=args.provider,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
