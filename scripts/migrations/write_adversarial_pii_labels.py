#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.training.bioes.data.adversarial_synthetic import (
    build_adversarial_pii_label_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write deterministic Meddies Labels adversarial examples as JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"data/processed/{LEGACY_ARTIFACT_TOKEN}.adversarial.jsonl"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in build_adversarial_pii_label_examples():
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
