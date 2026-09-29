#!/usr/bin/env python
"""Assemble the bioes-v2 corpus through the package-owned assembly seam."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path

from meddies_pii.training.bioes.assembly import (
    AssemblyGateError,
    assemble_bioes_v2_corpus,
    parse_waiver_cells,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "data/bioes-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the bioes-v2 training corpus.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--waive",
        action="append",
        default=[],
        metavar="LANG:LABEL",
        help="Recorded ADR 0008 exhaustion waiver for one language x label cell.",
    )
    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="Write the mix for inspection even when the ADR 0008 gate fails.",
    )
    args = parser.parse_args()

    try:
        result = assemble_bioes_v2_corpus(
            args.root,
            output_path=args.output,
            waivers=parse_waiver_cells(args.waive),
            enforce_gate=not args.allow_failed_gate,
        )
    except AssemblyGateError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result.to_summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
