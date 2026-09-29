#!/usr/bin/env python
"""Command-line wrapper over the PII350 mixed-generation aggregate assembler.

The assembly logic lives in
``meddies_pii.eval_baseline.pii350_release.aggregate``; this script only parses
arguments, validates the digests, and renders the result.
"""

from __future__ import annotations

# ruff: file-ignore[import-private-name]
# reason: `_parse_exclusion` is the private sibling of the `assemble` and `load_summary` this module already imports
# reason: from the same place, and a second exclusion parser here would drift from the one the aggregate uses. The
# reason: fix that would satisfy the rule is a public re-export inside `src/`, which this lane does not own.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.eval_baseline.pii350_release.aggregate import (
    _parse_exclusion,
    assemble,
    load_summary,
)
from meddies_pii.evaluation.identity import is_sha256

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ("assemble", "load_summary", "main")


MIN_CELL_SUMMARIES = 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Assemble a complete 17-cell PII350 aggregate from two or more cell_score_summary JSON files."),
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="PATH",
        help="cell_score_summary JSON file; repeat for each source workspace",
    )
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--trajectory-digest", required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GENERATION:CELL",
        help=("drop one cell from one source generation to resolve an intended duplicate; repeat per exclusion"),
    )
    parser.add_argument("--output", metavar="PATH", help="write the aggregate here instead of stdout")
    arguments = parser.parse_args(argv)

    if len(arguments.summary) < MIN_CELL_SUMMARIES:
        msg = "assembly requires at least two cell score summaries"
        raise SystemExit(msg)
    if not is_sha256(arguments.checkpoint_digest) or not is_sha256(arguments.trajectory_digest):
        msg = "checkpoint and trajectory digests must be lowercase SHA-256"
        raise SystemExit(msg)
    exclusions = [_parse_exclusion(value) for value in arguments.exclude]
    summaries = [load_summary(Path(path)) for path in arguments.summary]
    assembled = assemble(
        summaries,
        checkpoint_digest=arguments.checkpoint_digest,
        trajectory_digest=arguments.trajectory_digest,
        exclusions=exclusions,
    )
    rendered = json.dumps(assembled, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(rendered, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": arguments.output,
                    "assembled_digest": assembled["assembled_digest"],
                    "coverage_cells": assembled["coverage"]["coverage_cells"],
                    "rows": assembled["coverage"]["rows"],
                },
                sort_keys=True,
            ),
        )
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
