#!/usr/bin/env python
"""Render a self-contained HTML audit of rejected synthetic documents."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

from meddies_pii.training.bioes.reports.rejection_audit import (
    load_rejection_records,
    write_rejection_audit_report,
)

if TYPE_CHECKING:
    from collections.abc import Sized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_rejection_records()
    output_path = write_rejection_audit_report(records, Path(args.out))
    # reason: the fallback makes the value sized on the empty path, which is all len() needs here;
    # reason: the record itself is untyped JSON, so Sized states the assumption without overclaiming a list.
    reasons = sum(len(cast("Sized", record.get("errors") or ["(none)"])) for record in records)
    print(f"wrote {output_path} — {len(records)} docs, {reasons} rejection-reasons")


if __name__ == "__main__":
    main()
