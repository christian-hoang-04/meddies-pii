#!/usr/bin/env python
"""Reject incomplete or under-threshold coverage reports for active source."""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TypeAlias, TypeGuard

DEFAULT_MINIMUM_STATEMENT_COVERAGE = 80.0
DEFAULT_MINIMUM_BRANCH_COVERAGE = 70.1
REPO_ROOT = Path(__file__).resolve().parents[2]

JsonObject: TypeAlias = Mapping[str, object]


def discover_source_files(source_root: Path) -> set[Path]:
    """Return every active Python file, including namespace-package children."""
    return {path.resolve() for path in source_root.rglob("*.py")}


def _is_str_mapping(value: object) -> TypeGuard[JsonObject]:
    """Return whether a value is a mapping whose keys are all strings.

    A bare `isinstance(value, Mapping)` leaves the key type unknown, so the result stays unusable
    where `Mapping[str, object]` is required. This script deliberately imports nothing from the
    package under test, so it carries its own copy rather than reaching for `json_types`.
    """
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _as_object(value: object, description: str) -> JsonObject:
    if not _is_str_mapping(value):
        msg = f"coverage report {description} must be an object"
        raise ValueError(msg)
    return value


def _as_non_negative_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"coverage report {description} must be a non-negative integer"
        raise ValueError(msg)
    return value


def _reported_files(report: JsonObject, repo_root: Path) -> dict[Path, JsonObject]:
    files = _as_object(report.get("files"), "files")
    reported: dict[Path, JsonObject] = {}
    for filename, details in files.items():
        if not isinstance(filename, str):
            msg = "coverage report file names must be strings"
            raise ValueError(msg)
        path = Path(filename)
        resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        reported[resolved] = _as_object(details, f"entry for {filename}")
    return reported


def _coverage_counts(expected_files: set[Path], reported_files: Mapping[Path, JsonObject]) -> tuple[int, int, int, int]:
    covered_lines = 0
    total_statements = 0
    covered_branches = 0
    total_branches = 0

    for path in expected_files:
        summary = _as_object(reported_files[path].get("summary"), f"summary for {path}")
        covered_lines += _as_non_negative_int(summary.get("covered_lines"), f"covered_lines for {path}")
        total_statements += _as_non_negative_int(summary.get("num_statements"), f"num_statements for {path}")
        covered_branches += _as_non_negative_int(summary.get("covered_branches"), f"covered_branches for {path}")
        total_branches += _as_non_negative_int(summary.get("num_branches"), f"num_branches for {path}")

    return covered_lines, total_statements, covered_branches, total_branches


def _below_threshold(label: str, covered: int, total: int, threshold: float) -> str | None:
    if total == 0:
        return f"{label} coverage has no reportable opportunities"

    percentage = covered / total * 100
    exact_percentage = Decimal(covered) / Decimal(total) * 100
    if exact_percentage < Decimal(str(threshold)):
        return f"{label} coverage {percentage:.1f}% ({covered}/{total}) is below {threshold:.1f}%"
    return None


def validate_coverage(
    report: JsonObject,
    *,
    source_root: Path,
    statement_threshold: float = DEFAULT_MINIMUM_STATEMENT_COVERAGE,
    branch_threshold: float = DEFAULT_MINIMUM_BRANCH_COVERAGE,
) -> list[str]:
    """Validate complete active-source inventory and independent coverage floors."""
    if statement_threshold < 0 or branch_threshold < 0:
        msg = "coverage thresholds must be non-negative"
        raise ValueError(msg)

    source_root = source_root.resolve()
    expected_files = discover_source_files(source_root)
    if not expected_files:
        msg = f"no Python source files found under {source_root}"
        raise ValueError(msg)

    errors: list[str] = []
    meta = _as_object(report.get("meta"), "meta")
    if meta.get("branch_coverage") is not True:
        return ["coverage report was not collected with branch coverage"]

    reported_files = _reported_files(report, repo_root=source_root.parents[1])
    missing_files = sorted(expected_files - reported_files.keys())
    if missing_files:
        relative_files = [path.relative_to(source_root.parents[1]).as_posix() for path in missing_files]
        noun = "file" if len(relative_files) == 1 else "files"
        errors.append(f"coverage report omitted {len(relative_files)} active source {noun}: " + ", ".join(relative_files))
        return errors

    covered_lines, total_statements, covered_branches, total_branches = _coverage_counts(expected_files, reported_files)
    statement_error = _below_threshold("statement", covered_lines, total_statements, statement_threshold)
    if statement_error:
        errors.append(statement_error)
    branch_error = _below_threshold("branch", covered_branches, total_branches, branch_threshold)
    if branch_error:
        errors.append(branch_error)
    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "src/meddies_pii")
    parser.add_argument(
        "--min-statements",
        "--statement-threshold",
        dest="statement_threshold",
        type=float,
        default=DEFAULT_MINIMUM_STATEMENT_COVERAGE,
    )
    parser.add_argument(
        "--min-branches",
        "--branch-threshold",
        dest="branch_threshold",
        type=float,
        default=DEFAULT_MINIMUM_BRANCH_COVERAGE,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = _as_object(json.loads(args.coverage_json.read_text()), "root")
        errors = validate_coverage(
            report,
            source_root=args.source_root.resolve(),
            statement_threshold=args.statement_threshold,
            branch_threshold=args.branch_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"coverage gate error: {error}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"coverage gate error: {error}", file=sys.stderr)
        return 1

    print("coverage gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
