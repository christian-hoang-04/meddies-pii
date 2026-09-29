#!/usr/bin/env python
"""Reject release artifacts that ship forbidden repository files."""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

FORBIDDEN_DIRECTORIES = frozenset({"experiments"})
FORBIDDEN_FILENAMES = frozenset({"coverage.json", "coverage.xml"})


def _is_forbidden_member(member: str) -> bool:
    path = PurePosixPath(member)
    return path.name in FORBIDDEN_FILENAMES or bool(FORBIDDEN_DIRECTORIES.intersection(path.parts))


def _archive_members(artifact: Path, *, kind: str) -> tuple[str, ...]:
    try:
        if kind == "wheel":
            with zipfile.ZipFile(artifact) as archive:
                return tuple(archive.namelist())
        with tarfile.open(artifact, "r:gz") as archive:
            return tuple(archive.getnames())
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        msg = f"cannot read {kind} {artifact.name}: {error}"
        raise ValueError(msg) from error


def _artifact_errors(artifact: Path, *, kind: str) -> list[str]:
    try:
        members = _archive_members(artifact, kind=kind)
    except ValueError as error:
        return [str(error)]

    return [f"{artifact.name} contains forbidden member: {member}" for member in members if _is_forbidden_member(member)]


def validate_package_inventory(artifact_dir: Path) -> list[str]:
    """Validate one wheel and sdist that exclude forbidden repository files."""
    if not artifact_dir.exists():
        return [f"artifact directory does not exist: {artifact_dir}"]
    if not artifact_dir.is_dir():
        return [f"artifact path is not a directory: {artifact_dir}"]

    wheels = sorted(path for path in artifact_dir.glob("*.whl") if path.is_file())
    source_distributions = sorted(path for path in artifact_dir.glob("*.tar.gz") if path.is_file())
    errors: list[str] = []
    if len(wheels) != 1:
        errors.append(f"expected exactly one wheel (*.whl), found {len(wheels)}")
    if len(source_distributions) != 1:
        errors.append(f"expected exactly one source distribution (*.tar.gz), found {len(source_distributions)}")
    if errors:
        return errors

    return [
        *_artifact_errors(wheels[0], kind="wheel"),
        *_artifact_errors(source_distributions[0], kind="source distribution"),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        errors = validate_package_inventory(args.artifact_dir)
    except OSError as exception:
        print(f"package inventory gate error: {exception}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"package inventory gate error: {error}", file=sys.stderr)
        return 1

    print("package inventory gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
