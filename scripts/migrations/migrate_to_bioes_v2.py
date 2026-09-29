#!/usr/bin/env python
"""Migrate scattered Meddies-PII training data into the canonical `data/bioes-v2/` home.

Implements the one-time migration of ADR 0007 decision 5. The plan is data-driven:
it reads `data/bioes-v2/MANIFEST.json` and acts only on entries whose `action` is a
move. Dry-run by default — it prints the planned filesystem operations and exits 0
WITHOUT touching anything. Pass `--execute` to actually perform the moves.

HARD CONSTRAINT (ADR 0007 decision 5): the PINNED eval gold
(`data/run2a/component_a_v1.jsonl` and `data/run2a/eval_gold/`) is never moved,
renamed, or deleted. Any manifest entry with `"pinned": true` is refused as a guard,
even if its `action` were mistakenly set to a move — the guard is belt-and-suspenders.

Manifest `action` values handled:
  keep-in-place  / leave-as-is  -> no-op (cataloged for provenance only)
  move                          -> move a single file to `target_path`
  move-accepted                 -> move only `accepted.*.jsonl` out of a directory
                                   into `target_path/`, prefixing pm-sourced files
                                   with `pm_` to avoid collision with the preview copy

Usage:
  uv run python scripts/migrations/migrate_to_bioes_v2.py            # dry-run (default)
  uv run python scripts/migrations/migrate_to_bioes_v2.py --execute  # perform the moves
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.json_types import is_str_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "bioes-v2" / "MANIFEST.json"

MOVE_ACTIONS = frozenset({"move", "move-accepted"})
NOOP_ACTIONS = frozenset({"keep-in-place", "leave-as-is"})


@dataclass(frozen=True)
class PlannedMove:
    """A single source -> destination file move resolved from a manifest entry."""

    dataset_id: str
    src: Path
    dst: Path


def load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        sys.exit(f"ERROR: manifest not found at {path}")
    with path.open(encoding="utf-8") as fh:
        value: object = json.load(fh)
    if not is_str_mapping(value):
        sys.exit("ERROR: manifest root must be an object with string keys.")
    return dict(value)


def _require_str(entry: Mapping[str, object], key: str) -> str:
    """Read a string field off an untrusted manifest entry, or stop with the cause named."""
    value = entry.get(key)
    if not isinstance(value, str):
        sys.exit(f"ERROR: manifest entry field '{key}' must be a string, got {type(value).__name__}.")
    return value


def resolve_moves(manifest: dict[str, object]) -> list[PlannedMove]:
    """Turn manifest entries into concrete file moves, enforcing the pinned guard.

    Belt-and-suspenders: a pinned entry must never resolve to a move.

    """
    moves: list[PlannedMove] = []
    entries = manifest["datasets"]
    if not isinstance(entries, list):
        sys.exit(f"ERROR: manifest 'datasets' must be a list, got {type(entries).__name__}.")
    for entry in entries:
        if not is_str_mapping(entry):
            sys.exit("ERROR: every manifest 'datasets' item must be an object with string keys.")
        dataset_id = _require_str(entry, "id")
        action = _require_str(entry, "action")

        if entry.get("pinned"):
            if action in MOVE_ACTIONS:
                sys.exit(
                    f"ERROR: manifest entry '{dataset_id}' is pinned but has a move "
                    f"action '{action}'. Pinned eval gold must never move (ADR 0007 "
                    f"decision 5). Refusing to proceed.",
                )
            continue

        if action in NOOP_ACTIONS:
            continue
        if action not in MOVE_ACTIONS:
            sys.exit(f"ERROR: unknown action '{action}' for entry '{dataset_id}'.")

        current = REPO_ROOT / _require_str(entry, "current_path")
        target = REPO_ROOT / _require_str(entry, "target_path")

        if action == "move":
            moves.append(PlannedMove(dataset_id, current, target))
        elif action == "move-accepted":
            moves.extend(_resolve_accepted_moves(dataset_id, entry, current, target))

    return moves


def _resolve_accepted_moves(
    dataset_id: str,
    entry: Mapping[str, object],
    src_dir: Path,
    dst_dir: Path,
) -> list[PlannedMove]:
    """For `move-accepted`: move each `accepted.*.jsonl` into `dst_dir`.

    project-manager-sourced files get a `pm_` prefix so they don't collide with the
    preview copy that lands in the same `synthetic/opencode_zen/` directory.
    """
    is_pm = _require_str(entry, "current_path").startswith("project-manager/")
    prefix = "pm_" if is_pm else ""
    moves: list[PlannedMove] = [
        PlannedMove(dataset_id, src, dst_dir / f"{prefix}{src.name}") for src in sorted(src_dir.glob("accepted.*.jsonl"))
    ]
    return moves


def rel(path: Path) -> str:
    """Repo-relative display path (falls back to absolute if outside the repo)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def execute_moves(moves: list[PlannedMove]) -> None:
    for mv in moves:
        if not mv.src.exists():
            sys.exit(f"ERROR: source missing, aborting: {rel(mv.src)}")
        if mv.dst.exists():
            sys.exit(f"ERROR: destination exists, refusing to overwrite: {rel(mv.dst)}")
    for mv in moves:
        mv.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mv.src), str(mv.dst))
        print(f"MOVED  {rel(mv.src)}  ->  {rel(mv.dst)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the moves. Without this flag the script only prints the plan.",
    )
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST_PATH)
    moves = resolve_moves(manifest)

    print(f"Manifest: {rel(MANIFEST_PATH)}")
    print(f"Planned file moves: {len(moves)}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN (no changes)'}")
    print("-" * 72)

    if not moves:
        print("Nothing to move — all entries are keep-in-place / leave-as-is.")
        return 0

    for mv in moves:
        status = "ok" if mv.src.exists() else "MISSING-SRC"
        collision = "  [DEST-EXISTS]" if mv.dst.exists() else ""
        print(f"[{status:11}] {rel(mv.src)}  ->  {rel(mv.dst)}{collision}")

    print("-" * 72)
    if not args.execute:
        print("Dry-run only. Re-run with --execute to perform these moves.")
        return 0

    execute_moves(moves)
    print(f"Done. {len(moves)} file(s) moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
