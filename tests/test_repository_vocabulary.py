from __future__ import annotations

import re

# reason: git is the only authority on which files this repository tracks, and the point of the test
# reason: is to scan exactly that set rather than a directory walk that would miss the ignore rules.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_JARGON = re.compile(r"meddies(?:[-_ ]?" + "9)", re.IGNORECASE)
APPROVED_HISTORICAL_JARGON_PATHS = frozenset({
    "src/meddies_pii/historical_artifacts.py",
    "tests/test_historical_artifacts.py",
    "tests/test_repository_vocabulary.py",
})


def test_historical_jargon_is_confined_to_the_explicit_legacy_contract() -> None:
    tracked_files = subprocess.run(
        # reason: `git` stays a bare name deliberately — this test runs on developer machines and in CI,
        # reason: where git lives at different absolute paths, and it reads the tree it is already inside.
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],  # ruff: ignore[start-process-with-partial-path]
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    matches: set[str] = set()
    for relative_path in tracked_files:
        path = REPOSITORY_ROOT / relative_path
        if path.is_file() and HISTORICAL_JARGON.search(path.read_text(encoding="utf-8")):
            matches.add(relative_path)

    assert matches <= APPROVED_HISTORICAL_JARGON_PATHS
