from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ACTIVE_TARGETS = (
    REPO / "src",
    REPO / "scripts",
    REPO / "README.md",
    REPO / "CONTRIBUTING.md",
    REPO / "CLAUDE.md",
)
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json"}
RETIRED_PATTERNS = (
    "project-manager/preview",
    "project-manager/reports",
    "project-manager/tasks",
    "project-manager/progress",
    "project-manager/adr",
    "preview/pii_bioes_general_aug",
)
ALLOWED_COMPATIBILITY_LINES = {
    "CLAUDE.md": (
        "There is no in-repo `project-manager/`",
        "Never write project docs to `project-manager/`",
    ),
    "scripts/migrations/migrate_to_bioes_v2.py": (
        "project-manager-sourced files get a `pm_` prefix",
        'startswith("project-manager/")',
    ),
}


def _active_files() -> list[Path]:
    files: list[Path] = []
    for target in ACTIVE_TARGETS:
        if target.is_file():
            files.append(target)
            continue
        files.extend(
            path
            for path in target.rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts
        )
    return sorted(files)


def _is_allowed_compatibility_line(relative_path: str, line: str) -> bool:
    return any(marker in line for marker in ALLOWED_COMPATIBILITY_LINES.get(relative_path, ()))


def test_active_surfaces_do_not_default_to_retired_storage_paths() -> None:
    offenders: list[str] = []
    for path in _active_files():
        relative_path = str(path.relative_to(REPO))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not any(pattern in line for pattern in RETIRED_PATTERNS):
                continue
            if _is_allowed_compatibility_line(relative_path, line):
                continue
            offenders.append(f"{relative_path}:{line_number}: {line.strip()}")

    assert offenders == []
