from __future__ import annotations

from pathlib import Path

import tomllib

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_source_type_gates_keep_mypy_strict_without_source_exclusions() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    mypy = project["tool"]["mypy"]
    basedpyright = project["tool"]["basedpyright"]

    assert mypy["strict"] is True
    assert "exclude" not in mypy
    assert basedpyright["typeCheckingMode"] == "standard"
    assert all(not str(excluded_path).startswith("src/") for excluded_path in basedpyright.get("exclude", []))
