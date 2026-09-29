"""Architecture checks for the baseline evaluation Module."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = REPOSITORY_ROOT / "src" / "meddies_pii" / "eval_baseline" / "baseline"
REGEX_RELEASE_MODULE = "meddies_pii.eval_baseline.regex_release"


def _imports_module(source_path: Path, module: str) -> bool:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == module or alias.name.startswith(f"{module}.") for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == module or node.module.startswith(f"{module}."))
        ):
            return True
    return False


def test_baseline_module_does_not_depend_on_regex_release_workflow() -> None:
    forbidden_importers = [
        source_path.relative_to(REPOSITORY_ROOT).as_posix()
        for source_path in BASELINE_ROOT.rglob("*.py")
        if _imports_module(source_path, REGEX_RELEASE_MODULE)
    ]

    assert forbidden_importers == []
