from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
SCRIPT_SUBDIRS = (
    REPO / "scripts/ops",
    REPO / "scripts/reports",
    REPO / "scripts/migrations",
)


UNREVIEWED_ROOT_PATTERNS = (
    re.compile(r'sys\.path\.insert\(0, ["\']/root["\']\)'),
    re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[1\]"),
    re.compile(r"Path\(__file__\)\.resolve\(\)\.parent\.parent"),
)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    return modules


def imported_names(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def test_active_modules_do_not_import_retired_training_modal_bootstrap() -> None:
    active_modules = (
        *(REPO / "src/meddies_pii").rglob("*.py"),
        *(path for path in (REPO / "scripts").rglob("*.py") if not path.is_relative_to(REPO / "scripts/archive")),
    )
    offenders = [
        path.relative_to(REPO)
        for path in sorted(active_modules)
        if "meddies_pii.training.bioes.modal.bootstrap" in imported_modules(path)
    ]

    assert offenders == []


def test_moved_scripts_do_not_use_unreviewed_root_hacks() -> None:
    offenders: list[str] = []
    for directory in SCRIPT_SUBDIRS:
        for path in sorted(directory.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            offenders.extend(
                f"{path.relative_to(REPO)}: {pattern.pattern}"
                for pattern in UNREVIEWED_ROOT_PATTERNS
                if pattern.search(text)
            )

    assert offenders == []


def test_ops_scripts_do_not_mutate_sys_path() -> None:
    offenders = [
        path.relative_to(REPO)
        for path in sorted((REPO / "scripts/ops").glob("*.py"))
        if "sys.path.insert" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def source_mount_calls(path: Path) -> tuple[ast.Call, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_dir"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "src"
    )


def test_ops_scripts_import_the_neutral_modal_source_mount_contract() -> None:
    ops_with_src_mounts = [path for path in sorted((REPO / "scripts/ops").glob("*.py")) if source_mount_calls(path)]

    assert ops_with_src_mounts
    for path in ops_with_src_mounts:
        assert {
            "MODAL_SOURCE_ROOT",
            "add_source_pythonpath",
        } <= imported_names(path, "meddies_pii.modal_runtime")
        assert all(
            any(
                keyword.arg == "remote_path"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "MODAL_SOURCE_ROOT"
                for keyword in call.keywords
            )
            for call in source_mount_calls(path)
        )


def import_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_migrate_to_bioes_v2_resolves_repo_root_after_move() -> None:
    module = import_script(
        REPO / "scripts/migrations/migrate_to_bioes_v2.py",
        "_migrate_to_bioes_v2_root_test",
    )

    assert module.REPO_ROOT == REPO
    assert module.MANIFEST_PATH == REPO / "data/bioes-v2/MANIFEST.json"
