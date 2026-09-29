"""Tests for the coverage inventory and independent threshold gate."""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

    import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_gate() -> ModuleType:
    path = REPO_ROOT / "scripts/quality/check_coverage.py"
    spec = importlib.util.spec_from_file_location("coverage_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_tree(tmp_path: Path) -> Path:
    """Training deliberately has no __init__.py: coverage must still inventory it."""
    source_root = tmp_path / "src/meddies_pii"
    (source_root / "regular").mkdir(parents=True)
    (source_root / "training/namespace_child").mkdir(parents=True)
    (source_root / "regular/__init__.py").write_text("")
    (source_root / "regular/module.py").write_text("value = 1\n")
    (source_root / "training/namespace_child/module.py").write_text("value = 2\n")
    return source_root


def _report(source_root: Path, *, branch_coverage: bool = True) -> dict[str, object]:
    files: dict[str, dict[str, dict[str, int]]] = {}
    for path in source_root.rglob("*.py"):
        files[str(path)] = {
            "summary": {
                "covered_lines": 80,
                "num_statements": 100,
                "covered_branches": 8,
                "num_branches": 10,
            },
        }
    return {"meta": {"branch_coverage": branch_coverage}, "files": files}


def test_accepts_complete_namespace_package_inventory(tmp_path: Path) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)

    errors = gate.validate_coverage(_report(source_root), source_root=source_root)

    assert errors == []


def test_accepts_the_measured_branch_ratchet_by_default(tmp_path: Path) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)
    report = _report(source_root)
    files = cast("dict[str, dict[str, dict[str, int]]]", report["files"])
    first_summary = next(iter(files.values()))["summary"]
    first_summary["covered_branches"] = 685
    first_summary["num_branches"] = 980

    errors = gate.validate_coverage(report, source_root=source_root)

    assert errors == []


def test_rejects_report_without_branch_metadata(tmp_path: Path) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)

    errors = gate.validate_coverage(_report(source_root, branch_coverage=False), source_root=source_root)

    assert errors == ["coverage report was not collected with branch coverage"]


def test_rejects_missing_namespace_package_file(tmp_path: Path) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)
    report = _report(source_root)
    missing = source_root / "training/namespace_child/module.py"
    files = cast("dict[str, object]", report["files"])
    del files[str(missing)]

    errors = gate.validate_coverage(report, source_root=source_root)

    assert errors == ["coverage report omitted 1 active source file: src/meddies_pii/training/namespace_child/module.py"]


def test_rejects_statement_coverage_below_threshold(tmp_path: Path) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)
    report = _report(source_root)
    files = cast("dict[str, dict[str, dict[str, int]]]", report["files"])
    first_summary = next(iter(files.values()))["summary"]
    first_summary["covered_lines"] = 79

    errors = gate.validate_coverage(report, source_root=source_root)

    assert errors == ["statement coverage 79.7% (239/300) is below 80.0%"]


def test_rejects_branch_coverage_below_threshold(tmp_path: Path) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)
    report = _report(source_root)
    files = cast("dict[str, dict[str, dict[str, int]]]", report["files"])
    first_summary = next(iter(files.values()))["summary"]
    first_summary["covered_branches"] = 684
    first_summary["num_branches"] = 980

    errors = gate.validate_coverage(report, source_root=source_root)

    assert errors == ["branch coverage 70.0% (700/1000) is below 70.1%"]


def test_non_source_files_cannot_inflate_statement_coverage(tmp_path: Path) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)
    report = _report(source_root)
    files = cast("dict[str, dict[str, dict[str, int]]]", report["files"])
    for details in files.values():
        details["summary"]["covered_lines"] = 79
    files[str(tmp_path / "outside.py")] = {
        "summary": {
            "covered_lines": 10_000,
            "num_statements": 10_000,
            "covered_branches": 10_000,
            "num_branches": 10_000,
        },
    }

    errors = gate.validate_coverage(report, source_root=source_root)

    assert errors == ["statement coverage 79.0% (237/300) is below 80.0%"]


def test_cli_threshold_flags_allow_focused_coverage_test(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gate = _load_gate()
    source_root = _source_tree(tmp_path)
    report_path = tmp_path / "coverage.json"
    report_path.write_text(json.dumps(_report(source_root)))

    exit_code = gate.main([
        str(report_path),
        "--source-root",
        str(source_root),
        "--min-statements",
        "80.1",
        "--min-branches",
        "80.1",
    ])

    assert exit_code == 1
    assert "statement coverage 80.0% (240/300) is below 80.1%" in capsys.readouterr().err
