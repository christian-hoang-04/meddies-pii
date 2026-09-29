from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_runner() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts/generation/run_opencode_zen_clinical_daily.py"
    spec = importlib.util.spec_from_file_location("run_opencode_zen_clinical_daily_for_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_max_concurrency_auto_uses_key_count() -> None:
    runner = _load_runner()

    assert runner.resolve_max_concurrency(0, 7) == 7


def test_resolve_max_concurrency_keeps_explicit_override() -> None:
    runner = _load_runner()

    assert runner.resolve_max_concurrency(2, 7) == 2


def test_daily_segments_use_additional_counts_and_dated_output_contract() -> None:
    runner = _load_runner()

    segments = runner.build_opencode_zen_daily_segments(8)
    # reason: `_summary_path` is pure path composition; the literal is an argument and an expected
    # reason: value, and no filesystem call reaches it. Proven by an audit-hook run over this module.
    summary_path = runner._summary_path(Path("/tmp/out"), "2026-06-14")  # ruff: ignore[hardcoded-temp-file]

    assert sum(segment.target_count for segment in segments) == 8
    assert {segment.count_mode for segment in segments} == {"additional"}
    assert summary_path == Path("/tmp/out/daily_runs/2026-06-14.summary.json")  # ruff: ignore[hardcoded-temp-file]
