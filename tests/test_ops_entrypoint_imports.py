"""Operational entrypoints must import from installed neutral contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENTRYPOINTS = (
    "scripts/ops/audit_internal_data.py",
    "scripts/ops/build_redistributed_mix.py",
    "scripts/ops/modal_dataset_audit.py",
    "scripts/ops/run_baseline_eval.py",
    "scripts/ops/run_bioes_inference_preview.py",
    "scripts/ops/run_gliner2_baseline.py",
    "scripts/ops/run_lfm25_pii_baseline.py",
    "scripts/ops/run_lfm_bioes_baseline.py",
    "scripts/ops/run_model_baseline.py",
    "scripts/ops/run_opf_baseline.py",
    "scripts/ops/run_opf_inference_benchmark.py",
    "scripts/ops/run_pdf_redaction_benchmark.py",
    "scripts/quality/profile_eval_aggregation.py",
)


@pytest.mark.parametrize("relative_path", _ENTRYPOINTS)
def test_operational_entrypoint_imports(relative_path: str) -> None:
    path = _REPO_ROOT / relative_path
    module_name = f"_entrypoint_import_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
