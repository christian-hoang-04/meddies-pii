from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from meddies_pii.eval_baseline.adapters import opf_backend
from meddies_pii.evaluation.runtime_provenance import (
    evaluation_runtime_source_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RUNNERS = (
    "run_model_baseline.py",
    "run_gliner2_baseline.py",
    "run_lfm25_pii_baseline.py",
    "run_lfm_bioes_baseline.py",
    "run_opf_baseline.py",
)


def _install_checkpoint_download(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, object]]) -> None:
    def snapshot_download(**kwargs: object) -> None:
        calls.append(kwargs)
        original = Path(str(kwargs["local_dir"])) / "original"
        original.mkdir(parents=True)
        (original / "config.json").write_text(json.dumps({"model_type": "openai_privacy_filter"}), encoding="utf-8")
        (original / "weights.safetensors").write_bytes(b"known-checkpoint-bytes")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )


def test_runtime_source_artifact_hashes_the_executable_package_closure(
    tmp_path: Path,
) -> None:
    package = tmp_path / "meddies_pii"
    package.mkdir()
    (package / "adapter.py").write_text("from .shared import decode\n", encoding="utf-8")
    shared = package / "shared.py"
    shared.write_text("def decode(): return 'first'\n", encoding="utf-8")

    runner = tmp_path / "runner.py"
    runner.write_text("run()\n", encoding="utf-8")
    before = evaluation_runtime_source_artifact(package, runner)
    shared.write_text("def decode(): return 'second'\n", encoding="utf-8")
    after_dependency_change = evaluation_runtime_source_artifact(package, runner)
    runner.write_text("run_with_new_batching()\n", encoding="utf-8")
    after_runner_change = evaluation_runtime_source_artifact(package, runner)

    assert before.reference == "meddies-pii-evaluation-runtime-source"
    assert before.revision.startswith("sha256:")
    assert before.sha256 != after_dependency_change.sha256
    assert after_dependency_change.sha256 != after_runner_change.sha256


def test_every_active_matrix_runner_uses_the_shared_runtime_source_artifact() -> None:
    for runner_name in BENCHMARK_RUNNERS:
        source = (REPO_ROOT / "scripts" / "ops" / runner_name).read_text(encoding="utf-8")
        assert "evaluation_runtime_source_artifact" in source, runner_name
        assert "local_adapter_source=source_artifact" not in source, runner_name
        assert "scorer_contract=source_artifact" not in source, runner_name


def test_opf_checkpoint_cache_reuses_only_the_verified_pinned_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert opf_backend.DEFAULT_NATIVE_CHECKPOINT_DIR.endswith(opf_backend.MODEL_REVISION)
    calls: list[dict[str, object]] = []
    _install_checkpoint_download(monkeypatch, calls)
    target = tmp_path / "checkpoint"

    assert opf_backend.prepare_native_checkpoint(target) == target
    assert len(calls) == 1
    assert calls[0]["repo_id"] == opf_backend.MODEL_ID
    assert calls[0]["revision"] == opf_backend.MODEL_REVISION
    assert calls[0]["allow_patterns"] == ["original/*"]
    assert str(calls[0]["local_dir"]) != str(target)
    assert opf_backend.prepare_native_checkpoint(target) == target
    assert len(calls) == 1

    (target / "weights.safetensors").write_bytes(b"tampered-checkpoint-bytes")
    with pytest.raises(RuntimeError, match="checkpoint cache content does not match"):
        opf_backend.prepare_native_checkpoint(target)


def test_opf_checkpoint_cache_rejects_arbitrary_existing_contents(
    tmp_path: Path,
) -> None:
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkpoint cache is unverified"):
        opf_backend.prepare_native_checkpoint(target)


def test_runtime_source_tree_excludes_generated_python_cache(tmp_path: Path) -> None:
    from meddies_pii.evaluation.runtime_provenance import source_tree_artifact

    package = tmp_path / "meddies_pii"
    package.mkdir()
    (package / "adapter.py").write_text("def predict(): pass\n", encoding="utf-8")
    before = source_tree_artifact("runtime", package)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "adapter.cpython-312.pyc").write_bytes(b"generated")

    assert source_tree_artifact("runtime", package) == before


def test_opf_contract_binds_mounted_vendor_bytes_not_an_unverified_git_claim() -> None:
    source = (REPO_ROOT / "scripts" / "ops" / "run_opf_baseline.py").read_text(encoding="utf-8")

    assert '"mounted-vendor://openai/privacy-filter"' in source
    assert "source_tree_artifact" in source
    assert "OPF_VENDOR_REVISION" not in source


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "wrong/model", "checkpoint cache model does not match"),
        ("model_revision", "0" * 40, "checkpoint cache revision does not match"),
    ],
)
def test_opf_checkpoint_cache_rejects_manifest_with_wrong_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    calls: list[dict[str, object]] = []
    _install_checkpoint_download(monkeypatch, calls)
    target = tmp_path / "checkpoint"
    opf_backend.prepare_native_checkpoint(target)

    manifest_path = next(tmp_path.glob("*.opf-checkpoint.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        opf_backend.prepare_native_checkpoint(target)
