from __future__ import annotations

# ruff: file-ignore[no-self-use]
# reason: the stateless volume double retains Modal's bound commit method shape.
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from meddies_pii.eval_baseline.regex_release.regex_ignore_list import (
    ignore_list_from_rows,
    write_ignore_list_jsonl,
)
from meddies_pii.eval_baseline.regex_release.regex_release_contract import ReleaseGateContract

if TYPE_CHECKING:
    from types import ModuleType


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "ops" / "run_regex_release_gate.py"
    spec = importlib.util.spec_from_file_location("run_regex_release_gate_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "split": "eval",
        "uid": "audit-doc-0c33d83fd43447dd",
        "start": 23,
        "end": 48,
        "text": "Bệnh Viện Đa Khoa Sài Gòn",
        "category": "company_name",
    }
    base.update(overrides)
    return base


def test_stage_and_verify_returns_none_when_contract_declares_no_list(
    tmp_path: Path,
) -> None:
    runner = _runner()
    contract = runner.ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")
    artifact_path = tmp_path / "mounted.jsonl"
    write_ignore_list_jsonl(artifact_path, ignore_list_from_rows([_row()]))

    digest = runner._stage_and_verify_ignore_list(tmp_path / "output", contract, local_artifact_path=artifact_path)

    assert digest is None
    assert not (tmp_path / "output").exists()


def test_stage_and_verify_refuses_a_digest_mismatch_before_persisting(
    tmp_path: Path,
) -> None:
    """Reject a mounted ignore-list whose digest differs from the pin.

    Deliberately not ignore_list.sha256 — the mounted artifact will never match, simulating a stale or wrong file baked
    into the image.
    """
    runner = _runner()
    ignore_list = ignore_list_from_rows([_row()])
    contract = runner.ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256="a" * 64,
    )
    artifact_path = tmp_path / "mounted.jsonl"
    write_ignore_list_jsonl(artifact_path, ignore_list)
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="does not match the contract pin"):
        runner._stage_and_verify_ignore_list(output_root, contract, local_artifact_path=artifact_path)

    assert not output_root.exists()


def test_stage_and_verify_persists_a_verified_copy_other_stages_can_load(
    tmp_path: Path,
) -> None:
    runner = _runner()
    ignore_list = ignore_list_from_rows([_row()])
    contract = runner.ReleaseGateContract.default(
        source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf",
        ignore_list_sha256=ignore_list.sha256,
    )
    artifact_path = tmp_path / "mounted.jsonl"
    write_ignore_list_jsonl(artifact_path, ignore_list)
    output_root = tmp_path / "output"

    digest = runner._stage_and_verify_ignore_list(output_root, contract, local_artifact_path=artifact_path)

    assert digest == ignore_list.sha256
    staged = runner._load_staged_ignore_list(output_root, contract)
    assert staged is not None
    assert staged.sha256 == ignore_list.sha256


def test_load_staged_ignore_list_returns_none_when_contract_declares_no_list(
    tmp_path: Path,
) -> None:
    runner = _runner()
    contract = runner.ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    assert runner._load_staged_ignore_list(tmp_path / "output", contract) is None


def test_ignore_list_artifact_dir_exists_after_import_on_a_fresh_clone() -> None:
    runner = _runner()

    assert runner.IGNORE_LIST_ARTIFACT_DIR.is_dir()


def test_checked_contract_accepts_a_render_pinned_with_an_ignore_list_digest() -> None:
    """Accept a checked contract reconstructed with the rendered pin.

    The local/remote handshake: render pins a digest computed WITH the
    ignore-list; _checked_contract (the construction path hydrate, run_eval,
    run_eval_challenge, and aggregate_5100 all share) must reconstruct the
    identical digest when given the same pin, or every remote stage refuses
    the approved contract.
    """
    runner = _runner()
    source_commit = "9431725cc96d7014a5433c13fbf3c7e257a5a7bf"
    ignore_list_sha256 = "e" * 64
    rendered = runner.ReleaseGateContract.default(source_commit=source_commit, ignore_list_sha256=ignore_list_sha256)

    accepted = runner._checked_contract(source_commit, rendered.sha256, ignore_list_sha256)

    assert accepted.sha256 == rendered.sha256
    assert accepted.ignore_list_sha256 == ignore_list_sha256


def test_checked_contract_refuses_a_pinned_render_without_the_pin() -> None:
    """Refuse a pinned render when contract reconstruction omits the pin.

    Reproduces the reported failure mode directly: reconstructing without
    the pin computes a different (unpinned) digest than a pinned render, so
    the contract is refused — this is the exact mechanism that made
    stage_approval_receipt raise 'approved contract SHA-256 does not match
    rendered contract' for every pinned contract before the fix.
    """
    runner = _runner()
    source_commit = "9431725cc96d7014a5433c13fbf3c7e257a5a7bf"
    rendered = runner.ReleaseGateContract.default(source_commit=source_commit, ignore_list_sha256="e" * 64)

    with pytest.raises(ValueError, match="does not match frozen inputs"):
        runner._checked_contract(source_commit, rendered.sha256)


def test_checked_contract_still_accepts_an_unpinned_render() -> None:
    """Accept an unpinned render through the unchanged default path.

    The unpinned path is unaffected: render's default digest still matches
    _checked_contract's default construction, so a no-ignore-list run is
    exactly as it was before this feature existed.
    """
    runner = _runner()
    source_commit = "9431725cc96d7014a5433c13fbf3c7e257a5a7bf"
    rendered = runner.ReleaseGateContract.default(source_commit=source_commit)

    accepted = runner._checked_contract(source_commit, rendered.sha256)

    assert accepted.sha256 == rendered.sha256
    assert accepted.ignore_list_sha256 is None


def test_stage_approval_construction_accepts_a_render_pinned_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the ignore-list pin through the real stage-approval body.

    Exercises _stage_approval's real body end to end (params → contract
    construction → the receipt call) without touching the real filesystem —
    stage_approval_receipt and the volume handle are captured, not executed,
    so this proves the construction step specifically, which is where the
    handshake broke.
    """
    runner = _runner()
    source_commit = "9431725cc96d7014a5433c13fbf3c7e257a5a7bf"
    ignore_list_sha256 = "e" * 64
    rendered = runner.ReleaseGateContract.default(source_commit=source_commit, ignore_list_sha256=ignore_list_sha256)
    captured: dict[str, Any] = {}

    def fake_stage_approval_receipt(_output_root: object, contract: object, *, approved_contract_sha256: str) -> None:
        captured["contract"] = contract
        captured["approved_contract_sha256"] = approved_contract_sha256

    class FakeVolume:
        def commit(self) -> None:
            return None

    monkeypatch.setattr(runner, "stage_approval_receipt", fake_stage_approval_receipt)
    monkeypatch.setattr(runner, "output_volume", FakeVolume())

    returned_sha256 = runner._stage_approval.local(source_commit, rendered.sha256, ignore_list_sha256)

    assert returned_sha256 == rendered.sha256
    assert captured["contract"].sha256 == rendered.sha256
    assert captured["approved_contract_sha256"] == rendered.sha256


def test_modal_functions_request_the_contract_compute_envelope() -> None:
    """Keep Modal decorators aligned with the contract compute envelope.

    The decorators carry literal resources while the contract declares the envelope the evidence records; nothing links
    them, so a contract-only amendment would silently leave the paid containers on the old shape.
    """
    runner = _runner()
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    assert contract.cpu_cores == runner.GATE_CPU_CORES
    assert contract.memory_mib == runner.GATE_MEMORY_MIB
    assert contract.timeout_seconds == runner.GATE_TIMEOUT_SECONDS

    for function in (
        runner._stage_approval,
        runner._hydrate,
        runner._run_child,
        runner._aggregate,
    ):
        assert function.spec.cpu == contract.cpu_cores
        assert function.spec.memory == contract.memory_mib


def test_onnx_runtime_thread_env_matches_the_contract_cores() -> None:
    """Keep ONNX Runtime threads aligned with the contracted CPU count.

    ONNX Runtime reads OMP_NUM_THREADS for intra-op parallelism, so a stale value silently caps decode below the cores
    the contract pays for.
    """
    runner = _runner()
    contract = ReleaseGateContract.default(source_commit="9431725cc96d7014a5433c13fbf3c7e257a5a7bf")

    assert runner.GATE_RUNTIME_ENV["OMP_NUM_THREADS"] == str(contract.cpu_cores)
