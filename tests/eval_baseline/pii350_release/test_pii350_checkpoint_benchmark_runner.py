from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[no-self-use]
# reason: stateless test doubles retain the bound method shape of compiler and Modal function interfaces they replace.
import base64
import hashlib
import importlib.util
import json
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from meddies_pii.eval_baseline.baseline.datasets import EVAL_DATASETS, EVAL_EXPECTED_ROWS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.eval_baseline.baseline.run import ShardSpec
    from meddies_pii.eval_baseline.pii350_release.checkpoint_benchmark import BenchmarkRequest


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "ops" / "run_pii350_checkpoint_benchmark.py"
    spec = importlib.util.spec_from_file_location("pii350_checkpoint_benchmark_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(runner: ModuleType, digest: str = "a" * 64) -> BenchmarkRequest:
    return cast(
        "BenchmarkRequest",
        runner.BenchmarkRequest(
            repo_id="Meddies/private-pii350",
            revision="b" * 40,
            artifact_path="trajectories/pii350/checkpoints/step-00000100",
            checkpoint_digest=digest,
            trajectory_digest="d" * 64,
            max_all_in_usd=2.6,
            all_in_rate_usd_per_second=runner.A10G_ALL_IN_RATE_USD_PER_SECOND,
            reserve_seconds_per_cell=300,
            profile="diffusionllm",
        ),
    )


def _approval(
    runner: ModuleType,
    request: BenchmarkRequest,
    purpose: str,
    nonce: str = "e" * 64,
) -> dict[str, Any]:
    private_key = Ed25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    )
    # reason: these tests pin the imported runner to a test-only issuer key by rebinding module globals.
    # reason: `ModuleType` declares no such attribute, so ty cannot type an assignment to one; reads through
    # reason: the module still resolve because `ModuleType.__getattr__` returns `Any`.
    runner.APPROVAL_PUBLIC_KEY_HEX = public_key_hex  # ty: ignore[unresolved-attribute]
    runner.APPROVAL_ISSUER_KEY_ID = (  # ty: ignore[unresolved-attribute]
        f"ed25519:{runner.canonical_sha256({'public_key_hex': public_key_hex})}"
    )
    body = runner._approval_body(
        request,
        purpose=purpose,
        launch_nonce=nonce,
        external_ledger_approval_id="external-ledger-test-approval",
        modeled_prior_attempts=(runner.MODELED_PRIOR_DIFFUSION_ATTEMPTS if request.profile == "diffusionllm" else ()),
    )
    return {
        **body,
        "approval_digest": runner.canonical_sha256(body),
        "signature": base64.b64encode(
            private_key.sign(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()),
        ).decode(),
    }


class _TestAtomicNonceAuthority:
    def __init__(self) -> None:
        self.claims: dict[str, object] = {}
        self.lock = threading.Lock()

    def put(self, key: str, value: object, *, skip_if_exists: bool) -> bool:
        assert skip_if_exists is True
        with self.lock:
            if key in self.claims:
                return False
            self.claims[key] = value
            return True


def _active_approval(
    runner: ModuleType,
    request: BenchmarkRequest,
    purpose: str,
    nonce: str = "e" * 64,
) -> dict[str, Any]:
    if not hasattr(runner, "_test_nonce_claim_authority"):
        authority = _TestAtomicNonceAuthority()
        runner._test_nonce_claim_authority = authority
        runner.nonce_claim_authority = authority
    approval = _approval(runner, request, purpose, nonce)
    runner.persist_active_launch_lease(request, purpose=purpose, approval=approval)
    return approval


# reason: Metrics and checkpoint, trajectory, evaluation, and comparison digests are independent cadence axes.
def _cadence_result(  # ruff: ignore[too-many-arguments]
    step: int,
    *,
    checkpoint: str,
    f1: float,
    recall: float,
    trajectory: str = "d" * 64,
    evaluation_contract: str = "e" * 64,
    comparison_contract: str = "f" * 64,
) -> dict[str, object]:
    return {
        "checkpoint_digest": checkpoint * 64,
        "trajectory_digest": trajectory,
        "evaluation_contract_digest": evaluation_contract,
        "comparison_contract_digest": comparison_contract,
        "optimizer_step": step,
        "fixed_nine_exact_typed": {
            "precision": 0.8,
            "recall": recall,
            "f1": f1,
        },
    }


def test_step60_baseline_is_a_fixed_complete_matrix_with_verified_v2_reuse() -> None:
    runner = _runner()
    contract = runner.benchmark_contract(_request(runner), purpose="step60_baseline")

    runner.require_step60_baseline_matrix(contract)
    assert tuple(contract["datasets"]) == EVAL_DATASETS
    assert contract["rows_by_cell"] == EVAL_EXPECTED_ROWS
    assert contract["total_rows"] == 263_785
    assert contract["expected_full9_gold_spans"] == runner.FULL_GOLD_SPANS
    assert contract["v2_eval_policy"] == "reuse_only_if_identity_verified_else_rerun"
    assert contract["v2_eval_challenge_policy"] == "require_existing_identity_verified_no_rerun"
    with pytest.raises(RuntimeError, match="optimizer step 60"):
        runner.require_step60_checkpoint(100)
    runner.require_step60_checkpoint(60)


def test_step60_baseline_report_is_complete_and_safety_veto_eligible() -> None:
    runner = _runner()
    report = runner.step60_baseline_coverage_report(v2_eval_reused=True)

    assert report == {
        "coverage_cells": "17/17",
        "v2_eval": {
            "status": "reused",
            "policy": "reuse_only_if_identity_verified_else_rerun",
        },
        "v2_eval_challenge": {
            "status": "reused",
            "policy": "require_existing_identity_verified_no_rerun",
        },
        "eligible_for_full_matrix_safety_veto": True,
    }
    assert runner.step60_baseline_coverage_report(v2_eval_reused=False)["v2_eval"]["status"] == "rerun"
    with pytest.raises(RuntimeError, match="exact 17-cell"):
        runner.require_full_matrix(runner.benchmark_contract(_request(runner), purpose="step60_baseline"))


def test_step60_partial_16of17_contract_is_explicit_and_never_full() -> None:
    runner = _runner()
    contract = runner.benchmark_contract(_request(runner), purpose="step60_partial_16of17")

    runner.require_step60_partial_16of17_matrix(contract)
    assert tuple(contract["datasets"]) == runner.STEP60_PARTIAL_16OF17_DATASETS
    assert "v2-eval-challenge" not in contract["datasets"]
    assert contract["rows_by_cell"] == {
        dataset: runner.EVAL_EXPECTED_ROWS[dataset] for dataset in runner.STEP60_PARTIAL_16OF17_DATASETS
    }
    assert contract["total_rows"] == 260_385
    assert runner.step60_partial_16of17_coverage_report() == {
        "coverage_cells": "16/17",
        "excluded_cells": ["v2-eval-challenge"],
        "eligible_for_full_matrix_safety_veto": False,
    }
    with pytest.raises(RuntimeError, match="exact 17-cell"):
        runner.require_full_matrix(contract)


def test_checkpoint_partial_16of17_accepts_step100_and_rejects_contract_drift() -> None:
    runner = _runner()
    contract = runner.benchmark_contract(_request(runner), purpose="checkpoint_partial_16of17")

    runner.require_checkpoint_partial_16of17_matrix(contract)
    for step in (60, 100, 150):
        runner.require_checkpoint_partial_optimizer_step(step)
    with pytest.raises(RuntimeError, match="step 60 or a 50-step cadence"):
        runner.require_checkpoint_partial_optimizer_step(101)
    with pytest.raises(RuntimeError, match="step 60 or a 50-step cadence"):
        runner.require_checkpoint_partial_optimizer_step(146)
    with pytest.raises(RuntimeError, match="exact fixed matrix"):
        runner.require_checkpoint_partial_16of17_matrix({
            **contract,
            "datasets": [*contract["datasets"], "v2-eval-challenge"],
        })
    with pytest.raises(RuntimeError, match="exact fixed matrix"):
        runner.require_checkpoint_partial_16of17_matrix({**contract, "excluded_cells": []})


def test_checkpoint_partial_terminal_identity_binds_digest_cursor_and_revision() -> None:
    runner = _runner()
    request = _request(runner, runner.TERMINAL_STEP146_CHECKPOINT_DIGEST)
    verified = SimpleNamespace(
        artifact=SimpleNamespace(revision=request.revision),
        lifecycle_state="terminal",
        optimizer_step=146,
        packed_cursor=18_688,
        world_size=2,
    )

    runner.require_checkpoint_partial_checkpoint(request, verified)
    for field, value, match in (
        ("packed_cursor", 18_687, "allowlisted step-146 identity"),
        ("optimizer_step", 145, "allowlisted step-146 identity"),
        ("world_size", 4, "allowlisted step-146 identity"),
        ("artifact", SimpleNamespace(revision="c" * 40), "revision"),
    ):
        invalid = SimpleNamespace(**vars(verified))
        setattr(invalid, field, value)
        with pytest.raises(RuntimeError, match=match):
            runner.require_checkpoint_partial_checkpoint(request, invalid)
    wrong_digest = _request(runner, "b" * 64)
    with pytest.raises(RuntimeError, match="allowlisted step-146 identity"):
        runner.require_checkpoint_partial_checkpoint(wrong_digest, verified)


def test_checkpoint_partial_16of17_state_machine_allows_only_fresh_repair_or_aggregate() -> None:
    runner = _runner()
    scheduled = runner.STEP60_PARTIAL_16OF17_DATASETS
    request = _request(runner)

    assert runner.require_complete_or_single_missing_partial_matrix(
        scheduled_datasets=scheduled,
        validated_completed_cells=(),
        missing_cells=scheduled,
        elapsed_seconds=0.0,
        request=request,
    ) == {"mode": "fresh_exact16"}
    with pytest.raises(RuntimeError, match="complete identity-proven matrix"):
        runner.require_complete_or_single_missing_partial_matrix(
            scheduled_datasets=scheduled,
            validated_completed_cells=scheduled[:1],
            missing_cells=scheduled[1:],
            elapsed_seconds=0.0,
            request=request,
        )
    repair = runner.require_complete_or_single_missing_partial_matrix(
        scheduled_datasets=scheduled,
        validated_completed_cells=scheduled[:-1],
        missing_cells=scheduled[-1:],
        elapsed_seconds=0.0,
        request=request,
    )
    assert repair is not None
    assert repair["mode"] == "single_missing_cell_resume"
    assert runner.require_complete_or_single_missing_partial_matrix(
        scheduled_datasets=scheduled,
        validated_completed_cells=scheduled,
        missing_cells=(),
        elapsed_seconds=0.0,
        request=request,
    ) == {"mode": "aggregate_only"}
    with pytest.raises(RuntimeError, match="complete identity-proven matrix"):
        runner.require_complete_or_single_missing_partial_matrix(
            scheduled_datasets=scheduled,
            validated_completed_cells=scheduled[:-2],
            missing_cells=scheduled[-1:],
            elapsed_seconds=0.0,
            request=request,
        )


def test_checkpoint_partial_terminal146_can_start_a_fresh_exact16_matrix() -> None:
    runner = _runner()
    request = _request(runner, runner.TERMINAL_STEP146_CHECKPOINT_DIGEST)
    verified = SimpleNamespace(
        artifact=SimpleNamespace(revision=request.revision),
        lifecycle_state="terminal",
        optimizer_step=146,
        packed_cursor=18_688,
        world_size=2,
    )

    runner.require_checkpoint_partial_checkpoint(request, verified)
    assert runner.require_complete_or_single_missing_partial_matrix(
        scheduled_datasets=runner.STEP60_PARTIAL_16OF17_DATASETS,
        validated_completed_cells=(),
        missing_cells=runner.STEP60_PARTIAL_16OF17_DATASETS,
        elapsed_seconds=0.0,
        request=request,
    ) == {"mode": "fresh_exact16"}


def test_checkpoint_partial_16of17_rejects_wrong_caller_digest_before_modal_work() -> None:
    runner = _runner()
    request = _request(runner)

    with pytest.raises(RuntimeError, match="caller checkpoint digest"):
        runner.run_checkpoint_partial_16of17.local(
            asdict(request),
            "b" * 64,
            _approval(runner, request, runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE),
        )


def test_launch_approval_binds_modeled_inventory_arithmetic_and_nonce_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)
    approval = _approval(runner, request, runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE, "9" * 64)

    assert (
        runner.verify_launch_approval(approval, request=request, purpose=runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE)
        == approval
    )
    assert approval["cumulative_modeled_ceiling_usd"] == "3.81484763390416214"
    assert approval["modeled_prior_attempts"] == list(runner.MODELED_PRIOR_DIFFUSION_ATTEMPTS)
    for field, value in (
        ("profile", "meddies-pii"),
        ("checkpoint_digest", "b" * 64),
        ("purpose", "full_benchmark"),
        ("new_attempt_ceiling_usd", "2.8"),
    ):
        tampered = {**approval, field: value}
        body = {key: item for key, item in tampered.items() if key != "approval_digest"}
        tampered["approval_digest"] = runner.canonical_sha256(body)
        with pytest.raises(RuntimeError):
            runner.verify_launch_approval(
                tampered,
                request=request,
                purpose=runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE,
            )

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeAtomicNonceAuthority:
        def __init__(self) -> None:
            self.claims: dict[str, object] = {}

        def put(self, key: str, value: object, *, skip_if_exists: bool) -> bool:
            assert skip_if_exists is True
            if key in self.claims:
                return False
            self.claims[key] = value
            return True

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "nonce_claim_authority", FakeAtomicNonceAuthority())
    runner.claim_launch_nonce(approval)
    with pytest.raises(RuntimeError, match="already claimed"):
        runner.claim_launch_nonce(approval)
    assert runner.preflight_failure_root(
        request,
        purpose=runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        approval=approval,
    ) != runner.preflight_failure_root(
        request,
        purpose=runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE,
        approval={**approval, "launch_nonce": "8" * 64},
    )


# reason: The barrier race and cross-container replay share one nonce authority; splitting would reset atomicity state.
def test_nonce_claim_is_atomic_under_concurrent_replay_and_shared_authority(  # ruff: ignore[complex-structure]
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)
    approval = _approval(runner, request, "full_benchmark", "f" * 64)

    class FakeVolume:
        def commit(self) -> None:
            pass

    class AtomicNonceAuthority:
        def __init__(self) -> None:
            self.claims: dict[str, object] = {}
            self.lock = threading.Lock()

        def put(self, key: str, value: object, *, skip_if_exists: bool) -> bool:
            assert skip_if_exists is True
            with self.lock:
                if key in self.claims:
                    return False
                self.claims[key] = value
                return True

    authority = AtomicNonceAuthority()
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "nonce_claim_authority", authority)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    outcomes_lock = threading.Lock()

    def contender() -> None:
        barrier.wait()
        try:
            runner.claim_launch_nonce(approval)
        except RuntimeError as error:
            outcome = str(error)
        # reason: this concurrent probe records unexpected exception types explicitly so a thread cannot fail silently.
        except Exception as error:  # ruff: ignore[blind-except]
            outcome = f"unexpected:{type(error).__name__}:{error}"
        else:
            outcome = "claimed"
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=contender) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("claimed") == 1
    assert outcomes.count("launch approval nonce was already claimed") == 7

    other_container = _runner()
    monkeypatch.setattr(other_container, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(other_container, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(other_container, "nonce_claim_authority", authority)
    with pytest.raises(RuntimeError, match="already claimed"):
        other_container.claim_launch_nonce(approval)


def test_concurrent_matrix_replays_write_one_model_gate_failure_and_no_tmp_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVolume:
        def commit(self) -> None:
            pass

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    approval = _active_approval(runner, request, "full_benchmark", "1" * 64)
    monkeypatch.setattr(
        runner,
        "enable_offline_runtime",
        lambda _: (_ for _ in ()).throw(RuntimeError("model gate refused")),
    )
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    outcomes_lock = threading.Lock()

    def contender() -> None:
        barrier.wait()
        try:
            runner.run_full_benchmark.local(asdict(request), request.checkpoint_digest, approval)
        except RuntimeError as error:
            outcome = str(error)
        # reason: this concurrent probe records unexpected exception types explicitly so a thread cannot fail silently.
        except Exception as error:  # ruff: ignore[blind-except]
            outcome = f"unexpected:{type(error).__name__}:{error}"
        else:
            outcome = "unexpected-success"
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=contender) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("launch approval nonce was already claimed") == 7
    assert outcomes.count("model gate refused") == 1
    root = runner.preflight_failure_root(request, purpose="full_benchmark", approval=approval)
    assert json.loads((root / "failure.json").read_text())["error"] == "model gate refused"
    assert not list(root.rglob("*.tmp"))


def _issuer_key_for_test(runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Generate a test-only key and pin this imported runner to its public half."""
    private_key = Ed25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    )
    key_path = tmp_path / "issuer.pem"
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    key_path.chmod(0o600)
    monkeypatch.setenv(runner.APPROVAL_PRIVATE_KEY_ENV, str(key_path))
    monkeypatch.setattr(runner, "APPROVAL_PUBLIC_KEY_HEX", public_key_hex)
    monkeypatch.setattr(
        runner,
        "APPROVAL_ISSUER_KEY_ID",
        f"ed25519:{runner.canonical_sha256({'public_key_hex': public_key_hex})}",
    )
    return key_path


def test_local_ed25519_issuer_bounds_allocation_and_remote_entrypoint_verifies_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second live reservation is allowed in principle.

    But diffusionllm has 1.21 of settled spend and 2.6 already active against a 4.0 allocation, so another 2.6 exceeds the
    cumulative bound.

    """
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    request = _request(runner)
    ledger_path = tmp_path / "ledger.json"
    approval = runner.issue_launch_approval(
        request,
        purpose="full_benchmark",
        launch_nonce="1" * 64,
        ledger_path=ledger_path,
    )
    assert approval["issuer_key_id"] == runner.APPROVAL_ISSUER_KEY_ID
    assert runner.verify_launch_approval(approval, request=request, purpose="full_benchmark") == approval
    with pytest.raises(RuntimeError, match="cumulative modeled ceiling exceeds"):
        runner.issue_launch_approval(
            request,
            purpose="full_benchmark",
            launch_nonce="2" * 64,
            ledger_path=ledger_path,
        )

    other_request = runner.BenchmarkRequest(**{**asdict(request), "profile": "meddies-pii"})
    other = runner.issue_launch_approval(
        other_request,
        purpose="full_benchmark",
        launch_nonce="3" * 64,
        ledger_path=ledger_path,
    )
    assert other["profile"] == "meddies-pii"

    calls: list[object] = []
    monkeypatch.setenv(runner.PROFILE_ENVIRONMENT_KEY, request.profile)
    monkeypatch.setattr(
        runner,
        "run_full_benchmark",
        SimpleNamespace(remote=lambda *args: calls.append(args) or {"status": "ok"}),
    )
    monkeypatch.setattr(
        runner,
        "activate_paid_lease_from_ledger",
        lambda **_: {"status": "active"},
    )
    runner.main(
        action="full_benchmark",
        request_json=json.dumps(asdict(request)),
        approval_json=json.dumps(approval),
        ledger_path=str(ledger_path),
    )
    assert len(calls) == 1


def test_local_issuer_key_and_ledger_fail_closed_without_paid_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)
    ledger_path = tmp_path / "ledger.json"
    monkeypatch.setattr(
        runner,
        "revoke_launch_lease",
        SimpleNamespace(remote=lambda *_: {"status": "released"}),
    )
    with pytest.raises(RuntimeError, match=runner.APPROVAL_PRIVATE_KEY_ENV):
        runner.issue_launch_approval(
            request,
            purpose="full_benchmark",
            launch_nonce="1" * 64,
            ledger_path=ledger_path,
        )
    key_path = _issuer_key_for_test(runner, monkeypatch, tmp_path)
    key_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        runner.issue_launch_approval(
            request,
            purpose="full_benchmark",
            launch_nonce="1" * 64,
            ledger_path=ledger_path,
        )
    key_path.chmod(0o600)
    ledger_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger"):
        runner.issue_launch_approval(
            request,
            purpose="full_benchmark",
            launch_nonce="1" * 64,
            ledger_path=ledger_path,
        )


def test_fabricated_or_wrong_issuer_approval_is_rejected_before_remote_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    request = _request(runner)
    approval = runner.issue_launch_approval(
        request,
        purpose="full_benchmark",
        launch_nonce="1" * 64,
        ledger_path=tmp_path / "ledger.json",
    )
    monkeypatch.setenv(runner.PROFILE_ENVIRONMENT_KEY, request.profile)
    monkeypatch.setattr(
        runner,
        "run_full_benchmark",
        SimpleNamespace(remote=lambda *_args: pytest.fail("paid remote was called")),
    )
    forged = {**approval, "issuer_key_id": "ed25519:wrong"}
    forged_body = {key: value for key, value in forged.items() if key not in {"approval_digest", "signature"}}
    forged["approval_digest"] = runner.canonical_sha256(forged_body)
    with pytest.raises(RuntimeError, match="issuer key ID"):
        runner.main(
            action="full_benchmark",
            request_json=json.dumps(asdict(request)),
            approval_json=json.dumps(forged),
        )
    forged = {**approval, "signature": base64.b64encode(b"fabricated").decode()}
    with pytest.raises(RuntimeError, match="signature"):
        runner.main(
            action="full_benchmark",
            request_json=json.dumps(asdict(request)),
            approval_json=json.dumps(forged),
        )


def test_released_live_lease_blocks_gpu_function_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVolume:
        def commit(self) -> None:
            pass

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    approval = _approval(runner, request, "full_benchmark", "4" * 64)
    runner.persist_active_launch_lease(request, purpose="full_benchmark", approval=approval)
    runner.persist_released_launch_lease(approval["approval_digest"])
    monkeypatch.setattr(
        runner,
        "enable_offline_runtime",
        lambda _: pytest.fail("released approval reached model runtime"),
    )

    with pytest.raises(RuntimeError, match="absent or revoked"):
        runner.run_full_benchmark.local(asdict(request), request.checkpoint_digest, approval)


def test_ledger_never_reissues_nonce_or_digest_after_release_or_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    request = _request(runner)
    ledger_path = tmp_path / "ledger.json"
    monkeypatch.setattr(
        runner,
        "revoke_launch_lease",
        SimpleNamespace(remote=lambda *_: {"status": "released"}),
    )
    first = runner.issue_launch_approval(
        request,
        purpose="full_benchmark",
        launch_nonce="5" * 64,
        ledger_path=ledger_path,
    )
    runner.release_launch_approval(ledger_path, approval_digest=first["approval_digest"])
    with pytest.raises(RuntimeError, match="nonce was already issued"):
        runner.issue_launch_approval(
            request,
            purpose="full_benchmark",
            launch_nonce="5" * 64,
            ledger_path=ledger_path,
        )
    second = runner.issue_launch_approval(
        request,
        purpose="full_benchmark",
        launch_nonce="6" * 64,
        ledger_path=ledger_path,
    )
    runner.settle_launch_approval(ledger_path, approval_digest=second["approval_digest"], modeled_cost_usd="0.1")
    with pytest.raises(RuntimeError, match="nonce was already issued"):
        runner.issue_launch_approval(
            request,
            purpose="full_benchmark",
            launch_nonce="6" * 64,
            ledger_path=ledger_path,
        )


def test_settled_approval_cannot_reach_paid_lease_or_gpu_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    request = _request(runner)
    ledger_path = tmp_path / "ledger.json"
    approval = runner.issue_launch_approval(
        request,
        purpose="full_benchmark",
        launch_nonce="9" * 64,
        ledger_path=ledger_path,
    )
    monkeypatch.setattr(
        runner,
        "revoke_launch_lease",
        SimpleNamespace(remote=lambda *_: {"status": "released"}),
    )
    runner.settle_launch_approval(ledger_path, approval_digest=approval["approval_digest"], modeled_cost_usd="0.1")
    monkeypatch.setenv(runner.PROFILE_ENVIRONMENT_KEY, request.profile)
    monkeypatch.setattr(
        runner,
        "activate_launch_lease",
        SimpleNamespace(remote=lambda *_: pytest.fail("settled approval reached lease activation")),
    )
    monkeypatch.setattr(
        runner,
        "run_full_benchmark",
        SimpleNamespace(remote=lambda *_: pytest.fail("settled approval reached GPU dispatch")),
    )
    with pytest.raises(RuntimeError, match="does not match an active"):
        runner.main(
            action="full_benchmark",
            request_json=json.dumps(asdict(request)),
            approval_json=json.dumps(approval),
            ledger_path=str(ledger_path),
        )


def test_paid_dispatch_requires_matching_active_external_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    request = _request(runner)
    ledger_path = tmp_path / "ledger.json"
    approval = runner.issue_launch_approval(
        request,
        purpose="full_benchmark",
        launch_nonce="a" * 64,
        ledger_path=ledger_path,
    )
    monkeypatch.setenv(runner.PROFILE_ENVIRONMENT_KEY, request.profile)
    monkeypatch.setattr(
        runner,
        "activate_launch_lease",
        SimpleNamespace(remote=lambda *_: pytest.fail("ledger mismatch reached lease activation")),
    )
    with pytest.raises(RuntimeError, match="ledger_path"):
        runner.main(
            action="full_benchmark",
            request_json=json.dumps(asdict(request)),
            approval_json=json.dumps(approval),
        )
    wrong_path = tmp_path / "wrong.json"
    with pytest.raises(RuntimeError, match="no reservation"):
        runner.main(
            action="full_benchmark",
            request_json=json.dumps(asdict(request)),
            approval_json=json.dumps(approval),
            ledger_path=str(wrong_path),
        )
    original = json.loads(ledger_path.read_text())
    for field, value in (
        ("profile", "meddies-pii"),
        ("purpose", "internal_control"),
        ("launch_nonce", "b" * 64),
    ):
        mutated = json.loads(json.dumps(original))
        mutated["reservations"][0][field] = value
        ledger_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(RuntimeError, match="does not match an active"):
            runner.main(
                action="full_benchmark",
                request_json=json.dumps(asdict(request)),
                approval_json=json.dumps(approval),
                ledger_path=str(ledger_path),
            )
    ledger_path.write_text(json.dumps(original), encoding="utf-8")


def test_settlement_revocation_is_seen_by_next_gpu_progress_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    request = _request(runner)
    ledger_path = tmp_path / "ledger.json"

    class FakeVolume:
        def commit(self) -> None:
            pass

        def reload(self) -> None:
            pass

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path / "benchmark"))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    approval = runner.issue_launch_approval(
        request,
        purpose="internal_control",
        launch_nonce="c" * 64,
        ledger_path=ledger_path,
    )
    runner.persist_active_launch_lease(request, purpose="internal_control", approval=approval)
    monkeypatch.setattr(
        runner,
        "revoke_launch_lease",
        SimpleNamespace(remote=runner.persist_released_launch_lease),
    )
    runner.settle_launch_approval(ledger_path, approval_digest=approval["approval_digest"], modeled_cost_usd="0.1")
    with pytest.raises(RuntimeError, match="absent or revoked"):
        runner.enforce_internal_control_progress_budget(
            root=runner.checkpoint_output_root(request) / "internal_control",
            request=request,
            approval=approval,
            done=100,
            total=1_700,
            elapsed_seconds=1.0,
        )


def test_live_lease_read_reloads_the_volume_before_observing_tombstone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)

    class SeedVolume:
        def commit(self) -> None:
            pass

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", SeedVolume())
    approval = _approval(runner, request, "internal_control", "d" * 64)
    runner.persist_active_launch_lease(request, purpose="internal_control", approval=approval)

    class StaleSnapshotVolume:
        reloads = 0

        def commit(self) -> None:
            pass

        def reload(self) -> None:
            self.reloads += 1
            runner._atomic_json(
                runner.launch_lease_path(approval["approval_digest"]),
                {
                    "status": "released",
                    "approval_digest": approval["approval_digest"],
                    "launch_nonce": approval["launch_nonce"],
                    "profile": approval["profile"],
                    "purpose": approval["purpose"],
                },
            )

    volume = StaleSnapshotVolume()
    monkeypatch.setattr(runner, "benchmark_volume", volume)
    with pytest.raises(RuntimeError, match="absent or revoked"):
        runner.require_active_launch_lease(request, purpose="internal_control", approval=approval)
    assert volume.reloads == 1


def test_internal_control_refuses_small_reservation_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = runner.BenchmarkRequest(**{**asdict(_request(runner)), "max_all_in_usd": 0.1})

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 100

    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "require_triton_c_compiler", lambda: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: SimpleNamespace(digest="f" * 64))
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(datasets_module, "load_eval_cell", lambda *_args, **_kwargs: [object()] * 1_700)
    approval = _active_approval(runner, request, "internal_control", "7" * 64)
    monkeypatch.setattr(
        adapter_module,
        "Pii350CheckpointAdapter",
        lambda *_: pytest.fail("budget refusal reached model load"),
    )

    with pytest.raises(runner.BudgetStop, match="before model load"):
        runner.run_internal_control.local(asdict(request), request.checkpoint_digest, [], approval)
    stop = json.loads(
        (runner.evaluation_generation_root(request, "f" * 64) / "internal_control" / "budget_stop.json").read_text(),
    )
    assert stop["rows_done"] == 0
    assert stop["projected_all_in_cost_usd"] > request.max_all_in_usd
    assert stop["evaluation_generation_digest"] == "f" * 64


def test_internal_control_progress_stops_at_all_in_ceiling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVolume:
        def commit(self) -> None:
            pass

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    approval = _active_approval(runner, request, "internal_control", "8" * 64)
    monkeypatch.setattr(runner, "budget_ceiling_reached", lambda **_: True)

    with pytest.raises(runner.BudgetStop, match="during inference"):
        runner.enforce_internal_control_progress_budget(
            root=runner.checkpoint_output_root(request) / "internal_control",
            request=request,
            approval=approval,
            done=100,
            total=1_700,
            elapsed_seconds=1.0,
        )
    stop = json.loads((runner.checkpoint_output_root(request) / "internal_control" / "budget_stop.json").read_text())
    assert stop["rows_done"] == 100
    assert stop["rows_total"] == 1_700


def test_triton_c_compiler_is_explicit_and_fails_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    missing = tmp_path / "gcc"
    monkeypatch.setenv("CC", str(missing))

    assert runner.TRITON_COMPILER_APT_PACKAGE == "gcc"
    assert runner.TRITON_C_COMPILER == "/usr/bin/gcc"
    with pytest.raises(RuntimeError, match="Triton C compiler"):
        runner.require_triton_c_compiler()

    compiler = tmp_path / "gcc-ready"
    compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    compiler.chmod(0o755)
    monkeypatch.setenv("CC", str(compiler))
    assert runner.require_triton_c_compiler() == str(compiler)


def test_triton_compiler_preflight_requires_a_real_compile_and_link(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    broken = tmp_path / "broken-gcc"
    broken.write_text("#!/bin/sh\nexit 71\n", encoding="utf-8")
    broken.chmod(0o755)
    monkeypatch.setenv("CC", str(broken))

    with pytest.raises(RuntimeError, match="compile and link"):
        runner.compile_triton_c_preflight()

    compiler = tmp_path / "gcc-stub"
    compiler.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then shift; output=$1; fi\n'
        "  shift\n"
        "done\n"
        'printf x > "$output"\n',
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    monkeypatch.setenv("CC", str(compiler))
    receipt = runner.compile_triton_c_preflight()

    assert receipt == {
        "cc": str(compiler),
        "proof": "c_shared_object_compile_link_v1",
        "proof_sha256": runner.canonical_sha256({
            "compiler": str(compiler),
            "source": "int triton_preflight(void) { return 0; }\n",
            "arguments": ["-shared", "-fPIC", "-x", "c", "-", "-o", "<output>"],
        }),
    }


def test_triton_compiler_preflight_command_is_cpu_only_and_profile_bound() -> None:
    runner = _runner()
    command = runner.render_triton_compiler_preflight_command("diffusionllm")

    assert "--action triton_compiler_preflight --profile diffusionllm" in command
    with pytest.raises(ValueError, match="approved evaluation"):
        runner.render_triton_compiler_preflight_command("unknown")


# reason: The exact-16 schedule, stale-marker guard, routing, coverage receipt, and state form one partial-run contract.
@pytest.mark.parametrize(
    ("optimizer_step", "terminal"),
    [(60, False), (100, False), (146, True)],
    ids=["step60", "step100", "terminal146"],
)
# reason: pytest binds this parameter from the parametrize argnames tuple, so the boolean is labelled at every call site.
def test_checkpoint_partial_fresh_runs_exact16_and_never_uses_full_fixture(  # ruff: ignore[too-many-statements]
    optimizer_step: int,
    terminal: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(
        runner,
        runner.TERMINAL_STEP146_CHECKPOINT_DIGEST if terminal else "a" * 64,
    )
    compiler_checks: list[str] = []
    step = optimizer_step

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = step
        packed_cursor = {60: 7_680, 100: 12_800, 146: 18_688}[step]
        lifecycle_state = "terminal" if terminal else "update_committed"
        world_size = 2 if terminal else 4
        artifact = SimpleNamespace(revision=request.revision)

    class FakeAdapter:
        supported_labels = frozenset()
        truncated_documents = 0

        def __init__(self, *_: object) -> None:
            pass

        def load(self) -> None:
            assert compiler_checks == ["checked"]

    class FakeRows:
        def __init__(self, count: int) -> None:
            self.count = count

        def __len__(self) -> int:
            return self.count

    class FakeEvaluation:
        digest = "f" * 64

    identity = SimpleNamespace(digest="2" * 64)
    scheduled: list[str] = []
    matrix_roots: list[Path] = []
    force_values: list[bool] = []
    read_calls = 0
    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import aggregate as aggregate_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module
    from meddies_pii.eval_baseline.baseline import run as run_module
    from meddies_pii.evaluation import identity as identity_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    old_generation = runner.evaluation_generation_root(request, "0" * 64)
    old_markers = [
        old_generation / "results" / dataset / "full.meta.json" for dataset in runner.STEP60_PARTIAL_16OF17_DATASETS[:-1]
    ]
    for old_marker in old_markers:
        old_marker.parent.mkdir(parents=True, exist_ok=True)
        old_marker.write_text("old evaluator result", encoding="utf-8")
    expected_generation = runner.evaluation_generation_root(request, "f" * 64)
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(
        runner,
        "require_triton_c_compiler",
        lambda: compiler_checks.append("checked"),
    )
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: FakeEvaluation())
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)
    monkeypatch.setattr(
        datasets_module,
        "load_eval_cell",
        lambda dataset, **_: FakeRows(runner.EVAL_EXPECTED_ROWS[dataset]),
    )
    monkeypatch.setattr(identity_module, "dataset_shard_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(
        run_module,
        "run_shard",
        lambda **kwargs: (
            scheduled.append(kwargs["spec"].dataset)
            or force_values.append(kwargs["force"])
            or SimpleNamespace(skipped=False, output_path=tmp_path / "result.jsonl")
        ),
    )
    monkeypatch.setattr(
        run_module,
        "expected_matrix_shard_identities",
        lambda _evaluation, datasets: dict.fromkeys(datasets, identity),
    )

    def read_matrix(_root: str | Path, _model: str, datasets: Sequence[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal read_calls
        matrix_roots.append(Path(_root))
        assert Path(_root) == expected_generation
        read_calls += 1
        if read_calls == 1:
            return SimpleNamespace(
                missing_datasets=tuple(datasets),
                rows_by_dataset={},
                result_sha256_by_dataset={},
                shard_identities_by_dataset={},
                timing_by_dataset={},
            )
        return SimpleNamespace(
            missing_datasets=(),
            rows_by_dataset={dataset: object() for dataset in datasets},
            result_sha256_by_dataset=dict.fromkeys(datasets, "1" * 64),
            shard_identities_by_dataset=dict.fromkeys(datasets, identity),
            timing_by_dataset={},
        )

    monkeypatch.setattr(run_module, "read_matrix_results", read_matrix)
    monkeypatch.setattr(
        aggregate_module,
        "aggregate_results",
        lambda *_args, **_kwargs: {
            "overall": {
                "rows": runner.STEP60_PARTIAL_16OF17_ROWS,
                "rows_per_second": 1.0,
            },
        },
    )
    monkeypatch.setattr(
        run_module,
        "assert_frozen_fixture",
        lambda *_: (_ for _ in ()).throw(AssertionError("partial benchmark must not assert the 17-cell fixture")),
    )
    monkeypatch.setattr(runner, "budget_allows_next_cell", lambda **_: 1.0)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)),
    )

    payload = runner.run_checkpoint_partial_16of17.local(
        asdict(request),
        request.checkpoint_digest,
        _active_approval(runner, request, runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE),
    )

    assert matrix_roots == [expected_generation, expected_generation]
    assert len(old_markers) == 15
    assert all(old_marker.read_text(encoding="utf-8") == "old evaluator result" for old_marker in old_markers)
    assert tuple(scheduled) == runner.STEP60_PARTIAL_16OF17_DATASETS
    assert force_values == [False] * 16
    assert "v2-eval-challenge" not in scheduled
    assert payload["status"] == "checkpoint_partial_16of17_aggregate_terminal"
    assert payload["evaluation_generation_digest"] == "f" * 64
    runtime_state = json.loads((expected_generation / "state.json").read_text())
    assert runtime_state["evaluation_generation_digest"] == "f" * 64
    assert payload["coverage"] == runner.step60_partial_16of17_coverage_report()
    assert payload["coverage"]["eligible_for_full_matrix_safety_veto"] is False


def test_checkpoint_partial_aggregate_only_does_not_load_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request = _request(runner)
    adapter_constructed = False

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 100
        packed_cursor = 12_800
        lifecycle_state = "update_committed"
        world_size = 4
        artifact = SimpleNamespace(revision=request.revision)

    class FakeEvaluation:
        digest = "f" * 64

    identity = SimpleNamespace(digest="2" * 64)
    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import aggregate as aggregate_module
    from meddies_pii.eval_baseline.baseline import run as run_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "require_triton_c_compiler", lambda: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: FakeEvaluation())
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())

    def forbidden_adapter(*_args: object) -> object:
        nonlocal adapter_constructed
        adapter_constructed = True
        msg = "aggregate-only matrix must not load the adapter"
        raise AssertionError(msg)

    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", forbidden_adapter)
    monkeypatch.setattr(
        run_module,
        "expected_matrix_shard_identities",
        lambda _evaluation, datasets: dict.fromkeys(datasets, identity),
    )
    monkeypatch.setattr(
        run_module,
        "read_matrix_results",
        lambda _root, _model, datasets, **_kwargs: SimpleNamespace(
            missing_datasets=(),
            rows_by_dataset={dataset: object() for dataset in datasets},
            result_sha256_by_dataset=dict.fromkeys(datasets, "1" * 64),
            shard_identities_by_dataset=dict.fromkeys(datasets, identity),
            timing_by_dataset={},
        ),
    )
    monkeypatch.setattr(
        aggregate_module,
        "aggregate_results",
        lambda *_args, **_kwargs: {
            "overall": {
                "rows": runner.STEP60_PARTIAL_16OF17_ROWS,
                "rows_per_second": 1.0,
            },
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)),
    )

    payload = runner.run_checkpoint_partial_16of17.local(
        asdict(request),
        request.checkpoint_digest,
        _active_approval(runner, request, runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE),
    )

    assert adapter_constructed is False
    assert payload["status"] == "checkpoint_partial_16of17_aggregate_terminal"
    assert payload["cost_ledger"] == []


def test_step60_partial_single_missing_resume_reuses_validated_cells_without_budget_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    original = _request(runner)
    request = runner.BenchmarkRequest(
        original.repo_id,
        original.revision,
        original.artifact_path,
        original.checkpoint_digest,
        original.trajectory_digest,
        2.60,
        original.all_in_rate_usd_per_second,
        original.reserve_seconds_per_cell,
        original.profile,
    )

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 60

    class FakeAdapter:
        supported_labels = frozenset()
        truncated_documents = 0

        def __init__(self, *_: object) -> None:
            pass

        def load(self) -> None:
            pass

    class FakeRows:
        def __init__(self, count: int) -> None:
            self.count = count

        def __len__(self) -> int:
            return self.count

    class FakeEvaluation:
        digest = "f" * 64

    identity = SimpleNamespace(digest="2" * 64)
    missing = runner.STEP60_PARTIAL_16OF17_DATASETS[-1]
    reused = runner.STEP60_PARTIAL_16OF17_DATASETS[:-1]
    read_calls = 0
    shard_calls: list[tuple[str, bool]] = []
    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import aggregate as aggregate_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module
    from meddies_pii.eval_baseline.baseline import run as run_module
    from meddies_pii.evaluation import identity as identity_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "require_triton_c_compiler", lambda: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: FakeEvaluation())
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)
    monkeypatch.setattr(
        datasets_module,
        "load_eval_cell",
        lambda dataset, **_: FakeRows(runner.EVAL_EXPECTED_ROWS[dataset]),
    )
    monkeypatch.setattr(identity_module, "dataset_shard_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(
        run_module,
        "expected_matrix_shard_identities",
        lambda _evaluation, datasets: dict.fromkeys(datasets, identity),
    )

    def read_matrix(_root: str | Path, _model: str, datasets: Sequence[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal read_calls
        read_calls += 1
        cells = tuple(datasets)
        completed = reused if read_calls == 1 else cells
        return SimpleNamespace(
            missing_datasets=(missing,) if read_calls == 1 else (),
            rows_by_dataset={dataset: object() for dataset in completed},
            result_sha256_by_dataset=dict.fromkeys(completed, "1" * 64),
            shard_identities_by_dataset=dict.fromkeys(completed, identity),
            timing_by_dataset={},
        )

    monkeypatch.setattr(run_module, "read_matrix_results", read_matrix)

    def run_shard(*, spec: ShardSpec, force: bool, **_kwargs: object) -> SimpleNamespace:
        dataset = spec.dataset
        shard_calls.append((dataset, force))
        return SimpleNamespace(
            skipped=dataset in reused,
            output_path=tmp_path / f"{dataset}.jsonl",
        )

    monkeypatch.setattr(run_module, "run_shard", run_shard)
    monkeypatch.setattr(
        runner,
        "budget_allows_next_cell",
        lambda **_: (_ for _ in ()).throw(AssertionError("resume must not use stale full-cell admission")),
    )
    monkeypatch.setattr(
        aggregate_module,
        "aggregate_results",
        lambda *_args, **_kwargs: {
            "overall": {
                "rows": runner.STEP60_PARTIAL_16OF17_ROWS,
                "rows_per_second": 1.0,
            },
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)),
    )

    payload = runner.run_step60_partial_16of17.local(
        asdict(request),
        request.checkpoint_digest,
        _active_approval(runner, request, runner.STEP60_PARTIAL_16OF17_PURPOSE),
    )

    assert read_calls == 2
    assert shard_calls == [(dataset, False) for dataset in reused] + [(missing, False)]
    assert payload["resume"]["mode"] == "single_missing_cell_resume"
    assert payload["resume"]["reused_cells"] == list(reused)
    assert payload["resume"]["missing_cells"] == [missing]
    assert (runner.evaluation_generation_root(request, "f" * 64) / "resume.json").is_file()


def test_step60_partial_16of17_render_command_is_profile_bound() -> None:
    runner = _runner()
    request = _request(runner)
    command = runner.render_step60_partial_16of17_command(
        request,
        _approval(runner, request, runner.STEP60_PARTIAL_16OF17_PURPOSE),
    )

    assert f"--action {runner.STEP60_PARTIAL_16OF17_PURPOSE}" in command
    assert f"{runner.PROFILE_ENVIRONMENT_KEY}=diffusionllm" in command
    assert request.checkpoint_digest in command
    assert request.trajectory_digest in command


def test_single_missing_cell_resume_avoids_skipped_bootstrap_poisoning() -> None:
    runner = _runner()
    original = _request(runner)
    request = runner.BenchmarkRequest(
        original.repo_id,
        original.revision,
        original.artifact_path,
        original.checkpoint_digest,
        original.trajectory_digest,
        2.80,
        original.all_in_rate_usd_per_second,
        original.reserve_seconds_per_cell,
        original.profile,
    )
    completed = runner.STEP60_PARTIAL_16OF17_DATASETS[:-1]
    missing = (runner.STEP60_PARTIAL_16OF17_DATASETS[-1],)

    with pytest.raises(runner.BudgetRefusal, match="refuses next cell"):
        runner.budget_allows_next_cell(
            elapsed_seconds=0,
            next_dataset=missing[0],
            next_rows=runner.EVAL_EXPECTED_ROWS[missing[0]],
            observed_seconds_per_row=(
                runner.BOOTSTRAP_CELL_SECONDS / runner.EVAL_EXPECTED_ROWS[runner.BOOTSTRAP_DATASET],
            ),
            request=request,
        )

    resume = runner.single_missing_cell_resume_admission(
        scheduled_datasets=runner.STEP60_PARTIAL_16OF17_DATASETS,
        validated_completed_cells=completed,
        missing_cells=missing,
        elapsed_seconds=0,
        request=request,
    )
    assert resume == {
        "mode": "single_missing_cell_resume",
        "hard_ceiling_usd": 2.80,
        "reused_cells": list(completed),
        "missing_cells": list(missing),
        "persistence_reserve_seconds": 300,
        "progress_window_seconds": runner.SINGLE_MISSING_CELL_RESUME_PROGRESS_WINDOW_SECONDS,
    }
    assert (
        runner.single_missing_cell_resume_admission(
            scheduled_datasets=runner.STEP60_PARTIAL_16OF17_DATASETS,
            validated_completed_cells=completed[:-1],
            missing_cells=runner.STEP60_PARTIAL_16OF17_DATASETS[-2:],
            elapsed_seconds=0,
            request=request,
        )
        is None
    )
    too_small = runner.BenchmarkRequest(
        original.repo_id,
        original.revision,
        original.artifact_path,
        original.checkpoint_digest,
        original.trajectory_digest,
        0.1,
        original.all_in_rate_usd_per_second,
        original.reserve_seconds_per_cell,
        original.profile,
    )
    with pytest.raises(runner.BudgetRefusal, match="single-missing-cell resume"):
        runner.single_missing_cell_resume_admission(
            scheduled_datasets=runner.STEP60_PARTIAL_16OF17_DATASETS,
            validated_completed_cells=completed,
            missing_cells=missing,
            elapsed_seconds=0,
            request=too_small,
        )


@pytest.mark.parametrize("ceiling", [4.01, 500.0])
def test_request_rejects_ceiling_above_selected_profile_cap(ceiling: float) -> None:
    runner = _runner()
    request = _request(runner)
    with pytest.raises(ValueError, match="profile-pinned"):
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            ceiling,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            request.profile,
        )


def test_step60_baseline_render_command_starts_no_remote_work() -> None:
    runner = _runner()
    request = _request(runner)
    command = runner.render_step60_baseline_command(request, _approval(runner, request, runner.STEP60_BASELINE_PURPOSE))

    assert "--action step60_baseline" in command
    assert request.checkpoint_digest in command
    assert request.trajectory_digest in command


def test_step60_baseline_action_refuses_non_step60_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 100

    class FakeAdapter:
        def __init__(self, *_: object) -> None:
            msg = "step-60 gate must run before adapter loading"
            raise AssertionError(msg)

    class FakeVolume:
        def commit(self) -> None:
            pass

    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)

    with pytest.raises(RuntimeError, match="optimizer step 60"):
        runner.run_step60_baseline.local(
            asdict(request),
            request.checkpoint_digest,
            "f" * 64,
            _active_approval(runner, request, runner.STEP60_BASELINE_PURPOSE),
        )


def test_step60_baseline_refuses_to_rerun_unverified_challenge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 60

    class FakeAdapter:
        def __init__(self, *_: object) -> None:
            msg = "missing challenge evidence must stop before loading"
            raise AssertionError(msg)

    class FakeRows:
        def __init__(self, count: int) -> None:
            self.count = count

        def __len__(self) -> int:
            return self.count

    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module
    from meddies_pii.eval_baseline.baseline import run as run_module
    from meddies_pii.evaluation import identity as identity_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: SimpleNamespace(digest="f" * 64))
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)
    monkeypatch.setattr(
        datasets_module,
        "load_eval_cell",
        lambda dataset, **_: FakeRows(runner.EVAL_EXPECTED_ROWS[dataset]),
    )
    monkeypatch.setattr(identity_module, "dataset_shard_identity", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        run_module,
        "read_matrix_results",
        lambda *_args, **_kwargs: SimpleNamespace(
            missing_datasets=("v2-eval-challenge",),
            rows_by_dataset={"v2-eval": object()},
        ),
    )

    with pytest.raises(RuntimeError, match="refusing user-excluded rerun"):
        runner.run_step60_baseline.local(
            asdict(request),
            request.checkpoint_digest,
            "f" * 64,
            _active_approval(runner, request, runner.STEP60_BASELINE_PURPOSE),
        )


def test_step60_cpu_receipt_refuses_missing_challenge_before_any_a10g_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    request = _request(runner)
    calls = {"cpu": 0, "a10g": 0}

    class CpuReceiptFailure:
        def remote(self, _payload: dict[str, object]) -> dict[str, object]:
            calls["cpu"] += 1
            msg = (
                "step-60 baseline cannot prove existing v2-eval-challenge artifact identity; refusing user-excluded rerun"
            )
            raise RuntimeError(
                msg,
            )

    class A10GFunction:
        def remote(self, *_args: object) -> dict[str, object]:
            calls["a10g"] += 1
            msg = "A10G must not dispatch after a CPU gate failure"
            raise AssertionError(msg)

    monkeypatch.setenv(runner.PROFILE_ENVIRONMENT_KEY, request.profile)
    monkeypatch.setattr(runner, "step60_reuse_receipt", CpuReceiptFailure())
    monkeypatch.setattr(runner, "run_step60_baseline", A10GFunction())

    with pytest.raises(RuntimeError, match="refusing user-excluded rerun"):
        runner.main(
            action="step60_baseline",
            request_json=json.dumps(asdict(request)),
            approval_json=json.dumps(_approval(runner, request, runner.STEP60_BASELINE_PURPOSE)),
        )

    assert calls == {"cpu": 1, "a10g": 0}


def test_step60_reuse_receipt_rejects_missing_challenge_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 60

    class FakeRows:
        def __init__(self, count: int) -> None:
            self.count = count

        def __len__(self) -> int:
            return self.count

    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module
    from meddies_pii.eval_baseline.baseline import run as run_module
    from meddies_pii.evaluation import identity as identity_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: SimpleNamespace(digest="f" * 64))
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(
        datasets_module,
        "load_eval_cell",
        lambda dataset, **_: FakeRows(runner.EVAL_EXPECTED_ROWS[dataset]),
    )
    monkeypatch.setattr(identity_module, "dataset_shard_identity", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        run_module,
        "read_matrix_results",
        lambda *_args, **_kwargs: SimpleNamespace(missing_datasets=("v2-eval-challenge",), rows_by_dataset={}),
    )

    with pytest.raises(RuntimeError, match="refusing user-excluded rerun"):
        runner.step60_reuse_receipt.local(asdict(request))


def test_step60_reuse_receipt_is_bound_to_all_reused_result_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    body = runner._step60_reuse_receipt_body(
        request=request,
        optimizer_step=60,
        contract_digest="1" * 64,
        evaluation_digest="2" * 64,
        shard_identity_digest_by_cell={
            "v2-eval": "3" * 64,
            "v2-eval-challenge": "4" * 64,
        },
        result_sha256_by_cell={
            "v2-eval": "5" * 64,
            "v2-eval-challenge": "6" * 64,
        },
        scored_report={"overall": {"rows": 3_400}},
    )
    receipt = {**body, "receipt_digest": runner.canonical_sha256(body)}
    path = runner._step60_reuse_receipt_path(request, body["evaluation_contract_digest"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    assert (
        runner._require_step60_reuse_receipt(
            request=request,
            expected_receipt_digest=receipt["receipt_digest"],
            body=body,
        )
        == receipt
    )
    tampered = {**body, "rows_by_cell": {"v2-eval": 1_700, "v2-eval-challenge": 1}}
    with pytest.raises(RuntimeError, match="does not match checkpoint"):
        runner._require_step60_reuse_receipt(
            request=request,
            expected_receipt_digest=receipt["receipt_digest"],
            body=tampered,
        )


def test_step60_baseline_reuses_only_the_verified_v2_result_and_aggregates_17_cells(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 60

    class FakeAdapter:
        supported_labels = frozenset()
        truncated_documents = 0

        def __init__(self, *_: object) -> None:
            pass

        def load(self) -> None:
            pass

    class FakeRows:
        def __init__(self, count: int) -> None:
            self.count = count

        def __len__(self) -> int:
            return self.count

    class FakeEvaluation:
        digest = "f" * 64

    seen_force: list[bool] = []
    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import aggregate as aggregate_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module
    from meddies_pii.eval_baseline.baseline import run as run_module
    from meddies_pii.evaluation import identity as identity_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: FakeEvaluation())
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)
    identity = SimpleNamespace(digest="2" * 64)
    monkeypatch.setattr(identity_module, "dataset_shard_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(
        datasets_module,
        "load_eval_cell",
        lambda dataset, **_: FakeRows(runner.EVAL_EXPECTED_ROWS[dataset]),
    )
    monkeypatch.setattr(
        run_module,
        "run_shard",
        lambda **kwargs: (
            seen_force.append(kwargs["force"])
            or SimpleNamespace(
                skipped=kwargs["spec"].dataset == "v2-eval",
                output_path=tmp_path / f"{kwargs['spec'].dataset}.jsonl",
            )
        ),
    )
    monkeypatch.setattr(
        run_module,
        "expected_matrix_shard_identities",
        lambda _evaluation, datasets: {dataset: object() for dataset in datasets},
    )
    monkeypatch.setattr(
        run_module,
        "read_matrix_results",
        lambda _root, _model, datasets, **_kwargs: SimpleNamespace(
            missing_datasets=(),
            rows_by_dataset={dataset: object() for dataset in datasets},
            result_sha256_by_dataset=dict.fromkeys(datasets, "1" * 64),
            shard_identities_by_dataset=dict.fromkeys(datasets, identity),
            timing_by_dataset={},
        ),
    )
    monkeypatch.setattr(
        aggregate_module,
        "aggregate_results",
        lambda *_args, **_kwargs: {
            "fixture": {
                "sha256": "1ccd86a42833d45ce20c9d222e8891c6cd334e07ca774ba5ab64debfd5378cb5",
                "full9_gold_spans": runner.FULL_GOLD_SPANS,
            },
            "overall": {"rows": runner.FULL_ROWS, "rows_per_second": 1.0},
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)),
    )

    monkeypatch.setattr(runner, "_require_step60_reuse_receipt", lambda **_: None)
    monkeypatch.setattr(runner, "budget_allows_next_cell", lambda **_: 1.0)
    payload = runner.run_step60_baseline.local(
        asdict(request),
        request.checkpoint_digest,
        "f" * 64,
        _active_approval(runner, request, runner.STEP60_BASELINE_PURPOSE),
    )

    assert len(seen_force) == 15
    assert seen_force == [False] * 15
    assert payload["status"] == "step60_baseline_aggregate_terminal"
    assert payload["coverage"] == runner.step60_baseline_coverage_report(v2_eval_reused=True)


def test_cadence_patience_stops_after_two_consecutive_f1_and_recall_degradations() -> None:
    runner = _runner()
    decision = runner.cadence_early_stop_decision([
        _cadence_result(50, checkpoint="a", f1=0.7000, recall=0.7100),
        _cadence_result(100, checkpoint="b", f1=0.6950, recall=0.7100),
        _cadence_result(150, checkpoint="c", f1=0.6940, recall=0.7000),
    ])
    assert decision["status"] == "early_stop"
    assert decision["best_checkpoint"]["optimizer_step"] == 50
    assert decision["trigger_checkpoint"]["optimizer_step"] == 150
    assert decision["consecutive_degradations"] == 2


def test_cadence_patience_continues_when_recall_recovers() -> None:
    runner = _runner()
    decision = runner.cadence_early_stop_decision([
        _cadence_result(50, checkpoint="a", f1=0.7000, recall=0.7100),
        _cadence_result(100, checkpoint="b", f1=0.6950, recall=0.7200),
        _cadence_result(150, checkpoint="c", f1=0.6940, recall=0.7000),
    ])
    assert decision["status"] == "continue"
    assert decision["consecutive_degradations"] == 1


def test_cadence_history_allows_checkpoint_specific_evaluation_digests() -> None:
    runner = _runner()
    decision = runner.cadence_early_stop_decision([
        _cadence_result(
            50,
            checkpoint="a",
            f1=0.7000,
            recall=0.7100,
            evaluation_contract="1" * 64,
        ),
        _cadence_result(
            100,
            checkpoint="b",
            f1=0.6980,
            recall=0.7200,
            evaluation_contract="2" * 64,
        ),
    ])
    assert decision["status"] == "continue"


def test_comparison_contract_excludes_checkpoint_specific_model_identity() -> None:
    runner = _runner()

    class FakeEvaluation:
        def __init__(self, checkpoint_digest: str) -> None:
            self.checkpoint_digest = checkpoint_digest

        def to_payload(self) -> dict[str, object]:
            return {
                "schema_version": 1,
                "model": {"checkpoint_digest": self.checkpoint_digest},
                "vendor_inference_source": {"sha256": "1" * 64},
                "scorer_contract": {"sha256": "2" * 64},
            }

    first = runner.comparison_contract_digest(FakeEvaluation("a" * 64), fixture_digest="3" * 64)
    second = runner.comparison_contract_digest(FakeEvaluation("b" * 64), fixture_digest="3" * 64)
    assert first == second


def test_cadence_receipt_rejects_history_and_decision_digest_tampering() -> None:
    runner = _runner()
    receipt = runner.build_cadence_decision_receipt([_cadence_result(100, checkpoint="a", f1=0.7, recall=0.7)])
    assert runner.verify_cadence_decision_receipt(receipt) == receipt

    history_tampered = json.loads(json.dumps(receipt))
    history_tampered["history"][0]["fixed_nine_exact_typed"]["f1"] = 0.2
    with pytest.raises(ValueError, match="digest"):
        runner.verify_cadence_decision_receipt(history_tampered)

    decision_tampered = json.loads(json.dumps(receipt))
    decision_tampered["decision"]["status"] = "early_stop"
    with pytest.raises(ValueError, match="decision"):
        runner.verify_cadence_decision_receipt(decision_tampered)


def test_durable_cadence_history_isolated_by_comparison_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    trajectory_digest = "d" * 64
    legacy_comparison_digest = "f" * 64
    current_comparison_digest = "1" * 64
    history = [
        _cadence_result(
            100,
            checkpoint="a",
            f1=0.7,
            recall=0.7,
            trajectory=trajectory_digest,
            comparison_contract=legacy_comparison_digest,
        ),
    ]
    receipt = runner.build_cadence_decision_receipt(history)
    legacy_root = tmp_path / "trajectories" / trajectory_digest / "internal-control"
    runner._atomic_json(legacy_root / "history.json", {"history": history})
    runner._atomic_json(legacy_root / "decision-receipt.json", receipt)
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))

    assert runner._read_legacy_durable_cadence_history(trajectory_digest) == history
    assert runner._read_durable_cadence_history(trajectory_digest, current_comparison_digest) is None
    assert (
        runner._require_prior_cadence_history(
            [],
            durable_history=None,
            trajectory_digest=trajectory_digest,
            stable_comparison_digest=current_comparison_digest,
            current_step=100,
        )
        == []
    )
    with pytest.raises(ValueError, match="does not match the current trajectory"):
        runner._require_prior_cadence_history(
            history,
            durable_history=None,
            trajectory_digest=trajectory_digest,
            stable_comparison_digest=current_comparison_digest,
            current_step=150,
        )


def test_persisted_cadence_receipt_cannot_cross_comparison_generations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    trajectory_digest = "d" * 64
    legacy_comparison_digest = "f" * 64
    current_comparison_digest = "1" * 64
    legacy_history = [
        _cadence_result(
            100,
            checkpoint="a",
            f1=0.7,
            recall=0.7,
            trajectory=trajectory_digest,
            comparison_contract=legacy_comparison_digest,
        ),
    ]
    current_history = [
        _cadence_result(
            100,
            checkpoint="b",
            f1=0.71,
            recall=0.71,
            trajectory=trajectory_digest,
            comparison_contract=current_comparison_digest,
        ),
    ]
    legacy_receipt = runner.build_cadence_decision_receipt(legacy_history)
    current_receipt = runner.build_cadence_decision_receipt(current_history)
    legacy_root = tmp_path / "trajectories" / trajectory_digest / "internal-control"
    runner._atomic_json(legacy_root / "history.json", {"history": legacy_history})
    runner._atomic_json(legacy_root / "decision-receipt.json", legacy_receipt)

    class FakeVolume:
        def commit(self) -> None:
            pass

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    runner._persist_cadence_decision(
        trajectory_digest=trajectory_digest,
        stable_comparison_digest=current_comparison_digest,
        receipt=current_receipt,
    )

    assert runner._read_durable_cadence_history(trajectory_digest, current_comparison_digest) == current_history
    assert runner._read_legacy_durable_cadence_history(trajectory_digest) == legacy_history
    with pytest.raises(ValueError, match="does not match its durable generation"):
        runner._persist_cadence_decision(
            trajectory_digest=trajectory_digest,
            stable_comparison_digest=legacy_comparison_digest,
            receipt=current_receipt,
        )


def test_two_new_comparison_generations_keep_independent_durable_histories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    trajectory_digest = "d" * 64
    first_comparison_digest = "1" * 64
    second_comparison_digest = "2" * 64
    first_history = [
        _cadence_result(
            100,
            checkpoint="a",
            f1=0.70,
            recall=0.70,
            trajectory=trajectory_digest,
            comparison_contract=first_comparison_digest,
        ),
    ]
    second_history = [
        _cadence_result(
            100,
            checkpoint="b",
            f1=0.71,
            recall=0.71,
            trajectory=trajectory_digest,
            comparison_contract=second_comparison_digest,
        ),
    ]

    class FakeVolume:
        def commit(self) -> None:
            pass

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    runner._persist_cadence_decision(
        trajectory_digest=trajectory_digest,
        stable_comparison_digest=first_comparison_digest,
        receipt=runner.build_cadence_decision_receipt(first_history),
    )
    runner._persist_cadence_decision(
        trajectory_digest=trajectory_digest,
        stable_comparison_digest=second_comparison_digest,
        receipt=runner.build_cadence_decision_receipt(second_history),
    )

    first_path = runner._trajectory_history_path(trajectory_digest, first_comparison_digest)
    second_path = runner._trajectory_history_path(trajectory_digest, second_comparison_digest)
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()
    assert runner._read_durable_cadence_history(trajectory_digest, first_comparison_digest) == first_history
    assert runner._read_durable_cadence_history(trajectory_digest, second_comparison_digest) == second_history


def test_internal_control_render_requires_explicit_prior_history_json() -> None:
    runner = _runner()
    request = _request(runner)
    command = runner.render_internal_control_command(request, "[]", _approval(runner, request, "internal_control"))
    assert "--action internal_control" in command
    assert "--prior-history-json '[]'" in command
    with pytest.raises(ValueError, match="valid"):
        runner.render_internal_control_command(
            _request(runner),
            "not-json",
            _approval(runner, _request(runner), "internal_control"),
        )


def test_internal_control_entrypoint_fails_closed_without_prior_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    request = _request(runner)
    monkeypatch.setenv(runner.PROFILE_ENVIRONMENT_KEY, request.profile)
    with pytest.raises(RuntimeError, match="prior_history_json is required"):
        runner.main(
            action="internal_control",
            request_json=json.dumps(asdict(request)),
            approval_json=json.dumps(_approval(runner, request, "internal_control")),
        )


@pytest.mark.parametrize(
    ("prior_history", "stable_comparison_digest", "message"),
    [
        (
            [_cadence_result(100, checkpoint="a", f1=0.7, recall=0.7)],
            "f" * 64,
            "rollback, omission, or reordering",
        ),
        (
            [
                _cadence_result(100, checkpoint="a", f1=0.7, recall=0.7),
                _cadence_result(150, checkpoint="b", f1=0.69, recall=0.69),
            ],
            "1" * 64,
            "does not match the current trajectory",
        ),
    ],
)
def test_later_cross_profile_control_rejects_omission_and_comparison_drift(
    prior_history: list[dict[str, object]],
    stable_comparison_digest: str,
    message: str,
) -> None:
    runner = _runner()
    with pytest.raises(ValueError, match=message):
        runner._require_prior_cadence_history(
            prior_history,
            durable_history=None,
            trajectory_digest="d" * 64,
            stable_comparison_digest=stable_comparison_digest,
            current_step=200,
        )


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ([], "at least one"),
        (
            [_cadence_result(50, checkpoint="a", f1=0.7, recall=0.7) | {"extra": 1}],
            "missing or unexpected",
        ),
        (
            [
                _cadence_result(50, checkpoint="a", f1=0.7, recall=0.7),
                _cadence_result(150, checkpoint="b", f1=0.69, recall=0.7),
            ],
            "missing or out of order",
        ),
        (
            [
                _cadence_result(100, checkpoint="a", f1=0.7, recall=0.7),
                _cadence_result(50, checkpoint="b", f1=0.69, recall=0.7),
            ],
            "missing or out of order",
        ),
        (
            [
                _cadence_result(50, checkpoint="a", f1=0.7, recall=0.7),
                _cadence_result(
                    100,
                    checkpoint="b",
                    f1=0.69,
                    recall=0.7,
                    trajectory="f" * 64,
                ),
            ],
            "mixes trajectory or comparison identities",
        ),
        (
            [
                _cadence_result(50, checkpoint="a", f1=0.7, recall=0.7),
                _cadence_result(
                    100,
                    checkpoint="b",
                    f1=0.69,
                    recall=0.7,
                    comparison_contract="3" * 64,
                ),
            ],
            "mixes trajectory or comparison identities",
        ),
    ],
)
def test_cadence_patience_rejects_incomplete_or_incompatible_history(
    history: list[dict[str, object]],
    message: str,
) -> None:
    runner = _runner()
    with pytest.raises(ValueError, match=message):
        runner.cadence_early_stop_decision(history)


def test_full_benchmark_contract_is_exact_17_cell_schedule_and_not_internal_control() -> None:
    runner = _runner()
    request = _request(runner)
    contract = runner.benchmark_contract(request, purpose="full_benchmark")

    runner.require_full_matrix(contract)
    assert tuple(contract["datasets"]) == EVAL_DATASETS
    assert contract["rows_by_cell"] == EVAL_EXPECTED_ROWS
    assert contract["total_rows"] == 263_785
    assert runner.benchmark_contract(request, purpose="internal_control")["total_rows"] == 1_700


def test_evaluation_generation_root_separates_source_generations() -> None:
    runner = _runner()
    request = _request(runner)

    old = runner.evaluation_generation_root(request, "0" * 64)
    current = runner.evaluation_generation_root(request, "f" * 64)

    assert old != current
    assert old.parent == current.parent
    assert old.name == "0" * 64
    assert current.name == "f" * 64


def test_checkpoint_specific_root_prevents_done_file_collision_between_checkpoints() -> None:
    runner = _runner()
    assert runner.checkpoint_output_root(_request(runner, "a" * 64)) != runner.checkpoint_output_root(
        _request(runner, "c" * 64),
    )
    assert runner.checkpoint_model_name(_request(runner, "a" * 64)) != runner.checkpoint_model_name(
        _request(runner, "c" * 64),
    )


def test_budget_refuses_next_cell_before_dispatch() -> None:
    runner = _runner()
    with pytest.raises(RuntimeError, match="refuses next cell"):
        runner.budget_allows_next_cell(
            elapsed_seconds=8_000,
            next_dataset="v2-eval",
            next_rows=1_700,
            observed_seconds_per_row=(),
            request=_request(runner),
        )


def test_budget_refuses_a_large_next_cell_from_conservative_observation() -> None:
    runner = _runner()
    request = _request(runner)
    constrained = runner.BenchmarkRequest(
        request.repo_id,
        request.revision,
        request.artifact_path,
        request.checkpoint_digest,
        request.trajectory_digest,
        0.5,
        runner.A10G_ALL_IN_RATE_USD_PER_SECOND,
        request.reserve_seconds_per_cell,
        request.profile,
    )
    with pytest.raises(RuntimeError, match="refuses next cell"):
        runner.budget_allows_next_cell(
            elapsed_seconds=10,
            next_dataset="nemotron_en",
            next_rows=99_892,
            observed_seconds_per_row=(0.1, 0.08),
            request=constrained,
        )


def test_request_requires_its_workspace_pinned_budget_ceiling() -> None:
    runner = _runner()
    request = _request(runner)
    assert runner.PROFILE_ALL_IN_CEILING_USD == {
        "diffusionllm": 4.0,
        "huyhoang0411ha": 8.5,
        "meddies-pii": 3.8,
        "huyhoang041100": 3.0,
        "meddies-run": 2.9,
    }
    for profile, ceiling in runner.PROFILE_ALL_IN_CEILING_USD.items():
        pinned = runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            ceiling,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            profile,
        )
        assert pinned.max_all_in_usd == ceiling
        remaining_attempt = runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            ceiling - 0.1,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            profile,
        )
        assert remaining_attempt.max_all_in_usd == ceiling - 0.1
    with pytest.raises(ValueError, match="profile-pinned"):
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            500.0,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            request.profile,
        )


def test_bootstrap_is_explicit_and_bounded_to_v2_eval() -> None:
    runner = _runner()
    assert (
        runner.predict_next_cell_seconds(next_dataset="v2-eval", next_rows=1_700, observed_seconds_per_row=())
        == runner.BOOTSTRAP_CELL_SECONDS
    )
    with pytest.raises(RuntimeError, match="bounded bootstrap"):
        runner.predict_next_cell_seconds(next_dataset="nemotron_en", next_rows=99_892, observed_seconds_per_row=())


def test_full_contract_rejects_row_and_revision_drift() -> None:
    runner = _runner()
    contract = runner.benchmark_contract(_request(runner), purpose="full_benchmark")
    broken_rows = {
        **contract,
        "rows_by_cell": {**contract["rows_by_cell"], "v2-eval": 1},
    }
    with pytest.raises(RuntimeError, match="row schedule"):
        runner.require_full_matrix(broken_rows)
    broken_revision = {**contract, "v2_revision": "0" * 40}
    with pytest.raises(RuntimeError, match="revision drifted"):
        runner.require_full_matrix(broken_revision)


def test_cpu_and_gpu_use_the_same_nested_checkpoint_artifact_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request = _request(runner)
    monkeypatch.setattr(runner, "CHECKPOINT_CACHE_MOUNT", str(tmp_path))
    artifact_root = tmp_path / request.checkpoint_digest / request.artifact_path
    artifact_root.mkdir(parents=True)
    assert runner._checkpoint_local_root(request) == artifact_root.resolve()


def test_rendered_commands_preserve_immutable_receipt_identity() -> None:
    runner = _runner()
    request = _request(runner)
    receipt = runner.render_cpu_receipt_command(request)
    benchmark = runner.render_full_benchmark_command(request, _approval(runner, request, "full_benchmark"))
    for command in (receipt, benchmark):
        assert request.revision in command
        assert request.artifact_path in command
        assert request.checkpoint_digest in command
        assert request.trajectory_digest in command
    assert "--action receipt" in receipt
    assert "--action full_benchmark" in benchmark


def test_request_rejects_a_moving_revision_or_escaping_artifact_path() -> None:
    runner = _runner()
    request = _request(runner)
    with pytest.raises(ValueError, match="immutable Git SHA"):
        runner.BenchmarkRequest(
            request.repo_id,
            "main",
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            request.max_all_in_usd,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            request.profile,
        )
    with pytest.raises(ValueError, match="artifact path"):
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            "../escape",
            request.checkpoint_digest,
            request.trajectory_digest,
            request.max_all_in_usd,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            request.profile,
        )


def test_checkpoint_namespace_rejects_stale_done_from_another_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from meddies_pii.eval_baseline.baseline.run import ShardSpec, shard_done_path

    runner = _runner()
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    old_request = _request(runner, "a" * 64)
    request = _request(runner, "c" * 64)
    old_spec = ShardSpec(model=runner.checkpoint_model_name(old_request), dataset="v2-eval", shard="full")
    new_spec = ShardSpec(model=runner.checkpoint_model_name(request), dataset="v2-eval", shard="full")
    stale_done = shard_done_path(runner.checkpoint_output_root(old_request), old_spec)
    stale_done.parent.mkdir(parents=True)
    stale_done.write_text("done\n", encoding="utf-8")
    assert not shard_done_path(runner.checkpoint_output_root(request), new_spec).exists()


def test_full_contract_rejects_a_partial_matrix() -> None:
    runner = _runner()
    contract = runner.benchmark_contract(_request(runner), purpose="full_benchmark")
    partial = {**contract, "datasets": list(contract["datasets"][:-1])}
    with pytest.raises(RuntimeError, match="exact 17-cell"):
        runner.require_full_matrix(partial)


def test_request_requires_frozen_trajectory_and_pinned_a10g_rate() -> None:
    runner = _runner()
    request = _request(runner)
    with pytest.raises(ValueError, match="trajectory digest"):
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            "not-a-digest",
            request.max_all_in_usd,
            runner.A10G_ALL_IN_RATE_USD_PER_SECOND,
            request.reserve_seconds_per_cell,
            request.profile,
        )
    with pytest.raises(ValueError, match="A10G rate"):
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            request.max_all_in_usd,
            0.0001,
            request.reserve_seconds_per_cell,
            request.profile,
        )


def test_trajectory_mismatch_is_refused_before_evaluation() -> None:
    runner = _runner()
    with pytest.raises(RuntimeError, match="does not match"):
        runner.require_expected_trajectory(_request(runner), "e" * 64)


def test_profile_must_match_before_modal_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    request = _request(runner)
    monkeypatch.delenv(runner.PROFILE_ENVIRONMENT_KEY, raising=False)
    with pytest.raises(RuntimeError, match="does not match"):
        runner.require_local_profile(request)
    monkeypatch.setenv(runner.PROFILE_ENVIRONMENT_KEY, request.profile)
    runner.require_local_profile(request)


def test_progress_is_printed_and_committed_during_a_long_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setattr(runner, "benchmark_volume", volume)
    request = _request(runner)
    runner.publish_progress(
        root=tmp_path,
        request=request,
        dataset="nemotron_en",
        rows_done=runner.PROGRESS_EVERY,
        rows_total=99_892,
    )
    payload = (tmp_path / "progress" / "nemotron_en.json").read_text(encoding="utf-8")
    assert '"rows_done": 2000' in payload
    assert volume.commits == 1
    assert "PII350_BENCHMARK_PROGRESS::" in capsys.readouterr().out


def _write_cpu_receipts(runner: ModuleType, request: BenchmarkRequest, root: Path) -> None:
    snapshot = root / "huggingface" / "hub" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    preflight_body = {
        "schema_version": 1,
        "profile": request.profile,
        "offline_cache": {
            "base_model": {
                "id": "LiquidAI/LFM2.5-Encoder-350M-PII-Detector",
                "revision": "b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2",
                "snapshot_path": str(snapshot),
                "tokenizer_source": "same_pinned_base_snapshot",
            },
            "datasets": {
                "cache_path": "/cache/huggingface/datasets",
                "v2_revision": runner.V2_DATASET_REVISION,
                "external_revision": runner.EXTERNAL_DATASET_REVISION,
                "rows_by_cell": runner.EVAL_EXPECTED_ROWS,
                "total_rows": runner.FULL_ROWS,
            },
        },
    }
    preflight = {
        **preflight_body,
        "receipt_digest": runner.canonical_sha256(preflight_body),
    }
    preflight_path = runner._cache_preflight_receipt_path(request.profile)
    preflight_path.parent.mkdir(parents=True)
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    checkpoint_receipt = {
        "checkpoint_digest": request.checkpoint_digest,
        "trajectory_digest": request.trajectory_digest,
        "revision": request.revision,
        "profile": request.profile,
        "cache_preflight_receipt_digest": preflight["receipt_digest"],
    }
    receipt_path = runner._receipt_path(request)
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(checkpoint_receipt), encoding="utf-8")


def test_gpu_refuses_without_both_cpu_receipts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request = _request(runner)
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    with pytest.raises(RuntimeError, match="checkpoint receipt"):
        runner.enable_offline_runtime(request)


def test_cpu_receipts_bind_base_dataset_revisions_and_full_matrix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request = _request(runner)
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    _write_cpu_receipts(runner, request, tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    runner.enable_offline_runtime(request)
    assert runner._cache_preflight_receipt(request.profile)["offline_cache"]["datasets"]["total_rows"] == 263_785
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_cache_preflight_command_is_profile_bound_and_has_no_checkpoint_input() -> None:
    runner = _runner()
    command = runner.render_cache_preflight_command("diffusionllm")
    assert "--action cache_preflight --profile diffusionllm" in command
    assert "request-json" not in command
    with pytest.raises(ValueError, match="approved evaluation"):
        runner.render_cache_preflight_command("unknown")


def test_request_requires_minimum_persistence_reserve_and_contract_records_it() -> None:
    runner = _runner()
    request = _request(runner)
    with pytest.raises(ValueError, match="minimum persistence reserve"):
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            request.max_all_in_usd,
            request.all_in_rate_usd_per_second,
            runner.MINIMUM_PERSISTENCE_RESERVE_SECONDS - 1,
            request.profile,
        )
    contract = runner.benchmark_contract(request, purpose="full_benchmark")
    assert contract["budget"]["persistence_reserve_seconds"] == 300
    assert contract["budget"]["minimum_persistence_reserve_seconds"] == 300


def test_budget_ceiling_guard_stops_at_or_above_reserve_projection() -> None:
    runner = _runner()
    request = _request(runner)
    elapsed = request.max_all_in_usd / runner.A10G_ALL_IN_RATE_USD_PER_SECOND - request.reserve_seconds_per_cell
    assert runner.budget_ceiling_reached(elapsed_seconds=elapsed, request=request)


def test_budget_stop_is_durable_and_structured_inside_a_long_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setattr(runner, "benchmark_volume", volume)
    request = _request(runner)
    payload = runner.publish_budget_stop(
        root=tmp_path,
        request=request,
        dataset="nemotron_en",
        rows_done=2_000,
        rows_total=99_892,
        elapsed_seconds=8_100,
    )
    assert payload["status"] == "budget_stop"
    assert (tmp_path / "budget_stop.json").is_file()
    assert (tmp_path / "state.budget_stop.json").is_file()
    assert volume.commits == 1
    assert "PII350_BENCHMARK_BUDGET_STOP::" in capsys.readouterr().out


def test_full_executor_persists_pre_dispatch_budget_stop_not_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    request = _request(runner)
    constrained = request

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = constrained.trajectory_digest

    class FakeAdapter:
        supported_labels = frozenset()

        def __init__(self, verified: FakeVerified) -> None:
            self.verified = verified

        def load(self) -> None:
            pass

    import meddies_pii.eval_baseline.adapters.pii350_checkpoint as adapter_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: SimpleNamespace(digest="f" * 64))
    monkeypatch.setattr(runner, "BOOTSTRAP_CELL_SECONDS", 10_000.0)
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)

    with pytest.raises(runner.BudgetStop, match="before dispatch"):
        runner.run_full_benchmark.local(
            asdict(constrained),
            constrained.checkpoint_digest,
            _active_approval(runner, constrained, "full_benchmark"),
        )

    root = runner.evaluation_generation_root(constrained, "f" * 64)
    budget_stop = json.loads((root / "budget_stop.json").read_text(encoding="utf-8"))
    assert budget_stop["dataset"] == "v2-eval"
    assert budget_stop["rows_done"] == 0
    assert budget_stop["rows_total"] == 1_700
    assert budget_stop["prior_completed_cell"] is None
    assert budget_stop["prior_progress"] is None
    assert budget_stop["predicted_next_cell_seconds"] == runner.BOOTSTRAP_CELL_SECONDS
    assert budget_stop["projected_all_in_cost_usd"] > constrained.max_all_in_usd
    assert not (root / "failure.json").exists()
    assert json.loads((root / "state.budget_stop.json").read_text(encoding="utf-8"))["status"] == "budget_stop"
    assert "PII350_BENCHMARK_BUDGET_STOP::" in capsys.readouterr().out


# reason: Fixed-nine metrics, shard/result digests, receipt, history, and early stop form one provenance chain.
def test_internal_control_returns_fixed_nine_metrics_bound_to_verified_shard(  # ruff: ignore[too-many-statements]
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request = _request(runner)

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 100

    adapter_loads: list[object] = []

    class FakeAdapter:
        supported_labels = frozenset()

        def __init__(self, verified: FakeVerified) -> None:
            self.verified = verified

        def load(self) -> None:
            adapter_loads.append(self)

    class FakeEvaluation:
        def __init__(self, checkpoint_digest: str) -> None:
            self.digest = checkpoint_digest
            self.checkpoint_digest = checkpoint_digest

        def to_payload(self) -> dict[str, object]:
            return {
                "schema_version": 1,
                "model": {"checkpoint_digest": self.checkpoint_digest},
                "scorer_contract": {"sha256": "4" * 64},
            }

    identity = SimpleNamespace(
        digest="1" * 64,
        fixture_identity=SimpleNamespace(digest="3" * 64),
    )
    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import aggregate as aggregate_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module
    from meddies_pii.eval_baseline.baseline import run as run_module
    from meddies_pii.evaluation import identity as identity_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    legacy_result_path = runner.checkpoint_output_root(request) / "internal_control" / "result.jsonl"
    legacy_result_path.parent.mkdir(parents=True)
    legacy_result_path.write_bytes(b"legacy evaluator result")
    current_root = runner.evaluation_generation_root(request, request.checkpoint_digest) / "internal_control"
    current_result_path = current_root / "result.jsonl"
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(
        runner,
        "_evaluation_contract",
        lambda run_request, *_: FakeEvaluation(run_request.checkpoint_digest),
    )
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)
    monkeypatch.setattr(datasets_module, "load_eval_cell", lambda *_args, **_kwargs: [object()] * 1_700)

    def run_current_generation_shard(**kwargs: object) -> SimpleNamespace:
        output_root = kwargs["output_root"]
        assert isinstance(output_root, Path)
        output_path = output_root / "result.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"current evaluator result")
        return SimpleNamespace(
            skipped=False,
            rows_in=1_700,
            spans_out=42,
            output_path=output_path,
        )

    monkeypatch.setattr(run_module, "run_shard", run_current_generation_shard)
    monkeypatch.setattr(identity_module, "dataset_shard_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(
        run_module,
        "read_matrix_results",
        lambda *_args, **_kwargs: SimpleNamespace(
            missing_datasets=(),
            rows_by_dataset={"v2-eval": object()},
            result_sha256_by_dataset={"v2-eval": "2" * 64},
            shard_identities_by_dataset={"v2-eval": identity},
            timing_by_dataset={"v2-eval": {"elapsed_seconds": 1.0}},
        ),
    )
    metrics = {"precision": 0.71, "recall": 0.61, "f1": 0.66}

    monkeypatch.setattr(
        aggregate_module,
        "aggregate_results",
        lambda *_args, **_kwargs: {
            "overall": {
                "rows": 1_700,
                "full9_exact_precision": metrics["precision"],
                "full9_exact_recall": metrics["recall"],
                "full9_exact_f1": metrics["f1"],
            },
        },
    )

    payload = runner.run_internal_control.local(
        asdict(request),
        request.checkpoint_digest,
        [],
        _active_approval(runner, request, "internal_control"),
    )

    assert payload["rows"] == 1_700
    assert payload["optimizer_step"] == 100
    assert payload["evaluation_contract_digest"] == "a" * 64
    assert payload["comparison_contract_digest"] == runner.comparison_contract_digest(
        FakeEvaluation(request.checkpoint_digest),
        fixture_digest="3" * 64,
    )
    assert payload["dataset_shard_identity_digest"] == "1" * 64
    assert payload["result_sha256"] == "2" * 64
    assert payload["fixed_nine_exact_typed"] == {
        "precision": 0.71,
        "recall": 0.61,
        "f1": 0.66,
    }
    assert payload["status"] == "continue"
    assert payload["skipped"] is False
    assert len(adapter_loads) == 1
    assert legacy_result_path.read_bytes() == b"legacy evaluator result"
    assert current_result_path.read_bytes() == b"current evaluator result"
    assert Path(payload["result_path"]) == current_result_path
    assert payload["decision_receipt"] == runner.verify_cadence_decision_receipt(payload["decision_receipt"])
    durable_history_path = runner._trajectory_history_path(
        request.trajectory_digest,
        payload["comparison_contract_digest"],
    )
    assert durable_history_path.is_file()

    request_150 = runner.BenchmarkRequest(
        request.repo_id,
        request.revision,
        request.artifact_path,
        "b" * 64,
        request.trajectory_digest,
        runner.PROFILE_ALL_IN_CEILING_USD["meddies-pii"],
        request.all_in_rate_usd_per_second,
        request.reserve_seconds_per_cell,
        "meddies-pii",
    )
    cross_profile_volume = tmp_path / "meddies-pii-workspace"
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(cross_profile_volume))
    FakeVerified.optimizer_step = 150
    metrics.update({"precision": 0.70, "recall": 0.60, "f1": 0.65})
    payload_150 = runner.run_internal_control.local(
        asdict(request_150),
        request_150.checkpoint_digest,
        payload["decision_receipt"]["history"],
        _active_approval(runner, request_150, "internal_control", "d" * 64),
    )
    assert payload_150["status"] == "continue"
    assert (
        runner._trajectory_history_path(request.trajectory_digest, payload_150["comparison_contract_digest"]).parent
        == cross_profile_volume
        / "trajectories"
        / request.trajectory_digest
        / "internal-control"
        / "generations"
        / payload_150["comparison_contract_digest"]
    )
    assert payload_150["evaluation_contract_digest"] == "b" * 64
    assert payload_150["decision_receipt"]["history_digest"] == runner.canonical_sha256(
        payload_150["decision_receipt"]["history"],
    )
    with pytest.raises(RuntimeError, match="disagrees with durable"):
        runner._require_prior_cadence_history(
            payload["decision_receipt"]["history"],
            durable_history=runner._read_durable_cadence_history(
                request.trajectory_digest,
                payload_150["comparison_contract_digest"],
            ),
            trajectory_digest=request.trajectory_digest,
            stable_comparison_digest=payload_150["comparison_contract_digest"],
            current_step=200,
        )

    request_200 = runner.BenchmarkRequest(
        request.repo_id,
        request.revision,
        request.artifact_path,
        "c" * 64,
        request.trajectory_digest,
        runner.PROFILE_ALL_IN_CEILING_USD["meddies-pii"],
        request.all_in_rate_usd_per_second,
        request.reserve_seconds_per_cell,
        "meddies-pii",
    )
    FakeVerified.optimizer_step = 200
    metrics.update({"precision": 0.69, "recall": 0.59, "f1": 0.64})
    payload_200 = runner.run_internal_control.local(
        asdict(request_200),
        request_200.checkpoint_digest,
        payload_150["decision_receipt"]["history"],
        _active_approval(runner, request_200, "internal_control", "c" * 64),
    )
    assert payload_200["status"] == "early_stop"
    assert payload_200["decision_receipt"]["status"] == "early_stop"
    assert payload_200["decision_receipt"]["last_evaluated_optimizer_step"] == 200


def test_cache_preflight_progress_is_printed_and_committed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setattr(runner, "benchmark_volume", volume)
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    runner.publish_cache_preflight_progress(
        profile="diffusionllm",
        stage="dataset:nemotron_en",
        cells_done=17,
        rows_done=263_785,
    )
    assert (tmp_path / "cache-preflight" / "diffusionllm" / "progress.json").is_file()
    assert volume.commits == 1
    assert "PII350_CACHE_PREFLIGHT_PROGRESS::" in capsys.readouterr().out


def _fill_approval(
    runner: ModuleType,
    request: BenchmarkRequest,
    cells: Sequence[str],
    nonce: str = "e" * 64,
) -> dict[str, Any]:
    """Sign a real fill approval so the cell list is covered by the signature."""
    private_key = Ed25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    )
    # reason: these tests pin the imported runner to a test-only issuer key by rebinding module globals.
    # reason: `ModuleType` declares no such attribute, so ty cannot type an assignment to one; reads through
    # reason: the module still resolve because `ModuleType.__getattr__` returns `Any`.
    runner.APPROVAL_PUBLIC_KEY_HEX = public_key_hex  # ty: ignore[unresolved-attribute]
    runner.APPROVAL_ISSUER_KEY_ID = (  # ty: ignore[unresolved-attribute]
        f"ed25519:{runner.canonical_sha256({'public_key_hex': public_key_hex})}"
    )
    body = runner._approval_body(
        request,
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        launch_nonce=nonce,
        external_ledger_approval_id="external-ledger-fill-approval",
        modeled_prior_attempts=(runner.MODELED_PRIOR_DIFFUSION_ATTEMPTS if request.profile == "diffusionllm" else ()),
        cells=cells,
    )
    return {
        **body,
        "approval_digest": runner.canonical_sha256(body),
        "signature": base64.b64encode(
            private_key.sign(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()),
        ).decode(),
    }


def test_fill_cell_list_is_validated_and_canonically_ordered() -> None:
    runner = _runner()
    assert runner.require_fill_cells(["gretel_en", "v2-eval"]) == (
        "v2-eval",
        "gretel_en",
    )
    assert runner._parse_cells_argument(" gretel_en , v2-eval ") == (
        "v2-eval",
        "gretel_en",
    )
    with pytest.raises(ValueError, match="at least one cell"):
        runner.require_fill_cells([])
    with pytest.raises(ValueError, match="unknown cells"):
        runner.require_fill_cells(["v2-eval", "not_a_cell"])
    with pytest.raises(ValueError, match="repeats a cell"):
        runner.require_fill_cells(["v2-eval", "v2-eval"])
    with pytest.raises(ValueError, match="at least one cell"):
        runner._parse_cells_argument("")


def test_fill_contract_binds_only_the_requested_cells() -> None:
    """A contract for two cells cannot satisfy a different cell list."""
    runner = _runner()
    request = _request(runner)
    contract = runner.benchmark_contract(
        request,
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        cells=["gretel_en", "v2-eval"],
    )

    assert contract["datasets"] == ["v2-eval", "gretel_en"]
    assert contract["total_rows"] == (EVAL_EXPECTED_ROWS["v2-eval"] + EVAL_EXPECTED_ROWS["gretel_en"])
    runner.require_fill_missing_cells_matrix(contract, cells=["v2-eval", "gretel_en"])
    with pytest.raises(RuntimeError, match="exact approved cell list"):
        runner.require_fill_missing_cells_matrix(contract, cells=["v2-eval"])
    with pytest.raises(ValueError, match="binds an explicit cell list"):
        runner.benchmark_contract(request, purpose=runner.FILL_MISSING_CELLS_PURPOSE)
    with pytest.raises(ValueError, match="binds an explicit cell list"):
        runner.benchmark_contract(request, purpose="full_benchmark", cells=["v2-eval"])


def test_fill_approval_signature_covers_the_cell_list() -> None:
    runner = _runner()
    request = _request(runner)
    approval = _fill_approval(runner, request, ["v2-eval", "gretel_en"])

    verified = runner.verify_launch_approval(approval, request=request, purpose=runner.FILL_MISSING_CELLS_PURPOSE)
    assert runner.approved_fill_cells(verified) == ("v2-eval", "gretel_en")

    widened = {**approval, "cells": list(EVAL_DATASETS)}
    with pytest.raises(RuntimeError, match="digest is invalid"):
        runner.verify_launch_approval(widened, request=request, purpose=runner.FILL_MISSING_CELLS_PURPOSE)
    stripped = {key: value for key, value in approval.items() if key != "cells"}
    with pytest.raises(RuntimeError, match="missing or unexpected fields"):
        runner.verify_launch_approval(stripped, request=request, purpose=runner.FILL_MISSING_CELLS_PURPOSE)


def test_non_fill_approval_never_carries_a_cell_list() -> None:
    runner = _runner()
    request = _request(runner)
    approval = _approval(runner, request, "full_benchmark")
    assert "cells" not in approval

    smuggled = {**approval, "cells": ["v2-eval"]}
    with pytest.raises(RuntimeError, match="missing or unexpected fields"):
        runner.verify_launch_approval(smuggled, request=request, purpose="full_benchmark")
    with pytest.raises(ValueError, match="binds a cell list"):
        runner._approval_body(
            request,
            purpose="full_benchmark",
            launch_nonce="e" * 64,
            external_ledger_approval_id="x",
            modeled_prior_attempts=(),
            cells=["v2-eval"],
        )


def test_fill_refuses_a_cell_already_completed_in_this_generation() -> None:
    runner = _runner()
    scheduled = ("v2-eval", "gretel_en")

    runner.require_fillable_cells(scheduled, scheduled)
    with pytest.raises(RuntimeError, match="already completed in this generation"):
        runner.require_fillable_cells(scheduled, ("gretel_en",))
    with pytest.raises(RuntimeError, match=r"\['v2-eval', 'gretel_en'\]"):
        runner.require_fillable_cells(scheduled, ())


def _seed_scored_cell(root: Path, model: str, dataset: str, generation: str, *, hits: int) -> None:
    """Write one done cell where the first `hits` rows predict their gold span."""
    directory = root / "results" / model / dataset
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(EVAL_EXPECTED_ROWS[dataset]):
        gold = {"start": 0, "end": 4, "text": "Anna", "label": "human_name"}
        predicted = gold if index < hits else {"start": 5, "end": 9, "text": "Bobb", "label": "human_name"}
        lines.append(
            json.dumps({
                "id": index,
                "doc_id": f"{dataset}-{index}",
                "pred_spans": [predicted],
                "gold_spans": [gold],
                "language": "en",
                "slice": "full",
            }),
        )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    (directory / "full.jsonl").write_bytes(payload)
    (directory / "full.done").write_bytes(b"done\n")
    (directory / "full.meta.json").write_text(
        json.dumps(
            {
                "rows_in": EVAL_EXPECTED_ROWS[dataset],
                "result_sha256": hashlib.sha256(payload).hexdigest(),
                "fixture_sha256": "7" * 64,
                "dataset_shard_identity_sha256": "8" * 64,
                "evaluation_contract_sha256": generation,
                "elapsed_seconds": 3.5,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_cell_score_summary_emits_raw_counts_the_assembler_can_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One gold and one predicted span per row; only the first `hits` rows match.

    The assembler must accept this payload unchanged; a schema drift between the producer and the consumer would otherwise
    only surface during a real run.

    """
    runner = _runner()
    request = _request(runner)
    generation = "9" * 64
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    root = runner.evaluation_generation_root(request, generation)
    model = runner.checkpoint_model_name(request)
    hits = 400
    _seed_scored_cell(root, model, "v2-eval", generation, hits=hits)

    summary = runner.cell_score_summary.local(asdict(request), generation)

    cell = summary["cells"]["v2-eval"]
    rows = EVAL_EXPECTED_ROWS["v2-eval"]
    assert cell["rows"] == rows
    assert cell["counts"]["exact"] == {
        "tp": hits,
        "pred_total": rows,
        "gold_total": rows,
    }
    assert cell["counts_by_language"]["en"]["rows"] == rows
    assert cell["counts_by_label"]["human_name"]["exact"]["tp"] == hits
    body = {key: value for key, value in summary.items() if key != "summary_digest"}
    assert summary["summary_digest"] == runner.canonical_sha256(body)

    assembler_path = Path(__file__).resolve().parents[3] / "scripts" / "ops" / "assemble_pii350_checkpoint_aggregate.py"
    spec = importlib.util.spec_from_file_location("pii350_assemble_probe", assembler_path)
    assert spec is not None
    assert spec.loader is not None
    assembler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = assembler
    spec.loader.exec_module(assembler)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    loaded = assembler.load_summary(summary_path)
    assert loaded["cells"]["v2-eval"]["counts"]["exact"]["tp"] == hits


def _concurrent_request(runner: ModuleType, ceiling: float, digest: str = "a" * 64) -> BenchmarkRequest:
    base = _request(runner)
    return cast(
        "BenchmarkRequest",
        runner.BenchmarkRequest(
            base.repo_id,
            base.revision,
            base.artifact_path,
            digest,
            base.trajectory_digest,
            ceiling,
            base.all_in_rate_usd_per_second,
            base.reserve_seconds_per_cell,
            "huyhoang0411ha",
        ),
    )


def test_one_profile_holds_several_active_reservations_within_its_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    ledger_path = tmp_path / "ledger.json"
    assert runner.PROFILE_ALL_IN_CEILING_USD["huyhoang0411ha"] == 8.5

    first = runner.issue_launch_approval(
        _concurrent_request(runner, 4.0),
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        launch_nonce="1" * 64,
        ledger_path=ledger_path,
        cells=["v2-eval"],
    )
    second = runner.issue_launch_approval(
        _concurrent_request(runner, 4.0, digest="b" * 64),
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        launch_nonce="2" * 64,
        ledger_path=ledger_path,
        cells=["gretel_en"],
    )

    assert first["approval_digest"] != second["approval_digest"]
    ledger = runner.inspect_approval_ledger(ledger_path)
    active = [
        reservation
        for reservation in ledger["reservations"]
        if reservation["profile"] == "huyhoang0411ha" and reservation["status"] == "active"
    ]
    assert len(active) == 2


def test_a_reservation_beyond_the_remaining_allocation_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """4.0 + 4.0 are already live against an 8.5 allocation; 1.0 more does not fit."""
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    ledger_path = tmp_path / "ledger.json"
    for index, nonce in enumerate(("1" * 64, "2" * 64)):
        runner.issue_launch_approval(
            _concurrent_request(runner, 4.0, digest=chr(ord("a") + index) * 64),
            purpose=runner.FILL_MISSING_CELLS_PURPOSE,
            launch_nonce=nonce,
            ledger_path=ledger_path,
            cells=["v2-eval"],
        )

    with pytest.raises(RuntimeError, match="cumulative modeled ceiling exceeds"):
        runner.issue_launch_approval(
            _concurrent_request(runner, 1.0, digest="c" * 64),
            purpose=runner.FILL_MISSING_CELLS_PURPOSE,
            launch_nonce="3" * 64,
            ledger_path=ledger_path,
            cells=["gretel_en"],
        )


def test_settling_a_reservation_frees_none_of_its_settled_spend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settle the first at its full ceiling.

    The spend moves from an active ceiling to recorded cost; it must keep consuming the same allocation.

    """
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    ledger_path = tmp_path / "ledger.json"
    monkeypatch.setattr(runner, "revoke_launch_lease", SimpleNamespace(remote=lambda _digest: None))
    first = runner.issue_launch_approval(
        _concurrent_request(runner, 4.0),
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        launch_nonce="1" * 64,
        ledger_path=ledger_path,
        cells=["v2-eval"],
    )
    runner.issue_launch_approval(
        _concurrent_request(runner, 4.0, digest="b" * 64),
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        launch_nonce="2" * 64,
        ledger_path=ledger_path,
        cells=["gretel_en"],
    )

    runner.settle_launch_approval(ledger_path, approval_digest=first["approval_digest"], modeled_cost_usd="4.0")

    with pytest.raises(RuntimeError, match="cumulative modeled ceiling exceeds"):
        runner.issue_launch_approval(
            _concurrent_request(runner, 1.0, digest="c" * 64),
            purpose=runner.FILL_MISSING_CELLS_PURPOSE,
            launch_nonce="3" * 64,
            ledger_path=ledger_path,
            cells=["gretel_en"],
        )
    ledger = runner.inspect_approval_ledger(ledger_path)
    settled = ledger["prior_modeled_attempts_by_profile"]["huyhoang0411ha"]
    assert [entry["modeled_all_in_cost_usd"] for entry in settled] == ["4.0"]


def test_duplicate_nonce_and_duplicate_digest_are_still_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real body embeds the nonce, so two issuances can never collide on the digest through the public path.

    Pin the body to prove the digest check is still armed as the second line of defence behind the nonce check.

    """
    runner = _runner()
    _issuer_key_for_test(runner, monkeypatch, tmp_path)
    ledger_path = tmp_path / "ledger.json"
    request = _concurrent_request(runner, 1.0)
    runner.issue_launch_approval(
        request,
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        launch_nonce="1" * 64,
        ledger_path=ledger_path,
        cells=["v2-eval"],
    )

    with pytest.raises(RuntimeError, match="launch nonce was already issued"):
        runner.issue_launch_approval(
            _concurrent_request(runner, 2.0, digest="b" * 64),
            purpose=runner.FILL_MISSING_CELLS_PURPOSE,
            launch_nonce="1" * 64,
            ledger_path=ledger_path,
            cells=["gretel_en"],
        )

    monkeypatch.setattr(
        runner,
        "_approval_body",
        lambda *_args, **_kwargs: {
            "schema_version": runner.LAUNCH_APPROVAL_SCHEMA_VERSION,
            "profile": request.profile,
            "checkpoint_digest": request.checkpoint_digest,
            "purpose": runner.FILL_MISSING_CELLS_PURPOSE,
            "launch_nonce": "1" * 64,
            "external_ledger_approval_id": "fixed-external-id",
            "modeled_prior_attempts": [],
            "new_attempt_ceiling_usd": "1.0",
            "cumulative_modeled_ceiling_usd": "1.0",
            "profile_allocation_usd": "8.5",
            "issuer_key_id": runner.APPROVAL_ISSUER_KEY_ID,
            "cells": ["v2-eval"],
        },
    )
    runner.issue_launch_approval(
        request,
        purpose=runner.FILL_MISSING_CELLS_PURPOSE,
        launch_nonce="8" * 64,
        ledger_path=ledger_path,
        cells=["v2-eval"],
    )
    with pytest.raises(RuntimeError, match="digest was already issued"):
        runner.issue_launch_approval(
            request,
            purpose=runner.FILL_MISSING_CELLS_PURPOSE,
            launch_nonce="9" * 64,
            ledger_path=ledger_path,
            cells=["v2-eval"],
        )


def test_cell_prediction_uses_the_median_not_the_startup_outlier() -> None:
    """The first cell carries one-time kernel-compile overhead; the rest are the steady state.

    A max-based projection would carry that outlier all run.

    Under the pinned A10G rate this is a ~$1.7 reservation, not the ~$4.8 the max-based projection demanded for the same
    cell.

    """
    runner = _runner()
    observed = (0.08, 0.017, 0.017, 0.018)

    predicted = runner.predict_next_cell_seconds(
        next_dataset="nemotron_en",
        next_rows=99_892,
        observed_seconds_per_row=observed,
    )

    median_rate = 0.0175
    assert predicted == pytest.approx(median_rate * 2.0 * 99_892)
    assert predicted < max(observed) * 1.25 * 99_892
    projected_usd = (predicted + 300) * runner.A10G_ALL_IN_RATE_USD_PER_SECOND
    assert projected_usd < 2.0


def test_single_observation_prediction_stays_conservative() -> None:
    runner = _runner()

    predicted = runner.predict_next_cell_seconds(
        next_dataset="gretel_en",
        next_rows=1_000,
        observed_seconds_per_row=(0.05,),
    )

    assert predicted == pytest.approx(0.05 * runner.PREDICTED_CELL_MARGIN * 1_000)


def test_fill_runs_the_cheapest_approved_cell_first() -> None:
    """creddata_en is the smallest cell but sorts last of these three in matrix order.

    So this set separates the schedule from the canonical identity.

    Identity keeps canonical matrix order; only the schedule is reordered.

    """
    runner = _runner()
    requested = ["ai4privacy_de", "creddata_en", "v2-eval"]

    ordered = runner.fill_execution_order(requested)

    assert ordered == ("creddata_en", "v2-eval", "ai4privacy_de")
    rows = [EVAL_EXPECTED_ROWS[dataset] for dataset in ordered]
    assert rows == sorted(rows)
    assert runner.require_fill_cells(requested) == (
        "v2-eval",
        "ai4privacy_de",
        "creddata_en",
    )
    assert runner.require_fill_cells(requested) != ordered


def test_fill_prediction_needs_no_bootstrap_cell() -> None:
    """A fill list without v2-eval would hit the shared projector's cold-start branch and refuse the first cell.

    after the paid lease was already active.

    """
    runner = _runner()
    with pytest.raises(RuntimeError, match="bounded bootstrap"):
        runner.predict_next_cell_seconds(
            next_dataset="nemotron_en",
            next_rows=99_892,
            observed_seconds_per_row=(),
        )

    predicted = runner.predict_fill_cell_seconds(next_rows=99_892)
    assert predicted == pytest.approx(99_892 * runner.FILL_PRIOR_SECONDS_PER_ROW)


def test_fill_budget_admits_a_large_cell_the_old_projector_refused() -> None:
    """Observed on the step-200 run.

    898s elapsed after the first cell, with ai4privacy_en (40,580 rows) still to dispatch under a $2.60 ceiling.

    """
    runner = _runner()
    request = _request(runner)

    predicted = runner.fill_budget_allows_next_cell(elapsed_seconds=898.0, next_rows=40_580, request=request)

    assert predicted == pytest.approx(40_580 * 0.03)
    projected = (898.0 + predicted + request.reserve_seconds_per_cell) * runner.A10G_ALL_IN_RATE_USD_PER_SECOND
    assert projected < request.max_all_in_usd

    tight = runner.BenchmarkRequest(
        request.repo_id,
        request.revision,
        request.artifact_path,
        request.checkpoint_digest,
        request.trajectory_digest,
        0.2,
        request.all_in_rate_usd_per_second,
        request.reserve_seconds_per_cell,
        request.profile,
    )
    with pytest.raises(runner.BudgetRefusal, match="fill budget refuses next cell"):
        runner.fill_budget_allows_next_cell(elapsed_seconds=0.0, next_rows=40_580, request=tight)


def _fill_executor_world(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    request: BenchmarkRequest,
) -> tuple[list[str], Any]:
    """Stub every remote dependency so the fill executor can run in-process."""

    class FakeVolume:
        def commit(self) -> None:
            pass

    class FakeVerified:
        trajectory_digest = request.trajectory_digest
        optimizer_step = 150
        packed_cursor = 19_200
        lifecycle_state = "update_committed"
        world_size = 4
        artifact = SimpleNamespace(revision=request.revision)

    class FakeRows:
        def __init__(self, count: int) -> None:
            self.count = count

        def __len__(self) -> int:
            return self.count

    class FakeAdapter:
        supported_labels = frozenset()
        truncated_documents = 0
        on_progress = None

        def __init__(self, *_: object) -> None:
            pass

        def load(self) -> None:
            pass

    class FakeEvaluation:
        digest = "f" * 64

    from meddies_pii.eval_baseline.adapters import pii350_checkpoint as adapter_module
    from meddies_pii.eval_baseline.baseline import datasets as datasets_module
    from meddies_pii.eval_baseline.baseline import run as run_module

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path))
    monkeypatch.setattr(runner, "benchmark_volume", FakeVolume())
    monkeypatch.setattr(runner, "enable_offline_runtime", lambda _: None)
    monkeypatch.setattr(runner, "require_triton_c_compiler", lambda: None)
    monkeypatch.setattr(runner, "_checkpoint_local_root", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_evaluation_contract", lambda *_: FakeEvaluation())
    monkeypatch.setattr(adapter_module, "verify_checkpoint_artifact", lambda *_: FakeVerified())
    monkeypatch.setattr(adapter_module, "Pii350CheckpointAdapter", FakeAdapter)
    monkeypatch.setattr(
        datasets_module,
        "load_eval_cell",
        lambda dataset, **_: FakeRows(runner.EVAL_EXPECTED_ROWS[dataset]),
    )
    identity = SimpleNamespace(digest="2" * 64)
    monkeypatch.setattr(
        run_module,
        "expected_matrix_shard_identities",
        lambda _evaluation, datasets: dict.fromkeys(datasets, identity),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)),
    )
    ran: list[str] = []
    monkeypatch.setattr(
        run_module,
        "run_shard",
        lambda **kwargs: (
            ran.append(kwargs["spec"].dataset) or SimpleNamespace(skipped=False, output_path=tmp_path / "result.jsonl")
        ),
    )

    def read_matrix(_root: str | Path, _model: str, datasets: Sequence[str], **_kwargs: object) -> SimpleNamespace:
        remaining = tuple(dataset for dataset in datasets if dataset not in ran)
        return SimpleNamespace(
            missing_datasets=remaining,
            rows_by_dataset={},
            result_sha256_by_dataset=dict.fromkeys(datasets, "1" * 64),
            shard_identities_by_dataset={},
            timing_by_dataset={},
        )

    monkeypatch.setattr(run_module, "read_matrix_results", read_matrix)
    return ran, FakeEvaluation


def test_fill_executor_completes_and_persists_its_approved_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Matrix order here is v2-eval, ai4privacy_de, creddata_en.

    The schedule must differ from it, so a canonical-order regression cannot pass this test.

    """
    runner = _runner()
    request = _request(runner)
    cells = ["ai4privacy_de", "creddata_en", "v2-eval"]
    ran, evaluation = _fill_executor_world(runner, monkeypatch, tmp_path, request=request)
    approval = _fill_approval(runner, request, cells)
    # reason: the same module-global rebinding; `ModuleType` declares no `nonce_claim_authority` to assign to.
    runner.nonce_claim_authority = _TestAtomicNonceAuthority()  # ty: ignore[unresolved-attribute]
    runner.persist_active_launch_lease(request, purpose=runner.FILL_MISSING_CELLS_PURPOSE, approval=approval)

    payload = runner.run_fill_missing_cells.local(asdict(request), request.checkpoint_digest, approval)

    assert ran == ["creddata_en", "v2-eval", "ai4privacy_de"]
    assert payload["status"] == "fill_terminal"
    assert payload["filled_cells"] == ["v2-eval", "ai4privacy_de", "creddata_en"]
    assert payload["evaluation_generation_digest"] == evaluation.digest
    assert payload["coverage"]["coverage_cells"] == f"3/{len(EVAL_DATASETS)}"
    assert payload["coverage"]["eligible_for_full_matrix_safety_veto"] is False
    assert [record["dataset"] for record in payload["cost_ledger"]] == ran
    root = runner.evaluation_generation_root(request, evaluation.digest)
    assert json.loads((root / "fill.json").read_text())["status"] == "fill_terminal"
    assert json.loads((root / "state.json").read_text())["status"] == "fill_terminal"


def test_fill_executor_stops_on_budget_and_keeps_the_completed_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """v2-eval predicts 51s and fits; gretel_en predicts 150s and does not."""
    runner = _runner()
    base = _request(runner)
    constrained = runner.BenchmarkRequest(
        base.repo_id,
        base.revision,
        base.artifact_path,
        base.checkpoint_digest,
        base.trajectory_digest,
        0.2,
        base.all_in_rate_usd_per_second,
        base.reserve_seconds_per_cell,
        base.profile,
    )
    ran, evaluation = _fill_executor_world(runner, monkeypatch, tmp_path, request=constrained)
    approval = _fill_approval(runner, constrained, ["v2-eval", "gretel_en"])
    # reason: the same module-global rebinding; `ModuleType` declares no `nonce_claim_authority` to assign to.
    runner.nonce_claim_authority = _TestAtomicNonceAuthority()  # ty: ignore[unresolved-attribute]
    runner.persist_active_launch_lease(constrained, purpose=runner.FILL_MISSING_CELLS_PURPOSE, approval=approval)

    with pytest.raises(runner.BudgetStop):
        runner.run_fill_missing_cells.local(asdict(constrained), constrained.checkpoint_digest, approval)

    assert ran == ["v2-eval"]
    root = runner.evaluation_generation_root(constrained, evaluation.digest)
    stop = json.loads((root / "budget_stop.json").read_text())
    assert stop["status"] == "budget_stop"
    assert stop["dataset"] == "gretel_en"
    assert stop["prior_completed_cell"] == "v2-eval"
    assert json.loads((root / "progress" / "v2-eval.json").read_text())["status"] == "cell_terminal"


def test_fill_coverage_never_claims_whole_matrix_eligibility() -> None:
    runner = _runner()
    coverage = runner.fill_missing_cells_coverage_report(filled_cells=["v2-eval", "gretel_en"], generation_digest="9" * 64)
    assert coverage["coverage_cells"] == f"2/{len(EVAL_DATASETS)}"
    assert coverage["eligible_for_full_matrix_safety_veto"] is False


class _FakeCommitVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class _FakeHub:
    """Stand in for the private HF repo: records upload order, serves downloads."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_order: list[str] = []
        self._commits = 0

    def upload_file(
        self,
        *,
        path_or_fileobj: object,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> SimpleNamespace:
        assert repo_id == "Meddies/pii350-trajectories-private"
        assert repo_type == "model"
        assert commit_message
        payload = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(str(path_or_fileobj)).read_bytes()
        self.objects[path_in_repo] = payload
        self.upload_order.append(path_in_repo)
        self._commits += 1
        return SimpleNamespace(oid=f"{self._commits:040x}")

    def download(self, *, repo_id: str, path_in_repo: str, revision: str, destination: str) -> str:
        assert repo_id == "Meddies/pii350-trajectories-private"
        assert revision
        if path_in_repo not in self.objects:
            raise FileNotFoundError(path_in_repo)
        target = Path(destination) / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.objects[path_in_repo])
        return str(target)


def _seed_result_cell(root: Path, model: str, dataset: str, *, rows: bytes, done: bool = True) -> None:
    directory = root / "results" / model / dataset
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "full.jsonl").write_bytes(rows)
    (directory / "full.meta.json").write_text(
        json.dumps(
            {"rows_in": 1, "result_sha256": hashlib.sha256(rows).hexdigest()},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if done:
        (directory / "full.done").write_bytes(b"done\n")


def _destination_request(runner: ModuleType, request: BenchmarkRequest) -> BenchmarkRequest:
    """Request the same checkpoint and generation from the new workspace."""
    return cast(
        "BenchmarkRequest",
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            8.5,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            "huyhoang0411ha",
        ),
    )


def _export_transfer(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cells: tuple[str, ...] = ("v2-eval", "gretel_en"),
) -> tuple[BenchmarkRequest, str, Path, _FakeHub, dict[str, Any]]:
    source = tmp_path / "source-workspace"
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(source))
    monkeypatch.setattr(runner, "benchmark_volume", _FakeCommitVolume())
    request = _request(runner)
    generation = "9" * 64
    root = runner.evaluation_generation_root(request, generation)
    model = runner.checkpoint_model_name(request)
    for index, dataset in enumerate(cells):
        _seed_result_cell(root, model, dataset, rows=b'{"id":%d}\n' % index)
    hub = _FakeHub()
    monkeypatch.setattr(runner, "_hf_api", lambda: hub)
    monkeypatch.setattr(runner, "_hf_download", hub.download)
    receipt = runner.export_cell_results.local(asdict(request), generation)
    return request, generation, root, hub, receipt


def test_new_evaluation_workspace_has_its_own_pinned_ceiling() -> None:
    runner = _runner()
    request = _request(runner)
    assert runner.PROFILE_ALL_IN_CEILING_USD["huyhoang0411ha"] == 8.5
    assert "huyhoang0411ha" in runner.EVALUATION_PROFILES
    pinned = _destination_request(runner, request)
    assert pinned.max_all_in_usd == 8.5
    assert pinned.profile == "huyhoang0411ha"
    with pytest.raises(ValueError, match="profile-pinned"):
        runner.BenchmarkRequest(
            request.repo_id,
            request.revision,
            request.artifact_path,
            request.checkpoint_digest,
            request.trajectory_digest,
            8.51,
            request.all_in_rate_usd_per_second,
            request.reserve_seconds_per_cell,
            "huyhoang0411ha",
        )


def test_export_uploads_every_cell_file_before_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    _request_used, generation, root, hub, receipt = _export_transfer(runner, monkeypatch, tmp_path)

    assert receipt["cells"] == ["v2-eval", "gretel_en"]
    assert len(receipt["files"]) == 6
    assert receipt["evaluation_generation_digest"] == generation
    assert receipt["source_profile"] == "diffusionllm"
    assert receipt["total_bytes"] == sum(entry["bytes"] for entry in receipt["files"])
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    assert receipt["receipt_digest"] == runner.canonical_sha256(body)
    assert hub.upload_order[-1].endswith("/manifest.json")
    assert len(hub.upload_order) == 7
    for entry in receipt["files"]:
        uploaded = hub.objects[f"{receipt['repo_prefix']}/files/{entry['path']}"]
        assert uploaded == (root / entry["path"]).read_bytes()
        assert hashlib.sha256(uploaded).hexdigest() == entry["sha256"]
    assert "PII350_RESULT_TRANSFER_EXPORT::" in capsys.readouterr().out


def test_export_refuses_a_cell_whose_shard_trio_is_incomplete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    source = tmp_path / "source-workspace"
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(source))
    request = _request(runner)
    generation = "9" * 64
    root = runner.evaluation_generation_root(request, generation)
    model = runner.checkpoint_model_name(request)
    _seed_result_cell(root, model, "v2-eval", rows=b'{"id":0}\n')
    _seed_result_cell(root, model, "gretel_en", rows=b'{"id":1}\n', done=False)
    monkeypatch.setattr(runner, "_hf_api", _FakeHub)

    with pytest.raises(RuntimeError, match="incomplete and cannot be transferred"):
        runner.export_cell_results.local(asdict(request), generation)


def test_export_refuses_a_generation_with_no_completed_cell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path / "source-workspace"))
    request = _request(runner)
    monkeypatch.setattr(runner, "_hf_api", _FakeHub)

    with pytest.raises(RuntimeError, match="no completed result cell"):
        runner.export_cell_results.local(asdict(request), "9" * 64)


def test_import_writes_identical_bytes_at_identical_relative_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    request, generation, source_root, _hub, receipt = _export_transfer(runner, monkeypatch, tmp_path)
    capsys.readouterr()

    destination_request = _destination_request(runner, request)
    volume = _FakeCommitVolume()
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path / "destination"))
    monkeypatch.setattr(runner, "benchmark_volume", volume)
    imported = runner.import_cell_results.local(asdict(destination_request), receipt, receipt["hf_commit_sha"])

    destination_root = runner.evaluation_generation_root(destination_request, generation)
    assert destination_root != source_root
    assert sorted(imported["written_files"]) == sorted(entry["path"] for entry in receipt["files"])
    assert imported["already_present_files"] == []
    assert imported["destination_profile"] == "huyhoang0411ha"
    assert imported["source_receipt_digest"] == receipt["receipt_digest"]
    for entry in receipt["files"]:
        assert (destination_root / entry["path"]).read_bytes() == (source_root / entry["path"]).read_bytes()
    assert volume.commits == 1
    assert "PII350_RESULT_TRANSFER_IMPORT::" in capsys.readouterr().out

    repeated = runner.import_cell_results.local(asdict(destination_request), receipt, receipt["hf_commit_sha"])
    assert repeated["written_files"] == []
    assert sorted(repeated["already_present_files"]) == sorted(entry["path"] for entry in receipt["files"])


def test_import_refuses_a_transferred_file_whose_digest_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request, _generation, _root, hub, receipt = _export_transfer(runner, monkeypatch, tmp_path)
    poisoned = f"{receipt['repo_prefix']}/files/{receipt['files'][0]['path']}"
    original = hub.objects[poisoned]
    hub.objects[poisoned] = b"X" * len(original)

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path / "destination"))
    monkeypatch.setattr(runner, "benchmark_volume", _FakeCommitVolume())
    with pytest.raises(RuntimeError, match="failed hash verification"):
        runner.import_cell_results.local(
            asdict(_destination_request(runner, request)),
            receipt,
            receipt["hf_commit_sha"],
        )


def test_import_refuses_when_a_transferred_file_is_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request, _generation, _root, hub, receipt = _export_transfer(runner, monkeypatch, tmp_path)
    del hub.objects[f"{receipt['repo_prefix']}/files/{receipt['files'][0]['path']}"]

    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path / "destination"))
    monkeypatch.setattr(runner, "benchmark_volume", _FakeCommitVolume())
    with pytest.raises(RuntimeError, match="absent from the pinned revision"):
        runner.import_cell_results.local(
            asdict(_destination_request(runner, request)),
            receipt,
            receipt["hf_commit_sha"],
        )


def test_import_never_overwrites_a_divergent_destination_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner()
    request, generation, _root, _hub, receipt = _export_transfer(runner, monkeypatch, tmp_path)
    destination_request = _destination_request(runner, request)
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path / "destination"))
    monkeypatch.setattr(runner, "benchmark_volume", _FakeCommitVolume())
    destination_root = runner.evaluation_generation_root(destination_request, generation)
    divergent = destination_root / receipt["files"][0]["path"]
    divergent.parent.mkdir(parents=True, exist_ok=True)
    divergent.write_bytes(b"a different evaluator result\n")

    with pytest.raises(RuntimeError, match="already exists with different bytes"):
        runner.import_cell_results.local(asdict(destination_request), receipt, receipt["hf_commit_sha"])
    assert divergent.read_bytes() == b"a different evaluator result\n"


def test_import_refuses_a_receipt_that_does_not_bind_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner()
    request, _generation, _root, _hub, receipt = _export_transfer(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "BENCHMARK_MOUNT", str(tmp_path / "destination"))
    monkeypatch.setattr(runner, "benchmark_volume", _FakeCommitVolume())
    destination_request = _destination_request(runner, request)

    with pytest.raises(RuntimeError, match="immutable manifest commit"):
        runner.import_cell_results.local(asdict(destination_request), receipt, "0" * 40)
    tampered = {**receipt, "total_bytes": int(receipt["total_bytes"]) + 1}
    with pytest.raises(RuntimeError, match="receipt digest is invalid"):
        runner.import_cell_results.local(asdict(destination_request), tampered, receipt["hf_commit_sha"])


def test_checkpoint_partial_16of17_render_command_is_profile_bound() -> None:
    runner = _runner()
    request = _request(runner)
    command = runner.render_checkpoint_partial_16of17_command(
        request,
        _approval(runner, request, runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE),
    )

    assert f"--action {runner.CHECKPOINT_PARTIAL_16OF17_PURPOSE}" in command
    assert f"{runner.PROFILE_ENVIRONMENT_KEY}=diffusionllm" in command
    assert request.checkpoint_digest in command
    assert request.trajectory_digest in command


def test_checkpoint_partial_16of17_render_command_refuses_another_purposes_approval() -> None:
    runner = _runner()
    request = _request(runner)
    approval = _approval(runner, request, runner.STEP60_PARTIAL_16OF17_PURPOSE)

    with pytest.raises(RuntimeError):
        runner.render_checkpoint_partial_16of17_command(request, approval)
