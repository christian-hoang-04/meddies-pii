from __future__ import annotations

from typing import TYPE_CHECKING

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
import pytest
import torch

from meddies_pii.training.bioes.trainers import full_run_runtime
from meddies_pii.training.bioes.trainers import pii350_ddp_runtime as runtime

if TYPE_CHECKING:
    from pathlib import Path


def test_four_rank_step60_resume_lifecycle_requires_all_gates_before_checkpoint() -> None:
    state = runtime.ResumeState(
        step=60,
        cursor=7_680,
        parent_checkpoint_digest="source-step60",
        source_world_size=1,
    )
    plan = runtime.begin_four_rank_resume(state)
    assert plan.windows == (
        (7_680, 7_712),
        (7_712, 7_744),
        (7_744, 7_776),
        (7_776, 7_808),
    )
    committed = runtime.commit_four_rank_update(
        plan,
        finite_loss=True,
        finite_gradients=True,
        gradient_drift=0.0,
        parameter_drift=0.0,
        rank_successes=(True, True, True, True),
    )
    assert (committed.step, committed.cursor) == (61, 7_808)
    with pytest.raises(RuntimeError, match="rank consensus"):
        runtime.commit_four_rank_update(
            plan,
            finite_loss=True,
            finite_gradients=True,
            gradient_drift=0.0,
            parameter_drift=0.0,
            rank_successes=(True, True, False, True),
        )


def test_two_rank_checkpoint_resumes_on_four_ranks_with_new_destination_windows() -> None:
    state = runtime.ResumeState(
        step=61,
        cursor=7_808,
        parent_checkpoint_digest="wave-one-step61",
        source_world_size=2,
    )
    plan = runtime.begin_four_rank_resume(state, execution_world_size=4)
    assert plan.rng_transition == "two_rank_to_four_rank_v1"
    assert plan.windows == (
        (7_808, 7_840),
        (7_840, 7_872),
        (7_872, 7_904),
        (7_904, 7_936),
    )
    with pytest.raises(RuntimeError, match="reduce"):
        runtime.begin_four_rank_resume(
            runtime.ResumeState(
                step=61,
                cursor=7_808,
                parent_checkpoint_digest="four-rank-parent",
                source_world_size=4,
            ),
            execution_world_size=2,
        )


def test_checkpoint_cadence_saves_step50_or_terminal_only_after_consensus() -> None:
    assert runtime.should_save_checkpoint(step=100, terminal=False, rank_consensus=True)
    assert runtime.should_save_checkpoint(step=101, terminal=True, rank_consensus=True)
    assert not runtime.should_save_checkpoint(step=101, terminal=False, rank_consensus=True)
    assert not runtime.should_save_checkpoint(step=100, terminal=False, rank_consensus=False)


def test_rank_rng_state_gather_restore_requires_exact_four_rank_mapping() -> None:
    rng = runtime.rank_rng_records([b"cpu0", b"cpu1", b"cpu2", b"cpu3"], [b"cuda0", b"cuda1", b"cuda2", b"cuda3"])
    assert [record["rank"] for record in rng] == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="runtime rank"):
        runtime.require_four_rank_rng_records(rng[:3])


def test_explicit_gradient_reduction_and_zero_drift_use_the_real_torch_tensors() -> None:
    gradients = [torch.tensor([float(value)]) for value in (1, 2, 3, 4)]
    assert torch.equal(runtime.mean_four_rank_gradient_tensors(gradients), torch.tensor([2.5]))
    with pytest.raises(RuntimeError, match="non-finite"):
        runtime.require_finite_tensors([torch.tensor([float("nan")])])


def test_verified_resume_restores_payload_before_applying_explicit_rank_rng_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        full_run_runtime,
        "_restore_pii_training_payloads",
        lambda *_args, **_kwargs: events.append("restore") or {"cpu": b"one", "cuda": [b"one"]},
    )
    result = full_run_runtime.load_verified_pii_resume_state(
        str(tmp_path),
        tagger=object(),
        optimizer=object(),
        verified_metadata={
            "hash_verified": True,
            "optimizer_step": 60,
            "packed_cursor": 7_680,
            "world_size": 1,
            "checkpoint_digest": "source",
        },
        rank_local_rng_policy=lambda _rng: events.append("rng_policy"),
    )
    assert result == {"optimizer_step": 60, "packed_cursor": 7_680}
    assert events == ["restore", "rng_policy"]
    with pytest.raises(RuntimeError, match="hash-verified"):
        full_run_runtime.load_verified_pii_resume_state(
            str(tmp_path),
            tagger=object(),
            optimizer=object(),
            verified_metadata={},
            rank_local_rng_policy=lambda _rng: None,
        )


def test_verified_resume_accepts_two_rank_parent_for_later_four_rank_transition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        full_run_runtime,
        "_restore_pii_training_payloads",
        lambda *_args, **_kwargs: {"rank_rng_states": []},
    )
    assert full_run_runtime.load_verified_pii_resume_state(
        str(tmp_path),
        tagger=object(),
        optimizer=object(),
        verified_metadata={
            "hash_verified": True,
            "optimizer_step": 61,
            "packed_cursor": 7_808,
            "world_size": 2,
            "checkpoint_digest": "two-rank-source",
        },
        rank_local_rng_policy=lambda _rng: None,
    ) == {"optimizer_step": 61, "packed_cursor": 7_808}


def test_budget_gate_reserves_persistence_before_warmup_prediction_exists() -> None:
    assert runtime.projected_budget_stop(
        worker_elapsed_seconds=1.0,
        predicted_step_seconds=0.0,
        all_in_rate_usd_per_second=1.0,
        ceiling_usd=600.5,
        persistence_reserve_seconds=600.0,
    )


def test_epoch_terminal_resume_and_overrun_are_rejected() -> None:
    state = runtime.ResumeState(
        step=938,
        cursor=120_036,
        parent_checkpoint_digest="terminal",
        source_world_size=4,
    )
    with pytest.raises(RuntimeError, match="epoch-terminal"):
        runtime.begin_four_rank_resume(state)


def test_epoch_terminal_state_skips_resume_planning_so_checkpoint_logic_can_run() -> None:
    terminal = runtime.ResumeState(
        step=938,
        cursor=120_036,
        parent_checkpoint_digest="terminal",
        source_world_size=4,
    )
    from meddies_pii.training.bioes.modal import (
        pii350_ddp_continuation as modal_continuation,
    )

    assert modal_continuation._next_plan_or_terminal(terminal) is None


def _fake_torch(device_count: int, *, name: str = "NVIDIA A100-SXM4-40GB") -> object:
    from types import SimpleNamespace

    total_memory = 40 * 1024**3 - 1
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: device_count,
            get_device_properties=lambda _index: SimpleNamespace(name=name, total_memory=total_memory),
        ),
    )


def test_four_rank_topology_helper_pins_the_world_size_its_name_states() -> None:
    inventory = runtime.require_four_a100_40gb(_fake_torch(4))

    assert [entry["index"] for entry in inventory] == [0, 1, 2, 3]
    assert inventory == runtime.require_a100_40gb(_fake_torch(4), world_size=4)


def test_four_rank_topology_helper_refuses_a_two_rank_machine() -> None:
    with pytest.raises(RuntimeError, match="contract-visible CUDA devices"):
        runtime.require_four_a100_40gb(_fake_torch(2))


def test_four_rank_topology_helper_refuses_a_non_a100_device() -> None:
    with pytest.raises(RuntimeError, match="A100-40GB devices"):
        runtime.require_four_a100_40gb(_fake_torch(4, name="NVIDIA H100 80GB HBM3"))
