from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from meddies_pii.training.bioes.trainers import pii350_ddp_continuation as continuation

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path


def _gloo_explicit_mean_worker(rank: int, init_file: str, results: MutableMapping[int, float]) -> None:
    from torch import distributed

    # reason: torch declares the torch.distributed members behind an availability gate, so no static
    # reason: reader can prove they are present; this path runs only inside the multi-rank container.
    distributed.init_process_group(  # ty: ignore[possibly-missing-attribute]
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
    )
    gradient = torch.tensor([float(rank + 1)])
    # reason: torch declares the torch.distributed members behind an availability gate, so no static
    # reason: reader can prove they are present; this path runs only inside the multi-rank container.
    distributed.all_reduce(gradient, op=distributed.ReduceOp.SUM)  # ty: ignore[possibly-missing-attribute]
    gradient.div_(4)
    results[rank] = float(gradient.item())
    # reason: torch declares the torch.distributed members behind an availability gate, so no static
    # reason: reader can prove they are present; this path runs only inside the multi-rank container.
    distributed.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def test_step60_resume_assigns_four_windows_and_commits_step61_cursor7808() -> None:
    windows = continuation.rank_windows(cursor=7_680)
    assert windows == [(7_680, 7_712), (7_712, 7_744), (7_744, 7_776), (7_776, 7_808)]
    assert continuation.commit_update(step=60, cursor=7_680, rank_consensus=True) == (
        61,
        7_808,
    )
    with pytest.raises(RuntimeError, match="consensus"):
        continuation.commit_update(step=60, cursor=7_680, rank_consensus=False)


def test_trajectory_digest_survives_profile_change_but_execution_digest_does_not() -> None:
    first = continuation.render_segment_contract("meddiesresearch")
    second = continuation.render_segment_contract("huyhoang0411ha")
    assert continuation.WAVE_ORDER == (
        "meddiesresearch",
        "huyhoang0411ha",
        "meddies-ocr",
    )
    assert second["execution"]["timeout_seconds"] == 10_800
    assert second["execution"]["all_in_ceiling_usd"] == 30.0
    assert second["execution"]["auxiliary_billing_reserve"] == {
        "usd": 0.40,
        "cpu_rate_usd_per_second": continuation.CPU_AUXILIARY_RATE_USD_PER_SECOND,
        "max_cpu_calls": 4,
        "per_call_timeout_seconds": 1_800,
        "basis": "bounded 2CPU+8GiB preflight/receipt/transport plus volume-network uncertainty",
    }
    assert first["trajectory_digest"] == second["trajectory_digest"]
    assert first["execution_contract_digest"] != second["execution_contract_digest"]
    assert first["execution"]["trajectory_digest"] == first["trajectory_digest"]
    with pytest.raises(ValueError, match="approved training profile"):
        continuation.render_segment_contract("huyhoangha0411")


def test_schema_v2_records_explicit_single_to_two_rank_transition_for_wave_one() -> None:
    contract = continuation.render_segment_contract("meddiesresearch")
    metadata = continuation.checkpoint_metadata_v2(
        contract,
        parent_checkpoint_digest="source-step60",
        step=60,
        cursor=7_680,
        lifecycle_state="resume_source",
        rank_rng_states=(),
        execution_world_size=2,
        source_world_size=1,
    )
    assert metadata["schema_version"] == 2
    assert metadata["rng_transition"]["kind"] == "single_rank_to_two_rank_v1"
    assert [item["seed"] for item in metadata["rng_transition"]["rank_seeds"]] == [
        continuation.derived_rank_seed("source-step60", rank, world_size=2) for rank in range(2)
    ]


def test_two_rank_checkpoint_can_transition_to_four_without_reusing_rng_streams() -> None:
    wave_one = continuation.render_segment_contract("meddiesresearch")
    rank_rng_states = [{"rank": rank, "cpu_rng": "00", "cuda_rng": "00"} for rank in range(2)]
    metadata = continuation.checkpoint_metadata_v2(
        wave_one,
        parent_checkpoint_digest="source-step60",
        step=61,
        cursor=7_808,
        lifecycle_state="update_committed",
        rank_rng_states=rank_rng_states,
        execution_world_size=2,
        source_world_size=1,
        wave_profile="meddiesresearch",
    )
    assert metadata["world_size"] == 2
    assert metadata["rng_transition"]["kind"] == "single_rank_to_two_rank_v1"
    assert continuation.rng_transition_kind(2, 4) == "two_rank_to_four_rank_v1"
    assert [continuation.derived_rank_seed("wave-one-step61", rank, world_size=4) for rank in range(4)] != [
        continuation.derived_rank_seed("wave-one-step61", rank, world_size=2) for rank in range(2)
    ]


def test_explicit_four_rank_gradient_mean_and_parameter_drift_are_exact() -> None:
    gradients = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
        torch.tensor([7.0, 8.0]),
    ]
    assert torch.equal(continuation.explicit_gradient_mean(gradients), torch.tensor([4.0, 5.0]))
    assert continuation.max_parameter_drift([torch.tensor([1.0]), torch.tensor([1.0])] * 2) == 0.0
    with pytest.raises(RuntimeError, match="drift"):
        continuation.require_zero_parameter_drift([torch.tensor([1.0]), torch.tensor([1.001])])


def test_four_rank_gloo_explicit_gradient_mean_is_identical_on_every_rank(
    tmp_path: Path,
) -> None:
    from torch import multiprocessing

    manager = multiprocessing.Manager()
    results = manager.list([0.0] * 4)
    init_file = tmp_path / "gloo-init"
    multiprocessing.spawn(_gloo_explicit_mean_worker, args=(str(init_file), results), nprocs=4, join=True)
    assert list(results) == [2.5, 2.5, 2.5, 2.5]


def test_manifest_last_transport_rejects_unverified_file_before_deserialization(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "adapter.bin"
    payload.write_bytes(b"adapter")
    manifest = continuation.build_manifest({"adapter.bin": payload})
    continuation.verify_manifest_files(tmp_path, manifest)
    payload.write_bytes(b"tamper!")
    with pytest.raises(RuntimeError, match="SHA-256"):
        continuation.verify_manifest_files(tmp_path, manifest)


def test_checkpoint_digest_binds_committed_manifest_and_canonical_metadata(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "classifier.pt"
    payload.write_bytes(b"state")
    manifest = continuation.build_manifest({"classifier.pt": payload})
    assert continuation.checkpoint_digest({"step": 60}, manifest) != continuation.checkpoint_digest({"step": 61}, manifest)


def test_budget_and_whole_checkpoint_evaluation_scheduler_are_deterministic() -> None:
    assert continuation.budget_allows_next_step(
        elapsed_seconds=10.0,
        predicted_step_seconds=2.0,
        ceiling_usd=14.0,
        rate_usd_per_second=1.0,
        reserve_seconds=1.0,
    )
    assert not continuation.budget_allows_next_step(
        elapsed_seconds=10.0,
        predicted_step_seconds=2.0,
        ceiling_usd=14.0,
        rate_usd_per_second=1.0,
        reserve_seconds=3.1,
    )
    queue = continuation.assign_evaluation_checkpoints(["step-100", "step-150", "step-200", "step-250"])
    assert queue == [
        ("step-100", "diffusionllm"),
        ("step-150", "meddies-pii"),
        ("step-200", "huyhoang041100"),
        ("step-250", "meddies-run"),
    ]


def test_launch_digest_binds_parent_identity_and_terminal_checkpoint_never_overwrites_cadence() -> None:
    contract = continuation.render_segment_contract("meddiesresearch")
    source = {
        "checkpoint_digest": "parent-100",
        "optimizer_step": 100,
        "packed_cursor": 12_800,
        "world_size": 4,
    }
    launch = continuation.launch_digest(contract["execution_contract_digest"], source)
    assert launch != continuation.launch_digest(
        contract["execution_contract_digest"],
        {**source, "checkpoint_digest": "other-parent"},
    )
    assert continuation.checkpoint_directory_name(step=100, terminal=False) == "step-00000100"
    assert continuation.checkpoint_directory_name(step=100, terminal=True) == "terminal-step-00000100"


def test_preflight_digest_is_independent_of_future_stage_two_source_root() -> None:
    contract = continuation.render_segment_contract("huyhoang0411ha")
    attestation = {"body_tensor_count": 10, "body_tensor_values_sha256": "body"}
    first = continuation.preflight_digest(
        contract["execution_contract_digest"],
        packed_paths_sha256="packed",
        shard_count=3,
        encoder_checkpoint_attestation=attestation,
    )
    second = continuation.preflight_digest(
        contract["execution_contract_digest"],
        packed_paths_sha256="packed",
        shard_count=3,
        encoder_checkpoint_attestation=attestation,
    )
    assert first == second


def test_three_stage_rendered_commands_require_explicit_preflight_and_parent_receipts() -> None:
    preflight = continuation.render_preflight_command("meddiesresearch")
    source = continuation.render_cpu_receipt_command("meddiesresearch", "/artifacts/step60")
    train = continuation.render_train_command("meddiesresearch")
    assert "--preflight" in preflight
    assert "--source-root '/artifacts/step60'" in source
    assert "--source-receipt-json" in train
    assert "--preflight-receipt-json" in train
    assert "--source-launch-digest" in train
    assert "--launch-attempt-nonce" in train
    assert "--interruption-cost-receipt-json" in train


def test_default_source_receipt_command_is_paste_ready_for_the_frozen_step60_parent() -> None:
    command = continuation.render_cpu_receipt_command("meddiesresearch")
    assert "<SOURCE_CHECKPOINT_ROOT>" not in command
    assert "/artifacts/full-runs/pii350/2e679221fd7c4ec6925e13f27cf96769/checkpoints/step-00000060" in command


def test_exact_terminal_partial_covers_all_remaining_units_once_in_four_equal_slices() -> None:
    windows = continuation.rank_windows(
        cursor=continuation.FINAL_GLOBAL_START,
    )
    assert windows == [
        (119_936, 119_961),
        (119_961, 119_986),
        (119_986, 120_011),
        (120_011, 120_036),
    ]
    assert [unit for start, end in windows for unit in range(start, end)] == list(range(119_936, 120_036))
    assert continuation.commit_update(
        step=937,
        cursor=119_936,
        rank_consensus=True,
        consumed_units=continuation.FINAL_GLOBAL_BATCH_SIZE,
    ) == (938, 120_036)


def test_only_the_terminal_partial_metadata_may_break_step_times_128() -> None:
    contract = continuation.render_segment_contract("meddies-ocr")
    rng = [{"rank": rank, "cpu_rng": "00", "cuda_rng": "00"} for rank in range(4)]
    metadata = continuation.checkpoint_metadata_v2(
        contract,
        parent_checkpoint_digest="step-937",
        step=938,
        cursor=120_036,
        lifecycle_state="terminal",
        rank_rng_states=rng,
        wave_profile="meddies-ocr",
        wave_cumulative_all_in_cost_usd=70.0,
    )
    assert metadata["epoch_complete"] is True
    with pytest.raises(RuntimeError, match="inconsistent"):
        continuation.checkpoint_metadata_v2(
            contract,
            parent_checkpoint_digest="bad",
            step=938,
            cursor=120_000,
            lifecycle_state="terminal",
            rank_rng_states=rng,
        )


def test_wave_order_and_epoch_terminal_refuse_invalid_next_launches() -> None:
    assert continuation.validate_wave_source("meddiesresearch", {"wave_profile": None}) == 0.0
    assert continuation.validate_wave_source("huyhoang0411ha", {"wave_profile": "meddiesresearch"}) == 0.0
    assert continuation.validate_wave_source("meddies-ocr", {"wave_profile": "huyhoang0411ha"}) == 0.0
    with pytest.raises(RuntimeError, match="prior wave"):
        continuation.validate_wave_source("meddies-ocr", {"wave_profile": "meddiesresearch"})
    with pytest.raises(RuntimeError, match="epoch-terminal"):
        continuation.validate_wave_source("meddies-ocr", {"epoch_complete": True, "wave_profile": "meddies-ocr"})


def test_auxiliary_reserve_is_applied_once_for_new_wave_and_not_again_on_resume() -> None:
    new_ocr = {
        "wave_profile": "huyhoang0411ha",
        "wave_cumulative_all_in_cost_usd": 30.0,
    }
    resumed_ocr = {
        "wave_profile": "meddies-ocr",
        "wave_cumulative_all_in_cost_usd": 12.5,
    }
    assert continuation.effective_wave_carried_cost("meddies-ocr", new_ocr) == 1.2
    assert continuation.effective_wave_carried_cost("meddies-ocr", resumed_ocr) == 12.5
    contract = continuation.render_segment_contract("meddies-ocr")
    reserve = contract["execution"]["auxiliary_billing_reserve"]
    assert reserve["usd"] == 1.2
    assert reserve["max_cpu_calls"] == 12


def test_gpu_budget_uses_full_ceiling_minus_effective_auxiliary_carry() -> None:
    effective_carry = continuation.effective_wave_carried_cost("meddies-ocr", {"wave_profile": "huyhoang0411ha"})
    assert not continuation.budget_allows_next_step(
        elapsed_seconds=69.0,
        predicted_step_seconds=1.0,
        ceiling_usd=71.0 - effective_carry,
        rate_usd_per_second=1.0,
        reserve_seconds=1.0,
    )


def test_static_slice_denominations_admit_hangs_without_exceeding_wave_ceiling() -> None:
    assert continuation.select_slice_timeout_seconds("meddiesresearch", carried_cost_usd=0.0) == 7_900
    assert continuation.select_slice_timeout_seconds("huyhoang0411ha", carried_cost_usd=0.4) == 10_800
    assert continuation.select_slice_timeout_seconds("meddies-ocr", carried_cost_usd=1.2) == 10_800
    assert continuation.select_slice_timeout_seconds("meddies-ocr", carried_cost_usd=60.0) == 4_000
    with pytest.raises(RuntimeError, match="no approved"):
        continuation.select_slice_timeout_seconds("meddiesresearch", carried_cost_usd=3.0)


def test_wave_one_admission_uses_live_balance_without_double_counting_history() -> None:
    carried = continuation.effective_wave_carried_cost("meddiesresearch", {"wave_profile": None})
    assert pytest.approx(2.16724464) == continuation.WAVE_ONE_FAILED_CONTAINER_CARRY_USD
    assert carried == 0.0
    contract = continuation.render_segment_contract("meddiesresearch")
    rate = contract["execution"]["all_in_rate_usd_per_second"]
    assert rate == pytest.approx(0.00136048)
    assert contract["execution"]["all_in_ceiling_usd"] == 11.0
    assert contract["execution"]["budget_basis"] == ("reported_modal_live_remaining_balance_usd")
    assert contract["execution"]["slice_timeout_schedule_seconds"] == [7_900]
    assert contract["execution"]["historical_failed_container_carry_usd"] == (pytest.approx(2.16724464))
    total = carried + 7_900 * rate
    assert total == pytest.approx(10.747792)
    assert 11.0 - total == pytest.approx(0.252208)
    continuation.require_slice_admission("meddiesresearch", carried_cost_usd=carried, slice_timeout_seconds=7_900)
    with pytest.raises(RuntimeError, match="largest approved"):
        continuation.require_slice_admission("meddiesresearch", carried_cost_usd=carried, slice_timeout_seconds=8_000)
    with pytest.raises(RuntimeError, match="largest approved"):
        continuation.require_slice_admission("meddiesresearch", carried_cost_usd=carried, slice_timeout_seconds=8_300)
    assert continuation.select_slice_timeout_seconds("meddiesresearch", carried_cost_usd=carried) == 7_900
    with pytest.raises(RuntimeError, match="no approved"):
        continuation.select_slice_timeout_seconds("meddiesresearch", carried_cost_usd=carried + 0.3)


# reason: Checkpoint, evaluation, and comparison digests plus metrics are independent cadence-receipt axes.
def _benchmark_cadence_result(  # ruff: ignore[too-many-arguments]
    *,
    contract: dict[str, Any],
    step: int,
    checkpoint_digest: str,
    comparison_contract_digest: str,
    evaluation_contract_digest: str,
    precision: float,
    recall: float,
    f1: float,
) -> dict[str, object]:
    return {
        "checkpoint_digest": checkpoint_digest,
        "trajectory_digest": contract["trajectory_digest"],
        "evaluation_contract_digest": evaluation_contract_digest,
        "comparison_contract_digest": comparison_contract_digest,
        "optimizer_step": step,
        "fixed_nine_exact_typed": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def _cadence_receipt(
    *,
    contract: dict[str, Any],
    history: list[dict[str, object]],
    comparison_contract_digest: str,
) -> dict[str, object]:
    decision = continuation._preregistered_early_stop_decision(history)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "trajectory_digest": contract["trajectory_digest"],
        "comparison_contract_digest": comparison_contract_digest,
        "history": history,
        "history_digest": continuation._canonical_sha256(history),
        "decision": decision,
        "status": decision["status"],
        "last_evaluated_optimizer_step": history[-1]["optimizer_step"],
    }
    receipt["decision_digest"] = continuation._canonical_sha256(receipt)
    return receipt


def test_post_step60_continue_receipt_requires_benchmark_semantics_and_complete_cadence() -> None:
    contract = continuation.render_segment_contract("huyhoang0411ha")
    comparison_contract_digest = "a" * 64
    history = [
        _benchmark_cadence_result(
            contract=contract,
            step=100,
            checkpoint_digest="1" * 64,
            comparison_contract_digest=comparison_contract_digest,
            evaluation_contract_digest="2" * 64,
            precision=0.80,
            recall=0.70,
            f1=0.74,
        ),
        _benchmark_cadence_result(
            contract=contract,
            step=150,
            checkpoint_digest="3" * 64,
            comparison_contract_digest=comparison_contract_digest,
            evaluation_contract_digest="4" * 64,
            precision=0.81,
            recall=0.72,
            f1=0.75,
        ),
    ]
    receipt = _cadence_receipt(
        contract=contract,
        history=history,
        comparison_contract_digest=comparison_contract_digest,
    )
    continuation.validate_early_stop_receipt(contract, source_step=150, receipt=receipt)
    with pytest.raises(RuntimeError, match="requires"):
        continuation.validate_early_stop_receipt(contract, source_step=150, receipt=None)
    with pytest.raises(RuntimeError, match="complete cadence"):
        continuation.validate_early_stop_receipt(contract, source_step=200, receipt=receipt)


def test_recomputed_digests_cannot_turn_real_early_stop_into_continue() -> None:
    contract = continuation.render_segment_contract("huyhoang0411ha")
    comparison_contract_digest = "a" * 64
    history = [
        _benchmark_cadence_result(
            contract=contract,
            step=100,
            checkpoint_digest="1" * 64,
            comparison_contract_digest=comparison_contract_digest,
            evaluation_contract_digest="2" * 64,
            precision=0.80,
            recall=0.72,
            f1=0.75,
        ),
        _benchmark_cadence_result(
            contract=contract,
            step=150,
            checkpoint_digest="3" * 64,
            comparison_contract_digest=comparison_contract_digest,
            evaluation_contract_digest="4" * 64,
            precision=0.79,
            recall=0.72,
            f1=0.745,
        ),
        _benchmark_cadence_result(
            contract=contract,
            step=200,
            checkpoint_digest="5" * 64,
            comparison_contract_digest=comparison_contract_digest,
            evaluation_contract_digest="6" * 64,
            precision=0.78,
            recall=0.71,
            f1=0.740,
        ),
    ]
    early_stop_receipt = _cadence_receipt(
        contract=contract,
        history=history,
        comparison_contract_digest=comparison_contract_digest,
    )
    assert early_stop_receipt["status"] == "early_stop"

    forged_decision = dict(cast("dict[str, Any]", early_stop_receipt["decision"]))
    forged_decision["status"] = "continue"
    forged_continue = {
        **early_stop_receipt,
        "decision": forged_decision,
        "status": "continue",
    }
    forged_continue["decision_digest"] = continuation._canonical_sha256({
        key: value for key, value in forged_continue.items() if key != "decision_digest"
    })
    with pytest.raises(RuntimeError, match="decision"):
        continuation.validate_early_stop_receipt(contract, source_step=200, receipt=forged_continue)


def test_rendered_timeout_is_the_profile_static_cap_not_the_legacy_10800_default() -> None:
    contract = continuation.render_segment_contract("meddiesresearch")
    assert contract["execution"]["timeout_seconds"] == 7_900


def test_step200_interruption_receipt_carries_replayed_attempt_cost_without_reset() -> None:
    profile = "huyhoang0411ha"
    contract = continuation.render_segment_contract(profile)
    source = {
        "checkpoint_digest": "10c305bfc0d4c33ec16a192574c631366461bb8955903035ade5424dea4172e2",
        "optimizer_step": 200,
        "packed_cursor": 25_600,
        "world_size": 4,
        "wave_profile": profile,
        "wave_cumulative_all_in_cost_usd": 7.6887616350,
    }
    source_launch_digest = continuation.launch_digest(contract["execution_contract_digest"], source)
    receipt = {
        "schema_version": 1,
        "profile": profile,
        "execution_contract_digest": contract["execution_contract_digest"],
        "source_checkpoint_digest": source["checkpoint_digest"],
        "source_launch_digest": source_launch_digest,
        "observed_modal_cost_usd": 10.5361509104,
        "automatic_retry_cost_usd": 0.125,
        "wave_cumulative_all_in_cost_usd": 10.6611509104,
    }

    carried = continuation.effective_wave_carried_cost(profile, source, interruption_cost_receipt=receipt)
    assert carried == pytest.approx(10.6611509104)
    assert continuation.select_slice_timeout_seconds(profile, carried_cost_usd=carried) == 7_000
    with pytest.raises(RuntimeError, match="requires a reconciled"):
        continuation.effective_wave_carried_cost(profile, source)

    with pytest.raises(RuntimeError, match="cannot be lower"):
        continuation.effective_wave_carried_cost(
            profile,
            source,
            interruption_cost_receipt={
                **receipt,
                "observed_modal_cost_usd": 10.0,
                "wave_cumulative_all_in_cost_usd": 10.125,
            },
        )


def test_final_local_batch_size_is_the_four_rank_share_the_runtime_derives() -> None:
    assert continuation.FINAL_LOCAL_BATCH_SIZE == continuation.FINAL_GLOBAL_BATCH_SIZE // 4

    windows = continuation.rank_windows(cursor=continuation.FINAL_GLOBAL_START, world_size=4)
    assert [end - start for start, end in windows] == [continuation.FINAL_LOCAL_BATCH_SIZE] * 4


def test_compatibility_exports_still_equal_the_four_rank_authorization_pair() -> None:
    tokens = continuation.authorization_tokens(4)

    assert tokens["confirmation"] == continuation.TRAIN_CONFIRMATION
    assert tokens["primary_action"] == continuation.PRIMARY_ACTION
