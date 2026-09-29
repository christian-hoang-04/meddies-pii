from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# reason: the CLI purity probe runs this interpreter with a fixed module and flags as list-form argv.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from typing import cast

import pytest
import torch

from meddies_pii.training.bioes.trainers import pii350_ddp_smoke


def test_rank_assignment_is_seed_ordered_non_overlapping_and_exhaustive() -> None:
    assignments = [pii350_ddp_smoke.rank_window(global_step=3, rank=rank) for rank in range(pii350_ddp_smoke.WORLD_SIZE)]

    assert assignments == [(256, 320), (320, 384)]
    assert pii350_ddp_smoke.validate_rank_windows(global_step=3, windows=assignments) == {
        "global_step": 3,
        "global_window": (256, 384),
        "rank_windows": assignments,
    }

    with pytest.raises(RuntimeError, match="overlap"):
        pii350_ddp_smoke.validate_rank_windows(global_step=0, windows=[(0, 64), (63, 127)])


def test_weighted_rank_losses_equal_global_token_mean_gradient() -> None:
    """RED first: DDP's mean reduction needs world-size token weighting."""
    local_losses = [
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(5.0, requires_grad=True),
    ]
    active = [torch.tensor(3), torch.tensor(7)]

    scaled = [
        pii350_ddp_smoke.scale_local_token_mean_loss(loss, local_active=count, global_active=10)
        for loss, count in zip(local_losses, active, strict=True)
    ]
    ddp_mean_objective = cast("torch.Tensor", sum(scaled) / pii350_ddp_smoke.WORLD_SIZE)

    assert ddp_mean_objective.item() == pytest.approx((2.0 * 3 + 5.0 * 7) / 10)
    ddp_mean_objective.backward()
    assert local_losses[0].grad is not None
    assert local_losses[1].grad is not None
    assert local_losses[0].grad.item() == pytest.approx(3 / 10)
    assert local_losses[1].grad.item() == pytest.approx(7 / 10)


def test_contract_is_exactly_two_a100s_and_matches_constant3_optimizer() -> None:
    contract = pii350_ddp_smoke.render_ddp_smoke_contract()

    assert contract["model_id"] == "LiquidAI/LFM2.5-Encoder-350M-PII-Detector"
    assert contract["topology"] == {
        "gpu": "A100-40GB:2",
        "world_size": 2,
        "local_batch_size": 64,
        "global_batch_size": 128,
        "model_parallelism": False,
    }
    assert contract["training"] == {
        "optimizer_steps": 10,
        "evaluation": False,
        "checkpoint": False,
        "optimizer": {
            "name": "adamw",
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
            "fused": False,
            "scheduler": None,
        },
        "loss_reduction": (
            "explicit_gradient_bucket_all_reduce_mean_of_world_size_times_local_active_labels_over_global_active_labels"
        ),
    }
    assert contract["config_digest"] == pii350_ddp_smoke.contract_digest(contract)
    assert contract["attestation"]["interconnect_evidence"] == (
        "pre_torchrun_uuid_inventory_plus_raw_nvlink_status_and_symmetric_peer_access_nccl_readiness_v1"
    )
    assert contract["attestation"]["gradient_reduction"] == ("no_sync_then_explicit_all_reduce_mean_v1")
    assert contract["attestation"]["require_step_one_parameter_gradient_diagnostics"]
    assert contract["attestation"]["step_one_diagnostic_persistence"] == (
        "rank_zero_atomic_json_with_fresh_nested_parent_v1"
    )
    pii350_ddp_smoke.require_ddp_smoke_execute(
        contract,
        execute=True,
        confirmation=pii350_ddp_smoke.DDP_SMOKE_CONFIRMATION,
        primary_action=pii350_ddp_smoke.DDP_SMOKE_PRIMARY_ACTION,
    )


def test_commands_are_reviewable_and_paid_dispatch_is_explicit() -> None:
    preflight = pii350_ddp_smoke.render_ddp_smoke_preflight_command()
    prewarm = pii350_ddp_smoke.render_ddp_smoke_prewarm_command()
    launch = pii350_ddp_smoke.render_ddp_smoke_launch_command()

    assert preflight.startswith("MODAL_PROFILE=diffusionllm ")
    assert "--preflight-assets" in preflight
    assert "--execute" not in preflight
    assert prewarm == (
        "MODAL_PROFILE=diffusionllm uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.pii350_ddp_smoke --prewarm-assets"
    )
    assert "--execute" not in prewarm
    assert "--confirmation" not in prewarm
    assert "--primary-action" not in prewarm
    assert "--config-digest" not in prewarm
    assert "--execute" in launch
    assert f"--confirmation {pii350_ddp_smoke.DDP_SMOKE_CONFIRMATION}" in launch
    assert f"--primary-action {pii350_ddp_smoke.DDP_SMOKE_PRIMARY_ACTION}" in launch
    assert "--config-digest " in launch


def test_cpu_cli_only_renders_and_never_imports_modal() -> None:
    command = [
        sys.executable,
        "-m",
        "meddies_pii.training.bioes.trainers.pii350_ddp_smoke",
        "--render-launch-command",
    ]
    # reason: command is assembled immediately above from sys.executable and fixed module/flag literals; no shell.
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "modal run" in completed.stdout


def test_identity_coverage_rejects_a_real_rank_slice_overlap() -> None:
    global_identities = [f"unit-{index}" for index in range(128)]
    assert pii350_ddp_smoke.validate_rank_identity_coverage(
        global_identities=global_identities,
        rank_identities=[global_identities[:64], global_identities[64:]],
    ) == pii350_ddp_smoke.ordered_identity_digest(global_identities)
    with pytest.raises(RuntimeError, match="overlap"):
        pii350_ddp_smoke.validate_rank_identity_coverage(
            global_identities=global_identities,
            rank_identities=[global_identities[:64], global_identities[63:127]],
        )


def test_aggregate_throughput_uses_the_slowest_rank_duration() -> None:
    assert pii350_ddp_smoke.aggregate_real_tokens_per_second(
        global_real_tokens=1_280,
        rank_step_seconds=[0.2, 0.5],
    ) == pytest.approx(2_560.0)


def test_collective_readiness_requires_bidirectional_peer_access_and_nccl_sum() -> None:
    records = [
        {"rank": 0, "peer_rank": 1, "can_access_peer": True},
        {"rank": 1, "peer_rank": 0, "can_access_peer": True},
    ]
    assert pii350_ddp_smoke.validate_collective_readiness(peer_access_records=records, observed_all_reduce_sum=3)[
        "nccl_all_reduce"
    ] == {
        "inputs_by_rank": [1, 2],
        "expected_sum": 3,
        "observed_sum": 3,
    }
    with pytest.raises(RuntimeError, match="bidirectional CUDA peer access"):
        pii350_ddp_smoke.validate_collective_readiness(
            peer_access_records=[
                {"rank": 0, "peer_rank": 1, "can_access_peer": True},
                {"rank": 1, "peer_rank": 0, "can_access_peer": False},
            ],
            observed_all_reduce_sum=3,
        )
    with pytest.raises(RuntimeError, match="returned 2, expected 3"):
        pii350_ddp_smoke.validate_collective_readiness(peer_access_records=records, observed_all_reduce_sum=2)


def test_absolute_deadline_charges_parent_and_child_initialization_to_the_same_budget() -> None:
    """400 seconds of parent receipt/allocation/spawn plus 1,099 seconds in the child remains valid.

    The next second is the shared soft deadline.

    """
    outer_modal_entry = 1_000.0
    deadline = pii350_ddp_smoke.absolute_training_deadline(outer_started_monotonic=outer_modal_entry)

    assert deadline == 2_500.0
    assert not pii350_ddp_smoke.training_deadline_reached(absolute_deadline_monotonic=deadline, now_monotonic=2_499.0)
    assert pii350_ddp_smoke.training_deadline_reached(absolute_deadline_monotonic=deadline, now_monotonic=2_500.0)
