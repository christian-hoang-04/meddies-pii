from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[suspicious-subprocess-import]
# reason: the scout launch contract has to be observed as the CLI RENDERS it, not as an in-process
# reason: import renders it — the assertions pin that `--execute` is absent from the preflight command
# reason: and present in the launch one, which is the spend gate this module guards. The two call
# reason: sites carry their own argv reason.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
import json
import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.trainers import full_run

if TYPE_CHECKING:
    from collections.abc import Callable


def _qualified_artifact() -> dict[str, object]:
    return {
        "candidate": "encoder350",
        "batch_size": 192,
        **full_run.MANUAL_TRANSFER_ACCEPTANCES["encoder350"],
    }


def test_four_deterministic_contracts_are_exact_and_have_no_default_backend() -> None:
    rendered = full_run.render_all_full_runs()
    assert set(rendered) == {"base230", "encoder230", "encoder350", "pii350"}
    assert rendered["base230"]["backend"] == {
        "backend": "unsloth",
        "loader": "FastLanguageModel.from_pretrained",
        "image": "bioes-full-run-unsloth-isolated",
        "bf16": True,
        "gradient_checkpointing": "use_gradient_checkpointing='unsloth'",
        "attention_implementation": "unsloth_managed",
        "output_hidden_states": True,
        "packed_segment_isolation": True,
        "use_cache": False,
        "trust_remote_code": False,
    }
    for key in ("encoder230", "encoder350"):
        assert rendered[key]["backend"]["backend"] == "unsloth"
        assert rendered[key]["backend"]["loader"] == "FastModel.from_pretrained(auto_model=AutoModelForMaskedLM)"
        assert rendered[key]["backend"]["image"] == "bioes-full-run-unsloth-2026-7-4-encoder-isolated"
        assert rendered[key]["backend"]["gradient_checkpointing"] == "use_gradient_checkpointing='unsloth'"
        assert rendered[key]["backend"]["attention_implementation"] == "unsloth_managed"
        assert rendered[key]["backend"]["trust_remote_code"] is True
    assert rendered["pii350"]["backend"] == {
        "backend": "unsloth",
        "loader": "FastModel.from_pretrained(auto_model=AutoModelForTokenClassification)",
        "image": "bioes-full-run-unsloth-2026-7-4-encoder-isolated",
        "bf16": True,
        "gradient_checkpointing": "use_gradient_checkpointing='unsloth'",
        "attention_implementation": "unsloth_managed",
        "output_hidden_states": False,
        "packed_segment_isolation": True,
        "use_cache": False,
        "trust_remote_code": True,
    }
    with pytest.raises(ValueError, match="known candidate"):
        full_run.backend_for("default")


def test_fixed_full_population_pins_optimizer_and_final_eval() -> None:
    contract = full_run.render_full_run("encoder350")
    assert contract["gpu"] == "H100!"
    assert contract["seed"] == 3407
    assert contract["runtime"] == {
        "python_version": "3.11",
        "packages": [
            "torch==2.10.0",
            "transformers==5.2.0",
            "peft==0.19.1",
            "pyarrow==23.0.0",
            "datasets==4.3.0",
            "huggingface_hub==1.11.0",
            "nvidia-ml-py==13.590.44",
            "unsloth==2026.7.4",
            "unsloth_zoo==2026.7.4",
        ],
    }
    assert contract["packed_dataset"] == {
        "id": "Meddies/meddies-pii-mixed",
        "revision": "11fd43ec9ebb187e1d0f94fe77bcf1090a2a18ee",
        "config": "packed",
        "manifest_sha256": "7cadda8e81ef4b2a1111f37a8b508492158a983e90f54fecafdc24764f792969",
        "shard_count": 469,
        "raw_source_population": 1_000_000,
        "raw_source_revision": "f5d88d6ea0fbc6775d19097ceb406ceeaef19fdf",
        "accepted_row_count": 836_000,
        "rejected_row_count": 164_000,
        "packed_unit_count": 120_036,
        "permutation_seed": 3407,
        "permutation_digest": "f12ac0ba125c17b0d64058a0e2cd943f4f76eb3d272745e77a9b792054fa4c21",
        "real_token_count": 850_154_043,
        "physical_token_count": 983_334_912,
        "padding_token_count": 133_180_869,
        "packing_utilization": 0.864562045570899,
        "tokenizer_digests": {
            "causal": "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd",
            "encoder": "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711",
        },
        "compatibility_decision": "accepted",
        "boundary_contract": "lfm2_segment_isolation_v1",
        "boundary_token_count": 0,
        "max_length": 8192,
        "full_seeded_artifact": True,
        "row_limit": None,
        "train_limit": None,
        "first_n": None,
    }
    assert contract["evaluation"]["config"] == "eval"
    assert contract["evaluation"]["split"] == "train"
    assert contract["evaluation"]["expected_rows"] == 1700
    assert contract["evaluation"]["timing"] == "final_only"
    assert contract["optimizer"] == {
        "name": "adamw",
        "lr": 1e-4,
        "schedule": "constant",
        "fused": False,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
    }
    assert contract["training"]["epochs"] == 1
    assert contract["training"]["gradient_accumulation_steps"] == 1
    assert "trust_remote_code" not in contract["training"]["mechanics"]
    assert contract["training"]["full_epoch_geometry"] == {
        "packed_unit_count": 120_036,
        "optimizer_steps": 626,
        "final_physical_batch_size": 36,
        "max_physical_tokens_per_step": 1_572_864,
        "descriptive_only": True,
    }
    assert contract["training"]["lora"] == {
        "rank": 64,
        "alpha": 128,
        "dropout": 0.0,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
        ],
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("backend", "image"), "wrong-image", "backend"),
        (("gpu",), "H100", "gpu"),
        (("training", "epochs"), 2, "training"),
        (("packed_dataset", "train_limit"), 1, "packed_dataset"),
        (("budget", "gpu_only_budget_usd"), 9.999, "budget"),
    ],
)
def test_contract_gate_rejects_every_mutable_launch_escape(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    contract = full_run.render_full_run("encoder350")
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        full_run.validate_full_run_contract(contract)


def test_rendered_contract_copies_nested_defaults_and_manual_acceptance() -> None:
    contract = full_run.render_full_run("encoder230")
    contract["training"]["mechanics"]["lora"]["rank"] = 1
    contract["qualification"]["manual_transfer_acceptance"]["accepted_by"] = "other"

    later = full_run.render_full_run("encoder230")
    assert later["training"]["mechanics"]["lora"]["rank"] == 64
    assert later["qualification"]["manual_transfer_acceptance"]["accepted_by"] == "Ha"
    assert later == full_run.render_full_run("encoder230")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda contract: contract.__setitem__("seed", 0),
        lambda contract: contract.__setitem__("qualification", {"status": "wrong"}),
        lambda contract: contract.__setitem__("artifacts", {"final_checkpoint": False}),
        # reason: the `True` is DATA written into a contract dict to corrupt it, not a flag argument.
        # reason: The rule guards an unreadable boolean at a call site; `__setitem__(key, value)` has
        # reason: no such ambiguity, and this mutation IS the corruption case under test.
        lambda contract: contract.__setitem__("unexpected", True),  # ruff: ignore[boolean-positional-value-in-call]
    ],
)
def test_contract_gate_rejects_every_changed_or_extra_top_level_field(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    contract = full_run.render_full_run("encoder350")
    mutate(contract)
    with pytest.raises(ValueError, match="full-run contract mismatch"):
        full_run.validate_full_run_contract(contract)


def test_full_epoch_geometry_is_descriptive_and_exact_for_primary_batch_sizes() -> None:
    assert full_run.render_full_run("base230")["training"]["full_epoch_geometry"] == {
        "packed_unit_count": 120_036,
        "optimizer_steps": 289,
        "final_physical_batch_size": 228,
        "max_physical_tokens_per_step": 3_407_872,
        "descriptive_only": True,
    }
    assert full_run.render_full_run("encoder230")["training"]["full_epoch_geometry"] == {
        "packed_unit_count": 120_036,
        "optimizer_steps": 501,
        "final_physical_batch_size": 36,
        "max_physical_tokens_per_step": 1_966_080,
        "descriptive_only": True,
    }


def test_full_run_contract_pins_the_exact_ordered_bioes_label_vocabulary() -> None:
    contract = full_run.render_full_run("base230")
    assert contract["label_vocabulary"] == [
        "O",
        "B-address",
        "I-address",
        "E-address",
        "S-address",
        "B-company_name",
        "I-company_name",
        "E-company_name",
        "S-company_name",
        "B-date",
        "I-date",
        "E-date",
        "S-date",
        "B-email_address",
        "I-email_address",
        "E-email_address",
        "S-email_address",
        "B-human_name",
        "I-human_name",
        "E-human_name",
        "S-human_name",
        "B-id_number",
        "I-id_number",
        "E-id_number",
        "S-id_number",
        "B-phone_number",
        "I-phone_number",
        "E-phone_number",
        "S-phone_number",
        "B-private_url",
        "I-private_url",
        "E-private_url",
        "S-private_url",
        "B-secret",
        "I-secret",
        "E-secret",
        "S-secret",
    ]
    assert "label_vocabulary.json" in contract["artifacts"]["run_root_files"]


def test_full_launch_needs_exact_confirmation_and_attached_qualification() -> None:
    contract = full_run.render_full_run("encoder350")
    with pytest.raises(RuntimeError, match="requires --execute"):
        full_run.require_full_run_execute(
            contract,
            execute=False,
            confirmation="",
            qualification_artifact=_qualified_artifact(),
        )
    with pytest.raises(RuntimeError, match="qualification artifact"):
        full_run.require_full_run_execute(
            contract,
            execute=True,
            confirmation=full_run.FULL_RUN_CONFIRMATION,
            qualification_artifact=None,
        )
    wrong = {**_qualified_artifact(), "batch_size": 200}
    with pytest.raises(RuntimeError, match="manual transfer acceptance"):
        full_run.require_full_run_execute(
            contract,
            execute=True,
            confirmation=full_run.FULL_RUN_CONFIRMATION,
            qualification_artifact=wrong,
        )
    full_run.require_full_run_execute(
        contract,
        execute=True,
        confirmation=full_run.FULL_RUN_CONFIRMATION,
        qualification_artifact=_qualified_artifact(),
    )


@pytest.mark.parametrize("candidate", ["base230"])
def test_base230_refuses_any_payload_except_exact_ha_live_acceptance(candidate: str) -> None:
    with pytest.raises(RuntimeError, match="exact accepted"):
        full_run.require_full_run_execute(
            full_run.render_full_run(candidate),
            execute=True,
            confirmation=full_run.FULL_RUN_CONFIRMATION,
            qualification_artifact={
                "candidate": candidate,
                "batch_size": 1,
                "execution_id": "file",
            },
        )


@pytest.mark.parametrize("candidate", ["encoder230", "encoder350", "pii350"])
def test_transfer_accepted_candidate_requires_its_recorded_manual_provenance(
    candidate: str,
) -> None:
    expected = {
        "candidate": candidate,
        "batch_size": full_run.EXPECTED_FULL_BATCHES[candidate],
        **full_run.MANUAL_TRANSFER_ACCEPTANCES[candidate],
    }
    full_run.require_full_run_execute(
        full_run.render_full_run(candidate),
        execute=True,
        confirmation=full_run.FULL_RUN_CONFIRMATION,
        qualification_artifact=expected,
    )
    with pytest.raises(RuntimeError, match="manual transfer acceptance"):
        full_run.require_full_run_execute(
            full_run.render_full_run(candidate),
            execute=True,
            confirmation=full_run.FULL_RUN_CONFIRMATION,
            qualification_artifact={"candidate": candidate, "batch_size": 1},
        )


def test_encoder350_evidence_selects_192_and_rejects_stable_slower_200() -> None:
    qualification = full_run.qualification_for("encoder350")
    assert qualification.selected_batch_size == 192
    assert qualification.execution_id == "ded3bd6ec1c04ee486500308f4f05035"
    assert qualification.median_real_tokens_per_second == 11248.277676680911
    assert "6905.214375535371" in qualification.rationale


def test_resume_rejects_pin_digest_and_cursor_mismatches() -> None:
    contract = full_run.render_full_run("encoder350")
    state = full_run.ResumeState(
        "encoder350",
        contract["config_digest"],
        full_run.PACKED_DATASET_REVISION,
        full_run.PACKED_MANIFEST_SHA256,
        full_run.EVAL_DATASET_REVISION,
        10,
        2,
    )
    full_run.validate_resume_state(contract, state)
    for bad in (
        replace(state, config_digest="wrong"),
        replace(state, packed_revision="wrong"),
        replace(state, eval_revision="wrong"),
        replace(state, packed_cursor=full_run.PACKED_UNIT_COUNT + 1),
    ):
        with pytest.raises(RuntimeError):
            full_run.validate_resume_state(contract, bad)


def test_full_run_budget_is_telemetry_only_and_deadline_reserves_shutdown() -> None:
    contract = full_run.render_full_run("base230")
    budget = contract["budget"]
    assert budget["gpu_only_target_usd"] == 10.0
    assert budget["gpu_only_timeout_estimate_usd"] == 9.873
    assert budget["all_in_timeout_estimate_usd"] == 11.62332
    assert budget["training_deadline_seconds"] == 8_400
    assert budget["shutdown_reserve_seconds"] == 600


def test_eval_inventory_requires_pinned_eval_train_and_1700_before_encoding() -> None:
    inventory = {
        "id": full_run.EVAL_DATASET_ID,
        "revision": full_run.EVAL_DATASET_REVISION,
        "config": "eval",
        "split": "train",
        "rows": 1700,
    }
    full_run.validate_pinned_eval_inventory(inventory)
    with pytest.raises(RuntimeError, match="1,700"):
        full_run.validate_pinned_eval_inventory({**inventory, "rows": 1699})


def test_full_run_renderer_exposes_no_encoder_or_pii_qualification_probes() -> None:
    assert not hasattr(full_run, "render_missing_candidate_probe")
    assert not hasattr(full_run, "render_probe_command")


def test_full_run_command_renderer_has_one_explicit_candidate_and_exact_profile() -> None:
    command = full_run.render_full_run_command("encoder350")
    assert "--candidate encoder350" in command
    assert full_run.FULL_RUN_CONFIRMATION in command
    assert "ded3bd6ec1c04ee486500308f4f05035" in command
    assert "SUCCESSFUL_QUALIFICATION_JSON" not in command
    assert "MODAL_PROFILE=huyhoang041100" in command
    assert "uv run modal run --detach --timestamps -m" in command
    assert "qualification_evidence_backend" in command
    assert "source_backend" not in command
    assert full_run.render_full_run("encoder350")["backend"]["backend"] == "unsloth"


@pytest.mark.parametrize(
    ("candidate", "profile"),
    [
        ("base230", "hahuyhoang411"),
        ("encoder230", "huyhoangha0411"),
        ("encoder350", "huyhoang041100"),
        ("pii350", "huyhoang041100"),
    ],
)
def test_preflight_command_renders_the_cpu_receipt_prerequisite(candidate: str, profile: str) -> None:
    command = full_run.render_full_run_preflight_command(candidate)
    assert command == (
        f"MODAL_PROFILE={profile} uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate {candidate} --preflight-assets"
    )
    assert "--execute" not in command


def test_transfer_command_renders_the_exact_recorded_acceptance() -> None:
    command = full_run.render_full_run_command("encoder230")
    assert "ha-accepted-native-base230-b240-transfer" in command
    assert "SUCCESSFUL_QUALIFICATION_JSON" not in command
    assert "qualification_evidence_candidate" in command
    assert "source_candidate" not in command


def test_full_lane_or_multi_candidate_names_cannot_be_rendered_or_dispatched() -> None:
    with pytest.raises(ValueError, match="known candidate"):
        full_run.render_full_run("base230,encoder230")
    contract = full_run.render_full_run("encoder350")
    contract["candidate_key"] = "full_lane"
    with pytest.raises(ValueError, match="known candidate"):
        full_run.validate_full_run_contract(contract)


def test_pure_renderer_never_imports_modal_or_dispatches() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.trainers.full_run",
            "--render-config",
            "encoder350",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rendered = json.loads(completed.stdout)
    assert rendered["candidate_key"] == "encoder350"
    imports = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import meddies_pii.training.bioes.trainers.full_run; assert 'modal' not in "
                "sys.modules; print('pure')"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert imports.stdout.strip() == "pure"
    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.trainers.full_run",
            "--render-preflight-command",
            "encoder350",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--preflight-assets" in preflight.stdout
    assert "--execute" not in preflight.stdout
    fallback = subprocess.run(
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.trainers.full_run",
            "--render-base230-oom-fallback-command",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--base230-oom-fallback" in fallback.stdout
    assert "--execute" in fallback.stdout


def test_base230_b416_is_the_only_launchable_fresh_primary_contract() -> None:
    contract = full_run.render_full_run("base230")
    assert contract["training"]["batch_size"] == 416
    assert contract["training"]["fresh_run"] is True
    assert contract["training"]["resume_from_checkpoint"] is None
    assert contract["training"]["optimizer_step_cap"] is None
    assert full_run.EXPECTED_FULL_BATCHES["base230"] == 416
    command = full_run.render_full_run_command("base230")
    assert "--resume" not in command
    assert "max-steps" not in command


def test_base230_b384_fallback_requires_the_exact_persisted_b416_oom() -> None:
    fallback = full_run.render_base230_oom_fallback_contract()
    evidence = full_run.BASE230_B416_OOM_EVIDENCE
    assert fallback["training"]["batch_size"] == 384
    assert fallback["fallback_launch"] == {
        "automatic_launch": False,
        "required_primary_oom_evidence": evidence,
        "requires_primary_action": full_run.BASE230_OOM_FALLBACK_PRIMARY_ACTION,
    }
    assert fallback["backend"] == full_run.render_full_run("base230")["backend"]
    assert fallback["training"]["lora"] == full_run.render_full_run("base230")["training"]["lora"]
    command = full_run.render_base230_oom_fallback_command()
    assert "--base230-oom-fallback" in command
    assert "--execute" in command
    assert "--resume" not in command
    assert "ap-3oMP3pkzSyphMm539IxTzl" in command
    with pytest.raises(RuntimeError, match="OOM evidence"):
        full_run.require_base230_oom_fallback_execute(
            fallback,
            primary_oom_evidence={**evidence, "failed_optimizer_step": 3},
            primary_action=full_run.BASE230_OOM_FALLBACK_PRIMARY_ACTION,
        )
    with pytest.raises(RuntimeError, match="primary action"):
        full_run.require_base230_oom_fallback_execute(
            fallback,
            primary_oom_evidence=evidence,
            primary_action="automatic",
        )
    full_run.require_base230_oom_fallback_execute(
        fallback,
        primary_oom_evidence=evidence,
        primary_action=full_run.BASE230_OOM_FALLBACK_PRIMARY_ACTION,
    )


def test_pii350_utilization_run_requires_completed_b192_evidence_and_action() -> None:
    contract = full_run.render_pii350_utilization_run_contract()
    evidence = full_run.PII350_B192_COMPLETE_EVIDENCE
    assert contract["training"]["batch_size"] == 256
    assert contract["training"]["fresh_run"] is True
    assert contract["training"]["resume_from_checkpoint"] is None
    assert contract["training"]["optimizer_step_cap"] is None
    assert contract["config_digest"] == "25580ae7fca399e23a5cad1db4f0d3f04fc13fe3b3e41cc0f904c88f1628e14c"
    assert contract["manual_launch"] == {
        "purpose": "maximize_stable_h100_utilization",
        "automatic_launch": False,
        "required_source_run_evidence": evidence,
        "requires_primary_action": full_run.PII350_UTILIZATION_PRIMARY_ACTION,
        "modal_profile": "retraction",
        "target_total_device_memory_percent": {"lower": 85, "upper": 90},
    }
    assert contract["backend"] == full_run.render_full_run("pii350")["backend"]
    assert contract["training"]["lora"] == full_run.render_full_run("pii350")["training"]["lora"]
    command = full_run.render_pii350_utilization_run_command()
    assert command.startswith("MODAL_PROFILE=retraction ")
    assert "--pii350-utilization-run" in command
    assert "--execute" in command
    assert "--resume" not in command
    prewarm = full_run.render_pii350_utilization_prewarm_command()
    preflight = full_run.render_pii350_utilization_preflight_command()
    assert prewarm.startswith("MODAL_PROFILE=retraction ")
    assert "--prewarm-assets" in prewarm
    assert preflight.startswith("MODAL_PROFILE=retraction ")
    assert "--preflight-assets" in preflight
    with pytest.raises(RuntimeError, match="source run evidence"):
        full_run.require_pii350_utilization_run_execute(
            contract,
            source_run_evidence={**evidence, "optimizer_steps": 58},
            primary_action=full_run.PII350_UTILIZATION_PRIMARY_ACTION,
        )
    with pytest.raises(RuntimeError, match="primary action"):
        full_run.require_pii350_utilization_run_execute(
            contract,
            source_run_evidence=evidence,
            primary_action="automatic",
        )
    full_run.require_pii350_utilization_run_execute(
        contract,
        source_run_evidence=evidence,
        primary_action=full_run.PII350_UTILIZATION_PRIMARY_ACTION,
    )


def test_pii350_capacity_run_uses_authorized_batch256_and_r128a256() -> None:
    baseline = full_run.render_full_run("pii350")
    capacity = full_run.render_pii350_capacity_run_contract()
    evidence = full_run.PII350_B192_COMPLETE_EVIDENCE
    assert capacity["training"]["batch_size"] == 256
    assert full_run.PII350_CAPACITY_PRIMARY_ACTION == "HA_AUTHORIZE_PII350_R128A256_BATCH256_FULL_BUDGET"
    assert capacity["training"]["lora"] == {
        **baseline["training"]["lora"],
        "rank": 128,
        "alpha": 256,
    }
    assert capacity["training"]["mechanics"]["lora"] == {
        **baseline["training"]["mechanics"]["lora"],
        "rank": 128,
        "alpha": 256,
    }
    assert capacity["training"]["fresh_run"] is True
    assert capacity["training"]["resume_from_checkpoint"] is None
    assert capacity["training"]["optimizer_step_cap"] is None
    assert capacity["config_digest"] == "c0a5550264dabefa7403f59c48520cc4e59d9480aa5d29fa658e8af0304f1085"
    assert capacity["manual_launch"] == {
        "purpose": "measure_lora_capacity_at_fixed_h100_budget",
        "automatic_launch": False,
        "required_source_run_evidence": evidence,
        "requires_primary_action": full_run.PII350_CAPACITY_PRIMARY_ACTION,
        "modal_profile": "retraction",
        "oom_policy": {
            "automatic_fallback": False,
            "manual_fallback_batch_size": 160,
            "requires_persisted_primary_oom": True,
        },
    }
    assert capacity["backend"] == baseline["backend"]
    assert capacity["packed_dataset"] == baseline["packed_dataset"]
    assert capacity["optimizer"] == baseline["optimizer"]
    command = full_run.render_pii350_capacity_run_command()
    assert command.startswith("MODAL_PROFILE=retraction ")
    assert "--pii350-capacity-run" in command
    assert "R128A256_BATCH256" in command
    assert "--execute" in command
    assert "--resume" not in command
    with pytest.raises(RuntimeError, match="source run evidence"):
        full_run.require_pii350_capacity_run_execute(
            capacity,
            source_run_evidence={**evidence, "optimizer_steps": 58},
            primary_action=full_run.PII350_CAPACITY_PRIMARY_ACTION,
        )
    with pytest.raises(RuntimeError, match="primary action"):
        full_run.require_pii350_capacity_run_execute(
            capacity,
            source_run_evidence=evidence,
            primary_action="automatic",
        )
    full_run.require_pii350_capacity_run_execute(
        capacity,
        source_run_evidence=evidence,
        primary_action=full_run.PII350_CAPACITY_PRIMARY_ACTION,
    )


@pytest.mark.parametrize(
    (
        "candidate",
        "expected_batch",
        "expected_action",
        "expected_profile",
        "expected_digest",
    ),
    [
        (
            "base230",
            320,
            "HA_AUTHORIZE_BASE230_B320_R64A128_FULL_BUDGET",
            "meddies-pii",
            "b6c936c2bc48816f570f6185094c0c87b08c10fd080a571b1e2b63ddc1e3c1b4",
        ),
        (
            "encoder230",
            288,
            "HA_AUTHORIZE_ENCODER230_B288_R64A128_FULL_BUDGET",
            "hahuyhoang041100",
            "2fd30d386077b03af086c337145765a8cd6f05f03032da6240d9437b7789c681",
        ),
    ],
)
def test_m230_comparison_run_pins_batch_backend_version_and_user_action(
    candidate: str,
    expected_batch: int,
    expected_action: str,
    expected_profile: str,
    expected_digest: str,
) -> None:
    baseline = full_run.render_full_run(candidate)
    comparison = full_run.render_m230_comparison_run_contract(candidate)
    assert comparison["training"]["batch_size"] == expected_batch
    assert full_run.M230_COMPARISON_BATCH_SIZES[candidate] == expected_batch
    assert comparison["config_digest"] == expected_digest
    assert comparison["training"]["lora"] == {
        **baseline["training"]["lora"],
        "rank": 64,
        "alpha": 128,
    }
    assert comparison["training"]["mechanics"]["lora"] == {
        **baseline["training"]["mechanics"]["lora"],
        "rank": 64,
        "alpha": 128,
    }
    assert comparison["training"]["fresh_run"] is True
    assert comparison["training"]["resume_from_checkpoint"] is None
    assert comparison["training"]["optimizer_step_cap"] is None
    assert comparison["runtime"]["packages"][-2:] == [
        "unsloth==2026.7.4",
        "unsloth_zoo==2026.7.4",
    ]
    assert comparison["manual_launch"] == {
        "purpose": "compare_230m_base_and_encoder_at_fixed_budget",
        "automatic_launch": False,
        "requires_primary_action": full_run.M230_COMPARISON_PRIMARY_ACTIONS[candidate],
        "modal_profile": expected_profile,
        "oom_policy": {"automatic_fallback": False},
    }
    assert comparison["packed_dataset"] == baseline["packed_dataset"]
    assert comparison["evaluation"] == baseline["evaluation"]
    assert comparison["optimizer"] == baseline["optimizer"]
    command = full_run.render_m230_comparison_run_command(candidate)
    assert command.startswith(f"MODAL_PROFILE={expected_profile} ")
    assert f"--candidate {candidate}" in command
    assert "--m230-comparison-run" in command
    assert expected_action in command
    assert full_run.M230_COMPARISON_PRIMARY_ACTIONS[candidate] == expected_action
    assert "--execute" in command
    assert "--resume" not in command
    prewarm = full_run.render_m230_comparison_prewarm_command(candidate)
    preflight = full_run.render_m230_comparison_preflight_command(candidate)
    assert prewarm.startswith(f"MODAL_PROFILE={expected_profile} ")
    assert "--prewarm-assets" in prewarm
    assert preflight.startswith(f"MODAL_PROFILE={expected_profile} ")
    assert "--preflight-assets" in preflight
    with pytest.raises(RuntimeError, match="primary action"):
        full_run.require_m230_comparison_run_execute(
            comparison,
            candidate_key=candidate,
            primary_action="automatic",
        )
    full_run.require_m230_comparison_run_execute(
        comparison,
        candidate_key=candidate,
        primary_action=full_run.M230_COMPARISON_PRIMARY_ACTIONS[candidate],
    )
    opposite_candidate = "encoder230" if candidate == "base230" else "base230"
    with pytest.raises(RuntimeError, match="primary action"):
        full_run.require_m230_comparison_run_execute(
            comparison,
            candidate_key=candidate,
            primary_action=full_run.M230_COMPARISON_PRIMARY_ACTIONS[opposite_candidate],
        )


def test_m230_comparison_run_rejects_non_230m_candidate_and_crossed_contract() -> None:
    with pytest.raises(ValueError, match="base230 or encoder230"):
        full_run.render_m230_comparison_run_contract("encoder350")
    with pytest.raises(RuntimeError, match="exact rendered contract"):
        full_run.require_m230_comparison_run_execute(
            full_run.render_m230_comparison_run_contract("base230"),
            candidate_key="encoder230",
            primary_action=full_run.M230_COMPARISON_PRIMARY_ACTIONS["encoder230"],
        )


def test_base230_b240_resume_state_cannot_match_the_new_b416_contract() -> None:
    contract = full_run.render_full_run("base230")
    old_state = full_run.ResumeState(
        "base230",
        "b240-config-digest",
        full_run.PACKED_DATASET_REVISION,
        full_run.PACKED_MANIFEST_SHA256,
        full_run.EVAL_DATASET_REVISION,
        2_400,
        10,
    )
    with pytest.raises(RuntimeError, match="config digest"):
        full_run.validate_resume_state(contract, old_state)


def test_base230_uses_exact_ha_accepted_three_step_unsloth_evidence() -> None:
    assert full_run.qualification_for("base230").status == "accepted_live"
    full_run.require_full_run_execute(
        full_run.render_full_run("base230"),
        execute=True,
        confirmation=full_run.FULL_RUN_CONFIRMATION,
        qualification_artifact=full_run.BASE230_LIVE_ACCEPTANCE,
    )


def test_full_run_hard_timeout_is_exact_and_records_gpu_and_all_in_costs() -> None:
    assert full_run.FULL_RUN_HARD_TIMEOUT_SECONDS == 9_000
    contract = full_run.render_full_run("base230")
    assert contract["budget"]["gpu_only_timeout_estimate_usd"] == 9.873
    assert contract["budget"]["all_in_timeout_estimate_usd"] == 11.62332
    assert contract["budget"]["budget_enforcement"] == ("Modal hard timeout only; no early runtime budget stop")


def test_base230_resets_gate_seed_immediately_before_head_construction() -> None:
    import inspect

    from meddies_pii.training.bioes.trainers import full_run_runtime

    source = inspect.getsource(full_run_runtime._unsloth_tagger)
    assert source.index("torch.manual_seed(GATE_SEED)") < source.index("HiddenStateTokenTagger(")


def test_full_runtime_uses_unsloth_fastmodel_for_encoders_and_native_packing() -> None:
    import inspect

    from meddies_pii.training.bioes.trainers import full_run_runtime

    source = inspect.getsource(full_run_runtime)
    assert "from transformers import AutoModelForMaskedLM" in source
    assert "AutoModelForTokenClassification" in source
    assert "FastModel.from_pretrained" in source
    assert "auto_model=auto_model" in source
    assert "AutoModelForMaskedLM" in source
    assert "encoder FastModel wrapper has no .lfm2 encoder body" in source
    assert "packed_segment_isolation=True" in source
    assert "torch.isfinite(loss)" in source
    assert "LambdaLR" not in source


def test_encoder_runtime_loads_unsloth_before_transformers_and_peft() -> None:
    import inspect

    from meddies_pii.training.bioes.trainers import full_run_runtime

    source = inspect.getsource(full_run_runtime._unsloth_encoder_tagger)
    unsloth_load = 'fast_model = import_module("unsloth").FastModel'
    assert source.index(unsloth_load) < source.index("from peft import TaskType")
    assert source.index(unsloth_load) < source.index("from transformers import AutoModel")


def test_every_unsloth_import_site_patches_before_transformers_and_peft() -> None:
    """Unsloth patches transformers and peft at import time, so it must import first everywhere.

    Guarding one function let a mechanical import sort move unsloth below peft/transformers in the
    encoder350 probe, where no local test runs. This walks every deferred import block in the
    shipped package instead, so the contract cannot regress at a site nobody thought to pin.
    """
    import ast
    from pathlib import Path

    import meddies_pii

    assert meddies_pii.__file__ is not None
    package_root = Path(meddies_pii.__file__).parent
    patched = ("transformers", "peft")
    checked = 0
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            imports = [child for child in ast.walk(node) if isinstance(child, (ast.Import, ast.ImportFrom))]
            unsloth = [
                child.lineno
                for child in imports
                if isinstance(child, ast.ImportFrom)
                and child.module is not None
                and child.module.split(".")[0] == "unsloth"
            ]
            unsloth.extend(
                child.lineno
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "import_module"
                and len(child.args) == 1
                and isinstance(child.args[0], ast.Constant)
                and child.args[0].value == "unsloth"
            )
            if not unsloth:
                continue
            checked += 1
            blocked = [
                child.lineno
                for child in imports
                if isinstance(child, ast.ImportFrom) and child.module is not None and child.module.split(".")[0] in patched
            ]
            if not blocked:
                continue
            where = f"{path.relative_to(package_root)}:{node.name}"
            assert max(unsloth) < min(blocked), f"{where} imports unsloth after transformers/peft"

    assert checked >= 4, f"expected every known unsloth import site to be walked, saw {checked}"


@pytest.mark.parametrize(
    ("candidate", "loader"),
    [
        ("encoder230", "FastModel.from_pretrained(auto_model=AutoModelForMaskedLM)"),
        ("encoder350", "FastModel.from_pretrained(auto_model=AutoModelForMaskedLM)"),
        (
            "pii350",
            "FastModel.from_pretrained(auto_model=AutoModelForTokenClassification)",
        ),
    ],
)
def test_every_encoder_contract_uses_its_actual_fastmodel_loader(candidate: str, loader: str) -> None:
    contract = full_run.render_full_run(candidate)
    assert contract["backend"]["loader"] == loader


def test_packed_contract_distinguishes_source_rows_from_accepted_and_packed_units() -> None:
    packed = full_run.render_full_run("base230")["packed_dataset"]
    assert packed["raw_source_population"] == 1_000_000
    assert packed["accepted_row_count"] == 836_000
    assert packed["packed_unit_count"] == 120_036
    assert packed["boundary_contract"] == "lfm2_segment_isolation_v1"
    assert packed["boundary_token_count"] == 0


@pytest.mark.parametrize("candidate", ["base230", "encoder230", "encoder350", "pii350"])
def test_full_run_contract_reports_actual_hidden_state_and_packing_mode(
    candidate: str,
) -> None:
    backend = full_run.render_full_run(candidate)["backend"]
    assert backend["packed_segment_isolation"] is True
    assert backend["output_hidden_states"] is (candidate == "base230")


def test_resume_cursor_is_bounded_by_packed_units_not_raw_source_rows() -> None:
    contract = full_run.render_full_run("encoder350")
    state = full_run.ResumeState(
        "encoder350",
        contract["config_digest"],
        full_run.PACKED_DATASET_REVISION,
        full_run.PACKED_MANIFEST_SHA256,
        full_run.EVAL_DATASET_REVISION,
        120_036,
        1,
    )
    full_run.validate_resume_state(contract, state)
    with pytest.raises(RuntimeError, match="packed cursor"):
        full_run.validate_resume_state(
            contract,
            replace(state, packed_cursor=120_037),
        )


def test_full_runtime_uses_segment_isolation_and_keeps_the_final_partial_batch() -> None:
    import inspect

    from meddies_pii.training.bioes.trainers import full_run_runtime

    source = inspect.getsource(full_run_runtime)
    assert "packed_segment_isolation=True" in source
    assert "request_hidden_states=False" in source
    assert "def _physical_batches(" in source


def test_classifier_digest_is_deterministic_and_seed_sensitive() -> None:
    import torch

    from meddies_pii.training.bioes.trainers.full_run_runtime import _classifier_digest

    torch.manual_seed(3407)
    first = torch.nn.Linear(4, 37, dtype=torch.bfloat16)
    torch.manual_seed(3407)
    second = torch.nn.Linear(4, 37, dtype=torch.bfloat16)
    torch.manual_seed(3408)
    different = torch.nn.Linear(4, 37, dtype=torch.bfloat16)

    assert _classifier_digest(first) == _classifier_digest(second)
    assert _classifier_digest(first) != _classifier_digest(different)


def test_first_packed_batch_order_attestation_hashes_source_identities_only() -> None:
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _packed_order_attestation,
    )

    rows = [
        {"row_uids": ["a", "b"], "input_ids": [1, 2]},
        {"row_uids": ["c"], "input_ids": [3, 4]},
    ]
    attestation = _packed_order_attestation(rows, packed_cursor=17)

    assert attestation["first_packed_unit_cursor"] == 17
    assert attestation["last_packed_unit_cursor"] == 18
    assert attestation["source_identity_count"] == 3
    digest = attestation["ordered_source_identity_sha256"]
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_physical_batches_preserve_the_final_36_packed_units() -> None:
    from meddies_pii.training.bioes.trainers.full_run_runtime import _physical_batches

    rows = iter({"row_uids": [str(index)]} for index in range(276))

    assert [len(batch) for batch in _physical_batches(rows, 240)] == [240, 36]


def test_final_eval_uses_inference_only_eval_rows_without_training_preparation_filter() -> None:
    import inspect

    from meddies_pii.training.bioes.trainers import full_run_runtime

    source = inspect.getsource(full_run_runtime._final_evaluate)
    assert "_prepare_rows" not in source
    assert "_load_final_eval_rows" in source
    assert "truncation" in source


def test_encoder_runtime_pin_changes_only_runtime_packages_and_backend_image() -> None:
    from copy import deepcopy

    historical_digests = {
        "encoder230": "6a6cce4e33b1e795dd8887289e24974d0ec92ac7305969f48ca536696245e18d",
        "encoder350": "8d4f8879be2434341b9a917fb86ec583e3969f7ef558340604b29eabb194889c",
        "pii350": "7595c1fb523e8918c25c6c4e5b038e088c1b842700f71be7b21347f01958081c",
    }
    for candidate, historical_digest in historical_digests.items():
        current = full_run.render_full_run(candidate)
        historical = deepcopy(current)
        historical.pop("config_digest")
        historical["runtime"]["packages"] = list(full_run.FULL_RUN_IMAGE_PACKAGES)
        historical["backend"]["image"] = "bioes-full-run-unsloth-isolated"
        assert full_run._canonical_digest(historical) == historical_digest
        assert current["config_digest"] != historical_digest


def test_candidate_runtime_package_tuples_are_isolated_and_base_is_unchanged() -> None:
    base = full_run.render_full_run("base230")
    assert base["runtime"]["packages"] == list(full_run.FULL_RUN_IMAGE_PACKAGES)
    for candidate in ("encoder230", "encoder350", "pii350"):
        rendered = full_run.render_full_run(candidate)
        assert rendered["runtime"]["packages"] == list(full_run.ENCODER_FULL_RUN_IMAGE_PACKAGES)
        assert rendered["runtime"]["packages"][-2:] == [
            "unsloth==2026.7.4",
            "unsloth_zoo==2026.7.4",
        ]
        rendered["runtime"]["packages"].append("leak")
        assert full_run.render_full_run(candidate)["runtime"]["packages"] == list(full_run.ENCODER_FULL_RUN_IMAGE_PACKAGES)


def test_encoder350_unsloth_step1_evidence_is_recorded_without_mutating_full_contract() -> None:
    assert full_run.ENCODER350_UNSLOTH_STEP1_QUALIFICATION == {
        "app_id": "ap-Yuc2x2y2LN1VEtKe6KM9er",
        "candidate": "encoder350",
        "runtime_backend": "unsloth",
        "unsloth": "2026.7.4",
        "unsloth_zoo": "2026.7.4",
        "batch_size": 192,
        "optimizer_steps": 1,
        "status": "ok",
    }


@pytest.mark.parametrize(
    ("profile", "batch_size", "milestone_step", "action"),
    [
        (
            "meddies-pii",
            192,
            40,
            "HA_AUTHORIZE_PII350_R128A256_B192_MILESTONE7680_FULL_BUDGET",
        ),
        (
            "anhthunguyenump",
            160,
            48,
            "HA_AUTHORIZE_PII350_R128A256_B160_MILESTONE7680_FULL_BUDGET",
        ),
    ],
)
def test_pii350_milestone_decision_contracts_are_exact_and_continue_after_equal_cursor(
    profile: str,
    batch_size: int,
    milestone_step: int,
    action: str,
) -> None:
    decision = full_run.render_pii350_milestone_run_contract(profile)
    assert decision["config_digest"] == full_run.render_pii350_milestone_run_contract(profile)["config_digest"]
    assert len(decision["config_digest"]) == 64
    assert decision["training"]["batch_size"] == batch_size
    assert decision["training"]["lora"]["rank"] == 128
    assert decision["training"]["lora"]["alpha"] == 256
    assert decision["evaluation"]["timing"] == "milestone_and_final"
    assert decision["training"]["checkpoint_every_optimizer_steps"] == 10
    assert decision["training"]["optimizer_step_cap"] is None
    assert decision["training"]["milestone"] == {
        "optimizer_step": milestone_step,
        "packed_cursor": 7680,
        "evaluation_rows": 1700,
        "checkpoint": True,
        "evaluate": True,
        "continues_training": True,
    }
    assert decision["budget"] == {
        "all_in_credit_ceiling_usd": 9.80,
        "all_in_observed_runtime_multiplier": 1.177284,
        "all_in_projected_timeout_estimate_usd": 9.6525,
        "budget_rationale": (
            "USD 9.80 is an all-in account credit; equal H100 wall time uses a conservative USD 8.20 H100 ceiling"
        ),
        "gpu_only_target_usd": 8.20,
        "gpu_only_rate_usd_per_second": 0.001097,
        "hard_timeout_seconds": 7474,
        "shutdown_reserve_seconds": 600,
        "training_deadline_seconds": 6874,
        "gpu_only_timeout_estimate_usd": 8.198978,
        "budget_enforcement": "Modal hard timeout only; no early runtime budget stop",
    }
    assert decision["manual_launch"]["modal_profile"] == profile
    assert decision["manual_launch"]["requires_primary_action"] == action
    qualification = decision["qualification"]
    assert qualification["selected_batch_size"] == batch_size
    assert qualification["source_run"] == {
        "candidate": "pii350",
        "runtime_backend": "unsloth",
        "batch_size": 192,
        "execution_id": full_run.PII350_B192_COMPLETE_EVIDENCE["execution_id"],
        "status": "completed_ok_deadline_reached",
    }
    if profile == "meddies-pii":
        assert qualification["status"] == "accepted_live"
        assert qualification["transfer_basis"] is None
    else:
        assert qualification["status"] == "accepted_transfer"
        assert qualification["transfer_basis"] == ("accepted_lower_batch_transfer_from_strict_unsloth_pii350_b192")
    assert decision["runtime"]["packages"][-2:] == [
        "unsloth==2026.7.4",
        "unsloth_zoo==2026.7.4",
    ]


def test_pii350_milestone_run_gate_rejects_wrong_profile_budget_batch_cursor_and_action() -> None:
    decision = full_run.render_pii350_milestone_run_contract("meddies-pii")
    action = decision["manual_launch"]["requires_primary_action"]
    evidence = full_run.PII350_B192_COMPLETE_EVIDENCE
    with pytest.raises(ValueError, match="profile"):
        full_run.render_pii350_milestone_run_contract("wrong")
    with pytest.raises(RuntimeError, match="primary action"):
        full_run.require_pii350_milestone_run_execute(
            decision,
            profile="meddies-pii",
            source_run_evidence=evidence,
            primary_action="HA_AUTHORIZE_PII350_R128A256_B160_MILESTONE7680_FULL_BUDGET",
        )
    for path, value, message in (
        (("budget", "hard_timeout_seconds"), 8934, "exact rendered contract"),
        (("training", "batch_size"), 160, "exact rendered contract"),
        (("training", "milestone", "packed_cursor"), 7_679, "exact rendered contract"),
        (("training", "milestone", "optimizer_step"), 48, "exact rendered contract"),
        (("training", "milestone", "evaluate"), False, "exact rendered contract"),
        (("training", "optimizer_step_cap"), 40, "exact rendered contract"),
    ):
        changed = full_run.render_pii350_milestone_run_contract("meddies-pii")
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(RuntimeError, match=message):
            full_run.require_pii350_milestone_run_execute(
                changed,
                profile="meddies-pii",
                source_run_evidence=evidence,
                primary_action=action,
            )


def test_pii350_milestone_commands_render_cpu_preflight_and_explicit_h100_launch_only() -> None:
    preflight = full_run.render_pii350_milestone_run_preflight_command("meddies-pii")
    launch = full_run.render_pii350_milestone_run_command("anhthunguyenump")
    assert preflight.startswith("MODAL_PROFILE=meddies-pii ")
    assert "--pii350-milestone-run meddies-pii --preflight-assets" in preflight
    assert "--execute" not in preflight
    assert launch.startswith("MODAL_PROFILE=anhthunguyenump ")
    assert "--pii350-milestone-run anhthunguyenump --execute" in launch
    assert "HA_AUTHORIZE_PII350_R128A256_B160_MILESTONE7680_FULL_BUDGET" in launch
    rendered_preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.trainers.full_run",
            "--render-pii350-milestone-run-preflight-command",
            "meddies-pii",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert rendered_preflight.stdout.strip() == preflight


@pytest.mark.parametrize(
    ("arm", "lr", "expected_profile", "expected_action", "expected_digest"),
    [
        (
            "lr1e-4",
            1e-4,
            "hocdatacamp-hoangthu",
            "HA_AUTHORIZE_PII350_R128A256_B128_LR1E4_SCOUT",
            "8bfbbc035d4f9762617f0b8c3d6e9996d443f3a3f53b836b4108ffbfb091afed",
        ),
        (
            "lr2e-4",
            2e-4,
            "hocdatacamp-hoangthu",
            "HA_AUTHORIZE_PII350_R128A256_B128_LR2E4_SCOUT",
            "adea0c34b2bfc007f09455c8f68239ef9478c975d7afc06bb807f4053dc45128",
        ),
        (
            "lr3e-4",
            3e-4,
            "hocdatacamp-hoangthu",
            "HA_AUTHORIZE_PII350_R128A256_B128_LR3E4_SCOUT",
            "ba280317b4c9bceee7de56501b02434a20c123b3cafb781f9c24fb673832346a",
        ),
    ],
)
def test_pii350_scout_contracts_are_exact_batch128_anchors(
    arm: str,
    lr: float,
    expected_profile: str,
    expected_action: str,
    expected_digest: str,
) -> None:
    scout = full_run.render_pii350_scout_contract(arm)
    assert scout["config_digest"] == expected_digest
    assert scout["training"]["batch_size"] == 128
    assert scout["training"]["milestone"]["optimizer_step"] == 60
    assert scout["training"]["milestone"]["packed_cursor"] == 7680
    assert scout["training"]["milestone"]["evaluation_rows"] == 1700
    assert scout["training"]["lora"] == {
        **full_run.render_full_run("pii350")["training"]["lora"],
        "rank": 128,
        "alpha": 256,
    }
    assert scout["training"]["mechanics"]["scheduler"] is None
    assert scout["optimizer"]["lr"] == lr
    assert scout["optimizer"]["schedule"] == "constant"
    assert scout["training"]["mechanics"]["adamw"]["lr"] == lr
    assert scout["evaluation"]["timing"] == "milestone_and_final"
    assert scout["evaluation"]["expected_rows"] == 1700
    assert scout["backend"]["backend"] == "unsloth"
    assert scout["runtime"]["packages"][-2:] == [
        "unsloth==2026.7.4",
        "unsloth_zoo==2026.7.4",
    ]
    assert scout["packed_dataset"]["revision"] == ("11fd43ec9ebb187e1d0f94fe77bcf1090a2a18ee")
    assert scout["seed"] == 3407
    assert scout["budget"]["hard_timeout_seconds"] == 7474
    assert scout["manual_launch"]["modal_profile"] == expected_profile
    assert scout["manual_launch"]["requires_primary_action"] == expected_action
    with pytest.raises(RuntimeError, match="primary action"):
        full_run.require_pii350_scout_execute(
            scout,
            arm=arm,
            source_run_evidence=full_run.PII350_B192_COMPLETE_EVIDENCE,
            primary_action="wrong",
        )


@pytest.mark.parametrize(
    ("arm", "profile"),
    [
        ("lr1e-4", "hocdatacamp-hoangthu"),
        ("lr2e-4", "hocdatacamp-hoangthu"),
        ("lr3e-4", "hocdatacamp-hoangthu"),
        ("lr4e-4", "meddiesresearch"),
    ],
)
def test_pii350_scout_commands_bind_each_arm_to_its_own_cpu_receipt_and_h100_launch(arm: str, profile: str) -> None:
    scout = full_run.render_pii350_scout_contract(arm)
    preflight = full_run.render_pii350_scout_preflight_command(arm)
    launch = full_run.render_pii350_scout_command(arm)

    assert preflight == (
        f"MODAL_PROFILE={profile} "
        "uv run modal run --detach --timestamps -m "
        "meddies_pii.training.bioes.modal.full_run "
        f"--candidate pii350 --pii350-scout-run {arm} --preflight-assets"
    )
    assert "--execute" not in preflight
    assert launch.startswith(f"MODAL_PROFILE={profile} ")
    assert f"--candidate pii350 --pii350-scout-run {arm} --execute" in launch
    assert scout["manual_launch"]["requires_primary_action"] in launch
    assert "--resume" not in launch

    # reason: argv is a list, so no shell parses it, and every element is fixed — `sys.executable`,
    # reason: this repo's own module path, a literal flag, and `arm` from the parametrize list above.
    # reason: The launch-command call below is the same shape.
    rendered_preflight = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.trainers.full_run",
            "--render-pii350-scout-preflight-command",
            arm,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered_launch = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            sys.executable,
            "-m",
            "meddies_pii.training.bioes.trainers.full_run",
            "--render-pii350-scout-command",
            arm,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert rendered_preflight.stdout.strip() == preflight
    assert rendered_launch.stdout.strip() == launch


def test_pii350_lr3e4_scout_changes_only_learning_rate_and_identity_fields() -> None:
    baseline = full_run.render_pii350_scout_contract("lr2e-4")
    candidate = full_run.render_pii350_scout_contract("lr3e-4")

    for contract in (baseline, candidate):
        contract.pop("config_digest")
        contract["manual_launch"].pop("requires_primary_action")
        contract["optimizer"].pop("lr")
        contract["training"]["mechanics"]["adamw"].pop("lr")

    assert baseline == candidate


def test_pii350_lr4e4_scout_is_the_exact_constant_60_step_terminal_contract() -> None:
    scout = full_run.render_pii350_scout_contract("lr4e-4")

    assert scout["config_digest"] == ("2e5bdd06d0e2268deb5192d7a68a3d0cd370bd43c4a21fc2139b3d5a041cd2c8")
    assert scout["optimizer"]["lr"] == 4e-4
    assert scout["optimizer"]["schedule"] == "constant"
    assert scout["training"]["mechanics"]["scheduler"] is None
    assert scout["training"]["optimizer_step_cap"] == 60
    assert scout["training"]["milestone"] is None
    assert scout["evaluation"]["timing"] == "final_only"
    assert scout["evaluation"]["expected_rows"] == 1700
    assert scout["manual_launch"]["modal_profile"] == "meddiesresearch"
    assert scout["manual_launch"]["requires_primary_action"] == ("HA_AUTHORIZE_PII350_R128A256_B128_LR4E4_60STEP_SCOUT")

    preflight = full_run.render_pii350_scout_preflight_command("lr4e-4")
    launch = full_run.render_pii350_scout_command("lr4e-4")
    assert preflight.startswith("MODAL_PROFILE=meddiesresearch ")
    assert "--pii350-scout-run lr4e-4 --preflight-assets" in preflight
    assert launch.startswith("MODAL_PROFILE=meddiesresearch ")
    assert "--pii350-scout-run lr4e-4 --execute" in launch
    assert "HA_AUTHORIZE_PII350_R128A256_B128_LR4E4_60STEP_SCOUT" in launch


def test_pii350_wsd3e4_scout_is_an_exact_fresh_60_step_schedule_contract() -> None:
    scout = full_run.render_pii350_scout_contract("wsd3e-4")

    assert scout["config_digest"] == ("15f4fd2d7cf5312df759f7e8befa46617444a21690f9db1dd6a2ad8ba8ba6264")
    assert scout["training"]["batch_size"] == 128
    assert scout["training"]["lora"]["rank"] == 128
    assert scout["training"]["lora"]["alpha"] == 256
    assert scout["training"]["optimizer_step_cap"] == 60
    assert scout["training"]["milestone"] is None
    assert scout["evaluation"]["timing"] == "final_only"
    assert scout["evaluation"]["expected_rows"] == 1700
    assert scout["manual_launch"] == {
        "purpose": "anchor_pii350_r128a256_batch128_wsd_learning_rate",
        "automatic_launch": False,
        "required_source_run_evidence": full_run.PII350_B192_COMPLETE_EVIDENCE,
        "requires_primary_action": "HA_AUTHORIZE_PII350_R128A256_B128_WSD3E4_SCOUT",
        "modal_profile": "meddiesresearch",
        "oom_policy": {"automatic_fallback": False},
    }
    expected_schedule = {
        "peak_lr": 3e-4,
        "total_steps": 60,
        "warmup_steps": 6,
        "stable_steps": 48,
        "decay_steps": 6,
        "warmup_type": "linear",
        "decay_type": "cosine",
        "min_lr_ratio": 0.0,
    }
    assert scout["optimizer"] == {
        **full_run.render_pii350_scout_contract("lr3e-4")["optimizer"],
        "schedule": "wsd",
        **expected_schedule,
    }
    assert scout["training"]["mechanics"]["scheduler"] == {
        "name": "wsd",
        **expected_schedule,
    }

    preflight = full_run.render_pii350_scout_preflight_command("wsd3e-4")
    launch = full_run.render_pii350_scout_command("wsd3e-4")
    assert preflight.startswith("MODAL_PROFILE=meddiesresearch ")
    assert "--pii350-scout-run wsd3e-4 --preflight-assets" in preflight
    assert launch.startswith("MODAL_PROFILE=meddiesresearch ")
    assert "--pii350-scout-run wsd3e-4 --execute" in launch
    assert "HA_AUTHORIZE_PII350_R128A256_B128_WSD3E4_SCOUT" in launch

    malformed = full_run.render_pii350_scout_contract("wsd3e-4")
    malformed["optimizer"]["decay_steps"] = 5
    with pytest.raises(RuntimeError, match="exact rendered contract"):
        full_run.require_pii350_scout_execute(
            malformed,
            arm="wsd3e-4",
            source_run_evidence=full_run.PII350_B192_COMPLETE_EVIDENCE,
            primary_action=scout["manual_launch"]["requires_primary_action"],
        )


def test_pii350_scout_rejects_an_arm_mutation_or_opposite_action() -> None:
    scout = full_run.render_pii350_scout_contract("lr1e-4")
    mutated = full_run.render_pii350_scout_contract("lr1e-4")
    mutated["training"]["mechanics"]["adamw"]["lr"] = 2e-4

    with pytest.raises(RuntimeError, match="exact rendered contract"):
        full_run.require_pii350_scout_execute(
            mutated,
            arm="lr1e-4",
            source_run_evidence=full_run.PII350_B192_COMPLETE_EVIDENCE,
            primary_action=scout["manual_launch"]["requires_primary_action"],
        )
    with pytest.raises(RuntimeError, match="exact rendered contract"):
        full_run.require_pii350_scout_execute(
            scout,
            arm="lr2e-4",
            source_run_evidence=full_run.PII350_B192_COMPLETE_EVIDENCE,
            primary_action=scout["manual_launch"]["requires_primary_action"],
        )
