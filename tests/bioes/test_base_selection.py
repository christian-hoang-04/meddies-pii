from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from meddies_pii.training.bioes.trainers import base_selection


class TinyBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2)
        self.k_proj = nn.Linear(2, 2)
        self.v_proj = nn.Linear(2, 2)
        self.out_proj = nn.Linear(2, 2)
        self.in_proj = nn.Linear(2, 2)
        self.w1 = nn.Linear(2, 2)
        self.w2 = nn.Linear(2, 2)
        self.w3 = nn.Linear(2, 2)


class MissingTargetBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2)


class FakePeft:
    class TaskType:
        FEATURE_EXTRACTION = "feature"

    @dataclass
    class LoraConfig:
        r: int
        lora_alpha: int
        lora_dropout: float
        bias: str
        task_type: str
        target_modules: list[str]
        use_rslora: bool
        use_dora: bool

    @staticmethod
    def get_peft_model(model: nn.Module, config: object) -> nn.Module:
        # reason: peft attaches its config onto the module exactly this way, but torch's
        # reason: `Module.__setattr__` stub admits only `Tensor | Module`, so no annotation on the
        # reason: double's parameter lets the assignment typecheck while it still stands in for `nn.Module`.
        model.peft_config = config  # ty: ignore[invalid-assignment]
        return model


def test_four_exact_candidates_and_two_h100_lanes_render_without_launch() -> None:
    rendered = base_selection.render_gate_config()
    assert set(rendered["models"]) == {"base230", "encoder230", "encoder350", "pii350"}
    assert rendered["models"]["base230"]["revision"] == "9d2be5519834990d30996f878b6771cccbd24f2c"
    assert rendered["models"]["pii350"]["revision"] == "b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2"
    assert rendered["probe_lanes"] == {
        "230": {
            "gpu": "H100!",
            "models": ["base230", "encoder230"],
            "max_estimated_cost_usd": 3.0,
            "modal_profile": "hahuyhoang411",
        },
        "350": {
            "gpu": "H100!",
            "models": ["encoder350", "pii350"],
            "max_estimated_cost_usd": 3.0,
            "modal_profile": "hahuyhoang411",
        },
    }
    assert all(run["max_estimated_cost_usd"] == 10.0 for run in rendered["full_runs"].values())
    assert rendered["framework"]["torch_compile"] is False


def test_cost_gate_fails_closed_before_remote_launch() -> None:
    with pytest.raises(RuntimeError, match="budget"):
        base_selection.require_estimated_cost_within_budget(3.01, budget_usd=3.0)
    base_selection.require_estimated_cost_within_budget(3.0, budget_usd=3.0)


def test_lora_requires_all_logical_targets_and_keeps_head_trainable() -> None:
    body = TinyBody()
    head = base_selection.build_meddies_head(seed=3407)
    adapted, evidence = base_selection.apply_gate_lora(body, head, peft_module=FakePeft)
    assert adapted is body
    assert evidence.logical_targets == base_selection.LORA_TARGET_MODULES
    assert evidence.resolved_tensor_names
    assert all(parameter.requires_grad for parameter in head.parameters())
    with pytest.raises(RuntimeError, match="missing"):
        base_selection.apply_gate_lora(MissingTargetBody(), head, peft_module=FakePeft)


def test_head_is_bf16_and_seeded_identically() -> None:
    first = base_selection.build_meddies_head(seed=3407)
    torch.manual_seed(999)
    second = base_selection.build_meddies_head(seed=3407)
    first_dropout, first_linear = first[0], first[1]
    second_linear = second[1]
    assert isinstance(first_dropout, nn.Dropout)
    assert isinstance(first_linear, nn.Linear)
    assert isinstance(second_linear, nn.Linear)
    assert first_linear.in_features == 1024
    assert first_linear.out_features == 37
    assert first_dropout.p == 0.1
    assert first_linear.weight.dtype == torch.bfloat16
    assert torch.equal(first_linear.weight, second_linear.weight)


def test_pii_wrapper_retains_only_loaded_body_and_rejects_missing_body() -> None:
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lfm2 = TinyBody()
            self.classifier = nn.Linear(2, 161)

    class FakeTokenClassifier:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[nn.Module, dict[str, list[str]]]:
            return Wrapper(), {"missing_keys": [], "mismatched_keys": []}

    loaded = base_selection.load_pii350_body(FakeTokenClassifier)
    assert isinstance(loaded.body, TinyBody)
    assert loaded.loading_info["missing_keys"] == ()

    class BadTokenClassifier(FakeTokenClassifier):
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[nn.Module, dict[str, list[str]]]:
            return Wrapper(), {
                "missing_keys": ["lfm2.q_proj.weight"],
                "mismatched_keys": [],
            }

    with pytest.raises(RuntimeError, match="body tensors"):
        base_selection.load_pii350_body(BadTokenClassifier)


def test_loading_info_accepts_transformers_sets_and_rejects_invalid_collections() -> None:
    normalized = base_selection._normalized_loading_info({
        "missing_keys": {"lfm2.layers.0.weight"},
        "mismatched_keys": {("lfm2.layers.1.weight", (1024, 1024), (512, 1024))},
        "unexpected_keys": {"lm_head.weight"},
    })
    assert normalized == {
        "missing_keys": ("lfm2.layers.0.weight",),
        "mismatched_keys": ("lfm2.layers.1.weight",),
        "unexpected_keys": ("lm_head.weight",),
    }
    with pytest.raises(RuntimeError, match="missing_keys is invalid"):
        base_selection._normalized_loading_info({"missing_keys": "lfm2.weight"})
    with pytest.raises(RuntimeError, match="mismatched_keys is invalid"):
        base_selection._normalized_loading_info({"mismatched_keys": {"not-a-tuple"}})


def test_encoder_masked_lm_wrapper_extracts_loaded_body_and_rejects_prefix_mismatch() -> None:
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lfm2 = TinyBody()
            self.lm_head = nn.Linear(2, 2, bias=False)

    class FakeMaskedLM:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[nn.Module, dict[str, set[object]]]:
            return Wrapper(), {"missing_keys": set(), "mismatched_keys": set()}

    loaded = base_selection.load_encoder_body(FakeMaskedLM, base_selection.CANDIDATES["encoder350"])
    assert isinstance(loaded.body, TinyBody)
    assert loaded.loading_info["missing_keys"] == ()

    class BadMaskedLM(FakeMaskedLM):
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> tuple[nn.Module, dict[str, set[object]]]:
            return Wrapper(), {
                "missing_keys": {"lfm2.layers.0.q_proj.weight"},
                "mismatched_keys": set(),
                "unexpected_keys": {"layers.0.q_proj.weight"},
            }

    with pytest.raises(RuntimeError, match="body tensors"):
        base_selection.load_encoder_body(BadMaskedLM, base_selection.CANDIDATES["encoder350"])


def test_model_loaders_bind_h100_cache_and_forbid_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "/cache/hf")
    captured: dict[str, object] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **kwargs: object) -> tuple[nn.Module, dict[str, list[str]]]:
            captured.update(kwargs)
            return TinyBody(), {"missing_keys": [], "mismatched_keys": []}

    base_selection.load_standard_body(FakeAutoModel, base_selection.CANDIDATES["base230"])
    assert captured["cache_dir"] == "/cache/hf"
    assert captured["local_files_only"] is True

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lfm2 = TinyBody()

    class FakeTokenClassifier:
        @staticmethod
        def from_pretrained(*_args: object, **kwargs: object) -> tuple[nn.Module, dict[str, list[str]]]:
            captured.update(kwargs)
            return Wrapper(), {"missing_keys": [], "mismatched_keys": []}

    base_selection.load_pii350_body(FakeTokenClassifier)
    assert captured["cache_dir"] == "/cache/hf"
    assert captured["local_files_only"] is True

    class EncoderWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lfm2 = TinyBody()

    class FakeMaskedLM:
        @staticmethod
        def from_pretrained(*_args: object, **kwargs: object) -> tuple[nn.Module, dict[str, set[str]]]:
            captured.update(kwargs)
            return EncoderWrapper(), {"missing_keys": set(), "mismatched_keys": set()}

    base_selection.load_encoder_body(FakeMaskedLM, base_selection.CANDIDATES["encoder230"])
    assert captured["cache_dir"] == "/cache/hf"
    assert captured["local_files_only"] is True


def test_memory_contract_requires_native_non_reentrant_checkpointing() -> None:
    class CheckpointableBody(TinyBody):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.is_gradient_checkpointing = False
            self.checkpointing_kwargs: dict[str, bool] | None = None

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs: dict[str, bool]) -> None:
            self.checkpointing_kwargs = gradient_checkpointing_kwargs
            self.is_gradient_checkpointing = True

    body = CheckpointableBody()
    contract = base_selection.enable_native_gradient_checkpointing(body)
    assert body.config.use_cache is False
    assert body.checkpointing_kwargs == {"use_reentrant": False}
    assert contract.gradient_checkpointing is True
    assert contract.output_hidden_states is False
    assert contract.use_cache is False

    class NoCheckpointBody(TinyBody):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)

    with pytest.raises(RuntimeError, match="does not expose native gradient checkpointing"):
        base_selection.enable_native_gradient_checkpointing(NoCheckpointBody())


def test_probe_plan_uses_high_anchor_discovery_and_stable_measurement() -> None:
    plan = base_selection.probe_batch_plan()
    assert plan.discovery_batches == (224,)
    assert plan.oom_fallback_batches == (192, 160, 128, 96, 64)
    assert plan.discovery_steps == 2
    assert plan.steps == {"cold": 1, "warmup": 2, "measured": 7}


def test_rendered_telemetry_prioritizes_throughput_not_vram_fill() -> None:
    protocol = base_selection.render_gate_config()["probe_protocol"]
    assert protocol["metric"] == "median_real_bioes_tokens_per_second"
    assert protocol["vrm_occupancy"] == "stable_95_to_100_percent_allowed_not_selected"
    assert protocol["anchor_batch"] == 224
    assert protocol["oom_fallback"] == [192, 160, 128, 96, 64]
    assert protocol["discovery_steps"] == 2
    framework = base_selection.render_gate_config()["framework"]
    assert framework["attention_implementation"] == "sdpa"
    assert framework["fused_adamw"] is False
    assert framework["gradient_checkpointing"] == "native_non_reentrant_required"
    assert framework["output_hidden_states"] is False
    assert framework["use_cache"] is False
    assert framework["runtime_precision"] == "bf16_16bit"
    assert framework["packing_max_length"] == 8192
    assert framework["gradient_accumulation_steps"] == 1
    assert framework["june_baseline_deviations"] == ["transformers_peft_native_checkpointing_replaces_unsloth"]
    lora = base_selection.render_gate_config()["lora"]
    assert lora["dropout"] == 0.0
    assert lora["alpha_to_rank"] == 2.0
    optimizer = base_selection.render_gate_config()["optimizer"]
    assert optimizer == {
        "name": "adamw",
        "lr": 1e-4,
        "fused": False,
        "gradient_accumulation_steps": 1,
    }
    assert "gpu_sm_utilization" in protocol["telemetry"]
    assert "gpu_memory_controller_utilization" in protocol["telemetry"]


def test_fused_adamw_is_adopted_only_past_the_noise_margin() -> None:
    assert base_selection.select_fused_adamw(1_000.0, 1_030.0) is True
    assert base_selection.select_fused_adamw(1_000.0, 1_029.0) is False
    assert base_selection.select_fused_adamw(1_000.0, 900.0) is False


@pytest.mark.parametrize(("unfused", "fused"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_fused_adamw_refuses_non_positive_throughput(unfused: float, fused: float) -> None:
    with pytest.raises(ValueError, match="throughput measurements must be positive"):
        base_selection.select_fused_adamw(unfused, fused)


def test_finalization_reserve_never_drops_below_ten_minutes() -> None:
    # reason: both results are exact in binary floating point - the floor is returned verbatim and
    # reason: 1.25*600 + 1.5*200 + 120 is a sum of exactly representable values, so a tolerance would
    # reason: only hide a change to the formula this test exists to pin.
    floor = base_selection.finalization_reserve_seconds(final_eval_seconds=0.0, save_seconds=0.0)
    measured = base_selection.finalization_reserve_seconds(final_eval_seconds=600.0, save_seconds=200.0)

    assert floor == 600.0
    assert measured == 1_170.0


def test_finalization_reserve_refuses_negative_measurements() -> None:
    with pytest.raises(ValueError, match="finalization measurements must be non-negative"):
        base_selection.finalization_reserve_seconds(final_eval_seconds=-1.0, save_seconds=0.0)


def test_checkpoint_payload_copies_its_mappings_so_later_mutation_cannot_reach_it() -> None:
    digests = {"packed": "a" * 64}
    rng_state = {"cpu": "state"}
    payload = base_selection.checkpoint_payload(
        manifest_digests=digests,
        data_cursor=7_680,
        rng_state=rng_state,
    )

    digests["packed"] = "b" * 64
    rng_state["cpu"] = "mutated"

    assert payload == {
        "data_cursor": 7_680,
        "manifest_digests": {"packed": "a" * 64},
        "rng_state": {"cpu": "state"},
    }


def test_checkpoint_payload_refuses_a_cursor_or_manifest_it_cannot_resume_from() -> None:
    with pytest.raises(ValueError, match="checkpoint needs a cursor and manifest digests"):
        base_selection.checkpoint_payload(manifest_digests={}, data_cursor=0, rng_state={})
    with pytest.raises(ValueError, match="checkpoint needs a cursor and manifest digests"):
        base_selection.checkpoint_payload(manifest_digests={"packed": "a"}, data_cursor=-1, rng_state={})
