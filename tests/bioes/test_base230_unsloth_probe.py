from __future__ import annotations

import json

# reason: these purity probes run this interpreter with fixed module/script argv and no shell.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from typing import TYPE_CHECKING, ClassVar

import pytest

from meddies_pii.training.bioes.trainers import base230_unsloth_probe as probe

if TYPE_CHECKING:
    from collections.abc import Callable


class _FakeFastLanguageModel:
    loaded: ClassVar[dict[str, object]] = {}
    peft: ClassVar[dict[str, object]] = {}

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> tuple[str, str]:
        cls.loaded = kwargs
        return "base", "tokenizer"

    @classmethod
    def get_peft_model(cls, model: object, **kwargs: object) -> str:
        cls.peft = kwargs
        return f"adapted-{model}"


def test_exact_two_fair_unsloth_cells_render_with_same_runtime_contract() -> None:
    rendered = {batch: probe.render_probe(batch) for batch in probe.PROBE_BATCHES}
    assert set(rendered) == {224, 240}
    for batch, config in rendered.items():
        assert config["candidate_key"] == "base230"
        assert config["backend"] == "unsloth"
        assert config["batch_size"] == batch
        assert config["optimizer_steps"] == 10
        assert config["use_gradient_checkpointing"] == "unsloth"
        assert config["load_in_16bit"] is True
        assert config["torch_dtype"] == "bfloat16"
        assert config["packed_dataset"]["revision"] == "11fd43ec9ebb187e1d0f94fe77bcf1090a2a18ee"
        assert (
            config["packed_dataset"]["manifest_sha256"]
            == "7cadda8e81ef4b2a1111f37a8b508492158a983e90f54fecafdc24764f792969"
        )
        assert config["training"]["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
        ]
        assert config["training"]["fused_adamw"] is False
        assert config["cost"] == {
            "rate_usd_per_second": 0.001097,
            "target_actual_usd": 3.0,
            "billing_overhead_reserve": 1.20,
            "maximum_live_estimate_usd": 2.50,
            "hard_timeout_seconds": probe.PROBE_HARD_TIMEOUT_SECONDS,
            "shutdown_commit_reserve_seconds": 120,
            "child_deadline_seconds": probe.PROBE_CHILD_DEADLINE_SECONDS,
            "stop_before_next_step": True,
        }


def test_unsloth_loader_receives_exact_causal_bf16_and_lora_arguments() -> None:
    """In Unsloth 2026.7.4, the vision flag prevents the new-model route from narrowing an explicit target list.

    Base230 has no vision tower, but the explicit True preserves LFM2's convolutional in_proj target.

    """
    model, tokenizer = probe.build_unsloth_base230(
        _FakeFastLanguageModel,
        dtype="torch.bfloat16",
        tokenizer_name="/cache/hf/pinned-snapshot",
    )
    assert (model, tokenizer) == ("adapted-base", "tokenizer")
    assert _FakeFastLanguageModel.loaded == {
        "model_name": "LiquidAI/LFM2.5-230M-Base",
        "revision": "9d2be5519834990d30996f878b6771cccbd24f2c",
        "tokenizer_name": "/cache/hf/pinned-snapshot",
        "max_seq_length": 8192,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_in_16bit": True,
        "dtype": "torch.bfloat16",
        "full_finetuning": False,
        "fast_inference": False,
        "trust_remote_code": False,
    }
    assert _FakeFastLanguageModel.peft["r"] == 64
    assert _FakeFastLanguageModel.peft["lora_alpha"] == 128
    assert _FakeFastLanguageModel.peft["use_gradient_checkpointing"] == "unsloth"
    assert _FakeFastLanguageModel.peft["use_rslora"] is False
    assert _FakeFastLanguageModel.peft["use_dora"] is False
    assert _FakeFastLanguageModel.peft["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "in_proj",
        "w1",
        "w2",
        "w3",
    ]
    assert {
        key: _FakeFastLanguageModel.peft[key]
        for key in (
            "finetune_vision_layers",
            "finetune_language_layers",
            "finetune_attention_modules",
            "finetune_mlp_modules",
        )
    } == {
        "finetune_vision_layers": True,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: {**value, "backend": "transformers_peft"}, "backend"),
        (lambda value: {**value, "batch_size": 192}, "batch"),
        (lambda value: {**value, "optimizer_steps": 9}, "optimizer_steps"),
        (lambda value: {**value, "candidate_key": "encoder230"}, "candidate_key"),
    ],
)
def test_probe_contract_fails_closed_for_backend_cell_and_candidate(
    mutator: Callable[[dict[str, object]], dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        probe.validate_probe_contract(mutator(probe.render_probe(224)))


def test_multi_cell_and_wrong_confirmation_cannot_dispatch() -> None:
    with pytest.raises(ValueError, match="batch"):
        probe.render_probe(224240)
    with pytest.raises(RuntimeError, match="requires"):
        probe.require_probe_execute(probe.render_probe(224), execute=False, confirmation="")
    with pytest.raises(RuntimeError, match="requires"):
        probe.require_probe_execute(probe.render_probe(240), execute=True, confirmation="wrong")


def test_pure_review_commands_do_not_import_modal_or_dispatch() -> None:
    for batch in (224, 240):
        command = probe.render_command(batch)
        assert f"--batch-size {batch}" in command
        # reason: batch comes from the fixed tuple above; sys.executable, module, flags, and list-form argv are trusted.
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                sys.executable,
                "-m",
                "meddies_pii.training.bioes.trainers.base230_unsloth_probe",
                "--render-config",
                "--batch-size",
                str(batch),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        rendered = json.loads(completed.stdout)
        assert rendered["batch_size"] == batch
    modules = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import meddies_pii.training.bioes.trainers.base230_unsloth_probe; assert "
                "'modal' not in sys.modules; assert 'unsloth' not in sys.modules; print('pure')"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert modules.stdout.strip() == "pure"


def test_probe_hard_wall_cannot_exceed_the_dedicated_live_cost_cap() -> None:
    assert probe.PROBE_HARD_TIMEOUT_SECONDS == 2278
    assert probe.PROBE_HARD_TIMEOUT_SECONDS * probe.PROBE_RATE_USD_PER_SECOND <= probe.PROBE_MAX_LIVE_ESTIMATE_USD
    assert 0 < probe.PROBE_CHILD_DEADLINE_SECONDS < probe.PROBE_HARD_TIMEOUT_SECONDS


def test_pinned_snapshot_is_resolved_before_unsloth_and_forwarded_as_tokenizer_name() -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/cache/hf/models--LiquidAI--LFM2.5-230M-Base/snapshots/pinned"

    snapshot = probe.resolve_base230_snapshot(snapshot_download)
    probe.build_unsloth_base230(
        _FakeFastLanguageModel,
        dtype="torch.bfloat16",
        tokenizer_name=snapshot,
    )
    assert calls == [
        {
            "repo_id": "LiquidAI/LFM2.5-230M-Base",
            "revision": "9d2be5519834990d30996f878b6771cccbd24f2c",
            "cache_dir": "/cache/hf",
            "local_files_only": True,
        },
    ]
    assert _FakeFastLanguageModel.loaded["model_name"] == "LiquidAI/LFM2.5-230M-Base"
    assert _FakeFastLanguageModel.loaded["revision"] == "9d2be5519834990d30996f878b6771cccbd24f2c"
    assert _FakeFastLanguageModel.loaded["tokenizer_name"] == snapshot


def test_unsloth_loader_cannot_omit_pinned_local_tokenizer_snapshot() -> None:
    with pytest.raises(TypeError):
        # reason: omitting the tokenizer snapshot is the point of the test, so the call is deliberately incomplete.
        probe.build_unsloth_base230(_FakeFastLanguageModel, dtype="torch.bfloat16")  # ty: ignore[missing-argument]
