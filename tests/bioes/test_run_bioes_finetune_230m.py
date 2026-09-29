from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import tomllib

if TYPE_CHECKING:
    from types import ModuleType

    import pytest


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "run_bioes_finetune_230m.py"
    spec = importlib.util.spec_from_file_location("run_bioes_finetune_230m_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_recipe_binds_every_settled_issue_78_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    captured: dict[str, object] = {}

    def capture_config(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(runner, "SmokeTrainingConfig", capture_config)

    assert runner.ISSUE_78_URL == "https://github.com/meddies-ai/meddies-pii/issues/78"
    runner.build_config(runner.parse_args([]))

    assert (
        captured.items()
        >= {
            "backend": "unsloth",
            "model_id": "LiquidAI/LFM2.5-230M-Base",
            "dataset_id": "Meddies/meddies-pii-mixed",
            "train_config": "default",
            "dataset_split": "train",
            "eval_dataset_id": "Meddies/meddies-pii-v2",
            "eval_config": "eval",
            "eval_dataset_split": "train",
            "max_length": 8192,
            "lora_rank": 128,
            "lora_alpha": 256,
            "packing": True,
            "batch_size": 128,
            "gradient_accumulation_steps": 1,
            "steps": 150,
            "checkpoint_every_steps": 10,
            "learning_rate": 1e-4,
            "fused_adamw": False,
        }.items()
    )
    assert {"custom_lora_kernel", "custom_mlp_kernel"}.isdisjoint(captured)


def test_hf_backend_disables_packing_without_changing_other_requested_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()

    config = runner.build_config(runner.parse_args(["--backend", "hf"]))

    assert config.packing is False
    assert (config.max_length, config.lora_rank, config.fused_adamw) == (8192, 128, False)
    assert "packing requires the unsloth backend" in capsys.readouterr().out


def test_deptry_does_not_trust_the_retired_experiments_namespace() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())

    assert "experiments" not in project.get("tool", {}).get("deptry", {}).get("known_first_party", ())
