from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

from meddies_pii.training.bioes.modal import train as modal_train
from meddies_pii.training.bioes.trainers import trainer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from meddies_pii.training.bioes.trainers.config import SmokeTrainingConfig


def test_run_smoke_local_forwards_smoke_config_to_training_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run_smoke_training(config: SmokeTrainingConfig) -> dict[str, str]:
        captured["config"] = config
        return {"status": "fake"}

    monkeypatch.setattr(trainer, "run_smoke_training", fake_run_smoke_training)

    result = modal_train.run_smoke.local(
        steps=3,
        train_limit=7,
        eval_limit=5,
        model_id="test-model",
    )

    assert result == {"status": "fake"}
    assert captured["config"].steps == 3
    assert captured["config"].train_limit == 7
    assert captured["config"].eval_limit == 5
    assert captured["config"].model_id == "test-model"


def test_run_smoke_local_commits_and_marks_volume_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    commits: list[str] = []

    def fake_run_smoke_training(
        config: SmokeTrainingConfig,
        *,
        artifact_root: str,
        artifact_commit: Callable[[], None] | None,
    ) -> dict[str, str]:
        assert config.steps == 10
        assert artifact_root == "/artifacts/runs/smoke"
        assert artifact_commit is not None
        artifact_commit()
        return {"status": "fake"}

    monkeypatch.setattr(trainer, "run_smoke_training", fake_run_smoke_training)
    monkeypatch.setattr(
        modal_train,
        "artifact_volume",
        SimpleNamespace(commit=lambda: commits.append("commit")),
    )

    result = modal_train.run_smoke.local(artifact_root="/artifacts/runs/smoke")

    assert result == {
        "status": "fake",
        "artifact_root": "/artifacts/runs/smoke",
        "artifact_volume": "meddies-pii-bioes-artifacts",
        "artifact_persisted": True,
    }
    assert commits == ["commit", "commit"]


def test_run_smoke_local_propagates_training_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_training(config: SmokeTrainingConfig) -> NoReturn:
        msg = f"bad config: {config.backend}"
        raise RuntimeError(msg)

    monkeypatch.setattr(trainer, "run_smoke_training", fail_training)

    with pytest.raises(RuntimeError, match="bad config: unsloth"):
        modal_train.run_smoke.local()


class _FakeModalDispatch:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def remote(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.result


def test_modal_train_entrypoint_selects_a10_and_writes_local_result_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    a10 = _FakeModalDispatch({"status": "fake"})
    h100 = _FakeModalDispatch({"status": "wrong-dispatch"})
    out = tmp_path / "result.json"
    monkeypatch.setattr(modal_train, "run_smoke", a10)
    monkeypatch.setattr(modal_train, "run_smoke_h100", h100)

    modal_train.main(
        gpu="a10g",
        steps=3,
        train_limit=9,
        artifact_root="/artifacts/runs/a10",
        out=str(out),
        raw_log="raw.log",
        modal_url="https://modal.example/app",
    )

    rendered = json.loads(out.read_text())
    assert a10.calls == [
        {
            "backend": "unsloth",
            "steps": 3,
            "train_limit": 9,
            "artifact_root": "/artifacts/runs/a10",
        },
    ]
    assert h100.calls == []
    assert rendered["result"] == {"status": "fake"}
    assert rendered["provenance"]["command_kwargs"]["gpu"] == "A10G"
    assert rendered["provenance"]["raw_log"] == "raw.log"
    assert rendered["provenance"]["modal_url"] == "https://modal.example/app"
    assert '"status": "fake"' in capsys.readouterr().out


def test_modal_train_entrypoint_selects_h100_and_rejects_unknown_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a10 = _FakeModalDispatch({"status": "wrong-dispatch"})
    h100 = _FakeModalDispatch({"status": "fake"})
    monkeypatch.setattr(modal_train, "run_smoke", a10)
    monkeypatch.setattr(modal_train, "run_smoke_h100", h100)

    modal_train.main(gpu="h100-80gb", epochs=2, target_eval_slice="at_dot_obfuscation")

    assert h100.calls == [
        {
            "backend": "unsloth",
            "epochs": 2,
            "target_eval_slice": "at_dot_obfuscation",
            "target_eval_limit": 4,
            "target_eval_scan_multiplier": 32,
        },
    ]
    with pytest.raises(ValueError, match="Unsupported GPU 'L4'"):
        modal_train.main(gpu="L4")
    assert a10.calls == []


def test_modal_train_entrypoint_forwards_non_default_options_to_fake_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched = _FakeModalDispatch({"status": "fake"})
    monkeypatch.setattr(modal_train, "run_smoke", dispatched)
    monkeypatch.setattr(modal_train, "run_smoke_h100", _FakeModalDispatch({}))

    modal_train.main(
        backend="hf",
        steps=2,
        logging_steps=2,
        checkpoint_every_steps=1,
        resume_from_checkpoint="checkpoint",
        max_length=64,
        train_limit=8,
        eval_limit=4,
        batch_size=4,
        gradient_accumulation_steps=2,
        lora_rank=8,
        lora_alpha=16,
        packing=True,
        length_bucketing=True,
        viterbi_calibration_path="calibration.json",
        dataset_id="fake/train",
        dataset_split="validation",
        eval_dataset_split="test",
        eval_dataset_id="fake/eval",
        eval_dataset_revision="eval-revision",
        dataset_revision="train-revision",
        model_id="fake/model",
        model_revision="model-revision",
        tokenizer_id="fake/tokenizer",
        tokenizer_revision="tokenizer-revision",
        train_config="configured-train",
        eval_config="configured-eval",
        trust_remote_code=True,
        allow_same_local_eval=True,
    )

    kwargs = dispatched.calls[0]
    assert kwargs["backend"] == "hf"
    assert kwargs["packing"] is True
    assert kwargs["length_bucketing"] is True
    assert kwargs["allow_same_local_eval"] is True
    assert kwargs["tokenizer_revision"] == "tokenizer-revision"
