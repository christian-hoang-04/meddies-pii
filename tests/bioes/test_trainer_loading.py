"""Back-compat.

Unset eval_dataset_id => eval loads from the same repo/revision as train, exactly as before this field existed.

Cross-repo: train on mixed, eval on v2. The eval repo's own revision is used, independent of the train repo's revision.

eval repo differs but no eval revision pinned => None (not train's revision).

"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from meddies_pii.historical_artifacts import legacy_jsonl_locator
from meddies_pii.training.bioes.trainers.trainer import (
    SmokeTrainingConfig,
    _load_rows,
    _load_training_checkpoint,
    _resolve_training_loop_plan,
    _save_training_checkpoint,
    validate_smoke_training_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_rows_accepts_local_jsonl_dataset_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / legacy_jsonl_locator("train")
    records = [
        {"text": "Patient Alice", "label": []},
        {"text": "Patient Bob", "label": []},
        {"text": "Patient Cora", "label": []},
    ]
    dataset_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    rows = _load_rows(
        "unused",
        2,
        scan_multiplier=1,
        dataset_id=str(dataset_path),
    )

    assert [row["text"] for row in rows] == ["Patient Alice", "Patient Bob"]
    assert rows[0]["_dataset_index"] == 0
    assert rows[0]["uid"] == f"{dataset_path}:0"


def test_smoke_training_defaults_to_unsloth_lora_backend() -> None:
    config = SmokeTrainingConfig()

    assert config.backend == "unsloth"
    assert config.lora_rank == 4
    assert config.lora_alpha is None
    assert config.logging_steps == 1


def test_eval_dataset_id_defaults_to_train_repo() -> None:
    config = SmokeTrainingConfig(dataset_id="Meddies/meddies-pii", dataset_revision="abc123")

    assert config.resolved_eval_dataset_id() == "Meddies/meddies-pii"
    assert config.resolved_eval_revision() == "abc123"


def test_eval_dataset_id_can_differ_from_train_repo() -> None:
    config = SmokeTrainingConfig(
        dataset_id="Meddies/meddies-pii-mixed",
        dataset_revision="train-sha",
        eval_dataset_id="Meddies/meddies-pii-v2",
        eval_dataset_revision="eval-sha",
    )

    assert config.resolved_eval_dataset_id() == "Meddies/meddies-pii-v2"
    assert config.resolved_eval_revision() == "eval-sha"


def test_eval_dataset_id_set_without_revision_is_none() -> None:
    config = SmokeTrainingConfig(
        dataset_id="Meddies/meddies-pii-mixed",
        dataset_revision="train-sha",
        eval_dataset_id="Meddies/meddies-pii-v2",
    )

    assert config.resolved_eval_dataset_id() == "Meddies/meddies-pii-v2"
    assert config.resolved_eval_revision() is None


def test_bioes_packing_is_allowed_for_unsloth_backend() -> None:
    config = SmokeTrainingConfig(backend="unsloth", packing=True)

    validate_smoke_training_config(config)


def test_bioes_packing_requires_unsloth_backend() -> None:
    config = SmokeTrainingConfig(backend="hf", packing=True)

    with pytest.raises(ValueError, match="packing=True requires backend='unsloth'"):
        validate_smoke_training_config(config)


@pytest.mark.parametrize("kwargs", [{"lora_rank": 0}, {"lora_alpha": 0}])
# reason: this parameter is spread as SmokeTrainingConfig(**kwargs) into a dataclass whose fields are not all int,
# reason: so no annotation on it typechecks without changing what the parametrize case passes.
def test_lora_configuration_requires_positive_values(
    kwargs,  # ruff: ignore[missing-type-function-argument]
) -> None:
    with pytest.raises(ValueError, match=r"lora_(rank|alpha) must be positive"):
        validate_smoke_training_config(SmokeTrainingConfig(**kwargs))


def test_bioes_custom_length_bucketing_is_disabled_until_upstream_sampler() -> None:
    config = SmokeTrainingConfig(length_bucketing=True)

    with pytest.raises(NotImplementedError, match="upstream trainer/sampler"):
        validate_smoke_training_config(config)


def test_epochs_resolve_optimizer_steps_from_training_units() -> None:
    config = SmokeTrainingConfig(
        steps=None,
        epochs=3,
        batch_size=128,
        gradient_accumulation_steps=1,
    )

    plan = _resolve_training_loop_plan(config, training_unit_count=21_600)

    assert plan.steps_per_epoch == 169
    assert plan.resolved_steps == 507
    assert plan.epochs == 3


def test_epochs_must_be_positive_when_set() -> None:
    config = SmokeTrainingConfig(steps=None, epochs=0)

    with pytest.raises(ValueError, match="epochs must be positive"):
        validate_smoke_training_config(config)


def test_checkpoint_every_steps_must_be_non_negative() -> None:
    config = SmokeTrainingConfig(checkpoint_every_steps=-1)

    with pytest.raises(ValueError, match="checkpoint_every_steps must be >= 0"):
        validate_smoke_training_config(config)


def test_training_checkpoint_restores_trainable_state_and_optimizer(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = torch.tensor([[1.0, 2.0]])
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    expected_weight = model.weight.detach().clone()

    checkpoint_dir = tmp_path / "step-00010"
    saved_path = _save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        config=SmokeTrainingConfig(steps=10, checkpoint_every_steps=10),
        completed_steps=10,
        train_loss_values=[0.7, 0.5],
    )

    with torch.no_grad():
        model.weight.add_(10.0)

    loaded = _load_training_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
    )

    assert saved_path == str(checkpoint_dir)
    assert loaded.completed_steps == 10
    assert loaded.train_loss_values == [0.7, 0.5]
    assert torch.allclose(model.weight, expected_weight)
