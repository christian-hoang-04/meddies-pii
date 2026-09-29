from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
# ruff: file-ignore[import-private-name]
# reason: checkpoint persistence must use `_save_backbone_adapter` from the data-artifact implementation so an
# reason: adapter checkpoint and its metadata are written atomically by one implementation. That serializer has no
# reason: public seam because it is only valid inside BIOES training; exposing it would invite callers to bypass
# reason: checkpoint ownership.
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from meddies_pii.training.bioes.data.artifacts import _save_backbone_adapter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .config import SmokeTrainingConfig


@dataclass(frozen=True, slots=True)
class TrainingLoopPlan:
    resolved_steps: int
    steps_per_epoch: int
    epochs: int | None


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    path: str
    completed_steps: int
    train_loss_values: list[float]


def _resolve_training_loop_plan(config: SmokeTrainingConfig, *, training_unit_count: int) -> TrainingLoopPlan:
    if training_unit_count <= 0:
        msg = "training_unit_count must be positive"
        raise ValueError(msg)
    units_per_optimizer_step = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(training_unit_count / units_per_optimizer_step)
    if config.epochs is not None and config.steps is None:
        return TrainingLoopPlan(
            resolved_steps=config.epochs * steps_per_epoch,
            steps_per_epoch=steps_per_epoch,
            epochs=config.epochs,
        )
    if config.steps is None:
        msg = "either steps or epochs must be set"
        raise ValueError(msg)
    return TrainingLoopPlan(
        resolved_steps=config.steps,
        steps_per_epoch=steps_per_epoch,
        epochs=config.epochs,
    )


def _checkpoint_path(artifact_root: str, backend: str, completed_steps: int) -> Path:
    return Path(artifact_root) / backend / "checkpoints" / f"step-{completed_steps:05d}"


def _trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in module.named_parameters() if parameter.requires_grad}


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


# reason: save training keeps model/commit at its adapter seam; bundling would hide required inputs.
def _save_training_checkpoint(  # ruff: ignore[too-many-arguments]
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    config: SmokeTrainingConfig,
    completed_steps: int,
    train_loss_values: Sequence[float],
    commit_callback: Callable[[], None] | None = None,
) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state": _trainable_state_dict(model),
            "completed_steps": completed_steps,
        },
        checkpoint_dir / "trainable_state.pt",
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    adapter_path = _save_backbone_adapter(model.backbone, checkpoint_dir) if hasattr(model, "backbone") else None
    trainer_state = {
        "completed_steps": completed_steps,
        "train_loss_values": [float(value) for value in train_loss_values],
        "config": asdict(config),
        "adapter_path": adapter_path,
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if commit_callback is not None:
        commit_callback()
    return str(checkpoint_dir)


def _load_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    device: str,
) -> LoadedTrainingCheckpoint:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainable_state_path = checkpoint_dir / "trainable_state.pt"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if not trainer_state_path.exists():
        msg = f"Missing trainer state: {trainer_state_path}"
        raise FileNotFoundError(msg)
    if not trainable_state_path.exists():
        msg = f"Missing trainable state: {trainable_state_path}"
        raise FileNotFoundError(msg)
    if not optimizer_path.exists():
        msg = f"Missing optimizer state: {optimizer_path}"
        raise FileNotFoundError(msg)

    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainable_payload = torch.load(trainable_state_path, map_location=device)
    trainable_state = trainable_payload.get("trainable_state")
    if not isinstance(trainable_state, dict):
        msg = f"Invalid trainable checkpoint payload: {trainable_state_path}"
        raise ValueError(msg)
    incompatible = model.load_state_dict(trainable_state, strict=False)
    if incompatible.unexpected_keys:
        msg = f"Checkpoint contains unexpected model keys: {sorted(incompatible.unexpected_keys)}"
        raise RuntimeError(msg)
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    _move_optimizer_state_to_device(optimizer, device)
    return LoadedTrainingCheckpoint(
        path=str(checkpoint_dir),
        completed_steps=int(trainer_state["completed_steps"]),
        train_loss_values=[float(value) for value in trainer_state.get("train_loss_values", [])],
    )
