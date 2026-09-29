from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn

import pytest

from meddies_pii.training.bioes.trainers import trainer
from meddies_pii.training.bioes.trainers.config import SmokeTrainingConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from meddies_pii.json_types import JsonValue
    from meddies_pii.training.bioes.trainers.config import SmokeTrainingPayload


def _mapping_field(payload: SmokeTrainingPayload, key: str) -> dict[str, JsonValue]:
    """Return a payload field the caller reads by key, asserting it really is a JSON object.

    `SmokeTrainingPayload` is the erased `asdict()` of a ~60-field dataclass, so every field reads
    as `JsonValue`. Asserting the shape here pins what the test relies on instead of duplicating
    the dataclass in a parallel TypedDict that would have to be hand-synced forever.

    Returns:
        The named field, narrowed to a JSON object the caller can read by key.

    """
    value = payload[key]
    assert isinstance(value, dict), f"{key} must be a JSON object, got {type(value).__name__}"
    return value


def _list_field(payload: SmokeTrainingPayload, key: str) -> list[JsonValue]:
    """Return a payload field the caller indexes or slices, asserting it really is a JSON list.

    Returns:
        The named field, narrowed to a JSON list the caller can index or slice.

    """
    value = payload[key]
    assert isinstance(value, list), f"{key} must be a JSON list, got {type(value).__name__}"
    return value


def test_run_smoke_training_returns_fake_lifecycle_output_and_checkpoints(
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    checkpoints, _ = install_smoke_training_fakes()

    result = trainer.run_smoke_training(
        SmokeTrainingConfig(steps=2, batch_size=1, checkpoint_every_steps=1),
        # reason: `install_smoke_training_fakes` patches every consumer of `artifact_root` —
        # reason: `_save_training_checkpoint`, `_save_classifier_state`, `_save_backbone_adapter` —
        # reason: so the root is only ever joined by the pure `_checkpoint_path` and never opened.
        artifact_root="/tmp/fake-artifacts",  # ruff: ignore[hardcoded-temp-file]
    )

    assert result["completed_steps"] == 2
    assert result["latest_checkpoint"] == "checkpoint-2"
    assert result["train_loss_values"] == pytest.approx([1.0, 0.64])
    assert result["eval_exact_span_f1_delta"] == 0.5
    assert checkpoints == [1, 2]
    assert result["model_artifact"] == "adapter"


def test_run_smoke_training_resumes_from_checkpoint_before_training(
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    checkpoints, _ = install_smoke_training_fakes(resumed=True)

    result = trainer.run_smoke_training(
        SmokeTrainingConfig(
            steps=1,
            batch_size=1,
            checkpoint_every_steps=1,
            # reason: `install_smoke_training_fakes(resumed=True)` patches `_load_training_checkpoint`,
            # reason: the only reader of this path, so the directory is never opened or stat-ed.
            resume_from_checkpoint="/tmp/checkpoint-2",  # ruff: ignore[hardcoded-temp-file]
        ),
    )

    assert result["resume_from_checkpoint"] == "checkpoint-2"
    assert result["completed_steps"] == 3
    assert _list_field(result, "train_loss_values")[:2] == [0.9, 0.8]
    assert checkpoints == [3]


def test_run_smoke_training_rejects_invalid_config_before_model_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_build(_config: SmokeTrainingConfig) -> NoReturn:
        msg = "model loading must not run for invalid config"
        raise AssertionError(msg)

    monkeypatch.setattr(trainer, "_build_artifacts", should_not_build)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        trainer.run_smoke_training(SmokeTrainingConfig(batch_size=0))


def test_run_smoke_training_evaluates_fake_logits_without_loading_a_model(
    monkeypatch: pytest.MonkeyPatch,
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    from meddies_pii.training.bioes.trainers import evaluation

    install_smoke_training_fakes()
    monkeypatch.setattr(trainer, "_evaluate", evaluation._evaluate)

    result = trainer.run_smoke_training(SmokeTrainingConfig(steps=1, batch_size=1))

    assert result["initial_eval_exact_span_f1"] == 0.0
    assert result["eval_exact_span_f1"] == 0.0


def test_run_smoke_training_reads_viterbi_artifact_and_local_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    from meddies_pii.annotations.bioes import zero_viterbi_transition_biases
    from meddies_pii.training.bioes.trainers import runtime

    install_smoke_training_fakes()
    calibration_path = tmp_path / "viterbi.json"
    expected_biases = zero_viterbi_transition_biases()
    calibration_path.write_text(json.dumps({"operating_points": {"default": {"biases": expected_biases}}}))
    monkeypatch.setattr(
        trainer,
        "_resolve_viterbi_transition_biases",
        runtime._resolve_viterbi_transition_biases,
    )
    monkeypatch.setattr(trainer, "_optimizer_for_config", runtime._optimizer_for_config)
    monkeypatch.setattr(trainer, "_package_versions", runtime._package_versions)
    monkeypatch.setattr(trainer, "_cuda_memory_report", runtime._cuda_memory_report)

    result = trainer.run_smoke_training(
        SmokeTrainingConfig(steps=1, batch_size=1, viterbi_calibration_path=str(calibration_path)),
    )

    assert result["viterbi_transition_biases"] == expected_biases
    assert _mapping_field(result, "cuda_memory")["available"] is False
    assert "torch" in _mapping_field(result, "package_versions")


def test_run_smoke_training_uses_targeted_eval_selection_at_public_boundary(
    monkeypatch: pytest.MonkeyPatch,
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    from meddies_pii.training.bioes.eval import selection
    from meddies_pii.training.bioes.trainers import data_loading

    _, metric = install_smoke_training_fakes()
    targeted = SimpleNamespace(
        prepared_rows=[],
        slice_name="at_dot_obfuscation",
        to_report=lambda: {"slice_name": "at_dot_obfuscation", "selected": 0},
    )
    monkeypatch.setattr(
        data_loading,
        "_load_rows",
        lambda *_args, **_kwargs: [{"uid": "target-source"}],
    )
    monkeypatch.setattr(selection, "at_dot_obfuscate_email_rows", lambda rows, **_kwargs: rows)
    monkeypatch.setattr(
        selection,
        "select_prepared_rows_for_adversarial_slice",
        lambda *_args, **_kwargs: targeted,
    )
    monkeypatch.setattr(
        trainer,
        "_load_targeted_eval_selection",
        data_loading._load_targeted_eval_selection,
    )
    monkeypatch.setattr(trainer, "_evaluate", lambda *_args, **_kwargs: metric(0.5))

    result = trainer.run_smoke_training(
        SmokeTrainingConfig(
            steps=1,
            batch_size=1,
            target_eval_slice="at_dot_obfuscation",
            target_eval_require_support=False,
        ),
    )

    targeted_result = _mapping_field(result, "targeted_eval")
    assert targeted_result["slice_name"] == "at_dot_obfuscation"
    assert targeted_result["selected"] == 0
    assert targeted_result["row_hashes"] == []
    assert targeted_result["exact_span_f1_delta"] == 0.0
    assert targeted_result["containment_span_f1_delta"] == 0.0
    assert targeted_result["final_adversarial_slice"] is None


def test_run_smoke_training_rejects_conflicting_viterbi_lifecycle_inputs(
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    install_smoke_training_fakes()

    with pytest.raises(ValueError, match="Specify either viterbi_transition_biases"):
        trainer.run_smoke_training(
            SmokeTrainingConfig(
                viterbi_transition_biases={},
                viterbi_calibration_path="calibration.json",
            ),
        )
