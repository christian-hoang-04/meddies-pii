from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from meddies_pii.training.bioes.trainers import trainer
from meddies_pii.training.bioes.trainers.config import SmokeTrainingConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from meddies_pii.training.bioes.data.preparation import PreparedRow


class _BoundaryModel(torch.nn.Module):
    """A real `torch.nn.Module` standing in for the tagger whose packing boundary is measured.

    `prepare_training_units` declares `torch.nn.Module` because the contamination probe calls the
    model. These tests patch that probe out, but the parameter's contract is still a Module, so the
    double subclasses it rather than standing in as a looser namespace.
    """

    def __init__(self, *, packed_segment_isolation: bool = False) -> None:
        super().__init__()
        self.packed_segment_isolation = packed_segment_isolation


def test_prepare_training_units_runs_packing_probe_with_faked_model_boundary(
    monkeypatch: pytest.MonkeyPatch,
    prepared_row_factory: Callable[[str], PreparedRow],
) -> None:
    from meddies_pii.training.bioes.trainers import packing_runtime

    rows = [prepared_row_factory("prefix"), prepared_row_factory("target")]
    packed = [SimpleNamespace(name="packed")]
    probe_inputs = []
    logs: list[str] = []
    monkeypatch.setattr(packing_runtime, "pack_prepared_rows", lambda *_args, **_kwargs: packed)
    monkeypatch.setattr(packing_runtime, "packing_utilization", lambda _packed: 0.75)
    monkeypatch.setattr(
        packing_runtime,
        "run_packed_attention_contamination_probe",
        lambda _model, **kwargs: (
            probe_inputs.append(kwargs) or SimpleNamespace(to_dict=lambda: {"passed": True, "max_abs_diff": 0.0})
        ),
    )

    prepared = packing_runtime.prepare_training_units(
        SmokeTrainingConfig(packing=True, max_length=16),
        model=_BoundaryModel(packed_segment_isolation=True),
        tokenizer=SimpleNamespace(pad_token_id=7),
        train_rows=rows,
        device="cpu",
        log=logs.append,
    )

    assert prepared.units == packed
    assert prepared.utilization == 0.75
    assert prepared.boundary_token_count == 0
    assert prepared.attention_probe == {"passed": True, "max_abs_diff": 0.0}
    assert probe_inputs[0]["target_row_uid"] == "target"
    assert any("packing prepared" in line for line in logs)


def test_prepare_training_units_rejects_empty_packing(
    monkeypatch: pytest.MonkeyPatch,
    prepared_row_factory: Callable[[str], PreparedRow],
) -> None:
    from meddies_pii.training.bioes.trainers import packing_runtime

    row = prepared_row_factory("only-row")
    monkeypatch.setattr(packing_runtime, "pack_prepared_rows", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="produced 0 packed training units"):
        packing_runtime.prepare_training_units(
            SmokeTrainingConfig(packing=True),
            model=_BoundaryModel(),
            tokenizer=SimpleNamespace(pad_token_id=0),
            train_rows=[row],
            device="cpu",
            log=lambda _message: None,
        )


def test_prepare_training_units_returns_rows_without_packing(
    prepared_row_factory: Callable[[str], PreparedRow],
) -> None:
    from meddies_pii.training.bioes.trainers import packing_runtime

    rows = [prepared_row_factory("plain")]
    prepared = packing_runtime.prepare_training_units(
        SmokeTrainingConfig(packing=False),
        model=_BoundaryModel(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        train_rows=rows,
        device="cpu",
        log=lambda _message: None,
    )

    assert prepared.units == rows
    assert prepared.packed_rows is None


def test_prepare_training_units_rejects_failed_attention_probe(
    monkeypatch: pytest.MonkeyPatch,
    prepared_row_factory: Callable[[str], PreparedRow],
) -> None:
    from meddies_pii.training.bioes.trainers import packing_runtime

    rows = [prepared_row_factory("prefix"), prepared_row_factory("target")]
    packed = [SimpleNamespace()]
    monkeypatch.setattr(packing_runtime, "pack_prepared_rows", lambda *_args, **_kwargs: packed)
    monkeypatch.setattr(packing_runtime, "packing_utilization", lambda _packed: 1.0)
    monkeypatch.setattr(
        packing_runtime,
        "run_packed_attention_contamination_probe",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: {"passed": False, "max_abs_diff": 0.25}),
    )

    with pytest.raises(RuntimeError, match=r"contamination probe failed: max_abs_diff=0\.25"):
        packing_runtime.prepare_training_units(
            SmokeTrainingConfig(packing=True),
            model=_BoundaryModel(packed_segment_isolation=True),
            tokenizer=SimpleNamespace(pad_token_id=0),
            train_rows=rows,
            device="cpu",
            log=lambda _message: None,
        )


def test_prepare_training_units_requires_two_rows_for_attention_probe(
    monkeypatch: pytest.MonkeyPatch,
    prepared_row_factory: Callable[[str], PreparedRow],
) -> None:
    from meddies_pii.training.bioes.trainers import packing_runtime

    row = prepared_row_factory("only-row")
    monkeypatch.setattr(
        packing_runtime,
        "pack_prepared_rows",
        lambda *_args, **_kwargs: [SimpleNamespace()],
    )
    monkeypatch.setattr(packing_runtime, "packing_utilization", lambda _packed: 1.0)

    with pytest.raises(RuntimeError, match="requires at least 2 prepared train rows"):
        packing_runtime.prepare_training_units(
            SmokeTrainingConfig(packing=True),
            model=_BoundaryModel(packed_segment_isolation=True),
            tokenizer=SimpleNamespace(pad_token_id=None, eos_token_id=9),
            train_rows=[row],
            device="cpu",
            log=lambda _message: None,
        )


def test_run_smoke_training_loads_faked_hugging_face_rows_at_public_boundary(
    monkeypatch: pytest.MonkeyPatch,
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    from meddies_pii.training.bioes.trainers import data_loading

    install_smoke_training_fakes()
    dataset_calls = []

    class _Dataset:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict[str, int]:
            return {"value": index}

    def fake_load_dataset(
        dataset_id: str,
        config_name: str,
        *,
        split: str,
        revision: str,
    ) -> _Dataset:
        dataset_calls.append((dataset_id, config_name, split, revision))
        return _Dataset()

    monkeypatch.setattr(data_loading, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(trainer, "_load_rows", data_loading._load_rows)

    result = trainer.run_smoke_training(
        SmokeTrainingConfig(
            dataset_id="fake/repo",
            dataset_revision="revision-1",
            train_limit=1,
            eval_limit=1,
            steps=1,
            batch_size=1,
        ),
    )

    assert result["train_examples"] == 2
    assert dataset_calls == [
        ("fake/repo", "train", "train", "revision-1"),
        ("fake/repo", "test", "train", "revision-1"),
    ]


def test_run_smoke_training_reads_local_rows_through_public_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    install_smoke_training_fakes: Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]],
) -> None:
    from meddies_pii.training.bioes.trainers import data_loading

    install_smoke_training_fakes()
    local_jsonl = tmp_path / "fixture.jsonl"
    local_jsonl.write_text('{"text": "first"}\n\n{"text": "second"}\n', encoding="utf-8")
    monkeypatch.setattr(trainer, "_load_rows", data_loading._load_rows)

    result = trainer.run_smoke_training(
        SmokeTrainingConfig(
            dataset_id=str(local_jsonl),
            allow_same_local_eval=True,
            train_limit=1,
            eval_limit=1,
            steps=1,
            batch_size=1,
        ),
    )

    assert result["train_examples"] == 2
