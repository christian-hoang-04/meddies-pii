from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import datasets

from meddies_pii.publishing import huggingface

if TYPE_CHECKING:
    import pytest


class CapturedPush(TypedDict, total=False):
    """What the fake Dataset recorded: the rows it was built from and the push arguments."""

    rows: object
    repo_id: str
    kwargs: dict[str, object]


def _capture_push(monkeypatch: pytest.MonkeyPatch) -> CapturedPush:
    captured: CapturedPush = {}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows: object) -> FakeDataset:
            captured["rows"] = rows
            return cls()

        @staticmethod
        def push_to_hub(repo_id: str, **kwargs: object) -> None:
            captured["repo_id"] = repo_id
            captured["kwargs"] = kwargs

    monkeypatch.setattr(datasets, "Dataset", FakeDataset)
    return captured


def test_push_config_pushes_rows_as_named_config_and_split(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_push(monkeypatch)
    rows = [
        {"text": "a", "label": [], "info": {}},
        {"text": "b", "label": [], "info": {}},
    ]

    count = huggingface.push_config("Meddies/meddies-pii", "eval", rows, split="train")

    assert count == 2
    assert captured["rows"] == rows
    assert captured["repo_id"] == "Meddies/meddies-pii"
    assert captured["kwargs"]["config_name"] == "eval"
    assert captured["kwargs"]["split"] == "train"


def test_push_config_omits_private_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default must not touch repo visibility — mirrors mix.py's push_to_hub call."""
    captured = _capture_push(monkeypatch)

    huggingface.push_config("Meddies/meddies-pii", "eval", [{"text": "a"}])

    assert "private" not in captured["kwargs"]
    assert captured["kwargs"]["split"] == "train"
