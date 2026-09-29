from __future__ import annotations

from types import SimpleNamespace

import pytest

from meddies_pii.generation import datasets_adapter


class _DatasetsBoundary:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_dataset(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.result


def _provide_module(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    monkeypatch.setattr(
        datasets_adapter.importlib,
        "import_module",
        lambda _name: module,
    )


def test_streaming_train_returns_the_iterable_train_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _DatasetsBoundary({"train": iter([{"id": "a"}, {"id": "b"}])})
    _provide_module(monkeypatch, module)

    rows = datasets_adapter.load_streaming_train("Meddies/source")

    assert list(rows) == [{"id": "a"}, {"id": "b"}]
    assert module.calls == [(("Meddies/source",), {"streaming": True})]


def test_streaming_train_rejects_an_invalid_dataset_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provide_module(monkeypatch, _DatasetsBoundary(object()))

    with pytest.raises(RuntimeError, match="invalid streaming dataset"):
        datasets_adapter.load_streaming_train("Meddies/source")


def test_streaming_train_rejects_a_noniterable_train_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provide_module(monkeypatch, _DatasetsBoundary({"train": 7}))

    with pytest.raises(RuntimeError, match="train split is not iterable"):
        datasets_adapter.load_streaming_train("Meddies/source")


def test_materialized_train_returns_sized_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = [{"id": "a"}]
    module = _DatasetsBoundary(materialized)
    _provide_module(monkeypatch, module)

    rows = datasets_adapter.load_train_rows("Meddies/source", "en")

    assert rows is materialized
    assert module.calls == [(("Meddies/source", "en"), {"split": "train"})]


def test_materialized_train_rejects_rows_without_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provide_module(monkeypatch, _DatasetsBoundary(iter([{"id": "a"}])))

    with pytest.raises(RuntimeError, match="invalid train split"):
        datasets_adapter.load_train_rows("Meddies/source", "en")


def test_dataset_loading_rejects_a_module_without_the_required_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provide_module(monkeypatch, SimpleNamespace())

    with pytest.raises(RuntimeError, match="does not expose load_dataset"):
        datasets_adapter.load_streaming_train("Meddies/source")
