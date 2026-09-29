from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from meddies_pii.eval_baseline.baseline.datasets import (
    EXTERNAL_DATASET_REVISION,
    V2_DATASET_REVISION,
    load_external_rows,
    load_rows,
    load_v2_eval_rows,
)

if TYPE_CHECKING:
    import pytest


def _capture_empty_dataset_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> list[dict[str, object]]:
        calls.append((*args, kwargs))
        return []

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )
    return calls


def test_load_rows_passes_the_pinned_dataset_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> list[dict[str, object]]:
        calls.append((*args, kwargs))
        return [
            {
                "text": "John",
                "label": [
                    {
                        "category": "human_name",
                        "start": 0,
                        "end": 4,
                        "text": "John",
                    },
                ],
                "info": {"id": "one", "language": "en"},
            },
        ]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    rows = load_rows(
        "Meddies/example",
        "config",
        split="eval",
        dataset="external",
        revision="pinned-sha",
    )

    assert len(rows) == 1
    assert calls == [
        (
            "Meddies/example",
            "config",
            {
                "split": "eval",
                "revision": "pinned-sha",
            },
        ),
    ]


def test_load_external_rows_uses_the_benchmark_revision_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_empty_dataset_loads(monkeypatch)

    assert load_external_rows("ai4privacy_en") == []
    assert calls == [
        (
            "Meddies/meddies-pii-external",
            "ai4privacy_en",
            {
                "split": "eval",
                "revision": EXTERNAL_DATASET_REVISION,
            },
        ),
    ]


def test_load_v2_eval_rows_uses_the_benchmark_revision_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_empty_dataset_loads(monkeypatch)

    assert load_v2_eval_rows("eval-challenge") == []
    assert calls == [
        (
            "Meddies/meddies-pii-v2",
            "eval-challenge",
            {
                "split": "train",
                "revision": V2_DATASET_REVISION,
            },
        ),
    ]


def test_benchmark_dataset_loaders_allow_explicit_revision_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_empty_dataset_loads(monkeypatch)

    assert load_external_rows("ai4privacy_en", revision="external-override") == []
    assert load_v2_eval_rows("eval", revision="v2-override") == []
    assert calls == [
        (
            "Meddies/meddies-pii-external",
            "ai4privacy_en",
            {
                "split": "eval",
                "revision": "external-override",
            },
        ),
        (
            "Meddies/meddies-pii-v2",
            "eval",
            {
                "split": "train",
                "revision": "v2-override",
            },
        ),
    ]
