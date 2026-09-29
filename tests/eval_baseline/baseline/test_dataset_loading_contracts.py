from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from meddies_pii.eval_baseline.baseline.datasets import (
    EVAL_DATASETS,
    EXTERNAL_CONFIGS,
    canonical_language,
    load_all_external_rows,
    load_eval_cell,
    load_external_rows,
    load_external_stratified_subset,
    load_first_light_rows,
    load_first_light_subset,
    load_rows,
    load_v2_eval_rows,
    row_from_record,
    select_eval_datasets,
)


def test_selection_preserves_order_and_rejects_duplicates_and_unknowns() -> None:
    assert select_eval_datasets(" nemotron_en, v2-eval ") == (
        "nemotron_en",
        "v2-eval",
    )
    assert select_eval_datasets(None) == EVAL_DATASETS
    assert select_eval_datasets(" , ") == ()
    with pytest.raises(ValueError, match="duplicate datasets: \\['v2-eval'\\]"):
        select_eval_datasets("v2-eval,v2-eval")
    with pytest.raises(ValueError, match="unknown datasets: \\['missing'\\]"):
        select_eval_datasets("v2-eval,missing")


def test_row_from_record_uses_metadata_and_drops_bad_spans() -> None:
    row = row_from_record(
        {
            "raw": "Call Ana on 555.",
            "info": {
                "uid": "stable-1",
                "language": "English",
                "source": "fixture",
                "edge_cases": ["unicode", 7],
            },
            "label": [
                {"category": "human_name", "start": 5, "end": 8, "text": "Ana"},
                {"category": "unsupported", "start": 0, "end": 4},
                {"category": "date", "start": 9, "end": 9},
                {"category": "phone_number", "start": 12, "end": 15, "text": "nope"},
                "not-a-span",
            ],
        },
        dataset="external",
        index=9,
    )
    assert row.doc_id == "stable-1"
    assert row.shard == "en"
    assert row.language == "en"
    assert [(span.start, span.end, span.label) for span in row.gold_spans] == [(5, 8, "human_name")]
    assert row.slices == frozenset({"source=fixture", "edge_case=unicode", "edge_case=7"})
    assert row.stable_id == "external:en:stable-1"
    assert len(row.text_sha256) == 64


@pytest.mark.parametrize("record", [{}, {"text": ""}, {"text": 7}])
def test_row_from_record_rejects_missing_nonempty_text(
    record: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="missing non-empty text/raw"):
        row_from_record(record, dataset="v2-eval", index=0)


def test_load_rows_passes_limit_skips_bad_rows_and_preserves_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> list[object]:
        calls.append((args, kwargs))
        return [
            {
                "text": "Ana",
                "spans": [{"label": "human_name", "start": 0, "end": 3}],
                "language": "vi",
            },
            "not-a-record",
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    rows = load_rows(
        "Meddies/fake",
        "fixture",
        split="eval",
        dataset="external",
        shard="fixture-shard",
        limit=2,
    )
    assert calls == [(("Meddies/fake", "fixture"), {"split": "eval[:2]"})]
    assert len(rows) == 1
    assert rows[0].dataset == "external"
    assert rows[0].shard == "fixture-shard"
    assert rows[0].gold_spans[0].text == "Ana"


def test_loader_dispatch_rejects_bad_config_and_routes_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> list[object]:
        calls.append((args, kwargs))
        return []

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    assert load_v2_eval_rows("eval", limit=3, revision="v2-pin") == []
    assert load_external_rows(EXTERNAL_CONFIGS[0], limit=4, revision="external-pin") == []
    assert (
        load_eval_cell(
            "v2-eval-challenge",
            v2_limit=5,
            external_limit=6,
            v2_revision="v2-cell",
            external_revision="external-cell",
        )
        == []
    )
    assert (
        load_eval_cell(
            EXTERNAL_CONFIGS[0],
            v2_limit=5,
            external_limit=6,
            v2_revision="v2-cell",
            external_revision="external-cell",
        )
        == []
    )
    assert calls == [
        (
            ("Meddies/meddies-pii-v2", "eval"),
            {"split": "train[:3]", "revision": "v2-pin"},
        ),
        (
            ("Meddies/meddies-pii-external", EXTERNAL_CONFIGS[0]),
            {"split": "eval[:4]", "revision": "external-pin"},
        ),
        (
            ("Meddies/meddies-pii-v2", "eval-challenge"),
            {"split": "train[:5]", "revision": "v2-cell"},
        ),
        (
            ("Meddies/meddies-pii-external", EXTERNAL_CONFIGS[0]),
            {"split": "eval[:6]", "revision": "external-cell"},
        ),
    ]
    with pytest.raises(ValueError, match="unsupported v2 eval config"):
        load_v2_eval_rows("train")
    with pytest.raises(ValueError, match="unsupported external config"):
        load_external_rows("unknown")
    with pytest.raises(ValueError, match="unknown dataset"):
        load_eval_cell("unknown", v2_limit=None, external_limit=None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("English", "en"), ("fil", "fil"), ("unknown-language", "unknown")],
)
def test_canonical_language_normalizes_supported_aliases(value: str, expected: str) -> None:
    assert canonical_language(value) == expected


def test_subset_loaders_use_all_external_configs_and_light_source_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(repo_id: str, config: str, **_kwargs: object) -> list[dict[str, object]]:
        calls.append((repo_id, config))
        return [
            {
                "text": "Ana",
                "label": [{"category": "human_name", "start": 0, "end": 3}],
                "info": {"id": config, "language": "en"},
            },
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    all_external = load_all_external_rows(limit_per_config=1)
    assert len(all_external) == len(EXTERNAL_CONFIGS)
    stratified = load_external_stratified_subset(target_spans=1, max_rows_per_config=1, limit_per_config=1)
    assert len(stratified.rows) == 1
    light_rows = load_first_light_rows(limit=2, per_source_limit=1)
    light_subset = load_first_light_subset(limit=100, per_source_limit=1)
    assert len(light_rows) == 2
    assert len(light_subset.rows) == 6
    assert calls.count(("Meddies/meddies-pii-v2", "eval")) == 2
    assert calls.count(("Meddies/meddies-pii-v2", "eval-challenge")) == 2
