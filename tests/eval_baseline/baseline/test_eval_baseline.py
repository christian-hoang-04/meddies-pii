from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import json
from typing import TYPE_CHECKING, Any

import pytest

from meddies_pii.eval_baseline.adapters.openmed import (
    OPENMED_SUPPORTED_LABELS,
    map_openmed_label,
    spans_from_pipeline_entities,
)
from meddies_pii.eval_baseline.baseline.datasets import EvalRow
from meddies_pii.eval_baseline.baseline.run import (
    ShardSpec,
    filter_gold_to_supported,
    read_matrix_results,
    run_shard,
    score_reports,
)
from meddies_pii.eval_baseline.baseline.subset import (
    StratifiedSubsetSpec,
    build_external_stratified_subset,
    build_pinned_subset,
    build_smoke_subset,
)
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    EvaluationContract,
    dataset_shard_identity,
    file_sha256,
)
from meddies_pii.jsonl import read_jsonl, write_jsonl
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.eval_baseline.baseline.run import MatrixResults


def _evaluation_contract() -> EvaluationContract:
    def artifact(name: str, digest: str) -> ArtifactIdentity:
        return ArtifactIdentity(name, "f" * 40, digest * 64)

    return EvaluationContract(
        model=artifact("model", "a"),
        vendor_inference_source=artifact("vendor", "b"),
        local_adapter_source=artifact("adapter", "c"),
        applied_label_prediction_contract=artifact("labels", "d"),
        decoder_contract=artifact("decoder", "e"),
        resolved_runtime_environment=artifact("runtime", "f"),
        scorer_contract=artifact("scorer", "0"),
        supported_labels=tuple(sorted(OPENMED_SUPPORTED_LABELS)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


# reason: Dataset, shard, language, text, and slices are evaluation-row identity axes varied independently.
def _row(  # ruff: ignore[too-many-arguments]
    doc_id: str,
    *,
    dataset: str,
    shard: str,
    language: str,
    text: str = "Patient John Smith visited Bạch Mai.",
    slices: frozenset[str] = frozenset({"smoke"}),
) -> EvalRow:
    return EvalRow(
        doc_id=doc_id,
        dataset=dataset,
        shard=shard,
        text=text,
        gold_spans=(CharSpan(start=8, end=18, text="John Smith", label="human_name"),),
        language=language,
        slices=slices,
    )


def _read_completed_v2_matrix(
    tmp_path: Path,
    contract: EvaluationContract,
    rows: list[EvalRow],
) -> MatrixResults:
    return read_matrix_results(
        tmp_path,
        "openmed",
        ("v2-eval",),
        require_done=True,
        expected_rows={"v2-eval": len(rows)},
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities={
            "v2-eval": dataset_shard_identity(contract, dataset="v2-eval", shard="full", rows=rows),
        },
    )


def test_openmed_label_map_reuses_meddies_taxonomy_and_keeps_url_private_url() -> None:
    assert map_openmed_label("B-FIRST_NAME") == "human_name"
    assert map_openmed_label("I-date_time") == "date"
    assert map_openmed_label("url") == "private_url"
    assert map_openmed_label("CVV") == "secret"
    assert map_openmed_label("O") is None
    assert map_openmed_label("diagnosis") is None
    assert OPENMED_SUPPORTED_LABELS == PII_LABEL_SET


def test_openmed_pipeline_entities_become_original_text_charspans() -> None:
    text = "Patient John Smith used https://public.example for login token abc123."
    entities: list[dict[str, Any]] = [
        {"entity_group": "FIRST_NAME", "word": "John", "start": 8, "end": 12},
        {
            "entity_group": "URL",
            "word": "https://public.example",
            "start": 24,
            "end": 46,
        },
        {"entity_group": "PASSWORD", "word": "abc123", "start": 63, "end": 69},
        {"entity_group": "DIAGNOSIS", "word": "login", "start": 51, "end": 56},
    ]

    spans = spans_from_pipeline_entities(text, entities)

    assert spans == [
        CharSpan(start=8, end=12, text="John", label="human_name"),
        CharSpan(
            start=24,
            end=46,
            text="https://public.example",
            label="private_url",
        ),
        CharSpan(start=63, end=69, text="abc123", label="secret"),
    ]
    assert all(text[span.start : span.end] == span.text for span in spans)
    assert {span.label for span in spans} <= OPENMED_SUPPORTED_LABELS


def test_stratified_subset_balances_strata_and_pins_selected_ids() -> None:
    rows = (
        [_row(f"ext-en-{i}", dataset="external", shard="ai4privacy_en", language="en") for i in range(3)]
        + [_row(f"ext-vi-{i}", dataset="external", shard="ai4privacy_vi", language="vi") for i in range(3)]
        + [_row(f"v2-en-{i}", dataset="v2-eval", shard="en", language="en") for i in range(3)]
        + [_row(f"v2-vi-{i}", dataset="v2-eval", shard="vi", language="vi") for i in range(3)]
    )

    subset = build_pinned_subset(
        rows,
        StratifiedSubsetSpec(
            name="unit",
            target_rows=4,
            strata=("dataset", "language"),
        ),
    )

    assert [row.stable_id for row in subset.rows] == [
        "external:ai4privacy_en:ext-en-0",
        "external:ai4privacy_vi:ext-vi-0",
        "v2-eval:en:v2-en-0",
        "v2-eval:vi:v2-vi-0",
    ]
    assert subset.sha256 == ("bf53fc9b4daac92b35e01d69df08c889465dc7ed38c648fbb14701c1ff5df509")
    assert subset.stratum_counts == {
        "dataset=external|language=en": 1,
        "dataset=external|language=vi": 1,
        "dataset=v2-eval|language=en": 1,
        "dataset=v2-eval|language=vi": 1,
    }


def test_named_subset_builders_pin_smoke_rows_and_cap_external_configs() -> None:
    rows = (
        [_row(f"ext-en-{i}", dataset="external", shard="ai4privacy_en", language="en") for i in range(3)]
        + [_row(f"ext-vi-{i}", dataset="external", shard="ai4privacy_vi", language="vi") for i in range(3)]
        + [_row(f"v2-en-{i}", dataset="v2-eval", shard="en", language="en") for i in range(3)]
    )

    smoke = build_smoke_subset(rows, target_rows=5)
    external = build_external_stratified_subset(
        rows,
        target_spans=4,
        max_rows_per_config=1,
    )

    assert smoke.name == "smoke-200"
    assert len(smoke.rows) == 5
    assert len(external.rows) == 2
    assert {row.shard for row in external.rows} == {"ai4privacy_en", "ai4privacy_vi"}
    assert external.stratum_counts == {
        "dataset=external|shard=ai4privacy_en": 1,
        "dataset=external|shard=ai4privacy_vi": 1,
    }


def test_supported_subset_scoring_filters_gold_before_metrics() -> None:
    predicted_by_doc = {"doc-1": [CharSpan(start=0, end=4, text="John", label="human_name")]}
    gold_by_doc = {
        "doc-1": [
            CharSpan(start=0, end=4, text="John", label="human_name"),
            CharSpan(start=14, end=27, text="secret portal", label="private_url"),
        ],
    }

    reports = score_reports(
        predicted_by_doc,
        gold_by_doc,
        supported_labels=frozenset({"human_name"}),
    )

    assert filter_gold_to_supported(gold_by_doc, frozenset({"human_name"})) == predicted_by_doc
    assert reports["supported_exact"]["typed"]["gold_total"] == 1.0
    assert reports["full_exact"]["typed"]["gold_total"] == 2.0
    assert reports["supported_exact"]["typed"]["f1"] == 1.0
    assert reports["full_exact"]["typed"]["recall"] == 0.5


def test_run_shard_skips_matching_fixture_without_predicting(tmp_path: Path) -> None:
    spec = ShardSpec(model="openmed", dataset="smoke", shard="part-000")
    rows = [_row("doc-1", dataset="smoke", shard="part-000", language="en")]
    run_shard(
        adapter=_FixedAdapter([[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=_CountingVolume(),
    )

    adapter = _ExplodingAdapter()
    volume = _CountingVolume()
    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=volume,
    )

    assert result.skipped is True
    assert result.rows_in == 0
    assert result.spans_out == 0
    assert volume.commits == 0
    assert not adapter.predict_called


def test_run_shard_does_not_resume_when_persisted_predictions_change(
    tmp_path: Path,
) -> None:
    """The completion sidecar binds the exact JSONL bytes, not only its fixture."""
    spec = ShardSpec(model="openmed", dataset="smoke", shard="part-000")
    rows = [_row("doc-1", dataset="smoke", shard="part-000", language="en")]
    contract = _evaluation_contract()
    run_shard(
        adapter=_FixedAdapter([[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=contract,
        volume=_CountingVolume(),
    )

    completed = run_shard(
        adapter=_ExplodingAdapter(),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=contract,
        volume=_CountingVolume(),
    )
    assert completed.skipped is True

    output_path = tmp_path / "results" / "openmed" / "smoke" / "part-000.jsonl"
    persisted_rows = list(read_jsonl(output_path))
    persisted_rows[0]["pred_spans"] = []
    write_jsonl(output_path, persisted_rows)

    with pytest.raises(AssertionError, match="predict must not be called"):
        run_shard(
            adapter=_ExplodingAdapter(),
            rows=rows,
            output_root=tmp_path,
            spec=spec,
            evaluation_contract=contract,
            volume=_CountingVolume(),
        )


def test_run_shard_reruns_when_same_spec_has_a_different_fixture(
    tmp_path: Path,
) -> None:
    spec = ShardSpec(model="openmed", dataset="smoke", shard="part-000")
    prediction = [[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]
    run_shard(
        adapter=_FixedAdapter(prediction),
        rows=[_row("doc-1", dataset="smoke", shard="part-000", language="en")],
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=_CountingVolume(),
    )

    result = run_shard(
        adapter=_FixedAdapter(prediction),
        rows=[_row("doc-2", dataset="smoke", shard="part-000", language="en")],
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=_CountingVolume(),
    )

    assert result.skipped is False


@pytest.mark.parametrize(
    ("first_language", "second_language", "first_slices", "second_slices"),
    [
        ("en", "vi", frozenset({"smoke"}), frozenset({"smoke"})),
        ("en", "en", frozenset({"smoke"}), frozenset({"challenge"})),
    ],
)
def test_run_shard_reruns_when_aggregate_fields_change(
    tmp_path: Path,
    first_language: str,
    second_language: str,
    first_slices: frozenset[str],
    second_slices: frozenset[str],
) -> None:
    spec = ShardSpec(model="openmed", dataset="smoke", shard="part-000")
    prediction = [[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]
    run_shard(
        adapter=_FixedAdapter(prediction),
        rows=[
            _row(
                "doc-1",
                dataset="smoke",
                shard="part-000",
                language=first_language,
                slices=first_slices,
            ),
        ],
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=_CountingVolume(),
    )

    result = run_shard(
        adapter=_FixedAdapter(prediction),
        rows=[
            _row(
                "doc-1",
                dataset="smoke",
                shard="part-000",
                language=second_language,
                slices=second_slices,
            ),
        ],
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=_CountingVolume(),
    )

    assert result.skipped is False


def test_run_shard_writes_jsonl_done_and_commits(tmp_path: Path) -> None:
    spec = ShardSpec(model="openmed", dataset="smoke", shard="part-000")
    adapter = _FixedAdapter([[CharSpan(start=8, end=18, text="John Smith", label="human_name")]])
    volume = _CountingVolume()

    result = run_shard(
        adapter=adapter,
        rows=[_row("doc-1", dataset="smoke", shard="part-000", language="en")],
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=volume,
    )

    jsonl_path = tmp_path / "results" / "openmed" / "smoke" / "part-000.jsonl"
    done_path = tmp_path / "results" / "openmed" / "smoke" / "part-000.done"
    meta_path = tmp_path / "results" / "openmed" / "smoke" / "part-000.meta.json"
    written = list(read_jsonl(jsonl_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert result.skipped is False
    assert result.rows_in == 1
    assert result.spans_out == 1
    assert result.output_path == jsonl_path
    assert done_path.read_text(encoding="utf-8") == "done\n"
    assert metadata["result_sha256"] == file_sha256(str(jsonl_path))
    assert volume.commits == 1
    assert written[0]["pred_spans"] == [{"start": 8, "end": 18, "text": "John Smith", "label": "human_name"}]
    assert written[0]["gold_spans"] == [{"start": 8, "end": 18, "text": "John Smith", "label": "human_name"}]


def test_read_matrix_results_rejects_sidecar_unbound_from_persisted_rows(
    tmp_path: Path,
) -> None:
    spec = ShardSpec(model="openmed", dataset="v2-eval", shard="full")
    run_shard(
        adapter=_FixedAdapter([[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]),
        rows=[_row("doc-1", dataset="v2-eval", shard="full", language="en")],
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_evaluation_contract(),
        volume=_CountingVolume(),
    )

    accepted = read_matrix_results(
        tmp_path,
        "openmed",
        ("v2-eval", "nemotron_en"),
        require_done=True,
        expected_rows={"v2-eval": 1, "nemotron_en": 99_892},
        expected_evaluation_contract=_evaluation_contract(),
        expected_dataset_shard_identities={
            "v2-eval": dataset_shard_identity(
                _evaluation_contract(),
                dataset="v2-eval",
                shard="full",
                rows=[_row("doc-1", dataset="v2-eval", shard="full", language="en")],
            ),
        },
    )
    assert set(accepted.rows_by_dataset) == {"v2-eval"}
    assert accepted.missing_datasets == ("nemotron_en",)
    assert accepted.timing_by_dataset["v2-eval"]["rows"] == 1.0

    meta_path = tmp_path / "results" / "openmed" / "v2-eval" / "full.meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["fixture_sha256"] = "a" * 64
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    matrix = read_matrix_results(
        tmp_path,
        "openmed",
        ("v2-eval",),
        require_done=True,
        expected_rows={"v2-eval": 1},
        expected_evaluation_contract=_evaluation_contract(),
        expected_dataset_shard_identities={
            "v2-eval": dataset_shard_identity(
                _evaluation_contract(),
                dataset="v2-eval",
                shard="full",
                rows=[_row("doc-1", dataset="v2-eval", shard="full", language="en")],
            ),
        },
    )

    assert matrix.rows_by_dataset == {}
    assert matrix.missing_datasets == ("v2-eval",)


def test_read_matrix_results_rejects_prediction_mutation_after_completion(
    tmp_path: Path,
) -> None:
    spec = ShardSpec(model="openmed", dataset="v2-eval", shard="full")
    rows = [_row("doc-1", dataset="v2-eval", shard="full", language="en")]
    contract = _evaluation_contract()
    run_shard(
        adapter=_FixedAdapter([[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=contract,
        volume=_CountingVolume(),
    )
    output_path = tmp_path / "results" / "openmed" / "v2-eval" / "full.jsonl"
    result_rows = list(read_jsonl(output_path))
    result_rows[0]["pred_spans"] = []
    write_jsonl(output_path, result_rows)

    matrix = _read_completed_v2_matrix(tmp_path, contract, rows)

    assert matrix.rows_by_dataset == {}
    assert matrix.missing_datasets == ("v2-eval",)


def test_read_matrix_results_rejects_legacy_sidecar_without_result_digest(
    tmp_path: Path,
) -> None:
    spec = ShardSpec(model="openmed", dataset="v2-eval", shard="full")
    rows = [_row("doc-1", dataset="v2-eval", shard="full", language="en")]
    contract = _evaluation_contract()
    run_shard(
        adapter=_FixedAdapter([[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=contract,
        volume=_CountingVolume(),
    )
    meta_path = tmp_path / "results" / "openmed" / "v2-eval" / "full.meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    del metadata["result_sha256"]
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    matrix = _read_completed_v2_matrix(tmp_path, contract, rows)

    assert matrix.rows_by_dataset == {}
    assert matrix.missing_datasets == ("v2-eval",)


@pytest.mark.parametrize("corruption", ["missing", "truncated", "stale_pair"])
def test_read_matrix_results_rejects_incomplete_or_stale_result_artifacts(
    tmp_path: Path,
    corruption: str,
) -> None:
    spec = ShardSpec(model="openmed", dataset="v2-eval", shard="full")
    rows = [_row("doc-1", dataset="v2-eval", shard="full", language="en")]
    contract = _evaluation_contract()
    first_prediction = [[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]
    run_shard(
        adapter=_FixedAdapter(first_prediction),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=contract,
        volume=_CountingVolume(),
    )
    output_path = tmp_path / "results" / "openmed" / "v2-eval" / "full.jsonl"
    if corruption == "missing":
        output_path.unlink()
    elif corruption == "truncated":
        output_path.write_bytes(b'{"pred_spans":')
    else:
        stale_result = output_path.read_bytes()
        run_shard(
            adapter=_FixedAdapter([[]]),
            rows=rows,
            output_root=tmp_path,
            spec=spec,
            evaluation_contract=contract,
            volume=_CountingVolume(),
            force=True,
        )
        output_path.write_bytes(stale_result)

    matrix = _read_completed_v2_matrix(tmp_path, contract, rows)

    assert matrix.rows_by_dataset == {}
    assert matrix.missing_datasets == ("v2-eval",)


def test_result_source_revalidates_digest_when_consumed(tmp_path: Path) -> None:
    spec = ShardSpec(model="openmed", dataset="v2-eval", shard="full")
    rows = [_row("doc-1", dataset="v2-eval", shard="full", language="en")]
    contract = _evaluation_contract()
    run_shard(
        adapter=_FixedAdapter([[CharSpan(start=8, end=18, text="John Smith", label="human_name")]]),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=contract,
        volume=_CountingVolume(),
    )
    matrix = _read_completed_v2_matrix(tmp_path, contract, rows)
    output_path = tmp_path / "results" / "openmed" / "v2-eval" / "full.jsonl"
    output_path.write_bytes(b'{"pred_spans":')

    with pytest.raises(ValueError, match="result artifact digest mismatch"):
        list(matrix.rows_by_dataset["v2-eval"]())


class _CountingVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class _FixedAdapter:
    name = "openmed"
    supported_labels = OPENMED_SUPPORTED_LABELS

    def __init__(self, predictions: list[list[CharSpan]]) -> None:
        self._predictions = predictions

    @staticmethod
    def load() -> None:
        return None

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        assert texts
        return self._predictions


class _ExplodingAdapter:
    name = "openmed"
    supported_labels = OPENMED_SUPPORTED_LABELS

    def __init__(self) -> None:
        self.predict_called = False

    @staticmethod
    def load() -> None:
        return None

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        self.predict_called = True
        msg = f"predict must not be called: {texts!r}"
        raise AssertionError(msg)
