"""Regression tests for bounded-memory baseline aggregation."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import hashlib
from typing import TYPE_CHECKING

from meddies_pii.eval_baseline.baseline.aggregate import aggregate_results
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    DatasetShardIdentity,
    EvaluationContract,
    FixtureRowIdentity,
    canonical_sha256,
    dataset_shard_identity_from_result_rows,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import pytest

    from meddies_pii.evaluation.span_metrics import SpanMode
    from meddies_pii.json_types import JsonObject, JsonValue
    from meddies_pii.spans import CharSpan
    from meddies_pii.taxonomy import PiiLabel


def _contract(labels: frozenset[PiiLabel]) -> EvaluationContract:
    def artifact(reference: str, character: str) -> ArtifactIdentity:
        return ArtifactIdentity(reference, "f" * 40, character * 64)

    return EvaluationContract(
        model=artifact("model", "a"),
        vendor_inference_source=artifact("vendor", "b"),
        local_adapter_source=artifact("adapter", "c"),
        applied_label_prediction_contract=artifact("labels", "d"),
        decoder_contract=artifact("decoder", "e"),
        resolved_runtime_environment=artifact("runtime", "f"),
        scorer_contract=artifact("scorer", "0"),
        supported_labels=tuple(sorted(labels)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


def _span(start: int, end: int, text: str, label: str) -> JsonObject:
    return {"start": start, "end": end, "text": text, "label": label}


def _row(config: str, index: int) -> JsonObject:
    label = "human_name" if index % 3 else "email_address"
    gold = _span(0, 4, "John", label)
    prediction: list[JsonValue] = []
    if index % 4 == 0:
        prediction.append(gold)
    elif index % 4 == 1:
        prediction.append(_span(1, 4, "ohn", label))
    elif index % 4 == 2:
        prediction.append(
            _span(
                0,
                4,
                "John",
                "email_address" if label == "human_name" else "human_name",
            ),
        )
    return {
        "id": hashlib.sha256(f"{config}:{index}".encode()).hexdigest(),
        "doc_id": f"{config}:{index}",
        "pred_spans": prediction,
        "gold_spans": [gold],
        "language": "en" if config.endswith("en") else "vi",
        "slice": ["edge"] if index % 2 else [],
    }


def _rows(config: str, count: int) -> list[JsonObject]:
    return [_row(config, index) for index in range(count)]


def _expected_identities(
    contract: EvaluationContract,
    rows_by_config: dict[str, list[JsonObject]],
) -> dict[str, DatasetShardIdentity]:
    return {
        config: dataset_shard_identity_from_result_rows(contract, dataset=config, shard="full", rows=rows)
        for config, rows in rows_by_config.items()
    }


def _result_digests(configs: Iterable[object]) -> dict[str, str]:
    return {
        config: hashlib.sha256(f"synthetic-result:{config}".encode()).hexdigest()
        for config in configs
        if isinstance(config, str)
    }


def test_streaming_aggregate_matches_frozen_legacy_report_payload() -> None:
    """The canonical digest covers every report byte.

    Including its evaluation contract, fixture-only identity, and scored result-artifact provenance.

    """
    labels: frozenset[PiiLabel] = frozenset({"email_address", "human_name"})
    rows_by_config = {"alpha_en": _rows("alpha_en", 11), "beta_vi": _rows("beta_vi", 7)}
    contract = _contract(labels)
    report = aggregate_results(
        rows_by_config,
        supported_labels=labels,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=_expected_identities(contract, rows_by_config),
        result_sha256_by_config=_result_digests(rows_by_config),
        timing_by_config={
            "alpha_en": {"elapsed_seconds": 1.25},
            "beta_vi": {"elapsed_seconds": 2.5},
        },
    )

    assert canonical_sha256(report) == "6311ea1603f722b7e81912e1402f7b8b277ea35d2de45aa86ddd3cf5a604004c"


def test_aggregate_accepts_reiterable_row_factories_without_matrix_lists() -> None:
    """A single source pass validates fixture order and merges count observations."""
    labels: frozenset[PiiLabel] = frozenset({"email_address", "human_name"})
    contract = _contract(labels)
    calls = 0

    def rows() -> Iterator[JsonObject]:
        nonlocal calls
        calls += 1
        yield from (_row("stream_en", index) for index in range(1_000))

    expected = {
        "stream_en": dataset_shard_identity_from_result_rows(
            contract,
            dataset="stream_en",
            shard="full",
            rows=list(rows()),
        ),
    }
    report = aggregate_results(
        {"stream_en": rows},
        supported_labels=labels,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=expected,
        result_sha256_by_config=_result_digests(expected),
    )

    assert calls == 2
    assert report["overall"]["rows"] == 1_000


def test_fixture_validation_visits_each_row_without_a_second_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stream comparator validates every row directly against frozen identity."""
    from meddies_pii.eval_baseline.baseline import aggregate as aggregate_module

    labels: frozenset[PiiLabel] = frozenset({"email_address", "human_name"})
    contract = _contract(labels)
    rows_by_config = {"bounded_en": _rows("bounded_en", 13)}
    expected = _expected_identities(contract, rows_by_config)
    original = aggregate_module._result_row_matches_fixture
    calls = 0

    def tracked(row: aggregate_module.EvalRowDict, fixture_row: FixtureRowIdentity) -> bool:
        nonlocal calls
        calls += 1
        return original(row, fixture_row)

    monkeypatch.setattr(aggregate_module, "_result_row_matches_fixture", tracked)
    aggregate_results(
        rows_by_config,
        supported_labels=labels,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=expected,
        result_sha256_by_config=_result_digests(expected),
    )

    assert calls == 13


def test_each_row_calls_general_metric_helpers_four_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config/language/overall reports merge one shared metric observation."""
    from meddies_pii.eval_baseline.baseline import aggregate as aggregate_module

    labels: frozenset[PiiLabel] = frozenset({"email_address", "human_name"})
    contract = _contract(labels)
    rows_by_config = {"counts_en": _rows("counts_en", 5)}
    expected = _expected_identities(contract, rows_by_config)
    exact_calls = 0
    containment_calls = 0
    original_exact = aggregate_module.exact_span_counts
    original_containment = aggregate_module.containment_span_counts

    def exact(
        predicted: Sequence[CharSpan],
        gold: Sequence[CharSpan],
        *,
        mode: SpanMode = "typed",
    ) -> dict[str, int]:
        nonlocal exact_calls
        exact_calls += 1
        return original_exact(predicted, gold, mode=mode)

    def containment(
        predicted: Sequence[CharSpan],
        gold: Sequence[CharSpan],
        *,
        mode: SpanMode = "typed",
    ) -> dict[str, int]:
        nonlocal containment_calls
        containment_calls += 1
        return original_containment(predicted, gold, mode=mode)

    monkeypatch.setattr(aggregate_module, "exact_span_counts", exact)
    monkeypatch.setattr(aggregate_module, "containment_span_counts", containment)
    aggregate_results(
        rows_by_config,
        supported_labels=labels,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=expected,
        result_sha256_by_config=_result_digests(expected),
    )

    assert exact_calls == 10
    assert containment_calls == 10
