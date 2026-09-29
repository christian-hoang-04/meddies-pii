from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.eval_baseline.baseline.aggregate import (
    aggregate_results as _aggregate_results,
)
from meddies_pii.eval_baseline.baseline.aggregate import (
    format_aggregate_report,
)
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    EvaluationContract,
    dataset_shard_identity_from_result_rows,
)

if TYPE_CHECKING:
    from meddies_pii.eval_baseline.baseline.aggregate import AggregateReport
    from meddies_pii.taxonomy import PiiLabel

_ROWS = {
    "cfg_en": [
        {
            "doc_id": "d1",
            "language": "en",
            "pred_spans": [{"start": 0, "end": 4, "text": "John", "label": "human_name"}],
            "gold_spans": [{"start": 0, "end": 4, "text": "John", "label": "human_name"}],
        },
        {
            "doc_id": "d2",
            "language": "en",
            "pred_spans": [{"start": 0, "end": 5, "text": "Johns", "label": "human_name"}],
            "gold_spans": [{"start": 0, "end": 4, "text": "John", "label": "human_name"}],
        },
    ],
    "cfg_vi": [
        {
            "doc_id": "d3",
            "language": "vi",
            "pred_spans": [],
            "gold_spans": [{"start": 0, "end": 3, "text": "Lan", "label": "human_name"}],
        },
    ],
}
"""Three docs, human_name only, hand-computed.

D1 exact match; d2 boundary-off (0,5) vs gold (0,4); d3 a pure miss. exact:        tp=1, pred=2, gold=3 -> P=0.5  R=0.333
F1=0.4 containment:  precision_tp=1 (d1), recall_tp=2 (d1,d2), pred=2, gold=3 -> P=0.5 R=0.667 F1=0.5714.

"""
_SUPPORTED: frozenset[PiiLabel] = frozenset({"human_name"})


def _result_sha256(rows: list[dict[str, object]]) -> str:
    persisted = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows)
    return hashlib.sha256(persisted).hexdigest()


# reason: the real parameter type is Mapping[str, EvalRowSource], and the erased row dicts this helper builds are
# reason: not assignable to it; narrowing the whole EvalRowSource chain is its own unit of work.
def aggregate_results(
    rows_by_config,  # ruff: ignore[missing-type-function-argument]
    *,
    supported_labels,  # ruff: ignore[missing-type-function-argument]
    evaluation_contract: EvaluationContract | None = None,
    **kwargs,  # ruff: ignore[missing-type-kwargs]
) -> AggregateReport:
    def artifact(reference: str, character: str) -> ArtifactIdentity:
        return ArtifactIdentity(reference, "f" * 40, character * 64)

    contract = evaluation_contract or EvaluationContract(
        model=artifact("model", "a"),
        vendor_inference_source=artifact("vendor", "b"),
        local_adapter_source=artifact("adapter", "c"),
        applied_label_prediction_contract=artifact("labels", "d"),
        decoder_contract=artifact("decoder", "e"),
        resolved_runtime_environment=artifact("runtime", "f"),
        scorer_contract=artifact("scorer", "0"),
        supported_labels=tuple(sorted(supported_labels)),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )
    normalized = {
        name: [
            {
                **row,
                "id": hashlib.sha256(str(row.get("id", row["doc_id"])).encode("utf-8")).hexdigest(),
                "slice": row.get("slice", []),
            }
            for row in rows
        ]
        for name, rows in rows_by_config.items()
    }
    identities = {
        name: dataset_shard_identity_from_result_rows(
            contract,
            dataset=name,
            shard="full",
            rows=rows,
        )
        for name, rows in normalized.items()
    }
    return _aggregate_results(
        normalized,
        supported_labels=supported_labels,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=identities,
        result_sha256_by_config={name: _result_sha256(rows) for name, rows in normalized.items()},
        **kwargs,
    )


def test_overall_exact_and_containment_f1() -> None:
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    overall = report["overall"]
    assert overall["rows"] == 3
    assert overall["gold_spans"] == 3
    assert overall["pred_spans"] == 2
    assert overall["exact_f1"] == pytest.approx(0.4, abs=1e-3)
    assert overall["containment_f1"] == pytest.approx(0.5714, abs=1e-3)


def test_aggregate_binds_the_evaluation_contract_payload_and_digest() -> None:
    def artifact(reference: str, character: str) -> ArtifactIdentity:
        return ArtifactIdentity(reference, "f" * 40, character * 64)

    contract = EvaluationContract(
        model=artifact("model-a", "a"),
        vendor_inference_source=artifact("vendor", "b"),
        local_adapter_source=artifact("adapter", "c"),
        applied_label_prediction_contract=artifact("labels", "d"),
        decoder_contract=artifact("decoder", "e"),
        resolved_runtime_environment=artifact("runtime", "f"),
        scorer_contract=artifact("scorer", "0"),
        supported_labels=("human_name",),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )

    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED, evaluation_contract=contract)

    assert report["evaluation_contract"] == {
        "payload": contract.to_payload(),
        "sha256": contract.digest,
    }


def test_per_language_split() -> None:
    """en: tp=1 pred=2 gold=2 -> F1=0.5 ; vi: pure miss -> 0."""
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    per_language = report["per_language"]
    assert set(per_language) == {"en", "vi"}
    assert per_language["en"]["exact_f1"] == pytest.approx(0.5, abs=1e-3)
    assert per_language["vi"]["exact_f1"] == pytest.approx(0.0, abs=1e-3)


def test_per_config_preserves_source_breakdown() -> None:
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    assert set(report["per_config"]) == {"cfg_en", "cfg_vi"}
    assert report["per_config"]["cfg_vi"]["gold_spans"] == 1
    assert report["per_config"]["cfg_vi"]["pred_spans"] == 0


def test_per_label_and_macro() -> None:
    """Single supported label -> macro == that label's F1."""
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    human = report["per_label"]["human_name"]["exact"]
    assert human["f1"] == pytest.approx(0.4, abs=1e-3)
    assert report["overall"]["macro_exact_f1"] == pytest.approx(0.4, abs=1e-3)


def test_full_taxonomy_scores_penalize_labels_the_model_cannot_emit() -> None:
    rows = {
        **_ROWS,
        "cfg_company": [
            {
                "doc_id": "d4",
                "language": "en",
                "pred_spans": [],
                "gold_spans": [
                    {
                        "start": 0,
                        "end": 4,
                        "text": "Acme",
                        "label": "company_name",
                    },
                ],
            },
        ],
    }

    report = aggregate_results(rows, supported_labels=_SUPPORTED)

    assert report["overall"]["exact_f1"] == pytest.approx(0.4, abs=1e-3)
    assert report["overall"]["full9_exact_f1"] == pytest.approx(1 / 3, abs=1e-3)
    assert report["overall"]["full9_gold_spans"] == 4
    company = report["per_label_full9"]["company_name"]["exact"]
    assert company["gold_total"] == 1
    assert company["pred_total"] == 0
    assert company["f1"] == 0.0


def test_fixture_digest_tracks_order_and_gold_identity() -> None:
    forward = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    reversed_rows = {config: list(reversed(rows)) for config, rows in reversed(list(_ROWS.items()))}
    reordered = aggregate_results(reversed_rows, supported_labels=_SUPPORTED)
    changed = {
        **_ROWS,
        "cfg_en": [
            *_ROWS["cfg_en"][:-1],
            {
                **_ROWS["cfg_en"][-1],
                "gold_spans": [
                    {
                        "start": 0,
                        "end": 5,
                        "text": "Johns",
                        "label": "human_name",
                    },
                ],
            },
        ],
    }
    changed_report = aggregate_results(changed, supported_labels=_SUPPORTED)

    assert forward["fixture"]["sha256"] != reordered["fixture"]["sha256"]
    assert forward["fixture"]["sha256"] != changed_report["fixture"]["sha256"]


def test_fixture_digest_tracks_slices_through_dataset_shard_identity() -> None:
    forward = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    sliced = {
        **_ROWS,
        "cfg_en": [{**row, "slice": ["challenge"]} for row in _ROWS["cfg_en"]],
    }

    changed = aggregate_results(sliced, supported_labels=_SUPPORTED)

    assert forward["fixture"]["sha256"] != changed["fixture"]["sha256"]
    assert "fixture_sha256" in forward["fixture"]["configs"]["cfg_en"]


def test_same_result_artifacts_remain_bound_to_distinct_model_contracts() -> None:
    first = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    second_contract = EvaluationContract(
        model=ArtifactIdentity("model", "f" * 40, "9" * 64),
        vendor_inference_source=ArtifactIdentity("vendor", "f" * 40, "b" * 64),
        local_adapter_source=ArtifactIdentity("adapter", "f" * 40, "c" * 64),
        applied_label_prediction_contract=ArtifactIdentity("labels", "f" * 40, "d" * 64),
        decoder_contract=ArtifactIdentity("decoder", "f" * 40, "e" * 64),
        resolved_runtime_environment=ArtifactIdentity("runtime", "f" * 40, "f" * 64),
        scorer_contract=ArtifactIdentity("scorer", "f" * 40, "0" * 64),
        supported_labels=("human_name",),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )
    second = aggregate_results(
        _ROWS,
        supported_labels=_SUPPORTED,
        evaluation_contract=second_contract,
    )

    assert first["fixture"]["sha256"] == second["fixture"]["sha256"]
    assert first["fixture"]["configs"] == second["fixture"]["configs"]
    assert first["provenance"] == second["provenance"]
    assert first["evaluation_contract"]["sha256"] != second["evaluation_contract"]["sha256"]


def test_result_provenance_distinguishes_score_changing_reruns() -> None:
    first = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    rerun = aggregate_results(
        {
            **_ROWS,
            "cfg_en": [{**_ROWS["cfg_en"][0], "pred_spans": []}, *_ROWS["cfg_en"][1:]],
        },
        supported_labels=_SUPPORTED,
    )

    assert first["fixture"] == rerun["fixture"]
    assert first["overall"]["exact_f1"] != rerun["overall"]["exact_f1"]
    assert first["provenance"]["result_set_sha256"] != rerun["provenance"]["result_set_sha256"]
    assert set(first["provenance"]["result_sha256_by_config"]) == set(_ROWS)


def test_duplicate_row_identifiers_are_scored_independently() -> None:
    duplicate_identity_rows = {
        "cfg_a": [
            {
                "id": "same-text-sha256",
                "doc_id": "same-source-id",
                "language": "en",
                "pred_spans": [
                    {
                        "start": 0,
                        "end": 4,
                        "text": "John",
                        "label": "human_name",
                    },
                ],
                "gold_spans": [
                    {
                        "start": 0,
                        "end": 4,
                        "text": "John",
                        "label": "human_name",
                    },
                ],
            },
            {
                "id": "same-text-sha256",
                "doc_id": "same-source-id",
                "language": "en",
                "pred_spans": [
                    {
                        "start": 5,
                        "end": 8,
                        "text": "Doe",
                        "label": "human_name",
                    },
                ],
                "gold_spans": [
                    {
                        "start": 5,
                        "end": 8,
                        "text": "Doe",
                        "label": "human_name",
                    },
                ],
            },
        ],
        "cfg_b": [
            {
                "id": "same-text-sha256",
                "doc_id": "same-source-id",
                "language": "en",
                "pred_spans": [
                    {
                        "start": 9,
                        "end": 14,
                        "text": "Smith",
                        "label": "human_name",
                    },
                ],
                "gold_spans": [
                    {
                        "start": 9,
                        "end": 14,
                        "text": "Smith",
                        "label": "human_name",
                    },
                ],
            },
        ],
    }

    report = aggregate_results(duplicate_identity_rows, supported_labels=_SUPPORTED)
    reordered = aggregate_results(
        {config: list(reversed(rows)) for config, rows in duplicate_identity_rows.items()},
        supported_labels=_SUPPORTED,
    )

    assert report["fixture"]["full9_gold_spans"] == 3
    assert report["fixture"]["full9_gold_spans"] == report["overall"]["full9_gold_spans"]
    assert report["fixture"]["sha256"] != reordered["fixture"]["sha256"]
    assert report["overall"]["full9_gold_spans"] == 3
    assert report["overall"]["full9_pred_spans"] == 3
    assert report["per_language"]["en"]["full9_gold_spans"] == 3
    assert report["per_language"]["en"]["full9_pred_spans"] == 3
    assert report["per_config"]["cfg_a"]["full9_gold_spans"] == 2
    assert report["per_config"]["cfg_a"]["full9_pred_spans"] == 2
    assert report["per_config"]["cfg_b"]["full9_gold_spans"] == 1
    assert report["per_config"]["cfg_b"]["full9_pred_spans"] == 1
    for config, fixture_summary in report["fixture"]["configs"].items():
        assert fixture_summary["full9_gold_spans"] == report["per_config"][config]["full9_gold_spans"]


def test_format_report_renders_all_sections() -> None:
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    text = format_aggregate_report(report, model="openmed")
    assert "openmed baseline" in text
    assert "per config" in text
    assert "per language" in text
    assert "human_name" in text


_TIMING = {"cfg_en": {"elapsed_seconds": 4.0}, "cfg_vi": {"elapsed_seconds": 2.0}}


def test_timing_adds_rows_per_second_per_config_and_overall() -> None:
    """cfg_en: 2 rows / 4.0s = 0.5 rows/s ; cfg_vi: 1 / 2.0 = 0.5."""
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED, timing_by_config=_TIMING)
    assert report["per_config"]["cfg_en"]["elapsed_seconds"] == pytest.approx(4.0)
    assert report["per_config"]["cfg_en"]["rows_per_second"] == pytest.approx(0.5)
    assert report["overall"]["elapsed_seconds"] == pytest.approx(6.0)
    assert report["overall"]["rows_per_second"] == pytest.approx(0.5)


def test_no_timing_omits_speed_keys() -> None:
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED)
    assert "rows_per_second" not in report["overall"]
    assert "rows_per_second" not in report["per_config"]["cfg_en"]


def test_format_report_shows_rows_per_second_when_timed() -> None:
    report = aggregate_results(_ROWS, supported_labels=_SUPPORTED, timing_by_config=_TIMING)
    assert "rows/s" in format_aggregate_report(report, model="opf")


def test_exact_and_containment_keep_boundary_and_label_failures_distinct() -> None:
    rows = {
        "cfg": [
            {
                "doc_id": "inside",
                "language": "en",
                "pred_spans": [{"start": 1, "end": 3, "text": "oh", "label": "human_name"}],
                "gold_spans": [{"start": 0, "end": 4, "text": "John", "label": "human_name"}],
            },
            {
                "doc_id": "around",
                "language": "en",
                "pred_spans": [{"start": 4, "end": 10, "text": "Jane D", "label": "human_name"}],
                "gold_spans": [{"start": 5, "end": 9, "text": "Jane", "label": "human_name"}],
            },
            {
                "doc_id": "wrong-label",
                "language": "en",
                "pred_spans": [{"start": 0, "end": 4, "text": "Mary", "label": "email_address"}],
                "gold_spans": [{"start": 0, "end": 4, "text": "Mary", "label": "human_name"}],
            },
        ],
    }

    report = aggregate_results(
        rows,
        supported_labels=frozenset({"human_name", "email_address"}),
    )
    overall = report["overall"]

    assert overall["full9_exact_f1"] == 0.0
    assert overall["full9_containment_f1"] == pytest.approx(1 / 3, abs=1e-4)
    assert overall["full9_containment_precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert overall["full9_containment_recall"] == pytest.approx(1 / 3, abs=1e-4)


def test_aggregate_rejects_malformed_persisted_fixture_spans() -> None:
    with pytest.raises(ValueError, match="invalid gold span"):
        aggregate_results(
            {
                "cfg": [
                    {
                        "doc_id": "d1",
                        "language": "en",
                        "pred_spans": [
                            {
                                "start": "0",
                                "end": 4,
                                "text": "John",
                                "label": "human_name",
                            },
                        ],
                        "gold_spans": [
                            {
                                "start": 0,
                                "end": 4,
                                "text": "John",
                                "label": "human_name",
                            },
                            {"start": 8, "end": 12, "text": "Jane"},
                        ],
                    },
                ],
            },
            supported_labels=_SUPPORTED,
        )
