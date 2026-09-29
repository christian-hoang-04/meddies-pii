"""ai4privacy_vi only in openmed -> gliner2 cell is the missing marker.

An aggregate with no timing keys still renders -- the speed cell is the missing marker, not a crash.

"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    EvaluationContract,
    canonical_sha256,
)
from meddies_pii.json_types import is_str_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping

_SPEC = importlib.util.spec_from_file_location(
    "build_baseline_eval_report",
    Path(__file__).resolve().parents[3] / "scripts" / "reports" / "build_baseline_eval_report.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
build_report = _MOD.build_report

_ALL_CONFIGS = (
    "v2-eval",
    "v2-eval-challenge",
    "ai4privacy_de",
    "ai4privacy_en",
    "ai4privacy_es",
    "ai4privacy_fil",
    "ai4privacy_fr",
    "ai4privacy_id",
    "ai4privacy_ja",
    "ai4privacy_ko",
    "ai4privacy_ms",
    "ai4privacy_pt",
    "ai4privacy_vi",
    "ai4privacy_zh",
    "creddata_en",
    "gretel_en",
    "nemotron_en",
)


def _section(report: str, heading: str) -> str:
    start = report.index(heading)
    next_heading = report.find("\n## ", start + len(heading))
    return report[start:] if next_heading == -1 else report[start:next_heading]


def _row(section: str, key: str) -> str:
    return next(line for line in section.splitlines() if line.startswith(f"| {key} |"))


def _digests_of(payload: Mapping[str, object]) -> dict[str, str]:
    """Return a payload's config-to-digest map, narrowing the two blocks it sits behind.

    `_provenance` is declared `dict[str, object]`, so both `payload["provenance"]` and the
    `result_sha256_by_config` block inside it read as `object` and cannot be copied into a
    dict until the shape is stated.

    Returns:
        A fresh copy of the map, which the caller then corrupts to build a malformed payload.

    """
    provenance = payload["provenance"]
    assert is_str_mapping(provenance), "provenance must be a mapping"
    by_config = provenance["result_sha256_by_config"]
    assert is_str_mapping(by_config), "result_sha256_by_config must be a mapping"
    digests: dict[str, str] = {}
    for config, digest in by_config.items():
        assert isinstance(digest, str), f"digest for {config} must be a str"
        digests[config] = digest
    return digests


def _provenance(configs: tuple[str, ...], marker: str) -> dict[str, object]:
    return _provenance_for_digests({
        config: hashlib.sha256(f"{marker}:{config}".encode()).hexdigest() for config in configs
    })


def _provenance_for_digests(
    result_sha256_by_config: dict[str, str],
) -> dict[str, object]:
    result_set_sha256 = canonical_sha256({"result_sha256_by_config": result_sha256_by_config})
    return {
        "result_set_sha256": result_set_sha256,
        "result_sha256_by_config": result_sha256_by_config,
    }


def _fixture_configs(configs: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        config: {
            "fixture_sha256": hashlib.sha256(config.encode()).hexdigest(),
            "rows": 1,
            "full9_gold_spans": 5,
        }
        for config in configs
    }


def _evaluation_contract(model_reference: str) -> dict[str, object]:
    def artifact(reference: str, value: str) -> ArtifactIdentity:
        return ArtifactIdentity(
            reference=reference,
            revision="a" * 40,
            sha256=hashlib.sha256(value.encode()).hexdigest(),
        )

    contract = EvaluationContract(
        model=artifact(model_reference, f"{model_reference}:model"),
        vendor_inference_source=artifact("vendor", f"{model_reference}:vendor"),
        local_adapter_source=artifact("adapter", f"{model_reference}:adapter"),
        applied_label_prediction_contract=artifact("labels", f"{model_reference}:labels"),
        decoder_contract=artifact("decoder", f"{model_reference}:decoder"),
        resolved_runtime_environment=artifact("runtime", f"{model_reference}:runtime"),
        scorer_contract=artifact("scorer", f"{model_reference}:scorer"),
        supported_labels=("human_name",),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )
    return {"payload": contract.to_payload(), "sha256": contract.digest}


_OPENMED = {
    "evaluation_contract": _evaluation_contract("hf://openmed"),
    "fixture": {
        "sha256": "same-fixture",
        "rows": 100,
        "full9_gold_spans": 500,
        "configs": _fixture_configs(_ALL_CONFIGS),
    },
    "provenance": _provenance(_ALL_CONFIGS, "openmed"),
    "overall": {
        "rows": 100,
        "gold_spans": 500,
        "exact_f1": 0.45,
        "macro_exact_f1": 0.42,
        "containment_f1": 0.64,
        "macro_containment_f1": 0.56,
        "full9_gold_spans": 500,
        "full9_exact_f1": 0.45,
        "full9_macro_exact_f1": 0.42,
        "full9_containment_f1": 0.64,
        "full9_macro_containment_f1": 0.56,
    },
    "per_config": {
        config: {
            "exact_f1": 0.50,
            "containment_f1": 0.60,
            "full9_exact_f1": 0.50,
            "full9_containment_f1": 0.60,
        }
        for config in _ALL_CONFIGS
    }
    | {
        "nemotron_en": {
            "exact_f1": 0.91,
            "containment_f1": 0.93,
            "full9_exact_f1": 0.91,
            "full9_containment_f1": 0.93,
        },
        "ai4privacy_vi": {
            "exact_f1": 0.50,
            "containment_f1": 0.71,
            "full9_exact_f1": 0.50,
            "full9_containment_f1": 0.71,
        },
    },
    "per_language": {
        "en": {
            "exact_f1": 0.62,
            "containment_f1": 0.75,
            "full9_exact_f1": 0.62,
            "full9_containment_f1": 0.75,
        },
        "vi": {
            "exact_f1": 0.50,
            "containment_f1": 0.71,
            "full9_exact_f1": 0.50,
            "full9_containment_f1": 0.71,
        },
    },
    "per_label": {
        "human_name": {
            "exact": {"f1": 0.39},
            "containment": {"f1": 0.59},
        },
        "company_name": {
            "exact": {"f1": 0.20},
            "containment": {"f1": 0.40},
        },
    },
    "per_label_full9": {
        "human_name": {
            "exact": {"f1": 0.39},
            "containment": {"f1": 0.59},
        },
        "company_name": {
            "exact": {"f1": 0.20},
            "containment": {"f1": 0.40},
        },
    },
}
"""openmed supports all 9; gliner2 drops company_name + private_url."""
_GLINER2 = {
    "evaluation_contract": _evaluation_contract("hf://gliner2"),
    "fixture": {
        "sha256": "same-fixture",
        "rows": 100,
        "full9_gold_spans": 500,
        "configs": _fixture_configs(_ALL_CONFIGS),
    },
    "provenance": _provenance(_ALL_CONFIGS, "gliner2"),
    "overall": {
        "rows": 100,
        "gold_spans": 480,
        "exact_f1": 0.40,
        "macro_exact_f1": 0.38,
        "containment_f1": 0.60,
        "macro_containment_f1": 0.52,
        "full9_gold_spans": 500,
        "full9_exact_f1": 0.35,
        "full9_macro_exact_f1": 0.30,
        "full9_containment_f1": 0.55,
        "full9_macro_containment_f1": 0.45,
    },
    "per_config": {
        "nemotron_en": {
            "exact_f1": 0.70,
            "containment_f1": 0.84,
            "full9_exact_f1": 0.65,
            "full9_containment_f1": 0.80,
        },
    },
    "per_language": {
        "en": {
            "exact_f1": 0.55,
            "containment_f1": 0.69,
            "full9_exact_f1": 0.50,
            "full9_containment_f1": 0.64,
        },
    },
    "per_label": {
        "human_name": {
            "exact": {"f1": 0.41},
            "containment": {"f1": 0.61},
        },
    },
    "per_label_full9": {
        "human_name": {
            "exact": {"f1": 0.41},
            "containment": {"f1": 0.61},
        },
        "company_name": {
            "exact": {"f1": 0.0},
            "containment": {"f1": 0.0},
        },
    },
}


def test_overall_has_every_model() -> None:
    report = build_report({"openmed": _OPENMED, "gliner2": _GLINER2}, generated="2026-06-30")
    assert "| openmed |" in report
    assert "| gliner2 |" in report
    assert "0.910" in report
    assert "fixed nine-label taxonomy" in report
    assert "Result artifact provenance" in report
    assert _OPENMED["provenance"]["result_set_sha256"] in report
    assert _GLINER2["provenance"]["result_set_sha256"] in report


def test_report_keeps_same_result_artifacts_distinct_by_evaluation_contract() -> None:
    same_result_provenance = _provenance(_ALL_CONFIGS, "same-result-artifacts")
    first = {
        **_OPENMED,
        "provenance": same_result_provenance,
        "evaluation_contract": _evaluation_contract("hf://first-model"),
    }
    second = {
        **_GLINER2,
        "provenance": same_result_provenance,
        "evaluation_contract": _evaluation_contract("hf://second-model"),
    }

    report = build_report({"first": first, "second": second}, generated="2026-06-30")

    first_contract = first["evaluation_contract"]
    second_contract = second["evaluation_contract"]
    assert first["fixture"] == second["fixture"]
    assert first["provenance"] == second["provenance"]
    assert first_contract["sha256"] != second_contract["sha256"]
    assert same_result_provenance["result_set_sha256"] in report
    assert f"| first | `{first_contract['sha256']}` | hf://first-model |" in report
    assert f"| second | `{second_contract['sha256']}` | hf://second-model |" in report


@pytest.mark.parametrize("corruption", ["missing", "malformed", "mismatched"])
def test_report_rejects_invalid_evaluation_contract_provenance(corruption: str) -> None:
    aggregate = dict(_OPENMED)
    if corruption == "missing":
        aggregate.pop("evaluation_contract")
    elif corruption == "malformed":
        aggregate["evaluation_contract"] = {
            "payload": {"schema_version": 2},
            "sha256": "0" * 64,
        }
    else:
        aggregate["evaluation_contract"] = {
            **_OPENMED["evaluation_contract"],
            "sha256": "0" * 64,
        }

    with pytest.raises(ValueError, match="evaluation contract"):
        build_report({"openmed": aggregate}, generated="2026-06-30")


def test_every_config_renders_in_separate_exact_and_containment_sections() -> None:
    report = build_report({"openmed": _OPENMED, "gliner2": _GLINER2}, generated="2026-06-30")
    exact = _section(report, "## Per config (source x language) — fixed nine-label exact F1")
    containment = _section(report, "## Per config (source x language) — fixed nine-label containment F1")

    assert len(_ALL_CONFIGS) == 17
    for config in _ALL_CONFIGS:
        assert f"| {config} |" in exact
        assert f"| {config} |" in containment
    assert "0.910" in _row(exact, "nemotron_en")
    assert "0.930" in _row(containment, "nemotron_en")


def test_language_and_label_sections_render_both_metric_definitions() -> None:
    report = build_report({"openmed": _OPENMED, "gliner2": _GLINER2}, generated="2026-06-30")

    language_exact = _section(report, "## Per language (ISO) — fixed nine-label exact F1")
    language_containment = _section(report, "## Per language (ISO) — fixed nine-label containment F1")
    label_exact = _section(report, "## Per label — fixed nine-label exact F1")
    label_containment = _section(report, "## Per label — fixed nine-label containment F1")

    assert "0.500" in _row(language_exact, "vi")
    assert "0.710" in _row(language_containment, "vi")
    assert "0.200" in _row(label_exact, "company_name")
    assert "0.400" in _row(label_containment, "company_name")


def test_unsupported_label_renders_not_supported() -> None:
    report = build_report({"openmed": _OPENMED, "gliner2": _GLINER2}, generated="2026-06-30")
    full9_exact = _section(report, "## Per label — fixed nine-label exact F1")
    supported_exact = _section(report, "## Supported-subset per label — exact F1 capability diagnostic")
    assert "0.200" in _row(full9_exact, "company_name")
    assert "0.000" in _row(full9_exact, "company_name")
    assert "n/s" in _row(supported_exact, "company_name")


def test_config_union_across_models() -> None:
    gliner_only = {
        **_GLINER2,
        "per_config": {
            "nemotron_en": _GLINER2["per_config"]["nemotron_en"],
            "gliner_only": {
                "full9_exact_f1": 0.44,
                "full9_containment_f1": 0.54,
            },
        },
    }
    report = build_report({"openmed": _OPENMED, "gliner2": gliner_only}, generated="2026-06-30")
    exact = _section(report, "## Per config (source x language) — fixed nine-label exact F1")
    vi_row = _row(exact, "ai4privacy_vi")
    assert "0.500" in vi_row
    assert "—" in vi_row


def test_inference_speed_section_renders_rows_per_second() -> None:
    timed = {
        "openmed": {
            **_OPENMED,
            "overall": {
                **_OPENMED["overall"],
                "elapsed_seconds": 50.0,
                "rows_per_second": 2.0,
            },
        },
    }
    report = build_report(timed, generated="2026-06-30")
    assert "## Inference speed" in report
    assert "| openmed | 2.0 |" in report


def test_inference_speed_missing_marker_when_untimed() -> None:
    report = build_report({"gliner2": _GLINER2}, generated="2026-06-30")
    assert "## Inference speed" in report
    assert "| gliner2 | — |" in report


def test_report_rejects_mixed_fixtures() -> None:
    mismatched = {
        **_GLINER2,
        "fixture": {**_GLINER2["fixture"], "sha256": "different-fixture"},
    }

    with pytest.raises(ValueError, match="fixture"):
        build_report(
            {"openmed": _OPENMED, "gliner2": mismatched},
            generated="2026-06-30",
        )


def test_report_rejects_mismatched_fixture_config_summary() -> None:
    mismatched = {
        **_GLINER2,
        "fixture": {
            **_GLINER2["fixture"],
            "configs": {"nemotron_en": {"rows": 99, "full9_gold_spans": 500}},
        },
    }

    with pytest.raises(ValueError, match="fixture"):
        build_report({"openmed": _OPENMED, "gliner2": mismatched}, generated="2026-06-30")


def test_report_compares_fixture_config_identity_not_result_identity() -> None:
    fixture_config = {
        "nemotron_en": {
            "fixture_sha256": "same-cell-fixture",
            "rows": 99_892,
            "full9_gold_spans": 500,
        },
    }
    first = {
        **_OPENMED,
        "provenance": _provenance(("nemotron_en",), "first"),
        "fixture": {
            **_OPENMED["fixture"],
            "configs": {
                **fixture_config,
                "nemotron_en": {
                    **fixture_config["nemotron_en"],
                    "dataset_shard_identity_sha256": "model-a-result",
                },
            },
        },
    }
    second = {
        **_GLINER2,
        "provenance": _provenance(("nemotron_en",), "second"),
        "fixture": {
            **_GLINER2["fixture"],
            "configs": {
                **fixture_config,
                "nemotron_en": {
                    **fixture_config["nemotron_en"],
                    "dataset_shard_identity_sha256": "model-b-result",
                },
            },
        },
    }

    report = build_report({"openmed": first, "gliner2": second}, generated="2026-06-30")

    assert "Fixture SHA-256: `same-fixture`." in report


@pytest.mark.parametrize("corruption", ["missing", "extra"])
def test_report_rejects_result_provenance_with_nonfixture_config_keys(
    corruption: str,
) -> None:
    digests = _digests_of(_OPENMED)
    if corruption == "missing":
        del digests[_ALL_CONFIGS[0]]
    else:
        digests["unexpected_config"] = hashlib.sha256(b"unexpected").hexdigest()
    malformed = {
        **_OPENMED,
        "provenance": _provenance_for_digests(digests),
    }

    with pytest.raises(ValueError, match="config keys do not match fixture configs"):
        build_report({"openmed": malformed}, generated="2026-06-30")


def test_inference_speed_disclaims_universal_h100_and_normalized_ranking() -> None:
    report = build_report({"openmed": _OPENMED, "gliner2": _GLINER2}, generated="2026-06-30")

    assert "throughput on one H100" not in report
    assert "Hardware and batching can differ" in report
    assert "not a normalized ranking" in report
