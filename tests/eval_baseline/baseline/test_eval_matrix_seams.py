from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from meddies_pii.eval_baseline.baseline import datasets
from meddies_pii.eval_baseline.baseline.aggregate import aggregate_results
from meddies_pii.eval_baseline.baseline.run import read_matrix_results
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    DatasetShardIdentity,
    EvaluationContract,
    file_sha256,
)
from meddies_pii.jsonl import write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from meddies_pii.json_types import JsonObject


def _expected_identity() -> tuple[EvaluationContract, DatasetShardIdentity]:
    def artifact(name: str, value: str) -> ArtifactIdentity:
        return ArtifactIdentity(name, "f" * 40, value * 64)

    contract = EvaluationContract(
        model=artifact("model", "a"),
        vendor_inference_source=artifact("vendor", "b"),
        local_adapter_source=artifact("adapter", "c"),
        applied_label_prediction_contract=artifact("labels", "d"),
        decoder_contract=artifact("decoder", "e"),
        resolved_runtime_environment=artifact("runtime", "f"),
        scorer_contract=artifact("scorer", "0"),
        supported_labels=("human_name",),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )
    return contract, DatasetShardIdentity(contract, "v2-eval", "full", (), 0)


def test_select_eval_datasets_preserves_requested_order_and_rejects_duplicates() -> None:
    assert datasets.select_eval_datasets("nemotron_en,v2-eval") == (
        "nemotron_en",
        "v2-eval",
    )
    with pytest.raises(ValueError, match=r"duplicate datasets: \['nemotron_en'\]"):
        datasets.select_eval_datasets("nemotron_en,v2-eval,nemotron_en")


def test_load_eval_cell_passes_explicit_limits_to_its_dataset_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int | None]] = []

    def load_v2(config: str, *, limit: int | None, **_: object) -> list[str]:
        calls.append(("v2", config, limit))
        return [config]

    def load_external(config: str, *, limit: int | None, **_: object) -> list[str]:
        calls.append(("external", config, limit))
        return [config]

    monkeypatch.setattr(datasets, "load_v2_eval_rows", load_v2)
    monkeypatch.setattr(datasets, "load_external_rows", load_external)

    assert datasets.load_eval_cell("v2-eval-challenge", v2_limit=12, external_limit=34) == ["eval-challenge"]
    assert datasets.load_eval_cell("nemotron_en", v2_limit=12, external_limit=34) == ["nemotron_en"]
    assert calls == [
        ("v2", "eval-challenge", 12),
        ("external", "nemotron_en", 34),
    ]


def test_matrix_selection_and_cell_loader_reject_unknown_dataset() -> None:
    with pytest.raises(ValueError, match=r"unknown datasets: \['missing'\]"):
        datasets.select_eval_datasets("missing")
    with pytest.raises(ValueError, match="unknown dataset: 'missing'"):
        datasets.load_eval_cell("missing", v2_limit=None, external_limit=None)


def test_read_matrix_results_rejects_wrong_frozen_row_counts(tmp_path: Path) -> None:
    contract, identity = _expected_identity()
    result_dir = tmp_path / "results" / "openmed" / "v2-eval"
    result_dir.mkdir(parents=True)
    write_jsonl(result_dir / "full.jsonl", [{"doc_id": "row-1"}])
    (result_dir / "full.meta.json").write_text(
        '{"elapsed_seconds": 1.25, "rows_in": 1, "spans_out": 0, '
        '"fixture_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen matrix row counts changed"):
        read_matrix_results(
            tmp_path,
            "openmed",
            ("v2-eval",),
            expected_rows={"v2-eval": 2},
            expected_evaluation_contract=contract,
            expected_dataset_shard_identities={"v2-eval": identity},
        )


def test_read_matrix_results_reports_missing_fixture_metadata(tmp_path: Path) -> None:
    contract, identity = _expected_identity()
    result_dir = tmp_path / "results" / "openmed" / "v2-eval"
    result_dir.mkdir(parents=True)
    write_jsonl(result_dir / "full.jsonl", [{"doc_id": "row-1"}])
    (result_dir / "full.done").write_text("done\n", encoding="utf-8")

    found = read_matrix_results(
        tmp_path,
        "openmed",
        ("v2-eval",),
        require_done=True,
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities={"v2-eval": identity},
    )

    assert found.rows_by_dataset == {}
    assert found.missing_datasets == ("v2-eval",)


def test_read_matrix_results_reports_invalid_fixture_metadata(tmp_path: Path) -> None:
    contract, identity = _expected_identity()
    result_dir = tmp_path / "results" / "openmed" / "v2-eval"
    result_dir.mkdir(parents=True)
    write_jsonl(result_dir / "full.jsonl", [{"doc_id": "row-1"}])
    (result_dir / "full.done").write_text("done\n", encoding="utf-8")
    (result_dir / "full.meta.json").write_text(
        '{"elapsed_seconds": 1.25, "rows_in": 1, "spans_out": 0, "fixture_sha256": "not-a-sha256"}',
        encoding="utf-8",
    )

    found = read_matrix_results(
        tmp_path,
        "openmed",
        ("v2-eval",),
        require_done=True,
        expected_rows={"v2-eval": 1},
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities={"v2-eval": identity},
    )

    assert found.rows_by_dataset == {}
    assert found.missing_datasets == ("v2-eval",)


def test_every_runner_uses_the_shared_matrix_seams() -> None:
    root = Path(__file__).resolve().parents[3]
    runner_paths = (
        "run_model_baseline.py",
        "run_opf_baseline.py",
        "run_gliner2_baseline.py",
        "run_lfm_bioes_baseline.py",
        "run_lfm25_pii_baseline.py",
    )

    for runner_name in runner_paths:
        source = (root / "scripts" / "ops" / runner_name).read_text(encoding="utf-8")
        assert "EVAL_DATASETS" in source
        assert "load_eval_cell" in source
        assert "select_eval_datasets" in source
        assert "read_matrix_results" in source
        assert "assert_frozen_fixture(report)" in source
        assert "EXTERNAL_CELLS" not in source


def test_read_matrix_results_defers_jsonl_open_until_aggregate_source_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matrix discovery validates the sidecar; the aggregate owns the sole row pass."""
    from meddies_pii.eval_baseline.baseline import run as run_module
    from meddies_pii.eval_baseline.baseline.run import write_shard_meta

    contract, identity = _expected_identity()
    result_dir = tmp_path / "results" / "openmed" / "v2-eval"
    result_dir.mkdir(parents=True)
    output_path = result_dir / "full.jsonl"
    write_jsonl(output_path, [])
    (result_dir / "full.done").write_text("done\n", encoding="utf-8")
    write_shard_meta(
        result_dir / "full.meta.json",
        elapsed_seconds=0.0,
        rows_in=0,
        spans_out=0,
        fixture_sha256=identity.digest,
        result_sha256=file_sha256(str(output_path)),
        environment_fingerprint={},
        evaluation_contract=contract,
        dataset_shard_identity=identity,
    )
    original_read = run_module.read_jsonl
    opens = 0

    def tracked(path: str | Path) -> Iterator[JsonObject]:
        nonlocal opens
        opens += 1
        yield from original_read(path)

    monkeypatch.setattr(run_module, "read_jsonl", tracked)
    matrix = read_matrix_results(
        tmp_path,
        "openmed",
        ("v2-eval",),
        require_done=True,
        expected_rows={"v2-eval": 0},
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities={"v2-eval": identity},
    )

    assert opens == 0
    assert matrix.result_sha256_by_dataset == {"v2-eval": file_sha256(str(output_path))}
    report = aggregate_results(
        matrix.rows_by_dataset,
        supported_labels=frozenset({"human_name"}),
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
        result_sha256_by_config=matrix.result_sha256_by_dataset,
    )
    assert report["overall"]["rows"] == 0
    assert opens == 1

    with pytest.raises(ValueError, match="source digest does not match"):
        aggregate_results(
            matrix.rows_by_dataset,
            supported_labels=frozenset({"human_name"}),
            expected_evaluation_contract=contract,
            expected_dataset_shard_identities=matrix.shard_identities_by_dataset,
            result_sha256_by_config={"v2-eval": "0" * 64},
        )
