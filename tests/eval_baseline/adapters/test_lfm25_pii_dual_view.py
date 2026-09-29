from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from meddies_pii.eval_baseline.adapters.lfm25_pii import Lfm25PiiSpaceAdapter
from meddies_pii.eval_baseline.baseline import artifact_export
from meddies_pii.eval_baseline.baseline import run as run_module
from meddies_pii.eval_baseline.baseline.artifact_export import (
    export_hub_safe_shard_package,
    validate_hub_safe_shard_package,
)
from meddies_pii.eval_baseline.baseline.datasets import EvalRow
from meddies_pii.eval_baseline.baseline.run import (
    ShardSpec,
    read_matrix_results,
    run_dual_view_shards,
    shard_output_path,
)
from meddies_pii.eval_baseline.baseline.views import (
    MODEL_CORE_VIEW,
    VENDOR_HYBRID_VIEW,
)
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    EvaluationContract,
    dataset_shard_identity,
    file_sha256,
)
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass
class _FakeDecoder:
    model_calls: list[str]

    def model_spans(self, text: str, _tokenizer: object, _model: object) -> list[dict[str, object]]:
        self.model_calls.append(text)
        if text == "no pii":
            return []
        return [{"start": 0, "end": 5, "type": "identity.person_name"}]

    @staticmethod
    def hybrid_spans(text: str, _model_spans: list[dict[str, object]]) -> list[dict[str, object]]:
        if text == "no pii":
            return []
        return [
            {"start": 0, "end": 5, "type": "identity.person_name"},
            {"start": 0, "end": 9, "type": "credential.api_key"},
        ]


class _FakeDualViewDetector:
    def __init__(self) -> None:
        self.hd = _FakeDecoder([])
        self.tok = object()
        self.model = object()
        self.lock = _TrackingLock()
        self.auth_priority_types = frozenset({"credential.api_key"})
        self.detect_calls: list[str] = []

    def detect(self, text: str) -> object:
        self.detect_calls.append(text)
        if text == "no pii":
            return {"spans": []}
        return {
            "spans": [
                {"start": 0, "end": 9, "type": "credential.api_key"},
            ],
        }


class _TrackingLock:
    def __init__(self) -> None:
        self.entries = 0

    def __enter__(self) -> object:
        self.entries += 1
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> bool:
        return False


def test_predict_views_runs_the_model_once_per_text_and_returns_named_views() -> None:
    detector = _FakeDualViewDetector()
    adapter = Lfm25PiiSpaceAdapter(detector)

    views = adapter.predict_views(["token1234", "no pii"])

    assert detector.hd.model_calls == ["token1234", "no pii"]
    assert detector.lock.entries == 2
    assert detector.detect_calls == []
    assert [[(span.start, span.end, span.label) for span in result.model_core] for result in views] == [
        [(0, 5, "human_name")],
        [],
    ]
    assert [[(span.start, span.end, span.label) for span in result.vendor_hybrid] for result in views] == [
        [(0, 9, "secret")],
        [],
    ]


def test_named_view_is_an_additive_shard_dimension_not_a_model_name(
    tmp_path: Path,
) -> None:
    core = ShardSpec(
        model="lfm25-pii",
        dataset="smoke",
        shard="full",
        view=MODEL_CORE_VIEW,
    )
    hybrid = ShardSpec(
        model="lfm25-pii",
        dataset="smoke",
        shard="full",
        view=VENDOR_HYBRID_VIEW,
    )

    assert core.model == hybrid.model == "lfm25-pii"
    assert shard_output_path(tmp_path, core) == (
        tmp_path / "results" / "lfm25-pii" / "model_core" / "smoke" / "full.jsonl"
    )
    assert shard_output_path(tmp_path, hybrid) == (
        tmp_path / "results" / "lfm25-pii" / "vendor_hybrid" / "smoke" / "full.jsonl"
    )


class _CountingVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


def _contract(view: str) -> EvaluationContract:
    def artifact(name: str, digest: str) -> ArtifactIdentity:
        return ArtifactIdentity(name, "f" * 40, digest * 64)

    return EvaluationContract(
        model=artifact("model", "a"),
        vendor_inference_source=artifact("vendor", "b"),
        local_adapter_source=artifact("adapter", "c"),
        applied_label_prediction_contract=artifact(f"labels-{view}", "d"),
        decoder_contract=artifact(f"decoder-{view}", "e"),
        resolved_runtime_environment=artifact("runtime", "f"),
        scorer_contract=artifact("scorer", "0"),
        supported_labels=("human_name", "secret"),
        result_schema=(
            "id",
            "doc_id",
            "pred_spans",
            "gold_spans",
            "language",
            "slice",
            "evaluation_view",
            "shared_inference_identity",
        ),
    )


def test_run_dual_view_shards_writes_two_bound_scoreable_artifacts(
    tmp_path: Path,
) -> None:
    detector = _FakeDualViewDetector()
    adapter = Lfm25PiiSpaceAdapter(detector)
    rows = [
        EvalRow(
            doc_id="doc-1",
            dataset="smoke",
            shard="full",
            text="token1234",
            gold_spans=(CharSpan(start=0, end=5, text="token", label="human_name"),),
            language="en",
            slices=frozenset({"smoke"}),
        ),
    ]
    volume = _CountingVolume()

    result = run_dual_view_shards(
        adapter=adapter,
        rows=rows,
        output_root=tmp_path,
        specs={
            MODEL_CORE_VIEW: ShardSpec("lfm25-pii", "smoke", "full", MODEL_CORE_VIEW),
            VENDOR_HYBRID_VIEW: ShardSpec("lfm25-pii", "smoke", "full", VENDOR_HYBRID_VIEW),
        },
        evaluation_contracts={
            MODEL_CORE_VIEW: _contract(MODEL_CORE_VIEW),
            VENDOR_HYBRID_VIEW: _contract(VENDOR_HYBRID_VIEW),
        },
        volume=volume,
    )

    assert detector.hd.model_calls == ["token1234"]
    assert set(result.by_view) == {MODEL_CORE_VIEW, VENDOR_HYBRID_VIEW}
    assert volume.commits == 1
    assert len(result.shared_inference_identity) == 64
    for view, shard_result in result.by_view.items():
        row = json.loads(shard_result.output_path.read_text(encoding="utf-8"))
        meta = json.loads(shard_result.output_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert row["evaluation_view"] == view
        assert row["shared_inference_identity"] == result.shared_inference_identity
        assert meta["evaluation_view"] == view
        assert meta["shared_inference_identity"] == result.shared_inference_identity

    for view, model in (
        (MODEL_CORE_VIEW, "lfm25-pii"),
        (VENDOR_HYBRID_VIEW, "lfm25-pii"),
    ):
        matrix = read_matrix_results(
            tmp_path,
            model,
            ("smoke",),
            require_done=True,
            expected_rows={"smoke": 1},
            expected_evaluation_contract=_contract(view),
            expected_dataset_shard_identities={
                "smoke": dataset_shard_identity(_contract(view), dataset="smoke", shard="full", rows=rows),
            },
            view=view,
        )
        assert set(matrix.rows_by_dataset) == {"smoke"}


def test_hub_safe_export_is_deterministic_and_removes_span_surface_strings(
    tmp_path: Path,
) -> None:
    result = run_dual_view_shards(
        adapter=Lfm25PiiSpaceAdapter(_FakeDualViewDetector()),
        rows=[
            EvalRow(
                doc_id="doc-1",
                dataset="smoke",
                shard="full",
                text="token1234",
                gold_spans=(CharSpan(start=0, end=5, text="token", label="human_name"),),
                language="en",
                slices=frozenset({"smoke"}),
            ),
        ],
        output_root=tmp_path / "internal",
        specs={
            MODEL_CORE_VIEW: ShardSpec("lfm25-pii", "smoke", "full", MODEL_CORE_VIEW),
            VENDOR_HYBRID_VIEW: ShardSpec("lfm25-pii", "smoke", "full", VENDOR_HYBRID_VIEW),
        },
        evaluation_contracts={
            MODEL_CORE_VIEW: _contract(MODEL_CORE_VIEW),
            VENDOR_HYBRID_VIEW: _contract(VENDOR_HYBRID_VIEW),
        },
        volume=_CountingVolume(),
    )
    shard = result.by_view[MODEL_CORE_VIEW]

    first = export_hub_safe_shard_package(
        result_path=shard.output_path,
        metadata_path=shard.output_path.with_suffix(".meta.json"),
        output_dir=tmp_path / "safe",
    )
    second = export_hub_safe_shard_package(
        result_path=shard.output_path,
        metadata_path=shard.output_path.with_suffix(".meta.json"),
        output_dir=tmp_path / "safe-repeat",
    )

    safe_rows = first.rows_path.read_text(encoding="utf-8")
    safe_manifest = first.manifest_path.read_text(encoding="utf-8")
    assert first.sha256 == second.sha256
    assert "token1234" not in safe_rows
    assert "doc-1" not in safe_rows
    assert '"text"' not in safe_rows
    assert '"text"' not in safe_manifest
    assert "dataset_shard_identity" not in safe_manifest
    manifest = json.loads(safe_manifest)
    assert manifest["dataset"] == "smoke"
    assert manifest["shard"] == "full"
    assert manifest["safe_rows_file"] == first.rows_path.name
    assert manifest["safe_rows_sha256"] == file_sha256(str(first.rows_path))
    assert manifest["package_sha256"] == first.sha256
    assert validate_hub_safe_shard_package(first.manifest_path) == first

    first.rows_path.write_bytes(first.rows_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="safe rows digest"):
        validate_hub_safe_shard_package(first.manifest_path)


def test_safe_export_manifest_replace_preserves_previous_generation_on_fault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = run_dual_view_shards(
        adapter=Lfm25PiiSpaceAdapter(_FakeDualViewDetector()),
        rows=[
            EvalRow(
                doc_id="doc-1",
                dataset="smoke",
                shard="full",
                text="token1234",
                gold_spans=(CharSpan(start=0, end=5, text="token", label="human_name"),),
                language="en",
                slices=frozenset({"smoke"}),
            ),
        ],
        output_root=tmp_path / "internal",
        specs={
            MODEL_CORE_VIEW: ShardSpec("lfm25-pii", "smoke", "full", MODEL_CORE_VIEW),
            VENDOR_HYBRID_VIEW: ShardSpec("lfm25-pii", "smoke", "full", VENDOR_HYBRID_VIEW),
        },
        evaluation_contracts={
            MODEL_CORE_VIEW: _contract(MODEL_CORE_VIEW),
            VENDOR_HYBRID_VIEW: _contract(VENDOR_HYBRID_VIEW),
        },
        volume=_CountingVolume(),
    )
    shard = result.by_view[MODEL_CORE_VIEW]
    output_dir = tmp_path / "safe"
    previous = export_hub_safe_shard_package(
        result_path=shard.output_path,
        metadata_path=shard.output_path.with_suffix(".meta.json"),
        output_dir=output_dir,
    )
    previous_manifest = previous.manifest_path.read_bytes()
    previous_rows = previous.rows_path.read_bytes()

    source_row = json.loads(shard.output_path.read_text(encoding="utf-8"))
    source_row["pred_spans"] = []
    shard.output_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    metadata_path = shard.output_path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["result_sha256"] = file_sha256(str(shard.output_path))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    original_replace = artifact_export.os.replace

    def fail_manifest_replace(source: Path, destination: Path) -> None:
        if destination == output_dir / "manifest.json":
            msg = "injected manifest replace failure"
            raise OSError(msg)
        original_replace(source, destination)

    monkeypatch.setattr(artifact_export.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="injected manifest replace failure"):
        export_hub_safe_shard_package(
            result_path=shard.output_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
        )

    assert previous.manifest_path.read_bytes() == previous_manifest
    assert previous.rows_path.read_bytes() == previous_rows
    assert validate_hub_safe_shard_package(previous.manifest_path) == previous


def test_dual_view_resume_rejects_paired_outputs_with_a_stale_shared_identity(
    tmp_path: Path,
) -> None:
    rows = [
        EvalRow(
            doc_id="doc-1",
            dataset="smoke",
            shard="full",
            text="token1234",
            gold_spans=(CharSpan(start=0, end=5, text="token", label="human_name"),),
            language="en",
            slices=frozenset({"smoke"}),
        ),
    ]
    specs = {
        MODEL_CORE_VIEW: ShardSpec("lfm25-pii", "smoke", "full", MODEL_CORE_VIEW),
        VENDOR_HYBRID_VIEW: ShardSpec("lfm25-pii", "smoke", "full", VENDOR_HYBRID_VIEW),
    }
    contracts = {
        MODEL_CORE_VIEW: _contract(MODEL_CORE_VIEW),
        VENDOR_HYBRID_VIEW: _contract(VENDOR_HYBRID_VIEW),
    }
    first = run_dual_view_shards(
        adapter=Lfm25PiiSpaceAdapter(_FakeDualViewDetector()),
        rows=rows,
        output_root=tmp_path,
        specs=specs,
        evaluation_contracts=contracts,
        volume=_CountingVolume(),
    )
    stale_identity = first.shared_inference_identity
    core_path = first.by_view[MODEL_CORE_VIEW].output_path
    core_row = json.loads(core_path.read_text(encoding="utf-8"))
    core_row["pred_spans"] = []
    core_path.write_text(json.dumps(core_row) + "\n", encoding="utf-8")
    core_meta_path = core_path.with_suffix(".meta.json")
    core_meta = json.loads(core_meta_path.read_text(encoding="utf-8"))
    core_meta["result_sha256"] = file_sha256(str(core_path))
    core_meta_path.write_text(json.dumps(core_meta), encoding="utf-8")

    detector = _FakeDualViewDetector()
    rerun = run_dual_view_shards(
        adapter=Lfm25PiiSpaceAdapter(detector),
        rows=rows,
        output_root=tmp_path,
        specs=specs,
        evaluation_contracts=contracts,
        volume=_CountingVolume(),
    )

    assert detector.hd.model_calls == ["token1234"]
    assert rerun.by_view[MODEL_CORE_VIEW].skipped is False
    assert rerun.shared_inference_identity == stale_identity


def test_force_removes_both_done_markers_before_rewriting_either_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        EvalRow(
            doc_id="doc-1",
            dataset="smoke",
            shard="full",
            text="token1234",
            gold_spans=(CharSpan(start=0, end=5, text="token", label="human_name"),),
            language="en",
            slices=frozenset({"smoke"}),
        ),
    ]
    specs = {
        MODEL_CORE_VIEW: ShardSpec("lfm25-pii", "smoke", "full", MODEL_CORE_VIEW),
        VENDOR_HYBRID_VIEW: ShardSpec("lfm25-pii", "smoke", "full", VENDOR_HYBRID_VIEW),
    }
    contracts = {
        MODEL_CORE_VIEW: _contract(MODEL_CORE_VIEW),
        VENDOR_HYBRID_VIEW: _contract(VENDOR_HYBRID_VIEW),
    }
    initial = run_dual_view_shards(
        adapter=Lfm25PiiSpaceAdapter(_FakeDualViewDetector()),
        rows=rows,
        output_root=tmp_path,
        specs=specs,
        evaluation_contracts=contracts,
        volume=_CountingVolume(),
    )
    done_paths = [shard.done_path for shard in initial.by_view.values()]
    original = run_module.write_jsonl_atomically
    observed: list[bool] = []

    def assert_pair_is_unpublished(path: Path, records: Sequence[dict[str, object]]) -> None:
        observed.append(all(not done_path.exists() for done_path in done_paths))
        original(path, records)

    monkeypatch.setattr(run_module, "write_jsonl_atomically", assert_pair_is_unpublished)
    run_dual_view_shards(
        adapter=Lfm25PiiSpaceAdapter(_FakeDualViewDetector()),
        rows=rows,
        output_root=tmp_path,
        specs=specs,
        evaluation_contracts=contracts,
        volume=_CountingVolume(),
        force=True,
    )

    assert observed == [True, True]
