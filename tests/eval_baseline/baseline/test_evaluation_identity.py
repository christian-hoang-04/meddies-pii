from __future__ import annotations

import builtins
import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from meddies_pii.eval_baseline.baseline import datasets as datasets_module
from meddies_pii.eval_baseline.baseline import run as run_module
from meddies_pii.eval_baseline.baseline.aggregate import AggregateReport, aggregate_results
from meddies_pii.eval_baseline.baseline.datasets import EvalRow
from meddies_pii.eval_baseline.baseline.run import (
    ShardSpec,
    assert_frozen_fixture,
    run_shard,
)
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    DatasetShardIdentity,
    EvaluationContract,
    FixtureRowIdentity,
    canonical_json_bytes,
    canonical_sha256,
    dataset_shard_identity,
    file_sha256,
)
from meddies_pii.jsonl import read_jsonl
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType
    from typing import IO, Self

    from meddies_pii.taxonomy import PiiLabel


def _digest(character: str) -> str:
    return character * 64


def _artifact(name: str, character: str) -> ArtifactIdentity:
    return ArtifactIdentity(name, "f" * 40, _digest(character))


def _contract() -> EvaluationContract:
    return EvaluationContract(
        model=_artifact("hf://model", "a"),
        vendor_inference_source=_artifact("git://vendor", "b"),
        local_adapter_source=_artifact("src/adapter.py", "c"),
        applied_label_prediction_contract=_artifact("labels", "d"),
        decoder_contract=_artifact("decoder", "e"),
        resolved_runtime_environment=_artifact("runtime", "f"),
        scorer_contract=_artifact("scorer", "0"),
        supported_labels=("human_name",),
        result_schema=("id", "doc_id", "pred_spans", "gold_spans", "language", "slice"),
    )


def _fixture_row() -> FixtureRowIdentity:
    return FixtureRowIdentity(
        stable_id="v2-eval:en:row-1",
        text_sha256=_digest("1"),
        gold_spans=(CharSpan(0, 4, "John", "human_name"),),
        language="en",
        slices=("smoke",),
    )


def test_frozen_fixture_anchors_modal_canonical_digest() -> None:
    """Reject the legacy pre-separation fixture digest.

    Modal ap-nugnbHR8de7ShjETk4r12g: 263,785 rows and 1,601,262 full-nine spans.

    """
    observed = "1ccd86a42833d45ce20c9d222e8891c6cd334e07ca774ba5ab64debfd5378cb5"
    legacy = "32708bdf4bb083bae31a8487b438d9bc60a8782c16750d9b31fdd386318a5aeb"

    assert observed == datasets_module.EVAL_FIXTURE_SHA256
    assert observed == run_module.EVAL_FIXTURE_SHA256
    assert_frozen_fixture({"fixture": {"sha256": observed}})
    with pytest.raises(ValueError, match="frozen fixture hash changed"):
        assert_frozen_fixture({"fixture": {"sha256": legacy}})


def test_artifact_identity_requires_immutable_reference_revision_and_digest() -> None:
    with pytest.raises(ValueError, match="reference"):
        ArtifactIdentity("", "revision", _digest("a"))
    with pytest.raises(ValueError, match="revision"):
        ArtifactIdentity("model", "", _digest("a"))
    with pytest.raises(ValueError, match="immutable"):
        ArtifactIdentity("model", "main", _digest("a"))
    with pytest.raises(ValueError, match="64-hex"):
        ArtifactIdentity("model", "f" * 40, "a" * 63)


def test_canonical_json_is_deterministic_but_fixture_order_is_preserved() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes({"a": 1, "b": 2})
    first = DatasetShardIdentity(_contract(), "v2-eval", "full", (_fixture_row(),), 1)
    second = DatasetShardIdentity(
        _contract(),
        "v2-eval",
        "full",
        (_fixture_row(), replace(_fixture_row(), stable_id="v2-eval:en:row-2")),
        2,
    )
    assert first.digest != second.digest


def test_fixture_identity_is_stable_across_evaluation_contracts() -> None:
    fixture = (_fixture_row(),)
    first = DatasetShardIdentity(_contract(), "v2-eval", "full", fixture, 1)
    second = DatasetShardIdentity(
        replace(_contract(), model=replace(_contract().model, sha256=_digest("2"))),
        "v2-eval",
        "full",
        fixture,
        1,
    )

    assert first.digest != second.digest
    assert first.fixture_identity.digest == second.fixture_identity.digest


def test_file_sha256_streams_in_bounded_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"a" * (2 * 1024 * 1024 + 17)
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(payload)
    original_open = builtins.open
    requested_sizes: list[int] = []

    class TrackingReader:
        def __init__(self, source: IO[bytes]) -> None:
            self.source = source

        def __enter__(self) -> Self:
            self.source.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            self.source.__exit__(exc_type, exc, traceback)

        def read(self, size: int = -1) -> bytes:
            requested_sizes.append(size)
            return self.source.read(size)

        def fileno(self) -> int:
            return self.source.fileno()

    def tracking_open(file: str, mode: str = "r") -> TrackingReader:
        return TrackingReader(original_open(file, mode))

    monkeypatch.setattr(builtins, "open", tracking_open)

    assert file_sha256(str(path)) == hashlib.sha256(payload).hexdigest()
    assert requested_sizes == [1024 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 1024]


class _Volume:
    def commit(self) -> None:
        pass


class _Adapter:
    name = "identity-test"
    supported_labels: frozenset[PiiLabel] = frozenset({"human_name"})

    def __init__(self) -> None:
        self.predict_calls = 0

    def load(self) -> None:
        pass

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        self.predict_calls += 1
        return [[] for _ in texts]


def _eval_row(
    *,
    doc_id: str = "row-1",
    text: str = "John",
    language: str = "en",
    slices: frozenset[str] = frozenset({"smoke"}),
) -> EvalRow:
    return EvalRow(
        doc_id=doc_id,
        dataset="v2-eval",
        shard="full",
        text=text,
        gold_spans=(CharSpan(0, 4, text[:4], "human_name"),),
        language=language,
        slices=slices,
    )


@pytest.mark.parametrize(
    "changed_rows",
    [
        [_eval_row(doc_id="row-2")],
        [_eval_row(text="Jane")],
        [replace(_eval_row(), gold_spans=(CharSpan(0, 2, "Jo", "human_name"),))],
        [_eval_row(language="vi")],
        [_eval_row(slices=frozenset({"challenge"}))],
        [_eval_row(doc_id="row-2"), _eval_row(doc_id="row-1")],
    ],
    ids=("row-id", "text", "gold", "language", "slices", "order"),
)
def test_fixture_identity_changes_when_scored_fixture_changes(
    changed_rows: list[EvalRow],
) -> None:
    original_rows = [_eval_row(), _eval_row(doc_id="row-2")] if len(changed_rows) == 2 else [_eval_row()]

    original = dataset_shard_identity(_contract(), dataset="v2-eval", shard="full", rows=original_rows)
    changed = dataset_shard_identity(_contract(), dataset="v2-eval", shard="full", rows=changed_rows)

    assert original.fixture_identity.digest != changed.fixture_identity.digest


def test_frozen_fixture_check_accepts_one_fixture_across_model_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_eval_row()]
    first_contract = _contract()
    second_contract = replace(_contract(), model=replace(_contract().model, sha256=_digest("2")))

    def report_for(contract: EvaluationContract, name: str) -> tuple[AggregateReport, DatasetShardIdentity]:
        result = run_shard(
            adapter=_Adapter(),
            rows=rows,
            output_root=tmp_path,
            spec=ShardSpec(name, "v2-eval", "full"),
            evaluation_contract=contract,
            volume=_Volume(),
        )
        identity = dataset_shard_identity(contract, dataset="v2-eval", shard="full", rows=rows)
        return (
            aggregate_results(
                {"v2-eval": list(read_jsonl(result.output_path))},
                supported_labels=frozenset({"human_name"}),
                expected_evaluation_contract=contract,
                expected_dataset_shard_identities={"v2-eval": identity},
                result_sha256_by_config={"v2-eval": file_sha256(str(result.output_path))},
            ),
            identity,
        )

    first_report, first_identity = report_for(first_contract, "first")
    second_report, second_identity = report_for(second_contract, "second")

    assert first_identity.digest != second_identity.digest
    assert first_report["fixture"]["sha256"] == second_report["fixture"]["sha256"]
    monkeypatch.setattr(run_module, "EVAL_FIXTURE_SHA256", first_report["fixture"]["sha256"])
    assert_frozen_fixture(first_report)
    assert_frozen_fixture(second_report)


@pytest.mark.parametrize(
    "changed_contract",
    [
        lambda value: replace(value, model=replace(value.model, sha256=_digest("2"))),
        lambda value: replace(
            value,
            vendor_inference_source=replace(value.vendor_inference_source, sha256=_digest("3")),
        ),
        lambda value: replace(
            value,
            local_adapter_source=replace(value.local_adapter_source, sha256=_digest("4")),
        ),
        lambda value: replace(
            value,
            applied_label_prediction_contract=replace(value.applied_label_prediction_contract, sha256=_digest("5")),
        ),
        lambda value: replace(
            value,
            decoder_contract=replace(value.decoder_contract, sha256=_digest("6")),
        ),
        lambda value: replace(
            value,
            resolved_runtime_environment=replace(value.resolved_runtime_environment, sha256=_digest("7")),
        ),
        lambda value: replace(
            value,
            scorer_contract=replace(value.scorer_contract, sha256=_digest("8")),
        ),
        lambda value: replace(value, supported_labels=("address",)),
        lambda value: replace(
            value,
            result_schema=(
                "id",
                "doc_id",
                "pred_spans",
                "gold_spans",
                "slice",
                "language",
            ),
        ),
    ],
    ids=(
        "model",
        "vendor-source",
        "local-adapter",
        "label-prediction",
        "decoder",
        "runtime",
        "scorer",
        "supported-labels",
        "result-schema",
    ),
)
def test_run_shard_refuses_resume_when_any_contract_component_changes(
    tmp_path: Path,
    changed_contract: Callable[[EvaluationContract], EvaluationContract],
) -> None:
    rows = [_eval_row()]
    spec = ShardSpec("identity-test", "v2-eval", "full")
    adapter = _Adapter()
    run_shard(
        adapter=adapter,
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=_contract(),
        volume=_Volume(),
    )

    result = run_shard(
        adapter=adapter,
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=changed_contract(_contract()),
        volume=_Volume(),
    )

    assert result.skipped is False
    assert adapter.predict_calls == 2


@pytest.mark.parametrize(
    ("initial_rows", "changed_rows", "changed_spec"),
    [
        (
            [_eval_row()],
            [_eval_row(doc_id="row-2")],
            ShardSpec("identity-test", "v2-eval", "full"),
        ),
        (
            [_eval_row()],
            [_eval_row(text="Jane")],
            ShardSpec("identity-test", "v2-eval", "full"),
        ),
        (
            [_eval_row()],
            [
                replace(
                    _eval_row(),
                    gold_spans=(CharSpan(0, 2, "Jo", "human_name"),),
                ),
            ],
            ShardSpec("identity-test", "v2-eval", "full"),
        ),
        (
            [_eval_row()],
            [_eval_row(language="vi")],
            ShardSpec("identity-test", "v2-eval", "full"),
        ),
        (
            [_eval_row()],
            [_eval_row(slices=frozenset({"challenge"}))],
            ShardSpec("identity-test", "v2-eval", "full"),
        ),
        (
            [_eval_row(doc_id="row-1"), _eval_row(doc_id="row-2")],
            [_eval_row(doc_id="row-2"), _eval_row(doc_id="row-1")],
            ShardSpec("identity-test", "v2-eval", "full"),
        ),
        (
            [_eval_row()],
            [_eval_row()],
            ShardSpec("identity-test", "other-dataset", "full"),
        ),
        (
            [_eval_row()],
            [_eval_row()],
            ShardSpec("identity-test", "v2-eval", "other-shard"),
        ),
        (
            [_eval_row()],
            [_eval_row(), _eval_row(doc_id="row-2")],
            ShardSpec("identity-test", "v2-eval", "full"),
        ),
    ],
    ids=(
        "fixture-row-id",
        "text-content-and-hash",
        "gold-spans",
        "language",
        "slices",
        "row-order",
        "dataset",
        "shard",
        "row-count",
    ),
)
def test_run_shard_refuses_resume_when_any_dataset_shard_component_changes(
    tmp_path: Path,
    initial_rows: list[EvalRow],
    changed_rows: list[EvalRow],
    changed_spec: ShardSpec,
) -> None:
    adapter = _Adapter()
    run_shard(
        adapter=adapter,
        rows=initial_rows,
        output_root=tmp_path,
        spec=ShardSpec("identity-test", "v2-eval", "full"),
        evaluation_contract=_contract(),
        volume=_Volume(),
    )

    result = run_shard(
        adapter=adapter,
        rows=changed_rows,
        output_root=tmp_path,
        spec=changed_spec,
        evaluation_contract=_contract(),
        volume=_Volume(),
    )

    assert result.skipped is False
    assert adapter.predict_calls == 2


def test_aggregate_requires_the_exact_contract_and_dataset_identity(
    tmp_path: Path,
) -> None:
    rows = [_eval_row()]
    contract = _contract()
    spec = ShardSpec("identity-test", "v2-eval", "full")
    result = run_shard(
        adapter=_Adapter(),
        rows=rows,
        output_root=tmp_path,
        spec=spec,
        evaluation_contract=contract,
        volume=_Volume(),
    )
    records = list(read_jsonl(result.output_path))
    identity = dataset_shard_identity(contract, dataset="v2-eval", shard="full", rows=rows)
    report = aggregate_results(
        {"v2-eval": records},
        supported_labels=frozenset({"human_name"}),
        expected_evaluation_contract=contract,
        expected_dataset_shard_identities={"v2-eval": identity},
        result_sha256_by_config={"v2-eval": file_sha256(str(result.output_path))},
    )
    assert report["overall"]["rows"] == 1
    assert report["fixture"]["sha256"] == canonical_sha256({
        "dataset_shard_fixture_identities": {"v2-eval": identity.fixture_identity.to_payload()},
    })
    with pytest.raises(ValueError, match="mixed evaluation contracts"):
        aggregate_results(
            {"v2-eval": records},
            supported_labels=frozenset({"human_name"}),
            expected_evaluation_contract=replace(contract, model=replace(contract.model, sha256=_digest("9"))),
            expected_dataset_shard_identities={"v2-eval": identity},
            result_sha256_by_config={"v2-eval": file_sha256(str(result.output_path))},
        )
    with pytest.raises(ValueError, match="fixture does not match"):
        aggregate_results(
            {"v2-eval": records},
            supported_labels=frozenset({"human_name"}),
            expected_evaluation_contract=contract,
            expected_dataset_shard_identities={
                "v2-eval": dataset_shard_identity(
                    contract,
                    dataset="v2-eval",
                    shard="full",
                    rows=[_eval_row(doc_id="other")],
                ),
            },
            result_sha256_by_config={"v2-eval": file_sha256(str(result.output_path))},
        )
