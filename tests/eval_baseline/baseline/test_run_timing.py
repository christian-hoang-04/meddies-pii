"""No meta files -> empty, not an error (a model evaluated before timing existed still aggregates.

It just has no rows/s).

"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import meddies_pii.eval_baseline.baseline.run as run_module
from meddies_pii.eval_baseline.baseline.run import (
    ShardSpec,
    read_shard_timing,
    resolved_environment_fingerprint,
    shard_meta_path,
    write_shard_meta,
)
from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    DatasetShardIdentity,
    EvaluationContract,
    canonical_json_bytes,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _contract() -> EvaluationContract:
    def artifact(reference: str, digest: str) -> ArtifactIdentity:
        return ArtifactIdentity(reference, "f" * 40, digest * 64)

    return EvaluationContract(
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


def test_write_and_read_shard_timing_roundtrips(tmp_path: Path) -> None:
    """Only the written cell appears; rows are surfaced as float for rows/s math."""
    spec = ShardSpec(model="opf", dataset="v2-eval", shard="full")
    meta_path = shard_meta_path(tmp_path, spec)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_sha256 = hashlib.sha256(b"fixture").hexdigest()
    environment_fingerprint = {
        "schema_version": 1,
        "distributions": ["torch==2.13.0"],
        "sha256": hashlib.sha256(b"torch==2.13.0").hexdigest(),
    }
    contract = _contract()
    identity = DatasetShardIdentity(contract, "v2-eval", "full", (), 0)
    write_shard_meta(
        meta_path,
        elapsed_seconds=12.345,
        rows_in=100,
        spans_out=400,
        fixture_sha256=fixture_sha256,
        result_sha256=hashlib.sha256(b"result").hexdigest(),
        environment_fingerprint=environment_fingerprint,
        evaluation_contract=contract,
        dataset_shard_identity=identity,
    )

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload == json.loads(
        canonical_json_bytes({
            "elapsed_seconds": 12.345,
            "rows_in": 100,
            "spans_out": 400,
            "fixture_sha256": fixture_sha256,
            "result_sha256": hashlib.sha256(b"result").hexdigest(),
            "environment_fingerprint": environment_fingerprint,
            "evaluation_contract": contract.to_payload(),
            "evaluation_contract_sha256": contract.digest,
            "dataset_shard_identity": identity.to_payload(),
            "dataset_shard_identity_sha256": identity.digest,
        }),
    )

    timing = read_shard_timing(tmp_path, "opf", ["v2-eval", "missing-cell"])
    assert timing == {"v2-eval": {"elapsed_seconds": 12.345, "rows": 100.0}}


def test_read_shard_timing_skips_configs_without_meta(tmp_path: Path) -> None:
    assert read_shard_timing(tmp_path, "gliner2", ["v2-eval", "creddata_en"]) == {}


def test_resolved_environment_fingerprint_covers_every_identity_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Distribution:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    discovered = [
        Distribution("torch", "2.13.0"),
        Distribution("transformers", "5.11.0"),
    ]
    monkeypatch.setattr(
        run_module.importlib.metadata,
        "distributions",
        lambda: discovered,
    )
    monkeypatch.setattr(run_module.sys, "version", "Python 3.11.0")
    monkeypatch.setattr(run_module.platform, "platform", lambda: "Linux-a")

    baseline = resolved_environment_fingerprint()
    identity = {key: value for key, value in baseline.items() if key != "sha256"}
    expected = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    ).hexdigest()
    assert baseline["sha256"] == expected

    monkeypatch.setattr(run_module.sys, "version", "Python 3.12.0")
    assert resolved_environment_fingerprint()["sha256"] != baseline["sha256"]

    monkeypatch.setattr(run_module.sys, "version", "Python 3.11.0")
    monkeypatch.setattr(run_module.platform, "platform", lambda: "Linux-b")
    assert resolved_environment_fingerprint()["sha256"] != baseline["sha256"]

    monkeypatch.setattr(run_module.platform, "platform", lambda: "Linux-a")
    discovered[0] = Distribution("torch", "2.13.1")
    changed_distribution_digest = resolved_environment_fingerprint()["sha256"]
    assert changed_distribution_digest != baseline["sha256"]

    discovered.reverse()
    assert resolved_environment_fingerprint()["sha256"] == changed_distribution_digest
