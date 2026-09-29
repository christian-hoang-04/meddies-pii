from __future__ import annotations

import json
from typing import TYPE_CHECKING

import onnx
import pytest
from onnx import TensorProto, helper

from meddies_pii.eval_baseline.pii350_release import q8_head_preserved as q8_module
from meddies_pii.eval_baseline.pii350_release.q8_head_preserved import (
    Q8_BITS,
    Q8_BLOCK_SIZE,
    Q8_IS_SYMMETRIC,
    SourceArtifactIdentity,
    build_r1_preflight,
    export_head_preserved_q8,
    verify_r1_manifest,
    verify_r1_preflight_receipt,
    write_r1_preflight_receipt,
)
from meddies_pii.evaluation.identity import canonical_sha256, file_sha256

if TYPE_CHECKING:
    from pathlib import Path


def _write_graph(
    path: Path,
    *,
    label_count: int = 37,
    extra_head: bool = False,
) -> None:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, label_count])
    backbone_weight = helper.make_tensor("backbone.weight", TensorProto.FLOAT, [64, 64], [0.01] * (64 * 64))
    head_weight = helper.make_tensor(
        "classifier.weight",
        TensorProto.FLOAT,
        [64, label_count],
        [0.01] * (64 * label_count),
    )
    nodes = [
        helper.make_node(
            "MatMul",
            ["input", "backbone.weight"],
            ["hidden"],
            name="/backbone/MatMul",
        ),
        helper.make_node(
            "MatMul",
            ["hidden", "classifier.weight"],
            ["logits"],
            name="/classifier/MatMul",
        ),
    ]
    initializers = [backbone_weight, head_weight]
    if extra_head:
        initializers.append(
            helper.make_tensor(
                "ambiguous.weight",
                TensorProto.FLOAT,
                [64, 37],
                [0.02] * (64 * 37),
            ),
        )
        nodes.insert(
            1,
            helper.make_node(
                "MatMul",
                ["hidden", "ambiguous.weight"],
                ["unused_head"],
                name="/ambiguous/MatMul",
            ),
        )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "synthetic-pii350",
            [input_info],
            [output_info],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.save_model(model, path)


def _source(path: Path) -> SourceArtifactIdentity:
    return SourceArtifactIdentity(
        repository="modal-volume://meddies-pii350-release",
        revision="f" * 64,
        path="meddies-pii-v2-onnx/model.onnx",
        bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def test_r1_quantizes_the_backbone_and_preserves_the_unique_37_output_head(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.onnx"
    output = tmp_path / "model.int8.onnx"
    manifest_path = tmp_path / "model.int8.r1-manifest.json"
    _write_graph(source)

    manifest = export_head_preserved_q8(
        source,
        output,
        manifest_path,
        source=_source(source),
        exporter_commit="a" * 40,
    )

    exported = onnx.load(output, load_external_data=False)
    nodes = {node.name: node.op_type for node in exported.graph.node}
    assert nodes["/backbone/MatMul_Q8"] == "MatMulNBits"
    assert nodes["/classifier/MatMul"] == "MatMul"
    assert manifest["recipe"] == {
        "bits": Q8_BITS,
        "block_size": Q8_BLOCK_SIZE,
        "is_symmetric": Q8_IS_SYMMETRIC,
        "kind": "weight-only",
        "op": "MatMulNBits",
    }
    assert manifest["nodes"]["included"] == ["/backbone/MatMul"]
    assert manifest["nodes"]["excluded"] == ["/classifier/MatMul"]
    assert manifest["compatibility"]["producer_onnxruntime"] == "1.28.0"
    assert manifest["compatibility"]["required_consumer"] == "onnxruntime-web@1.27.0"
    assert manifest["compatibility"]["browser_arm_admitted"] is False
    assert "parity smoke" in manifest["compatibility"]["blocking_gate"]
    assert manifest["classifier_proof"] == {
        "node": "/classifier/MatMul",
        "output_count": 37,
        "output_name": "logits",
        "weight": "classifier.weight",
        "weight_shape": [64, 37],
    }
    assert manifest["determinism"]["rerun_identical"] is True
    assert manifest["output"]["bytes"] == output.stat().st_size
    assert manifest["output"]["sha256"] == file_sha256(output)
    assert verify_r1_manifest(source, output, manifest_path) == manifest


def test_r1_fails_closed_when_the_37_output_head_is_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.onnx"
    _write_graph(source, extra_head=True)

    with pytest.raises(RuntimeError, match="ambiguous 37-output classifier head"):
        export_head_preserved_q8(
            source,
            tmp_path / "out.onnx",
            tmp_path / "manifest.json",
            source=_source(source),
            exporter_commit="a" * 40,
        )


def test_r1_fails_closed_when_the_graph_output_is_not_37_labels(tmp_path: Path) -> None:
    source = tmp_path / "wrong-output.onnx"
    _write_graph(source, label_count=36)

    with pytest.raises(RuntimeError, match="output shape mismatch"):
        export_head_preserved_q8(
            source,
            tmp_path / "out.onnx",
            tmp_path / "manifest.json",
            source=_source(source),
            exporter_commit="a" * 40,
        )


def test_r1_rejects_source_identity_and_manifest_mismatches(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    output = tmp_path / "out.onnx"
    manifest_path = tmp_path / "manifest.json"
    _write_graph(source)
    wrong = _source(source)
    wrong = SourceArtifactIdentity(
        repository=wrong.repository,
        revision=wrong.revision,
        path=wrong.path,
        bytes=wrong.bytes,
        sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="source SHA-256 mismatch"):
        export_head_preserved_q8(
            source,
            output,
            manifest_path,
            source=wrong,
            exporter_commit="a" * 40,
        )

    manifest = export_head_preserved_q8(
        source,
        output,
        manifest_path,
        source=_source(source),
        exporter_commit="a" * 40,
    )
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    body["recipe"]["block_size"] = 64
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest identity mismatch"):
        verify_r1_manifest(source, output, manifest_path)
    assert manifest["recipe"]["block_size"] == 32


def test_r1_rejects_tampered_q8_even_when_self_reported_hashes_are_resealed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.onnx"
    output = tmp_path / "candidate.onnx"
    manifest_path = tmp_path / "candidate.manifest.json"
    _write_graph(source)
    export_head_preserved_q8(
        source,
        output,
        manifest_path,
        source=_source(source),
        exporter_commit="a" * 40,
    )
    output.write_bytes(output.read_bytes() + b"tampered")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["bytes"] = output.stat().st_size
    manifest["output"]["sha256"] = file_sha256(output)
    manifest["determinism"]["rerun_bytes"] = output.stat().st_size
    manifest["determinism"]["rerun_sha256"] = file_sha256(output)
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = canonical_sha256(body)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="deterministic re-quantization differs"):
        verify_r1_manifest(source, output, manifest_path)


def test_r1_keeps_the_previous_artifact_if_the_deterministic_rerun_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.onnx"
    output = tmp_path / "candidate.onnx"
    manifest_path = tmp_path / "candidate.manifest.json"
    _write_graph(source)
    output.write_bytes(b"previous candidate")
    manifest_path.write_bytes(b"previous manifest")
    original_quantize = q8_module._quantize_once
    calls = 0

    def nondeterministic_quantize(
        source_model: onnx.ModelProto,
        output: Path,
        *,
        head_node: str,
    ) -> None:
        nonlocal calls
        calls += 1
        original_quantize(source_model, output, head_node=head_node)
        if calls == 2:
            output.write_bytes(output.read_bytes() + b"changed")

    monkeypatch.setattr(q8_module, "_quantize_once", nondeterministic_quantize)

    with pytest.raises(RuntimeError, match="deterministic re-quantization differs"):
        export_head_preserved_q8(
            source,
            output,
            manifest_path,
            source=_source(source),
            exporter_commit="a" * 40,
        )

    assert output.read_bytes() == b"previous candidate"
    assert manifest_path.read_bytes() == b"previous manifest"


def test_r1_preflight_is_a_non_launching_exact_contract() -> None:
    contract = build_r1_preflight(
        source=SourceArtifactIdentity(
            repository="modal-volume://meddies-pii350-release",
            revision="f" * 64,
            path="meddies-pii-v2-onnx/model.onnx",
            bytes=1_420_000_000,
            sha256="1" * 64,
        ),
        exporter_commit="a" * 40,
        estimated_spend_usd=0.42,
        approval_link="evidence://ha/pending-r1-approval",
    )

    assert contract["launch"] is False
    assert contract["modal"] == {
        "app": "meddies-pii350-release-export",
        "cpu": 8,
        "gpu": None,
        "memory_mib": 32768,
        "timeout_seconds": 5400,
        "workspace_profile": "huyhoang041100",
    }
    assert contract["transfer"] == {
        "browser_cache_bytes": 0,
        "local_filesystem_bytes": 0,
        "remote_source_bytes": 1_420_000_000,
    }
    assert contract["approval"]["required_before_launch"] is True
    assert contract["approval"]["status"] == "pending"
    assert contract["approval"]["link"] == "evidence://ha/pending-r1-approval"
    assert len(contract["receipt_sha256"]) == 64
    assert contract["artifact_home"].startswith("modal-volume://")


def test_r1_preflight_rejects_a_mutable_revision_or_traversing_source_path() -> None:
    with pytest.raises(ValueError, match="immutable"):
        build_r1_preflight(
            source=SourceArtifactIdentity(
                repository="modal-volume://meddies-pii350-release",
                revision="latest",
                path="meddies-pii-v2-onnx/model.onnx",
                bytes=1,
                sha256="1" * 64,
            ),
            exporter_commit="a" * 40,
            estimated_spend_usd=0.01,
            approval_link="evidence://ha/pending-r1-approval",
        )
    with pytest.raises(ValueError, match="non-traversing"):
        build_r1_preflight(
            source=SourceArtifactIdentity(
                repository="modal-volume://meddies-pii350-release",
                revision="f" * 64,
                path="../model.onnx",
                bytes=1,
                sha256="1" * 64,
            ),
            exporter_commit="a" * 40,
            estimated_spend_usd=0.01,
            approval_link="evidence://ha/pending-r1-approval",
        )


def test_r1_export_authority_requires_a_persisted_approved_receipt(
    tmp_path: Path,
) -> None:
    source = SourceArtifactIdentity(
        repository="modal-volume://meddies-pii350-release",
        revision="f" * 64,
        path="meddies-pii-v2-onnx/model.onnx",
        bytes=1_420_000_000,
        sha256="1" * 64,
    )
    approved = build_r1_preflight(
        source=source,
        exporter_commit="a" * 40,
        estimated_spend_usd=0.42,
        approval_link="evidence://ha/r1-approval-20260806",
        approval_status="approved",
    )
    receipt_path = write_r1_preflight_receipt(approved, tmp_path / "preflight")

    assert receipt_path.name == f"{approved['receipt_sha256']}.json"
    assert verify_r1_preflight_receipt(receipt_path, approved["receipt_sha256"]) == approved

    pending = build_r1_preflight(
        source=source,
        exporter_commit="a" * 40,
        estimated_spend_usd=0.42,
        approval_link="evidence://ha/r1-pending-20260806",
        approval_status="pending",
    )
    pending_path = write_r1_preflight_receipt(pending, tmp_path / "preflight")
    with pytest.raises(RuntimeError, match="not approved"):
        verify_r1_preflight_receipt(pending_path, pending["receipt_sha256"])


def test_r1_export_authority_rejects_status_string_and_tampered_receipt(
    tmp_path: Path,
) -> None:
    source = SourceArtifactIdentity(
        repository="modal-volume://meddies-pii350-release",
        revision="f" * 64,
        path="meddies-pii-v2-onnx/model.onnx",
        bytes=1,
        sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="not a status string"):
        build_r1_preflight(
            source=source,
            exporter_commit="a" * 40,
            estimated_spend_usd=0.01,
            approval_link="approved",
            approval_status="approved",
        )

    receipt = build_r1_preflight(
        source=source,
        exporter_commit="a" * 40,
        estimated_spend_usd=0.01,
        approval_link="evidence://ha/r1-approval-20260806",
        approval_status="approved",
    )
    receipt_path = write_r1_preflight_receipt(receipt, tmp_path)
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["modal"]["cpu"] = 16
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt content"):
        verify_r1_preflight_receipt(receipt_path, receipt["receipt_sha256"])
