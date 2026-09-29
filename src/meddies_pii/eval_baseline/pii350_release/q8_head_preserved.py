"""Deterministic q8 weight-only export with the 37-label classifier kept fp32.

This module is deliberately independent of Modal. It turns one already-pinned fp32
ONNX file into the R1 candidate, writes a content-addressed manifest, and repeats the
quantization to prove byte identity. Model acquisition and paid execution remain an
explicit caller responsibility.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: ONNX is needed only by export and verification calls; keeping its three imports inside
# reason: those calls leaves contract and manifest inspection usable without loading the model stack.
import json
import platform
import re
import tempfile
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    from onnx import ModelProto, ValueInfoProto

from meddies_pii.evaluation.identity import (
    canonical_sha256,
    file_sha256,
    is_sha256,
)

R1_SCHEMA = "meddies-pii/pii350-q8-head-preserved/v1"
PREFLIGHT_SCHEMA = "meddies-pii/pii350-r1-preflight/v1"
Q8_BITS = 8
Q8_BLOCK_SIZE = 32
Q8_IS_SYMMETRIC = True
LABEL_COUNT = 37
PRODUCER_ONNXRUNTIME = "1.28.0"
REQUIRED_CONSUMER = "onnxruntime-web@1.27.0"
MAX_PREFLIGHT_RECEIPT_BYTES = 64 * 1024
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40,64}")


def _is_git_commit(value: str) -> bool:
    return GIT_COMMIT_PATTERN.fullmatch(value) is not None


@dataclass(frozen=True)
class SourceArtifactIdentity:
    """Immutable identity of the fp32 ONNX bytes being quantized."""

    repository: str
    revision: str
    path: str
    bytes: int
    sha256: str

    def validate(self) -> None:
        if not self.repository.strip():
            msg = "source repository/URI must not be empty"
            raise ValueError(msg)
        if not _is_git_commit(self.revision):
            msg = "source revision must be an immutable 40- or 64-hex ID"
            raise ValueError(msg)
        relative_path = Path(self.path)
        if not self.path.strip() or relative_path.is_absolute() or ".." in relative_path.parts:
            msg = "source path must be a non-traversing repository-relative path"
            raise ValueError(msg)
        if self.bytes <= 0:
            msg = "source byte count must be positive"
            raise ValueError(msg)
        if not is_sha256(self.sha256):
            msg = "source SHA-256 must be a complete lowercase digest"
            raise ValueError(msg)


def _fixed_dimension(value_info: ValueInfoProto, axis: int) -> int | None:
    dimensions = value_info.type.tensor_type.shape.dim
    if not dimensions:
        return None
    dimension = dimensions[axis]
    return int(dimension.dim_value) if dimension.HasField("dim_value") else None


# reason: one graph-identity proof must jointly establish output shape, unique classifier node, and output reachability.
def _classifier_proof(model: ModelProto) -> dict[str, Any]:  # ruff: ignore[complex-structure]
    label_outputs = [output for output in model.graph.output if _fixed_dimension(output, -1) == LABEL_COUNT]
    if len(label_outputs) != 1:
        observed = {output.name: _fixed_dimension(output, -1) for output in model.graph.output}
        msg = (
            "ONNX output shape mismatch: expected exactly one graph output with "
            f"final dimension {LABEL_COUNT}; observed {observed}"
        )
        raise RuntimeError(msg)
    graph_output = label_outputs[0]
    initializer_shapes = {
        initializer.name: [int(dimension) for dimension in initializer.dims] for initializer in model.graph.initializer
    }
    candidates: list[Any] = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            continue
        weight_shape = initializer_shapes.get(node.input[1])
        if weight_shape and len(weight_shape) == 2 and weight_shape[-1] == LABEL_COUNT:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            candidates.append(node)
    if len(candidates) != 1:
        names = sorted(node.name or "<unnamed>" for node in candidates)
        msg = (
            "unknown or ambiguous 37-output classifier head: expected one MatMul "
            f"with a constant [hidden,{LABEL_COUNT}] RHS; found {names}"
        )
        raise RuntimeError(msg)
    head = candidates[0]
    if not head.name:
        msg = "unknown or ambiguous 37-output classifier head: node is unnamed"
        raise RuntimeError(msg)

    reachable = set(head.output)
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if any(name in reachable for name in node.input):
                for name in node.output:
                    if name not in reachable:
                        reachable.add(name)
                        changed = True
    if graph_output.name not in reachable:
        msg = (
            "unknown or ambiguous 37-output classifier head: selected MatMul does "
            f"not reach graph output {graph_output.name!r}"
        )
        raise RuntimeError(msg)
    weight = head.input[1]
    return {
        "node": head.name,
        "output_count": LABEL_COUNT,
        "output_name": graph_output.name,
        "weight": weight,
        "weight_shape": initializer_shapes[weight],
    }


def _validate_source(path: Path, source: SourceArtifactIdentity) -> None:
    source.validate()
    if not path.is_file():
        msg = f"source ONNX file is absent: {path}"
        raise RuntimeError(msg)
    actual_bytes = path.stat().st_size
    if actual_bytes != source.bytes:
        msg = f"source byte-count mismatch: manifest={source.bytes}, actual={actual_bytes}"
        raise RuntimeError(msg)
    actual_sha = file_sha256(str(path))
    if actual_sha != source.sha256:
        msg = f"source SHA-256 mismatch: manifest={source.sha256}, actual={actual_sha}"
        raise RuntimeError(msg)


def _quantize_once(source_model: ModelProto, output: Path, *, head_node: str) -> None:
    """Quantize the model once while preserving the requested head node.

    ORT does not publish typing metadata. Dynamic loading keeps that untyped
    third-party boundary explicit rather than suppressing analysis for this file.
    """
    import onnx

    quantizer_module = import_module("onnxruntime.quantization.matmul_nbits_quantizer")
    quantizer: Any = quantizer_module.MatMulNBitsQuantizer(
        source_model,
        bits=Q8_BITS,
        block_size=Q8_BLOCK_SIZE,
        is_symmetric=Q8_IS_SYMMETRIC,
        nodes_to_exclude=[head_node],
    )
    quantizer.process()
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        quantizer.model.model,
        output,
        save_as_external_data=False,
    )


def _source_matmuls(model: ModelProto) -> dict[tuple[str, ...], str]:
    return {tuple(node.output): node.name for node in model.graph.node if node.op_type == "MatMul" and node.name}


def _node_manifest(
    source_matmuls: dict[tuple[str, ...], str],
    output_model: ModelProto,
    *,
    head_node: str,
) -> dict[str, list[str]]:
    included = sorted(
        source_matmuls[tuple(node.output)]
        for node in output_model.graph.node
        if node.op_type == "MatMulNBits" and tuple(node.output) in source_matmuls
    )
    if not included:
        msg = "R1 q8 export quantized no eligible backbone MatMul nodes"
        raise RuntimeError(msg)
    preserved = [node for node in output_model.graph.node if node.name == head_node and node.op_type == "MatMul"]
    if len(preserved) != 1:
        msg = f"R1 classifier preservation failed for {head_node!r}: found {len(preserved)} fp32 MatMul nodes"
        raise RuntimeError(msg)
    if head_node in included:
        msg = f"R1 classifier {head_node!r} was included in quantization"
        raise RuntimeError(msg)
    return {"included": included, "excluded": [head_node]}


def _producer_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "onnx": version("onnx"),
        "onnxruntime_quantizer": version("onnxruntime"),
        "onnx_ir": version("onnx-ir"),
    }
    if versions["onnxruntime_quantizer"] != PRODUCER_ONNXRUNTIME:
        msg = (
            f"R1 producer version mismatch: expected {PRODUCER_ONNXRUNTIME}, observed {versions['onnxruntime_quantizer']}"
        )
        raise RuntimeError(msg)
    return versions


def _compatibility_contract() -> dict[str, Any]:
    return {
        "producer_onnxruntime": PRODUCER_ONNXRUNTIME,
        "required_consumer": REQUIRED_CONSUMER,
        "browser_arm_admitted": False,
        "blocking_gate": (
            "onnxruntime-web@1.27.0 WebGPU operator-compatibility and decoded-span "
            "parity smoke on these exact output bytes"
        ),
        "interpretation": ("artifact compatibility risk only; this export is not an ORT version-lane result"),
    }


def _recipe() -> dict[str, Any]:
    return {
        "bits": Q8_BITS,
        "block_size": Q8_BLOCK_SIZE,
        "is_symmetric": Q8_IS_SYMMETRIC,
        "kind": "weight-only",
        "op": "MatMulNBits",
    }


def _model_opset(model: ModelProto) -> int:
    versions = [entry.version for entry in model.opset_import if entry.domain in {"", "ai.onnx"}]
    if len(versions) != 1:
        msg = f"expected exactly one default ONNX opset; found {versions}"
        raise RuntimeError(msg)
    return int(versions[0])


def export_head_preserved_q8(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    source: SourceArtifactIdentity,
    exporter_commit: str,
) -> dict[str, Any]:
    """Create and verify R1 without acquiring or uploading model bytes.

    Both files are built in a same-filesystem staging directory. The manifest is
    replaced last, so an interrupted export can never advertise incomplete bytes.

    Returns:
        The verified R1 release-manifest payload.

    Raises:
        ValueError: If the exporter identity or output paths violate the release contract.
        RuntimeError: If source, graph, quantization, or determinism verification fails.

    """
    if not _is_git_commit(exporter_commit):
        msg = "exporter commit must be a complete lowercase Git SHA"
        raise ValueError(msg)
    _validate_source(source_path, source)
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.resolve() != manifest_path.parent.resolve():
        msg = "R1 output and manifest must share one artifact directory"
        raise ValueError(msg)
    source_model = onnx.load(source_path, load_external_data=True)
    proof = _classifier_proof(source_model)
    source_matmuls = _source_matmuls(source_model)
    opset = _model_opset(source_model)

    with tempfile.TemporaryDirectory(prefix=".pii350-r1-staging-", dir=output_path.parent) as directory:
        staging_root = Path(directory)
        staged_output = staging_root / output_path.name
        staged_manifest = staging_root / manifest_path.name
        _quantize_once(source_model, staged_output, head_node=proof["node"])
        output_model = onnx.load(staged_output, load_external_data=False)
        if _classifier_proof(output_model) != proof:
            msg = "R1 output classifier proof differs from the fp32 source"
            raise RuntimeError(msg)
        nodes = _node_manifest(source_matmuls, output_model, head_node=proof["node"])
        output_sha = file_sha256(str(staged_output))
        output_bytes = staged_output.stat().st_size

        body: dict[str, Any] = {
            "schema": R1_SCHEMA,
            "source": asdict(source),
            "exporter_commit": exporter_commit,
            "producer": _producer_versions(),
            "compatibility": _compatibility_contract(),
            "opset": opset,
            "recipe": _recipe(),
            "nodes": nodes,
            "classifier_proof": proof,
            "output": {
                "path": output_path.name,
                "bytes": output_bytes,
                "sha256": output_sha,
            },
            "determinism": {
                "rerun_bytes": output_bytes,
                "rerun_sha256": output_sha,
                "rerun_identical": True,
                "verified_by": "verify_r1_manifest",
            },
        }
        manifest = {**body, "manifest_digest": canonical_sha256(body)}
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_r1_manifest(source_path, staged_output, staged_manifest)
        staged_output.replace(output_path)
        staged_manifest.replace(manifest_path)
    return manifest


def _load_r1_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        msg = "manifest root must be an object"
        raise TypeError(msg)
    manifest = cast("dict[str, Any]", manifest_value)
    digest = manifest["manifest_digest"]
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if canonical_sha256(body) != digest:
        msg = "manifest digest does not match its body"
        raise ValueError(msg)
    if manifest["schema"] != R1_SCHEMA:
        msg = "schema differs"
        raise ValueError(msg)
    return manifest


def _verify_r1_source_contract(manifest: dict[str, Any], source_path: Path) -> tuple[ModelProto, dict[str, Any]]:
    source = SourceArtifactIdentity(**manifest["source"])
    _validate_source(source_path, source)
    import onnx

    if manifest["recipe"] != _recipe():
        msg = "quantization recipe differs"
        raise ValueError(msg)
    if manifest["producer"] != _producer_versions():
        msg = "producer versions differ"
        raise ValueError(msg)
    if manifest["compatibility"] != _compatibility_contract():
        msg = "runtime compatibility contract differs"
        raise ValueError(msg)
    if not _is_git_commit(manifest["exporter_commit"]):
        msg = "exporter commit is invalid"
        raise ValueError(msg)
    source_model = onnx.load(source_path, load_external_data=True)
    proof = _classifier_proof(source_model)
    if manifest["classifier_proof"] != proof:
        msg = "classifier proof differs"
        raise ValueError(msg)
    if manifest["opset"] != _model_opset(source_model):
        msg = "opset differs"
        raise ValueError(msg)
    return source_model, proof


def _verify_r1_output_contract(
    manifest: dict[str, Any],
    source_path: Path,
    output_path: Path,
    source_model: ModelProto,
    proof: dict[str, Any],
) -> None:
    import onnx

    if manifest["output"]["path"] != output_path.name:
        msg = "output path differs"
        raise ValueError(msg)
    if output_path.stat().st_size != manifest["output"]["bytes"]:
        msg = "output byte count differs"
        raise ValueError(msg)
    if file_sha256(str(output_path)) != manifest["output"]["sha256"]:
        msg = "output SHA-256 differs"
        raise ValueError(msg)
    with tempfile.TemporaryDirectory(prefix="pii350-r1-verify-") as directory:
        rerun_path = Path(directory) / output_path.name
        _quantize_once(
            onnx.load(source_path, load_external_data=True),
            rerun_path,
            head_node=proof["node"],
        )
        rerun_bytes = rerun_path.stat().st_size
        rerun_sha = file_sha256(str(rerun_path))
    if rerun_bytes != manifest["output"]["bytes"] or rerun_sha != manifest["output"]["sha256"]:
        msg = (
            "deterministic re-quantization differs from declared output: "
            f"declared={manifest['output']['bytes']}/{manifest['output']['sha256']}, "
            f"rerun={rerun_bytes}/{rerun_sha}"
        )
        raise ValueError(msg)
    output_model = onnx.load(output_path, load_external_data=False)
    if _classifier_proof(output_model) != proof:
        msg = "output classifier proof differs"
        raise ValueError(msg)
    nodes = _node_manifest(_source_matmuls(source_model), output_model, head_node=proof["node"])
    if manifest["nodes"] != nodes:
        msg = "included/excluded node names differ"
        raise ValueError(msg)
    determinism = manifest["determinism"]
    if determinism != {
        "rerun_bytes": manifest["output"]["bytes"],
        "rerun_sha256": manifest["output"]["sha256"],
        "rerun_identical": True,
        "verified_by": "verify_r1_manifest",
    }:
        msg = "deterministic rerun identity differs"
        raise ValueError(msg)


def verify_r1_manifest(source_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Fail closed unless source, recipe, graph selection, and output all agree.

    Returns:
        The verified manifest payload.

    Raises:
        RuntimeError: If any manifest, source, graph, output, or determinism identity check fails.

    """
    try:
        manifest = _load_r1_manifest(manifest_path)
        source_model, proof = _verify_r1_source_contract(manifest, source_path)
        _verify_r1_output_contract(manifest, source_path, output_path, source_model, proof)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        msg = f"R1 manifest identity mismatch: {error}"
        raise RuntimeError(msg) from error
    return manifest


def build_r1_preflight(
    *,
    source: SourceArtifactIdentity,
    exporter_commit: str,
    estimated_spend_usd: float,
    approval_link: str,
    approval_status: str = "pending",
) -> dict[str, Any]:
    """Seal the exact non-launching contract that must precede a Modal run.

    Returns:
        The sealed non-launching preflight payload and its content digest.

    Raises:
        ValueError: If the source, exporter, spend, approval link, or approval status is invalid.

    """
    source.validate()
    if not _is_git_commit(exporter_commit):
        msg = "exporter commit must be a complete lowercase Git SHA"
        raise ValueError(msg)
    if estimated_spend_usd < 0:
        msg = "estimated spend must not be negative"
        raise ValueError(msg)
    parsed_link = urlparse(approval_link)
    if (
        approval_link.strip().lower() in {"approved", "pending", "none"}
        or parsed_link.scheme not in {"evidence", "http", "https"}
        or not (parsed_link.netloc or parsed_link.path)
    ):
        msg = "approval link must be an http(s) or evidence URI, not a status string"
        raise ValueError(msg)
    if approval_status not in {"pending", "approved"}:
        msg = "approval status must be pending or approved"
        raise ValueError(msg)
    body: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "launch": False,
        "candidate": "r1_q8_head_preserved",
        "source": asdict(source),
        "exporter_commit": exporter_commit,
        "recipe": _recipe(),
        "compatibility": _compatibility_contract(),
        "modal": {
            "app": "meddies-pii350-release-export",
            "workspace_profile": "huyhoang041100",
            "gpu": None,
            "cpu": 8,
            "memory_mib": 32_768,
            "timeout_seconds": 90 * 60,
        },
        "estimated_spend_usd": estimated_spend_usd,
        "transfer": {
            "remote_source_bytes": source.bytes,
            "local_filesystem_bytes": 0,
            "browser_cache_bytes": 0,
        },
        "artifact_home": ("modal-volume://meddies-pii350-release/meddies-pii-v2-onnx/model.int8.r1-head-preserved.onnx"),
        "manifest_home": ("modal-volume://meddies-pii350-release/meddies-pii-v2-onnx/model.int8.r1-manifest.json"),
        "approval": {
            "required_before_launch": True,
            "link": approval_link,
            "status": approval_status,
            "requirements": [
                "approve the exact Modal resource configuration and spend estimate",
                "approve access to the digest-pinned remote fp32 source bytes",
                "approve the immutable remote artifact and manifest homes",
            ],
        },
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def write_r1_preflight_receipt(receipt: dict[str, Any], directory: Path) -> Path:
    """Persist a receipt under its content digest without launching Modal.

    Returns:
        The path of the persisted content-addressed receipt.

    Raises:
        ValueError: If the receipt digest is absent or does not match its body.
        RuntimeError: If an existing content-addressed receipt has different bytes.

    """
    digest = receipt.get("receipt_sha256")
    if not is_sha256(digest):
        msg = "preflight receipt is missing a complete SHA-256"
        raise ValueError(msg)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if canonical_sha256(body) != digest:
        msg = "preflight receipt SHA-256 does not match its body"
        raise ValueError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.json"
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != payload:
            msg = f"content-addressed preflight collision: {destination}"
            raise RuntimeError(msg)
        return destination
    staging = directory / f".{digest}.tmp"
    staging.write_text(payload, encoding="utf-8")
    staging.replace(destination)
    return destination


def verify_r1_preflight_receipt(
    receipt_path: Path,
    expected_sha256: str,
    *,
    require_approved: bool = True,
) -> dict[str, Any]:
    """Verify the persisted execution authority before any artifact write.

    Returns:
        The verified preflight receipt payload.

    Raises:
        RuntimeError: If the expected digest, receipt path, receipt bytes, or approval state is invalid.

    """
    if not is_sha256(expected_sha256):
        msg = "expected preflight receipt SHA-256 is invalid"
        raise RuntimeError(msg)
    if receipt_path.name != f"{expected_sha256}.json":
        msg = "preflight receipt path is not content-addressed by expected SHA"
        raise RuntimeError(msg)
    try:
        payload = receipt_path.read_bytes()
    except OSError as error:
        msg = f"R1 preflight receipt mismatch: {error}"
        raise RuntimeError(msg) from error
    return verify_r1_preflight_receipt_bytes(
        payload,
        expected_sha256,
        require_approved=require_approved,
    )


def _validated_r1_preflight_receipt(
    payload: bytes,
    expected_sha256: str,
    *,
    require_approved: bool,
) -> dict[str, Any]:
    value: object = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        msg = "receipt root must be an object"
        raise TypeError(msg)
    receipt = cast("dict[str, Any]", value)
    digest = receipt["receipt_sha256"]
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if digest != expected_sha256 or canonical_sha256(body) != expected_sha256:
        msg = "receipt content does not match expected SHA-256"
        raise ValueError(msg)
    source = SourceArtifactIdentity(**receipt["source"])
    expected = build_r1_preflight(
        source=source,
        exporter_commit=receipt["exporter_commit"],
        estimated_spend_usd=receipt["estimated_spend_usd"],
        approval_link=receipt["approval"]["link"],
        approval_status=receipt["approval"]["status"],
    )
    if receipt != expected:
        msg = "receipt differs from the exact R1 preflight contract"
        raise ValueError(msg)
    if require_approved and receipt["approval"]["status"] != "approved":
        msg = "preflight receipt is not approved"
        raise ValueError(msg)
    return receipt


def verify_r1_preflight_receipt_bytes(
    payload: bytes,
    expected_sha256: str,
    *,
    require_approved: bool = True,
) -> dict[str, Any]:
    """Verify received receipt bytes before they are allowed onto the volume.

    Returns:
        The verified preflight receipt payload.

    Raises:
        RuntimeError: If the receipt digest, size, content, identity, or approval is invalid.

    """
    if not is_sha256(expected_sha256):
        msg = "expected preflight receipt SHA-256 is invalid"
        raise RuntimeError(msg)
    if not payload or len(payload) > MAX_PREFLIGHT_RECEIPT_BYTES:
        msg = f"R1 preflight receipt mismatch: receipt byte count must be in 1..{MAX_PREFLIGHT_RECEIPT_BYTES}"
        raise RuntimeError(msg)
    try:
        return _validated_r1_preflight_receipt(
            payload,
            expected_sha256,
            require_approved=require_approved,
        )
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        msg = f"R1 preflight receipt mismatch: {error}"
        raise RuntimeError(msg) from error
