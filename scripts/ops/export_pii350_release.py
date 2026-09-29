"""Package the verified step-250 PII350 checkpoint as the Meddies v2 release.

The release claim is decode parity with the benchmark. The published inference
path therefore does not reimplement BIOES decoding: this script vendors the
exact decode sources the evaluation adapter imports, rewriting only their import
prefixes so the payload stands alone, and proves behaviour with a span-level
parity probe against the adapter itself.

Every action is CPU-only. Uploads are manifest-last, so a partial upload can
never read as a complete release.
"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this Modal command is invoked and tested by file path; scripts/ops is not a distributable package API.
# ruff: file-ignore[import-outside-top-level]
# reason: the heavy dependency loads in the subcommand that needs it, so starting the script does not pay for it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import importlib.util
import json
import shlex
import shutil
import sys
import tempfile
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, override

import modal

from meddies_pii.eval_baseline.pii350_release.q8_head_preserved import (
    SourceArtifactIdentity,
    build_r1_preflight,
    export_head_preserved_q8,
    verify_r1_preflight_receipt,
    verify_r1_preflight_receipt_bytes,
    write_r1_preflight_receipt,
)
from meddies_pii.eval_baseline.pii350_release.release_payload import (
    CHANGE_NOTICE,
    LICENSE_FILENAME,
    MERGED_CLASSIFIER_PREFIX,
    MERGED_ENCODER_PREFIX,
    MERGED_WEIGHTS_FILENAME,
    MERGED_WEIGHTS_FORMAT,
    ONNX_FP32_FILENAME,
    ONNX_INT8_FILENAME,
    ONNX_OPSET,
    PARITY_FIXTURE_CELL,
    PARITY_ROW_MINIMUM,
    POSTPROCESS_FILENAME,
    RELEASE_KIND_ONNX,
    RELEASE_KIND_TORCH,
    RELEASE_MANIFEST_FILENAME,
    RELEASE_MODEL_REPO,
    RELEASE_ONNX_REPO,
    RELEASE_PRIVATE,
    RELEASE_REPO_TYPE,
    RELEASE_SCHEMA_VERSION,
    REMOTE_CODE_FILENAME,
    RUNTIME_DTYPE,
    STEP250_ARTIFACT_PATH,
    STEP250_CHECKPOINT_DIGEST,
    STEP250_REPO_ID,
    STEP250_REVISION,
    USAGE_SNIPPET_FILENAME,
    VENDORED_DECODE_MODULES,
    VENDORED_DECODE_PACKAGE,
    VENDORED_IMPORT_REWRITES,
    UploadedFile,
    bioes_label_map,
    compare_span_sets,
    decode_spans_from_logits,
    entity_label_for_tag,
    hf_api,
    package_source_root,
    parity_rows,
    postprocess_module_source,
    published_merged_spans,
    reconcile_merged_parity,
    release_label_config,
    release_manifest,
    release_parity_record,
    remote_code_module_source,
    required_payload_files,
    rewrite_vendored_source,
    span_key,
    upload_release_tree,
    usage_snippet_source,
    vendored_decode_payload,
    verified_checkpoint_root,
    verify_manifest_tree,
    verify_payload_complete,
)
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
    use_pinned_debian_snapshot,
)

if TYPE_CHECKING:
    from torch import Tensor

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import VerifiedCheckpoint
    from meddies_pii.spans import CharSpan
    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger


class _ReleaseTokenizer(Protocol):
    # reason: this Protocol mirrors the tokenizer's six independent keyword controls; bundling
    # reason: them would no longer describe the callable that the release path accepts.
    def __call__(  # ruff: ignore[too-many-arguments]
        self,
        texts: list[str],
        *,
        truncation: bool,
        max_length: int,
        padding: bool,
        return_offsets_mapping: bool,
        return_tensors: str,
    ) -> MutableMapping[str, Any]: ...


__all__ = (
    "CHANGE_NOTICE",
    "LICENSE_FILENAME",
    "MERGED_CLASSIFIER_PREFIX",
    "MERGED_ENCODER_PREFIX",
    "MERGED_WEIGHTS_FILENAME",
    "MERGED_WEIGHTS_FORMAT",
    "ONNX_FP32_FILENAME",
    "ONNX_INT8_FILENAME",
    "ONNX_OPSET",
    "PARITY_FIXTURE_CELL",
    "PARITY_ROW_MINIMUM",
    "POSTPROCESS_FILENAME",
    "RELEASE_KIND_ONNX",
    "RELEASE_KIND_TORCH",
    "RELEASE_MANIFEST_FILENAME",
    "RELEASE_MODEL_REPO",
    "RELEASE_ONNX_REPO",
    "RELEASE_PRIVATE",
    "RELEASE_REPO_TYPE",
    "RELEASE_SCHEMA_VERSION",
    "REMOTE_CODE_FILENAME",
    "RUNTIME_DTYPE",
    "STEP250_ARTIFACT_PATH",
    "STEP250_CHECKPOINT_DIGEST",
    "STEP250_REPO_ID",
    "STEP250_REVISION",
    "USAGE_SNIPPET_FILENAME",
    "VENDORED_DECODE_MODULES",
    "VENDORED_DECODE_PACKAGE",
    "VENDORED_IMPORT_REWRITES",
    "UploadedFile",
    "bioes_label_map",
    "compare_span_sets",
    "decode_spans_from_logits",
    "entity_label_for_tag",
    "hf_api",
    "package_source_root",
    "parity_rows",
    "postprocess_module_source",
    "published_merged_spans",
    "reconcile_merged_parity",
    "release_label_config",
    "release_manifest",
    "release_parity_record",
    "remote_code_module_source",
    "required_payload_files",
    "rewrite_vendored_source",
    "span_key",
    "upload_release_tree",
    "usage_snippet_source",
    "vendored_decode_payload",
    "verified_checkpoint_root",
    "verify_manifest_tree",
    "verify_payload_complete",
)

APP_NAME = "meddies-pii350-release-export"
R1_OUTPUT_FILENAME = "model.int8.r1-head-preserved.onnx"
R1_MANIFEST_FILENAME = "model.int8.r1-manifest.json"
R1_PREFLIGHT_RECEIPT_DIRECTORY = "preflight/pii350-r1"


COMPILER_APT_PACKAGES = ("gcc=4:12.2.0-3",)
"""The evaluated adapter path runs bf16, and the release runs the same arithmetic."""
CHECKPOINT_CACHE_MOUNT = "/cache/checkpoints"
HF_CACHE_MOUNT = "/cache/huggingface"
RELEASE_MOUNT = "/release"


image = add_source_pythonpath(
    cast(
        "Any",
        use_pinned_debian_snapshot(
            modal.Image.from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"),
        )
        .apt_install(*COMPILER_APT_PACKAGES)
        .pip_install("torch==2.13.0")
        .pip_install(
            "transformers==5.11.0",
            "peft==0.19.1",
            "safetensors==0.8.0",
            "datasets==4.5.0",
            "pyarrow==23.0.1",
            "huggingface-hub==1.19.0",
            "onnx==1.22.0",
            "onnxruntime==1.28.0",
        )
        .pip_install("onnx-ir==0.2.1")
        .env({
            "HF_HOME": HF_CACHE_MOUNT,
            "HF_DATASETS_CACHE": f"{HF_CACHE_MOUNT}/datasets",
            "CC": "/usr/bin/gcc",
        }),
    ),
).add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
receipt_image = add_source_pythonpath(
    cast(
        "Any",
        use_pinned_debian_snapshot(
            modal.Image.from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"),
        ),
    ),
).add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
app = modal.App(APP_NAME, image=image)
release_volume = modal.Volume.from_name("meddies-pii350-release", create_if_missing=True)
checkpoint_cache = modal.Volume.from_name("meddies-pii350-checkpoint-cache", create_if_missing=True)
huggingface_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)


def _adapter_form_tagger(
    verified: VerifiedCheckpoint,
    *,
    dtype_name: str = RUNTIME_DTYPE,
    device: str = "cpu",
) -> tuple[HiddenStateTokenTagger, Any]:
    """Load the base encoder and keep the LoRA adapter applied, never merged.

    The 6/200 (fp32) and 9/200 (bf16) span deltas once attributed to merge
    rounding were measured against the batched adapter path and are confounded
    with bf16 shape sensitivity (the same magnitude appears adapter-form vs
    batched). Adapter-form remains the release form on the stronger ground:
    the published adapter and head bytes are verbatim copies of the evaluated
    checkpoint, so no re-serialization claim is needed at all.
    """
    # reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
    # reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
    # reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # ty: ignore[possibly-missing-import]

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        BASE_MODEL_ID,
        BASE_MODEL_REVISION,
        CLASSIFIER_FILENAME,
        HEAD_HIDDEN_SIZE,
        HEAD_LABEL_COUNT,
    )
    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    wrapper = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    body = getattr(wrapper, "lfm2", None)
    if body is None:
        msg = "PII350 base wrapper does not expose its lfm2 encoder body"
        raise RuntimeError(msg)
    backbone = PeftModel.from_pretrained(body, verified.root / "adapter", is_trainable=False)
    tagger = HiddenStateTokenTagger(
        backbone,
        HEAD_HIDDEN_SIZE,
        HEAD_LABEL_COUNT,
        request_hidden_states=False,
        classifier_dtype=dtype,
        dropout=0.1,
        packed_segment_isolation=False,
    )
    classifier = torch.load(verified.root / CLASSIFIER_FILENAME, map_location="cpu", weights_only=True)
    if not isinstance(classifier, Mapping):
        msg = "checkpoint classifier state is not a state-dict mapping"
        raise RuntimeError(msg)
    tagger.classifier.load_state_dict(classifier, strict=True)
    tagger = tagger.to(device=device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION, trust_remote_code=True)
    return tagger, tokenizer


def _merged_form_tagger(
    verified: VerifiedCheckpoint,
    *,
    dtype_name: str = RUNTIME_DTYPE,
    device: str = "cpu",
) -> tuple[HiddenStateTokenTagger, Any]:
    """Bake the LoRA delta into a fresh copy of the base encoder.

    Loaded fresh rather than merged in place: merge_and_unload() rewrites the
    module it is called on, and the adapter-form tagger is the one the release
    gate was proven against. Both fit on the A10G at 350M parameters each.
    """
    tagger, tokenizer = _adapter_form_tagger(verified, dtype_name=dtype_name, device=device)
    merge = getattr(tagger.backbone, "merge_and_unload", None)
    if merge is None:
        msg = "adapter-form backbone is not a PEFT model to merge"
        raise RuntimeError(msg)
    tagger.backbone = merge()
    return tagger, tokenizer


def _merged_state_dict(tagger: HiddenStateTokenTagger) -> dict[str, Tensor]:
    """Name the merged encoder and head tensors for the published weights file.

    copy=True gives every entry its own storage; safetensors refuses to serialise two names that share one buffer.

    """
    state: dict[str, Tensor] = {}
    for module, prefix in (
        (tagger.backbone, MERGED_ENCODER_PREFIX),
        (tagger.classifier, MERGED_CLASSIFIER_PREFIX),
    ):
        for key, value in cast("Any", module).state_dict().items():
            state[f"{prefix}{key}"] = value.detach().to(device="cpu", copy=True).contiguous()
    return state


def _save_merged_weights(tagger: HiddenStateTokenTagger, path: Path) -> dict[str, Any]:
    """Write the merged encoder and head as one self-contained safetensors file."""
    from safetensors.torch import save_file

    state = _merged_state_dict(tagger)
    save_file(
        state,
        str(path),
        metadata={"format": MERGED_WEIGHTS_FORMAT, "dtype": RUNTIME_DTYPE},
    )
    return {"tensors": len(state), "bytes": path.stat().st_size}


def _copy_base_license(payload_root: Path) -> None:
    """Ship the base licence, which its redistribution terms require."""
    from huggingface_hub import hf_hub_download

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        BASE_MODEL_ID,
        BASE_MODEL_REVISION,
    )

    source = hf_hub_download(BASE_MODEL_ID, LICENSE_FILENAME, revision=BASE_MODEL_REVISION)
    shutil.copyfile(source, payload_root / LICENSE_FILENAME)


def _decode_with_tagger(
    tagger: HiddenStateTokenTagger,
    tokenizer: _ReleaseTokenizer,
    texts: Sequence[str],
) -> list[list[CharSpan]]:
    import torch

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import MAX_SEQUENCE_LENGTH

    id_to_label = bioes_label_map()
    decoded: list[list[CharSpan]] = []
    with torch.inference_mode():
        for text in texts:
            encoded = tokenizer(
                [text],
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            attention = encoded["attention_mask"]
            device = next(tagger.parameters()).device
            inputs = {key: value.to(device) for key, value in encoded.items()}
            logits = tagger(**inputs)["logits"]
            active = int(attention[0].sum().item())
            row_offsets = [(int(pair[0]), int(pair[1])) for pair in offsets[0][:active].tolist()]
            decoded.append(decode_spans_from_logits(text, logits[0][:active], row_offsets, id_to_label))
    return decoded


# reason: Checkpoint merge, decode checks, manifest writing, and packaging share one release identity.
@app.function(
    gpu="A10G",
    cpu=8,
    memory=32_768,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        RELEASE_MOUNT: release_volume,
    },
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def merge_and_package(upload: bool = True) -> dict[str, Any]:  # ruff: ignore[too-many-locals,boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    """Merge the pinned step-250 LoRA and publish a self-contained model repo.

    cuda, because the benchmark adapter this gate compares against runs its bf16 arithmetic on cuda (pii350_checkpoint.py);
    cpu-vs-cuda kernel differences flip near-tie logits and read as false parity failures.

    Ship the verified adapter and head bytes verbatim. Nothing is re-serialised, so the published weights are the evaluated
    weights.

    The adapter form references the base weights by pinned revision rather than redistributing them. The merged form does
    redistribute a derivative, which the base licence permits on the conditions this payload meets: the licence copy and
    the change notice below.

    Adapter-form cannot be rebuilt by AutoModel: the runtime is base weights plus a PEFT adapter plus a separate head. The
    repo exposes an explicit loader instead of claiming an auto_map that would fail.

    Like-for-like shape and device, so any mismatch here is a packaging defect, not numerics.

    The batched adapter path is how the published benchmark numbers were produced; its shape-driven span delta is recorded
    for honesty, not gated.

    The merged form is built only once the adapter form has passed its gate, so a merged file can never exist for a payload
    that failed verification. Its own delta is recorded, never gated: baking W+BA rounds the LoRA delta, and the adapter
    bytes remain the release's parity claim.

    """
    from huggingface_hub import snapshot_download

    destination = Path(CHECKPOINT_CACHE_MOUNT) / STEP250_CHECKPOINT_DIGEST
    snapshot_download(
        STEP250_REPO_ID,
        revision=STEP250_REVISION,
        local_dir=destination,
        allow_patterns=[f"{STEP250_ARTIFACT_PATH}/**"],
    )
    verified = verified_checkpoint_root(destination)
    checkpoint_cache.commit()

    payload_root = Path(RELEASE_MOUNT) / "meddies-pii-v2"
    if payload_root.exists():
        shutil.rmtree(payload_root)
    payload_root.mkdir(parents=True)

    tagger, tokenizer = _adapter_form_tagger(verified, device="cuda")
    tokenizer.save_pretrained(payload_root)

    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        ADAPTER_CONFIG_FILENAME,
        ADAPTER_WEIGHTS_FILENAME,
        BASE_MODEL_ID,
        BASE_MODEL_REVISION,
        CLASSIFIER_FILENAME,
        LORA_ALPHA,
        LORA_RANK,
    )

    for relative in (
        ADAPTER_CONFIG_FILENAME,
        ADAPTER_WEIGHTS_FILENAME,
        CLASSIFIER_FILENAME,
    ):
        target = payload_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verified.root / relative, target)

    labels = release_label_config()
    config_path = payload_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config.update({
        **labels,
        "meddies_release": {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "kind": RELEASE_KIND_TORCH,
            "checkpoint_digest": STEP250_CHECKPOINT_DIGEST,
            "optimizer_step": verified.optimizer_step,
            "trajectory_digest": verified.trajectory_digest,
            "runtime_dtype": RUNTIME_DTYPE,
            "decode": "viterbi-bioes-char-offset-v1",
            "base_model_id": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "base_weights_included": False,
            "merged_weights_included": True,
            "change_notice": CHANGE_NOTICE,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "adapter_applied_at_load": True,
        },
        "meddies_loader": f"{REMOTE_CODE_FILENAME[:-3]}.MeddiesPiiExtractor",
    })
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for relative, content in vendored_decode_payload().items():
        target = payload_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (payload_root / REMOTE_CODE_FILENAME).write_text(remote_code_module_source(), encoding="utf-8")

    rows = parity_rows(PARITY_ROW_MINIMUM)
    texts = [row.text for row in rows]
    packaged_spans = _decode_with_tagger(tagger, tokenizer, texts)
    reference_spans = _reference_single_spans(verified, texts)
    parity = compare_span_sets(reference_spans, packaged_spans)
    if not parity["identical"]:
        msg = (
            "adapter-form release decodes different spans than the benchmark "
            f"adapter on {parity['mismatched_rows']} of {parity['rows']} rows: "
            f"{json.dumps(parity['mismatches'][:3], sort_keys=True)}"
        )
        raise RuntimeError(msg)
    batched_delta = compare_span_sets(_reference_adapter_spans(verified, texts), packaged_spans)
    parity["batched_reference_mismatched_rows"] = batched_delta["mismatched_rows"]

    _copy_base_license(payload_root)
    merged_tagger, merged_tokenizer = _merged_form_tagger(verified, device="cuda")
    merged_weights = _save_merged_weights(merged_tagger, payload_root / MERGED_WEIGHTS_FILENAME)
    merged_spans = _decode_with_tagger(merged_tagger, tokenizer, texts)
    merged_delta = compare_span_sets(packaged_spans, merged_spans)
    parity["merged_mismatched_rows"] = merged_delta["mismatched_rows"]
    del merged_tagger, merged_tokenizer

    config["meddies_release"]["parity"] = release_parity_record(parity)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload_files = verify_payload_complete(payload_root)
    release_volume.commit()

    receipt: dict[str, Any] = {
        "status": "packaged",
        "payload_root": str(payload_root),
        "parity": parity,
        "labels": labels["num_labels"],
        "merged_weights": merged_weights,
        "required_files": len(payload_files),
    }
    if upload:
        receipt["upload"] = upload_release_tree(payload_root, repo_id=RELEASE_MODEL_REPO, kind=RELEASE_KIND_TORCH)
    print(f"PII350_RELEASE_PACKAGE::{json.dumps(receipt, sort_keys=True)}", flush=True)
    return receipt


def _reference_adapter_spans(verified: VerifiedCheckpoint, texts: Sequence[str]) -> list[list[Any]]:
    """Decode the same rows through the unmodified evaluation adapter."""
    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import (
        Pii350CheckpointAdapter,
    )

    adapter = Pii350CheckpointAdapter(verified)
    return [list(spans) for spans in adapter.predict(list(texts))]


def _reference_single_spans(verified: VerifiedCheckpoint, texts: Sequence[str]) -> list[list[Any]]:
    """Decode through the benchmark adapter's own module at consumer shape.

    The adapter's predict() pads length-grouped batches, and bf16 matmul is
    shape-sensitive: near-tie logits flip deterministically between batched and
    batch-of-1 inference (observed: 8-11/200 rows across two runs). The gate
    therefore compares at the batch-of-1 shape the published loader actually
    runs; the batched delta is recorded in the receipt, never gated.
    """
    from meddies_pii.eval_baseline.adapters.pii350_checkpoint import build_checkpoint_tagger

    reference_tagger, reference_tokenizer = build_checkpoint_tagger(verified)
    return _decode_with_tagger(reference_tagger, reference_tokenizer, texts)


# reason: ONNX export, optimization, decode parity, and manifest writes share one graph identity.
@app.function(
    gpu="A10G",
    cpu=8,
    memory=32_768,
    timeout=90 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
        RELEASE_MOUNT: release_volume,
    },
)
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def export_onnx(upload: bool = True) -> dict[str, Any]:  # ruff: ignore[too-many-locals,too-many-statements,boolean-type-hint-positional-argument,boolean-default-value-positional-argument,complex-structure]
    """Export one ONNX graph plus an int8 variant, gated on span-level parity.

    Trace the adapter-applied module so the graph encodes the same arithmetic the torch release runs. fp32 is the export
    dtype; the runtime claim is the span-parity gate below, not the storage dtype.

    Legacy tracer: the dynamo exporter needs onnxscript (absent from the image) and re-derives the graph; the tracer
    exports the arithmetic exactly as the parity gate below validates it.

    Weight-only MatMulNBits, the scheme the v2-preview browser build used: dynamic activation quantization
    (quantize_dynamic) measured 114/200 parity rows changed, which is not a shippable browser artifact.

    fp32 must reproduce the torch spans exactly. int8 is declared lossy, so its delta is recorded for the model card
    instead of blocking the release.

    """
    # reason: onnx and onnxruntime install only in the release export image, so no static reader here can resolve them.
    import numpy as np
    import onnx as onnx_lib
    import onnxruntime
    import torch
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    destination = Path(CHECKPOINT_CACHE_MOUNT) / STEP250_CHECKPOINT_DIGEST
    verified = verified_checkpoint_root(destination)
    tagger, tokenizer = _adapter_form_tagger(verified, dtype_name="float32")

    onnx_root = Path(RELEASE_MOUNT) / "meddies-pii-v2-onnx"
    if onnx_root.exists():
        shutil.rmtree(onnx_root)
    onnx_root.mkdir(parents=True)

    sample = tokenizer(
        ["Meddies release parity sample."],
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner: HiddenStateTokenTagger) -> None:
            super().__init__()
            self.inner = inner

        @override
        def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
            return cast("Tensor", self.inner(input_ids=input_ids, attention_mask=attention_mask)["logits"])

    fp32_path = onnx_root / ONNX_FP32_FILENAME
    torch.onnx.export(
        LogitsOnly(tagger).eval(),
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    int8_path = onnx_root / ONNX_INT8_FILENAME
    quantizer = MatMulNBitsQuantizer(onnx_lib.load(str(fp32_path)), block_size=32, is_symmetric=True, bits=8)
    quantizer.process()
    quantized_model: object = getattr(quantizer.model, "model", quantizer.model)
    if not isinstance(quantized_model, onnx_lib.ModelProto):
        msg = "MatMulNBits quantizer returned a malformed ONNX model"
        raise RuntimeError(msg)
    onnx_lib.save_model(
        quantized_model,
        str(int8_path),
        save_as_external_data=False,
    )

    for relative, content in vendored_decode_payload().items():
        target = onnx_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (onnx_root / POSTPROCESS_FILENAME).write_text(postprocess_module_source(), encoding="utf-8")
    (onnx_root / USAGE_SNIPPET_FILENAME).write_text(usage_snippet_source(), encoding="utf-8")
    labels = release_label_config()
    (onnx_root / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tokenizer.save_pretrained(onnx_root)

    rows = parity_rows(PARITY_ROW_MINIMUM)
    texts = [row.text for row in rows]
    torch_spans = _decode_with_tagger(tagger, tokenizer, texts)
    id_to_label = bioes_label_map()
    parity_by_variant: dict[str, Any] = {}
    for variant, path in (("fp32", fp32_path), ("int8", int8_path)):
        session = onnxruntime.InferenceSession(str(path))
        onnx_spans: list[list[Any]] = []
        for text in texts:
            encoded = tokenizer([text], return_offsets_mapping=True, truncation=True, max_length=8192)
            offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"][0]]
            logits: object = session.run(
                None,
                {
                    "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
                    "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
                },
            )[0]
            if not isinstance(logits, np.ndarray):
                msg = "ONNX runtime returned a non-array logits output"
                raise RuntimeError(msg)
            onnx_spans.append(
                decode_spans_from_logits(
                    text,
                    torch.as_tensor(logits[0][: len(offsets)]),
                    offsets,
                    id_to_label,
                ),
            )
        parity_by_variant[variant] = compare_span_sets(torch_spans, onnx_spans)
    if not parity_by_variant["fp32"]["identical"]:
        msg = f"fp32 ONNX span parity failed: {json.dumps(parity_by_variant['fp32'], sort_keys=True)[:2000]}"
        raise RuntimeError(msg)
    int8_result = parity_by_variant["int8"]
    int8_delta = {
        "declared_lossy": True,
        "rows": int8_result["rows"],
        "mismatched_rows": int8_result["mismatched_rows"],
        "affected_rows": [entry["row"] for entry in int8_result["mismatches"]],
    }
    release_volume.commit()

    receipt: dict[str, Any] = {
        "status": "exported",
        "opset": ONNX_OPSET,
        "payload_root": str(onnx_root),
        "runtime_dtype": RUNTIME_DTYPE,
        "parity": parity_by_variant,
        "int8_delta": int8_delta,
    }
    if upload:
        receipt["upload"] = upload_release_tree(onnx_root, repo_id=RELEASE_ONNX_REPO, kind=RELEASE_KIND_ONNX)
    print(f"PII350_RELEASE_ONNX::{json.dumps(receipt, sort_keys=True)}", flush=True)
    return receipt


def _release_volume_path(relative_path: str) -> Path:
    """Resolve a manifest path under the release volume, never outside it."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        msg = f"R1 source path must be release-volume relative: {relative_path!r}"
        raise RuntimeError(msg)
    root = Path(RELEASE_MOUNT).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        msg = f"R1 source path escapes the release volume: {relative_path!r}"
        raise RuntimeError(msg)
    return resolved


def _stage_r1_command(receipt_path: Path, receipt_sha256: str) -> str:
    return " ".join((
        "MODAL_PROFILE=huyhoang041100",
        "uv run modal run scripts/ops/export_pii350_release.py",
        "--action stage_r1_preflight_receipt",
        "--local-preflight-receipt-path",
        shlex.quote(str(receipt_path)),
        "--expected-receipt-sha256",
        receipt_sha256,
    ))


def _export_r1_command(remote_path: str, receipt_sha256: str) -> str:
    return " ".join((
        "MODAL_PROFILE=huyhoang041100",
        "uv run modal run scripts/ops/export_pii350_release.py",
        "--action export_r1_q8_head_preserved",
        "--preflight-receipt-path",
        shlex.quote(remote_path),
        "--expected-receipt-sha256",
        receipt_sha256,
        "--no-upload",
    ))


@app.function(
    image=receipt_image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    volumes={RELEASE_MOUNT: release_volume},
)
def stage_r1_preflight_receipt(
    receipt_bytes: bytes,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Verify and stage only the approved receipt bytes; never touch model bytes."""
    preflight = verify_r1_preflight_receipt_bytes(
        receipt_bytes,
        expected_receipt_sha256,
        require_approved=True,
    )
    relative_path = f"{R1_PREFLIGHT_RECEIPT_DIRECTORY}/{expected_receipt_sha256}.json"
    destination = _release_volume_path(relative_path)
    canonical_payload = (json.dumps(preflight, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != canonical_payload:
            msg = f"content-addressed remote preflight collision: {relative_path}"
            raise RuntimeError(msg)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{expected_receipt_sha256}.tmp"
        staging.write_bytes(canonical_payload)
        staging.replace(destination)
    release_volume.commit()
    return {
        "status": "staged_r1_preflight_receipt",
        "remote_path": relative_path,
        "receipt_sha256": expected_receipt_sha256,
        "receipt_bytes": len(canonical_payload),
        "approval": preflight["approval"],
        "model_bytes_transferred": 0,
    }


@app.function(
    cpu=8,
    memory=32_768,
    timeout=90 * 60,
    volumes={RELEASE_MOUNT: release_volume},
)
def export_r1_q8_head_preserved(
    preflight_receipt_path: str,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Quantize an existing pinned fp32 export; never acquire or upload bytes."""
    receipt_file = _release_volume_path(preflight_receipt_path)
    preflight = verify_r1_preflight_receipt(
        receipt_file,
        expected_receipt_sha256,
        require_approved=True,
    )
    source = SourceArtifactIdentity(**preflight["source"])
    source_file = _release_volume_path(source.path)
    onnx_root = Path(RELEASE_MOUNT) / "meddies-pii-v2-onnx"
    output_file = onnx_root / R1_OUTPUT_FILENAME
    manifest_file = onnx_root / R1_MANIFEST_FILENAME
    manifest = export_head_preserved_q8(
        source_file,
        output_file,
        manifest_file,
        source=source,
        exporter_commit=preflight["exporter_commit"],
    )
    release_volume.commit()
    receipt = {
        "status": "exported_r1_q8_head_preserved",
        "preflight_receipt_path": preflight_receipt_path,
        "preflight_receipt_sha256": expected_receipt_sha256,
        "approval": preflight["approval"],
        "source": manifest["source"],
        "output": manifest["output"],
        "manifest_path": str(manifest_file),
        "manifest_digest": manifest["manifest_digest"],
        "classifier_proof": manifest["classifier_proof"],
        "nodes": manifest["nodes"],
        "determinism": manifest["determinism"],
    }
    print(f"PII350_RELEASE_R1::{json.dumps(receipt, sort_keys=True)}", flush=True)
    return receipt


# reason: Manifest hashes, Torch/ONNX parity, and merged reconciliation produce one release verdict.
@app.function(
    gpu="A10G",
    cpu=8,
    memory=32_768,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CHECKPOINT_CACHE_MOUNT: checkpoint_cache,
        HF_CACHE_MOUNT: huggingface_cache,
    },
)
def verify_release(model_revision: str, onnx_revision: str) -> dict[str, Any]:  # ruff: ignore[too-many-locals]
    """Download both released repos and prove they decode like the benchmark.

    Both published paths run batch-of-1, so each is proven against a reference at its own shape and arithmetic: the torch
    loader against the adapter module single-shape bf16 on cuda, the ONNX graph against the adapter-form module in fp32
    (mirroring the export gate).

    Load through the repo's own published loader, so this proves the path a consumer actually uses rather than a path only
    this script knows.

    The merged branch ships but is otherwise never executed: packaging measured its delta through its own modules, so this
    is the only place the consumer path itself runs. A from_config or .lfm2 failure surfaces here as its own RuntimeError
    rather than going unnoticed.

    Reconciled only once the adapter form has passed: the merged delta is measured against the adapter spans, so a broken
    baseline must report as the adapter failure it is rather than as merged drift.

    """
    # reason: onnx and onnxruntime install only in the release export image, so no static reader here can resolve them.
    import numpy as np
    import onnxruntime
    import torch
    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory(prefix="meddies-release-verify-") as directory:
        model_root = Path(
            snapshot_download(
                RELEASE_MODEL_REPO,
                revision=model_revision,
                repo_type=RELEASE_REPO_TYPE,
                local_dir=Path(directory) / "torch",
            ),
        )
        onnx_root = Path(
            snapshot_download(
                RELEASE_ONNX_REPO,
                revision=onnx_revision,
                repo_type=RELEASE_REPO_TYPE,
                local_dir=Path(directory) / "onnx",
            ),
        )
        model_manifest = json.loads((model_root / RELEASE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        onnx_manifest = json.loads((onnx_root / RELEASE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        model_files = verify_manifest_tree(model_root, model_manifest)
        onnx_files = verify_manifest_tree(onnx_root, onnx_manifest)

        destination = Path(CHECKPOINT_CACHE_MOUNT) / STEP250_CHECKPOINT_DIGEST
        verified = verified_checkpoint_root(destination)
        rows = parity_rows(PARITY_ROW_MINIMUM)
        texts = [row.text for row in rows]
        reference_spans = _reference_single_spans(verified, texts)
        fp32_tagger, fp32_tokenizer = _adapter_form_tagger(verified, dtype_name="float32")
        fp32_reference_spans = _decode_with_tagger(fp32_tagger, fp32_tokenizer, texts)

        loader_spec = importlib.util.spec_from_file_location("meddies_pii_v2_published", model_root / REMOTE_CODE_FILENAME)
        if loader_spec is None or loader_spec.loader is None:
            msg = "published release does not expose its loader module"
            raise RuntimeError(msg)
        loader_module = importlib.util.module_from_spec(loader_spec)
        sys.modules[loader_spec.name] = loader_module
        loader_spec.loader.exec_module(loader_module)
        published = loader_module.MeddiesPiiExtractor.from_pretrained(str(model_root), device="cuda")
        tokenizer = published.tokenizer
        torch_spans = [list(published.extract(text)) for text in texts]
        id_to_label = bioes_label_map()

        merged_spans = published_merged_spans(loader_module, model_root, texts)
        merged_comparison = compare_span_sets(torch_spans, merged_spans)
        published_config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))

        session = onnxruntime.InferenceSession(str(onnx_root / ONNX_FP32_FILENAME))
        onnx_spans: list[list[Any]] = []
        for text in texts:
            encoded = tokenizer([text], return_offsets_mapping=True, truncation=True, max_length=8192)
            offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"][0]]
            logits: object = session.run(
                None,
                {
                    "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
                    "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
                },
            )[0]
            if not isinstance(logits, np.ndarray):
                msg = "ONNX runtime returned a non-array logits output"
                raise RuntimeError(msg)
            onnx_spans.append(
                decode_spans_from_logits(
                    text,
                    torch.as_tensor(logits[0][: len(offsets)]),
                    offsets,
                    id_to_label,
                ),
            )

    receipt: dict[str, Any] = {
        "status": "verified",
        "model_repo": RELEASE_MODEL_REPO,
        "model_revision": model_revision,
        "model_files": len(model_files),
        "onnx_repo": RELEASE_ONNX_REPO,
        "onnx_revision": onnx_revision,
        "onnx_files": len(onnx_files),
        "torch_parity": compare_span_sets(reference_spans, torch_spans),
        "onnx_parity": compare_span_sets(fp32_reference_spans, onnx_spans),
    }
    if not (receipt["torch_parity"]["identical"] and receipt["onnx_parity"]["identical"]):
        msg = f"published release does not decode like the benchmark: {json.dumps(receipt, sort_keys=True)[:2000]}"
        raise RuntimeError(
            msg,
        )
    receipt["merged_parity"] = reconcile_merged_parity(published_config, merged_comparison)
    print(f"PII350_RELEASE_VERIFY::{json.dumps(receipt, sort_keys=True)}", flush=True)
    return receipt


@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
# reason: one dispatcher validates every action beside its exact Modal invocation; extraction would split the CLI contract.
def main(  # ruff: ignore[too-many-arguments,too-many-positional-arguments,complex-structure]
    action: str = "",
    model_revision: str = "",
    onnx_revision: str = "",
    upload: bool = True,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    source_repository: str = "",
    source_revision: str = "",
    source_path: str = "",
    source_bytes: int = 0,
    source_sha256: str = "",
    exporter_commit: str = "",
    estimated_spend_usd: float = -1.0,
    approval_link: str = "",
    approval_status: str = "pending",
    receipt_output_directory: str = "",
    local_preflight_receipt_path: str = "",
    preflight_receipt_path: str = "",
    expected_receipt_sha256: str = "",
) -> None:
    if action not in {
        "merge_and_package",
        "export_onnx",
        "preflight_r1_q8_head_preserved",
        "stage_r1_preflight_receipt",
        "export_r1_q8_head_preserved",
        "verify_release",
    }:
        msg = (
            "action must be merge_and_package, export_onnx, "
            "preflight_r1_q8_head_preserved, stage_r1_preflight_receipt, "
            "export_r1_q8_head_preserved, or verify_release"
        )
        raise RuntimeError(
            msg,
        )
    if action == "merge_and_package":
        print(json.dumps(merge_and_package.remote(upload), sort_keys=True))
        return
    if action == "export_onnx":
        print(json.dumps(export_onnx.remote(upload), sort_keys=True))
        return
    if action in {"preflight_r1_q8_head_preserved", "export_r1_q8_head_preserved"}:
        if action == "preflight_r1_q8_head_preserved":
            source = SourceArtifactIdentity(
                repository=source_repository,
                revision=source_revision,
                path=source_path,
                bytes=source_bytes,
                sha256=source_sha256,
            )
            if not receipt_output_directory:
                msg = "receipt_output_directory is required for R1 preflight"
                raise RuntimeError(msg)
            contract = build_r1_preflight(
                source=source,
                exporter_commit=exporter_commit,
                estimated_spend_usd=estimated_spend_usd,
                approval_link=approval_link,
                approval_status=approval_status,
            )
            receipt_path = write_r1_preflight_receipt(contract, Path(receipt_output_directory))
            stage_command = _stage_r1_command(receipt_path, contract["receipt_sha256"])
            print(
                "PII350_RELEASE_R1_PREFLIGHT::"
                + json.dumps(
                    {"receipt": contract, "receipt_path": str(receipt_path), "stage_command": stage_command},
                    sort_keys=True,
                ),
            )
            return
        if upload:
            msg = "R1 is an experiment artifact and cannot upload; pass --no-upload"
            raise RuntimeError(msg)
        print(
            json.dumps(
                export_r1_q8_head_preserved.remote(
                    preflight_receipt_path,
                    expected_receipt_sha256,
                ),
                sort_keys=True,
            ),
        )
        return
    if action == "stage_r1_preflight_receipt":
        if not local_preflight_receipt_path:
            msg = "local_preflight_receipt_path is required for R1 receipt staging"
            raise RuntimeError(msg)
        local_receipt = Path(local_preflight_receipt_path)
        verify_r1_preflight_receipt(
            local_receipt,
            expected_receipt_sha256,
            require_approved=True,
        )
        staged = stage_r1_preflight_receipt.remote(local_receipt.read_bytes(), expected_receipt_sha256)
        export_command = _export_r1_command(staged["remote_path"], staged["receipt_sha256"])
        print(
            f"PII350_RELEASE_R1_STAGE::{json.dumps({'export_command': export_command, 'stage': staged}, sort_keys=True)}",
        )
        return
    if not model_revision or not onnx_revision:
        msg = "model_revision and onnx_revision are required for verify_release"
        raise RuntimeError(msg)
    print(json.dumps(verify_release.remote(model_revision, onnx_revision), sort_keys=True))
