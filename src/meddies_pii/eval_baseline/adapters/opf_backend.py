from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: adapters load their model stack inside `load()`, so listing an adapter costs nothing.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import fcntl
import importlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from meddies_pii.eval_baseline.adapters.opf_windowing import (
    TokenizedOpfDoc,
    pack_token_windows,
)
from meddies_pii.evaluation.identity import checkpoint_tree_artifact
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from meddies_pii.json_types import JsonObject

MODEL_ID = "openai/privacy-filter"
MODEL_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
DEFAULT_NATIVE_CHECKPOINT_DIR = f"/cache/opf-native-package-checkpoint-{MODEL_REVISION}"
_CACHE_MANIFEST_SUFFIX = ".opf-checkpoint.json"
_CACHE_LOCK_SUFFIX = ".opf-checkpoint.lock"
_CACHE_MANIFEST_SCHEMA_VERSION = 1


class NativePredictorConfig(Protocol):
    @property
    def batch_size(self) -> int: ...

    @property
    def triton(self) -> str: ...

    @property
    def compile_mode(self) -> str: ...


class NativePredictorDoc(Protocol):
    @property
    def doc_id(self) -> str: ...

    @property
    def text(self) -> str: ...


NativePredictFn = Callable[[Sequence[NativePredictorDoc]], dict[str, list[CharSpan]]]


def prepare_native_checkpoint(
    checkpoint_dir: str | Path = DEFAULT_NATIVE_CHECKPOINT_DIR,
    *,
    model_id: str = MODEL_ID,
) -> Path:
    """Return one verified native checkpoint, downloading its pinned revision once.

    The cache manifest binds the requested model and revision to the exact normalized
    checkpoint tree. A pre-existing directory without that manifest is deliberately
    rejected: its contents cannot be attributed to the pinned download.

    Returns:
        The checkpoint directory, verified either way: an existing tree is re-verified against
        its cache manifest before being reused, and a fresh download is verified after landing.
        The work happens under a lock, so concurrent callers share one download rather than
        racing into the same directory.

    """
    target = Path(checkpoint_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _checkpoint_cache_lock(target):
        if target.exists():
            _verify_cached_checkpoint(target, model_id)
            return target
        _download_and_cache_checkpoint(target, model_id)
        _verify_cached_checkpoint(target, model_id)
        return target


def _download_and_cache_checkpoint(target: Path, model_id: str) -> None:
    hub = importlib.import_module("huggingface_hub")
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.download-") as staging_root:
        staging = Path(staging_root)
        download_root = staging / "download"
        download_root.mkdir()
        hub.snapshot_download(
            repo_id=model_id,
            revision=MODEL_REVISION,
            local_dir=str(download_root),
            allow_patterns=["original/*"],
        )
        original = download_root / "original"
        if not original.is_dir():
            msg = f"native checkpoint download missing original subtree: {original}"
            raise RuntimeError(msg)
        prepared = staging / "checkpoint"
        prepared.mkdir()
        for path in original.iterdir():
            shutil.move(str(path), str(prepared / path.name))
        _normalize_native_config(prepared / "config.json")
        artifact = checkpoint_tree_artifact(f"hf://{model_id}", prepared)
        Path(prepared).replace(target)
        _write_cache_manifest(target, model_id, artifact.to_payload())


def _verify_cached_checkpoint(target: Path, model_id: str) -> None:
    if not target.is_dir():
        msg = f"checkpoint cache is unverified: {target} is not a directory"
        raise RuntimeError(msg)
    manifest_path = _cache_manifest_path(target)
    if not manifest_path.is_file():
        msg = f"checkpoint cache is unverified: {manifest_path} is missing"
        raise RuntimeError(msg)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"checkpoint cache is unverified: invalid {manifest_path}"
        raise RuntimeError(msg) from exc
    if not isinstance(manifest, dict):
        msg = f"checkpoint cache is unverified: invalid {manifest_path}"
        raise RuntimeError(msg)
    if manifest.get("schema_version") != _CACHE_MANIFEST_SCHEMA_VERSION:
        msg = f"checkpoint cache is unverified: unsupported {manifest_path}"
        raise RuntimeError(msg)
    if manifest.get("model_id") != model_id:
        msg = "checkpoint cache model does not match the requested model"
        raise RuntimeError(msg)
    if manifest.get("model_revision") != MODEL_REVISION:
        msg = "checkpoint cache revision does not match the pinned revision"
        raise RuntimeError(msg)
    actual = checkpoint_tree_artifact(f"hf://{model_id}", target).to_payload()
    if manifest.get("checkpoint") != actual:
        msg = "checkpoint cache content does not match the verified manifest"
        raise RuntimeError(msg)


def _normalize_native_config(config_path: Path) -> None:
    if not config_path.is_file():
        msg = f"native checkpoint is missing config: {config_path}"
        raise RuntimeError(msg)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"native checkpoint config is invalid JSON: {config_path}"
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"native checkpoint config is not a JSON object: {config_path}"
        raise RuntimeError(msg)
    if payload.get("model_type") == "openai_privacy_filter":
        payload["model_type"] = "privacy_filter"
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_cache_manifest(target: Path, model_id: str, checkpoint: JsonObject) -> None:
    manifest = {
        "schema_version": _CACHE_MANIFEST_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": MODEL_REVISION,
        "checkpoint": checkpoint,
    }
    manifest_path = _cache_manifest_path(target)
    temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(temporary_path).replace(manifest_path)


def _cache_manifest_path(target: Path) -> Path:
    return target.with_name(f"{target.name}{_CACHE_MANIFEST_SUFFIX}")


@contextmanager
def _checkpoint_cache_lock(target: Path) -> Iterator[None]:
    lock_path = target.with_name(f"{target.name}{_CACHE_LOCK_SUFFIX}")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def apply_native_env(config: NativePredictorConfig) -> None:
    os.environ["OPF_MOE_TRITON"] = "1" if config.triton == "on" else "0"
    if config.compile_mode == "none":
        os.environ.pop("OPF_TORCH_COMPILE", None)
        os.environ.pop("OPF_TORCH_COMPILE_MODE", None)
    else:
        os.environ["OPF_TORCH_COMPILE"] = "1"
        os.environ["OPF_TORCH_COMPILE_MODE"] = config.compile_mode


class _DetectedSpan(Protocol):
    """The four attributes a detector span must expose to become a `CharSpan`.

    The offsets are typed as coercible rather than `int` because both the native and the ONNX
    backend hand them back as whatever their runtime produced, and this is the one place that
    normalises them.
    """

    start: str | int | float
    end: str | int | float
    text: object
    label: object


def detected_span_to_char_span(span: _DetectedSpan) -> CharSpan:
    start = int(span.start)
    end = int(span.end)
    return CharSpan(
        start=start,
        end=end,
        text=str(span.text),
        label=str(span.label),
    )


def build_native_batched_predictor(
    config: NativePredictorConfig,
    *,
    checkpoint_dir: str | Path = DEFAULT_NATIVE_CHECKPOINT_DIR,
) -> NativePredictFn:
    apply_native_env(config)
    checkpoint = prepare_native_checkpoint(checkpoint_dir)
    runtime_mod = importlib.import_module("opf._core.runtime")
    decoding_mod = importlib.import_module("opf._core.decoding")
    spans_mod = importlib.import_module("opf._core.spans")
    runtime = runtime_mod.load_inference_runtime(
        checkpoint=str(checkpoint),
        device_name="cuda",
        n_ctx_override=None,
        trim_span_whitespace=True,
        discard_overlapping_predicted_spans=False,
        output_mode="typed",
    )
    decoder, _biases = decoding_mod.build_sequence_decoder(
        decode_mode="viterbi",
        label_info=runtime.label_info,
        viterbi_calibration_path=None,
        checkpoint_dir=str(checkpoint),
    )

    # reason: predict coordinates with offsets with pack windows; extra seams would misattribute row errors.
    def predict(docs: Sequence[NativePredictorDoc]) -> dict[str, list[CharSpan]]:  # ruff: ignore[too-many-locals]
        import torch
        from torch.nn import functional

        tokenized = tuple(
            TokenizedOpfDoc(
                example_id=doc.doc_id,
                token_ids=tuple(
                    int(tok)
                    for tok in runtime.encoding.encode(
                        doc.text,
                        allowed_special="all",
                    )
                ),
            )
            for doc in docs
        )
        batches = pack_token_windows(
            tokenized,
            window_size=int(runtime.n_ctx),
            batch_size=config.batch_size,
            pad_token_id=int(runtime.pad_token_id),
        )
        scores_by_example: dict[str, dict[int, Any]] = {}
        counts_by_example: dict[str, dict[int, int]] = {}
        with torch.inference_mode():
            for batch in batches:
                input_ids = torch.tensor(
                    batch.input_ids,
                    device=runtime.device,
                    dtype=torch.int32,
                )
                attention_mask = torch.tensor(
                    batch.attention_mask,
                    device=runtime.device,
                    dtype=torch.bool,
                )
                logits = runtime.model(input_ids, attention_mask=attention_mask)
                log_probs = functional.log_softmax(logits.float(), dim=-1).cpu()
                for row_idx, window in enumerate(batch.windows):
                    score_target = scores_by_example.setdefault(window.example_id, {})
                    count_target = counts_by_example.setdefault(window.example_id, {})
                    for token_pos, is_valid in enumerate(window.valid_mask):
                        if not is_valid or token_pos >= len(window.offsets):
                            continue
                        offset = window.offsets[token_pos]
                        score_vec = log_probs[row_idx, token_pos]
                        existing = score_target.get(offset)
                        score_target[offset] = (
                            score_vec.clone() if existing is None else torch.logaddexp(existing, score_vec)
                        )
                        count_target[offset] = count_target.get(offset, 0) + 1

        output: dict[str, list[CharSpan]] = {}
        for doc, token_doc in zip(docs, tokenized, strict=True):
            offset_scores = scores_by_example.get(doc.doc_id, {})
            offset_counts = counts_by_example.get(doc.doc_id, {})
            ordered_offsets = sorted(offset_scores)
            if not ordered_offsets:
                output[doc.doc_id] = []
                continue
            stacked_scores = torch.stack(
                [offset_scores[offset] - math.log(float(offset_counts[offset])) for offset in ordered_offsets],
                dim=0,
            )
            decoded = decoder.decode(stacked_scores) if decoder is not None else stacked_scores.argmax(dim=1).tolist()
            predicted_labels = {offset: int(label) for offset, label in zip(ordered_offsets, decoded, strict=True)}
            token_spans = spans_mod.labels_to_spans(
                predicted_labels,
                runtime.label_info,
            )
            decoded_text, char_starts, char_ends = spans_mod.decode_text_with_offsets(
                token_doc.token_ids,
                runtime.encoding,
            )
            source_text = decoded_text if decoded_text != doc.text else doc.text
            char_spans = spans_mod.token_spans_to_char_spans(
                token_spans,
                char_starts,
                char_ends,
            )
            char_spans = spans_mod.trim_char_spans_whitespace(char_spans, source_text)
            output[doc.doc_id] = [
                CharSpan(
                    start=int(start),
                    end=int(end),
                    text=source_text[int(start) : int(end)],
                    label=str(runtime.label_info.span_class_names[int(label_idx)]),
                )
                for label_idx, start, end in char_spans
                if 0 <= int(start) < int(end) <= len(source_text)
            ]
        return output

    return predict
