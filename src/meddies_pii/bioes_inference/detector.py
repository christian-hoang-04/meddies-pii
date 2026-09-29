"""Optional runtime dependency: dynamic import keeps the base package importable."""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the redaction runtime is an optional extra, and several backends are resolved by name at call time.
import importlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    decode_bioes_from_offsets,
    tokenize_and_align,
    viterbi_decode_numpy,
)
from meddies_pii.bioes_inference.artifact_staging import (
    copy_verified_file,
    copy_verified_tokenizer_directory,
    validated_tokenizer_manifest,
)
from meddies_pii.bioes_inference.contracts import ExpectedFileIdentity, SpanDetection, validate_sha256

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

_TOKENIZER_FILES = frozenset({"tokenizer.json", "tokenizer_config.json"})


class BioesInferenceBackend(Protocol):
    name: str

    def load(self, model_path: Path) -> None: ...

    def infer(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        *,
        bucket: int,
    ) -> object: ...


class _OrtSessionLike(Protocol):
    def run(
        self,
        output_names: None,
        inputs: Mapping[str, object],
    ) -> Sequence[object]: ...


class _OpenVinoCompiledLike(Protocol):
    def __call__(self, inputs: Mapping[str, object]) -> Mapping[object, object]: ...


class BioesSpanDetector:
    # reason: Model hash, tokenizer manifest, backend, loader, and buckets are the detector constructor contract.
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        model_path: Path,
        tokenizer_path: Path,
        expected_model_size_bytes: int,
        expected_model_sha256: str,
        expected_tokenizer_files: Mapping[str, ExpectedFileIdentity],
        backend: BioesInferenceBackend,
        tokenizer_loader: Callable[[Path], object] | None = None,
        buckets: Sequence[int] = (512, 2048, 8192),
    ) -> None:
        normalized_buckets = tuple(int(bucket) for bucket in buckets)
        if (
            not normalized_buckets
            or any(bucket <= 0 for bucket in normalized_buckets)
            or tuple(sorted(set(normalized_buckets))) != normalized_buckets
        ):
            msg = "buckets must be unique positive integers in ascending order"
            raise ValueError(msg)
        validate_sha256(expected_model_sha256, name="expected_model_sha256")
        if (
            isinstance(expected_model_size_bytes, bool)
            or not isinstance(expected_model_size_bytes, int)
            or expected_model_size_bytes <= 0
        ):
            msg = "expected_model_size_bytes must be a positive integer"
            raise ValueError(msg)

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.expected_model_size_bytes = expected_model_size_bytes
        self.expected_model_sha256 = expected_model_sha256.lower()
        self.expected_tokenizer_files = validated_tokenizer_manifest(expected_tokenizer_files)
        self.backend = backend
        self.tokenizer_loader = tokenizer_loader or _load_tokenizer
        self.buckets = normalized_buckets
        self._tokenizer: object | None = None
        label_space = build_bioes_label_space(ENTITY_LABELS)
        self._id_to_label = dict(enumerate(label_space))

    def load(self) -> None:
        if self._tokenizer is not None:
            return
        with tempfile.TemporaryDirectory(prefix="meddies-pii-runtime-") as stage:
            stage_root = Path(stage)
            Path(stage_root).chmod(0o700)
            staged_model = stage_root / "model.onnx"
            copy_verified_file(
                self.model_path,
                staged_model,
                expected_size_bytes=self.expected_model_size_bytes,
                expected_sha256=self.expected_model_sha256,
                identity_name="model",
            )
            staged_tokenizer = stage_root / "tokenizer"
            staged_tokenizer.mkdir(mode=0o700)
            copy_verified_tokenizer_directory(
                self.tokenizer_path,
                staged_tokenizer,
                self.expected_tokenizer_files,
            )
            tokenizer = self.tokenizer_loader(staged_tokenizer)
            self.backend.load(staged_model)
        self._tokenizer = tokenizer

    def detect(self, texts: Sequence[str]) -> list[SpanDetection]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            msg = "BioesSpanDetector.load() must be called before detect()"
            raise RuntimeError(msg)
        if any(not isinstance(text, str) or not text for text in texts):
            msg = "texts must contain non-empty strings"
            raise ValueError(msg)

        return [self._detect_one(tokenizer, text) for text in texts]

    def _detect_one(self, tokenizer: object, text: str) -> SpanDetection:
        example = tokenize_and_align(
            cast("PreTrainedTokenizerBase", tokenizer),
            text,
            [],
            max_length=self.buckets[-1],
        )
        num_tokens = len(example.input_ids)
        bucket = next(
            (candidate for candidate in self.buckets if num_tokens <= candidate),
            self.buckets[-1],
        )
        truncated = bool(example.truncated or num_tokens > bucket)
        input_ids = list(example.input_ids[:bucket])
        attention_mask = list(example.attention_mask[:bucket])
        offsets = list(example.offset_mapping[:bucket])
        padding = bucket - len(input_ids)
        input_ids.extend([_pad_token_id(tokenizer)] * padding)
        attention_mask.extend([0] * padding)
        offsets.extend([(0, 0)] * padding)

        logits = self.backend.infer(
            input_ids,
            attention_mask,
            bucket=bucket,
        )
        logits_row = _validated_logits_row(
            logits,
            expected_tokens=bucket,
            expected_labels=len(self._id_to_label),
        )
        pred_ids = viterbi_decode_numpy(logits_row, self._id_to_label, offsets)
        spans = decode_bioes_from_offsets(text, offsets, pred_ids, self._id_to_label)
        return SpanDetection(
            spans=spans,
            bucket=bucket,
            num_tokens=num_tokens,
            truncated=truncated,
        )


class OnnxRuntimeBackend:
    name = "onnxruntime-cpu"

    def __init__(
        self,
        *,
        threads: int,
        session_factory: Callable[[Path, int], _OrtSessionLike] | None = None,
    ) -> None:
        if threads <= 0:
            msg = "threads must be positive"
            raise ValueError(msg)
        self.threads = threads
        self._session_factory = session_factory or _create_ort_session
        self._session: _OrtSessionLike | None = None

    def load(self, model_path: Path) -> None:
        if self._session is None:
            self._session = self._session_factory(model_path, self.threads)

    def infer(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        *,
        bucket: int,  # ruff: ignore[unused-method-argument] reason: keyword is fixed by BioesInferenceBackend
    ) -> object:
        session = self._session
        if session is None:
            msg = "OnnxRuntimeBackend.load() must be called before infer()"
            raise RuntimeError(msg)
        import numpy as np

        outputs = session.run(
            None,
            {
                "input_ids": np.asarray([input_ids], dtype=np.int64),
                "attention_mask": np.asarray([attention_mask], dtype=np.int64),
            },
        )
        if len(outputs) != 1:
            msg = f"ONNX backend expected one output tensor, received {len(outputs)}"
            raise RuntimeError(msg)
        return outputs[0]


class OpenVinoBackend:
    name = "openvino-cpu"

    def __init__(
        self,
        *,
        threads: int,
        compiled_factory: Callable[[Path, int], tuple[_OpenVinoCompiledLike, object]] | None = None,
    ) -> None:
        if threads <= 0:
            msg = "threads must be positive"
            raise ValueError(msg)
        self.threads = threads
        self._compiled_factory = compiled_factory or _compile_openvino_model
        self._compiled: _OpenVinoCompiledLike | None = None
        self._output_key: object | None = None

    def load(self, model_path: Path) -> None:
        if self._compiled is not None:
            return
        self._compiled, self._output_key = self._compiled_factory(model_path, self.threads)

    def infer(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        *,
        bucket: int,  # ruff: ignore[unused-method-argument] reason: keyword is fixed by BioesInferenceBackend
    ) -> object:
        compiled = self._compiled
        if compiled is None or self._output_key is None:
            msg = "OpenVinoBackend.load() must be called before infer()"
            raise RuntimeError(msg)
        import numpy as np

        outputs = compiled({
            "input_ids": np.asarray([input_ids], dtype=np.int64),
            "attention_mask": np.asarray([attention_mask], dtype=np.int64),
        })
        try:
            return outputs[self._output_key]
        except KeyError as exc:
            msg = "OpenVINO backend did not return the logits output"
            raise RuntimeError(msg) from exc


def _load_tokenizer(path: Path) -> object:
    try:
        observed_files = {entry.name for entry in path.iterdir()}
        expected_files_are_regular = all(
            (path / filename).is_file() and not (path / filename).is_symlink() for filename in _TOKENIZER_FILES
        )
    except OSError as error:
        msg = "tokenizer directory is unreadable"
        raise RuntimeError(msg) from error
    if observed_files != _TOKENIZER_FILES or not expected_files_are_regular:
        msg = (
            "tokenizer directory identity mismatch: "
            f"expected_files={len(_TOKENIZER_FILES)} "
            f"observed_entries={len(observed_files)}"
        )
        raise RuntimeError(
            msg,
        )

    # reason: transformers declares its public names only under `TYPE_CHECKING` and serves them
    # reason: at runtime through `_LazyModule`, so a static reader cannot prove the symbol is
    # reason: present. Verified against the pinned 5.14.1 in this environment:
    # reason: `hasattr(transformers, "AutoTokenizer")` is True.
    from transformers import AutoTokenizer  # ty: ignore[possibly-missing-import]

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _create_ort_session(model_path: Path, threads: int) -> _OrtSessionLike:
    ort = cast("Any", importlib.import_module("onnxruntime"))

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return cast(
        "_OrtSessionLike",
        ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        ),
    )


def _compile_openvino_model(model_path: Path, threads: int) -> tuple[_OpenVinoCompiledLike, object]:
    ov = cast("Any", importlib.import_module("openvino"))

    core = ov.Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(
        model,
        "CPU",
        config={
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": str(threads),
            "NUM_STREAMS": "1",
        },
    )
    return cast("_OpenVinoCompiledLike", compiled), compiled.output(0)


def _pad_token_id(tokenizer: object) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    return int(value) if value is not None else 0


def _validated_logits_row(
    logits: object,
    *,
    expected_tokens: int,
    expected_labels: int,
) -> object:
    import numpy as np

    tensor = np.asarray(logits)
    expected_shape = (1, expected_tokens, expected_labels)
    if tuple(tensor.shape) != expected_shape:
        msg = f"backend logits shape mismatch: expected={expected_shape} actual={tuple(tensor.shape)}"
        raise RuntimeError(msg)
    return tensor[0].astype(np.float32, copy=False)
