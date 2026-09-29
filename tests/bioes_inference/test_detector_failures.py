from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

import meddies_pii.bioes_inference.detector as inference_module
from meddies_pii.bioes_inference import (
    BioesSpanDetector,
    ExpectedFileIdentity,
    OnnxRuntimeBackend,
    OpenVinoBackend,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class _Backend:
    name = "synthetic"

    # reason: this mirrors the inference `Backend` protocol, whose `load` the code under test calls on the
    # reason: backend instance it is handed, so the bound-method form is the API being stood in for.
    def load(self, model_path: Path) -> None:  # ruff: ignore[no-self-use]
        assert model_path.is_file()

    # reason: this mirrors the inference `Backend` protocol, whose `infer` the code under test calls on the
    # reason: backend instance it is handed, so the bound-method form is the API being stood in for.
    def infer(  # ruff: ignore[no-self-use]
        self,
        # reason: this signature mirrors the one `BioesSpanDetector` calls at `bioes_inference/detector.py:165-169` as
        # reason: `self.backend.infer(input_ids, attention_mask, bucket=bucket)`; the bucketed zeros this
        # reason: returns do not depend on either sequence, but dropping them would break that call.
        input_ids: list[int],  # ruff: ignore[unused-method-argument]
        attention_mask: list[int],  # ruff: ignore[unused-method-argument]
        *,
        bucket: int,
    ) -> np.ndarray:
        return np.zeros((1, bucket, 37), dtype=np.float32)


def _identity(payload: bytes) -> ExpectedFileIdentity:
    return ExpectedFileIdentity(
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _detector(
    *,
    buckets: tuple[int, ...] = (8,),
    expected_model_size_bytes: int = 1,
) -> BioesSpanDetector:
    return BioesSpanDetector(
        model_path=Path("model.onnx"),
        tokenizer_path=Path("tokenizer"),
        expected_model_size_bytes=expected_model_size_bytes,
        expected_model_sha256="a" * 64,
        expected_tokenizer_files={
            "tokenizer.json": ExpectedFileIdentity(
                size_bytes=1,
                sha256="b" * 64,
            ),
        },
        backend=_Backend(),
        buckets=buckets,
    )


@pytest.mark.parametrize("buckets", [(), (0,), (8, 4), (8, 8)])
def test_detector_rejects_ambiguous_bucket_configuration(
    buckets: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="unique positive integers"):
        _detector(buckets=buckets)


@pytest.mark.parametrize("size", [True, 0])
def test_detector_rejects_invalid_model_size(size: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _detector(expected_model_size_bytes=size)


def test_detector_rejects_inference_before_load_and_invalid_texts(
    tmp_path: Path,
) -> None:
    detector = _detector()

    with pytest.raises(RuntimeError, match=r"load\(\) must be called"):
        detector.detect(["Maya"])

    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    payload = b"{}"
    tokenizer.joinpath("tokenizer.json").write_bytes(payload)
    loaded = BioesSpanDetector(
        model_path=model,
        tokenizer_path=tokenizer,
        expected_model_size_bytes=model.stat().st_size,
        expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        expected_tokenizer_files={"tokenizer.json": _identity(payload)},
        backend=_Backend(),
        tokenizer_loader=lambda _: object(),
    )
    loaded.load()
    with pytest.raises(ValueError, match="non-empty strings"):
        loaded.detect([""])


def test_onnx_backend_rejects_invalid_lifecycle_and_output_cardinality(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    factory_calls = 0

    class _Session:
        # reason: this mirrors `onnxruntime`'s `InferenceSession.run`, which the code under test calls on the
        # reason: session instance the factory returns, so the bound-method form is the API being stood in for.
        def run(self, output_names: None, inputs: Mapping[str, object]) -> list[object]:  # ruff: ignore[no-self-use]
            assert output_names is None
            assert tuple(inputs) == ("input_ids", "attention_mask")
            return []

    def factory(path: Path, threads: int) -> _Session:
        nonlocal factory_calls
        assert path == model
        assert threads == 2
        factory_calls += 1
        return _Session()

    with pytest.raises(ValueError, match="threads must be positive"):
        OnnxRuntimeBackend(threads=0)

    unloaded = OnnxRuntimeBackend(threads=1)
    with pytest.raises(RuntimeError, match=r"load\(\) must be called"):
        unloaded.infer([1], [1], bucket=1)

    backend = OnnxRuntimeBackend(threads=2, session_factory=factory)
    backend.load(model)
    backend.load(model)
    with pytest.raises(RuntimeError, match="expected one output tensor"):
        backend.infer([1], [1], bucket=1)
    assert factory_calls == 1


def test_openvino_backend_rejects_invalid_lifecycle_and_missing_output(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    factory_calls = 0
    output_key = object()

    class _Compiled:
        def __call__(self, inputs: Mapping[str, object]) -> dict[object, object]:
            assert tuple(inputs) == ("input_ids", "attention_mask")
            return {}

    def factory(path: Path, threads: int) -> tuple[_Compiled, object]:
        nonlocal factory_calls
        assert path == model
        assert threads == 3
        factory_calls += 1
        return _Compiled(), output_key

    with pytest.raises(ValueError, match="threads must be positive"):
        OpenVinoBackend(threads=0)

    unloaded = OpenVinoBackend(threads=1)
    with pytest.raises(RuntimeError, match=r"load\(\) must be called"):
        unloaded.infer([1], [1], bucket=1)

    backend = OpenVinoBackend(threads=3, compiled_factory=factory)
    backend.load(model)
    backend.load(model)
    with pytest.raises(RuntimeError, match="did not return the logits output"):
        backend.infer([1], [1], bucket=1)
    assert factory_calls == 1


@pytest.mark.parametrize("backend_type", [OnnxRuntimeBackend, OpenVinoBackend])
def test_optional_inference_dependency_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_type: type[OnnxRuntimeBackend | OpenVinoBackend],
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")

    def missing_dependency(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(inference_module.importlib, "import_module", missing_dependency)

    with pytest.raises(ModuleNotFoundError):
        backend_type(threads=1).load(model)


def test_default_tokenizer_loader_accepts_only_the_fixed_local_file_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    payloads = {
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    for filename, payload in payloads.items():
        tokenizer.joinpath(filename).write_bytes(payload)

    loaded_paths: list[str] = []
    transformers = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, *, local_files_only: bool) -> object:
            assert local_files_only is True
            loaded_paths.append(path)
            return object()

    # reason: the code under test imports transformers by name out of sys.modules, so the stand-in must be a
    # reason: real module; ModuleType declares no such attribute and no annotation admits the write.
    transformers.AutoTokenizer = _AutoTokenizer  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    detector = BioesSpanDetector(
        model_path=model,
        tokenizer_path=tokenizer,
        expected_model_size_bytes=model.stat().st_size,
        expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        expected_tokenizer_files={filename: _identity(payload) for filename, payload in payloads.items()},
        backend=_Backend(),
    )
    detector.load()

    assert len(loaded_paths) == 1
    assert not Path(loaded_paths[0]).exists()


def test_default_tokenizer_loader_rejects_incomplete_manifest(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    payload = b"{}"
    tokenizer.joinpath("tokenizer.json").write_bytes(payload)
    detector = BioesSpanDetector(
        model_path=model,
        tokenizer_path=tokenizer,
        expected_model_size_bytes=model.stat().st_size,
        expected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        expected_tokenizer_files={"tokenizer.json": _identity(payload)},
        backend=_Backend(),
    )

    with pytest.raises(RuntimeError, match="tokenizer directory identity mismatch"):
        detector.load()
