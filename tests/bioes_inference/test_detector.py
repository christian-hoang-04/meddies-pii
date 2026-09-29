from __future__ import annotations

import builtins
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
import torch

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    viterbi_decode_logits,
    viterbi_decode_numpy,
)
from meddies_pii.bioes_inference import (
    BioesSpanDetector,
    ExpectedFileIdentity,
    OnnxRuntimeBackend,
    OpenVinoBackend,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, *, overflow: bool = False) -> None:
        self.overflow = overflow

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        tokens = list(range(1, len(text) + 1))
        offsets = [(index, index + 1) for index in range(len(text))]
        if self.overflow:
            split = max(1, len(tokens) // 2)
            return {
                "input_ids": [tokens[:split], tokens[split:]],
                "attention_mask": [
                    [1] * len(tokens[:split]),
                    [1] * len(tokens[split:]),
                ],
                "offset_mapping": [offsets[:split], offsets[split:]],
            }
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "offset_mapping": offsets,
            "num_truncated_tokens": 0,
        }


class FakeBackend:
    name = "fake"

    def __init__(self, *, bad_shape: bool = False) -> None:
        self.bad_shape = bad_shape
        self.load_count = 0

    def load(self, model_path: Path) -> None:
        assert model_path.is_file()
        self.load_count += 1

    def infer(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        *,
        bucket: int,
    ) -> np.ndarray:
        assert len(input_ids) == len(attention_mask) == bucket
        label_space = build_bioes_label_space(ENTITY_LABELS)
        labels = len(label_space) - int(self.bad_shape)
        logits = np.full((1, bucket, labels), -10.0, dtype=np.float32)
        logits[:, :, 0] = 10.0
        if not self.bad_shape:
            logits[0, 0, label_space.index("S-human_name")] = 30.0
        return logits


def _model_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"synthetic-model")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _tokenizer_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, ExpectedFileIdentity]]:
    path = tmp_path / "tokenizer"
    path.mkdir()
    payloads = {
        "tokenizer.json": b'{"synthetic":true}',
        "tokenizer_config.json": b'{"model_max_length":8192}',
    }
    for filename, payload in payloads.items():
        path.joinpath(filename).write_bytes(payload)
    return path, {
        filename: ExpectedFileIdentity(
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for filename, payload in payloads.items()
    }


def test_detector_verifies_model_and_loads_backend_once(tmp_path: Path) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    backend = FakeBackend()
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=backend,
        tokenizer_loader=lambda _: FakeTokenizer(),
        buckets=(8, 16),
    )

    detector.load()
    detector.load()

    assert backend.load_count == 1


def test_detector_consumes_model_from_open_handle_when_cache_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    expected_model = model_path.read_bytes()
    replacement_model = b"x" * len(expected_model)
    opened_model = tmp_path / "opened-model.onnx"
    pending_replacement = tmp_path / "replacement-model.onnx"
    pending_replacement.write_bytes(replacement_model)
    original_open = os.open
    replacement_happened = False
    consumed_model: bytes | None = None
    consumed_path: Path | None = None
    consumed_mode: int | None = None
    source_model_path = model_path

    def replace_after_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replacement_happened
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if not replacement_happened and dir_fd is None and Path(os.fsdecode(path)) == source_model_path:
            source_model_path.replace(opened_model)
            pending_replacement.replace(source_model_path)
            replacement_happened = True
        return descriptor

    class CapturingBackend(FakeBackend):
        @override
        def load(self, model_path: Path) -> None:
            nonlocal consumed_mode, consumed_model, consumed_path
            consumed_path = model_path
            consumed_model = model_path.read_bytes()
            consumed_mode = stat.S_IMODE(model_path.stat().st_mode)
            if replacement_happened:
                source_model_path.unlink()
                opened_model.replace(source_model_path)
            super().load(model_path)

    monkeypatch.setattr(os, "open", replace_after_open)
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=len(expected_model),
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=CapturingBackend(),
        tokenizer_loader=lambda _: FakeTokenizer(),
        buckets=(8, 16),
    )

    detector.load()

    assert replacement_happened is True
    assert consumed_model == expected_model
    assert consumed_mode == 0o400
    assert model_path.read_bytes() == expected_model
    assert consumed_path is not None
    assert not consumed_path.exists()


def test_detector_consumes_tokenizer_from_open_handle_when_cache_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    expected_tokenizer = b'{"version":"expected"}'
    replacement_tokenizer = b'{"version":"tampered"}'
    assert len(expected_tokenizer) == len(replacement_tokenizer)
    tokenizer_file = tokenizer_path / "tokenizer.json"
    tokenizer_file.write_bytes(expected_tokenizer)
    config_file = tokenizer_path / "tokenizer_config.json"
    config_file.write_bytes(b"{}")
    opened_tokenizer = tmp_path / "opened-tokenizer.json"
    pending_replacement = tmp_path / "replacement-tokenizer.json"
    pending_replacement.write_bytes(replacement_tokenizer)
    original_open = os.open
    replacement_happened = False
    consumed_tokenizer: bytes | None = None
    consumed_path: Path | None = None
    consumed_modes: tuple[int, int, int] | None = None

    def replace_after_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replacement_happened
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if not replacement_happened and dir_fd is not None and os.fsdecode(path) == "tokenizer.json":
            tokenizer_file.replace(opened_tokenizer)
            pending_replacement.replace(tokenizer_file)
            replacement_happened = True
        return descriptor

    def load_tokenizer(staged_tokenizer_path: Path) -> FakeTokenizer:
        nonlocal consumed_modes, consumed_tokenizer, consumed_path
        consumed_path = staged_tokenizer_path
        consumed_tokenizer = staged_tokenizer_path.joinpath("tokenizer.json").read_bytes()
        consumed_modes = (
            stat.S_IMODE(staged_tokenizer_path.parent.stat().st_mode),
            stat.S_IMODE(staged_tokenizer_path.stat().st_mode),
            stat.S_IMODE(staged_tokenizer_path.joinpath("tokenizer.json").stat().st_mode),
        )
        if replacement_happened:
            tokenizer_file.unlink()
            opened_tokenizer.replace(tokenizer_file)
        return FakeTokenizer()

    monkeypatch.setattr(os, "open", replace_after_open)
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files={
            "tokenizer.json": ExpectedFileIdentity(
                size_bytes=len(expected_tokenizer),
                sha256=hashlib.sha256(expected_tokenizer).hexdigest(),
            ),
            "tokenizer_config.json": ExpectedFileIdentity(
                size_bytes=config_file.stat().st_size,
                sha256=hashlib.sha256(config_file.read_bytes()).hexdigest(),
            ),
        },
        backend=FakeBackend(),
        tokenizer_loader=load_tokenizer,
        buckets=(8, 16),
    )

    detector.load()

    assert replacement_happened is True
    assert consumed_tokenizer == expected_tokenizer
    assert consumed_modes == (0o700, 0o700, 0o400)
    assert tokenizer_file.read_bytes() == expected_tokenizer
    assert consumed_path is not None
    assert not consumed_path.exists()


def test_detector_rejects_undeclared_tokenizer_files_before_importing_transformers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tokenizer_files = {
        filename: ExpectedFileIdentity(
            size_bytes=tokenizer_path.joinpath(filename).stat().st_size,
            sha256=hashlib.sha256(tokenizer_path.joinpath(filename).read_bytes()).hexdigest(),
        )
        for filename in ("tokenizer.json", "tokenizer_config.json")
    }
    (tokenizer_path / "added_tokens.json").write_text("{}", encoding="utf-8")
    backend = FakeBackend()
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=backend,
    )
    real_import = builtins.__import__
    transformers_imported = False

    def track_transformers(
        name: str,
        # reason: this replaces `builtins.__import__`, and these parameter names are that builtin's documented
        # reason: signature; renaming them would break any caller that imports by keyword.
        globals: Mapping[str, object] | None = None,  # ruff: ignore[builtin-argument-shadowing]
        locals: Mapping[str, object] | None = None,  # ruff: ignore[builtin-argument-shadowing]
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        nonlocal transformers_imported
        if name == "transformers" or name.startswith("transformers."):
            transformers_imported = True
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", track_transformers)

    with pytest.raises(RuntimeError, match="tokenizer directory identity mismatch"):
        detector.load()

    assert transformers_imported is False
    assert backend.load_count == 0


def test_detector_rejects_same_size_tokenizer_digest_mismatch_before_loading(
    tmp_path: Path,
) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    tokenizer_file = tokenizer_path / "tokenizer.json"
    tokenizer_file.write_bytes(b"x" * tokenizer_file.stat().st_size)
    backend = FakeBackend()
    tokenizer_loaded = False

    def load_tokenizer(_: Path) -> FakeTokenizer:
        nonlocal tokenizer_loaded
        tokenizer_loaded = True
        return FakeTokenizer()

    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=backend,
        tokenizer_loader=load_tokenizer,
    )

    with pytest.raises(RuntimeError, match=r"tokenizer/tokenizer\.json SHA-256 mismatch"):
        detector.load()

    assert tokenizer_loaded is False
    assert backend.load_count == 0


def test_detector_rejects_oversized_model_before_reading_or_leaving_a_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "oversized-model.onnx"
    model_path.write_bytes(b"x" * (4 * 1024 * 1024))
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    original_read = os.read
    bytes_read = 0

    def track_read(descriptor: int, size: int) -> bytes:
        nonlocal bytes_read
        chunk = original_read(descriptor, size)
        bytes_read += len(chunk)
        return chunk

    monkeypatch.setattr(os, "read", track_read)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    backend = FakeBackend()
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=1,
        expected_model_sha256=hashlib.sha256(b"x").hexdigest(),
        expected_tokenizer_files=tokenizer_files,
        backend=backend,
        tokenizer_loader=lambda _: FakeTokenizer(),
    )

    with pytest.raises(
        RuntimeError,
        match=r"model size mismatch: expected=1 actual=4194304",
    ):
        detector.load()

    assert bytes_read == 0
    assert backend.load_count == 0
    assert not tuple(tmp_path.glob("meddies-pii-runtime-*"))


def test_detector_bounds_copy_when_open_model_grows_after_fstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "growing-model.onnx"
    model_path.write_bytes(b"x")
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    original_open = os.open
    original_read = os.read
    model_descriptor: int | None = None
    model_grew = False
    model_bytes_read = 0

    def track_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal model_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None and Path(os.fsdecode(path)) == model_path:
            model_descriptor = descriptor
        return descriptor

    def grow_before_first_read(descriptor: int, size: int) -> bytes:
        nonlocal model_bytes_read, model_grew
        if descriptor == model_descriptor and not model_grew:
            with model_path.open("ab") as stream:
                stream.write(b"y" * (4 * 1024 * 1024))
            model_grew = True
        chunk = original_read(descriptor, size)
        if descriptor == model_descriptor:
            model_bytes_read += len(chunk)
        return chunk

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "read", grow_before_first_read)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    backend = FakeBackend()
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=1,
        expected_model_sha256=hashlib.sha256(b"x").hexdigest(),
        expected_tokenizer_files=tokenizer_files,
        backend=backend,
        tokenizer_loader=lambda _: FakeTokenizer(),
    )

    with pytest.raises(RuntimeError, match=r"model size mismatch: expected=1 actual=2"):
        detector.load()

    assert model_grew is True
    assert model_bytes_read == 2
    assert backend.load_count == 0
    assert not tuple(tmp_path.glob("meddies-pii-runtime-*"))


def test_detector_rejects_symlinked_model_and_tokenizer_sources(
    tmp_path: Path,
) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    model_link = tmp_path / "model-link.onnx"
    model_link.symlink_to(model_path)

    with pytest.raises(RuntimeError, match="model file is unreadable"):
        BioesSpanDetector(
            model_path=model_link,
            tokenizer_path=tokenizer_path,
            expected_model_size_bytes=model_path.stat().st_size,
            expected_model_sha256=expected_sha,
            expected_tokenizer_files=tokenizer_files,
            backend=FakeBackend(),
            tokenizer_loader=lambda _: FakeTokenizer(),
        ).load()

    tokenizer_file = tokenizer_path / "tokenizer.json"
    external_tokenizer = tmp_path / "external-tokenizer.json"
    tokenizer_file.replace(external_tokenizer)
    tokenizer_file.symlink_to(external_tokenizer)

    with pytest.raises(RuntimeError, match=r"tokenizer/tokenizer\.json file is unreadable"):
        BioesSpanDetector(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            expected_model_size_bytes=model_path.stat().st_size,
            expected_model_sha256=expected_sha,
            expected_tokenizer_files=tokenizer_files,
            backend=FakeBackend(),
            tokenizer_loader=lambda _: FakeTokenizer(),
        ).load()


def test_detector_rejects_fifo_model_without_blocking(tmp_path: Path) -> None:
    if os.name != "posix" or not hasattr(os, "mkfifo"):
        pytest.skip("FIFO probe requires POSIX")
    model_path = tmp_path / "model-fifo.onnx"
    os.mkfifo(model_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="model source is not a regular file"):
        BioesSpanDetector(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            expected_model_size_bytes=1,
            expected_model_sha256=hashlib.sha256(b"x").hexdigest(),
            expected_tokenizer_files=tokenizer_files,
            backend=FakeBackend(),
            tokenizer_loader=lambda _: FakeTokenizer(),
        ).load()


def test_detector_returns_span_bucket_and_truncation_metadata(tmp_path: Path) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=FakeBackend(),
        tokenizer_loader=lambda _: FakeTokenizer(),
        buckets=(8, 16),
    )
    detector.load()

    [result] = detector.detect(["Maya"])

    assert result.bucket == 8
    assert result.num_tokens == 4
    assert result.truncated is False
    assert [(span.start, span.end, span.text, span.label) for span in result.spans] == [(0, 1, "M", "human_name")]


def test_numpy_viterbi_matches_canonical_torch_decoder() -> None:
    rng = np.random.default_rng(20260713)
    label_space = build_bioes_label_space(ENTITY_LABELS)
    id_to_label = dict(enumerate(label_space))
    logits = rng.normal(size=(17, len(label_space))).astype(np.float32)
    offsets = [(0, 0), *((index, index + 1) for index in range(15)), (0, 0)]

    expected = viterbi_decode_logits(torch.from_numpy(logits), id_to_label, offsets)
    observed = viterbi_decode_numpy(logits, id_to_label, offsets)

    assert observed == expected


def test_pdf_detector_does_not_import_torch_at_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=FakeBackend(),
        tokenizer_loader=lambda _: FakeTokenizer(),
        buckets=(8, 16),
    )
    detector.load()
    real_import = builtins.__import__

    def reject_torch(
        name: str,
        # reason: this replaces `builtins.__import__`, and these parameter names are that builtin's documented
        # reason: signature; renaming them would break any caller that imports by keyword.
        globals: Mapping[str, object] | None = None,  # ruff: ignore[builtin-argument-shadowing]
        locals: Mapping[str, object] | None = None,  # ruff: ignore[builtin-argument-shadowing]
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "torch" or name.startswith("torch."):
            msg = "torch is intentionally unavailable"
            raise ModuleNotFoundError(msg)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_torch)

    [result] = detector.detect(["Maya"])

    assert result.spans


def test_detector_surfaces_tokenizer_overflow(tmp_path: Path) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=FakeBackend(),
        tokenizer_loader=lambda _: FakeTokenizer(overflow=True),
        buckets=(8,),
    )
    detector.load()

    [result] = detector.detect(["overflow"])

    assert result.truncated is True


def test_detector_rejects_hash_and_output_shape_mismatches(tmp_path: Path) -> None:
    model_path, expected_sha = _model_file(tmp_path)
    tokenizer_path, tokenizer_files = _tokenizer_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="model SHA-256 mismatch"):
        BioesSpanDetector(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            expected_model_size_bytes=model_path.stat().st_size,
            expected_model_sha256="0" * 64,
            expected_tokenizer_files=tokenizer_files,
            backend=FakeBackend(),
            tokenizer_loader=lambda _: FakeTokenizer(),
        ).load()

    detector = BioesSpanDetector(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        expected_model_size_bytes=model_path.stat().st_size,
        expected_model_sha256=expected_sha,
        expected_tokenizer_files=tokenizer_files,
        backend=FakeBackend(bad_shape=True),
        tokenizer_loader=lambda _: FakeTokenizer(),
    )
    detector.load()
    with pytest.raises(RuntimeError, match="backend logits shape mismatch"):
        detector.detect(["Maya"])


def test_onnx_backend_uses_one_cpu_session_and_int64_inputs(tmp_path: Path) -> None:
    model_path, _ = _model_file(tmp_path)
    calls: list[tuple[Path, int]] = []

    class Session:
        # reason: this mirrors `onnxruntime`'s `InferenceSession.run`, which the code under test calls on the
        # reason: session instance the factory returns, so the bound-method form is the API being stood in for.
        def run(self, output_names: None, inputs: Mapping[str, object]) -> list[object]:  # ruff: ignore[no-self-use]
            assert output_names is None
            assert np.asarray(inputs["input_ids"]).dtype == np.int64
            assert np.asarray(inputs["attention_mask"]).shape == (1, 4)
            return [np.zeros((1, 4, 37), dtype=np.float32)]

    def factory(path: Path, threads: int) -> Session:
        calls.append((path, threads))
        return Session()

    backend = OnnxRuntimeBackend(threads=4, session_factory=factory)
    backend.load(model_path)

    output = backend.infer([1, 2, 0, 0], [1, 1, 0, 0], bucket=4)

    assert calls == [(model_path, 4)]
    assert np.asarray(output).shape == (1, 4, 37)


def test_openvino_backend_pins_latency_threads_and_one_stream(tmp_path: Path) -> None:
    model_path, _ = _model_file(tmp_path)
    calls: list[tuple[Path, int]] = []

    class Compiled:
        def __call__(self, inputs: Mapping[str, object]) -> dict[object, object]:
            assert np.asarray(inputs["input_ids"]).dtype == np.int64
            return {"logits": np.zeros((1, 4, 37), dtype=np.float32)}

    def factory(path: Path, threads: int) -> tuple[Compiled, str]:
        calls.append((path, threads))
        return Compiled(), "logits"

    backend = OpenVinoBackend(threads=6, compiled_factory=factory)
    backend.load(model_path)

    output = backend.infer([1, 2, 0, 0], [1, 1, 0, 0], bucket=4)

    assert calls == [(model_path, 6)]
    assert np.asarray(output).shape == (1, 4, 37)
