from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar, Protocol, cast

import pytest

from meddies_pii.eval_baseline.adapters.lfm25_pii import (
    MODEL_ID,
    MODEL_REVISION,
    _snapshot_file,
    load_pinned_space_detector,
)


class _FakeModel:
    def __init__(self) -> None:
        self.moves: list[dict[str, object]] = []

    def to(self, **kwargs: object) -> _FakeModel:
        self.moves.append(kwargs)
        return self


class _FakeVendorModel(Protocol):
    moves: list[dict[str, object]]


class _FakeVendorDetector(Protocol):
    """What the fake vendor ``PiiDetector`` below exposes, named so its assertions can be checked.

    The fake is written to disk as source TEXT and imported at runtime, so its class does not
    exist when the type checker runs; the loader is declared to return the production detector,
    which carries none of these attributes. Naming the double's surface here states what the
    test expects it to expose, rather than silencing each read with its own disable.

    ``auth_priority_types`` is not the fake's own -- the loader assigns it onto whatever
    detector it built, after reading the decoder's ``_AUTH_TYPES``.
    """

    instances: ClassVar[int]
    imported_marker: str
    model_id_at_init: str
    decoder_at_init: str
    model: _FakeVendorModel
    auth_priority_types: frozenset[str]


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        float32="float32",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
    )


def _write_fake_space_source(path: Path) -> None:
    path.write_text(
        """
MODEL_ID = "remote-model"

def hf_hub_download(*_args, **_kwargs):
    raise AssertionError("runtime must replace the vendor downloader")

class FakeModel:
    def __init__(self):
        self.moves = []

    def to(self, **kwargs):
        self.moves.append(kwargs)
        return self

class FakeDecoder:
    _AUTH_TYPES = {"credential.api_key"}

class PiiDetector:
    instances = 0

    def __init__(self):
        type(self).instances += 1
        self.model_id_at_init = MODEL_ID
        self.decoder_at_init = hf_hub_download(
            MODEL_ID, "nested/pii_hybrid_decode.py", token="unused"
        )
        self.model = FakeModel()
        self.hd = FakeDecoder()

    def detect(self, text):
        return {"spans": []}
""".lstrip(),
        encoding="utf-8",
    )


def _load_runner_module() -> ModuleType:
    runner = Path(__file__).resolve().parents[3] / "scripts" / "ops" / "run_lfm25_pii_baseline.py"
    spec = importlib.util.spec_from_file_location("lfm25_pii_baseline_test", runner)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_downloads_only_the_pinned_model_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def snapshot_download(model_id: str, *, revision: str, cache_dir: str) -> str:
        calls.append((model_id, revision, cache_dir))
        return str(tmp_path / "snapshot")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    runner = _load_runner_module()

    assert runner._download_model_snapshot() == tmp_path / "snapshot"
    assert calls == [(MODEL_ID, MODEL_REVISION, "/cache/huggingface")]


def test_runner_defaults_to_meddies_workspace_volumes_and_exposes_bounded_smoke() -> None:
    runner = _load_runner_module()

    assert runner.BASELINE_VOLUME_NAME == "meddies-pii-baseline-eval-lfm25-pii"
    assert runner.CACHE_VOLUME_NAME == "lfm25-pii-eval-hf-cache"
    assert runner.MODEL_NAME == "lfm25-pii"
    assert callable(runner.smoke)
    assert callable(runner.run_parity_smoke)
    assert any("sk-" in text for text in runner.PARITY_SMOKE_TEXTS)


def test_runner_pins_a_debian_compiler_before_python_packages() -> None:
    runner = _load_runner_module()
    runner_path = runner.__file__
    assert runner_path is not None
    source = Path(runner_path).read_text(encoding="utf-8")

    assert runner.COMPILER_APT_PACKAGES == ("gcc=4:12.2.0-3",)
    assert "use_pinned_debian_snapshot(" in source
    assert (
        source.index("use_pinned_debian_snapshot(")
        < source.index(".apt_install(*COMPILER_APT_PACKAGES)")
        < source.index(".pip_install(")
        < source.index('.add_local_dir("src"')
    )


def _pinned_space_reference_available() -> bool:
    return any(
        (ancestor / "meddies-pii-context" / "references" / "liquidai-pii-detection-space").is_dir()
        for ancestor in Path(__file__).resolve().parents
    )


@pytest.mark.skipif(
    not _pinned_space_reference_available(),
    reason="verifies the real pinned reference clone, absent without the context repo",
)
def test_runner_finds_the_pinned_space_reference_from_a_nested_worktree() -> None:
    runner = _load_runner_module()

    reference = runner._space_reference_local()

    assert reference.is_dir()
    source = reference / "pii.py"
    assert source.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == runner.SPACE_SOURCE_SHA256


def test_runner_uses_the_mounted_space_reference_inside_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    monkeypatch.setattr(runner.modal, "is_local", lambda: False)

    assert runner._space_reference_local() == Path(runner.SPACE_REFERENCE_MOUNT)


def test_snapshot_file_accepts_huggingface_snapshot_symlink_into_model_blobs(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "models--liquidai--pii"
    snapshot = cache_root / "snapshots" / "pinned-revision"
    blobs = cache_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "decoder-digest"
    blob.write_text("decoder", encoding="utf-8")
    (snapshot / "pii_hybrid_decode.py").symlink_to("../../blobs/decoder-digest")

    assert _snapshot_file(snapshot, "pii_hybrid_decode.py") == str(snapshot / "pii_hybrid_decode.py")


def test_snapshot_file_preserves_decoder_import_name_for_snapshot_symlink(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "models--liquidai--pii"
    snapshot = cache_root / "snapshots" / "pinned-revision"
    blobs = cache_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "decoder-digest"
    blob.write_text('MARKER = "decoder-loaded"\n', encoding="utf-8")
    decoder_link = snapshot / "pii_hybrid_decode.py"
    decoder_link.symlink_to("../../blobs/decoder-digest")

    decoder_path = _snapshot_file(snapshot, "pii_hybrid_decode.py")
    source = tmp_path / "pii.py"
    source.write_text(
        """
import os
import sys

MODEL_ID = "remote-model"

class PiiDetector:
    def __init__(self):
        self.model = type("Model", (), {"to": lambda model, **_kwargs: model})()
        hd_path = hf_hub_download(MODEL_ID, "pii_hybrid_decode.py")
        sys.path.insert(0, os.path.dirname(hd_path))
        import pii_hybrid_decode
        self.imported_marker = pii_hybrid_decode.MARKER
        self.hd = type("Decoder", (), {"_AUTH_TYPES": set()})()

    def detect(self, text):
        return {"spans": []}
""".lstrip(),
        encoding="utf-8",
    )

    try:
        detector: _FakeVendorDetector = cast(
            "_FakeVendorDetector",
            load_pinned_space_detector(
                snapshot_path=snapshot,
                source_path=source,
                space_root=tmp_path,
                space_revision="pinned-space-revision",
                expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                torch_module=_fake_torch(),
            ),
        )
        assert decoder_path == str(decoder_link)
        assert detector.imported_marker == "decoder-loaded"
    finally:
        sys.modules.pop("pii_hybrid_decode", None)
        while str(snapshot) in sys.path:
            sys.path.remove(str(snapshot))


# reason: the absolute path is the escape attempt under test, not a temp file this test writes.
# reason: `_snapshot_file` must refuse it, and the assertion below is that refusal.
@pytest.mark.parametrize(
    "filename",
    ["../../blobs/decoder-digest", "/tmp/decoder"],  # ruff: ignore[hardcoded-temp-file]
)
def test_snapshot_file_rejects_lexical_escape_filenames(
    tmp_path: Path,
    filename: str,
) -> None:
    snapshot = tmp_path / "models--liquidai--pii" / "snapshots" / "pinned-revision"
    snapshot.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Liquid snapshot filename"):
        _snapshot_file(snapshot, filename)


def test_snapshot_file_rejects_symlink_escaping_model_cache(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--liquidai--pii" / "snapshots" / "pinned-revision"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside-decoder.py"
    outside.write_text("decoder", encoding="utf-8")
    (snapshot / "pii_hybrid_decode.py").symlink_to(outside)

    with pytest.raises(RuntimeError, match="Liquid snapshot file escapes model cache"):
        _snapshot_file(snapshot, "pii_hybrid_decode.py")


def test_load_pinned_space_detector_binds_the_fake_vendor_to_one_local_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    decoder = snapshot / "nested" / "pii_hybrid_decode.py"
    decoder.parent.mkdir(parents=True)
    decoder.write_text("decoder", encoding="utf-8")
    source = tmp_path / "pii.py"
    _write_fake_space_source(source)
    fake_torch = _fake_torch()

    detector: _FakeVendorDetector = cast(
        "_FakeVendorDetector",
        load_pinned_space_detector(
            snapshot_path=snapshot,
            source_path=source,
            space_root=tmp_path,
            space_revision="pinned-space-revision",
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            torch_module=fake_torch,
        ),
    )

    assert detector.model_id_at_init == str(snapshot)
    assert detector.decoder_at_init == str(decoder)
    assert detector.model.moves == [{"device": "cuda", "dtype": "float32"}]
    assert detector.auth_priority_types == frozenset({"credential.api_key"})
    assert fake_torch.backends.cuda.matmul.allow_tf32 is True
    assert type(detector).instances == 1


def test_load_pinned_space_detector_fails_closed_when_source_bytes_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pii.py"
    _write_fake_space_source(source)
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source hash changed"):
        load_pinned_space_detector(
            snapshot_path=tmp_path / "snapshot",
            source_path=source,
            space_root=tmp_path,
            space_revision="pinned-space-revision",
            expected_source_sha256=expected_hash,
            torch_module=_fake_torch(),
        )


def test_load_pinned_space_detector_rejects_source_outside_the_space_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "outside.py"
    _write_fake_space_source(source)
    space_root = tmp_path / "space"
    space_root.mkdir()

    with pytest.raises(RuntimeError, match="Space source escapes root"):
        load_pinned_space_detector(
            snapshot_path=tmp_path / "snapshot",
            source_path=source,
            space_root=space_root,
            space_revision="pinned-space-revision",
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            torch_module=_fake_torch(),
        )


# reason: the absolute path is the escape attempt under test, not a temp file this test writes.
# reason: the loader must refuse any vendor file outside the snapshot, and that refusal is asserted below.
@pytest.mark.parametrize(
    "filename",
    ["missing.py", "../outside.py", "/tmp/outside.py"],  # ruff: ignore[hardcoded-temp-file]
)
def test_load_pinned_space_detector_rejects_non_snapshot_vendor_files(
    tmp_path: Path,
    filename: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = tmp_path / "pii.py"
    source.write_text(
        f"""
MODEL_ID = "remote-model"

class PiiDetector:
    def __init__(self):
        self.model = type("Model", (), {{"to": lambda model, **_kwargs: model}})()
        self.requested = hf_hub_download(MODEL_ID, {filename!r})

    def detect(self, text):
        return {{"spans": []}}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Liquid snapshot"):
        load_pinned_space_detector(
            snapshot_path=snapshot,
            source_path=source,
            space_root=tmp_path,
            space_revision="pinned-space-revision",
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            torch_module=_fake_torch(),
        )
