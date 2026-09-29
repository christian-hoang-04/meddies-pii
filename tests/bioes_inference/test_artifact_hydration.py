from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from meddies_pii.bioes_inference.artifacts import (
    PUBLIC_Q8_ARTIFACT_SPEC,
    ArtifactHydrationError,
    PinnedArtifactSpec,
    PinnedFileSpec,
    create_default_detector,
    hydrate_artifact,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _small_spec() -> tuple[PinnedArtifactSpec, dict[str, bytes]]:
    payloads = {
        "onnx/model.onnx": b"synthetic-model",
        "tokenizer.json": b'{"synthetic":true}',
        "tokenizer_config.json": b'{"model_max_length":8192}',
    }
    files = tuple(
        PinnedFileSpec(
            relative_path=relative_path,
            size_bytes=len(payload),
            sha256=_sha256(payload),
            role="model" if relative_path.endswith(".onnx") else "tokenizer",
        )
        for relative_path, payload in payloads.items()
    )
    return (
        PinnedArtifactSpec(
            artifact_id="synthetic_q8",
            repo_id="Meddies/synthetic-public",
            revision="1" * 40,
            files=files,
        ),
        payloads,
    )


def test_public_q8_spec_pins_model_and_minimum_tokenizer_files() -> None:
    spec = PUBLIC_Q8_ARTIFACT_SPEC
    files = {file.relative_path: file for file in spec.files}

    assert spec.repo_id == "Meddies/meddies-pii-v2-onnx"
    assert spec.revision == "47b041b16ec3ecdf75871d42e38951f7f6d65a31"
    assert set(files) == {
        "onnx/model.q8.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert files["onnx/model.q8.onnx"].size_bytes == 592_471_465
    assert files["onnx/model.q8.onnx"].sha256 == "896b7cc8a9621c9b11e54a7893071091acf05c0055a2e7df7e35220587c73fe3"
    assert files["tokenizer.json"].size_bytes == 4_733_016
    assert files["tokenizer.json"].sha256 == "4905ab82b2cfc25e0c88adc8f4eeffe759c57c5626312b30b0aaeaf8ad3379bc"
    assert files["tokenizer_config.json"].size_bytes == 526
    assert files["tokenizer_config.json"].sha256 == "1c02b0dd850ea012fa8824ae9facdf4cfb367463d3b33134e567b9f434cc9241"


def test_hydration_downloads_only_declared_files_at_the_exact_revision(
    tmp_path: Path,
) -> None:
    spec, payloads = _small_spec()
    calls: list[tuple[str, str, str, str]] = []

    def fake_download(*, repo_id: str, filename: str, revision: str, local_dir: str) -> str:
        calls.append((repo_id, filename, revision, local_dir))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        return str(path)

    root = tmp_path / "hydrated"
    hydrated = hydrate_artifact(spec, root, downloader=fake_download)
    artifact_root = root / spec.artifact_id / spec.revision
    download_root = artifact_root / "files"
    tokenizer_root = artifact_root / "tokenizer"

    assert tuple(call[1] for call in calls) == tuple(file.relative_path for file in spec.files)
    assert all(call[0] == spec.repo_id for call in calls)
    assert all(call[2] == spec.revision for call in calls)
    assert all(call[3] == str(download_root) for call in calls)
    assert hydrated.repo_id == spec.repo_id
    assert hydrated.revision == spec.revision
    assert hydrated.root == artifact_root
    assert hydrated.model_path == download_root / "onnx/model.onnx"
    assert hydrated.tokenizer_root == tokenizer_root
    assert {path.name for path in tokenizer_root.iterdir()} == {
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert tuple(file.sha256 for file in hydrated.files) == tuple(file.sha256 for file in spec.files)


def test_hydration_rebuilds_an_exact_tokenizer_directory_idempotently(
    tmp_path: Path,
) -> None:
    spec, payloads = _small_spec()

    def fake_download(*, repo_id: str, filename: str, revision: str, local_dir: str) -> str:
        del repo_id, revision
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        return str(path)

    root = tmp_path / "hydrated"
    first = hydrate_artifact(spec, root, downloader=fake_download)
    (first.tokenizer_root / "added_tokens.json").write_text('{"unexpected": true}', encoding="utf-8")

    second = hydrate_artifact(spec, root, downloader=fake_download)

    assert first.tokenizer_root == second.tokenizer_root
    assert {path.relative_to(second.tokenizer_root).as_posix() for path in second.tokenizer_root.rglob("*")} == {
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert second.tokenizer_root.joinpath("tokenizer.json").read_bytes() == payloads["tokenizer.json"]


def test_hydration_fails_closed_on_size_before_accepting_identity(
    tmp_path: Path,
) -> None:
    spec, payloads = _small_spec()

    def wrong_size(*, repo_id: str, filename: str, revision: str, local_dir: str) -> str:
        del repo_id, revision
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = payloads[filename]
        path.write_bytes(payload + b"x" if filename.endswith(".onnx") else payload)
        return str(path)

    with pytest.raises(ArtifactHydrationError) as captured:
        hydrate_artifact(spec, tmp_path / "hydrated", downloader=wrong_size)

    assert captured.value.code == "size_mismatch"
    assert captured.value.relative_path == "onnx/model.onnx"


def test_hydration_fails_closed_on_same_size_digest_mismatch(tmp_path: Path) -> None:
    spec, payloads = _small_spec()

    def wrong_digest(*, repo_id: str, filename: str, revision: str, local_dir: str) -> str:
        del repo_id, revision
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = payloads[filename]
        path.write_bytes(b"x" * len(payload) if filename.endswith(".onnx") else payload)
        return str(path)

    with pytest.raises(ArtifactHydrationError) as captured:
        hydrate_artifact(spec, tmp_path / "hydrated", downloader=wrong_digest)

    assert captured.value.code == "digest_mismatch"
    assert captured.value.observed_sha256 is not None
    assert captured.value.observed_sha256 != captured.value.expected_sha256


def test_download_failures_never_expose_secret_values(tmp_path: Path) -> None:
    spec, _ = _small_spec()
    canary = "hf_secret_value_must_never_escape"

    def fails(*, repo_id: str, filename: str, revision: str, local_dir: str) -> str:
        del repo_id, filename, revision, local_dir
        raise RuntimeError(canary)

    with pytest.raises(ArtifactHydrationError) as captured:
        hydrate_artifact(spec, tmp_path / "hydrated", downloader=fails)

    failure = captured.value
    assert failure.code == "download_failed"
    assert failure.cause_digest == _sha256(f"RuntimeError:{canary}".encode())
    assert canary not in str(failure)
    assert canary not in repr(failure)


def test_specs_reject_mutable_revisions_and_path_traversal() -> None:
    file = PinnedFileSpec(
        relative_path="tokenizer.json",
        size_bytes=2,
        sha256=_sha256(b"{}"),
        role="tokenizer",
    )

    with pytest.raises(ValueError, match="40-character lowercase commit"):
        PinnedArtifactSpec(
            artifact_id="mutable",
            repo_id="Meddies/model",
            revision="main",
            files=(file,),
        )
    with pytest.raises(ValueError, match="relative repository path"):
        PinnedFileSpec(
            relative_path="../tokenizer.json",
            size_bytes=2,
            sha256=_sha256(b"{}"),
            role="tokenizer",
        )


def test_hydration_rejects_scoped_cache_symlink_before_writing_outside_root(
    tmp_path: Path,
) -> None:
    spec, _ = _small_spec()
    root = tmp_path / "hydrated"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / spec.artifact_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactHydrationError) as captured:
        hydrate_artifact(spec, root, downloader=lambda **_: "unused")

    assert captured.value.code == "path_escape"
    assert not (outside / spec.revision).exists()


def test_default_detector_factory_uses_verified_artifact_and_openvino_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.q8.onnx"
    model.write_bytes(b"synthetic")

    hydrate_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_hydrate(*args: object, **kwargs: object) -> object:
        hydrate_calls.append((args, kwargs))
        return SimpleNamespace(model_path=model, tokenizer_root=tmp_path)

    monkeypatch.setattr(
        "meddies_pii.bioes_inference.artifacts.hydrate_artifact",
        fake_hydrate,
    )

    detector = create_default_detector(
        tmp_path,
        threads=6,
    )

    assert hydrate_calls == [((PUBLIC_Q8_ARTIFACT_SPEC, tmp_path), {})]

    assert detector.model_path == model
    assert detector.tokenizer_path == tmp_path
    assert detector.expected_model_size_bytes == 592_471_465
    assert set(detector.expected_tokenizer_files) == {
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert (
        detector.expected_tokenizer_files["tokenizer.json"].sha256
        == "4905ab82b2cfc25e0c88adc8f4eeffe759c57c5626312b30b0aaeaf8ad3379bc"
    )
    assert detector.backend.name == "openvino-cpu"
