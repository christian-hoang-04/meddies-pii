from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import hashlib
from pathlib import Path

import pytest

import meddies_pii.bioes_inference.artifacts as model_artifacts_module
from meddies_pii.bioes_inference.artifacts import (
    ArtifactHydrationError,
    PinnedArtifactSpec,
    PinnedFileSpec,
    hydrate_artifact,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _spec(*, tokenizer: bool = True) -> tuple[PinnedArtifactSpec, dict[str, bytes]]:
    payloads: dict[str, bytes] = {"model.onnx": b"model"}
    if tokenizer:
        payloads["tokenizer.json"] = b"{}"
    return (
        PinnedArtifactSpec(
            artifact_id="synthetic",
            repo_id="Meddies/synthetic",
            revision="a" * 40,
            files=tuple(
                PinnedFileSpec(
                    relative_path=path,
                    size_bytes=len(payload),
                    sha256=_digest(payload),
                    role="tokenizer" if path.startswith("tokenizer") else "model",
                )
                for path, payload in payloads.items()
            ),
        ),
        payloads,
    )


@pytest.mark.parametrize(
    ("relative_path", "size_bytes", "sha256", "role", "message"),
    [
        ("model.onnx", 0, _digest(b"x"), "model", "size_bytes must be positive"),
        ("model.onnx", 1, "UPPER" * 13, "model", "lowercase SHA-256"),
        ("model.onnx", 1, _digest(b"x"), "weights", "role must be model"),
    ],
)
def test_pinned_file_spec_rejects_invalid_supply_chain_fields(
    relative_path: str,
    size_bytes: int,
    sha256: str,
    role: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PinnedFileSpec(
            relative_path=relative_path,
            size_bytes=size_bytes,
            sha256=sha256,
            # reason: the third parametrize case passes role="weights" precisely to assert the constructor
            # reason: refuses it, so the type error IS the assertion; narrowing to the Literal deletes the case.
            role=role,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("artifact_id", "repo_id", "file_count", "message"),
    [
        ("not safe", "Meddies/model", 1, "artifact_id must be a safe identifier"),
        ("safe", "not/a/slug", 1, "repo_id must be an owner/name identifier"),
        ("safe", "Meddies/model", 0, "artifact spec must declare at least one file"),
        ("safe", "Meddies/model", 2, "artifact file paths must be unique"),
    ],
)
def test_pinned_artifact_spec_rejects_unsafe_identity_and_duplicate_files(
    artifact_id: str,
    repo_id: str,
    file_count: int,
    message: str,
) -> None:
    file = PinnedFileSpec(
        relative_path="model.onnx",
        size_bytes=1,
        sha256=_digest(b"x"),
        role="model",
    )

    with pytest.raises(ValueError, match=message):
        PinnedArtifactSpec(
            revision="b" * 40,
            artifact_id=artifact_id,
            repo_id=repo_id,
            files=(file,) * file_count,
        )


def test_hydration_rejects_downloader_path_escape_and_never_copies_outside_root(
    tmp_path: Path,
) -> None:
    spec, _ = _spec()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"model")

    with pytest.raises(ArtifactHydrationError) as raised:
        hydrate_artifact(spec, tmp_path / "cache", downloader=lambda **_: str(outside))

    assert raised.value.code == "path_escape"
    assert raised.value.relative_path == "model.onnx"


@pytest.mark.parametrize("result_kind", ["missing", "directory"])
def test_hydration_rejects_missing_or_non_file_download_result(tmp_path: Path, result_kind: str) -> None:
    spec, _ = _spec()

    def download(*, local_dir: str, **_: str) -> str:
        path = Path(local_dir) / "model.onnx"
        if result_kind == "directory":
            path.mkdir(parents=True)
        return str(path)

    with pytest.raises(ArtifactHydrationError) as raised:
        hydrate_artifact(spec, tmp_path / result_kind, downloader=download)

    assert raised.value.code == "missing_file"


def test_hydration_classifies_missing_hub_dependency_without_network(
    tmp_path: Path,
) -> None:
    spec, _ = _spec()

    def missing_dependency(**_: str) -> str:
        error = ModuleNotFoundError("not installed")
        error.name = "huggingface_hub"
        raise error

    with pytest.raises(ArtifactHydrationError) as raised:
        hydrate_artifact(spec, tmp_path / "cache", downloader=missing_dependency)

    assert raised.value.code == "dependency_missing"
    assert raised.value.cause_type == "ModuleNotFoundError"


def test_model_only_artifact_has_no_tokenizer_root(tmp_path: Path) -> None:
    spec, payloads = _spec(tokenizer=False)

    def download(*, filename: str, local_dir: str, **_: str) -> str:
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        return str(path)

    hydrated = hydrate_artifact(spec, tmp_path / "cache", downloader=download)

    assert hydrated.model_path.read_bytes() == b"model"
    with pytest.raises(RuntimeError, match="no tokenizer files"):
        _ = hydrated.tokenizer_root


def test_hydration_preserves_file_unreadable_classification_for_hash_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, payloads = _spec(tokenizer=False)

    def download(*, filename: str, local_dir: str, **_: str) -> str:
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        return str(path)

    def unreadable(_: Path) -> str:
        msg = "synthetic unreadable artifact"
        raise PermissionError(msg)

    monkeypatch.setattr(model_artifacts_module, "file_sha256", unreadable)

    with pytest.raises(ArtifactHydrationError) as raised:
        hydrate_artifact(spec, tmp_path / "cache", downloader=download)

    assert raised.value.code == "file_unreadable"
    assert raised.value.relative_path == "model.onnx"


def test_hydrated_identity_requires_exactly_one_model_file(tmp_path: Path) -> None:
    from meddies_pii.bioes_inference.artifacts import HydratedArtifactIdentity

    identity = HydratedArtifactIdentity(
        artifact_id="synthetic",
        repo_id="Meddies/synthetic",
        revision="a" * 40,
        root=tmp_path,
        tokenizer_directory=tmp_path / "tokenizer",
        files=(),
    )

    with pytest.raises(RuntimeError, match="exactly one model"):
        _ = identity.model_path


def test_default_detector_rejects_invalid_thread_count_before_hydration(
    tmp_path: Path,
) -> None:
    from meddies_pii.bioes_inference.artifacts import create_default_detector

    with pytest.raises(ValueError, match="threads must be a positive integer"):
        create_default_detector(tmp_path, threads=0)
