from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import pytest

from meddies_pii.bioes_inference.artifact_staging import (
    copy_verified_file,
    copy_verified_tokenizer_directory,
    validated_tokenizer_manifest,
)
from meddies_pii.bioes_inference.contracts import ExpectedFileIdentity

if TYPE_CHECKING:
    from pathlib import Path


def _identity(payload: bytes) -> ExpectedFileIdentity:
    return ExpectedFileIdentity(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("relative_path", ["../tokenizer.json", "/tokenizer.json", "a\\b"])
def test_tokenizer_manifest_rejects_path_escape_forms(relative_path: str) -> None:
    with pytest.raises(ValueError, match="normalized relative paths"):
        validated_tokenizer_manifest({relative_path: _identity(b"{}")})


def test_tokenizer_manifest_rejects_overlapping_entries() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        validated_tokenizer_manifest({
            "vocab": _identity(b"x"),
            "vocab/merges.txt": _identity(b"y"),
        })


def test_copy_verified_file_sets_read_only_permissions_after_identity_check(
    tmp_path: Path,
) -> None:
    payload = b'{"tokenizer":"synthetic"}'
    source = tmp_path / "tokenizer.json"
    destination = tmp_path / "staged.json"
    source.write_bytes(payload)

    copy_verified_file(
        source,
        destination,
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        identity_name="tokenizer",
    )

    assert destination.read_bytes() == payload
    if os.name == "posix":
        assert destination.stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize(
    ("expected_size_bytes", "expected_sha256", "message"),
    [
        (3, hashlib.sha256(b"abc").hexdigest(), "size mismatch"),
        (4, hashlib.sha256(b"wxyz").hexdigest(), "SHA-256 mismatch"),
    ],
)
def test_copy_verified_file_rejects_size_and_digest_mismatches(
    tmp_path: Path,
    expected_size_bytes: int,
    expected_sha256: str,
    message: str,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"abcd")

    with pytest.raises(RuntimeError, match=message):
        copy_verified_file(
            source,
            destination,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            identity_name="model",
        )


def test_copy_verified_file_rejects_symlink_source(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"safe")
    source = tmp_path / "source-link.bin"
    source.symlink_to(target)

    with pytest.raises(RuntimeError, match="model file is unreadable"):
        copy_verified_file(
            source,
            tmp_path / "destination.bin",
            expected_size_bytes=4,
            expected_sha256=hashlib.sha256(b"safe").hexdigest(),
            identity_name="model",
        )


def test_copy_verified_tokenizer_directory_requires_an_exact_non_symlink_tree(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tokenizer-source"
    source.mkdir()
    nested = source / "nested"
    nested.mkdir()
    payloads = {"tokenizer.json": b"{}", "nested/merges.txt": b"a b"}
    for relative_path, payload in payloads.items():
        (source / relative_path).write_bytes(payload)
    manifest = {path: _identity(payload) for path, payload in payloads.items()}
    destination = tmp_path / "tokenizer-destination"
    destination.mkdir(mode=0o700)

    copy_verified_tokenizer_directory(source, destination, manifest)

    assert {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()} == set(payloads)

    (source / "unexpected.json").write_text("extra", encoding="utf-8")
    second_destination = tmp_path / "second"
    second_destination.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        copy_verified_tokenizer_directory(source, second_destination, manifest)


def test_copy_verified_tokenizer_directory_rejects_a_symlinked_member(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tokenizer-source"
    source.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    (source / "tokenizer.json").symlink_to(outside)
    manifest = {"tokenizer.json": _identity(b"{}")}

    destination = tmp_path / "destination"
    destination.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match=r"tokenizer/tokenizer\.json file is unreadable"):
        copy_verified_tokenizer_directory(source, destination, manifest)


def test_tokenizer_manifest_requires_entries_with_expected_identity_objects() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validated_tokenizer_manifest({})
    with pytest.raises(TypeError, match="ExpectedFileIdentity"):
        # reason: a bare object() is passed precisely to assert the manifest validator refuses a value that is
        # reason: not an ExpectedFileIdentity, so the type error IS the assertion this test makes.
        validated_tokenizer_manifest({"tokenizer.json": object()})  # ty: ignore[invalid-argument-type]


def test_copy_verified_file_rejects_missing_and_nonregular_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="missing model file"):
        copy_verified_file(
            tmp_path / "missing.bin",
            tmp_path / "destination.bin",
            expected_size_bytes=1,
            expected_sha256=hashlib.sha256(b"x").hexdigest(),
            identity_name="model",
        )

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="model source is not a regular file"):
        copy_verified_file(
            directory,
            tmp_path / "destination-directory.bin",
            expected_size_bytes=1,
            expected_sha256=hashlib.sha256(b"x").hexdigest(),
            identity_name="model",
        )


def test_copy_verified_tokenizer_directory_rejects_missing_source_and_file_instead_of_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir(mode=0o700)
    manifest = {"tokenizer.json": _identity(b"{}")}

    with pytest.raises(FileNotFoundError, match="missing tokenizer directory"):
        copy_verified_tokenizer_directory(tmp_path / "missing", destination, manifest)

    source_file = tmp_path / "not-a-directory"
    source_file.write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="tokenizer directory is unreadable"):
        copy_verified_tokenizer_directory(source_file, destination, manifest)


def test_copy_verified_tokenizer_directory_rejects_file_where_manifest_requires_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").write_bytes(b"not-a-directory")
    destination = tmp_path / "destination"
    destination.mkdir(mode=0o700)
    manifest = {"nested/tokenizer.json": _identity(b"{}")}

    with pytest.raises(RuntimeError, match="tokenizer directory is unreadable"):
        copy_verified_tokenizer_directory(source, destination, manifest)


def test_copy_verified_file_refuses_to_replace_an_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"safe")
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        copy_verified_file(
            source,
            destination,
            expected_size_bytes=4,
            expected_sha256=hashlib.sha256(b"safe").hexdigest(),
            identity_name="model",
        )

    assert destination.read_bytes() == b"existing"
