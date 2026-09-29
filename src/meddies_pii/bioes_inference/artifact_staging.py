from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from meddies_pii.bioes_inference.contracts import ExpectedFileIdentity

if TYPE_CHECKING:
    from collections.abc import Mapping


def validated_tokenizer_manifest(
    files: Mapping[str, ExpectedFileIdentity],
) -> dict[str, ExpectedFileIdentity]:
    if not files:
        msg = "expected_tokenizer_files must not be empty"
        raise ValueError(msg)
    normalized: dict[str, ExpectedFileIdentity] = {}
    paths: list[PurePosixPath] = []
    for relative_path, identity in files.items():
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != relative_path
        ):
            msg = "expected_tokenizer_files keys must be normalized relative paths"
            raise ValueError(msg)
        if not isinstance(identity, ExpectedFileIdentity):
            msg = "expected_tokenizer_files values must be ExpectedFileIdentity"
            raise TypeError(msg)
        normalized[relative_path] = identity
        paths.append(path)
    for path in paths:
        if any(other != path and other.is_relative_to(path) for other in paths):
            msg = "tokenizer manifest paths must not overlap"
            raise ValueError(msg)
    return normalized


def copy_verified_tokenizer_directory(
    source: Path,
    destination: Path,
    manifest: Mapping[str, ExpectedFileIdentity],
) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        msg = "runtime requires O_NOFOLLOW support"
        raise RuntimeError(msg)
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | no_follow
    try:
        source_descriptor = os.open(source, directory_flags)
    except FileNotFoundError:
        msg = "missing tokenizer directory"
        raise FileNotFoundError(msg) from None
    except OSError as error:
        msg = "tokenizer directory is unreadable"
        raise RuntimeError(msg) from error
    try:
        if not stat.S_ISDIR(os.fstat(source_descriptor).st_mode):
            msg = "tokenizer source is not a directory"
            raise RuntimeError(msg)
        expected = {PurePosixPath(path): identity for path, identity in manifest.items()}
        copy_verified_manifest_tree(
            source_descriptor,
            destination,
            expected,
            prefix=(),
        )
    finally:
        os.close(source_descriptor)


def copy_verified_manifest_tree(
    source_directory_descriptor: int,
    destination: Path,
    manifest: Mapping[PurePosixPath, ExpectedFileIdentity],
    *,
    prefix: tuple[str, ...],
) -> None:
    paths = tuple(path for path in manifest if path.parts[: len(prefix)] == prefix)
    expected_entries = {path.parts[len(prefix)] for path in paths}
    try:
        observed_entries = set(os.listdir(source_directory_descriptor))
    except OSError as error:
        msg = "tokenizer directory is unreadable"
        raise RuntimeError(msg) from error
    if observed_entries != expected_entries:
        msg = (
            "tokenizer directory identity mismatch: "
            f"expected_entries={len(expected_entries)} "
            f"observed_entries={len(observed_entries)}"
        )
        raise RuntimeError(
            msg,
        )

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        msg = "runtime requires O_NOFOLLOW support"
        raise RuntimeError(msg)
    for entry in sorted(expected_entries):
        child_prefix = (*prefix, entry)
        child_path = PurePosixPath(*child_prefix)
        identity = manifest.get(child_path)
        if identity is not None:
            source_file_descriptor = _open_regular_file_at(
                source_directory_descriptor,
                entry,
                identity_name=f"tokenizer/{child_path.as_posix()}",
            )
            destination_file = destination / entry
            destination_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                copy_verified_open_file(
                    source_file_descriptor,
                    destination_file,
                    expected_size_bytes=identity.size_bytes,
                    expected_sha256=identity.sha256,
                    identity_name=f"tokenizer/{child_path.as_posix()}",
                )
            finally:
                os.close(source_file_descriptor)
            continue

        directory_descriptor = _open_directory_at(
            source_directory_descriptor,
            entry,
        )
        destination_directory = destination / entry
        destination_directory.mkdir(mode=0o700)
        try:
            copy_verified_manifest_tree(
                directory_descriptor,
                destination_directory,
                manifest,
                prefix=child_prefix,
            )
        finally:
            os.close(directory_descriptor)


def _open_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    identity_name: str,
) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        msg = "runtime requires O_NOFOLLOW support"
        raise RuntimeError(msg)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | no_follow,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        msg = f"{identity_name} file is unreadable"
        raise RuntimeError(msg) from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        msg = f"{identity_name} source is not a regular file"
        raise RuntimeError(msg)
    return descriptor


def _open_directory_at(directory_descriptor: int, name: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        msg = "runtime requires O_NOFOLLOW support"
        raise RuntimeError(msg)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | no_follow,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        msg = "tokenizer directory is unreadable"
        raise RuntimeError(msg) from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        msg = "tokenizer source is not a directory"
        raise RuntimeError(msg)
    return descriptor


def copy_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    identity_name: str,
) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        msg = "runtime requires O_NOFOLLOW support"
        raise RuntimeError(msg)
    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | no_follow
    try:
        source_descriptor = os.open(source, source_flags)
    except FileNotFoundError:
        msg = f"missing {identity_name} file"
        raise FileNotFoundError(msg) from None
    except OSError as error:
        msg = f"{identity_name} file is unreadable"
        raise RuntimeError(msg) from error

    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            msg = f"{identity_name} source is not a regular file"
            raise RuntimeError(msg)
        copy_verified_open_file(
            source_descriptor,
            destination,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            identity_name=identity_name,
        )
    finally:
        os.close(source_descriptor)


def copy_verified_open_file(
    source_descriptor: int,
    destination: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    identity_name: str,
) -> None:
    source_stat = os.fstat(source_descriptor)
    if not stat.S_ISREG(source_stat.st_mode):
        msg = f"{identity_name} source is not a regular file"
        raise RuntimeError(msg)
    if source_stat.st_size != expected_size_bytes:
        msg = f"{identity_name} size mismatch: expected={expected_size_bytes} actual={source_stat.st_size}"
        raise RuntimeError(msg)

    destination_descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        observed_size, observed_sha256 = _stream_copy_and_hash(
            source_descriptor,
            destination_descriptor,
            expected_size_bytes=expected_size_bytes,
        )
        os.fchmod(destination_descriptor, 0o400)
    finally:
        os.close(destination_descriptor)

    if observed_size != expected_size_bytes:
        msg = f"{identity_name} size mismatch: expected={expected_size_bytes} actual={observed_size}"
        raise RuntimeError(msg)
    if observed_sha256 != expected_sha256:
        msg = f"{identity_name} SHA-256 mismatch: expected={expected_sha256[:12]} actual={observed_sha256[:12]}"
        raise RuntimeError(
            msg,
        )


def _stream_copy_and_hash(
    source_descriptor: int,
    destination_descriptor: int,
    *,
    expected_size_bytes: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    while size_bytes < expected_size_bytes:
        chunk = os.read(
            source_descriptor,
            min(8 * 1024 * 1024, expected_size_bytes - size_bytes),
        )
        if not chunk:
            break
        digest.update(chunk)
        size_bytes += len(chunk)
        remaining = memoryview(chunk)
        while remaining:
            written = os.write(destination_descriptor, remaining)
            if written <= 0:
                msg = "staged artifact write made no progress"
                raise OSError(msg)
            remaining = remaining[written:]
    size_bytes += len(os.read(source_descriptor, 1))
    return size_bytes, digest.hexdigest()
