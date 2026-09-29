from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from meddies_pii.bioes_inference.file_identity import file_sha256
from meddies_pii.pdf_redaction.verification_scans import _output_digest

if TYPE_CHECKING:
    from pathlib import Path


def test_file_sha256_hashes_file_bytes(tmp_path: Path) -> None:
    payload = b"meddies-pdf-identity" * 257
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)

    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_file_sha256_propagates_read_failures(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError):
        file_sha256(missing)


def test_verification_treats_an_unreadable_output_as_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.pdf"
    output.write_bytes(b"%PDF")

    def unreadable(_: Path) -> str:
        msg = "synthetic unreadable output"
        raise PermissionError(msg)

    monkeypatch.setattr(
        "meddies_pii.pdf_redaction.verification_scans.file_sha256",
        unreadable,
    )

    # reason: the empty string is the exact value proven here; `not _output_digest(output)` would also accept
    # reason: `None`, so the equality is what pins the unreadable output to `""` rather than a missing digest.
    assert _output_digest(output) == ""  # ruff: ignore[compare-to-empty-string]
