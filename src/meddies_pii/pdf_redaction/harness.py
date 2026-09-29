from __future__ import annotations

import hashlib
import json
import os
import traceback
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

SHA256_HEX_LENGTH = 64


class UnsafeResultPayloadError(RuntimeError):
    """The serialized result contains a forbidden synthetic value."""


def safe_failure_record(
    *,
    stage: str,
    candidate_id: str,
    error: Exception,
    forbidden_values: Sequence[str],
) -> dict[str, object]:
    message = unicodedata.normalize("NFC", str(error))
    for value in forbidden_values:
        if not value:
            msg = "forbidden_values must contain non-empty synthetic values"
            raise ValueError(msg)
        normalized = unicodedata.normalize("NFC", value)
        replacement = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        message = message.replace(normalized, f"<synthetic:{replacement}>")
    return {
        "candidate_id": candidate_id,
        "error_message": message,
        "error_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "error_type": type(error).__name__,
        "stack_frames": [
            {
                "file": Path(frame.filename).name,
                "function": frame.name,
                "line": frame.lineno,
            }
            for frame in traceback.extract_tb(error.__traceback__)[-12:]
        ],
        "stage": stage,
        "status": "failed",
    }


def safe_json_bytes(
    payload: object,
    *,
    forbidden_values: Sequence[str],
) -> bytes:
    forbidden = tuple(unicodedata.normalize("NFC", value) for value in forbidden_values)
    if not forbidden or any(not value for value in forbidden):
        msg = "forbidden_values must contain non-empty synthetic values"
        raise ValueError(msg)
    for text in _walk_strings(payload):
        normalized_text = unicodedata.normalize("NFC", text)
        if any(value in normalized_text for value in forbidden):
            msg = "result payload contains a forbidden synthetic value"
            raise UnsafeResultPayloadError(msg)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (serialized + "\n").encode("utf-8")


def write_json_once(
    destination: Path,
    payload: object,
    *,
    forbidden_values: Sequence[str],
) -> None:
    data = safe_json_bytes(payload, forbidden_values=forbidden_values)
    _write_once(destination, data)


def write_jsonl_once(
    destination: Path,
    rows: Sequence[object],
    *,
    forbidden_values: Sequence[str],
) -> None:
    data = b"".join(safe_json_bytes(row, forbidden_values=forbidden_values) for row in rows)
    _write_once(destination, data)


def copy_verified_once(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    if len(expected_sha256) != SHA256_HEX_LENGTH:
        msg = "expected_sha256 must be a 64-character digest"
        raise ValueError(msg)
    try:
        bytes.fromhex(expected_sha256)
    except ValueError as error:
        msg = "expected_sha256 must be a 64-character digest"
        raise ValueError(msg) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    output_stream = destination.open("xb")
    digest = hashlib.sha256()
    # reason: copy verified's try keeps write with read; splitting would split cleanup from writes.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        with source.open("rb") as input_stream, output_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(block)
                digest.update(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if digest.hexdigest() != expected_sha256:
            msg = "copied output SHA-256 mismatch"
            # reason: the mismatch must raise inside this cleanup boundary so the invalid destination is deleted.
            raise RuntimeError(msg)  # ruff: ignore[raise-within-try]
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _write_once(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = destination.open("xb")
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _walk_strings(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        output: list[str] = []
        for key, item in value.items():
            output.extend(_walk_strings(key))
            output.extend(_walk_strings(item))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        output = []
        for item in value:
            output.extend(_walk_strings(item))
        return output
    return ()
