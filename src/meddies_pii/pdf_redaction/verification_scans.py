from __future__ import annotations

import contextlib
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.bioes_inference.file_identity import file_sha256
from meddies_pii.pdf_redaction.verification_types import (
    ToolObservation,
    ToolRun,
    VerificationFinding,
    VerificationGate,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

_RISKY_OBJECT_MARKERS = (
    b"/AA",
    b"/AcroForm",
    b"/EmbeddedFiles",
    b"/Filespec",
    b"/ImportData",
    b"/JS",
    b"/JavaScript",
    b"/Launch",
    b"/Metadata",
    b"/OpenAction",
    b"/RichMedia",
    b"/SubmitForm",
    b"/Thumb",
    b"/URI",
    b"/XFA",
)


def _first_failed_run(runs: Sequence[ToolRun]) -> ToolRun | None:
    return next(
        (run for run in runs if run.status != "completed" or run.returncode != 0),
        None,
    )


def _tool_failure_finding(
    gate: VerificationGate,
    run: ToolRun,
) -> VerificationFinding:
    if run.status == "timeout":
        code = "tool_timeout"
        detail = "verification tool exceeded its bounded timeout"
    elif run.status == "unavailable":
        code = "tool_unavailable"
        detail = "required independent verification tool is unavailable"
    else:
        code = "tool_failed"
        detail = "verification tool returned a nonzero status"
    return VerificationFinding(gate=gate, passed=False, code=code, detail=detail)


def _observation(gate: VerificationGate, run: ToolRun) -> ToolObservation:
    return ToolObservation(
        gate=gate,
        tool=Path(run.tool).name,
        status=run.status,
        returncode=run.returncode,
        stdout_bytes=len(run.stdout),
        stderr_bytes=len(run.stderr),
        stdout_sha256=hashlib.sha256(run.stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(run.stderr).hexdigest(),
        duration_ms=run.duration_ms,
    )


def _matched_hashes(payload: bytes, values: Sequence[str]) -> tuple[str, ...]:
    normalized_payload = unicodedata.normalize("NFC", payload.decode("utf-8", errors="ignore"))
    return tuple(
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in values
        if value.encode("utf-8") in payload or unicodedata.normalize("NFC", value) in normalized_payload
    )


def _matched_string_hashes(
    payload: Iterable[str],
    values: Sequence[str],
) -> tuple[str, ...]:
    matched: set[str] = set()
    for candidate in payload:
        normalized_candidate = unicodedata.normalize("NFC", candidate)
        matched.update(value for value in values if unicodedata.normalize("NFC", value) in normalized_candidate)
    return tuple(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values if value in matched)


def _qpdf_json_string_values(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _qpdf_json_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _qpdf_json_string_values(child)
    elif isinstance(value, str):
        yield from _decode_qpdf_json_string(value)


def _has_json_attachments(inventory: dict[object, object]) -> bool:
    if "attachments" not in inventory or inventory["attachments"] is None:
        return False
    attachments = inventory["attachments"]
    if isinstance(attachments, (dict, list)):
        return bool(attachments)
    return True


def _decode_qpdf_json_string(value: str) -> tuple[str, ...]:
    if value.startswith("u:"):
        return (value[2:],)
    if not value.startswith("b:"):
        return (value,)

    try:
        raw = bytes.fromhex(value[2:])
    except ValueError:
        return (value,)

    candidates: list[str] = []
    if raw.startswith(b"\xfe\xff"):
        _append_decoded(candidates, raw[2:], "utf-16-be")
    elif raw.startswith(b"\xef\xbb\xbf"):
        _append_decoded(candidates, raw[3:], "utf-8")
    _append_decoded(candidates, raw, "utf-8")
    candidates.append(raw.decode("latin-1"))
    return tuple(dict.fromkeys(candidates))


def _append_decoded(candidates: list[str], raw: bytes, encoding: str) -> None:
    with contextlib.suppress(UnicodeDecodeError):
        candidates.append(raw.decode(encoding))


def _present_risky_markers(payload: bytes) -> tuple[str, ...]:
    return tuple(
        marker.decode("ascii")
        for marker in _RISKY_OBJECT_MARKERS
        if re.search(
            rb"(?<![A-Za-z0-9])" + re.escape(marker) + rb"(?![A-Za-z0-9])",
            payload,
        )
    )


def _output_digest(path: Path) -> str:
    try:
        return file_sha256(path) if path.is_file() else ""
    except OSError:
        return ""
