from __future__ import annotations

# reason: verifier tools run as list-form argv; executable paths come from the explicit operator-owned toolchain config.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from pathlib import Path

VerificationGate = Literal[
    "writer_safety",
    "fresh_output",
    "structure",
    "expanded_objects",
    "risky_objects",
    "text_extraction",
    "reopen_render",
    "ocr",
]
ToolStatus = Literal["completed", "timeout", "unavailable"]


@dataclass(frozen=True, slots=True)
class ToolRun:
    tool: str
    status: ToolStatus
    returncode: int | None
    stdout: bytes
    stderr: bytes
    duration_ms: float


class ToolRunner(Protocol):
    def run(self, args: tuple[str, ...], *, timeout_seconds: float) -> ToolRun: ...


class SubprocessToolRunner:
    # reason: this stateless implementation keeps the bound method shape required by the injectable ToolRunner protocol.
    def run(self, args: tuple[str, ...], *, timeout_seconds: float) -> ToolRun:  # ruff: ignore[no-self-use]
        started = time.perf_counter()
        try:
            # reason: args come from the verifier's operator-owned toolchain plus fixed flags and artifact paths; no shell.
            completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                args,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return ToolRun(
                tool=args[0],
                status="timeout",
                returncode=None,
                stdout=_as_bytes(error.stdout),
                stderr=_as_bytes(error.stderr),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except FileNotFoundError:
            return ToolRun(
                tool=args[0],
                status="unavailable",
                returncode=None,
                stdout=b"",
                stderr=b"",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return ToolRun(
            tool=args[0],
            status="completed",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class ToolObservation:
    gate: VerificationGate
    tool: str
    status: ToolStatus
    returncode: int | None
    stdout_bytes: int
    stderr_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    duration_ms: float


@dataclass(frozen=True, slots=True)
class VerificationFinding:
    gate: VerificationGate
    passed: bool
    code: str
    detail: str
    matched_value_sha256: tuple[str, ...] = ()
    risky_markers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationReport:
    output_path: Path
    output_sha256: str
    passed: bool
    findings: tuple[VerificationFinding, ...]
    observations: tuple[ToolObservation, ...]

    def finding(self, gate: VerificationGate) -> VerificationFinding:
        for finding in self.findings:
            if finding.gate == gate:
                return finding
        raise KeyError(gate)


@dataclass(frozen=True, slots=True)
class VerificationToolchain:
    qpdf: str = "qpdf"
    pdftotext: str = "pdftotext"
    pdftoppm: str = "pdftoppm"
    tesseract: str = "tesseract"
    timeout_seconds: float = 10.0
    render_dpi: int = 150
    ocr_languages: str = "eng+vie"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            msg = "verification timeout must be positive"
            raise ValueError(msg)
        if self.render_dpi <= 0:
            msg = "verification render DPI must be positive"
            raise ValueError(msg)
