"""PDFium accepts float scale at runtime, but pypdfium2 5.11 annotates it as int."""

from __future__ import annotations

# ruff: file-ignore[invalid-function-name]
# reason: the ReportLab canvas protocol mirrors its upstream camelCase method names exactly for structural typing.
import hashlib
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

from meddies_pii.pdf_redaction.errors import (
    InputDocumentError,
    OutputPathError,
    PdfRedactionError,
    RedactionApplyError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from io import BytesIO

    from PIL.Image import Image as PillowImage

    from meddies_pii.pdf_redaction.contracts import PageRegion

WriterSafety = Literal["destructive", "unsafe_overlay", "reference"]


QUARTER_TURN_DEGREES = 90
HALF_TURN_DEGREES = 180
THREE_QUARTER_TURN_DEGREES = 270


class _PdfiumBitmap(Protocol):
    def to_pil(self) -> PillowImage: ...


class _PdfiumPageWithFloatScale(Protocol):
    def render(self, *, scale: float) -> _PdfiumBitmap: ...

    def get_size(self) -> tuple[float, float]: ...

    def get_rotation(self) -> int: ...


class _PdfiumDocument(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _PdfiumPageWithFloatScale: ...

    def close(self) -> None: ...


# reason: this shared structural type is consumed by writer_raster; Ruff checks only this defining module.
class _PdfiumModule(Protocol):  # ruff: ignore[unused-private-protocol]
    PdfDocument: Callable[[Path], _PdfiumDocument]
    PdfiumError: type[Exception]


class _ReportLabCanvas(Protocol):
    def setAuthor(self, value: str) -> None: ...

    def setCreator(self, value: str) -> None: ...

    def setSubject(self, value: str) -> None: ...

    def setTitle(self, value: str) -> None: ...

    def setFillColorRGB(self, red: float, green: float, blue: float) -> None: ...

    def setPageSize(self, size: tuple[float, float]) -> None: ...

    def drawImage(
        self,
        image: object,
        x: float,
        y: float,
        *,
        width: float,
        height: float,
    ) -> None: ...

    # reason: _ReportLabCanvas exposes x/fill as its public contract; bundling would break callers.
    def rect(  # ruff: ignore[too-many-arguments]
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        stroke: int,
        fill: int,
    ) -> None: ...

    def showPage(self) -> None: ...

    def save(self) -> None: ...


class _ReportLabCanvasFactory(Protocol):
    def __call__(
        self,
        filename: str | BytesIO,
        *,
        pagesize: tuple[float, float],
        pageCompression: int = 0,  # ruff: ignore[invalid-argument-name] reason: mirrors ReportLab's keyword
        invariant: int = 0,
    ) -> _ReportLabCanvas: ...


# reason: this shared structural type is consumed by writer_overlay and writer_raster; Ruff checks only this module.
class _ReportLabCanvasModule(Protocol):  # ruff: ignore[unused-private-protocol]
    Canvas: _ReportLabCanvasFactory


# reason: this shared structural type is consumed by writer_raster; Ruff checks only this defining module.
class _ReportLabImageModule(Protocol):  # ruff: ignore[unused-private-protocol]
    ImageReader: Callable[[PillowImage], object]


@dataclass(frozen=True, slots=True)
class RedactionWriteResult:
    source_path: Path
    output_path: Path
    writer_id: str
    safety: WriterSafety
    page_count: int
    source_sha256: str
    output_sha256: str
    verification_state: Literal["unverified"] = "unverified"


def _validate_request(
    source: Path,
    regions: Sequence[PageRegion],
    destination: Path,
    *,
    allow_empty_regions: bool = False,
) -> tuple[Path, Path]:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        msg = "PDF input does not exist or is not a file"
        raise InputDocumentError(
            msg,
            stage="input",
            code="input_not_file",
        )
    if not regions and not allow_empty_regions:
        msg = "at least one redaction region is required"
        raise InputDocumentError(
            msg,
            stage="input",
            code="empty_regions",
        )
    if source.resolve() == destination.resolve():
        msg = "redaction output must use a different path"
        raise OutputPathError(
            msg,
            stage="output",
            code="in_place_output",
        )
    if destination.exists():
        msg = "redaction output path must not already exist"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_exists",
        )
    if not destination.parent.is_dir():
        msg = "redaction output parent directory does not exist"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_parent_missing",
        )
    return source, destination


def _validate_page_regions(regions: Sequence[PageRegion], page_count: int) -> None:
    if any(region.page_index < 0 or region.page_index >= page_count for region in regions):
        msg = "redaction region references a missing page"
        raise InputDocumentError(
            msg,
            stage="input",
            code="page_out_of_range",
        )


def _group_regions(
    regions: Sequence[PageRegion],
) -> dict[int, tuple[PageRegion, ...]]:
    grouped: dict[int, list[PageRegion]] = {}
    for region in regions:
        grouped.setdefault(region.page_index, []).append(region)
    return {page_index: tuple(items) for page_index, items in grouped.items()}


# reason: validate bounds keeps x0/height at its adapter seam; bundling would hide required inputs.
def _validate_bounds(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    height: float,
) -> None:
    coordinates = (x0, y0, x1, y1)
    if not all(math.isfinite(value) for value in coordinates) or x0 >= x1 or y0 >= y1:
        msg = "redaction region must be finite and non-empty"
        raise InputDocumentError(
            msg,
            stage="input",
            code="invalid_region",
        )
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        msg = "redaction region must be inside its page"
        raise InputDocumentError(
            msg,
            stage="input",
            code="region_outside_page",
        )


# reason: rotate bounds keeps x0/rotation at its adapter seam; bundling would hide required inputs.
def _rotate_bounds(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    height: float,
    rotation: int,
) -> tuple[float, float, float, float]:
    if rotation == 0:
        return x0, y0, x1, y1
    if rotation == QUARTER_TURN_DEGREES:
        return height - y1, x0, height - y0, x1
    if rotation == HALF_TURN_DEGREES:
        return width - x1, height - y1, width - x0, height - y0
    if rotation == THREE_QUARTER_TURN_DEGREES:
        return y0, width - x1, y1, width - x0
    msg = "PDF page rotation must be a multiple of 90 degrees"
    raise InputDocumentError(
        msg,
        stage="input",
        code="unsupported_page_rotation",
    )


def _temporary_output(destination: Path) -> Path:
    workspace = destination.parent / f".meddies-pdf-{uuid4().hex}.working"
    try:
        workspace.mkdir(mode=0o700)
    except OSError as error:
        msg = "private output staging could not be created"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_staging_failed",
            cause_type=type(error).__name__,
        ) from None
    return workspace / "output.pdf"


def _snapshot_source(source: Path, destination: Path) -> str:
    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        msg = "PDF input is not readable"
        raise InputDocumentError(
            msg,
            stage="input",
            code="source_unreadable",
            cause_type=type(error).__name__,
        ) from None
    # reason: snapshot source's try keeps write all with s isreg; splitting would split cleanup from writes.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            msg = "PDF input must be a regular file"
            raise InputDocumentError(
                msg,
                stage="input",
                code="source_not_regular",
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            digest = hashlib.sha256()
            remaining = source_stat.st_size
            while remaining:
                chunk = os.read(source_descriptor, min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    msg = "PDF input changed while creating a private snapshot"
                    raise InputDocumentError(
                        msg,
                        stage="input",
                        code="source_identity_changed",
                    )
                digest.update(chunk)
                _write_all(destination_descriptor, chunk)
                remaining -= len(chunk)
            if os.read(source_descriptor, 1):
                msg = "PDF input changed while creating a private snapshot"
                raise InputDocumentError(
                    msg,
                    stage="input",
                    code="source_identity_changed",
                )
            os.fchmod(destination_descriptor, 0o400)
        finally:
            os.close(destination_descriptor)
    except PdfRedactionError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as error:
        destination.unlink(missing_ok=True)
        msg = "PDF input could not be privately staged"
        raise InputDocumentError(
            msg,
            stage="input",
            code="source_staging_failed",
            cause_type=type(error).__name__,
        ) from None
    finally:
        os.close(source_descriptor)
    return digest.hexdigest()


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            msg = "private source staging write made no progress"
            raise OSError(msg)
        remaining = remaining[written:]


def _cleanup_temporary(temporary: Path) -> None:
    try:
        temporary.unlink(missing_ok=True)
        (temporary.parent / "source.pdf").unlink(missing_ok=True)
        temporary.parent.rmdir()
    except OSError as error:
        msg = "private output staging could not be removed"
        raise RedactionApplyError(
            msg,
            stage="apply",
            code="output_staging_cleanup_failed",
            cause_type=type(error).__name__,
        ) from None


def _promote_temporary(temporary: Path, destination: Path) -> None:
    try:
        temporary.chmod(0o600)
        os.link(temporary, destination)
    except FileExistsError as error:
        msg = "redaction output path was created by another process"
        raise OutputPathError(
            msg,
            stage="output",
            code="output_exists",
            cause_type=type(error).__name__,
        ) from error
    temporary.unlink()
