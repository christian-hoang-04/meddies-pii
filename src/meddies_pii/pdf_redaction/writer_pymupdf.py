from __future__ import annotations

# ruff: file-ignore[invalid-function-name]
# reason: the PyMuPDF module protocol must spell its upstream `Rect` constructor exactly for structural typing.
import importlib
from typing import TYPE_CHECKING, Protocol, cast

from typing_extensions import Self

from meddies_pii.bioes_inference.file_identity import file_sha256
from meddies_pii.pdf_redaction.errors import (
    InputDocumentError,
    PdfRedactionError,
    RedactionApplyError,
)
from meddies_pii.pdf_redaction.writer_common import (
    RedactionWriteResult,
    _cleanup_temporary,
    _promote_temporary,
    _snapshot_source,
    _temporary_output,
    _validate_bounds,
    _validate_page_regions,
    _validate_request,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from meddies_pii.pdf_redaction.contracts import PageRegion


class _Rect(Protocol):
    width: float
    height: float


class _Page(Protocol):
    cropbox: _Rect

    def add_redact_annot(self, rectangle: _Rect, *, fill: tuple[float, float, float], cross_out: bool) -> None: ...

    def apply_redactions(self) -> None: ...


class _Document(Protocol):
    needs_pass: bool
    page_count: int

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def __getitem__(self, index: int) -> _Page: ...

    def scrub(self) -> None: ...

    def save(self, path: Path, **kwargs: object) -> None: ...


class _PymupdfModule(Protocol):
    EmptyFileError: type[Exception]
    FileDataError: type[Exception]

    def open(self, path: Path) -> _Document: ...

    def Rect(self, x0: float, y0: float, x1: float, y1: float) -> _Rect: ...


class PyMuPdfRedactionWriter:
    writer_id = "pymupdf-object-aware"

    def write(
        self,
        source: Path,
        regions: Sequence[PageRegion],
        destination: Path,
    ) -> RedactionWriteResult:
        source, destination = _validate_request(source, regions, destination)
        temporary = _temporary_output(destination)
        staged_source = temporary.parent / "source.pdf"
        # reason: PyMuPdfRedactionWr's try keeps page regions with check bounds; splitting would split cleanup from writes.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            source_sha256 = _snapshot_source(source, staged_source)
            pymupdf = cast("_PymupdfModule", importlib.import_module("pymupdf"))

            try:
                document = pymupdf.open(staged_source)
            except (pymupdf.EmptyFileError, pymupdf.FileDataError) as error:
                msg = "PDF input is empty, malformed, or unsupported"
                raise InputDocumentError(
                    msg,
                    stage="input",
                    code="malformed_document",
                    cause_type=type(error).__name__,
                ) from error
            with document:
                if document.needs_pass:
                    msg = "encrypted PDF input is not supported"
                    # reason: keep the refusal inside the document and temporary-output cleanup boundary.
                    raise InputDocumentError(  # ruff: ignore[raise-within-try]
                        msg,
                        stage="input",
                        code="encrypted_document",
                    )
                _validate_page_regions(regions, document.page_count)
                touched_pages: set[int] = set()
                for region in regions:
                    bounds = region.quad.bounds()
                    page = document[region.page_index]
                    _validate_bounds(
                        bounds.x0,
                        bounds.y0,
                        bounds.x1,
                        bounds.y1,
                        page.cropbox.width,
                        page.cropbox.height,
                    )
                    rectangle = pymupdf.Rect(
                        bounds.x0,
                        bounds.y0,
                        bounds.x1,
                        bounds.y1,
                    )
                    page.add_redact_annot(
                        rectangle,
                        fill=(0.0, 0.0, 0.0),
                        cross_out=False,
                    )
                    touched_pages.add(region.page_index)

                for page_index in sorted(touched_pages):
                    document[page_index].apply_redactions()
                document.scrub()
                page_count = document.page_count
                document.save(
                    temporary,
                    garbage=4,
                    clean=True,
                    deflate=True,
                    deflate_images=True,
                    deflate_fonts=True,
                    incremental=False,
                    preserve_metadata=False,
                    use_objstms=1,
                )
            output_sha256 = file_sha256(temporary)
            _promote_temporary(temporary, destination)
        except PdfRedactionError:
            raise
        except Exception as error:
            msg = "PyMuPDF could not apply destructive redaction"
            raise RedactionApplyError(
                msg,
                stage="apply",
                code="pymupdf_apply_failed",
                cause_type=type(error).__name__,
            ) from error
        finally:
            _cleanup_temporary(temporary)

        return RedactionWriteResult(
            source_path=source,
            output_path=destination,
            writer_id=self.writer_id,
            safety="destructive",
            page_count=page_count,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
        )
