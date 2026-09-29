from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the redaction runtime is an optional extra, and several backends are resolved by name at call time.
import importlib
from io import BytesIO
from typing import TYPE_CHECKING, cast

from meddies_pii.bioes_inference.file_identity import file_sha256
from meddies_pii.pdf_redaction.errors import (
    InputDocumentError,
    PdfRedactionError,
    RedactionApplyError,
)
from meddies_pii.pdf_redaction.writer_common import (
    RedactionWriteResult,
    _cleanup_temporary,
    _group_regions,
    _promote_temporary,
    _ReportLabCanvasModule,
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


class UnsafeOverlayWriter:
    """Calibration-only mask that intentionally leaves source content recoverable."""

    writer_id = "pypdf-overlay-negative-control"

    # reason: validate and page regions share UnsafeOverlayWrite's state; extraction would split cleanup from writes.
    def write(  # ruff: ignore[too-many-locals]
        self,
        source: Path,
        regions: Sequence[PageRegion],
        destination: Path,
    ) -> RedactionWriteResult:
        source, destination = _validate_request(source, regions, destination)
        temporary = _temporary_output(destination)
        staged_source = temporary.parent / "source.pdf"
        # reason: UnsafeOverlayWrite's try keeps page regions with check bounds; splitting would split cleanup from writes.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            source_sha256 = _snapshot_source(source, staged_source)
            from pypdf import PdfReader, PdfWriter
            from pypdf.errors import PdfReadError

            canvas_module = cast(
                "_ReportLabCanvasModule",
                importlib.import_module("reportlab.pdfgen.canvas"),
            )

            try:
                reader = PdfReader(staged_source)
            except PdfReadError as error:
                msg = "PDF input is malformed, encrypted, or unsupported"
                raise InputDocumentError(
                    msg,
                    stage="input",
                    code="malformed_document",
                    cause_type=type(error).__name__,
                ) from error
            page_count = len(reader.pages)
            _validate_page_regions(regions, page_count)
            regions_by_page = _group_regions(regions)
            writer = PdfWriter()
            for page_index, page in enumerate(reader.pages):
                width_pt = float(page.mediabox.width)
                height_pt = float(page.mediabox.height)
                writer.add_page(page)
                output_page = writer.pages[-1]
                page_regions = regions_by_page.get(page_index, ())
                if page_regions:
                    overlay_bytes = BytesIO()
                    overlay = canvas_module.Canvas(
                        overlay_bytes,
                        pagesize=(width_pt, height_pt),
                        invariant=1,
                    )
                    overlay.setFillColorRGB(0, 0, 0)
                    for region in page_regions:
                        bounds = region.quad.bounds()
                        _validate_bounds(
                            bounds.x0,
                            bounds.y0,
                            bounds.x1,
                            bounds.y1,
                            width_pt,
                            height_pt,
                        )
                        overlay.rect(
                            bounds.x0,
                            height_pt - bounds.y1,
                            bounds.x1 - bounds.x0,
                            bounds.y1 - bounds.y0,
                            stroke=0,
                            fill=1,
                        )
                    overlay.showPage()
                    overlay.save()
                    overlay_bytes.seek(0)
                    output_page.merge_page(PdfReader(overlay_bytes).pages[0], over=True)
            writer.write(temporary)
            output_sha256 = file_sha256(temporary)
            _promote_temporary(temporary, destination)
        except PdfRedactionError:
            raise
        except Exception as error:
            msg = "pypdf could not create the unsafe overlay control"
            raise RedactionApplyError(
                msg,
                stage="apply",
                code="overlay_write_failed",
                cause_type=type(error).__name__,
            ) from error
        finally:
            _cleanup_temporary(temporary)

        return RedactionWriteResult(
            source_path=source,
            output_path=destination,
            writer_id=self.writer_id,
            safety="unsafe_overlay",
            page_count=page_count,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
        )
