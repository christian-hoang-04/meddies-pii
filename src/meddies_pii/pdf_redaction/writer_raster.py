from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import importlib
import math
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast, runtime_checkable

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
    _PdfiumModule,
    _promote_temporary,
    _ReportLabCanvas,
    _ReportLabCanvasModule,
    _ReportLabImageModule,
    _rotate_bounds,
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


@runtime_checkable
class _PillowImageDraw(Protocol):
    def rectangle(self, xy: tuple[int, int, int, int], *, fill: tuple[int, int, int]) -> None: ...


_PillowDrawFactory: TypeAlias = Callable[[object], object]
"""`PIL.ImageDraw.Draw`.

Reached through `getattr`, named so the call type-checks against a declared shape rather than against `object`.

"""


def _load_image_draw(image: object) -> _PillowImageDraw:
    module = importlib.import_module("PIL.ImageDraw")
    draw_factory: _PillowDrawFactory | None = getattr(module, "Draw", None)
    if not callable(draw_factory):
        msg = "PIL.ImageDraw.Draw is unavailable"
        raise ValueError(msg)
    draw: object = draw_factory(image)
    if not isinstance(draw, _PillowImageDraw):
        msg = "PIL.ImageDraw.Draw returned an invalid drawing context"
        raise ValueError(msg)
    return draw


@dataclass(frozen=True, slots=True)
class RasterRebuildWriter:
    dpi: int = 300

    @property
    def writer_id(self) -> str:
        return f"pdfium-raster-rebuild-{self.dpi}dpi"

    def write(
        self,
        source: Path,
        regions: Sequence[PageRegion],
        destination: Path,
    ) -> RedactionWriteResult:
        return self._write(
            source,
            regions,
            destination,
            allow_empty_regions=False,
            safety="destructive",
        )

    # reason: RasterRebuildWriter owns render and validate together; splitting would split cleanup from writes.
    def _write(  # ruff: ignore[too-many-locals,too-many-statements]
        self,
        source: Path,
        regions: Sequence[PageRegion],
        destination: Path,
        *,
        allow_empty_regions: bool,
        safety: Literal["destructive", "reference"],
    ) -> RedactionWriteResult:
        source, destination = _validate_request(
            source,
            regions,
            destination,
            allow_empty_regions=allow_empty_regions,
        )
        if self.dpi <= 0:
            msg = "raster rebuild DPI must be positive"
            raise InputDocumentError(
                msg,
                stage="input",
                code="invalid_dpi",
            )
        temporary = _temporary_output(destination)
        staged_source = temporary.parent / "source.pdf"
        # reason: RasterRebuildWrite's try keeps render with page regions; splitting would split cleanup from writes.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            source_sha256 = _snapshot_source(source, staged_source)
            pdfium = cast("_PdfiumModule", importlib.import_module("pypdfium2"))
            image_module = cast(
                "_ReportLabImageModule",
                importlib.import_module("reportlab.lib.utils"),
            )
            canvas_module = cast(
                "_ReportLabCanvasModule",
                importlib.import_module("reportlab.pdfgen.canvas"),
            )

            pixel_scale = self.dpi / 72
            try:
                document = pdfium.PdfDocument(staged_source)
            except pdfium.PdfiumError as error:
                msg = "PDF input is malformed, encrypted, or unsupported"
                raise InputDocumentError(
                    msg,
                    stage="input",
                    code="malformed_document",
                    cause_type=type(error).__name__,
                ) from error
            with closing(document):
                page_count = len(document)
                _validate_page_regions(regions, page_count)
                regions_by_page = _group_regions(regions)
                canvas: _ReportLabCanvas | None = None
                for page_index in range(page_count):
                    page = document[page_index]
                    width_pt, height_pt = page.get_size()
                    rotation = page.get_rotation()
                    if rotation in {90, 270}:
                        source_width_pt, source_height_pt = height_pt, width_pt
                    else:
                        source_width_pt, source_height_pt = width_pt, height_pt
                    image = page.render(scale=pixel_scale).to_pil().convert("RGB")
                    draw = _load_image_draw(image)
                    for region in regions_by_page.get(page_index, ()):
                        bounds = region.quad.bounds()
                        _validate_bounds(
                            bounds.x0,
                            bounds.y0,
                            bounds.x1,
                            bounds.y1,
                            source_width_pt,
                            source_height_pt,
                        )
                        x0, y0, x1, y1 = _rotate_bounds(
                            bounds.x0,
                            bounds.y0,
                            bounds.x1,
                            bounds.y1,
                            source_width_pt,
                            source_height_pt,
                            rotation,
                        )
                        draw.rectangle(
                            (
                                math.floor(x0 * pixel_scale),
                                math.floor(y0 * pixel_scale),
                                math.ceil(x1 * pixel_scale),
                                math.ceil(y1 * pixel_scale),
                            ),
                            fill=(0, 0, 0),
                        )
                    if canvas is None:
                        canvas = canvas_module.Canvas(
                            str(temporary),
                            pagesize=(width_pt, height_pt),
                            pageCompression=1,
                            invariant=1,
                        )
                        canvas.setAuthor("")
                        canvas.setCreator("")
                        canvas.setSubject("")
                        canvas.setTitle("")
                    else:
                        canvas.setPageSize((width_pt, height_pt))
                    canvas.drawImage(
                        image_module.ImageReader(image),
                        0,
                        0,
                        width=width_pt,
                        height=height_pt,
                    )
                    canvas.showPage()
                if canvas is None:
                    msg = "PDF input has no pages"
                    # reason: keep the refusal inside the temporary-output cleanup and domain-error boundary.
                    raise InputDocumentError(  # ruff: ignore[raise-within-try]
                        msg,
                        stage="input",
                        code="empty_document",
                    )
                canvas.save()
            output_sha256 = file_sha256(temporary)
            _promote_temporary(temporary, destination)
        except PdfRedactionError:
            raise
        except Exception as error:
            msg = "PDFium could not rebuild the PDF from redacted pixels"
            raise RedactionApplyError(
                msg,
                stage="apply",
                code="raster_rebuild_failed",
                cause_type=type(error).__name__,
            ) from error
        finally:
            _cleanup_temporary(temporary)

        return RedactionWriteResult(
            source_path=source,
            output_path=destination,
            writer_id=self.writer_id,
            safety=safety,
            page_count=page_count,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
        )
