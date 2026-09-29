"""PDF inspection and rendering facade."""

from meddies_pii.pdf_redaction.document_pdfium import PdfiumDocumentAdapter
from meddies_pii.pdf_redaction.document_pymupdf import PymupdfDocumentAdapter
from meddies_pii.pdf_redaction.document_types import (
    DEFAULT_MAX_RASTER_PIXELS,
    DEFAULT_MEANINGFUL_RASTER_AREA_RATIO,
    DEFAULT_RENDER_DPI,
    DocumentAdapter,
    DocumentAdapterError,
    DocumentInspection,
    PageInspection,
    PdfSource,
    RasterArtifact,
)

__all__ = [
    "DEFAULT_MAX_RASTER_PIXELS",
    "DEFAULT_MEANINGFUL_RASTER_AREA_RATIO",
    "DEFAULT_RENDER_DPI",
    "DocumentAdapter",
    "DocumentAdapterError",
    "DocumentInspection",
    "PageInspection",
    "PdfSource",
    "PdfiumDocumentAdapter",
    "PymupdfDocumentAdapter",
    "RasterArtifact",
]
