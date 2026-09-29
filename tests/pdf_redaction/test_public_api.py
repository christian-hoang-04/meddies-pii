from __future__ import annotations

from meddies_pii import bioes_inference, pdf_redaction


def test_public_api_exposes_only_product_composition_seams() -> None:
    assert set(pdf_redaction.__all__) == {
        "IndependentPdfVerifier",
        "InputDocumentError",
        "OutputPathError",
        "PdfGeometryPreparer",
        "PdfRedactionError",
        "PdfRedactionPipeline",
        "PdfiumExtractor",
        "PdfiumDocumentAdapter",
        "PipelineVerificationError",
        "PreparedDocument",
        "RapidOcrAdapter",
        "RasterRebuildWriter",
        "ReviewedRedaction",
        "RedactionApplyError",
        "RedactionPreview",
        "VerificationError",
        "VerificationToolchain",
        "VerifiedRedactionOutput",
        "create_default_detector",
    }
    assert not hasattr(pdf_redaction, "UnsafeOverlayWriter")
    assert not hasattr(pdf_redaction, "FullMatrixInputs")


def test_pdf_convenience_detector_export_is_the_neutral_inference_factory() -> None:
    assert pdf_redaction.create_default_detector is bioes_inference.create_default_detector
