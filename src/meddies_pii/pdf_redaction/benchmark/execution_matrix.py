from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the redaction runtime is an optional extra, and several backends are resolved by name at call time.
import statistics
import time
from typing import TYPE_CHECKING

from meddies_pii.pdf_redaction.benchmark.execution_report import (
    FullMatrixEvidence,
    build_full_matrix_result,
)
from meddies_pii.pdf_redaction.benchmark.execution_types import (
    FullMatrixExecution,
    FullMatrixInputs,
)
from meddies_pii.pdf_redaction.benchmark.results import (
    TIMING_STAGES,
    StageTimingRecord,
)
from meddies_pii.pdf_redaction.benchmark.runtime import (
    OcrCandidateGateError,
    OcrSuccess,
    fixed_failure_row,
    manifest_bytes,
    measure_model_gate,
    measure_ocr_candidate,
    measure_three,
    merge_prepared_pages,
    prepare_pages,
)
from meddies_pii.pdf_redaction.benchmark.spatial import (
    PageQuad,
    measure_spatial_coverage,
    model_coverage_breakdown,
)

if TYPE_CHECKING:
    from pathlib import Path

    from meddies_pii.pdf_redaction.verification import VerificationReport
    from meddies_pii.pdf_redaction.writers import RedactionWriteResult


# reason: Artifact hydration, document inspection, page execution, and receipts share one work root.
def execute_full_matrix(  # ruff: ignore[too-many-locals,too-many-statements]
    inputs: FullMatrixInputs,
    *,
    work_root: Path,
    artifact_root: Path,
    allocated_cpu_cores: float,
) -> FullMatrixExecution:
    from meddies_pii.bioes_inference import (
        BioesSpanDetector,
        ExpectedFileIdentity,
        OnnxRuntimeBackend,
        OpenVinoBackend,
    )
    from meddies_pii.bioes_inference.artifacts import (
        PUBLIC_Q8_ARTIFACT_SPEC,
        hydrate_artifact,
    )
    from meddies_pii.pdf_redaction.benchmark.fidelity import (
        measure_render_fidelity,
        write_raster_reference,
    )
    from meddies_pii.pdf_redaction.document import PdfiumDocumentAdapter
    from meddies_pii.pdf_redaction.extraction import PdfiumExtractor
    from meddies_pii.pdf_redaction.geometry import map_detections_to_regions
    from meddies_pii.pdf_redaction.ocr import (
        GeometryOcr,
        RapidOcrAdapter,
        RapidOcrEngineFactory,
        RapidOcrRuntimeConfig,
        SubprocessTesseractRunner,
        TesseractTsvAdapter,
    )
    from meddies_pii.pdf_redaction.verification import (
        IndependentPdfVerifier,
        VerificationToolchain,
    )
    from meddies_pii.pdf_redaction.writers import RasterRebuildWriter

    started = time.perf_counter()
    work_root.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=True)
    process_setup_seconds = time.perf_counter() - started

    hydration_started = time.perf_counter()
    artifact = hydrate_artifact(PUBLIC_Q8_ARTIFACT_SPEC, artifact_root)
    hydration_seconds = time.perf_counter() - hydration_started

    document_adapter = PdfiumDocumentAdapter(dpi=300)
    native_extractor = PdfiumExtractor()
    inspection = document_adapter.inspect(inputs.source)
    ocr_indices = tuple(page.signals.page_index for page in inspection.pages if page.decision.requires_full_page_ocr)
    raster_artifacts = document_adapter.render_pages(
        inputs.source,
        page_indices=ocr_indices,
    )
    ocr_candidates: tuple[tuple[str, GeometryOcr], ...] = (
        (
            "rapidocr-ppocrv6-small-openvino",
            RapidOcrAdapter(
                engine_factory=RapidOcrEngineFactory(
                    RapidOcrRuntimeConfig(backend="openvino", model_tier="small", threads=4),
                ),
            ),
        ),
        (
            "rapidocr-ppocrv6-small-onnxruntime",
            RapidOcrAdapter(
                engine_factory=RapidOcrEngineFactory(
                    RapidOcrRuntimeConfig(backend="onnxruntime", model_tier="small", threads=4),
                ),
            ),
        ),
        (
            "rapidocr-ppocrv6-tiny-openvino",
            RapidOcrAdapter(
                engine_factory=RapidOcrEngineFactory(
                    RapidOcrRuntimeConfig(backend="openvino", model_tier="tiny", threads=4),
                ),
            ),
        ),
        (
            "tesseract-fast-eng-vie",
            TesseractTsvAdapter(
                runner_factory=lambda: SubprocessTesseractRunner(language="eng+vie", psm=3, timeout_seconds=60),
            ),
        ),
    )
    ocr_rows: list[dict[str, object]] = []
    ocr_successes: list[OcrSuccess] = []
    for candidate_id, adapter in ocr_candidates:
        try:
            success = measure_ocr_candidate(
                candidate_id=candidate_id,
                adapter=adapter,
                artifacts=raster_artifacts,
                oracle_regions=inputs.oracle_regions,
                forbidden_values=inputs.forbidden_values,
            )
        # reason: a benchmark candidate may fail through any injected OCR backend; the failure row is the evidence product.
        except Exception as error:  # ruff: ignore[blind-except]
            ocr_rows.append(fixed_failure_row(candidate_id, "ocr_gate", error))
            continue
        ocr_successes.append(success)
        ocr_rows.append({
            "candidate_id": candidate_id,
            "status": "measured",
            "cold_seconds": success.cold_seconds,
            "warm_seconds": list(success.warm_seconds),
            "median_warm_seconds": statistics.median(success.warm_seconds),
            "exact_visible_recall": success.exact_visible_recall,
            "clean_page_misses": success.clean_page_misses,
            "geometry_pages": len(success.pages),
            "geometry_tokens": sum(len(page.tokens) for page in success.pages),
            "selected_model_bytes": manifest_bytes(success.model_manifest),
            "model_manifest": list(success.model_manifest),
        })
    eligible_ocr = tuple(
        result
        for result in ocr_successes
        if result.candidate_id.startswith("rapidocr-") and any(page.tokens for page in result.pages)
    )
    if not eligible_ocr:
        raise OcrCandidateGateError(ocr_rows)
    best_ocr = min(
        eligible_ocr,
        key=lambda result: (
            result.clean_page_misses,
            -result.exact_visible_recall,
            statistics.median(result.warm_seconds),
            manifest_bytes(result.model_manifest),
        ),
    )
    cold_start_seconds = best_ocr.cold_seconds

    native_pages = native_extractor.extract(inputs.source)
    prepared_pages = merge_prepared_pages(native_pages, best_ocr.pages)
    model_gate = measure_model_gate(
        model_path=artifact.model_path,
        tokenizer_root=artifact.tokenizer_root,
        expected_sha256=PUBLIC_Q8_ARTIFACT_SPEC.files[0].sha256,
        expected_model_size_bytes=PUBLIC_Q8_ARTIFACT_SPEC.files[0].size_bytes,
        expected_tokenizer_files={
            file.relative_path: ExpectedFileIdentity(
                size_bytes=file.size_bytes,
                sha256=file.sha256,
            )
            for file in PUBLIC_Q8_ARTIFACT_SPEC.files
            if file.role == "tokenizer"
        },
        pages=prepared_pages,
        openvino_backends=(
            OpenVinoBackend(threads=6),
            OpenVinoBackend(threads=8),
        ),
        ort_backend=OnnxRuntimeBackend(threads=6),
        detector_class=BioesSpanDetector,
    )
    model_regions = map_detections_to_regions(prepared_pages, model_gate.detections)
    if not model_regions:
        msg = "PII model returned no redaction regions"
        raise RuntimeError(msg)

    gold_quads = tuple(
        PageQuad(
            page_index=item.region.page_index,
            quad=item.region.quad,
            label=item.region.label,
        )
        for item in inputs.oracle_regions
    )
    model_quads = tuple(
        PageQuad(
            page_index=item.page_index,
            quad=item.quad,
            label=item.label,
        )
        for item in model_regions
    )
    model_coverage = measure_spatial_coverage(
        gold=gold_quads,
        applied=model_quads,
        negative_controls=inputs.negative_controls,
    )
    model_coverage_rows = model_coverage_breakdown(
        oracle_regions=inputs.oracle_regions,
        applied=model_quads,
        inspection=inspection,
    )
    oracle_regions = tuple(item.region for item in inputs.oracle_regions)
    oracle_coverage = measure_spatial_coverage(
        gold=gold_quads,
        applied=gold_quads,
        negative_controls=inputs.negative_controls,
    )
    if oracle_coverage.negative_control_touches:
        msg = "oracle masks overlap benchmark negative controls"
        raise RuntimeError(msg)

    writer = RasterRebuildWriter(dpi=200)
    verifier = IndependentPdfVerifier(toolchain=VerificationToolchain(timeout_seconds=60, render_dpi=150))
    page_classification_seconds = measure_three(lambda: document_adapter.inspect(inputs.source))
    extraction_seconds = measure_three(
        lambda: prepare_pages(
            inputs.source,
            document_adapter,
            native_extractor,
            best_ocr.adapter,
        ),
    )
    inference_seconds = measure_three(lambda: model_gate.detector.detect([page.text for page in prepared_pages]))
    mapping_seconds = measure_three(
        lambda: map_detections_to_regions(
            prepared_pages,
            model_gate.detections,
        ),
    )

    written_outputs: list[RedactionWriteResult] = []
    write_index = 0

    def write_oracle() -> RedactionWriteResult:
        nonlocal write_index
        destination = work_root / f"h4-oracle-{write_index}.pdf"
        write_index += 1
        written = writer.write(inputs.source, oracle_regions, destination)
        written_outputs.append(written)
        return written

    write_seconds = measure_three(write_oracle)
    promoted_write = written_outputs[-1]
    reports: list[VerificationReport] = []

    def verify_oracle() -> VerificationReport:
        report = verifier.verify(
            promoted_write,
            known_canaries=inputs.forbidden_values,
        )
        reports.append(report)
        return report

    verify_seconds = measure_three(verify_oracle)
    final_report = reports[-1]
    if not final_report.passed:
        msg = "oracle raster output failed independent verification"
        raise RuntimeError(msg)
    reference_path = work_root / "h4-raster-reference.pdf"
    write_raster_reference(inputs.source, reference_path, dpi=writer.dpi)
    render_fidelity = measure_render_fidelity(
        reference_path,
        promoted_write.output_path,
        oracle_regions,
        mask_geometry_source=inputs.source,
    )
    rasterization_fidelity = measure_render_fidelity(
        inputs.source,
        reference_path,
        (),
    )

    end_to_end_index = 0
    model_reports: list[VerificationReport] = []

    def end_to_end() -> VerificationReport:
        nonlocal end_to_end_index
        pages = prepare_pages(
            inputs.source,
            document_adapter,
            native_extractor,
            best_ocr.adapter,
        )
        detections = model_gate.detector.detect([page.text for page in pages])
        regions = map_detections_to_regions(pages, detections)
        destination = work_root / f"h4-model-timing-{end_to_end_index}.pdf"
        end_to_end_index += 1
        written = writer.write(inputs.source, regions, destination)
        try:
            report = verifier.verify(written, known_canaries=inputs.forbidden_values)
            model_reports.append(report)
            return report
        finally:
            destination.unlink(missing_ok=True)

    end_to_end_seconds = measure_three(end_to_end)
    stage_values: dict[str, tuple[float, ...]] = {
        "cold_start": (cold_start_seconds,),
        "model_load": (model_gate.load_seconds,),
        "page_classification": page_classification_seconds,
        "extraction_ocr": extraction_seconds,
        "pii_inference": inference_seconds,
        "geometry_mapping": mapping_seconds,
        "write_sanitize": write_seconds,
        "verify": verify_seconds,
        "end_to_end": end_to_end_seconds,
    }
    stage_timings = tuple(StageTimingRecord(stage=stage, seconds=stage_values[stage]) for stage in TIMING_STAGES)
    return FullMatrixExecution(
        result=build_full_matrix_result(
            FullMatrixEvidence(
                inputs=inputs,
                artifact=artifact,
                ocr_rows=ocr_rows,
                best_ocr=best_ocr,
                model_gate=model_gate,
                inspection=inspection,
                ocr_indices=ocr_indices,
                prepared_pages=prepared_pages,
                model_regions=model_regions,
                model_coverage=model_coverage,
                model_coverage_rows=model_coverage_rows,
                oracle_coverage=oracle_coverage,
                stage_timings=stage_timings,
                promoted_write=promoted_write,
                final_report=final_report,
                render_fidelity=render_fidelity,
                rasterization_fidelity=rasterization_fidelity,
                model_reports=model_reports,
                process_setup_seconds=process_setup_seconds,
                hydration_seconds=hydration_seconds,
                allocated_cpu_cores=allocated_cpu_cores,
                started_at=started,
            ),
        ),
        promoted_output=promoted_write.output_path,
        promoted_output_sha256=final_report.output_sha256,
    )
