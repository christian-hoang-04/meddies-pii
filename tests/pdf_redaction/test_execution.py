from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pypdf import PdfWriter
from pypdf.generic import NameObject, NumberObject, RectangleObject

from meddies_pii.bioes_inference import SpanDetection
from meddies_pii.pdf_redaction.benchmark.execution_environment import (
    _automated_redaction_status,
    _current_rss_bytes,
    _page_invariants,
)
from meddies_pii.pdf_redaction.benchmark.runtime import (
    ModelCandidateGateError,
    OcrCandidateGateError,
    _has_exact_detection_consensus,
    _ModelRuntimeMeasurement,
    _select_openvino_measurement,
    fixed_failure_row,
    select_ocr_model_files,
)
from meddies_pii.pdf_redaction.benchmark.spatial import (
    OracleRegion,
    PageQuad,
    measure_spatial_coverage,
    model_coverage_breakdown,
)
from meddies_pii.pdf_redaction.contracts import PageRegion, Point, Quad
from meddies_pii.pdf_redaction.document import DocumentInspection, PageInspection
from meddies_pii.pdf_redaction.ocr import OcrAdapterError
from meddies_pii.pdf_redaction.routing import PageRouteSignals, classify_page

if TYPE_CHECKING:
    from meddies_pii.bioes_inference import BioesSpanDetector


def _quad(x0: float, y0: float, x1: float, y1: float) -> Quad:
    return Quad(
        points=(
            Point(x0, y0),
            Point(x1, y0),
            Point(x1, y1),
            Point(x0, y1),
        ),
    )


def test_current_rss_probe_is_positive() -> None:
    assert _current_rss_bytes() > 0


def _write_box_fixture(
    path: Path,
    *,
    media_box: tuple[float, float, float, float],
    crop_box: tuple[float, float, float, float],
    rotation: int = 0,
) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(
        width=media_box[2] - media_box[0],
        height=media_box[3] - media_box[1],
    )
    page.mediabox = RectangleObject(media_box)
    page.cropbox = RectangleObject(crop_box)
    if rotation:
        page[NameObject("/Rotate")] = NumberObject(rotation)
    writer.write(path)


def test_page_invariants_compare_media_and_crop_box_coordinates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    changed_crop = tmp_path / "changed-crop.pdf"
    changed_media = tmp_path / "changed-media.pdf"
    _write_box_fixture(
        source,
        media_box=(0, 0, 200, 300),
        crop_box=(10, 20, 190, 280),
    )
    _write_box_fixture(
        changed_crop,
        media_box=(0, 0, 200, 300),
        crop_box=(0, 0, 180, 260),
    )
    _write_box_fixture(
        changed_media,
        media_box=(0, 0, 180, 260),
        crop_box=(0, 0, 180, 260),
    )

    assert _page_invariants(source, changed_crop) == (False, ())
    assert _page_invariants(source, changed_media) == (False, ())


def test_page_invariants_allow_only_full_box_rotation_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    normalized = tmp_path / "normalized.pdf"
    offset_crop = tmp_path / "offset-crop.pdf"
    _write_box_fixture(
        source,
        media_box=(0, 0, 200, 300),
        crop_box=(0, 0, 200, 300),
        rotation=90,
    )
    _write_box_fixture(
        normalized,
        media_box=(0, 0, 300, 200),
        crop_box=(0, 0, 300, 200),
    )
    _write_box_fixture(
        offset_crop,
        media_box=(0, 0, 300, 200),
        crop_box=(10, 0, 300, 200),
    )

    assert _page_invariants(source, normalized) == (True, (0,))
    assert _page_invariants(source, offset_crop) == (False, ())


def test_automated_redaction_requires_independently_verified_model_outputs() -> None:
    coverage = measure_spatial_coverage(
        gold=(PageQuad(page_index=0, quad=_quad(0, 0, 10, 10)),),
        applied=(PageQuad(page_index=0, quad=_quad(0, 0, 10, 10)),),
        negative_controls=(),
    )
    assert _automated_redaction_status(coverage, outputs_verified=False) == (
        False,
        "failed_model_output_verification",
    )
    assert _automated_redaction_status(coverage, outputs_verified=True) == (
        True,
        "eligible",
    )
    overredacted = measure_spatial_coverage(
        gold=(PageQuad(page_index=0, quad=_quad(0, 0, 10, 10)),),
        applied=(PageQuad(page_index=0, quad=_quad(-1, 0, 11, 10)),),
        negative_controls=(),
    )
    assert _automated_redaction_status(overredacted, outputs_verified=True) == (
        False,
        "failed_model_overredaction_gate",
    )


def test_openvino_selection_uses_warm_median_before_load_time() -> None:
    detector = cast("BioesSpanDetector", object())
    t6 = _ModelRuntimeMeasurement(
        detector=detector,
        detections=(),
        runtime_id="openvino-cpu-t6",
        load_seconds=1.0,
        warm_seconds=(2.0, 2.1, 2.2),
    )
    t8 = _ModelRuntimeMeasurement(
        detector=detector,
        detections=(),
        runtime_id="openvino-cpu-t8",
        load_seconds=9.0,
        warm_seconds=(1.8, 1.9, 2.0),
    )

    assert _select_openvino_measurement((t6, t8)).runtime_id.endswith("t8")
    divergent = _ModelRuntimeMeasurement(
        detector=detector,
        detections=(SpanDetection(spans=(), bucket=8, num_tokens=1, truncated=False),),
        runtime_id="onnxruntime-cpu-t6",
        load_seconds=0.1,
        warm_seconds=(1.0, 1.0, 1.0),
    )
    assert _has_exact_detection_consensus((t6, t8))
    assert not _has_exact_detection_consensus((t6, divergent))


def test_model_gate_error_preserves_phi_free_candidate_rows() -> None:
    rows = ({"runtime_id": "openvino-cpu-t6", "status": "measured"},)

    error = ModelCandidateGateError(rows)

    assert error.candidate_rows == rows
    assert str(error) == "PII runtimes failed exact-span consensus"


def test_spatial_coverage_uses_mask_union_and_counts_control_touches() -> None:
    gold = (
        PageQuad(page_index=0, quad=_quad(0, 0, 10, 10)),
        PageQuad(page_index=1, quad=_quad(0, 0, 10, 10)),
    )
    applied = (
        PageQuad(page_index=0, quad=_quad(0, 0, 6, 10)),
        PageQuad(page_index=0, quad=_quad(4, 0, 10, 10)),
        PageQuad(page_index=1, quad=_quad(0, 0, 5, 10)),
    )
    controls = (
        PageQuad(page_index=0, quad=_quad(20, 20, 25, 25)),
        PageQuad(page_index=1, quad=_quad(4, 4, 6, 6)),
    )

    result = measure_spatial_coverage(
        gold=gold,
        applied=applied,
        negative_controls=controls,
    )

    assert result.gold_regions_fully_covered == 1
    assert result.gold_region_recall == 0.5
    assert result.gold_regions_partially_covered == 2
    assert result.gold_region_partial_recall == 1.0
    assert result.undercoverage_rate == pytest.approx(0.25)
    assert result.overredaction_rate == pytest.approx(0.0)
    assert result.negative_control_touches == 1


def test_spatial_coverage_handles_gaps_between_disjoint_masks() -> None:
    result = measure_spatial_coverage(
        gold=(PageQuad(page_index=0, quad=_quad(0, 0, 10, 10)),),
        applied=(
            PageQuad(page_index=0, quad=_quad(0, 0, 2, 10)),
            PageQuad(page_index=0, quad=_quad(8, 0, 10, 10)),
        ),
        negative_controls=(),
    )

    assert result.gold_regions_fully_covered == 0
    assert result.gold_regions_partially_covered == 1
    assert result.gold_region_partial_recall == 1.0
    assert result.undercoverage_rate == pytest.approx(0.6)


def test_spatial_coverage_measures_area_outside_gold_union() -> None:
    result = measure_spatial_coverage(
        gold=(PageQuad(page_index=0, quad=_quad(0, 0, 10, 10)),),
        applied=(PageQuad(page_index=0, quad=_quad(-5, 0, 15, 10)),),
        negative_controls=(),
    )

    assert result.undercoverage_rate == 0.0
    assert result.overredaction_rate == pytest.approx(1.0)


def test_spatial_coverage_does_not_double_count_overlapping_gold() -> None:
    result = measure_spatial_coverage(
        gold=(
            PageQuad(page_index=0, quad=_quad(0, 0, 10, 10)),
            PageQuad(page_index=0, quad=_quad(5, 0, 15, 10)),
        ),
        applied=(PageQuad(page_index=0, quad=_quad(0, 0, 15, 10)),),
        negative_controls=(),
    )

    assert result.undercoverage_rate == 0.0
    assert result.overredaction_rate == 0.0


def test_breakdown_scopes_applied_regions_by_page_route_and_label() -> None:
    oracle = (
        OracleRegion(
            region=PageRegion(0, _quad(0, 0, 10, 10), "human_name", 0, 1),
            value_index=0,
            channel="visible_native",
        ),
        OracleRegion(
            region=PageRegion(1, _quad(0, 0, 10, 10), "phone_number", 0, 1),
            value_index=1,
            channel="visible_raster",
        ),
    )
    native_signals = PageRouteSignals(
        page_index=0,
        visible_native_characters=10,
        has_meaningful_raster=False,
        native_text_trusted=True,
    )
    image_signals = PageRouteSignals(
        page_index=1,
        visible_native_characters=0,
        has_meaningful_raster=True,
        native_text_trusted=True,
    )
    inspection = DocumentInspection(
        pages=(
            PageInspection(
                native_signals,
                classify_page(native_signals),
                0.0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            PageInspection(
                image_signals,
                classify_page(image_signals),
                1.0,
                1,
                0,
                0,
                0,
                0,
                0,
            ),
        ),
        attachment_count=0,
    )
    applied = (
        PageQuad(0, _quad(0, 0, 10, 10), "human_name"),
        PageQuad(1, _quad(0, 0, 10, 10), "phone_number"),
    )

    breakdown = model_coverage_breakdown(
        oracle_regions=oracle,
        applied=applied,
        inspection=inspection,
    )

    for scope in ("by_page", "by_route", "by_label"):
        rows = breakdown[scope]
        assert isinstance(rows, list)
        assert all(row["overredaction_rate"] == 0.0 for row in rows)


def test_ocr_gate_error_preserves_phi_free_candidate_rows() -> None:
    rows = (
        {
            "candidate_id": "rapidocr-ppocrv6-small-openvino",
            "status": "failed",
            "error_digest": "a" * 64,
        },
        {
            "candidate_id": "tesseract-fast-eng-vie",
            "status": "measured",
            "exact_visible_recall": 0.5,
        },
    )

    error = OcrCandidateGateError(rows)

    assert str(error) == "no independent-verifier-compatible OCR candidate passed"
    assert error.candidate_rows == rows


def test_ocr_failure_row_preserves_structured_diagnosis_without_text() -> None:
    row = fixed_failure_row(
        "rapidocr-ppocrv6-small-openvino",
        "ocr_gate",
        OcrAdapterError("rapidocr", "invalid_output", page_index=4),
    )

    assert row["error_code"] == "rapidocr.invalid_output"
    assert row["page_index"] == 4


def test_ocr_manifest_selects_only_candidate_weight_files() -> None:
    files = (
        (Path("small-det.onnx"), 9_929_594),
        (Path("small-rec.onnx"), 21_234_383),
        (Path("cls.onnx"), 585_532),
        (Path("tiny-det.onnx"), 1_829_618),
        (Path("eng.traineddata"), 4_113_088),
    )

    selected = select_ocr_model_files(
        "rapidocr-ppocrv6-small-openvino",
        files,
    )

    assert tuple((role, path.name) for role, path in selected) == (
        ("detector", "small-det.onnx"),
        ("recognizer", "small-rec.onnx"),
        ("classifier", "cls.onnx"),
    )


def test_full_matrix_report_assembles_measured_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator delegates the stable result schema to report assembly."""
    from meddies_pii.bioes_inference.artifacts import (
        HydratedArtifactIdentity,
        HydratedFileIdentity,
    )
    from meddies_pii.pdf_redaction.benchmark import execution_report
    from meddies_pii.pdf_redaction.benchmark.execution_report import (
        FullMatrixEvidence,
        build_full_matrix_result,
    )
    from meddies_pii.pdf_redaction.benchmark.execution_types import FullMatrixInputs
    from meddies_pii.pdf_redaction.benchmark.fidelity import RenderFidelity
    from meddies_pii.pdf_redaction.benchmark.results import (
        TIMING_STAGES,
        StageTimingRecord,
    )
    from meddies_pii.pdf_redaction.benchmark.runtime import ModelGate, OcrSuccess
    from meddies_pii.pdf_redaction.benchmark.spatial import (
        OracleRegion,
        SpatialCoverage,
    )
    from meddies_pii.pdf_redaction.contracts import GeometryPage
    from meddies_pii.pdf_redaction.ocr import GeometryOcr
    from meddies_pii.pdf_redaction.verification import (
        VerificationFinding,
        VerificationReport,
    )
    from meddies_pii.pdf_redaction.writers import RedactionWriteResult

    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    digest = "a" * 64
    oracle = OracleRegion(
        region=PageRegion(0, _quad(0, 0, 10, 10), "human_name", 0, 1),
        value_index=0,
        channel="visible_native",
    )
    inputs = FullMatrixInputs(
        source=source,
        oracle_regions=(oracle,),
        forbidden_values=("synthetic-canary",),
        negative_controls=(),
    )
    signals = PageRouteSignals(
        page_index=0,
        visible_native_characters=1,
        has_meaningful_raster=False,
        native_text_trusted=True,
    )
    inspection = DocumentInspection(
        pages=(
            PageInspection(
                signals,
                classify_page(signals),
                0.0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        ),
        attachment_count=0,
    )
    report = VerificationReport(
        output_path=output,
        output_sha256=digest,
        passed=True,
        findings=(VerificationFinding(gate="structure", passed=True, code="passed", detail=""),),
        observations=(),
    )
    coverage = SpatialCoverage(1, 1.0, 1, 1.0, 0.0, 0.0, 0)
    fidelity = RenderFidelity(1, 72, 16, 2, 100, 0, 0.0, ())
    artifact = HydratedArtifactIdentity(
        artifact_id="test_model",
        repo_id="Meddies/test-model",
        revision="b" * 40,
        root=tmp_path,
        tokenizer_directory=tmp_path,
        files=(
            HydratedFileIdentity(
                relative_path="model.onnx",
                path=source,
                size_bytes=source.stat().st_size,
                sha256=digest,
                role="model",
            ),
        ),
    )
    timings = tuple(
        StageTimingRecord(
            stage=stage,
            seconds=(0.1,) if stage in {"cold_start", "model_load"} else (0.1,) * 3,
        )
        for stage in TIMING_STAGES
    )
    monkeypatch.setattr(execution_report, "_page_invariants", lambda *_: (True, ()))
    monkeypatch.setattr(
        execution_report,
        "_environment_identity",
        lambda **_: {"test": "environment"},
    )
    monkeypatch.setattr(execution_report, "_peak_rss_bytes", lambda: 42)
    monkeypatch.setattr(execution_report, "_current_rss_bytes", lambda: 24)

    result = build_full_matrix_result(
        FullMatrixEvidence(
            inputs=inputs,
            artifact=artifact,
            ocr_rows=[{"candidate_id": "rapidocr", "status": "measured"}],
            best_ocr=OcrSuccess(
                candidate_id="rapidocr",
                adapter=cast("GeometryOcr", object()),
                pages=(),
                cold_seconds=0.1,
                warm_seconds=(0.1, 0.1, 0.1),
                exact_visible_recall=1.0,
                clean_page_misses=0,
                model_manifest=(),
            ),
            model_gate=ModelGate(
                detector=cast("BioesSpanDetector", object()),
                detections=(),
                load_seconds=0.1,
                warm_seconds=(0.1, 0.1, 0.1),
                selected_runtime_id="openvino-cpu-t6",
                component_rows=(),
            ),
            inspection=inspection,
            ocr_indices=(),
            prepared_pages=(cast("GeometryPage", object()),),
            model_regions=(),
            model_coverage=coverage,
            model_coverage_rows={"by_page": []},
            oracle_coverage=coverage,
            stage_timings=timings,
            promoted_write=RedactionWriteResult(
                source_path=source,
                output_path=output,
                writer_id="raster-rebuild",
                safety="destructive",
                page_count=1,
                source_sha256=digest,
                output_sha256=digest,
            ),
            final_report=report,
            render_fidelity=fidelity,
            rasterization_fidelity=fidelity,
            model_reports=[report],
            process_setup_seconds=0.1,
            hydration_seconds=0.2,
            allocated_cpu_cores=8.0,
            started_at=0.0,
        ),
    )

    candidates = cast("list[dict[str, object]]", result["candidate_results"])
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "H1",
        "H2",
        "H3",
        "H4",
    ]
    assert result["model_outputs_independently_verified"] is True
    assert result["automated_redaction_status"] == "eligible"
    assert result["memory_metrics"] == {
        "benchmark_process_peak_rss_bytes": 42,
        "post_matrix_current_rss_bytes": 24,
        "scope": ("single_process_high_water_and_post_matrix_snapshot; candidate runtimes may remain resident"),
    }
