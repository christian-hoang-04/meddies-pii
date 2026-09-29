from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from meddies_pii.pdf_redaction.benchmark.execution_report import FullMatrixEvidence
from meddies_pii.pdf_redaction.benchmark.execution_types import FullMatrixInputs
from meddies_pii.pdf_redaction.benchmark.runtime import (
    ModelGate,
    OcrCandidateGateError,
    OcrSuccess,
)
from meddies_pii.pdf_redaction.benchmark.spatial import OracleRegion, SpatialCoverage
from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
    GeometryToken,
    PageRegion,
    Point,
    Quad,
)
from meddies_pii.pdf_redaction.document import DocumentInspection, PageInspection
from meddies_pii.pdf_redaction.routing import PageRouteSignals, classify_page
from meddies_pii.pdf_redaction.verification import (
    VerificationFinding,
    VerificationReport,
)
from meddies_pii.pdf_redaction.writers import RedactionWriteResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from meddies_pii.bioes_inference import SpanDetection
    from meddies_pii.pdf_redaction.ocr import RasterPage


def _quad() -> Quad:
    return Quad(points=(Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)))


def _page(text: str = "synthetic") -> GeometryPage:
    return GeometryPage(0, 100, 100, text, ())


def _ocr_page() -> GeometryPage:
    return GeometryPage(
        0,
        100,
        100,
        "synthetic",
        (GeometryToken("synthetic", _quad(), 0, 9, "ocr", 1.0),),
    )


def _inputs(tmp_path: Path) -> FullMatrixInputs:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-tiny-synthetic")
    return FullMatrixInputs(
        source=source,
        oracle_regions=(
            OracleRegion(
                PageRegion(0, _quad(), "human_name", 0, 1),
                value_index=0,
                channel="visible_raster",
            ),
        ),
        forbidden_values=("synthetic-canary",),
        negative_controls=(),
    )


def _inspection() -> DocumentInspection:
    signals = PageRouteSignals(
        page_index=0,
        visible_native_characters=0,
        has_meaningful_raster=True,
        native_text_trusted=True,
    )
    return DocumentInspection(
        pages=(
            PageInspection(
                signals,
                classify_page(signals),
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


class _DocumentAdapter:
    # reason: this mirrors the `DocumentAdapter` protocol, whose `inspect` the code under test calls on the
    # reason: adapter instance it is handed, so the bound-method form is the API being stood in for.
    def inspect(self, _: Path) -> DocumentInspection:  # ruff: ignore[no-self-use]
        return _inspection()

    # reason: this mirrors the `DocumentAdapter` protocol, whose `render_pages` the code under test calls on the
    # reason: adapter instance it is handed, so the bound-method form is the API being stood in for.
    def render_pages(self, _: Path, *, page_indices: tuple[int, ...]) -> tuple[object, ...]:  # ruff: ignore[no-self-use]
        assert page_indices == (0,)
        return (object(),)


class _NativeExtractor:
    # reason: this mirrors the `NativeExtractor` protocol, whose `extract` the code under test calls on the
    # reason: extractor instance it is handed, so the bound-method form is the API being stood in for.
    def extract(self, _: Path) -> tuple[GeometryPage, ...]:  # ruff: ignore[no-self-use]
        return (_page(),)


class _Detector:
    @staticmethod
    def load() -> None:
        return

    @staticmethod
    def detect(texts: Sequence[str]) -> list[SpanDetection]:
        assert list(texts) == ["synthetic"]
        return []


class _Ocr:
    @staticmethod
    def extract(raster: RasterPage) -> GeometryPage:
        del raster
        return _ocr_page()


class _Writer:
    dpi = 200

    def __init__(self) -> None:
        self.destinations: list[Path] = []

    def write(self, source: Path, _: object, destination: Path) -> RedactionWriteResult:
        destination.write_bytes(b"redacted")
        self.destinations.append(destination)
        return RedactionWriteResult(
            source_path=source,
            output_path=destination,
            writer_id="fake-raster",
            safety="destructive",
            page_count=1,
            source_sha256="a" * 64,
            output_sha256="b" * 64,
        )


class _Verifier:
    def __init__(self) -> None:
        self.canaries: list[tuple[str, ...]] = []

    def verify(self, written: RedactionWriteResult, *, known_canaries: tuple[str, ...]) -> VerificationReport:
        self.canaries.append(known_canaries)
        return VerificationReport(
            output_path=written.output_path,
            output_sha256=written.output_sha256,
            passed=True,
            findings=(VerificationFinding(gate="structure", passed=True, code="passed", detail=""),),
            observations=(),
        )


def _measure_three(action: Callable[[], object]) -> tuple[float, float, float]:
    for _ in range(3):
        action()
    return 0.1, 0.2, 0.3


def _install_safe_matrix_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], _Writer, _Verifier]:
    import meddies_pii.pdf_redaction.benchmark.execution_matrix as matrix
    from meddies_pii.bioes_inference import artifacts
    from meddies_pii.pdf_redaction import document, extraction, geometry, verification, writers

    writer = _Writer()
    verifier = _Verifier()
    artifact = SimpleNamespace(model_path=tmp_path / "model.onnx", tokenizer_root=tmp_path)
    artifact.model_path.write_bytes(b"model")
    region = PageRegion(0, _quad(), "human_name", 0, 1)
    captured: dict[str, object] = {}

    monkeypatch.setattr(artifacts, "hydrate_artifact", lambda *_: artifact)
    monkeypatch.setattr(document, "PdfiumDocumentAdapter", lambda **_: _DocumentAdapter())
    monkeypatch.setattr(extraction, "PdfiumExtractor", _NativeExtractor)
    monkeypatch.setattr(geometry, "map_detections_to_regions", lambda *_: (region,))
    monkeypatch.setattr(writers, "RasterRebuildWriter", lambda **_: writer)
    monkeypatch.setattr(verification, "IndependentPdfVerifier", lambda **_: verifier)
    monkeypatch.setattr(matrix, "merge_prepared_pages", lambda *_: (_page(),))
    monkeypatch.setattr(matrix, "prepare_pages", lambda *_: (_page(),))
    monkeypatch.setattr(matrix, "measure_three", _measure_three)
    monkeypatch.setattr(
        matrix,
        "measure_model_gate",
        lambda **_: ModelGate(
            detector=_Detector(),
            detections=(),
            load_seconds=0.4,
            warm_seconds=(0.1, 0.2, 0.3),
            selected_runtime_id="openvino-fake-t6",
            component_rows=({"runtime_id": "openvino-fake-t6", "status": "measured"},),
        ),
    )
    monkeypatch.setattr(
        matrix,
        "measure_ocr_candidate",
        lambda **kwargs: _ocr_success(kwargs["candidate_id"]),
    )
    monkeypatch.setattr(
        matrix,
        "build_full_matrix_result",
        lambda evidence: captured.update(evidence=evidence) or {"schema_version": 2, "stage": "full_matrix"},
    )
    fidelity = SimpleNamespace(changed_outside_mask_rate=0.0)
    import meddies_pii.pdf_redaction.benchmark.fidelity as fidelity_module

    monkeypatch.setattr(
        fidelity_module,
        "write_raster_reference",
        lambda _, destination, **__: destination.write_bytes(b"reference") or destination,
    )
    monkeypatch.setattr(fidelity_module, "measure_render_fidelity", lambda *_, **__: fidelity)
    return captured, writer, verifier


def _ocr_success(candidate_id: str) -> OcrSuccess:
    if candidate_id == "rapidocr-ppocrv6-small-openvino":
        raise OcrCandidateGateError(({"candidate_id": candidate_id, "status": "failed"},))
    return OcrSuccess(
        candidate_id=candidate_id,
        adapter=_Ocr(),
        pages=(_ocr_page(),),
        cold_seconds=0.5,
        warm_seconds=(0.2, 0.1, 0.3),
        exact_visible_recall=1.0,
        clean_page_misses=0,
        model_manifest=({"role": "recognizer", "bytes": 5, "sha256": "c" * 64},),
    )


def test_execute_full_matrix_records_fake_measurement_contract_without_remote_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meddies_pii.pdf_redaction.benchmark.execution_matrix as matrix

    captured, writer, verifier = _install_safe_matrix_fakes(monkeypatch, tmp_path)
    inputs = _inputs(tmp_path)

    execution = matrix.execute_full_matrix(
        inputs,
        work_root=tmp_path / "work",
        artifact_root=tmp_path / "artifacts",
        allocated_cpu_cores=3.0,
    )

    evidence = captured["evidence"]
    assert isinstance(evidence, FullMatrixEvidence)
    assert execution.result == {"schema_version": 2, "stage": "full_matrix"}
    assert execution.promoted_output.name == "h4-oracle-2.pdf"
    assert execution.promoted_output_sha256 == "b" * 64
    assert [row["status"] for row in evidence.ocr_rows] == [
        "failed",
        "measured",
        "measured",
        "measured",
    ]
    assert evidence.best_ocr.candidate_id == "rapidocr-ppocrv6-small-onnxruntime"
    assert [timing.stage for timing in evidence.stage_timings] == [
        "cold_start",
        "model_load",
        "page_classification",
        "extraction_ocr",
        "pii_inference",
        "geometry_mapping",
        "write_sanitize",
        "verify",
        "end_to_end",
    ]
    assert all(len(timing.seconds) in {1, 3} for timing in evidence.stage_timings)
    assert verifier.canaries == [inputs.forbidden_values] * 6
    assert len(writer.destinations) == 6
    assert not any(path.name.startswith("h4-model-timing") and path.exists() for path in writer.destinations)


def test_execute_full_matrix_returns_safe_candidate_rows_when_no_ocr_candidate_is_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meddies_pii.pdf_redaction.benchmark.execution_matrix as matrix
    from meddies_pii.bioes_inference import artifacts
    from meddies_pii.pdf_redaction import document, extraction

    artifact = SimpleNamespace(model_path=tmp_path / "model.onnx", tokenizer_root=tmp_path)
    artifact.model_path.write_bytes(b"model")
    monkeypatch.setattr(artifacts, "hydrate_artifact", lambda *_: artifact)
    monkeypatch.setattr(document, "PdfiumDocumentAdapter", lambda **_: _DocumentAdapter())
    monkeypatch.setattr(extraction, "PdfiumExtractor", _NativeExtractor)
    monkeypatch.setattr(
        matrix,
        "measure_ocr_candidate",
        lambda **_: (_ for _ in ()).throw(RuntimeError("synthetic canary")),
    )

    with pytest.raises(OcrCandidateGateError) as raised:
        matrix.execute_full_matrix(
            _inputs(tmp_path),
            work_root=tmp_path / "work",
            artifact_root=tmp_path / "artifacts",
            allocated_cpu_cores=1.0,
        )

    rows = raised.value.candidate_rows
    assert len(rows) == 4
    assert all(row["status"] == "failed" for row in rows)
    assert all(len(cast("str", row["error_digest"])) == 64 for row in rows)
    assert all("error_message" not in row for row in rows)


@pytest.mark.parametrize(
    "failure",
    ["model_regions", "oracle_controls", "oracle_verification"],
)
def test_execute_full_matrix_stops_at_non_promotable_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import meddies_pii.pdf_redaction.benchmark.execution_matrix as matrix
    from meddies_pii.pdf_redaction import geometry

    _, _, verifier = _install_safe_matrix_fakes(monkeypatch, tmp_path)
    if failure == "model_regions":
        monkeypatch.setattr(geometry, "map_detections_to_regions", lambda *_: ())
        message = "no redaction regions"
    elif failure == "oracle_controls":
        calls = 0

        def coverage(**_: object) -> SpatialCoverage:
            nonlocal calls
            calls += 1
            return SpatialCoverage(1, 1.0, 1, 1.0, 0.0, 0.0, calls == 2)

        monkeypatch.setattr(matrix, "measure_spatial_coverage", coverage)
        message = "negative controls"
    else:
        monkeypatch.setattr(
            verifier,
            "verify",
            lambda written, **_: VerificationReport(
                output_path=written.output_path,
                output_sha256=written.output_sha256,
                passed=False,
                findings=(VerificationFinding(gate="structure", passed=False, code="failed", detail=""),),
                observations=(),
            ),
        )
        message = "independent verification"

    with pytest.raises(RuntimeError, match=message):
        matrix.execute_full_matrix(
            _inputs(tmp_path),
            work_root=tmp_path / "work",
            artifact_root=tmp_path / "artifacts",
            allocated_cpu_cores=1.0,
        )
