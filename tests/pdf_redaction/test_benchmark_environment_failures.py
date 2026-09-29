from __future__ import annotations

import hashlib
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from pypdf import PdfWriter
from pypdf.generic import NameObject, NumberObject, RectangleObject

import meddies_pii.pdf_redaction.benchmark.execution_environment as environment_module
import meddies_pii.pdf_redaction.benchmark.execution_report as report_module
from meddies_pii.bioes_inference.artifacts import (
    HydratedArtifactIdentity,
    HydratedFileIdentity,
)
from meddies_pii.pdf_redaction.benchmark.execution_report import (
    FullMatrixEvidence,
    build_full_matrix_result,
)
from meddies_pii.pdf_redaction.benchmark.execution_types import FullMatrixInputs
from meddies_pii.pdf_redaction.benchmark.fidelity import RenderFidelity
from meddies_pii.pdf_redaction.benchmark.results import (
    TIMING_STAGES,
    NoEligibleCandidateError,
    StageTimingRecord,
)
from meddies_pii.pdf_redaction.benchmark.runtime import ModelGate, OcrSuccess
from meddies_pii.pdf_redaction.benchmark.spatial import (
    OracleRegion,
    SpatialCoverage,
)
from meddies_pii.pdf_redaction.contracts import (
    GeometryPage,
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
    from pathlib import Path

    from meddies_pii.bioes_inference import BioesSpanDetector
    from meddies_pii.pdf_redaction.ocr import GeometryOcr


def _quad() -> Quad:
    return Quad(
        points=(
            Point(0, 0),
            Point(10, 0),
            Point(10, 10),
            Point(0, 10),
        ),
    )


def _write_pdf(
    path: Path,
    *,
    pages: int = 1,
    media_box: tuple[float, float, float, float] = (0, 0, 100, 100),
    crop_box: tuple[float, float, float, float] | None = None,
    rotation: int = 0,
) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(
            width=media_box[2] - media_box[0],
            height=media_box[3] - media_box[1],
        )
        page.mediabox = RectangleObject(media_box)
        page.cropbox = RectangleObject(crop_box or media_box)
        if rotation:
            page[NameObject("/Rotate")] = NumberObject(rotation)
    writer.write(path)


def _evidence(
    tmp_path: Path,
    *,
    source: Path,
    output: Path,
    model_coverage: SpatialCoverage | None = None,
) -> FullMatrixEvidence:
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    oracle = OracleRegion(
        region=PageRegion(0, _quad(), "human_name", 0, 1),
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
                signals=signals,
                decision=classify_page(signals),
                raster_area_ratio=0.0,
                displayed_image_count=0,
                invisible_native_characters=0,
                replacement_characters=0,
                annotation_count=0,
                widget_count=0,
                rotation_degrees=0,
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
    perfect = SpatialCoverage(1, 1.0, 1, 1.0, 0.0, 0.0, 0)
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
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
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
    return FullMatrixEvidence(
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
        prepared_pages=(GeometryPage(0, 100, 100, "x", ()),),
        model_regions=(),
        model_coverage=model_coverage or perfect,
        model_coverage_rows={"by_page": []},
        oracle_coverage=perfect,
        stage_timings=timings,
        promoted_write=RedactionWriteResult(
            source_path=source,
            output_path=output,
            writer_id="raster-rebuild",
            safety="destructive",
            page_count=1,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
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
    )


def _disable_unrelated_measurement_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_module, "_peak_rss_bytes", lambda: 42)


def test_report_records_available_and_missing_runtime_dependencies_and_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)
    _write_pdf(output)

    def package_version(name: str) -> str:
        if name == "pypdf":
            return "9.9"
        raise environment_module.importlib.metadata.PackageNotFoundError(name)

    def tool_version(
        command: tuple[str, ...],
        *,
        capture_output: bool,
        check: bool,
        timeout: int,
    ) -> SimpleNamespace:
        assert capture_output is True
        assert check is False
        assert timeout == 10
        if command[0] == "qpdf":
            return SimpleNamespace(stdout=b"qpdf 9.9\n", stderr=b"", returncode=0)
        if command[0] == "pdftotext":
            return SimpleNamespace(stdout=b"", stderr=b"poppler 9.9\n", returncode=0)
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=7)

    class _SystemFile:
        def __init__(self, path: str) -> None:
            self.path = path

        # reason: this stands in for `pathlib.Path`, and `execution_environment.py:169` and `:206` call
        # reason: `read_text(encoding="utf-8")` on it, so `encoding` is part of the signature being replaced.
        def read_text(self, encoding: str | None = None) -> str:  # ruff: ignore[unused-method-argument]
            if self.path == "/proc/cpuinfo":
                return "processor: 0\nmodel name: Synthetic CPU\n"
            return "Name:\tpython\nVmRSS:\t123 kB\n"

    monkeypatch.setattr(environment_module.importlib.metadata, "version", package_version)
    monkeypatch.setattr(environment_module.subprocess, "run", tool_version)
    monkeypatch.setattr(environment_module.platform, "platform", lambda: "synthetic")
    monkeypatch.setattr(environment_module, "Path", _SystemFile)
    _disable_unrelated_measurement_probes(monkeypatch)

    result = build_full_matrix_result(_evidence(tmp_path, source=source, output=output))

    environment = cast("dict[str, object]", result["environment"])
    packages = cast("dict[str, str]", environment["packages"])
    tools = cast("dict[str, str]", environment["tools"])
    assert packages["pypdf"] == "9.9"
    assert packages["torch"] == "unavailable"
    assert environment["cpu_model"] == "Synthetic CPU"
    assert tools == {
        "qpdf": "qpdf 9.9",
        "poppler": "poppler 9.9",
        "tesseract": "exit-7",
    }


@pytest.mark.parametrize("cpuinfo", ["unreadable", "missing-model"])
def test_report_falls_back_when_host_runtime_metadata_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpuinfo: str,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)
    _write_pdf(output)

    class _SystemFile:
        def __init__(self, path: str) -> None:
            self.path = path

        # reason: this stands in for `pathlib.Path`, and `execution_environment.py:169` and `:206` call
        # reason: `read_text(encoding="utf-8")` on it, so `encoding` is part of the signature being replaced.
        def read_text(self, encoding: str | None = None) -> str:  # ruff: ignore[unused-method-argument]
            if self.path == "/proc/cpuinfo" and cpuinfo == "missing-model":
                return "processor: 0\n"
            msg = "synthetic unavailable"
            raise OSError(msg)

    monkeypatch.setattr(environment_module, "Path", _SystemFile)
    monkeypatch.setattr(environment_module.platform, "platform", lambda: "synthetic")
    monkeypatch.setattr(environment_module.platform, "processor", lambda: "")
    monkeypatch.setattr(
        environment_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing tool")),
    )
    _disable_unrelated_measurement_probes(monkeypatch)

    result = build_full_matrix_result(_evidence(tmp_path, source=source, output=output))

    environment = cast("dict[str, object]", result["environment"])
    assert environment["cpu_model"] == "unavailable"
    assert environment["tools"] == {
        "qpdf": "unavailable",
        "poppler": "unavailable",
        "tesseract": "unavailable",
    }


def test_report_rejects_model_containment_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)
    _write_pdf(output)
    failed = SpatialCoverage(0, 0.0, 1, 1.0, 1.0, 0.0, 0)
    monkeypatch.setattr(report_module, "_environment_identity", lambda **_: {})
    _disable_unrelated_measurement_probes(monkeypatch)

    result = build_full_matrix_result(
        _evidence(
            tmp_path,
            source=source,
            output=output,
            model_coverage=failed,
        ),
    )

    assert result["automated_redaction_status"] == "failed_model_containment_gate"
    assert result["automated_redaction_winner"] is None


# reason: Explicit source and output page geometry lets each parametrized case isolate one page-invariant violation.
@pytest.mark.parametrize(
    (
        "source_pages",
        "source_box",
        "source_crop",
        "source_rotation",
        "output_box",
        "output_rotation",
    ),
    [
        (1, (0, 0, 100, 100), None, 0, (0, 0, 100, 100), 0),
        (1, (0, 0, 100, 200), None, 45, (0, 0, 100, 200), 0),
        (1, (0, 0, 100, 200), None, 0, (0, 0, 200, 100), 90),
        (
            1,
            (0, 0, 100, 200),
            (0, 0, 90, 200),
            90,
            (0, 0, 200, 100),
            0,
        ),
        (1, (10, 0, 110, 200), None, 90, (0, 0, 200, 100), 0),
    ],
)
def test_report_refuses_outputs_that_break_pdf_page_invariants(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_pages: int,
    source_box: tuple[float, float, float, float],
    source_crop: tuple[float, float, float, float] | None,
    source_rotation: int,
    output_box: tuple[float, float, float, float],
    output_rotation: int,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(
        source,
        pages=source_pages,
        media_box=source_box,
        crop_box=source_crop,
        rotation=source_rotation,
    )
    output_pages = 2 if source_rotation == 0 and source_box == output_box else 1
    _write_pdf(
        output,
        pages=output_pages,
        media_box=output_box,
        rotation=output_rotation,
    )
    monkeypatch.setattr(report_module, "_environment_identity", lambda **_: {})
    _disable_unrelated_measurement_probes(monkeypatch)

    with pytest.raises(NoEligibleCandidateError):
        build_full_matrix_result(_evidence(tmp_path, source=source, output=output))


def test_report_rejects_nonfinite_page_box_from_pdf_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)
    _write_pdf(output)
    reads = 0

    class _Page:
        mediabox = (0.0, 0.0, float("nan"), 100.0)
        cropbox = (0.0, 0.0, 100.0, 100.0)

        # reason: this mirrors `pypdf`'s `Page.get`, which the code under test calls on a page instance, so the
        # reason: bound-method form is the API being stood in for.
        def get(self, key: str, default: object) -> object:  # ruff: ignore[no-self-use]
            assert key == "/Rotate"
            return default

    class _Reader:
        def __init__(self, _: Path, *, strict: bool) -> None:
            nonlocal reads
            assert strict is True
            reads += 1
            self.pages = [_Page()]

    fake_pypdf = ModuleType("pypdf")
    # reason: build_full_matrix_result imports pypdf by name out of sys.modules, so the stand-in must be a
    # reason: real module; ModuleType declares no such attribute and no annotation admits the write.
    fake_pypdf.PdfReader = _Reader  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    with pytest.raises(ValueError, match="four finite coordinates"):
        build_full_matrix_result(_evidence(tmp_path, source=source, output=output))

    assert reads == 2
