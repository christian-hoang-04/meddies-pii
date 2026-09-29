from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.bioes_inference.file_identity import file_sha256
from meddies_pii.pdf_redaction.ocr import GeometryOcr, OcrAdapterError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from meddies_pii.bioes_inference import (
        BioesInferenceBackend,
        BioesSpanDetector,
        ExpectedFileIdentity,
        PiiSpanDetector,
        SpanDetection,
    )
    from meddies_pii.pdf_redaction.benchmark.spatial import OracleRegion
    from meddies_pii.pdf_redaction.contracts import GeometryPage
    from meddies_pii.pdf_redaction.document import (
        DocumentAdapter,
        RasterArtifact,
    )
    from meddies_pii.pdf_redaction.extraction import NativePdfExtractor

OCR_CHALLENGE_PAGE_INDEX = 5

_VISIBLE_OCR_CHANNELS = frozenset({"visible_raster"})
_RAPID_SELECTED_MODEL_SIZES = {
    "small": (
        ("detector", 9_929_594),
        ("recognizer", 21_234_383),
        ("classifier", 585_532),
    ),
    "tiny": (
        ("detector", 1_829_618),
        ("recognizer", 4_489_813),
        ("classifier", 585_532),
    ),
}
_TESSERACT_SELECTED_MODELS = (
    ("recognizer_eng", "eng.traineddata"),
    ("recognizer_vie", "vie.traineddata"),
)


class OcrCandidateGateError(RuntimeError):
    def __init__(self, candidate_rows: Sequence[Mapping[str, object]]) -> None:
        super().__init__("no independent-verifier-compatible OCR candidate passed")
        self.candidate_rows = tuple(dict(row) for row in candidate_rows)


class ModelCandidateGateError(RuntimeError):
    def __init__(self, candidate_rows: Sequence[Mapping[str, object]]) -> None:
        super().__init__("PII runtimes failed exact-span consensus")
        self.candidate_rows = tuple(dict(row) for row in candidate_rows)


@dataclass(frozen=True, slots=True)
class OcrSuccess:
    candidate_id: str
    adapter: GeometryOcr
    pages: tuple[GeometryPage, ...] = field(repr=False)
    cold_seconds: float
    warm_seconds: tuple[float, float, float]
    exact_visible_recall: float
    clean_page_misses: int
    model_manifest: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class ModelGate:
    detector: PiiSpanDetector = field(repr=False)
    detections: tuple[SpanDetection, ...] = field(repr=False)
    load_seconds: float
    warm_seconds: tuple[float, float, float]
    selected_runtime_id: str
    component_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _ModelRuntimeMeasurement:
    detector: BioesSpanDetector = field(repr=False)
    detections: tuple[SpanDetection, ...] = field(repr=False)
    runtime_id: str
    load_seconds: float
    warm_seconds: tuple[float, float, float]


def measure_ocr_candidate(
    *,
    candidate_id: str,
    adapter: GeometryOcr,
    artifacts: Sequence[RasterArtifact],
    oracle_regions: Sequence[OracleRegion],
    forbidden_values: Sequence[str],
) -> OcrSuccess:
    cold_started = time.perf_counter()
    extract_ocr_pages(adapter, artifacts)
    cold_seconds = time.perf_counter() - cold_started
    measured_pages: list[tuple[GeometryPage, ...]] = []

    def extract() -> tuple[GeometryPage, ...]:
        pages = extract_ocr_pages(adapter, artifacts)
        measured_pages.append(pages)
        return pages

    warm_seconds = measure_three(extract)
    pages = measured_pages[-1]
    page_indices = {page.page_index for page in pages}
    targets = {
        oracle.value_index
        for oracle in oracle_regions
        if oracle.channel in _VISIBLE_OCR_CHANNELS and oracle.region.page_index in page_indices
    }
    oracle_page_by_value = {oracle.value_index: oracle.region.page_index for oracle in oracle_regions}
    text_by_page = {page.page_index: page.text for page in pages}
    found = {
        value_index
        for value_index in targets
        if forbidden_values[value_index] in text_by_page[oracle_page_by_value[value_index]]
    }
    clean_targets = {
        oracle.value_index
        for oracle in oracle_regions
        if oracle.channel in _VISIBLE_OCR_CHANNELS and oracle.region.page_index == OCR_CHALLENGE_PAGE_INDEX
    }
    return OcrSuccess(
        candidate_id=candidate_id,
        adapter=adapter,
        pages=pages,
        cold_seconds=cold_seconds,
        warm_seconds=warm_seconds,
        exact_visible_recall=len(found) / len(targets) if targets else 0.0,
        clean_page_misses=len(clean_targets - found),
        model_manifest=discover_ocr_model_manifest(candidate_id),
    )


def extract_ocr_pages(
    adapter: GeometryOcr,
    artifacts: Sequence[RasterArtifact],
) -> tuple[GeometryPage, ...]:
    pages = tuple(adapter.extract(artifact.raster) for artifact in artifacts)
    expected = tuple(artifact.raster.page_index for artifact in artifacts)
    observed = tuple(page.page_index for page in pages)
    if observed != expected:
        msg = "OCR page indices changed during extraction"
        raise RuntimeError(msg)
    return pages


# reason: measure model exposes model path/detector as its public contract; bundling would break callers.
def measure_model_gate(  # ruff: ignore[too-many-arguments]
    *,
    model_path: Path,
    tokenizer_root: Path,
    expected_model_size_bytes: int,
    expected_sha256: str,
    expected_tokenizer_files: Mapping[str, ExpectedFileIdentity],
    pages: Sequence[GeometryPage],
    openvino_backends: Sequence[BioesInferenceBackend],
    ort_backend: BioesInferenceBackend,
    detector_class: type[BioesSpanDetector],
) -> ModelGate:
    texts = [page.text for page in pages]
    if any(not text for text in texts):
        msg = "prepared page text is empty"
        raise RuntimeError(msg)
    rows: list[dict[str, object]] = []

    if not openvino_backends:
        msg = "model gate requires at least one OpenVINO backend"
        raise ValueError(msg)
    openvino_measurements: list[_ModelRuntimeMeasurement] = []
    reference_signature: tuple[tuple[tuple[int, int, str], ...], ...] | None = None
    for openvino_backend in openvino_backends:
        runtime_id = _runtime_id(openvino_backend)
        # reason: measure model's try keeps measure with runtime row; splitting would let sample setup drift.
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            measurement = _measure_model_runtime(
                model_path=model_path,
                tokenizer_root=tokenizer_root,
                expected_model_size_bytes=expected_model_size_bytes,
                expected_sha256=expected_sha256,
                expected_tokenizer_files=expected_tokenizer_files,
                texts=texts,
                backend=openvino_backend,
                detector_class=detector_class,
            )
            signature = detection_signature(measurement.detections)
            if reference_signature is None:
                reference_signature = signature
            parity = signature == reference_signature
            rows.append(
                _model_runtime_row(
                    measurement,
                    openvino_backend,
                    parity_with_reference=parity,
                ),
            )
            openvino_measurements.append(measurement)
        # reason: each third-party runtime candidate must become a failure row so the gate can compare the complete matrix.
        except Exception as error:  # ruff: ignore[blind-except]
            rows.append(fixed_failure_row(runtime_id, "pii_runtime", error))
    if len(openvino_measurements) != len(openvino_backends) or not (_has_exact_detection_consensus(openvino_measurements)):
        raise ModelCandidateGateError(rows)
    selected = _select_openvino_measurement(openvino_measurements)

    try:
        ort_measurement = _measure_model_runtime(
            model_path=model_path,
            tokenizer_root=tokenizer_root,
            expected_model_size_bytes=expected_model_size_bytes,
            expected_sha256=expected_sha256,
            expected_tokenizer_files=expected_tokenizer_files,
            texts=texts,
            backend=ort_backend,
            detector_class=detector_class,
        )
        ort_parity = detection_signature(ort_measurement.detections) == detection_signature(selected.detections)
        rows.append(
            _model_runtime_row(
                ort_measurement,
                ort_backend,
                parity_with_reference=ort_parity,
            ),
        )
    except ModelCandidateGateError:
        raise
    # reason: the ORT candidate is a third-party boundary; every failure must be recorded before the model gate refuses it.
    except Exception as error:  # ruff: ignore[blind-except]
        rows.append(fixed_failure_row(_runtime_id(ort_backend), "pii_runtime", error))
        raise ModelCandidateGateError(rows) from None
    if not ort_parity:
        raise ModelCandidateGateError(rows)
    return ModelGate(
        detector=selected.detector,
        detections=selected.detections,
        load_seconds=selected.load_seconds,
        warm_seconds=selected.warm_seconds,
        selected_runtime_id=selected.runtime_id,
        component_rows=tuple(rows),
    )


# reason: measure model keeps model path/detector at its adapter seam; bundling would hide required inputs.
def _measure_model_runtime(  # ruff: ignore[too-many-arguments]
    *,
    model_path: Path,
    tokenizer_root: Path,
    expected_model_size_bytes: int,
    expected_sha256: str,
    expected_tokenizer_files: Mapping[str, ExpectedFileIdentity],
    texts: Sequence[str],
    backend: BioesInferenceBackend,
    detector_class: type[BioesSpanDetector],
) -> _ModelRuntimeMeasurement:
    detector = detector_class(
        model_path=model_path,
        tokenizer_path=tokenizer_root,
        expected_model_size_bytes=expected_model_size_bytes,
        expected_model_sha256=expected_sha256,
        expected_tokenizer_files=expected_tokenizer_files,
        backend=backend,
    )
    load_started = time.perf_counter()
    detector.load()
    load_seconds = time.perf_counter() - load_started
    detector.detect(texts)
    runs: list[tuple[SpanDetection, ...]] = []

    def infer() -> tuple[SpanDetection, ...]:
        detections = tuple(detector.detect(texts))
        runs.append(detections)
        return detections

    warm_seconds = measure_three(infer)
    detections = runs[-1]
    if any(detection.truncated for detection in detections):
        msg = "PII inference truncated a page"
        raise RuntimeError(msg)
    return _ModelRuntimeMeasurement(
        detector=detector,
        detections=detections,
        runtime_id=_runtime_id(backend),
        load_seconds=load_seconds,
        warm_seconds=warm_seconds,
    )


def _select_openvino_measurement(
    measurements: Sequence[_ModelRuntimeMeasurement],
) -> _ModelRuntimeMeasurement:
    if not measurements:
        msg = "OpenVINO selection requires measured runtimes"
        raise ValueError(msg)
    return min(
        measurements,
        key=lambda measurement: (
            statistics.median(measurement.warm_seconds),
            measurement.load_seconds,
            measurement.runtime_id,
        ),
    )


def _has_exact_detection_consensus(
    measurements: Sequence[_ModelRuntimeMeasurement],
) -> bool:
    if not measurements:
        return False
    reference = detection_signature(measurements[0].detections)
    return all(detection_signature(measurement.detections) == reference for measurement in measurements[1:])


def _model_runtime_row(
    measurement: _ModelRuntimeMeasurement,
    backend: BioesInferenceBackend,
    *,
    parity_with_reference: bool,
) -> dict[str, object]:
    return {
        "runtime": backend.name,
        "runtime_id": measurement.runtime_id,
        "status": "measured",
        "threads": getattr(backend, "threads", None),
        "load_seconds": measurement.load_seconds,
        "warm_seconds": list(measurement.warm_seconds),
        "detected_spans": sum(len(item.spans) for item in measurement.detections),
        "truncated_pages": 0,
        "exact_span_parity_with_reference": parity_with_reference,
    }


def _runtime_id(backend: BioesInferenceBackend) -> str:
    threads = getattr(backend, "threads", None)
    return backend.name if threads is None else f"{backend.name}-t{threads}"


def detection_signature(
    detections: Sequence[SpanDetection],
) -> tuple[tuple[tuple[int, int, str], ...], ...]:
    return tuple(tuple((span.start, span.end, span.label) for span in detection.spans) for detection in detections)


def merge_prepared_pages(
    native_pages: Sequence[GeometryPage],
    ocr_pages: Sequence[GeometryPage],
) -> tuple[GeometryPage, ...]:
    ocr_by_index = {page.page_index: page for page in ocr_pages}
    if len(ocr_by_index) != len(ocr_pages):
        msg = "OCR pages contain duplicate indices"
        raise RuntimeError(msg)
    pages = tuple(ocr_by_index.get(native.page_index, native) for native in native_pages)
    if tuple(page.page_index for page in pages) != tuple(range(len(pages))):
        msg = "prepared page indices are not contiguous"
        raise RuntimeError(msg)
    return pages


def prepare_pages(
    source: Path,
    document_adapter: DocumentAdapter,
    native_extractor: NativePdfExtractor,
    ocr: GeometryOcr,
) -> tuple[GeometryPage, ...]:
    inspection = document_adapter.inspect(source)
    native_pages = native_extractor.extract(source)
    ocr_indices = tuple(page.signals.page_index for page in inspection.pages if page.decision.requires_full_page_ocr)
    artifacts = document_adapter.render_pages(source, page_indices=ocr_indices)
    return merge_prepared_pages(native_pages, extract_ocr_pages(ocr, artifacts))


def measure_three(action: Callable[[], object]) -> tuple[float, float, float]:
    samples: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    return samples[0], samples[1], samples[2]


def fixed_failure_row(
    candidate_id: str,
    stage: str,
    error: Exception,
) -> dict[str, object]:
    error_type = type(error).__name__
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "stage": stage,
        "status": "failed",
        "error_type": error_type if error_type.isidentifier() else "Exception",
        "error_digest": hashlib.sha256(f"{error_type}:{error}".encode()).hexdigest(),
    }
    if isinstance(error, OcrAdapterError):
        row["error_code"] = f"{error.engine}.{error.stage}"
        row["page_index"] = error.page_index
    return row


def discover_ocr_model_manifest(
    candidate_id: str,
) -> tuple[dict[str, object], ...]:
    roots = [Path.home() / ".cache", Path("/usr/share/tesseract-ocr")]
    try:
        rapidocr_distribution = importlib.metadata.distribution("rapidocr")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        roots.append(Path(str(rapidocr_distribution.locate_file("rapidocr"))))
    suffixes = frozenset({".bin", ".onnx", ".param", ".traineddata", ".xml"})
    files: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                files.add(path.resolve())
    selected = select_ocr_model_files(
        candidate_id,
        tuple((path, path.stat().st_size) for path in files),
    )
    return tuple(
        {
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for role, path in selected
    )


def select_ocr_model_files(
    candidate_id: str,
    files: Sequence[tuple[Path, int]],
) -> tuple[tuple[str, Path], ...]:
    if candidate_id.startswith("rapidocr-"):
        tier = "tiny" if "-tiny-" in candidate_id else "small"
        specs: Sequence[tuple[str, int | str]] = _RAPID_SELECTED_MODEL_SIZES[tier]
    elif candidate_id == "tesseract-fast-eng-vie":
        specs = _TESSERACT_SELECTED_MODELS
    else:
        msg = "unknown OCR candidate"
        raise ValueError(msg)

    selected: list[tuple[str, Path]] = []
    for role, identity in specs:
        matches = tuple(
            path
            for path, size in files
            if (isinstance(identity, int) and path.suffix.lower() == ".onnx" and size == identity)
            or (isinstance(identity, str) and path.name == identity)
        )
        if len(matches) != 1:
            msg = "candidate OCR model manifest mismatch"
            raise RuntimeError(msg)
        selected.append((role, matches[0]))
    return tuple(selected)


def manifest_bytes(manifest: Sequence[Mapping[str, object]]) -> int:
    total = 0
    for row in manifest:
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int):
            msg = "OCR model manifest bytes must be integers"
            raise TypeError(msg)
        total += size
    return total


def manifest_sha256(manifest: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
