"""Modal CPU benchmark for destructive PDF PII redaction.

Run the cheap security/toolchain gate first, then the complete matrix:

    MODAL_PROFILE=huyhoang041100 uv run modal run \
      scripts/ops/run_pdf_redaction_benchmark.py::preflight
    MODAL_PROFILE=huyhoang041100 uv run modal run \
      scripts/ops/run_pdf_redaction_benchmark.py::main

Only deterministic synthetic fixtures are persisted on Modal. The private-fixture
entrypoint keeps its input and output in an ephemeral container, returns the
unreviewed output directly to the caller, and logs only hashes, counts, timings,
and reason codes.
"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import modal

from meddies_pii.json_types import is_object_dict
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
    use_pinned_debian_snapshot,
)
from meddies_pii.pdf_redaction.contracts import PageRegion, Point, Quad

if TYPE_CHECKING:
    from collections.abc import Iterator

APP_NAME = "meddies-pii-pdf-redaction-benchmark"
OUTPUT_VOLUME_NAME = "meddies-pii-pdf-redaction-benchmark"
OUTPUT_MOUNT = "/benchmark"
CACHE_VOLUME_NAME = "hf-cache"
CACHE_MOUNT = "/cache"
TIMEOUT_SECONDS = 30 * 60
CPU_CORES = 8.0
MEMORY_RANGE_MIB = (4096, 16384)
RESULT_SCHEMA_VERSION = 2
MAX_PRIVATE_FIXTURE_BYTES = 50 * 1024 * 1024

PDF_APT_PACKAGES = (
    "fonts-dejavu-core=2.37-6",
    "libgl1=1.6.0-1",
    "libglib2.0-0=2.74.6-2+deb12u9",
    "poppler-utils=22.12.0-2+deb12u2",
    "qpdf=11.3.0-1+deb12u1",
    "tesseract-ocr=5.3.0-2",
    "tesseract-ocr-eng=1:4.1.0-2",
    "tesseract-ocr-vie=1:4.1.0-2",
)

PDF_PACKAGES = (
    "pillow==12.3.0",
    "pymupdf==1.28.0",
    "pypdf==6.14.2",
    "pypdfium2==5.12.1",
    "reportlab==5.0.0",
)
RUNTIME_PACKAGES = (
    "huggingface_hub==1.19.0",
    "numpy==2.4.4",
    "onnxruntime==1.27.0",
    "openvino==2026.1.0",
    "rapidocr==3.9.1",
    "transformers==5.5.0",
)

output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


system_image = use_pinned_debian_snapshot(
    modal.Image.from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d"),
).apt_install(*PDF_APT_PACKAGES)
runtime_environment = {
    "HF_HOME": f"{CACHE_MOUNT}/huggingface",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
preflight_image = (
    add_source_pythonpath(system_image.pip_install(*PDF_PACKAGES))
    .env(runtime_environment)
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
)
benchmark_image = (
    add_source_pythonpath(system_image.pip_install(*PDF_PACKAGES, *RUNTIME_PACKAGES))
    .env(runtime_environment)
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)
)

app = modal.App(APP_NAME)


def _fixture_payload() -> tuple[
    bytes,
    list[dict[str, Any]],
    tuple[str, ...],
    list[dict[str, Any]],
]:
    from meddies_pii.pdf_redaction.benchmark.corpus import generate_challenge_corpus

    fixture = generate_challenge_corpus()
    regions: list[dict[str, Any]] = []
    for canary_index, canary in enumerate(fixture.gold.canaries):
        regions.extend(
            {
                "page_index": canary.page_index,
                "label": canary.label,
                "value_index": canary_index,
                "channel": canary.channel,
                "points": [(point.x, point.y) for point in quad.points],
            }
            for quad in canary.quads
        )
    controls = [
        {
            "page_index": control.page_index,
            "points": [(point.x, point.y) for point in quad.points],
        }
        for control in fixture.gold.negative_controls
        for quad in control.quads
    ]
    return (
        fixture.pdf_bytes,
        regions,
        tuple(canary.raw_value for canary in fixture.gold.canaries),
        controls,
    )


@app.function(
    image=preflight_image,
    cpu=CPU_CORES,
    memory=MEMORY_RANGE_MIB,
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
    secrets=[hf_secret],
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
)
def run_preflight(
    pdf_bytes: bytes,
    oracle_records: list[dict[str, Any]],
    forbidden_values: tuple[str, ...],
) -> str:
    return _run_preflight_impl(pdf_bytes, oracle_records, forbidden_values)


@app.function(
    image=benchmark_image,
    cpu=CPU_CORES,
    memory=MEMORY_RANGE_MIB,
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
    secrets=[hf_secret],
    volumes={OUTPUT_MOUNT: output_volume, CACHE_MOUNT: cache_volume},
)
def run_full_matrix(
    pdf_bytes: bytes,
    oracle_records: list[dict[str, Any]],
    forbidden_values: tuple[str, ...],
    control_records: list[dict[str, Any]],
) -> str:
    return _run_full_matrix_impl(
        pdf_bytes,
        oracle_records,
        forbidden_values,
        control_records,
    )


@app.function(
    image=benchmark_image,
    cpu=CPU_CORES,
    memory=MEMORY_RANGE_MIB,
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
    secrets=[hf_secret],
    volumes={CACHE_MOUNT: cache_volume},
)
def run_private_fixture(pdf_bytes: bytes) -> tuple[bytes, dict[str, object]]:
    return _run_private_fixture_impl(pdf_bytes)


@app.local_entrypoint()
def preflight() -> None:
    pdf_bytes, oracle_records, forbidden_values, _ = _fixture_payload()
    output_root = run_preflight.remote(
        pdf_bytes,
        oracle_records,
        forbidden_values,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "stage": "preflight",
                "output_root": output_root,
                "volume": OUTPUT_VOLUME_NAME,
            },
            sort_keys=True,
        ),
    )


@app.local_entrypoint()
def main() -> None:
    pdf_bytes, oracle_records, forbidden_values, control_records = _fixture_payload()
    output_root = run_full_matrix.remote(
        pdf_bytes,
        oracle_records,
        forbidden_values,
        control_records,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "stage": "full_matrix",
                "output_root": output_root,
                "volume": OUTPUT_VOLUME_NAME,
            },
            sort_keys=True,
        ),
    )


@app.local_entrypoint()
def private_fixture(source: str, destination: str) -> None:
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    _require_distinct_private_fixture_paths(source_path, destination_path)

    pdf_bytes = _read_private_fixture(source_path)
    output_bytes, result = run_private_fixture.remote(pdf_bytes)
    _write_private_fixture_output(
        destination_path,
        output_bytes,
        expected_sha256=str(result["output_sha256"]),
    )
    print(
        json.dumps(
            {**result, "destination_written": True},
            sort_keys=True,
        ),
    )


# reason: Preflight hydration, fixture execution, timing, and receipt writing share one benchmark run identity.
def _run_preflight_impl(  # ruff: ignore[too-many-locals]
    pdf_bytes: bytes,
    oracle_records: list[dict[str, Any]],
    forbidden_values: tuple[str, ...],
) -> str:
    from meddies_pii.pdf_redaction.harness import copy_verified_once, write_json_once
    from meddies_pii.pdf_redaction.risk import classify_pdf_risk
    from meddies_pii.pdf_redaction.verification import (
        IndependentPdfVerifier,
        VerificationToolchain,
    )
    from meddies_pii.pdf_redaction.writers import (
        PyMuPdfRedactionWriter,
        RasterRebuildWriter,
        UnsafeOverlayWriter,
    )

    started = time.perf_counter()
    run_id = _run_id("preflight")
    output_root = Path(OUTPUT_MOUNT) / run_id
    # reason: this body runs only inside the `run_preflight` Modal container, whose filesystem has no
    # reason: second principal, and `exist_ok=False` over a uuid-suffixed run id fails closed if the
    # reason: name is ever pre-created. The private-fixture path, which handles operator PDFs rather
    # reason: than the public benchmark fixture, uses the hardened `_private_fixture_workspace`.
    work_root = Path("/tmp") / run_id  # ruff: ignore[hardcoded-temp-file]
    work_root.mkdir(parents=True, exist_ok=False)
    source = work_root / "source.pdf"
    source.write_bytes(pdf_bytes)
    regions = _regions_from_payload(oracle_records)
    risk = classify_pdf_risk(source)
    verifier = IndependentPdfVerifier(toolchain=VerificationToolchain(timeout_seconds=60, render_dpi=150))

    writer_rows: list[dict[str, object]] = []
    promoted_outputs: list[tuple[Path, str, str]] = []
    writer_cases = (
        (
            UnsafeOverlayWriter(),
            False,
            "negative_control_must_fail",
        ),
        (
            PyMuPdfRedactionWriter(),
            risk.risk == "static",
            "static_documents_only",
        ),
        (
            RasterRebuildWriter(dpi=200),
            risk.risk != "unsupported",
            "fresh_raster_for_renderable_documents",
        ),
    )
    for writer, expected_pass, expectation_reason in writer_cases:
        destination = work_root / f"{writer.writer_id}.pdf"
        writer_started = time.perf_counter()
        written = writer.write(source, regions, destination)
        report = verifier.verify(written, known_canaries=forbidden_values)
        elapsed = time.perf_counter() - writer_started
        calibrated = report.passed is expected_pass
        if report.passed and expected_pass:
            persisted = output_root / f"oracle-{writer.writer_id}.pdf"
            promoted_outputs.append((destination, persisted.name, report.output_sha256))
        writer_rows.append({
            "writer_id": writer.writer_id,
            "declared_safety": written.safety,
            "expected_pass": expected_pass,
            "expectation_reason": expectation_reason,
            "passed": report.passed,
            "calibrated": calibrated,
            "elapsed_seconds": elapsed,
            "input_bytes": len(pdf_bytes),
            "output_bytes": destination.stat().st_size,
            "output_sha256": report.output_sha256,
            "finding_codes": [finding.code for finding in report.findings],
            "failed_gates": [finding.gate for finding in report.findings if not finding.passed],
            "matched_value_sha256": sorted({
                digest for finding in report.findings for digest in finding.matched_value_sha256
            }),
            "risky_markers": sorted({marker for finding in report.findings for marker in finding.risky_markers}),
            "tool_observations": [
                {
                    "gate": observation.gate,
                    "tool": observation.tool,
                    "status": observation.status,
                    "returncode": observation.returncode,
                    "stdout_bytes": observation.stdout_bytes,
                    "stderr_bytes": observation.stderr_bytes,
                    "stdout_sha256": observation.stdout_sha256,
                    "stderr_sha256": observation.stderr_sha256,
                    "duration_ms": observation.duration_ms,
                }
                for observation in report.observations
            ],
        })

    calibration_passed = all(bool(row["calibrated"]) for row in writer_rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": "preflight",
        "fixture_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "oracle_region_count": len(regions),
        "calibration_passed": calibration_passed,
        "risk": risk.risk,
        "risk_evidence_codes": list(risk.evidence_codes),
        "writer_rows": writer_rows,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json_once(
        output_root / "preflight.json",
        result,
        forbidden_values=forbidden_values,
    )
    for source_output, name, expected_sha256 in promoted_outputs:
        copy_verified_once(
            source_output,
            output_root / name,
            expected_sha256=expected_sha256,
        )
    output_volume.commit()
    if not calibration_passed:
        failed_writer_ids = ",".join(str(row["writer_id"]) for row in writer_rows if not bool(row["calibrated"]))
        msg = f"writer/verifier calibration failed: {failed_writer_ids}; evidence={output_root.name}"
        raise RuntimeError(msg)
    return str(output_root)


def _run_full_matrix_impl(
    pdf_bytes: bytes,
    oracle_records: list[dict[str, Any]],
    forbidden_values: tuple[str, ...],
    control_records: list[dict[str, Any]],
) -> str:
    preflight_root = _run_preflight_impl(
        pdf_bytes,
        oracle_records,
        forbidden_values,
    )
    from meddies_pii.pdf_redaction.benchmark.execution import (
        FullMatrixInputs,
        execute_full_matrix,
    )
    from meddies_pii.pdf_redaction.benchmark.runtime import (
        ModelCandidateGateError,
        OcrCandidateGateError,
    )
    from meddies_pii.pdf_redaction.benchmark.spatial import OracleRegion, PageQuad
    from meddies_pii.pdf_redaction.harness import (
        copy_verified_once,
        safe_failure_record,
        write_json_once,
        write_jsonl_once,
    )

    run_id = _run_id("full")
    output_root = Path(OUTPUT_MOUNT) / run_id
    # reason: same container-local case as the preflight work root above — `run_full_matrix` owns the
    # reason: filesystem alone, and `exist_ok=False` over a uuid-suffixed run id fails closed.
    work_root = Path("/tmp") / run_id  # ruff: ignore[hardcoded-temp-file]
    work_root.mkdir(parents=True, exist_ok=False)
    source = work_root / "source.pdf"
    source.write_bytes(pdf_bytes)
    regions = _regions_from_payload(oracle_records)
    oracle_regions = tuple(
        OracleRegion(
            region=region,
            value_index=int(record["value_index"]),
            channel=str(record["channel"]),
        )
        for region, record in zip(regions, oracle_records, strict=True)
    )
    controls = tuple(
        PageQuad(
            page_index=int(record["page_index"]),
            quad=_quad_from_payload(record),
        )
        for record in control_records
    )
    try:
        execution = execute_full_matrix(
            FullMatrixInputs(
                source=source,
                oracle_regions=oracle_regions,
                forbidden_values=forbidden_values,
                negative_controls=controls,
            ),
            work_root=work_root / "matrix",
            artifact_root=Path(CACHE_MOUNT) / "pdf-redaction/q8",
            allocated_cpu_cores=CPU_CORES,
        )
    # reason: the two gate errors that carry candidate diagnostics are dispatched by `isinstance` inside this handler
    # reason: and everything else becomes a safe failure record. A failed matrix must produce a failure artifact,
    # reason: never an unhandled exit.
    except Exception as error:  # ruff: ignore[blind-except]
        if isinstance(error, OcrCandidateGateError):
            write_jsonl_once(
                output_root / "ocr-candidate-diagnostics.jsonl",
                error.candidate_rows,
                forbidden_values=forbidden_values,
            )
        if isinstance(error, ModelCandidateGateError):
            write_jsonl_once(
                output_root / "pii-candidate-diagnostics.jsonl",
                error.candidate_rows,
                forbidden_values=forbidden_values,
            )
        failure = safe_failure_record(
            stage="full_matrix",
            candidate_id="matrix",
            error=error,
            forbidden_values=forbidden_values,
        )
        write_json_once(
            output_root / "failure.json",
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "run_id": run_id,
                "stage": "full_matrix",
                "fixture_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                "preflight_run_id": Path(preflight_root).name,
                "failure": failure,
            },
            forbidden_values=forbidden_values,
        )
        output_volume.commit()
        msg = f"full matrix failed; evidence={output_root.name}; error_digest={str(failure['error_message_sha256'])[:12]}"
        raise RuntimeError(
            msg,
        ) from None

    preflight_run_id = Path(preflight_root).name
    preflight_evidence_sha256 = hashlib.sha256((Path(preflight_root) / "preflight.json").read_bytes()).hexdigest()
    result = {
        **execution.result,
        "run_id": run_id,
        "fixture_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "preflight_run_id": preflight_run_id,
        "candidate_pruning_evidence": {
            "candidate_ids": ["H1", "H2", "H3"],
            "gate": "writer_preflight",
            "preflight_run_id": preflight_run_id,
            "preflight_evidence_sha256": preflight_evidence_sha256,
            "reason": "dynamic_object_failure_before_ocr_or_pii_measurement",
        },
    }
    candidate_rows = result.get("candidate_results")
    if not isinstance(candidate_rows, list):
        msg = "full matrix result omitted candidate rows"
        raise RuntimeError(msg)
    for row in candidate_rows:
        if not is_object_dict(row) or row.get("candidate_id") not in {
            "H1",
            "H2",
            "H3",
        }:
            continue
        row["artifacts"] = [
            {
                "artifact_id": "writer_preflight_evidence",
                "version": preflight_run_id,
                "sha256": preflight_evidence_sha256,
            },
        ]
    write_json_once(
        output_root / "results.json",
        result,
        forbidden_values=forbidden_values,
    )
    write_jsonl_once(
        output_root / "candidates.jsonl",
        candidate_rows,
        forbidden_values=forbidden_values,
    )
    copy_verified_once(
        execution.promoted_output,
        output_root / "winner-h4-oracle-redacted.pdf",
        expected_sha256=execution.promoted_output_sha256,
    )
    output_volume.commit()
    return str(output_root)


def _run_private_fixture_impl(
    pdf_bytes: bytes,
) -> tuple[bytes, dict[str, object]]:
    if not pdf_bytes.startswith(b"%PDF-"):
        msg = "private fixture must be a PDF"
        raise ValueError(msg)
    if len(pdf_bytes) > MAX_PRIVATE_FIXTURE_BYTES:
        msg = "private fixture exceeds the 50 MiB limit"
        raise ValueError(msg)

    started = time.perf_counter()
    run_id = _run_id("private")
    with _private_fixture_workspace(run_id) as work_root:
        return _execute_private_fixture(
            pdf_bytes,
            run_id=run_id,
            work_root=work_root,
            started=started,
        )


# reason: Private-fixture copy, execution, verification, and cleanup share one ephemeral workspace.
def _execute_private_fixture(  # ruff: ignore[too-many-locals]
    pdf_bytes: bytes,
    *,
    run_id: str,
    work_root: Path,
    started: float,
) -> tuple[bytes, dict[str, object]]:
    from meddies_pii.pdf_redaction import (
        IndependentPdfVerifier,
        PdfGeometryPreparer,
        PdfiumDocumentAdapter,
        PdfiumExtractor,
        PdfRedactionPipeline,
        RapidOcrAdapter,
        RasterRebuildWriter,
        VerificationToolchain,
        create_default_detector,
    )

    source = work_root / "source.pdf"
    destination = work_root / "model-redacted-unreviewed.pdf"
    source.write_bytes(pdf_bytes)

    preparer = PdfGeometryPreparer(
        document_adapter=PdfiumDocumentAdapter(dpi=300),
        native_extractor=PdfiumExtractor(),
        ocr=RapidOcrAdapter(),
    )
    pipeline = PdfRedactionPipeline(
        detector=create_default_detector(
            Path(CACHE_MOUNT) / "pdf-redaction/q8",
            threads=8,
        ),
        writer=RasterRebuildWriter(dpi=200),
        verifier=IndependentPdfVerifier(toolchain=VerificationToolchain(timeout_seconds=60, render_dpi=150)),
    )

    prepare_started = time.perf_counter()
    prepared = preparer.prepare(source)
    prepare_seconds = time.perf_counter() - prepare_started

    preview_started = time.perf_counter()
    preview = pipeline.preview(prepared)
    preview_seconds = time.perf_counter() - preview_started

    apply_started = time.perf_counter()
    output = pipeline.apply_verified(
        preview=preview,
        reviewed=preview.suggestions,
        destination=destination,
    )
    apply_seconds = time.perf_counter() - apply_started
    output_bytes = output.path.read_bytes()
    if hashlib.sha256(output_bytes).hexdigest() != output.sha256:
        msg = "returned private fixture output digest mismatch"
        raise RuntimeError(msg)

    route_counts: dict[str, int] = {}
    for decision in prepared.decisions:
        route_counts[decision.content_route] = route_counts.get(decision.content_route, 0) + 1
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "benchmark_status": "unscored",
        "safety_status": "human_review_required",
        "source_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "source_bytes": len(pdf_bytes),
        "output_sha256": output.sha256,
        "output_bytes": len(output_bytes),
        "page_count": output.page_count,
        "ocr_page_count": sum(decision.requires_full_page_ocr for decision in prepared.decisions),
        "route_counts": route_counts,
        "detected_span_count": sum(len(detection.spans) for detection in preview.detections),
        "region_count": sum(len(item.quads) for item in preview.suggestions),
        "verification_scope": output.verification_scope,
        "verification_code_count": len(output.verification_codes),
        "timings_seconds": {
            "prepare": prepare_seconds,
            "preview": preview_seconds,
            "apply_and_verify": apply_seconds,
            "wall": time.perf_counter() - started,
        },
    }
    return output_bytes, result


@contextmanager
def _private_fixture_workspace(run_id: str) -> Iterator[Path]:
    with TemporaryDirectory(prefix=f"{run_id}-", dir="/tmp") as directory:
        root = Path(directory)
        root.chmod(0o700)
        yield root


def _read_private_fixture(source: Path) -> bytes:
    if source.suffix.lower() != ".pdf":
        msg = "private fixture source must use the .pdf extension"
        raise ValueError(msg)
    # reason: read private's try keeps read with s isreg; splitting would let sample setup drift.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        stream = os.fdopen(descriptor, "rb")
        with stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                msg = "private fixture source must be a regular file"
                raise ValueError(msg)
            pdf_bytes = stream.read(MAX_PRIVATE_FIXTURE_BYTES + 1)
    except FileNotFoundError:
        msg = "private fixture source PDF is missing"
        raise FileNotFoundError(msg) from None
    except OSError as error:
        msg = f"private fixture source PDF is unreadable ({type(error).__name__})"
        raise RuntimeError(msg) from None
    if len(pdf_bytes) > MAX_PRIVATE_FIXTURE_BYTES:
        msg = "private fixture exceeds the 50 MiB limit"
        raise ValueError(msg)
    if not pdf_bytes.startswith(b"%PDF-"):
        msg = "private fixture source is not a PDF"
        raise ValueError(msg)
    return pdf_bytes


def _require_distinct_private_fixture_paths(
    source: Path,
    destination: Path,
) -> None:
    try:
        same_path = source.resolve(strict=False) == destination.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        msg = f"private fixture path validation failed ({type(error).__name__})"
        raise RuntimeError(msg) from None
    if same_path:
        msg = "private fixture output must differ from its source"
        raise ValueError(msg)


def _write_private_fixture_output(
    destination: Path,
    output_bytes: bytes,
    *,
    expected_sha256: str,
) -> None:
    if destination.suffix.lower() != ".pdf":
        msg = "private fixture output must use the .pdf extension"
        raise ValueError(msg)
    if not output_bytes.startswith(b"%PDF-"):
        msg = "private fixture output is not a PDF"
        raise RuntimeError(msg)
    observed_sha256 = hashlib.sha256(output_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        msg = "private fixture output SHA-256 mismatch"
        raise RuntimeError(msg)
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode=0o600,
        )
    except FileExistsError:
        msg = "private fixture destination already exists"
        raise RuntimeError(msg) from None
    except OSError as error:
        msg = f"private fixture destination is not writable ({type(error).__name__})"
        raise RuntimeError(msg) from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(output_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        _remove_private_fixture_output(destination)
        msg = f"private fixture output write failed ({type(error).__name__})"
        raise RuntimeError(msg) from None
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        _remove_private_fixture_output(destination)
        raise


def _remove_private_fixture_output(destination: Path) -> None:
    try:
        destination.unlink(missing_ok=True)
    except OSError as error:
        msg = f"private fixture partial output cleanup failed ({type(error).__name__})"
        raise RuntimeError(msg) from None


def _regions_from_payload(
    records: list[dict[str, Any]],
) -> tuple[PageRegion, ...]:
    from meddies_pii.taxonomy import require_pii_label

    regions: list[PageRegion] = [
        PageRegion(
            page_index=int(record["page_index"]),
            quad=_quad_from_payload(record),
            label=require_pii_label(str(record["label"])),
            span_start=0,
            span_end=1,
        )
        for record in records
    ]
    return tuple(regions)


def _quad_from_payload(record: dict[str, Any]) -> Quad:
    raw_points = record["points"]
    if not isinstance(raw_points, list) or len(raw_points) != 4:  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "oracle region must contain four points"
        raise ValueError(msg)
    point_0, point_1, point_2, point_3 = raw_points
    return Quad(
        points=(
            Point(float(point_0[0]), float(point_0[1])),
            Point(float(point_1[0]), float(point_1[1])),
            Point(float(point_2[0]), float(point_2[1])),
            Point(float(point_3[0]), float(point_3[1])),
        ),
    )


def _run_id(stage: str) -> str:
    return f"pdf-redaction-{stage}-{int(time.time())}-{uuid4().hex[:8]}"
