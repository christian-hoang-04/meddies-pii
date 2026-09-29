from __future__ import annotations

import hashlib
import json
import os
import runpy
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from meddies_pii.json_types import is_str_mapping
from meddies_pii.pdf_redaction.harness import (
    UnsafeResultPayloadError,
    copy_verified_once,
    safe_failure_record,
    safe_json_bytes,
    write_json_once,
    write_jsonl_once,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

MODAL_SCRIPT = Path(__file__).parents[2] / "scripts" / "ops" / "run_pdf_redaction_benchmark.py"


def _load_modal_helpers() -> dict[str, object]:
    namespace: object = runpy.run_path(str(MODAL_SCRIPT))
    assert is_str_mapping(namespace)
    return dict(namespace)


def test_safe_json_bytes_rejects_raw_canary_at_any_depth() -> None:
    canary = "private-synthetic-value"

    with pytest.raises(UnsafeResultPayloadError, match="forbidden synthetic value"):
        safe_json_bytes(
            {"rows": [{"detail": f"found {canary}"}]},
            forbidden_values=(canary,),
        )


def test_safe_json_bytes_rejects_canonically_equivalent_canary() -> None:
    canary = "Nguyễn Văn A"
    decomposed = unicodedata.normalize("NFD", canary)

    with pytest.raises(UnsafeResultPayloadError, match="forbidden synthetic value"):
        safe_json_bytes(
            {"detail": decomposed},
            forbidden_values=(canary,),
        )


def test_safe_json_bytes_is_deterministic_and_utf8() -> None:
    payload = {"route": "native_trusted", "note": "Tiếng Việt", "count": 10}

    first = safe_json_bytes(payload, forbidden_values=("unpublished-canary",))
    second = safe_json_bytes(payload, forbidden_values=("unpublished-canary",))

    assert first == second
    assert json.loads(first) == payload
    assert "Tiếng Việt".encode() in first


def test_write_json_once_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    write_json_once(
        destination,
        {"status": "passed"},
        forbidden_values=("private-synthetic-value",),
    )

    with pytest.raises(FileExistsError):
        write_json_once(
            destination,
            {"status": "changed"},
            forbidden_values=("private-synthetic-value",),
        )

    assert json.loads(destination.read_bytes()) == {"status": "passed"}


def test_write_jsonl_once_emits_one_safe_record_per_line(tmp_path: Path) -> None:
    destination = tmp_path / "results.jsonl"

    write_jsonl_once(
        destination,
        ({"candidate_id": "H1"}, {"candidate_id": "H4"}),
        forbidden_values=("private-synthetic-value",),
    )

    assert [json.loads(line) for line in destination.read_bytes().splitlines()] == [
        {"candidate_id": "H1"},
        {"candidate_id": "H4"},
    ]
    with pytest.raises(FileExistsError):
        write_jsonl_once(
            destination,
            ({"candidate_id": "H2"},),
            forbidden_values=("private-synthetic-value",),
        )


def test_copy_verified_once_checks_identity_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "winner.pdf"
    source.write_bytes(b"verified-pdf-bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    copy_verified_once(source, destination, expected_sha256=expected)

    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        copy_verified_once(source, destination, expected_sha256=expected)


def test_copy_verified_once_deletes_digest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "winner.pdf"
    source.write_bytes(b"wrong-bytes")

    with pytest.raises(RuntimeError, match="copied output SHA-256 mismatch"):
        copy_verified_once(source, destination, expected_sha256="0" * 64)

    assert not destination.exists()


def test_failure_record_preserves_diagnosis_without_canary() -> None:
    canary = "private-synthetic-value"

    record = safe_failure_record(
        stage="verify",
        candidate_id="pymupdf-object-aware",
        error=RuntimeError(f"qpdf recovered {canary} from object 17"),
        forbidden_values=(canary,),
    )
    encoded = safe_json_bytes(record, forbidden_values=(canary,))
    assert canary.encode() not in encoded
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "qpdf recovered <synthetic:c9c029bfee9c> from object 17"


def test_failure_record_redacts_canonically_equivalent_canary() -> None:
    canary = "Nguyễn Văn A"
    decomposed = unicodedata.normalize("NFD", canary)

    record = safe_failure_record(
        stage="verify",
        candidate_id="pdfium-raster",
        error=RuntimeError(f"recovered {decomposed}"),
        forbidden_values=(canary,),
    )

    encoded = safe_json_bytes(record, forbidden_values=(canary,))
    assert decomposed.encode() not in encoded
    assert canary.encode() not in encoded
    assert str(record["error_message"]).startswith("recovered <synthetic:")


def test_failure_record_preserves_safe_stack_frames() -> None:
    def fail_inside_test() -> None:
        msg = "list index out of range"
        raise IndexError(msg)

    with pytest.raises(IndexError) as caught:
        fail_inside_test()
    record = safe_failure_record(
        stage="pii_gate",
        candidate_id="openvino-cpu",
        error=caught.value,
        forbidden_values=("private-synthetic-value",),
    )

    frames = record["stack_frames"]
    assert isinstance(frames, list)
    last = frames[-1]
    assert is_str_mapping(last)
    assert last["file"] == "test_modal_harness.py"
    assert last["function"] == "fail_inside_test"
    assert isinstance(last["line"], int)


def test_modal_script_pins_cpu_only_synthetic_bounded_execution() -> None:
    source = MODAL_SCRIPT.read_text()

    assert 'OUTPUT_VOLUME_NAME = "meddies-pii-pdf-redaction-benchmark"' in source
    assert 'CACHE_VOLUME_NAME = "hf-cache"' in source
    assert 'modal.Secret.from_name("huggingface-secret")' in source
    assert "CPU_CORES = 8.0" in source
    assert "TIMEOUT_SECONDS = 30 * 60" in source
    assert "max_containers=1" in source
    assert "@app.local_entrypoint()\ndef main()" in source
    assert "gpu=" not in source
    assert "generate_challenge_corpus" in source


def test_full_matrix_is_real_and_uses_the_runtime_image() -> None:
    source = MODAL_SCRIPT.read_text()

    assert "full matrix implementation has not passed" not in source
    assert "@app.function(\n    image=benchmark_image" in source
    assert '"stage": "full_matrix"' in source
    assert "isinstance(error, OcrCandidateGateError)" in source
    assert 'output_root / "ocr-candidate-diagnostics.jsonl"' in source
    assert "isinstance(error, ModelCandidateGateError)" in source
    assert 'output_root / "pii-candidate-diagnostics.jsonl"' in source
    assert '"preflight_evidence_sha256": preflight_evidence_sha256' in source
    assert '"writer_preflight_evidence"' in source


def test_private_fixture_entrypoint_is_unscored_ephemeral_and_bounded() -> None:
    source = MODAL_SCRIPT.read_text()

    assert "MAX_PRIVATE_FIXTURE_BYTES = 50 * 1024 * 1024" in source
    assert "def run_private_fixture(" in source
    assert "def private_fixture(" in source
    assert '"benchmark_status": "unscored"' in source
    assert '"safety_status": "human_review_required"' in source
    assert "PdfRedactionPipeline(" in source
    assert "RasterRebuildWriter(dpi=200)" in source
    assert "create_default_detector(" in source
    assert "threads=8" in source
    assert '"destination": str(destination_path)' not in source
    assert '"region_count": sum(' in source
    assert "len(item.quads) for item in preview.suggestions" in source
    assert "mode=0o600" in source
    assert "root.chmod(0o700)" in source

    private_impl = source.split("def _run_private_fixture_impl(", maxsplit=1)[1]
    private_impl = private_impl.split("def _regions_from_payload(", maxsplit=1)[0]
    assert "output_volume.commit()" not in private_impl
    assert "with _private_fixture_workspace(run_id) as work_root:" in private_impl
    assert "source.stat().st_size" not in source
    assert "source.read_bytes()" not in source


def test_private_fixture_workspace_is_removed_after_success_and_failure() -> None:
    namespace = _load_modal_helpers()
    workspace = cast(
        "Callable[[str], AbstractContextManager[Path]]",
        namespace["_private_fixture_workspace"],
    )

    with workspace("private-test-success") as root:
        success_root = root
        (root / "source.pdf").write_bytes(b"%PDF-private")
        assert root.stat().st_mode & 0o777 == 0o700
    assert not success_root.exists()

    failure_root: Path | None = None
    # reason: `raises` must wrap the workspace context manager so the workspace `__exit__` runs inside it; that
    # reason: teardown-on-exception is exactly what the `not failure_root.exists()` assertion below proves.
    with (  # ruff: ignore[pytest-raises-with-multiple-statements]
        pytest.raises(RuntimeError, match="synthetic failure"),
        workspace("private-test-failure") as root,
    ):
        failure_root = root
        (root / "source.pdf").write_bytes(b"%PDF-private")
        msg = "synthetic failure"
        raise RuntimeError(msg)
    assert failure_root is not None
    assert not failure_root.exists()


def test_private_fixture_io_never_exposes_caller_paths(tmp_path: Path) -> None:
    namespace = _load_modal_helpers()
    read_private = cast(
        "Callable[[Path], bytes]",
        namespace["_read_private_fixture"],
    )
    write_private = cast(
        "Callable[..., None]",
        namespace["_write_private_fixture_output"],
    )
    validate_paths = cast(
        "Callable[[Path, Path], None]",
        namespace["_require_distinct_private_fixture_paths"],
    )
    missing = tmp_path / "maya-sato-HIV-source.pdf"
    with pytest.raises(FileNotFoundError, match="private fixture source PDF is missing") as read_error:
        read_private(missing)
    assert missing.name not in str(read_error.value)
    assert str(missing) not in str(read_error.value)

    existing = tmp_path / "maya-sato-HIV-result.pdf"
    existing.write_bytes(b"keep")
    with pytest.raises((ValueError, RuntimeError)) as write_error:
        write_private(
            existing,
            b"%PDF-private",
            expected_sha256=hashlib.sha256(b"%PDF-private").hexdigest(),
        )
    assert existing.name not in str(write_error.value)
    assert str(existing) not in str(write_error.value)
    assert existing.read_bytes() == b"keep"

    source_loop = tmp_path / "maya-sato-HIV-source-loop.pdf"
    source_loop.symlink_to(source_loop)
    with pytest.raises((ValueError, RuntimeError)) as path_error:
        validate_paths(source_loop, tmp_path / "output.pdf")
    assert source_loop.name not in str(path_error.value)
    assert str(source_loop) not in str(path_error.value)


def test_private_fixture_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    if os.name != "posix" or not hasattr(os, "mkfifo"):
        pytest.skip("FIFO probe requires POSIX")
    namespace = _load_modal_helpers()
    read_private = cast(
        "Callable[[Path], bytes]",
        namespace["_read_private_fixture"],
    )
    source = tmp_path / "private-source.pdf"
    os.mkfifo(source)

    with pytest.raises(ValueError, match="regular file"):
        read_private(source)
