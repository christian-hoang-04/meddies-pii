from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the redaction runtime is an optional extra, and several backends are resolved by name at call time.
import importlib.metadata
import math
import os
import platform
import resource

# reason: environment capture runs fixed version probes as list-form argv; it never processes document text as a command.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypdf.generic import RectangleObject

    from meddies_pii.pdf_redaction.benchmark.spatial import SpatialCoverage


def _automated_redaction_status(
    coverage: SpatialCoverage,
    *,
    outputs_verified: bool,
) -> tuple[bool, str]:
    # reason: the containment gate demands complete gold-region coverage and zero undercoverage; a
    # reason: tolerance here would pass a redaction that left part of a PII region visible.
    containment_passed = (
        coverage.gold_region_recall == 1.0  # ruff: ignore[float-equality-comparison]
        and coverage.undercoverage_rate == 0.0  # ruff: ignore[float-equality-comparison]
        and coverage.negative_control_touches == 0
    )
    if not containment_passed:
        return False, "failed_model_containment_gate"
    if coverage.overredaction_rate > 0.0:
        return False, "failed_model_overredaction_gate"
    if not outputs_verified:
        return False, "failed_model_output_verification"
    return True, "eligible"


def _page_invariants(source: Path, output: Path) -> tuple[bool, tuple[int, ...]]:
    from pypdf import PdfReader

    source_pages = PdfReader(source, strict=True).pages
    output_pages = PdfReader(output, strict=True).pages
    if len(source_pages) != len(output_pages):
        return False, ()

    normalized: list[int] = []
    for page_index, (source_page, output_page) in enumerate(zip(source_pages, output_pages, strict=True)):
        source_media = _box_coordinates(source_page.mediabox)
        source_crop = _box_coordinates(source_page.cropbox)
        output_media = _box_coordinates(output_page.mediabox)
        output_crop = _box_coordinates(output_page.cropbox)
        source_rotation = int(source_page.get("/Rotate", 0) or 0) % 360
        output_rotation = int(output_page.get("/Rotate", 0) or 0) % 360
        if source_rotation not in {0, 90, 180, 270} or output_rotation not in {
            0,
            90,
            180,
            270,
        }:
            return False, tuple(normalized)
        if source_rotation == output_rotation:
            if not _same_box(source_media, output_media) or not _same_box(source_crop, output_crop):
                return False, tuple(normalized)
            continue
        if not _is_full_box_rotation_normalization(
            source_media=source_media,
            source_crop=source_crop,
            source_rotation=source_rotation,
            output_media=output_media,
            output_crop=output_crop,
            output_rotation=output_rotation,
        ):
            return False, tuple(normalized)
        normalized.append(page_index)
    return True, tuple(normalized)


def _box_coordinates(box: RectangleObject) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in box)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
        msg = "PDF page box must contain four finite coordinates"
        raise ValueError(msg)
    return values


def _same_box(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(left, right, strict=True))


# reason: is full box keeps source media/output at its adapter seam; bundling would hide required inputs.
def _is_full_box_rotation_normalization(  # ruff: ignore[too-many-arguments]
    *,
    source_media: tuple[float, float, float, float],
    source_crop: tuple[float, float, float, float],
    source_rotation: int,
    output_media: tuple[float, float, float, float],
    output_crop: tuple[float, float, float, float],
    output_rotation: int,
) -> bool:
    if source_rotation == 0 or output_rotation != 0:
        return False
    if not _same_box(source_media, source_crop):
        return False
    if not _same_box(output_media, output_crop):
        return False
    if not _same_box(
        (source_media[0], source_media[1], 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ) or not _same_box(
        (output_media[0], output_media[1], 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ):
        return False
    source_width = source_media[2] - source_media[0]
    source_height = source_media[3] - source_media[1]
    expected_width, expected_height = (
        (source_height, source_width) if source_rotation in {90, 270} else (source_width, source_height)
    )
    return _same_box(
        output_media,
        (0.0, 0.0, expected_width, expected_height),
    )


def _installed_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _environment_identity(*, allocated_cpu_cores: float) -> dict[str, object]:
    package_names = (
        "onnxruntime",
        "openvino",
        "pillow",
        "pymupdf",
        "pypdf",
        "pypdfium2",
        "rapidocr",
        "reportlab",
        "torch",
        "transformers",
    )
    packages = {package: _installed_package_version(package) for package in package_names}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host_visible_cpu_count": os.cpu_count(),
        "allocated_cpu_cores": allocated_cpu_cores,
        "cpu_model": _cpu_model(),
        "packages": packages,
        "tools": {
            "qpdf": _tool_version(("qpdf", "--version")),
            "poppler": _tool_version(("pdftotext", "-v")),
            "tesseract": _tool_version(("tesseract", "--version")),
        },
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        rows = cpuinfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return platform.processor() or "unavailable"
    for row in rows:
        key, separator, value = row.partition(":")
        if separator and key.strip() in {"model name", "Hardware"}:
            return value.strip()[:200] or "unavailable"
    return platform.processor() or "unavailable"


def _tool_version(command: tuple[str, ...]) -> str:
    try:
        # reason: callers are the fixed qpdf, pdftotext, and tesseract version probes above; no shell parses argv.
        completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    payload = completed.stdout or completed.stderr
    line = payload.decode("utf-8", errors="replace").splitlines()
    return line[0][:200] if line else f"exit-{completed.returncode}"


def _peak_rss_bytes() -> int:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, ValueError):
        return 1
    multiplier = 1 if sys.platform == "darwin" else 1024
    return max(1, value * multiplier)


def _current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    try:
        rows = status.read_text(encoding="utf-8").splitlines()
    except OSError:
        return _peak_rss_bytes()
    for row in rows:
        if not row.startswith("VmRSS:"):
            continue
        fields = row.split()
        if len(fields) == 3 and fields[2] == "kB" and fields[1].isdigit():  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
            return max(1, int(fields[1]) * 1024)
        break
    return _peak_rss_bytes()
