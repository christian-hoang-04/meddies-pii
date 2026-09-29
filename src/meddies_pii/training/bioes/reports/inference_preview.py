"""Persist the BIOES inference preview and preserve its public API."""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: these validators state one contract, malformed payload -> ValueError, and raise it from isinstance
# reason: and non-isinstance guards alike inside the same function. Converting only the isinstance ones would
# reason: split that contract on the guard shape rather than on what went wrong, and tests match the type.
import json
from collections.abc import Mapping
from pathlib import Path

from .inference_preview_html import render_inference_preview_html
from .inference_preview_metrics import compare_spans, preview_metrics
from .inference_preview_style import SCRIPT, STYLE

__all__ = [
    "DEFAULT_INFERENCE_JSON",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RESULT_JSON",
    "SCRIPT",
    "STYLE",
    "compare_spans",
    "preview_metrics",
    "render_inference_preview_html",
    "write_inference_preview_report",
]

DEFAULT_OUTPUT_DIR = Path("reports/bioes-inference-previews/lfm25-bioes-full-8192-r128-a256-bs128-step150")
DEFAULT_RESULT_JSON = DEFAULT_OUTPUT_DIR / "training_result.json"
DEFAULT_INFERENCE_JSON = DEFAULT_OUTPUT_DIR / "inference_preview.json"


def write_inference_preview_report(
    *,
    preview_json: Path,
    result_json: Path,
    output_dir: Path,
) -> Path:
    """Render persisted preview and training-result JSON into ``index.html``.

    Returns:
        The path of the written ``index.html``, already on disk. Nothing is written unless
        BOTH payloads validate, so a failed call leaves no partial report behind.

    Raises:
        ValueError: When either payload decodes to something other than a mapping. The
            result file is accepted in two shapes -- a bare result mapping, or one wrapped
            under a ``result`` key, which is unwrapped first -- so this fires only when
            neither shape is a mapping.

    """
    preview = json.loads(preview_json.read_text(encoding="utf-8"))
    if not isinstance(preview, Mapping):
        msg = f"Inference preview payload is invalid: {preview_json}"
        raise ValueError(msg)
    result_payload = json.loads(result_json.read_text(encoding="utf-8"))
    result = result_payload.get("result", result_payload) if isinstance(result_payload, Mapping) else result_payload
    if not isinstance(result, Mapping):
        msg = f"Training result payload is invalid: {result_json}"
        raise ValueError(msg)
    html = render_inference_preview_html(
        preview=preview,
        result=result,
        preview_json_label=str(preview_json),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
