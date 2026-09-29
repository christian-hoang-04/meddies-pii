from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.reports import inference_preview

if TYPE_CHECKING:
    from pathlib import Path


def test_inference_preview_writer_uses_wrapped_result_and_persists_html(
    tmp_path: Path,
) -> None:
    preview_json = tmp_path / "preview.json"
    result_json = tmp_path / "result.json"
    preview_json.write_text(
        json.dumps({
            "records": [
                {
                    "uid": "row-1",
                    "text": "Alice",
                    "gold_spans": [],
                    "predicted_spans": [],
                },
            ],
        }),
        encoding="utf-8",
    )
    result_json.write_text(json.dumps({"result": {"eval_exact_span_f1": 1.0}}), encoding="utf-8")

    output_path = inference_preview.write_inference_preview_report(
        preview_json=preview_json,
        result_json=result_json,
        output_dir=tmp_path / "report",
    )

    assert output_path == tmp_path / "report" / "index.html"
    assert output_path.exists()
    assert "BIOES inference preview" in output_path.read_text(encoding="utf-8")
    assert '<div class="value">1.000</div>' in output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("preview_payload", "result_payload", "message"),
    [
        (["invalid"], {"result": {}}, "Inference preview payload is invalid"),
        ({"records": []}, ["invalid"], "Training result payload is invalid"),
    ],
)
def test_inference_preview_writer_rejects_non_object_payloads(
    tmp_path: Path,
    preview_payload: object,
    result_payload: object,
    message: str,
) -> None:
    preview_json = tmp_path / "preview.json"
    result_json = tmp_path / "result.json"
    preview_json.write_text(json.dumps(preview_payload), encoding="utf-8")
    result_json.write_text(json.dumps(result_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        inference_preview.write_inference_preview_report(
            preview_json=preview_json,
            result_json=result_json,
            output_dir=tmp_path / "report",
        )
