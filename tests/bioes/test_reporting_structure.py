from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.training.bioes.eval.audit import EvalAuditIssue, audit_records
from meddies_pii.training.bioes.reports import eval_prediction_audit, inference_preview, training_setup_preview
from meddies_pii.training.bioes.reports.eval_prediction_audit_html import (
    render_eval_prediction_audit_html,
)

if TYPE_CHECKING:
    import pytest

_GOLDEN_HTML_SHA256 = {
    "audit": "99de1632193aec2f335709067a6f6ee8bd58dc823ec9fcddaaf0f03dafb8357f",
    "inference": "3452510bb230979a526d4997162ef9db927a65397258bc56eac06417e4f84a2b",
    "setup": "f1f8eeeaf69ee751743bf0df8c48a9bc28771ad544e4f9f146158958593cfa06",
}
_FIXED_GENERATED_AT = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


class _FixedDatetime:
    @classmethod
    def now(cls, timezone: object) -> datetime:
        assert timezone is UTC
        return _FIXED_GENERATED_AT


def _records() -> list[dict[str, object]]:
    return [
        {
            "uid": "row-<script>alert(1)</script>",
            "text": "Patient <img src=x onerror=alert(1)> Alice",
            "gold_spans": [{"label": "human_name", "start": 35, "end": 40, "text": "Alice"}],
            "predicted_spans": [{"label": "human_name", "start": 35, "end": 40, "text": "Alice"}],
        },
    ]


def _write_jsonl(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_report_documents_match_fixed_clock_golden_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(eval_prediction_audit, "datetime", _FixedDatetime)
    monkeypatch.setattr(training_setup_preview, "datetime", _FixedDatetime)
    records = _records()
    issues = audit_records(records)

    audit_path = eval_prediction_audit.write_eval_prediction_audit_outputs(
        predictions_json=Path("hostile <input>.json"),
        output_dir=tmp_path / "audit",
        records=records,
        issues=issues,
        max_issues=50,
    )
    inference_html = inference_preview.render_inference_preview_html(
        preview={"records": records, "provenance": {"checkpoint": "<checkpoint>"}},
        result={
            "config": {"dataset_id": "<dataset>", "max_length": 128},
            "eval_exact_span_f1": 1.0,
        },
        preview_json_label="<preview.json>",
        generated_at=_FIXED_GENERATED_AT.isoformat(),
    )
    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    record: dict[str, object] = {
        "text": "Patient <img> Alice",
        "label": [{"category": "human_name", "start": 14, "end": 19}],
        "info": {"id": "<row>"},
    }
    _write_jsonl(train_jsonl, record)
    _write_jsonl(validation_jsonl, record)
    split_summary = tmp_path / "split.json"
    split_summary.write_text("{}", encoding="utf-8")
    dataset_summary = tmp_path / "dataset.json"
    dataset_summary.write_text("{}", encoding="utf-8")
    setup_path = training_setup_preview.render_html(
        train_jsonl=train_jsonl,
        validation_jsonl=validation_jsonl,
        split_summary_json=split_summary,
        dataset_summary_json=dataset_summary,
        smoke_result_json=None,
        output_dir=tmp_path / "setup",
        sample_count=1,
    )

    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == _GOLDEN_HTML_SHA256["audit"]
    assert hashlib.sha256(inference_html.encode()).hexdigest() == _GOLDEN_HTML_SHA256["inference"]
    assert hashlib.sha256(setup_path.read_bytes()).hexdigest() == _GOLDEN_HTML_SHA256["setup"]


def test_audit_report_escapes_hostile_issue_values() -> None:
    payload = "<script>alert(1)</script>"
    issue = EvalAuditIssue(
        uid=payload,
        issue_type=payload,
        severity="high",
        label=payload,
        start=0,
        end=len(payload),
        text=payload,
        reason=payload,
        recommended_action=payload,
        context=payload,
        evidence={"payload": payload},
    )

    report = render_eval_prediction_audit_html(
        records=[{"uid": payload, "text": payload}],
        issues=[issue],
        predictions_json=Path(payload),
        max_issues=1,
        generated_at=_FIXED_GENERATED_AT.isoformat(),
    )

    assert payload not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report


def test_inference_preview_escapes_hostile_record_values() -> None:
    payload = "<script>alert(1)</script>"

    report = inference_preview.render_inference_preview_html(
        preview={
            "records": [
                {
                    "uid": payload,
                    "text": payload,
                    "gold_spans": [],
                    "predicted_spans": [],
                },
            ],
        },
        result={},
        preview_json_label=payload,
        generated_at=_FIXED_GENERATED_AT.isoformat(),
    )

    assert payload not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
