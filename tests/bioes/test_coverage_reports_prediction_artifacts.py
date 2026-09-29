from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.eval.audit import EvalAuditIssue, Severity
from meddies_pii.training.bioes.reports import eval_prediction_artifacts

if TYPE_CHECKING:
    from pathlib import Path


def _issue(
    uid: str,
    action: str,
    *,
    label: str = "email_address",
    severity: Severity = "high",
    start: int = 1,
) -> EvalAuditIssue:
    return EvalAuditIssue(
        uid=uid,
        issue_type="test_issue",
        severity=severity,
        label=label,
        start=start,
        end=start + 2,
        text=f"{uid}-text",
        reason="test",
        recommended_action=action,
        context="context",
        evidence={"uid": uid},
    )


def test_prediction_artifact_reader_and_jsonl_writer_validate_payload_shape(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps({"records": [{"uid": "kept"}, ["discarded"], "discarded"]}),
        encoding="utf-8",
    )

    assert eval_prediction_artifacts.read_prediction_records(prediction_path) == [{"uid": "kept"}]
    written = tmp_path / "nested" / "rows.jsonl"
    eval_prediction_artifacts.write_jsonl(written, [{"z": 1, "a": "đ"}])
    assert written.read_text(encoding="utf-8") == '{"a": "đ", "z": 1}\n'

    bad_payload = tmp_path / "bad.json"
    bad_payload.write_text(json.dumps({"records": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="has no records list"):
        eval_prediction_artifacts.read_prediction_records(bad_payload)


def test_adjudication_batches_prioritize_once_sort_and_replace_stale_jsonl(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "audit" / "adjudication_batches"
    batch_dir.mkdir(parents=True)
    (batch_dir / "stale.jsonl").write_text("stale\n", encoding="utf-8")
    issues = [
        _issue("add", "review_gold_add_predicted_span"),
        _issue("company", "review_regex_candidate", label="company_name"),
        _issue("remove", "review_gold_remove_or_relabel"),
        _issue("mismatch", "review_gold_or_prediction_label"),
        _issue("boundary", "review_boundary_policy"),
        _issue("private", "review_regex_candidate", label="private_url", severity="medium"),
        _issue("miss", "review_model_miss", severity="low"),
        _issue("regex", "review_regex_candidate"),
        _issue("nested", "review_nested_candidate_conflict"),
        _issue("false-positive", "review_model_false_positive"),
        _issue("unbatched", "review_other"),
    ]

    eval_prediction_artifacts.write_adjudication_batches(tmp_path / "audit", issues)

    assert not (batch_dir / "stale.jsonl").exists()
    assert (
        json.loads((batch_dir / "01_metric_predicted_gold_add.jsonl").read_text())
        == _issue("add", "review_gold_add_predicted_span").to_dict()
    )
    assert (
        json.loads((batch_dir / "02_company_regex_candidates.jsonl").read_text())
        == _issue("company", "review_regex_candidate", label="company_name").to_dict()
    )
    assert (
        json.loads((batch_dir / "06_private_url_secret_review.jsonl").read_text())
        == _issue("private", "review_regex_candidate", label="private_url", severity="medium").to_dict()
    )
    summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
    assert [item["file"] for item in summary["batches"]] == [
        "01_metric_predicted_gold_add.jsonl",
        "02_company_regex_candidates.jsonl",
        "03_gold_remove_or_relabel.jsonl",
        "04_label_mismatch_review.jsonl",
        "05_boundary_policy_review.jsonl",
        "06_private_url_secret_review.jsonl",
        "07_model_miss_review.jsonl",
        "08_regex_candidate_remainder.jsonl",
        "09_nested_candidate_conflicts.jsonl",
        "10_model_false_positive_review.jsonl",
        "99_unbatched_review.jsonl",
    ]
    assert "reviewer queues, not automatic gold edits" in (batch_dir / "README.md").read_text(encoding="utf-8")


def test_adjudication_batches_emit_empty_named_queues_without_remainder(
    tmp_path: Path,
) -> None:
    eval_prediction_artifacts.write_adjudication_batches(tmp_path, [])

    batch_dir = tmp_path / "adjudication_batches"
    assert len(list(batch_dir.glob("*.jsonl"))) == 10
    assert "99_unbatched_review.jsonl" not in {path.name for path in batch_dir.glob("*.jsonl")}
