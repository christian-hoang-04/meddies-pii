from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from meddies_pii.training.bioes.reports import render_training_report_html
from meddies_pii.training.bioes.reports.models import (
    TrainingReportOptions,
    TrainingScan,
)

if TYPE_CHECKING:
    from pathlib import Path


def _options(tmp_path: Path) -> TrainingReportOptions:
    return TrainingReportOptions(train_jsonl=tmp_path / "train.jsonl", output_html=tmp_path / "report.html")


def test_training_report_renders_populated_split_and_external_policy_evidence(
    tmp_path: Path,
) -> None:
    report = render_training_report_html(
        options=_options(tmp_path),
        scan=TrainingScan(rows_seen=8, span_count=3, label_counts=Counter({"human_name": 3})),
        validation_scan=TrainingScan(rows_seen=2, span_count=1),
        summary={},
        split_summary={
            "input_rows": 10,
            "train": {
                "rows": 8,
                "domain_bucket_counts": {"clinical": 6, "other": 2},
                "language_bucket_counts": {"vi": 8},
                "source_counts": {"<train-source>": 8},
                "label_doc_counts": {"human_name": 5},
            },
            "validation": {
                "rows": 2,
                "domain_bucket_counts": {"clinical": 1, "other": 1},
                "language_bucket_counts": {"vi": 1, "en": 1},
                "source_counts": {"<validation-source>": 2},
                "label_doc_counts": {"human_name": 1},
            },
            "split_validation": {
                "train_validation_id_overlap": 0,
                "train_validation_text_overlap": 1,
            },
        },
        dataset_summary={
            "language_policy": "Only <approved> language codes.",
            "all_rows_summary": {
                "dropped_external_spans": {
                    "external:unsupported_language:xx": 3,
                    "external:unsupported_language:yy": 1,
                    "other": 99,
                },
            },
        },
        audit_samples={},
        adversarial=[],
        modal={},
        todo_text="",
        progress_text="",
    )

    assert 'total rows before split</td><td class="num">10' in report
    assert "&lt;train-source&gt;" in report
    assert "75.00%" in report
    assert 'human_name</code></td><td class="num">5' in report
    assert "Only &lt;approved&gt; language codes." in report
    assert report.index("unsupported_language:xx") < report.index("unsupported_language:yy")


def test_training_report_renders_empty_split_tables_and_default_policy(
    tmp_path: Path,
) -> None:
    report = render_training_report_html(
        options=_options(tmp_path),
        scan=TrainingScan(rows_seen=4),
        validation_scan=None,
        summary={},
        split_summary={},
        dataset_summary={},
        audit_samples={},
        adversarial=[],
        modal={},
        todo_text="",
        progress_text="",
    )

    assert 'total rows before split</td><td class="num">4' in report
    assert report.count("No split count summary attached.") == 3
    assert "No label document support summary attached." in report
    assert "No unsupported-language skips recorded." in report
