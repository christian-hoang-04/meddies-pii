from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.training.bioes.reports import html
from meddies_pii.training.bioes.reports.models import (
    LabelExample,
    RowPreview,
    TrainingReportOptions,
    TrainingScan,
)

if TYPE_CHECKING:
    import pytest

_FIXED_GENERATED_AT = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


class _FixedDatetime:
    @classmethod
    def now(cls, _timezone: object) -> datetime:
        return _FIXED_GENERATED_AT


def _render_report(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(html, "datetime", _FixedDatetime)
    options = TrainingReportOptions(
        train_jsonl=Path("train <untrusted>.jsonl"),
        output_html=Path("report.html"),
        title="Meddies <script>alert('title')</script> & readiness",
        rows_per_label=2,
        include_all_rows=True,
    )
    scan = TrainingScan(
        rows_seen=1,
        span_count=1,
        invalid_rows=["row <bad> & malformed"],
        label_counts=Counter({"human_name": 1}),
        label_examples={
            "human_name": [
                LabelExample(
                    row_index=1,
                    example_id="row <unsafe> & 1",
                    label="human_name",
                    span_text="<Alice> & Bob",
                    snippet_html='Patient <mark class="pii">&lt;Alice&gt;</mark> &amp; Bob',
                ),
            ],
        },
        row_previews=[
            RowPreview(
                row_index=1,
                example_id="row <unsafe> & 1",
                labels=("human_name",),
                text_len=13,
                span_count=1,
                snippet_html='Patient <mark class="pii">&lt;Alice&gt;</mark> &amp; Bob',
            ),
        ],
    )
    modal = {
        "provenance": {
            "modal_url": "https://modal.example/run?value=<unsafe>",
            "command_kwargs": {"gpu": "H100 <unsafe>", "steps": 3},
        },
        "result": {
            "config": {
                "model_id": "model <unsafe>",
                "dataset_id": "dataset & unsafe",
                "max_length": 512,
                "batch_size": 2,
                "packing": False,
            },
            "backend": "hf",
            "steps": 3,
        },
    }
    return html.render_training_report_html(
        options=options,
        scan=scan,
        validation_scan=None,
        summary={},
        split_summary={},
        dataset_summary={},
        audit_samples={},
        adversarial=[],
        modal=modal,
        todo_text="TODO <script>alert('todo')</script> & keep",
        progress_text="Progress <unsafe> & keep",
    )


def test_render_training_report_html_escapes_untrusted_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _render_report(monkeypatch)

    assert "<script>alert('title')</script>" not in report
    assert "&lt;script&gt;alert(&#x27;title&#x27;)&lt;/script&gt;" in report
    assert "TODO &lt;script&gt;alert(&#x27;todo&#x27;)&lt;/script&gt; &amp; keep" in report
    assert "--gpu H100 &lt;unsafe&gt;" in report
    assert "No validation JSONL attached." in report
    assert "No audit samples attached." in report
    assert "No rows attached." in report
    assert all(not line or line.strip() for line in report.splitlines()), (
        "rendered HTML must not contain whitespace-only lines"
    )
