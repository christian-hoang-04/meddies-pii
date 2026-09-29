from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.reports import training_data_breakdown
from meddies_pii.training.bioes.reports.training_data_breakdown import (
    TrainingDataSourceConfig,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {"label": [{"category": "human_name"}, {"label": "email_address"}]},
            (["human_name", "email_address"], "label-dicts"),
        ),
        (
            {"label": [[0, 4, "date"], [5, 8, "secret"]]},
            (["date", "secret"], "label-triples"),
        ),
        ({"text": "[Alice]<human_name>"}, (["human_name"], "inline-tags")),
        ({"text": "plain"}, ([], "no-labels")),
    ],
)
def test_training_data_row_label_formats_are_reported(record: dict[str, object], expected: tuple[list[str], str]) -> None:
    assert training_data_breakdown.row_labels(record) == expected


def test_training_data_scan_and_report_preserve_sampled_evidence(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "\n".join([
            json.dumps({
                "text": "Patient <unsafe> Alice",
                "label": [
                    {"category": "human_name"},
                    {"label": "not_pii"},
                ],
                "info": {"language": "VI"},
            }),
            json.dumps({
                "text": "Contact",
                "label": [[0, 3, "email_address"]],
                "lang": "en",
            }),
            json.dumps({"text": "[token]<secret>", "language": "FR"}),
            "{not json",
            json.dumps({"text": "beyond sample", "language": "de"}),
            "",
        ])
        + "\n",
        encoding="utf-8",
    )
    scanned_config = TrainingDataSourceConfig(
        name="<flagship>",
        role="internal-augmented",
        note="<note>",
        paths=(source_path, tmp_path / "missing.jsonl"),
        recommend="include",
    )
    scanned = training_data_breakdown.scan_training_data_source(scanned_config, sample_limit=4)
    unscanned = training_data_breakdown.scan_training_data_source(
        TrainingDataSourceConfig(
            name="remote",
            role="external-raw",
            note="remote only",
            paths=(),
            recommend="convert-first",
        ),
    )
    missing = training_data_breakdown.scan_training_data_source(
        TrainingDataSourceConfig(
            name="missing",
            role="eval",
            note="missing local",
            paths=(tmp_path / "nope.jsonl",),
            recommend="eval-only",
        ),
    )

    assert scanned.rows == 6
    assert scanned.existing_paths == 1
    assert scanned.fmt == "label-dicts, label-triples"
    assert scanned.labels == {"human_name": 1, "email_address": 1, "secret": 1}
    assert scanned.invalid_labels == {"not_pii": 1}
    assert scanned.languages == {"vi": 1, "en": 1, "fr": 1}
    assert scanned.sample_text == "Patient <unsafe> Alice"
    assert training_data_breakdown.row_language({"info": {"lang": " DE "}}) == "de"
    assert training_data_breakdown.row_language({"language": "PT"}) == "pt"
    assert training_data_breakdown.row_language({}) == "?"

    output_path = training_data_breakdown.write_training_data_breakdown_report(
        tmp_path / "nested" / "breakdown.html",
        [scanned, unscanned, missing],
        sample_limit=4,
    )
    report = output_path.read_text(encoding="utf-8")
    assert output_path.exists()
    assert "6</b><span>local training-eligible rows (scanned)" in report
    assert "1</b><span>sources not scanned" in report
    assert "&lt;flagship&gt;" in report
    assert "⚠ 1 non-PII tag types in sample (top: not_pii)" in report
    assert "<b>NOT scanned</b> (not locally staged)" in report
    assert "<b>NOT scanned</b> (local paths missing)" in report
    assert "Patient &lt;unsafe&gt; Alice" in report
