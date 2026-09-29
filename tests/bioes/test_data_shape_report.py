from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

import pytest

from meddies_pii.training.bioes.reports.data_shape import (
    render_data_shape_report_html,
    write_data_shape_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _bar_value(report: str, label: str) -> str:
    match = re.search(
        rf"<div class=key>{re.escape(label)}</div>.*?<div class=val>([^<]+)</div>",
        report,
    )
    assert match is not None
    return cast("str", match.group(1))


def _rich_stats() -> dict[str, object]:
    return {
        "per_format": {
            "plain clinical note": 5,
            "PLAIN_TEXT": 2,
            "unknown <script>": 1,
            "ignored": "8",
        },
        "per_language": {"vi": 6, "en": 4, "xx<script>": 2, "ignored": "5"},
        "per_label": {
            "human_name": 90,
            "id_number": 80,
            "date": 70,
            "address": 60,
            "phone_number": 50,
            "email_address": 40,
            "company_name": 30,
            "secret": 20,
            "private_url": 10,
            "ignored": "1",
        },
        "lang_label_docs": {
            "vi|human_name": 1_500_000,
            "en|human_name": 1_200,
            "xx<script>|company_name": 2,
            "vi|private_url": "not an int",
        },
        "per_source": {
            "meddies-pii-hf-config": 5,
            "train": 3,
            "validation": 2,
            "nvidia/clinical": 4,
            "grpo-hf": 1,
            "opencode_zen": 2,
            "openrouter": 1,
            "legacy <script>": 6,
        },
        "per_edge_case": {"long_name": 3, "<script>": 1},
        "span_density": {"0": 1, "4-6": 5, "21+": 2},
        "text_len": {"0-200": 2, "1k-2k": 5, "4k+": 1},
        "total_rows": 10,
        "total_spans": 450,
        "format_coverage": 8,
    }


def test_render_data_shape_report_aggregates_stats_and_escapes_persisted_values() -> None:
    baseline = {
        "per_label": {
            "human_name": {"f1": 0.4},
            "id_number": {"f1": 0.7},
            "date": {"f1": 0.9},
            "address": {"f1": "not numeric"},
        },
    }

    report = render_data_shape_report_html(_rich_stats(), baseline, sha="sha<script>")

    assert report == render_data_shape_report_html(_rich_stats(), baseline, sha="sha<script>")
    assert "10" in report
    assert "450" in report
    assert "3/27 cells filled" in report
    assert "24 waived cell(s)" in report
    assert "80% of rows tagged" in report
    assert "PLAIN_TEXT" in report
    assert "7</div>" in report
    assert "Vietnamese" in report
    assert "English" in report
    assert "per-language configs" in report
    assert "ai4privacy" in report
    assert "nvidia/Nemotron" in report
    assert "base-173k (augmented)" in report
    assert "GRPO" in report
    assert "synthetic (free pools)" in report
    assert _bar_value(report, "per-language configs") == "5"
    assert _bar_value(report, "ai4privacy") == "5"
    assert _bar_value(report, "nvidia/Nemotron") == "4"
    assert _bar_value(report, "base-173k (augmented)") == "6"
    assert _bar_value(report, "GRPO") == "1"
    assert _bar_value(report, "synthetic (free pools)") == "3"
    assert "Run-1 F1 0.40" in report
    assert "Run-1 F1 0.70" in report
    assert "Run-1 F1 0.90" in report
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "1.5M" in report
    assert "1k" in report


def test_data_shape_report_handles_empty_stats_and_file_backed_inputs(
    tmp_path: Path,
) -> None:
    empty_report = render_data_shape_report_html({})
    assert "Gate passes with zero waivers" in empty_report
    assert "0/0" in empty_report
    assert "no data" in empty_report

    stats_path = tmp_path / "stats.json"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "reports/data-shape.html"
    stats_path.write_text(json.dumps(_rich_stats()), encoding="utf-8")
    baseline_path.write_text(json.dumps({"per_label": {}}), encoding="utf-8")

    written = write_data_shape_report(
        stats_path=stats_path,
        baseline_path=baseline_path,
        output_path=output_path,
        sha="fixture-sha",
    )

    assert written == output_path
    assert "fixture-sha" in output_path.read_text(encoding="utf-8")

    without_baseline = write_data_shape_report(
        stats_path=stats_path,
        baseline_path=None,
        output_path=tmp_path / "reports/no-baseline.html",
    )
    assert "class=chip" not in without_baseline.read_text(encoding="utf-8")


def test_data_shape_report_rejects_non_object_stats_json(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        write_data_shape_report(
            stats_path=stats_path,
            baseline_path=None,
            output_path=tmp_path / "report.html",
        )
