from __future__ import annotations

import json
from pathlib import Path

from meddies_pii.training.bioes.reports.span_filter_review import (
    collect_span_filter_review,
    render_span_filter_review_html,
    source_label_for_path,
    span_context_html,
    write_span_filter_review_report,
)


def _record(spans: list[tuple[str, str]]) -> dict[str, object]:
    text = " | ".join(value for _, value in spans)
    offset = 0
    labels = []
    for label, value in spans:
        labels.append({
            "category": label,
            "text": value,
            "start": offset,
            "end": offset + len(value),
        })
        offset += len(value) + 3
    return {"text": text, "label": labels}


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_collect_span_filter_review_aggregates_rules_origins_and_safe_html(
    tmp_path: Path,
) -> None:
    base = tmp_path / "data/bioes-v2/base/base.jsonl"
    grpo = tmp_path / "data/bioes-v2/grpo/grpo.jsonl"
    synthetic = tmp_path / "data/bioes-v2/synthetic/daily/accepted.jsonl"
    _write_jsonl(
        base,
        [
            _record([
                ("id_number", "no digits"),
                ("company_name", "Acme <script>alert(1)</script>"),
                ("custom_label", "custom entity with enough text"),
            ]),
            "",
            "{bad json",
            [],
            {"text": 42, "label": []},
            {"text": "ignored", "label": {"not": "a list"}},
        ],
    )
    _write_jsonl(
        grpo,
        [
            _record([
                ("phone_number", "+" + "1" * 31),
                ("email_address", "a" * 41),
                ("address", "123 long but ordinary clinical address"),
            ]),
        ],
    )
    _write_jsonl(
        synthetic,
        [
            _record([
                ("human_name", "A" * 61 + ". B."),
                ("private_url", '{"resourceType":"Patient"}'),
                (
                    "secret",
                    'prefix "system": "x" https://hl7.org/terminology',
                ),
            ]),
        ],
    )

    review = collect_span_filter_review([str(base), str(grpo), str(synthetic)])

    assert review.total == 9
    assert review.total_removed == 6
    assert review.counts["company_name"].total == 1
    assert review.counts["company_name"].removed == 0
    assert review.reason_counts == {
        "V1_struct_prefix": 1,
        "V2_struct_body": 1,
        "V3_phone_too_long": 1,
        "V5_email_no_at": 1,
        "V7_name_paragraph": 1,
        "V7_no_digit": 1,
    }
    assert {example.source for items in review.removed.values() for example in items} == {
        "base",
        "grpo",
        "synthetic",
    }
    assert {example.source for items in review.kept.values() for example in items} == {
        "base",
        "grpo",
    }

    report = render_span_filter_review_html(review)
    assert report == render_span_filter_review_html(review)
    assert "6</b> of <b>9</b> (66.67%) removed" in report
    assert "custom_label" in report
    assert report.index("id_number") < report.index("custom_label")
    assert "<script>alert(1)</script>" not in report
    assert "Acme &lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "nothing removed for this label in the sample" in report
    assert "no long kept spans sampled for this label" in report


def test_span_filter_public_helpers_handle_empty_report_and_context_boundaries(
    tmp_path: Path,
) -> None:
    unreadable_as_jsonl = tmp_path / "unreadable.jsonl"
    unreadable_as_jsonl.mkdir()
    review = collect_span_filter_review([str(tmp_path / "missing*.jsonl"), str(unreadable_as_jsonl)])

    assert review.total == 0
    assert review.total_removed == 0
    assert "0</b> of <b>0</b> (0.00%) removed" in render_span_filter_review_html(review)
    output_path = write_span_filter_review_report(review, tmp_path / "nested/report.html")
    assert output_path.read_text(encoding="utf-8") == render_span_filter_review_html(review)

    assert source_label_for_path(Path("single.jsonl")) == "single.jsonl"
    assert source_label_for_path(Path("bioes-v2")) == "bioes-v2"
    assert (
        span_context_html("prefix <value> suffix", 7, 14, "fallback", context_chars=3)
        == "…ix <mark>&lt;value&gt;</mark> su…"
    )
    assert span_context_html("text", 4, 4, "<fallback>") == "text<mark>&lt;fallback&gt;</mark>"


def test_collect_span_filter_review_uses_spans_fallback_and_base_cap(
    tmp_path: Path,
) -> None:
    base = tmp_path / "data/bioes-v2/base/capped.jsonl"
    _write_jsonl(
        base,
        [
            {
                "text": "Name " + "A" * 30,
                "label": {"wrong": "shape"},
                "spans": [
                    {
                        "label": "human_name",
                        "text": "A" * 30,
                        "start": 5,
                        "end": 35,
                    },
                    {"label": "human_name", "text": "", "start": 0, "end": 0},
                    "not a span",
                ],
            },
            _record([("phone_number", "no digits")]),
        ],
    )

    review = collect_span_filter_review([str(base)], max_base=1)

    assert review.max_base == 1
    assert review.total == 1
    assert review.total_removed == 0
    assert review.kept["human_name"][0].source == "base"


def test_span_filter_report_caps_repeated_reason_examples(tmp_path: Path) -> None:
    source = tmp_path / "data/bioes-v2/grpo/repeated.jsonl"
    _write_jsonl(source, [_record([("id_number", "no digits")]) for _ in range(5)])

    review = collect_span_filter_review([str(source)])
    report = render_span_filter_review_html(review)

    assert review.reason_counts["V7_no_digit"] == 5
    assert report.count("<span class=code>V7_no_digit</span>") == 4
