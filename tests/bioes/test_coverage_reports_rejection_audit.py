from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from meddies_pii.training.bioes.eval.rejection_policy import (
    RejectionVerdict,
    classify_rejection_reason,
)
from meddies_pii.training.bioes.reports import rejection_audit

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("reason", "verdict", "recoverable"),
    [
        ("short_phone_span", RejectionVerdict.OVER_FILTER, True),
        ("leftover_label_marker", RejectionVerdict.PARSER_BUG, True),
        ("too_short", RejectionVerdict.CORRECT, False),
        ("unregistered_reason", RejectionVerdict.UNCLASSIFIED, False),
    ],
)
# reason: pytest binds this parameter from the parametrize argnames tuple, so the boolean is labelled at every call site.
def test_rejection_policy_is_explicit_and_fail_closed(
    reason: str,
    verdict: RejectionVerdict,
    recoverable: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
) -> None:
    classification = classify_rejection_reason(reason)

    assert classification.verdict is verdict
    assert classification.recoverable is recoverable
    assert classification.explanation


def test_rejection_audit_loads_valid_objects_and_writes_mixed_verdict_report(
    tmp_path: Path,
) -> None:
    rejected = tmp_path / "nested" / "rejected.synthetic.jsonl"
    rejected.parent.mkdir()
    rejected.write_text(
        "\n".join([
            json.dumps({
                "errors": ["short_phone_span"],
                "parsed_label_count": 7,
                "content": "prefix [0123]<phone_number> suffix",
                "language": "vi",
                "scenario": "contact",
                "text_format": "note",
            }),
            "{malformed json",
            json.dumps(["not an object"]),
            json.dumps({
                "errors": ["leftover_label_marker", "unknown_reason: detail"],
                "parsed_label_count": "not-an-int",
                "content": "FHIR [x]<tag>",
            }),
            json.dumps({"errors": "wrong-shape", "content": "plain"}),
        ])
        + "\n",
        encoding="utf-8",
    )

    records = rejection_audit.load_rejection_records([str(tmp_path / "**" / "rejected*.jsonl")])
    output_path = rejection_audit.write_rejection_audit_report(records, tmp_path / "reports" / "audit.html")

    assert len(records) == 3
    assert output_path == tmp_path / "reports" / "audit.html"
    assert output_path.exists()
    report = output_path.read_text(encoding="utf-8")
    assert "4 rejection-reasons across 3 dropped docs" in report
    assert "Recoverable</div>" in report
    assert "50%" in report
    assert "<mark>[0123]&lt;phone_number&gt;</mark>" in report
    assert "unknown_reason" in report
    assert re.search(
        r"<code>unknown_reason</code>.*?>unclassified</span>",
        report,
    )
    assert "(none)" in report
