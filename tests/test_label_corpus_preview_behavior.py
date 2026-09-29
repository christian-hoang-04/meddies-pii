from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from meddies_pii.generation.label_corpus import preview
from meddies_pii.jsonl import write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _record(
    *,
    identifier: str,
    scenario: str | None = "portal",
    include_required: bool = True,
) -> dict[str, object]:
    private_url = "https://portal.example.test/result?token=abc"
    # reason: this synthetic credential is the PII the labeller under test has to find, so it is the
    # reason: fixture's payload rather than a secret in use. Renaming it would stop describing the row.
    secret = "token-123"  # ruff: ignore[hardcoded-password-string]
    text = f"<unsafe> {identifier}: {private_url}; session {secret}."
    labels: list[dict[str, object]] = []
    if include_required:
        for category, value in (("private_url", private_url), ("secret", secret)):
            start = text.index(value)
            labels.append({"category": category, "start": start, "end": start + len(value)})
    else:
        value = identifier
        start = text.index(value)
        labels.append({"category": "human_name", "start": start, "end": start + len(value)})
    info: dict[str, object] = {"id": identifier}
    if scenario is not None:
        info["scenario"] = scenario
    return {"text": text, "label": labels, "info": info}


def test_preview_builds_escaped_deterministic_html_from_tiny_artifacts(
    tmp_path: Path,
) -> None:
    first = _record(identifier="alpha", scenario="triage")
    duplicate = first | {"info": {"id": "alpha-copy", "scenario": "triage"}}
    missing_required = _record(identifier="beta", scenario=None, include_required=False)
    write_jsonl(
        tmp_path / "accepted.vi.jsonl",
        [first, duplicate, missing_required, {"text": 7}],
    )
    with (tmp_path / "accepted.vi.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{malformed-json}\n")

    for code in ("vi", "en"):
        if code == "en":
            write_jsonl(tmp_path / f"accepted.{code}.jsonl", [_record(identifier=code)])
        repaired = [_record(identifier=f"{code}-repaired", scenario="zeta")]
        if code == "vi":
            repaired.append(_record(identifier="vi-repaired-copy", scenario="zeta"))
        write_jsonl(
            tmp_path / f"repaired_candidates.clean.{code}.jsonl",
            repaired,
        )
        write_jsonl(tmp_path / f"raw.{code}.jsonl", [{"source": code}])
        write_jsonl(tmp_path / f"rejected.{code}.jsonl", [{"source": code}])
        write_jsonl(tmp_path / f"api_errors.{code}.jsonl", [{"source": code}])
        write_jsonl(tmp_path / f"content_rejected.{code}.jsonl", [{"source": code}])
        write_jsonl(tmp_path / f"repaired_candidates.flagged.{code}.jsonl", [{"source": code}])
    write_jsonl(
        tmp_path / "unrepaired_content_rejected.vi.jsonl",
        [
            {
                "errors": ["repair <reason> &"],
                "content": "<script>alert(1)</script>" + "x" * 901,
                "repaired_content": "<b>still unsafe</b>",
            },
            {"errors": ["repair <reason> &"], "content": "second example"},
            {"errors": "not-a-list"},
        ],
    )
    write_jsonl(
        tmp_path / "unrepaired_content_rejected.en.jsonl",
        [{"errors": "not-a-list"}],
    )
    (tmp_path / "repair_summary.vi.json").write_text('{"kept": 1}', encoding="utf-8")

    output = preview.build_preview(
        tmp_path,
        tmp_path / "nested" / "preview.html",
        per_label=1,
        per_scenario=1,
    )
    html = output.read_text(encoding="utf-8")

    assert output.exists()
    assert "repaired_candidates.clean" in html
    assert html.index("<h2>VI</h2>") < html.index("<h2>EN</h2>")
    assert "<b>accepted</b> 4" in html
    assert "accepted bad probe</b> 1" in html
    assert "accepted duplicates</b> 1" in html
    vi_labels = html.split("VI accepted label distribution", 1)[1].split("</section>", 1)[0]
    assert vi_labels.index("human_name") < vi_labels.index("private_url")
    assert vi_labels.index("private_url") < vi_labels.index("secret")
    assert "vi-repaired" in html
    assert "vi-repaired-copy" not in html
    assert "&lt;unsafe&gt;" in html
    assert "<unsafe>" not in html
    assert "repair &lt;reason&gt; &amp;" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_preview_main_writes_empty_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report" / "preview.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preview.py",
            "--output-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--per-label",
            "3",
            "--per-scenario",
            "4",
            "--repaired-prefix",
            "explicit-prefix",
        ],
    )

    preview.main()

    assert capsys.readouterr().out.strip() == str(output)
    html = output.read_text(encoding="utf-8")
    assert "explicit-prefix" in html
    assert "No examples." in html
