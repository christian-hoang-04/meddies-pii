"""Quality+diversity assembly gate (ADR 0008 §2 + §3).

Each test is one vertical behavior slice of ``evaluate_corpus``:
floors block, zero/sparse cells block (unless waived), entropy targets gate,
and a full small corpus that meets every condition passes.

(a) a starved label below floor -> passed False + label_floor_failures.

(b) a zero cell -> not passed + listed in zero_cells.

(c) the SAME zero cell in waivers -> passed True + listed in waived_cells.

(c') a sparse (0 < docs < cell_min) cell blocks; waiving it passes.

(e) a full small corpus meeting all -> passed True.

Regression (codex HIGH): is_supported_meddies_language admits aliases like "us"/"jpn" but normalize_language rejects them;
without the alias map the row (e.g. Nemotron locale="us") was silently dropped, hiding company_name supply. "us"->en,
"jpn"->ja: both spans must count toward the floor.

count-per-source + merge + evaluate_counts must equal evaluate_corpus over the union — the property the Modal
parallel/incremental gate relies on.

"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import math
from typing import TYPE_CHECKING, cast

from meddies_pii.languages import LANGUAGE_PROFILES
from meddies_pii.taxonomy import PII_LABELS
from meddies_pii.training.bioes.data.corpus_gate import GateReport, evaluate_corpus

if TYPE_CHECKING:
    from collections.abc import Mapping

LANGUAGE_CODES: tuple[str, ...] = tuple(p.code for p in LANGUAGE_PROFILES.values())


def _row(language: str, labels: tuple[str, ...]) -> dict[str, object]:
    return {
        "text": f"doc-{language}-{'-'.join(labels)}",
        "label": [{"category": label, "start": 0, "end": 1, "text": "x"} for label in labels],
        "info": {"language": language},
    }


def _full_grid_rows(*, per_cell: int = 1) -> list[dict[str, object]]:
    """One single-label doc per (language, label) cell, repeated per_cell times."""
    rows: list[dict[str, object]] = []
    for code in LANGUAGE_CODES:
        for label in PII_LABELS:
            rows.extend(_row(code, (label,)) for _ in range(per_cell))
    return rows


def _slack_floors() -> dict[str, int]:
    return dict.fromkeys(PII_LABELS, 0)


def _slack_entropy() -> dict[str, float]:
    return {"language": 0.0, "label": 0.0}


def test_returns_gate_report() -> None:
    report = evaluate_corpus(
        _full_grid_rows(),
        label_floors=_slack_floors(),
        cell_min=1,
        entropy_targets=_slack_entropy(),
    )
    assert isinstance(report, GateReport)


def test_starved_label_below_floor_blocks() -> None:
    rows = _full_grid_rows()
    floors = _slack_floors()
    floors["secret"] = 100

    report = evaluate_corpus(
        rows,
        label_floors=floors,
        cell_min=1,
        entropy_targets=_slack_entropy(),
    )

    assert report.passed is False
    assert report.label_floor_failures["secret"] == 100 - len(LANGUAGE_CODES)
    assert "secret" not in {label for label in report.label_floor_failures if report.label_floor_failures[label] <= 0}
    assert any("secret" in reason for reason in report.blocking_reasons)


def test_zero_cell_blocks_and_is_listed() -> None:
    """Drop every (vi, secret) doc -> that cell becomes zero."""
    rows = _full_grid_rows()
    rows = [r for r in rows if not _is_cell(r, "vi", "secret")]

    report = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=1,
        entropy_targets=_slack_entropy(),
    )

    assert report.passed is False
    assert ("vi", "secret") in report.zero_cells
    assert ("vi", "secret") not in report.sparse_cells


def test_waived_zero_cell_passes() -> None:
    rows = _full_grid_rows()
    rows = [r for r in rows if not _is_cell(r, "vi", "secret")]

    report = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=1,
        entropy_targets=_slack_entropy(),
        waivers={("vi", "secret")},
    )

    assert report.passed is True
    assert ("vi", "secret") in report.waived_cells
    assert ("vi", "secret") not in report.zero_cells
    assert ("vi", "secret") not in report.sparse_cells


def test_sparse_cell_blocks_then_waives() -> None:
    """Make (en, date) sparse: keep only 1 doc -> below cell_min of 3."""
    rows = _full_grid_rows(per_cell=3)
    en_date = [r for r in rows if _is_cell(r, "en", "date")]
    rows = [r for r in rows if not _is_cell(r, "en", "date")] + en_date[:1]

    blocked = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=3,
        entropy_targets=_slack_entropy(),
    )
    assert blocked.passed is False
    assert ("en", "date") in blocked.sparse_cells
    assert ("en", "date") not in blocked.zero_cells

    waived = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=3,
        entropy_targets=_slack_entropy(),
        waivers={("en", "date")},
    )
    assert waived.passed is True
    assert ("en", "date") in waived.waived_cells
    assert ("en", "date") not in waived.sparse_cells


def test_entropy_below_target_blocks() -> None:
    """Demand more language entropy than log2(17) can ever provide."""
    rows = _full_grid_rows()
    max_lang_entropy = math.log2(len(LANGUAGE_CODES))

    report = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=1,
        entropy_targets={"language": max_lang_entropy + 1.0, "label": 0.0},
    )

    assert report.passed is False
    assert "language" in report.entropy_failures
    assert report.entropy_failures["language"] == report.entropy["language"]
    assert "label" not in report.entropy_failures
    assert any("entropy" in reason for reason in report.blocking_reasons)


def test_entropy_is_shannon_log2_and_uniform_is_maximal() -> None:
    rows = _full_grid_rows()
    report = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=1,
        entropy_targets=_slack_entropy(),
    )
    assert math.isclose(report.entropy["language"], math.log2(len(LANGUAGE_CODES)), abs_tol=1e-9)
    assert math.isclose(report.entropy["label"], math.log2(len(PII_LABELS)), abs_tol=1e-9)


def test_full_corpus_meeting_all_passes() -> None:
    rows = _full_grid_rows(per_cell=2)
    floors: dict[str, int] = dict.fromkeys(PII_LABELS, 30)

    report = evaluate_corpus(
        rows,
        label_floors=floors,
        cell_min=2,
        entropy_targets={"language": 4.0, "label": 3.0},
    )

    assert report.passed is True
    assert report.label_floor_failures == {}
    assert report.zero_cells == []
    assert report.sparse_cells == []
    assert report.entropy_failures == {}
    assert report.blocking_reasons == []


def test_unsupported_language_rows_are_excluded_from_grid() -> None:
    """The bogus language must not create phantom cells nor break the gate."""
    rows = _full_grid_rows()
    rows.append(_row("klingon", ("human_name",)))

    report = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=1,
        entropy_targets=_slack_entropy(),
    )

    assert report.passed is True
    assert all(lang in LANGUAGE_CODES for lang, _ in report.zero_cells)
    assert all(lang in LANGUAGE_CODES for lang, _ in report.sparse_cells)


def test_language_alias_is_normalized_to_canonical_code() -> None:
    """Replace the canonical "vi" docs with the alias "vietnamese".

    "vietnamese" collapses onto "vi" -> no zero cell for vi.

    """
    rows = _full_grid_rows()
    rows = [r for r in rows if not _is_language(r, "vi")]
    for label in PII_LABELS:
        rows.append(_row("vietnamese", (label,)))

    report = evaluate_corpus(
        rows,
        label_floors=_slack_floors(),
        cell_min=1,
        entropy_targets=_slack_entropy(),
    )

    assert ("vi", "human_name") not in report.zero_cells
    assert report.passed is True


def test_spans_key_is_accepted_as_well_as_label_key() -> None:
    """One extra doc using the "spans"/"label" span schema instead of "label"/"category".

    The alt-schema span counted, so human_name now clears the +1 floor.

    """
    rows = _full_grid_rows()
    rows.append({
        "text": "alt-schema-doc",
        "spans": [{"label": "human_name", "start": 0, "end": 1, "text": "x"}],
        "info": {"language": "en"},
    })

    report = evaluate_corpus(
        rows,
        label_floors={**_slack_floors(), "human_name": len(LANGUAGE_CODES) + 1},
        cell_min=1,
        entropy_targets=_slack_entropy(),
    )

    assert "human_name" not in report.label_floor_failures
    assert report.passed is True


def _is_cell(row: Mapping[str, object], language: str, label: str) -> bool:
    return _is_language(row, language) and _has_label(row, label)


def _is_language(row: Mapping[str, object], language: str) -> bool:
    info = cast("Mapping[str, object]", row["info"])
    return info.get("language") == language


def _has_label(row: Mapping[str, object], label: str) -> bool:
    spans = cast("list[Mapping[str, object]]", row.get("label", []))
    return any(span.get("category") == label for span in spans)


def test_locale_aliases_resolve_and_count_toward_floors() -> None:
    rows = [_row("us", ("company_name",)), _row("jpn", ("company_name",))]
    floors = _slack_floors()
    floors["company_name"] = 2
    report = evaluate_corpus(rows, label_floors=floors, cell_min=1, entropy_targets=_slack_entropy())
    assert "company_name" not in report.label_floor_failures


def test_incremental_counts_equal_monolithic() -> None:
    """Counts survive a JSON cache round-trip (per-source caching on the volume)."""
    from meddies_pii.training.bioes.data.corpus_gate import (
        CorpusCounts,
        count_corpus,
        evaluate_counts,
        merge_counts,
    )

    rows_a = _full_grid_rows(per_cell=2)
    rows_b = [_row("vi", ("company_name",)), _row("en", ("secret",))]
    floors = _slack_floors()
    targets = _slack_entropy()

    mono = evaluate_corpus([*rows_a, *rows_b], label_floors=floors, cell_min=1, entropy_targets=targets)
    merged = merge_counts([count_corpus(rows_a), count_corpus(rows_b)])
    inc = evaluate_counts(merged, label_floors=floors, cell_min=1, entropy_targets=targets)

    assert inc.passed == mono.passed
    assert sorted(map(tuple, inc.zero_cells)) == sorted(map(tuple, mono.zero_cells))
    assert inc.label_floor_failures == mono.label_floor_failures
    assert inc.entropy == mono.entropy
    restored = CorpusCounts.from_json(merged.to_json())
    assert restored.cell_doc_counts == merged.cell_doc_counts
