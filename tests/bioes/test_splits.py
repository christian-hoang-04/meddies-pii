from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
from collections import Counter

from meddies_pii.training.bioes.data.splits import (
    carve_heldout,
    normalize_text,
    row_language_bucket,
)


def _row(*, row_id: str, language: str, text: str, source: str = "") -> dict[str, object]:
    return {
        "text": text,
        "label": [],
        "info": {"id": row_id, "language": language, "source": source},
    }


def _row_id(row: dict[str, object]) -> str:
    info = row.get("info")
    assert isinstance(info, dict)
    row_id = info.get("id")
    assert isinstance(row_id, str)
    return row_id


def _skewed_corpus() -> list[dict[str, object]]:
    """Heavily English-skewed pool so balancing has to actively rebalance toward 30/30/40."""
    other_langs = ["zh", "ja", "ko", "th", "fr"]
    rows: list[dict[str, object]] = [_row(row_id=f"vi-{i}", language="vi", text=f"vietnamese row {i}") for i in range(40)]
    rows.extend(_row(row_id=f"en-{i}", language="en", text=f"english row {i}") for i in range(120))
    for i in range(60):
        lang = other_langs[i % len(other_langs)]
        rows.append(_row(row_id=f"other-{i}", language=lang, text=f"other {lang} row {i}"))
    return rows


def test_carved_heldout_hits_30_30_40_within_tolerance() -> None:
    rows = _skewed_corpus()
    heldout_n = 100
    tolerance = 0.05

    _train, heldout = carve_heldout(rows, heldout_rows=heldout_n, seed=7, tolerance=tolerance)

    assert len(heldout) == heldout_n
    counts = Counter(row_language_bucket(row) for row in heldout)
    fractions = {bucket: counts.get(bucket, 0) / heldout_n for bucket in ("vi", "en", "other")}
    assert abs(fractions["vi"] - 0.30) <= tolerance, fractions
    assert abs(fractions["en"] - 0.30) <= tolerance, fractions
    assert abs(fractions["other"] - 0.40) <= tolerance, fractions


def test_carved_heldout_disjoint_from_train_by_id_and_normalized_text() -> None:
    """Deliberate duplicate.

    Same normalized text appears under two different ids, one destined for held-out, one in the remaining train pool.

    """
    rows = _skewed_corpus()
    dup_text = "Patient Tran Van A,  Hanoi"
    rows.append(_row(row_id="dup-heldout", language="vi", text=dup_text))
    rows.append(_row(row_id="dup-train", language="vi", text="  patient tran van a, hanoi  "))

    train, heldout = carve_heldout(rows, heldout_rows=100, seed=7, tolerance=0.05)

    train_ids = {_row_id(row) for row in train}
    heldout_ids = {_row_id(row) for row in heldout}
    assert train_ids & heldout_ids == set()

    train_hashes = {normalize_text(str(row["text"])) for row in train}
    heldout_hashes = {normalize_text(str(row["text"])) for row in heldout}
    assert train_hashes & heldout_hashes == set()


def test_normalize_text_strips_lowercases_and_collapses_whitespace() -> None:
    assert normalize_text("  Hello   World  ") == normalize_text("hello world")
    assert normalize_text("A\t B\nC") == "a b c"
