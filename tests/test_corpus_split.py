"""Behavior tests for the train/eval corpus-split logic used by the v2 publish.

The split is the safety-critical part of publishing: the eval holdout MUST be
text-disjoint from train (no leakage), reproducible regardless of how the union
was assembled, and exactly the requested size.

Same key check_eval_disjoint.py uses, so dedup/leakage agree codebase-wide.

The republish leak guard: the local corpus is a superset that re-introduces the eval rows, so train must subtract every
held-out text hash.

Consistent with dedup_rows: a non-string-text row can't be a training example, and indexing row["text"] on it must not
raise.

"""

from __future__ import annotations

import hashlib

import pytest

from meddies_pii.publishing.corpus_split import (
    dedup_rows,
    drop_held_out,
    holdout_split,
    text_hash,
)


def _row(text: str, *, source: str = "x") -> dict[str, object]:
    return {
        "text": text,
        "label": [],
        "info": {"language": "Vietnamese", "source": source, "split_purpose": "train"},
    }


def test_text_hash_matches_disjoint_checker_convention() -> None:
    assert text_hash("héllo") == hashlib.sha256("héllo".encode()).hexdigest()


def test_dedup_keeps_first_occurrence_and_drops_exact_duplicates() -> None:
    """First occurrence wins (HF row kept, not the later local one)."""
    rows = [_row("a", source="hf"), _row("b"), _row("a", source="local"), _row("b")]
    out = dedup_rows(rows)
    assert [r["text"] for r in out] == ["a", "b"]
    assert out[0]["info"]["source"] == "hf"


def test_dedup_drops_rows_without_string_text() -> None:
    rows = [_row("a"), {"text": None, "label": []}, {"label": []}]
    assert [r["text"] for r in dedup_rows(rows)] == ["a"]


def test_holdout_split_sizes_and_disjointness() -> None:
    rows = [_row(f"doc-{i}") for i in range(50)]
    train, held = holdout_split(rows, 10, seed=0)
    assert len(held) == 10
    assert len(train) == 40
    train_texts = {r["text"] for r in train}
    held_texts = {r["text"] for r in held}
    assert train_texts.isdisjoint(held_texts)
    assert train_texts | held_texts == {r["text"] for r in rows}


def test_holdout_flags_eval_rows_and_leaves_train_unchanged() -> None:
    rows = [_row(f"doc-{i}") for i in range(20)]
    train, held = holdout_split(rows, 5, seed=0)
    assert all(r["info"]["split_purpose"] == "eval" for r in held)
    assert all(r["info"]["split_purpose"] == "train" for r in train)


def test_holdout_is_reproducible_regardless_of_input_order() -> None:
    """Same row set + same seed -> same eval docs, even though input order differs."""
    rows = [_row(f"doc-{i}") for i in range(50)]
    shuffled = list(reversed(rows))
    _, held_a = holdout_split(rows, 10, seed=7)
    _, held_b = holdout_split(shuffled, 10, seed=7)
    assert {r["text"] for r in held_a} == {r["text"] for r in held_b}


def test_holdout_refuses_to_empty_the_training_set() -> None:
    rows = [_row(f"doc-{i}") for i in range(5)]
    # reason: `holdout_split` raises ValueError from two guards, and only this one is the refusal
    # reason: under test — without the pattern a regression into the `n must be >= 0` branch passes.
    with pytest.raises(ValueError, match="holdout n=5 >= rows=5 leaves no training data"):
        holdout_split(rows, 5, seed=0)


def test_drop_held_out_removes_rows_whose_text_is_held_out() -> None:
    rows = [_row("keep-1"), _row("eval-doc"), _row("keep-2"), _row("gold-doc")]
    held_out = {text_hash("eval-doc"), text_hash("gold-doc")}
    out = drop_held_out(rows, held_out)
    assert [r["text"] for r in out] == ["keep-1", "keep-2"]
    assert {r["text"] for r in out}.isdisjoint({"eval-doc", "gold-doc"})


def test_drop_held_out_empty_set_is_identity() -> None:
    rows = [_row("a"), _row("b")]
    assert drop_held_out(rows, set()) == rows


def test_drop_held_out_drops_rows_without_string_text() -> None:
    rows = [_row("a"), {"text": None, "label": []}, {"label": []}]
    assert [r["text"] for r in drop_held_out(rows, set())] == ["a"]
