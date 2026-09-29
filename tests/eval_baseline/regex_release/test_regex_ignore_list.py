from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from meddies_pii.eval_baseline.regex_release.regex_ignore_list import (
    AUDIT_POOL_PROVENANCE_NOTE,
    IgnoreListEntry,
    count_ignored_false_positives,
    ignore_list_from_audit_proposal,
    ignore_list_from_rows,
    load_ignore_list_jsonl,
    write_ignore_list_jsonl,
    write_ignore_list_provenance,
)
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from pathlib import Path

_UID = "audit-doc-vi-0c33d83fd43447dd"


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "split": "eval",
        "uid": "audit-doc-vi-0c33d83fd43447dd",
        "start": 23,
        "end": 48,
        "text": "Bệnh Viện Đa Khoa Sài Gòn",
        "category": "company_name",
    }
    base.update(overrides)
    return base


def test_entry_requires_every_key_field() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        ignore_list_from_rows([_row(text=None)])
    with pytest.raises(ValueError, match="missing required fields"):
        ignore_list_from_rows([{k: v for k, v in _row().items() if k != "uid"}])


def test_entry_rejects_wrong_field_types() -> None:
    with pytest.raises(ValueError, match="wrong type"):
        ignore_list_from_rows([_row(start="23")])


def test_entry_rejects_a_split_outside_the_release_gate_configs() -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        ignore_list_from_rows([_row(split="train")])


def test_entry_rejects_a_category_that_is_not_a_known_pii_label() -> None:
    with pytest.raises(ValueError, match="not a known PII label"):
        ignore_list_from_rows([_row(category="not_a_real_label")])


def test_ignore_list_sha256_is_stable_regardless_of_row_order() -> None:
    forward = ignore_list_from_rows([_row(), _row(uid="other-uid")])
    backward = ignore_list_from_rows([_row(uid="other-uid"), _row()])

    assert forward.sha256 == backward.sha256
    assert len(forward.sha256) == 64


def test_ignore_list_sha256_changes_when_content_changes() -> None:
    original = ignore_list_from_rows([_row()])
    changed = ignore_list_from_rows([_row(end=49)])

    assert original.sha256 != changed.sha256


def test_fp_exclusion_works_on_exact_audited_coordinates() -> None:
    ignore_list = ignore_list_from_rows([_row()])
    gold: tuple[CharSpan, ...] = ()
    predicted = (CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name"),)

    ignored = count_ignored_false_positives(predicted, gold, ignore_list, split="eval", uid=_UID)

    assert ignored == 1


def test_duplicate_predicted_spans_exclude_at_most_one_fp_per_entry() -> None:
    """One audited coordinate excludes exactly one false positive.

    Three identical predicted spans against a single matching entry must leave
    two of them scored as ordinary false positives, not exclude all three.
    """
    ignore_list = ignore_list_from_rows([_row()])
    gold: tuple[CharSpan, ...] = ()
    span = CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name")
    predicted = (span, span, span)

    ignored = count_ignored_false_positives(predicted, gold, ignore_list, split="eval", uid=_UID)

    assert ignored == 1


def test_same_text_at_a_different_coordinate_is_not_exempted() -> None:
    """Same text, same doc, but a different span position than the audited one."""
    ignore_list = ignore_list_from_rows([_row()])
    gold: tuple[CharSpan, ...] = ()
    predicted = (CharSpan(100, 125, "Bệnh Viện Đa Khoa Sài Gòn", "company_name"),)

    ignored = count_ignored_false_positives(predicted, gold, ignore_list, split="eval", uid=_UID)

    assert ignored == 0


def test_same_coordinates_in_a_different_document_are_not_exempted() -> None:
    ignore_list = ignore_list_from_rows([_row()])
    gold: tuple[CharSpan, ...] = ()
    predicted = (CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name"),)

    ignored = count_ignored_false_positives(predicted, gold, ignore_list, split="eval", uid="a-different-uid")

    assert ignored == 0


def test_ignored_span_never_becomes_a_true_positive() -> None:
    """Keep an audited gold span eligible to become a true positive.

    The audited span is also present in gold (already labeled) — it must be scored as the true positive it is, never
    pulled out as an "ignored FP".
    """
    ignore_list = ignore_list_from_rows([_row()])
    span = CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name")
    gold = (span,)
    predicted = (span,)

    ignored = count_ignored_false_positives(predicted, gold, ignore_list, split="eval", uid=_UID)

    assert ignored == 0


def test_no_ignore_list_excludes_nothing() -> None:
    gold: tuple[CharSpan, ...] = ()
    predicted = (CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name"),)

    assert count_ignored_false_positives(predicted, gold, None, split="eval", uid="x") == 0


def test_jsonl_round_trip(tmp_path: Path) -> None:
    ignore_list = ignore_list_from_rows([_row(), _row(uid="other-uid")])
    path = write_ignore_list_jsonl(tmp_path / "ignore.jsonl", ignore_list)

    loaded = load_ignore_list_jsonl(path)

    assert loaded.sha256 == ignore_list.sha256
    assert set(loaded.entries) == set(ignore_list.entries)


def test_audit_proposal_builder_keeps_only_the_requested_tier() -> None:
    proposal_rows = [
        {
            "split": "eval",
            "uid": "uid-a",
            "start": 23,
            "end": 48,
            "text": "Bệnh Viện Đa Khoa Sài Gòn",
            "proposed_category": "company_name",
            "tier": "A",
        },
        {
            "split": "eval",
            "uid": "uid-b",
            "start": 10,
            "end": 20,
            "text": "Some Corp",
            "proposed_category": "company_name",
            "tier": "B",
        },
    ]

    ignore_list = ignore_list_from_audit_proposal(
        proposal_rows,
        document_uids={("eval", "uid-a"): "v2-eval:vi:uid-a"},
    )

    assert len(ignore_list.entries) == 1
    assert ignore_list.entries[0] == IgnoreListEntry(
        split="eval",
        uid="v2-eval:vi:uid-a",
        start=23,
        end=48,
        text="Bệnh Viện Đa Khoa Sài Gòn",
        category="company_name",
    )


def test_audit_proposal_builder_accepts_a_wider_tier_filter() -> None:
    proposal_rows = [
        {
            "split": "eval",
            "uid": "uid-b",
            "start": 10,
            "end": 20,
            "text": "Some Corp",
            "proposed_category": "company_name",
            "tier": "B",
        },
    ]

    ignore_list = ignore_list_from_audit_proposal(
        proposal_rows,
        tiers=frozenset({"A", "B"}),
        document_uids={("eval", "uid-b"): "v2-eval:vi:uid-b"},
    )

    assert len(ignore_list.entries) == 1


def test_provenance_sidecar_records_source_tiers_and_pool_caveat(
    tmp_path: Path,
) -> None:
    proposal_path = tmp_path / "proposal.jsonl"
    proposal_path.write_text('{"tier": "A"}\n', encoding="utf-8")
    ignore_list = ignore_list_from_rows([_row(), _row(uid="other-uid")])

    provenance_path = write_ignore_list_provenance(
        tmp_path / "ignore-list-tierA.provenance.json",
        ignore_list=ignore_list,
        source_proposal_path=proposal_path,
        tiers=frozenset({"A"}),
    )

    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["source_proposal_path"] == str(proposal_path)
    assert payload["source_proposal_sha256"] == hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    assert payload["tier_filter"] == ["A"]
    assert payload["entry_count"] == 2
    assert payload["ignore_list_sha256"] == ignore_list.sha256
    assert payload["pool_provenance_note"] == AUDIT_POOL_PROVENANCE_NOTE


def _proposal_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "split": "eval",
        "uid": "mimo_corpus_vi_0c33d83fd43447dd",
        "start": 23,
        "end": 48,
        "text": "Bệnh Viện Đa Khoa Sài Gòn",
        "proposed_category": "company_name",
        "tier": "A",
    }
    base.update(overrides)
    return base


_RUNTIME_UID = "v2-eval:vi:mimo_corpus_vi_0c33d83fd43447dd"
"""The scorer keys documents by EvalRow.stable_id, i.e. dataset:shard:doc_id."""
_DOCUMENT_UIDS = {("eval", "mimo_corpus_vi_0c33d83fd43447dd"): _RUNTIME_UID}


def test_proposal_entries_match_documents_keyed_the_way_the_scorer_keys_them() -> None:
    ignore_list = ignore_list_from_audit_proposal([_proposal_row()], document_uids=_DOCUMENT_UIDS)
    predicted = (CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name"),)

    assert ignore_list.entries[0].uid == _RUNTIME_UID
    assert count_ignored_false_positives(predicted, (), ignore_list, split="eval", uid=_RUNTIME_UID) == 1


def test_raw_proposal_uid_matches_no_document_the_scorer_ever_presents() -> None:
    """Regression for the silent-miss trap that produced fp_ignored=0.

    The proposal's bare uid is only the doc_id half of the scorer's key, so an
    ignore-list built from it matched nothing and every audited coordinate kept
    counting as a false positive.
    """
    stale = ignore_list_from_rows([
        {
            "split": "eval",
            "uid": "mimo_corpus_vi_0c33d83fd43447dd",
            "start": 23,
            "end": 48,
            "text": "Bệnh Viện Đa Khoa Sài Gòn",
            "category": "company_name",
        },
    ])
    predicted = (CharSpan(23, 48, "Bệnh Viện Đa Khoa Sài Gòn", "company_name"),)

    assert count_ignored_false_positives(predicted, (), stale, split="eval", uid=_RUNTIME_UID) == 0


def test_proposal_refuses_a_uid_absent_from_the_pinned_split() -> None:
    with pytest.raises(ValueError, match="outside the pinned split"):
        ignore_list_from_audit_proposal([_proposal_row(uid="not-a-real-doc")], document_uids=_DOCUMENT_UIDS)
