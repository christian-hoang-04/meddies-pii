from __future__ import annotations

import json

import pytest

from meddies_pii.annotations.bioes import (
    build_bioes_label_space,
    build_label_to_id,
)
from meddies_pii.training.bioes.eval.selection import (
    at_dot_obfuscate_email_rows,
    select_prepared_rows_for_adversarial_slice,
)


class _CharacterTokenizer:
    def __call__(
        self,
        text: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        offsets = [(0, 0), *((index, index + 1) for index in range(len(text))), (0, 0)]
        return {
            "input_ids": list(range(len(offsets))),
            "attention_mask": [1] * len(offsets),
            "offset_mapping": offsets,
            "num_truncated_tokens": 0,
        }


def test_email_obfuscation_derives_parseable_rows_with_stable_source_ids() -> None:
    rows: list[dict[str, object]] = [
        {"text": 42},
        {"text": "No tagged email here."},
        {"text": ("[Ada Lovelace]<human_name> uses [user@localhost]<email_address>.")},
        {"text": "[not-an-email]<email_address>"},
        {
            "uid": "uid-row",
            "text": ("[Ada Lovelace]<human_name> uses [ada@example.org]<email_address>."),
        },
        {"source": "source-row", "text": "[bob@example.net]<email_address>"},
        {"_dataset_index": 17, "text": "[carol@example.com]<email_address>"},
        {"text": "[dan@example.edu]<email_address>"},
    ]

    derived = at_dot_obfuscate_email_rows(rows)

    assert [row["uid"] for row in derived] == [
        "uid-row:at-dot-obfuscation",
        "source-row:at-dot-obfuscation",
        "dataset-index-17:at-dot-obfuscation",
        "row-7:at-dot-obfuscation",
    ]
    assert derived[0]["text"] == ("[Ada Lovelace]<human_name> uses [ada (at) example (dot) org]<email_address>.")
    assert json.loads(str(derived[0]["label"])) == {
        "email_address": ["ada (at) example (dot) org"],
        "human_name": ["Ada Lovelace"],
    }
    assert derived[0]["adversarial_derivation"] == {
        "kind": "email_at_dot_obfuscation",
        "source_uid": "uid-row",
    }


def test_email_obfuscation_limit_stops_after_requested_rows() -> None:
    derived = at_dot_obfuscate_email_rows(
        [
            {"uid": "first", "text": "[first@example.org]<email_address>"},
            {"uid": "second", "text": "[second@example.org]<email_address>"},
        ],
        limit=1,
    )

    assert [row["uid"] for row in derived] == ["first:at-dot-obfuscation"]


def test_targeted_selection_rejects_unknown_slice() -> None:
    with pytest.raises(ValueError, match="slice_name must be one of"):
        select_prepared_rows_for_adversarial_slice(
            [],
            _CharacterTokenizer(),
            slice_name="unknown",
            max_length=64,
            id_to_label={0: "O"},
        )


def test_targeted_selection_reports_source_and_prepared_support_before_limit() -> None:
    rows = at_dot_obfuscate_email_rows([
        {"uid": "first", "text": "[first@example.org]<email_address>"},
        {"uid": "second", "text": "[second@example.org]<email_address>"},
    ])
    label_vocab = build_bioes_label_space(("email_address",))
    id_to_label = {identifier: label for label, identifier in build_label_to_id(label_vocab).items()}

    selection = select_prepared_rows_for_adversarial_slice(
        rows,
        _CharacterTokenizer(),
        slice_name="at_dot_obfuscation",
        max_length=128,
        id_to_label=id_to_label,
        limit=1,
        entity_labels=("email_address",),
    )

    assert [row.uid for row in selection.prepared_rows] == ["first:at-dot-obfuscation"]
    assert selection.source_support_docs == 2
    assert selection.prepared_support_docs == 2
    assert selection.source_stats.accepted == 2
    assert selection.preparation_stats.accepted == 2
    assert selection.to_report()["prepared_row_uids"] == ["first:at-dot-obfuscation"]
    assert selection.slice_filter_report["at_dot_obfuscation"]["prepared_docs"] == 2
