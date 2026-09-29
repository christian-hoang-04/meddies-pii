from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import json
from typing import TYPE_CHECKING, cast

import pytest

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    build_label_to_id,
)
from meddies_pii.training.bioes.data.source_payloads import (
    parse_serialized_collection,
)
from meddies_pii.training.bioes.data.tokenizer_security import (
    validate_remote_code_tokenizer_policy,
)
from meddies_pii.training.bioes.eval.selection import (
    select_prepared_rows_for_adversarial_slice,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger


class _BehaviorTokenizer:
    def __call__(
        self,
        text: str,
        **_: object,
    ) -> dict[str, object]:
        entity_end = text.index("\n")
        if "alignment" in text:
            offsets = [(entity_end + 1, len(text))]
        elif "round-trip" in text:
            offsets = [(0, entity_end + 2)]
        else:
            offsets = [(0, entity_end)]
        return {
            "input_ids": [1],
            "attention_mask": [1],
            "offset_mapping": offsets,
            "num_truncated_tokens": int("truncated" in text),
        }


def _label_row(
    uid: str,
    name: str,
    suffix: str,
    *,
    label: object | None = None,
) -> dict[str, object]:
    text = f"{name}\n{suffix}"
    return {
        "uid": uid,
        "text": text,
        "label": (
            [
                {
                    "category": "human_name",
                    "start": 0,
                    "end": len(name),
                    "text": name,
                },
            ]
            if label is None
            else label
        ),
    }


def _id_to_label(entity_labels: Sequence[str] = ENTITY_LABELS) -> dict[int, str]:
    label_to_id = build_label_to_id(build_bioes_label_space(entity_labels))
    return {identifier: label for label, identifier in label_to_id.items()}


def test_adversarial_selection_reports_each_tokenization_rejection() -> None:
    selection = select_prepared_rows_for_adversarial_slice(
        [
            _label_row("kept", "Alice", "kept"),
            _label_row("alignment", "Bob", "alignment"),
            _label_row("truncated", "Cara", "truncated"),
            _label_row("round-trip", "Dina", "round-trip"),
        ],
        _BehaviorTokenizer(),
        slice_name="line_breaks",
        max_length=32,
        id_to_label=_id_to_label(),
    )

    assert [row.uid for row in selection.prepared_rows] == ["kept"]
    assert selection.preparation_stats.accepted == 1
    assert selection.preparation_stats.skipped_alignment == 1
    assert selection.preparation_stats.skipped_truncated == 1
    assert selection.preparation_stats.skipped_round_trip == 1


@pytest.mark.parametrize(
    ("row", "counter"),
    [
        ({}, "skipped_empty_text"),
        ({"text": "No private data\nhere"}, "skipped_no_spans"),
        ({"raw": "No private data\nhere"}, "skipped_no_spans"),
        (
            {
                "text": "[Alice]<human_name>\nvisited",
            },
            "skipped_missing_label_json",
        ),
        (
            {
                "text": "[Alice]<human_name>\nvisited",
                "label": "{}",
            },
            "skipped_empty_label_json",
        ),
        (
            {
                "text": "[Alice]<human_name>\nvisited",
                "label": json.dumps({"human_name": ["Bob"]}),
            },
            "skipped_label_mismatch",
        ),
        (
            {
                "text": "[Alice]<name>\nvisited",
                "label": json.dumps({"human_name": ["Alice"]}),
            },
            "skipped_label_repairs",
        ),
    ],
)
def test_adversarial_selection_rejects_invalid_source_rows(
    row: dict[str, object],
    counter: str,
) -> None:
    selection = select_prepared_rows_for_adversarial_slice(
        [row],
        _BehaviorTokenizer(),
        slice_name="line_breaks",
        max_length=32,
        id_to_label=_id_to_label(),
    )

    assert selection.prepared_rows == ()
    assert getattr(selection.source_stats, counter) == 1


def test_adversarial_selection_accepts_repaired_and_optional_label_json() -> None:
    repaired = select_prepared_rows_for_adversarial_slice(
        [
            {
                "source": "repaired-source",
                "raw": "[Alice]<name>\nvisited",
                "label": json.dumps({"human_name": ["Alice"]}),
            },
        ],
        _BehaviorTokenizer(),
        slice_name="line_breaks",
        max_length=32,
        id_to_label=_id_to_label(),
        allow_label_repairs=True,
    )
    optional_label = select_prepared_rows_for_adversarial_slice(
        [
            {"text": "[Bob]<human_name>\nvisited"},
            {"text": "[Eve]<human_name>\nvisited", "label": "{}"},
        ],
        _BehaviorTokenizer(),
        slice_name="line_breaks",
        max_length=32,
        id_to_label=_id_to_label(),
        require_label_json=False,
    )

    assert [row.uid for row in repaired.prepared_rows] == ["repaired-source"]
    assert [row.uid for row in optional_label.prepared_rows] == ["row-0", "row-1"]
    assert optional_label.prepared_rows[0].provided_label_json == {}
    assert optional_label.prepared_rows[1].provided_label_json == {}


def test_smoke_training_applies_the_prepared_row_limit_before_eval_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from meddies_pii.training.bioes.data.artifacts import ProbeArtifacts
    from meddies_pii.training.bioes.trainers import trainer
    from meddies_pii.training.bioes.trainers.config import SmokeTrainingConfig

    label_vocab = build_bioes_label_space(ENTITY_LABELS)
    label_to_id = build_label_to_id(label_vocab)
    artifacts = ProbeArtifacts(
        # reason: the double exposes only the members this call path uses; test-side narrowing is the
        # reason: owner-ruled shape for a stand-in that cannot be a real HiddenStateTokenTagger.
        tagger=cast("HiddenStateTokenTagger", torch.nn.Linear(1, 1)),
        # reason: the double exposes only the members this call path uses; test-side narrowing is the
        # reason: owner-ruled shape for a stand-in that cannot be a real PreTrainedTokenizerBase.
        tokenizer=cast("PreTrainedTokenizerBase", _BehaviorTokenizer()),
        label_vocab=label_vocab,
        label_to_id=label_to_id,
    )
    source_calls = 0

    def load_rows(*_: object, **__: object) -> list[dict[str, object]]:
        nonlocal source_calls
        source_calls += 1
        return [_label_row("train", "Alice", "kept")] if source_calls == 1 else [{}]

    monkeypatch.setattr(trainer, "_build_artifacts", lambda _config: artifacts)
    monkeypatch.setattr(trainer, "_load_rows", load_rows)

    with pytest.raises(RuntimeError, match="Prepared 0 eval rows"):
        trainer.run_smoke_training(
            SmokeTrainingConfig(
                train_limit=1,
                eval_limit=1,
                max_length=32,
                steps=1,
                batch_size=1,
            ),
        )


def test_adversarial_selection_rejects_non_json_label_value() -> None:
    with pytest.raises(TypeError, match="record label must be a JSON string"):
        select_prepared_rows_for_adversarial_slice(
            [{"text": "[Alice]<human_name>\nvisited", "label": 42}],
            _BehaviorTokenizer(),
            slice_name="line_breaks",
            max_length=32,
            id_to_label=_id_to_label(),
        )


@pytest.mark.parametrize(
    "label",
    [
        "not-a-list",
        {},
        [None],
        [{"category": 7, "start": 0, "end": 5}],
        [{"category": "human_name", "start": True, "end": 5}],
        [{"category": "human_name", "start": 0, "end": True}],
        [{"category": "human_name", "start": -1, "end": 5}],
        [{"category": "human_name", "start": 0, "end": 99}],
        [{"category": "human_name", "start": 0, "end": 5, "text": "Other"}],
    ],
)
def test_adversarial_selection_ignores_malformed_label_list_entries(
    label: object,
) -> None:
    selection = select_prepared_rows_for_adversarial_slice(
        [_label_row("invalid-label", "Alice", "visited", label=label)],
        _BehaviorTokenizer(),
        slice_name="line_breaks",
        max_length=32,
        id_to_label=_id_to_label(),
        require_label_json=False,
    )

    assert selection.prepared_rows == ()
    assert selection.source_stats.skipped_no_spans >= 1


def test_source_validation_caps_diagnostic_examples() -> None:
    selection = select_prepared_rows_for_adversarial_slice(
        [{} for _ in range(12)],
        _BehaviorTokenizer(),
        slice_name="line_breaks",
        max_length=32,
        id_to_label=_id_to_label(),
    )

    assert selection.source_stats.skipped_empty_text == 12
    assert len(selection.source_stats.skip_examples) == 10


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("None", None),
        ({"key": "value"}, {"key": "value"}),
        ([{"label": "human_name"}], [{"label": "human_name"}]),
        ("[{'label': 'human_name'}]", [{"label": "human_name"}]),
        ('{"key": "value"}', {"key": "value"}),
        ("not serialized data", None),
        (42, None),
    ],
)
def test_serialized_collection_decoding_is_data_only(
    value: object,
    expected: object,
) -> None:
    assert parse_serialized_collection(value) == expected


def test_remote_code_tokenizer_policy_requires_an_allowlisted_pinned_model() -> None:
    validate_remote_code_tokenizer_policy(
        tokenizer_model_id="untrusted/model",
        tokenizer_revision=None,
        trust_remote_code=False,
    )
    validate_remote_code_tokenizer_policy(
        tokenizer_model_id="LiquidAI/LFM2.5-350M-Base",
        tokenizer_revision="9960764e30892e01f29a6dc23df2533fcd8bd5ae",
        trust_remote_code=True,
    )

    with pytest.raises(ValueError, match="is not allowlisted"):
        validate_remote_code_tokenizer_policy(
            tokenizer_model_id="untrusted/model",
            tokenizer_revision="revision",
            trust_remote_code=True,
        )
    for revision in (None, "", "   "):
        with pytest.raises(ValueError, match="requires a pinned tokenizer revision"):
            validate_remote_code_tokenizer_policy(
                tokenizer_model_id="LiquidAI/LFM2.5-350M-Base",
                tokenizer_revision=revision,
                trust_remote_code=True,
            )
