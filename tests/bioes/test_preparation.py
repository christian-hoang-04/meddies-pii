from __future__ import annotations

from typing import TYPE_CHECKING, cast

from meddies_pii.annotations.bioes import TokenizedExample
from meddies_pii.annotations.tagged_text import ParsedTaggedDocument
from meddies_pii.spans import CharSpan
from meddies_pii.training.bioes.data import preparation
from meddies_pii.training.bioes.data.preparation import AuditedSourceRow, _prepare_rows

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest
    from transformers import PreTrainedTokenizerBase


def _audited(uid: str, raw: str) -> AuditedSourceRow:
    span = CharSpan(start=0, end=len(raw), text=raw, label="name")
    parsed = ParsedTaggedDocument(
        text=raw,
        normalized_text=raw,
        had_label_repairs=False,
        raw=raw,
        spans=(span,),
    )
    label_json = {"name": [raw]}
    return AuditedSourceRow(
        uid=uid,
        source={"uid": uid},
        parsed=parsed,
        provided_label_json=label_json,
        parsed_label_json=label_json,
    )


def test_prepare_rows_treats_limit_as_upper_bound_after_quality_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # reason: `_prepare_rows` calls `tokenize_and_align(tokenizer, raw, spans, max_length=..., entity_labels=...)`,
    # reason: so these two names are written at the call site and the stand-in cannot rename them.
    def fake_tokenize(
        _tokenizer: object,
        raw: str,
        target_spans: Sequence[CharSpan],
        *,
        max_length: int,  # ruff: ignore[unused-function-argument]
        entity_labels: Sequence[str],  # ruff: ignore[unused-function-argument]
    ) -> TokenizedExample:
        return TokenizedExample(
            raw=raw,
            spans=tuple(target_spans),
            input_ids=[1],
            attention_mask=[1],
            labels=[1],
            offset_mapping=[(0, len(raw))],
            truncated=False,
        )

    def fake_decode(
        raw: str,
        _offset_mapping: Sequence[tuple[int, int]],
        _labels: Sequence[int],
        _id_to_label: dict[int, str],
    ) -> list[CharSpan]:
        if raw == "bad":
            return []
        return [CharSpan(start=0, end=len(raw), text=raw, label="name")]

    monkeypatch.setattr(preparation, "tokenize_and_align", fake_tokenize)
    monkeypatch.setattr(preparation, "decode_bioes_from_offsets", fake_decode)

    prepared_rows, stats = _prepare_rows(
        [_audited("good", "good"), _audited("bad", "bad")],
        # reason: the double exposes only the members this call path uses; test-side narrowing is the
        # reason: owner-ruled shape for a stand-in that cannot be a real PreTrainedTokenizerBase.
        tokenizer=cast("PreTrainedTokenizerBase", object()),
        max_length=32,
        limit=2,
        id_to_label={1: "S-name"},
    )

    assert [row.uid for row in prepared_rows] == ["good"]
    assert stats.accepted == 1
    assert stats.skipped_round_trip == 1
