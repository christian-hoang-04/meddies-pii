"""Golden behavior for dependency-neutral annotation and span metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from meddies_pii.annotations import tagged_text
from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    build_label_to_id,
    decode_bioes_from_offsets,
    tokenize_and_align,
    viterbi_decode_numpy,
)
from meddies_pii.annotations.source_mapping import map_native_label_to_pii_label
from meddies_pii.annotations.span_records import redaction_record
from meddies_pii.annotations.tagged_text import parse_tagged_text
from meddies_pii.evaluation.span_metrics import exact_span_report_by_doc
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class _OverflowTokenizer:
    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "input_ids": [[101, 7, 102], [101, 8, 102]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
            "offset_mapping": [[(0, 0), (0, 4), (0, 0)], [(0, 0), (5, 9), (0, 0)]],
        }


def test_source_map_precedence_and_parser_repairs_are_stable() -> None:
    assert map_native_label_to_pii_label("B-FIRST_NAME") == "human_name"
    assert map_native_label_to_pii_label("PRIVATE_URL") == "private_url"
    assert map_native_label_to_pii_label("diagnosis") is None
    assert map_native_label_to_pii_label("name", overrides={"name": "secret"}) == "secret"

    parsed = parse_tagged_text(
        "x<date> then [Tran Bao]name at [18/7]<date>",
        normalize_labels=True,
    )
    assert parsed.had_label_repairs is True
    assert parsed.raw == "x then Tran Bao at 18/7"
    assert [(span.start, span.end, span.text, span.label) for span in parsed.spans] == [
        (7, 15, "Tran Bao", "human_name"),
        (19, 23, "18/7", "date"),
    ]


def test_tagged_text_does_not_expose_bioes_inference_contracts() -> None:
    assert not hasattr(tagged_text, "TokenizedExample")
    assert not hasattr(tagged_text, "VITERBI_BIAS_KEYS")


def test_bioes_vocabulary_ids_paths_spans_and_truncation_are_stable() -> None:
    vocabulary = build_bioes_label_space(ENTITY_LABELS)
    ids = build_label_to_id(vocabulary)
    assert vocabulary[:9] == (
        "O",
        "B-address",
        "I-address",
        "E-address",
        "S-address",
        "B-company_name",
        "I-company_name",
        "E-company_name",
        "S-company_name",
    )
    assert ids["S-human_name"] == 20

    offsets = [(0, 0), (0, 4), (5, 9), (10, 13), (0, 0)]
    logits = np.full((5, len(vocabulary)), -10.0, dtype=np.float32)
    logits[:, ids["O"]] = 10.0
    logits[2, ids["B-human_name"]] = 20.0
    logits[3, ids["E-human_name"]] = 20.0
    decoded = viterbi_decode_numpy(logits, dict(enumerate(vocabulary)), offsets)
    assert decoded == [-100, 0, 17, 19, -100]
    assert decode_bioes_from_offsets("Call John Doe", offsets, decoded, dict(enumerate(vocabulary))) == (
        CharSpan(start=5, end=13, text="John Doe", label="human_name"),
    )

    # reason: the double exposes only the members this call path uses; test-side narrowing is the
    # reason: owner-ruled shape for a stand-in that cannot be a real PreTrainedTokenizerBase.
    example = tokenize_and_align(
        cast("PreTrainedTokenizerBase", _OverflowTokenizer()),
        "John Doe",
        [],
        max_length=3,
    )
    assert example.input_ids == [101, 7, 102]
    assert example.truncated is True


def test_rendering_and_metric_payloads_are_stable() -> None:
    text = "Call John"
    span = CharSpan(start=5, end=9, text="John", label="human_name")
    assert redaction_record(text=text, spans=[span]) == {
        "schema_version": 1,
        "summary": {
            "output_mode": "typed",
            "span_count": 1,
            "by_label": {"human_name": 1},
            "decoded_mismatch": False,
        },
        "text": text,
        "detected_spans": [
            {
                "label": "human_name",
                "start": 5,
                "end": 9,
                "text": "John",
                "placeholder": "<HUMAN_NAME>",
            },
        ],
        "redacted_text": "Call <HUMAN_NAME>",
    }
    assert exact_span_report_by_doc({"doc": [span]}, {"doc": [span]}) == {
        "tp": 1.0,
        "pred_total": 1.0,
        "gold_total": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "typed": {
            "tp": 1.0,
            "pred_total": 1.0,
            "gold_total": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        "untyped": {
            "tp": 1.0,
            "pred_total": 1.0,
            "gold_total": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        "untyped_minus_typed": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    }
