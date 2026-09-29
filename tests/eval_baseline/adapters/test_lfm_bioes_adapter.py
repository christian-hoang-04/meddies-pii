"""CPU unit tests for the LFM BIOES spike adapter's decode path.

These exercise ``spans_from_logits`` with hand-built logits + a real BIOES label
space — no GPU, no checkpoint. They verify the Viterbi + BIOES-decode glue produces
the right char spans for single-token (S), multi-token (B/E), and empty (all-O)
cases. Tokenizer char-offset alignment on multibyte scripts (vi/ja/zh) is a property
of the fast tokenizer and is eye-verified on-device at first-light, not here — a
hand-fed offset_mapping can't test the tokenizer.
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
import torch

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    build_label_to_id,
)
from meddies_pii.eval_baseline.adapters.lfm_bioes import (
    LFM_BIOES_SUPPORTED_LABELS,
    LfmBioesAdapter,
    spans_from_logits,
)
from meddies_pii.taxonomy import PII_LABEL_SET

_LABEL_VOCAB = build_bioes_label_space(ENTITY_LABELS)
_LABEL_TO_ID = build_label_to_id(_LABEL_VOCAB)
_ID_TO_LABEL = {index: label for label, index in _LABEL_TO_ID.items()}
_NUM_LABELS = len(_LABEL_VOCAB)


def _logits_for(peaks: dict[int, str], *, num_tokens: int) -> torch.Tensor:
    """Build a ``[tokens, labels]`` tensor that peaks at the named label per token.

    ``peaks`` maps token index -> BIOES label (e.g. ``"S-human_name"``); any token not
    named peaks at ``"O"``. A peak of +10 against a -10 floor makes the argmax (and the
    Viterbi path, since these sequences are all BIOES-legal) unambiguous.
    """
    logits = torch.full((num_tokens, _NUM_LABELS), -10.0)
    for token_idx in range(num_tokens):
        label = peaks.get(token_idx, "O")
        logits[token_idx, _LABEL_TO_ID[label]] = 10.0
    return logits


def test_supported_labels_is_the_full_nine() -> None:
    """The spike is one of only two baselines (with openmed) that covers both weak labels.

    The whole reason it is the interesting own-architecture floor.

    """
    assert LFM_BIOES_SUPPORTED_LABELS == PII_LABEL_SET
    assert LfmBioesAdapter.supported_labels == LFM_BIOES_SUPPORTED_LABELS
    assert {"private_url", "secret"} <= LFM_BIOES_SUPPORTED_LABELS
    assert len(LFM_BIOES_SUPPORTED_LABELS) == 9


def test_single_token_entity_decodes_to_one_span() -> None:
    """Special (0,0), "Call" (0,4), "John" (5,9), "now" (10,13)."""
    text = "Call John now"
    offset_mapping = [(0, 0), (0, 4), (5, 9), (10, 13)]
    logits = _logits_for({2: "S-human_name"}, num_tokens=4)
    spans = spans_from_logits(text, offset_mapping, logits, _ID_TO_LABEL)
    assert len(spans) == 1
    span = spans[0]
    assert (span.start, span.end, span.label) == (5, 9, "human_name")
    assert span.text == text[span.start : span.end] == "John"


def test_multi_token_entity_merges_b_e_into_one_span() -> None:
    """Special (0,0), "to" (0,2), "Maria" (3,8), "Garcia" (9,15), "now" (16,19)."""
    text = "to Maria Garcia now"
    offset_mapping = [(0, 0), (0, 2), (3, 8), (9, 15), (16, 19)]
    logits = _logits_for({2: "B-human_name", 3: "E-human_name"}, num_tokens=5)
    spans = spans_from_logits(text, offset_mapping, logits, _ID_TO_LABEL)
    assert len(spans) == 1
    span = spans[0]
    assert (span.start, span.end, span.label) == (3, 15, "human_name")
    assert span.text == text[span.start : span.end] == "Maria Garcia"


def test_all_o_yields_no_spans() -> None:
    text = "nothing to see here"
    offset_mapping = [(0, 0), (0, 7), (8, 10), (11, 14), (15, 19)]
    logits = _logits_for({}, num_tokens=5)
    spans = spans_from_logits(text, offset_mapping, logits, _ID_TO_LABEL)
    assert spans == []
