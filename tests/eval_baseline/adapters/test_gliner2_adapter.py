"""One 40k-char minified blob -> 4000 dense subwords.

Word-windowing made ONE ~56 GiB window; subword-windowing splits it into bounded, fully-covering windows -- no char-cap gap
that would drop the token's interior (the creddata failure mode that hid embedded credentials).

600 whitespace subwords > MAX_SUBWORDS -> two overlapping windows.

"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
import re
from typing import TYPE_CHECKING, Any

from meddies_pii.eval_baseline.adapters.gliner2 import (
    GLINER2_LABEL_FOLD,
    GLINER2_SOURCE_LABELS,
    GLINER2_SUPPORTED_LABELS,
    MAX_SUBWORDS,
    WINDOW_STRIDE_SUBWORDS,
    Gliner2Adapter,
    entity_entries,
    map_gliner2_label,
    spans_from_entities_entry,
    stitch_doc_spans,
    subword_window_ranges,
)
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    import pytest


def _whitespace_offsets(text: str) -> list[tuple[int, int]]:
    """One synthetic subword per whitespace run -- enough to drive window math."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


class _FakeTokenizer:
    """Stands in for the mDeBERTa fast tokenizer: one subword per whitespace run."""

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": _whitespace_offsets(text)}


def test_label_fold_collapses_each_native_group_to_pii_label() -> None:
    """Case-insensitive; unknowns and out-of-scope labels fold to None."""
    assert map_gliner2_label("street_address") == "address"
    assert map_gliner2_label("postal_code") == "address"
    assert map_gliner2_label("country") == "address"
    assert map_gliner2_label("sensitive_date") == "date"
    assert map_gliner2_label("expiration_date") == "date"
    assert map_gliner2_label("email") == "email_address"
    assert map_gliner2_label("person") == "human_name"
    assert map_gliner2_label("last_name") == "human_name"
    assert map_gliner2_label("phone_number") == "phone_number"
    assert map_gliner2_label("passport_number") == "id_number"
    assert map_gliner2_label("account_number") == "id_number"
    assert map_gliner2_label("api_key") == "secret"
    assert map_gliner2_label("recovery_code") == "secret"
    assert map_gliner2_label("PERSON") == "human_name"
    assert map_gliner2_label("diagnosis") is None
    assert map_gliner2_label("company_name") is None
    assert map_gliner2_label("url") is None


def test_supported_labels_drop_company_name_and_private_url() -> None:
    """The schema we send the model is exactly the fold's source vocabulary."""
    assert frozenset(GLINER2_LABEL_FOLD.values()) == GLINER2_SUPPORTED_LABELS
    assert "company_name" not in GLINER2_SUPPORTED_LABELS
    assert "private_url" not in GLINER2_SUPPORTED_LABELS
    assert tuple(GLINER2_LABEL_FOLD) == GLINER2_SOURCE_LABELS


def test_entities_entry_folds_and_keeps_original_offsets() -> None:
    """Unsupported native label -> dropped, never emitted."""
    text = "Email a@b.co about John Smith, ignore diagnosis."
    email_start = text.index("a@b.co")
    name_start = text.index("John Smith")
    diag_start = text.index("diagnosis")
    entry: dict[str, Any] = {
        "email": [{"text": "a@b.co", "start": email_start, "end": email_start + 6}],
        "person": [{"text": "John Smith", "start": name_start, "end": name_start + 10}],
        "diagnosis": [{"text": "diagnosis", "start": diag_start, "end": diag_start + 9}],
    }

    spans = spans_from_entities_entry(text, entry, char_offset=0)

    assert spans == [
        CharSpan(email_start, email_start + 6, "a@b.co", "email_address"),
        CharSpan(name_start, name_start + 10, "John Smith", "human_name"),
    ]
    assert all(span.label in GLINER2_SUPPORTED_LABELS for span in spans)
    assert all(text[span.start : span.end] == span.text for span in spans)


def test_entities_entry_rejects_out_of_bounds_and_nonint_offsets() -> None:
    window = "John"
    entry: dict[str, Any] = {
        "person": [
            {"text": "John", "start": 0, "end": 99},
            {"text": "John", "start": "0", "end": 4},
            {"text": "ok", "start": 0, "end": 4},
        ],
    }
    spans = spans_from_entities_entry(window, entry, char_offset=0)
    assert spans == [CharSpan(0, 4, "John", "human_name")]


def test_entity_entries_accepts_list_and_dict_shapes() -> None:
    """engine.py shape: entities is a list holding one dict."""
    payload: dict[str, Any] = {"person": [{"text": "x", "start": 0, "end": 1}]}
    assert entity_entries({"entities": [payload]}) == [payload]
    assert entity_entries({"entities": payload}) == [payload]
    assert entity_entries({"entities": None}) == []
    assert entity_entries({}) == []


def test_subword_window_ranges_empty_and_single_window() -> None:
    """Zero-width specials dropped; <= cap -> one window spanning first..last subword."""
    assert subword_window_ranges([], max_subwords=480, stride=48) == []
    assert subword_window_ranges([(0, 0), (0, 3), (4, 9), (0, 0)], max_subwords=480, stride=48) == [(0, 9)]


def test_subword_window_ranges_overlaps_and_covers_all() -> None:
    """Adjacent windows overlap (window 2 starts before window 1 ends).

    The whole doc is covered: first subword start .. last subword end.

    """
    offsets = [(i, i + 1) for i in range(600)]
    ranges = subword_window_ranges(offsets, max_subwords=480, stride=48)
    assert len(ranges) == 2
    assert ranges[1][0] < ranges[0][1]
    assert (ranges[0][0], ranges[-1][1]) == (0, 600)


def test_subword_window_ranges_bounds_a_dense_megatoken() -> None:
    """Whole 40k-char blob covered, not truncated at a char cap."""
    offsets = [(i * 10, i * 10 + 10) for i in range(4000)]
    ranges = subword_window_ranges(offsets, max_subwords=480, stride=48)
    assert len(ranges) == 10
    assert (ranges[0][0], ranges[-1][1]) == (0, 40000)


def test_chunk_and_stitch_reassembles_original_offsets_without_dupes() -> None:
    text = " ".join(f"w{i}" for i in range(600))
    offsets = _whitespace_offsets(text)
    windows = subword_window_ranges(offsets, max_subwords=MAX_SUBWORDS, stride=WINDOW_STRIDE_SUBWORDS)
    assert len(windows) == 2
    (ws1, _we1), (ws2, _we2) = windows

    overlap_tok = offsets[450]
    tail_tok = offsets[550]
    overlap_surface = text[overlap_tok[0] : overlap_tok[1]]
    tail_surface = text[tail_tok[0] : tail_tok[1]]

    window_results = [
        {
            "entities": [
                {
                    "person": [
                        {
                            "text": overlap_surface,
                            "start": overlap_tok[0] - ws1,
                            "end": overlap_tok[1] - ws1,
                        },
                    ],
                },
            ],
        },
        {
            "entities": [
                {
                    "person": [
                        {
                            "text": overlap_surface,
                            "start": overlap_tok[0] - ws2,
                            "end": overlap_tok[1] - ws2,
                        },
                    ],
                },
                {
                    "city": [
                        {
                            "text": tail_surface,
                            "start": tail_tok[0] - ws2,
                            "end": tail_tok[1] - ws2,
                        },
                    ],
                },
            ],
        },
    ]

    out = stitch_doc_spans(text, windows, window_results)

    assert out == [
        CharSpan(overlap_tok[0], overlap_tok[1], overlap_surface, "human_name"),
        CharSpan(tail_tok[0], tail_tok[1], tail_surface, "address"),
    ]
    assert all(text[span.start : span.end] == span.text for span in out)


class _FakeModel:
    """Stands in for GLiNER2 on CPU: emits each target span in any window it covers.

    Driven by the same windows the adapter computes, so its window-local offsets
    match what the real model would return after the adapter slices each window.
    """

    def __init__(
        self,
        windows: list[tuple[int, int]],
        targets: list[tuple[int, int, str, str]],
    ) -> None:
        self.windows = windows
        self.targets = targets
        self.calls: list[dict[str, Any]] = []

    def batch_extract(self, texts: list[str], schema: object, **kwargs: object) -> list[dict[str, Any]]:
        self.calls.append({"schema": schema, **kwargs})
        results: list[dict[str, Any]] = []
        for _win_text, (win_start, win_end) in zip(texts, self.windows, strict=True):
            entries: list[dict[str, Any]] = []
            for start, end, native, surface in self.targets:
                if win_start <= start and end <= win_end:
                    entries.append({
                        native: [
                            {
                                "text": surface,
                                "start": start - win_start,
                                "end": end - win_start,
                            },
                        ],
                    })
            results.append({"entities": entries})
        return results


def test_predict_passes_spec_kwargs_and_stitches_single_window() -> None:
    text = "Contact John Smith."
    name_start = text.index("John Smith")
    windows = subword_window_ranges(
        _whitespace_offsets(text),
        max_subwords=MAX_SUBWORDS,
        stride=WINDOW_STRIDE_SUBWORDS,
    )
    target = (name_start, name_start + 10, "person", "John Smith")
    fake = _FakeModel(windows, [target])
    adapter = Gliner2Adapter()
    adapter._model = fake
    adapter._tokenizer = _FakeTokenizer()

    out = adapter.predict([text])

    assert out == [[CharSpan(name_start, name_start + 10, "John Smith", "human_name")]]
    assert fake.calls[0] == {
        "schema": {"entities": list(GLINER2_SOURCE_LABELS)},
        "batch_size": 64,
        "num_workers": 4,
        "include_spans": True,
        "max_len": 512,
    }


def test_predict_chunks_long_doc_dedups_and_logs_rate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = " ".join(f"w{i}" for i in range(600))
    offsets = _whitespace_offsets(text)
    windows = subword_window_ranges(offsets, max_subwords=MAX_SUBWORDS, stride=WINDOW_STRIDE_SUBWORDS)
    assert len(windows) == 2
    overlap_tok = offsets[450]
    tail_tok = offsets[550]
    overlap_surface = text[overlap_tok[0] : overlap_tok[1]]
    tail_surface = text[tail_tok[0] : tail_tok[1]]

    targets = [
        (overlap_tok[0], overlap_tok[1], "person", overlap_surface),
        (tail_tok[0], tail_tok[1], "city", tail_surface),
    ]
    fake = _FakeModel(windows, targets)
    adapter = Gliner2Adapter()
    adapter._model = fake
    adapter._tokenizer = _FakeTokenizer()

    out = adapter.predict([text])

    assert out == [
        [
            CharSpan(overlap_tok[0], overlap_tok[1], overlap_surface, "human_name"),
            CharSpan(tail_tok[0], tail_tok[1], tail_surface, "address"),
        ],
    ]
    captured = capsys.readouterr()
    assert "GLINER2_CHUNKED::docs=1 chunked=1" in captured.out
