"""parse_tagged_text — stray `<label>` marker handling (records.py).

The model forgot the `[...]` wrapper, leaving a bare `<address>` marker. It must NOT leak into the raw text (it is a
generation artifact, never real clinical text) — and the valid span in the same doc is preserved.

Offsets must index the cleaned raw text exactly even when a stray marker was removed before the span.

"""

from __future__ import annotations

from meddies_pii.annotations.tagged_text import parse_tagged_text


def test_parses_valid_inline_tag() -> None:
    p = parse_tagged_text("Call [Tran Bao]<human_name> today.")
    assert p.raw == "Call Tran Bao today."
    assert [(s.text, s.label) for s in p.spans] == [("Tran Bao", "human_name")]


def test_strips_stray_label_marker_without_brackets() -> None:
    p = parse_tagged_text("Dia chi 18/7 Trai, TP.HCM<address>; bs [Tran Bao]<human_name>.")
    assert "<address>" not in p.raw
    assert p.raw == "Dia chi 18/7 Trai, TP.HCM; bs Tran Bao."
    assert [(s.text, s.label) for s in p.spans] == [("Tran Bao", "human_name")]


def test_strips_closing_stray_marker() -> None:
    p = parse_tagged_text("token abc</secret> and [Tran Bao]<human_name>")
    assert "</secret>" not in p.raw
    assert [(s.label) for s in p.spans] == ["human_name"]


def test_span_offsets_stay_exact_after_stripping() -> None:
    p = parse_tagged_text("x<date> then [Tran Bao]<human_name> end")
    for span in p.spans:
        assert p.raw[span.start : span.end] == span.text
