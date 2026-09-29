"""GRPO Gemini-batch -> Meddies Labels conversion (scripts/migrations/convert_grpo_gemini.py).

extraction hallucination: value absent from the doc -> no span.

The HF grpo-train / grpo-hard-train configs carry {raw_text, source, answer(label->values JSON)} — the same (text,
extractions) shape find_spans converts. source is the language; the doc lives directly in raw_text.

source "nvidia-health"/"nvidia-non-health" are dataset names, not langs; the Nemotron-derived rows are English clinical
text -> map to en so they land in the grid instead of resolving to None (excluded).

"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import cast

from meddies_pii.training.bioes.data.grpo_convert import (
    convert_grpo_hf_row,
    extract_document,
    find_spans,
    to_record,
)


def test_moved_grpo_gemini_script_defaults_resolve_repo_root() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts/migrations/convert_grpo_gemini.py"
    spec = importlib.util.spec_from_file_location("convert_grpo_gemini", script)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert repo == module.REPO
    assert (module.REPO / "pyproject.toml").is_file()


def test_extract_document_stops_at_next_header() -> None:
    prompt = "## Text\nDr. Aung Myint at Yangon General.\n\n## Draft PII extraction\n{...}\n## Instructions\nx"
    assert extract_document(prompt) == "Dr. Aung Myint at Yangon General."


def test_find_spans_locates_each_value() -> None:
    text = "Dr. Aung Myint at Yangon General Hospital, call 09-876-5432."
    spans = find_spans(
        text,
        {
            "human_name": ["Aung Myint"],
            "company_name": ["Yangon General Hospital"],
            "phone_number": ["09-876-5432"],
        },
    )
    got = {(s["category"], s["text"]) for s in spans}
    assert got == {
        ("human_name", "Aung Myint"),
        ("company_name", "Yangon General Hospital"),
        ("phone_number", "09-876-5432"),
    }
    for s in spans:
        start, end = s["start"], s["end"]
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert text[start:end] == s["text"]


def test_find_spans_drops_value_not_in_text() -> None:
    spans = find_spans("Patient seen today.", {"human_name": ["Ghost Patient"]})
    assert spans == []


def test_find_spans_tags_every_occurrence() -> None:
    text = "Nguyen and Nguyen again."
    spans = find_spans(text, {"human_name": ["Nguyen"]})
    assert len(spans) == 2


def test_find_spans_resolves_overlap_to_longer() -> None:
    text = "Dr. Aung Myint Senior"
    spans = find_spans(text, {"human_name": ["Aung Myint", "Aung Myint Senior"]})
    assert len(spans) == 1
    assert spans[0]["text"] == "Aung Myint Senior"


def test_find_spans_drops_non_pii_label_label() -> None:
    spans = find_spans("Age 45 noted.", {"age": ["45"], "human_name": []})
    assert spans == []


def test_to_record_emits_pii_label_shape() -> None:
    prompt = "## Text\nCall Dr. Lee.\n## Instructions\nx"
    rec = to_record(prompt, {"human_name": ["Lee"]}, uid="grpo-7")
    assert rec is not None
    assert rec["text"] == "Call Dr. Lee."
    info = cast("dict[str, object]", rec["info"])
    assert info["source"] == "grpo-gemini"
    label = cast("list[dict[str, object]]", rec["label"])
    assert label[0]["category"] == "human_name"


def test_to_record_returns_none_when_no_spans() -> None:
    prompt = "## Text\nNo PII here at all.\n## Instructions\nx"
    assert to_record(prompt, {"human_name": ["Nobody"]}, uid="grpo-8") is None


def test_convert_grpo_hf_row_builds_pii_label_from_raw_text_and_answer() -> None:
    row = {
        "raw_text": "Dr. Aung Myint at Yangon General Hospital.",
        "source": "burmese",
        "answer": ('{"human_name": ["Aung Myint"], "company_name": ["Yangon General Hospital"]}'),
    }
    rec = convert_grpo_hf_row(row, uid="g-1")
    assert rec is not None
    info = cast("dict[str, object]", rec["info"])
    assert info["language"] == "burmese"
    label = cast("list[dict[str, object]]", rec["label"])
    assert {s["category"] for s in label} == {"human_name", "company_name"}
    text = cast("str", rec["text"])
    for s in label:
        start, end = cast("int", s["start"]), cast("int", s["end"])
        assert text[start:end] == s["text"]


def test_convert_grpo_hf_row_maps_nvidia_source_to_en() -> None:
    row = {
        "raw_text": "Call Dr. Lee now.",
        "source": "nvidia-health",
        "answer": '{"human_name": ["Lee"]}',
    }
    rec = convert_grpo_hf_row(row, uid="g-2")
    assert rec is not None
    info = cast("dict[str, object]", rec["info"])
    assert info["language"] == "en"


def test_convert_grpo_hf_row_maps_vietnamese_translated_to_vi() -> None:
    row = {
        "raw_text": "Goi Dr. Le.",
        "source": "vietnamese-translated",
        "answer": '{"human_name": ["Le"]}',
    }
    rec = convert_grpo_hf_row(row, uid="g-3")
    assert rec is not None
    info = cast("dict[str, object]", rec["info"])
    assert info["language"] == "vi"


def test_convert_grpo_hf_row_accepts_dict_answer() -> None:
    row = {
        "raw_text": "Call Lee.",
        "source": "english",
        "answer": {"human_name": ["Lee"]},
    }
    assert convert_grpo_hf_row(row, uid="g-4") is not None


def test_convert_grpo_hf_row_none_on_hallucinated_only() -> None:
    row = {
        "raw_text": "No PII present.",
        "source": "english",
        "answer": '{"human_name": ["Ghost"]}',
    }
    assert convert_grpo_hf_row(row, uid="g-5") is None


def test_convert_grpo_hf_row_none_on_bad_answer_json() -> None:
    row = {"raw_text": "Call Lee.", "source": "english", "answer": "not json"}
    assert convert_grpo_hf_row(row, uid="g-6") is None
