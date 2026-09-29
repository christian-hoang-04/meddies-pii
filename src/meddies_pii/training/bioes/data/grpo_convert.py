"""GRPO Gemini-batch hard-examples -> Meddies Labels conversion (core logic).

The hard-examples are a Gemini batch job, not the PII-label corpus: the source clinical doc
lives in the batch INPUT prompt (under a ``## Text`` header) and the PII
extraction (label -> [values]) in the batch OUTPUT. The functions here locate
the doc and rebuild ``{category,start,end,text}`` spans by finding each
extracted value in it. Gemini emitted 7 of the 9 labels (no private_url/secret);
values absent from the doc (hallucinations) are dropped, every found occurrence
is tagged, and cross-label overlaps resolve to the longer span.

I/O + CLI lives in ``scripts/migrations/convert_grpo_gemini.py``; this module is pure +
testable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import cast

from meddies_pii.taxonomy import PII_LABELS

VALID_LABELS = frozenset(PII_LABELS)
_NEXT_HEADER = re.compile(r"\n## ")

_GRPO_SOURCE_TO_LANGUAGE = {
    "nvidia-health": "en",
    "nvidia-non-health": "en",
    "vietnamese-translated": "vi",
}
"""The HF grpo configs tag rows by their origin config, not always by language.

These two are Nemotron-derived English clinical text; vietnamese-translated is Vietnamese. Everything else carries a real
language name (burmese, tamil, ...) that the gate resolver maps to a grid code directly.

"""


def extract_document(prompt: str) -> str:
    """Extract the clinical doc sitting between '## Text' and the next '## ' header.

    Returns:
        The document text, stripped. When the prompt carries no ``## Text`` marker the split
        yields the whole prompt, and when no following header exists the rest of the body is
        taken -- both degrade to returning more text rather than returning nothing.

    """
    body = prompt.split("## Text\n", 1)[-1]
    match = _NEXT_HEADER.search(body)
    return (body[: match.start()] if match else body).strip()


def find_spans(text: str, extractions: Mapping[str, object]) -> list[dict[str, object]]:
    """Tag every occurrence of each extracted value; overlaps resolve to longer.

    earliest start first, then longest, so the longer span wins an overlap.

    Returns:
        One span record per surviving occurrence. Every occurrence of every extracted value is
        tagged, then overlaps are resolved by taking the earliest start and the longest span,
        so a value contained inside a longer one is dropped rather than double-counted. A label
        outside the valid set, or a value list that is not a non-string sequence, contributes
        nothing.

    """
    raw: list[tuple[int, int, str, str]] = []
    for label, values in extractions.items():
        if label not in VALID_LABELS or not isinstance(values, Sequence) or isinstance(values, str):
            continue
        for raw_value in values:
            value = str(raw_value).strip()
            if not value:
                continue
            start = 0
            while (idx := text.find(value, start)) >= 0:
                raw.append((idx, idx + len(value), label, value))
                start = idx + len(value)
    raw.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    spans: list[dict[str, object]] = []
    last_end = -1
    for start, end, label, value in raw:
        if start >= last_end:
            spans.append({"category": label, "start": start, "end": end, "text": value})
            last_end = end
    return spans


def to_record(prompt: str, extractions: Mapping[str, object], *, uid: str) -> dict[str, object] | None:
    doc = extract_document(prompt)
    if not doc:
        return None
    spans = find_spans(doc, extractions)
    if not spans:
        return None
    return {
        "text": doc,
        "label": spans,
        "info": {
            "source": "grpo-gemini",
            "source_dataset": "hard-examples/gemini-batch",
            "language": "unknown",
            "uid": uid,
        },
    }


def convert_grpo_hf_row(row: Mapping[str, object], *, uid: str) -> dict[str, object] | None:
    """Convert one HF grpo-train / grpo-hard-train row to a Meddies Labels record.

    Those configs carry ``{raw_text, source, answer}`` where ``answer`` is a JSON
    string (or already-parsed dict) of ``label -> [values]`` — the same
    extraction shape :func:`find_spans` rebuilds. The doc lives directly in
    ``raw_text`` (no ``## Text`` prompt wrapper). ``source`` is the language,
    mapped through :data:`_GRPO_SOURCE_TO_LANGUAGE` for the two dataset-named
    origins. Returns ``None`` when the text is empty, the answer is unparseable,
    or no extracted value is found in the doc.

    Returns:
        The converted record, or ``None`` for the three unusable cases above. ``None`` means
        drop this row, not fail the conversion: these datasets carry rows whose stated answers
        do not appear in their own text, and a record with labels that match nothing would
        train the tagger against its own input.

    """
    raw_text = row.get("raw_text")
    if not isinstance(raw_text, str):
        return None
    text = raw_text.strip()
    if not text:
        return None

    answer = row.get("answer")
    if isinstance(answer, str):
        try:
            extractions = json.loads(answer)
        except json.JSONDecodeError:
            return None
    else:
        extractions = answer
    if not isinstance(extractions, Mapping):
        return None

    spans = find_spans(text, cast("Mapping[str, object]", extractions))
    if not spans:
        return None

    source = str(row.get("source") or "").strip()
    language = _GRPO_SOURCE_TO_LANGUAGE.get(source, source) or "unknown"
    return {
        "text": text,
        "label": spans,
        "info": {
            "source": "grpo-hf",
            "source_dataset": "Meddies/meddies-pii (grpo configs)",
            "language": language,
            "uid": uid,
        },
    }
