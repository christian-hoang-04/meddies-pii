"""GLiNER2 privacy-filter adapter for the shared PII eval loop.

``fastino/gliner2-privacy-filter-PII-multi`` is an open-vocabulary GLiNER2
bi-encoder (mDeBERTa-v3-base). We feed it the privacy filter's native label
strings as the schema, then fold each native label back to the PII-label.

Two things differ from the OpenMed (HF-pipeline) adapter and drive this module:

1. gliner2 is not a project dependency (it ships only in the Modal image), so we
   resolve it via ``importlib`` at call time — a static ``import gliner2`` would
   be an unresolved-import type error in the project gate.
2. gliner2 has NO sliding window. ``batch_extract`` counts length in WORD tokens
   and *silently drops* everything past ``max_len`` (``text_tokens[:max_len]``).
   The HF pipeline gave OpenMed chunk-and-stitch for free via ``stride``; here we
   implement it ourselves: split a long doc into overlapping word windows, map
   each window's spans back onto the original text, and dedup the overlap.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: adapters load their model stack inside `load()`, so listing an adapter costs nothing.
# ruff: file-ignore[print]
# reason: per-shard telemetry goes to the run log so a degraded cell is visible rather than silent.
import importlib
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from meddies_pii.eval_baseline.baseline.span_cleanup import clean_spans
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from meddies_pii.taxonomy import PiiLabel

MODEL_ID = "fastino/gliner2-privacy-filter-PII-multi"
MODEL_REVISION = "59894c087cb2923b01f337d4ee72f6ff84d5bdd6"

GLINER2_LABEL_FOLD: dict[str, PiiLabel] = {
    "address": "address",
    "street_address": "address",
    "city": "address",
    "state_or_region": "address",
    "postal_code": "address",
    "country": "address",
    "sensitive_date": "date",
    "document_date": "date",
    "expiration_date": "date",
    "transaction_date": "date",
    "email": "email_address",
    "person": "human_name",
    "full_name": "human_name",
    "first_name": "human_name",
    "last_name": "human_name",
    "phone_number": "phone_number",
    "government_id": "id_number",
    "national_id_number": "id_number",
    "passport_number": "id_number",
    "drivers_license_number": "id_number",
    "tax_id": "id_number",
    "account_number": "id_number",
    "password": "secret",
    "api_key": "secret",
    "access_token": "secret",
    "recovery_code": "secret",
}
"""The privacy filter's native open vocabulary -> the PII-label.

company_name and private_url have no native source label here, so they fall outside supported_labels (the eval scores gold
for them as out-of-scope, not as misses).

"""

GLINER2_SOURCE_LABELS: tuple[str, ...] = tuple(GLINER2_LABEL_FOLD)
"""Native label strings handed to gliner2 as the open-vocab entity schema."""

GLINER2_SUPPORTED_LABELS: frozenset[PiiLabel] = frozenset(GLINER2_LABEL_FOLD.values())

WINDOW_TOKENIZER_ID = "microsoft/mdeberta-v3-base"
"""gliner2 is mDeBERTa-v3-base; this is its subword tokenizer.

We window each doc by SUBWORD count (using its offset_mapping) so the sequence length the model sees is bounded regardless
of content.

"""
WINDOW_TOKENIZER_REVISION = "a0484667b22365f84929a935b5e50a51f71f159d"
MAX_SUBWORDS = 480
"""Window cap in mDeBERTa subwords.

Bounding by SUBWORDS (not words or chars) is what caps mDeBERTa's O(L^2) attention: a 512-WORD window of one minified
mega-token is ~6-8k subwords (the deterministic ~56 GiB OOM on creddata), but a 480-SUBWORD window is always ~480 tokens of
text. Headroom under gliner2's hard 512 leaves room for the schema-label tokens it prepends. Mirrors openmed's subword
pipeline- stride, which is exactly why openmed survived creddata where word-windowing OOM'd.

"""
WINDOW_STRIDE_SUBWORDS = 48
"""Overlap so an entity straddling a window boundary still lands whole in one window."""
GLINER2_MAX_LEN = 512
"""gliner2's hard internal word cap (its max_len).

Our windows are <=MAX_SUBWORDS subwords (so <=MAX_SUBWORDS words, since subwords >= words), so this never truncates a
window -- it's the documented sentinel-avoidance (max_len=None silently truncates at 512).

"""
MAX_WINDOWS_PER_CALL = 256
"""A code-heavy cell (creddata) explodes into many windows.

Passing them all to one batch_extract materializes every tokenized window at once -> OOM. Slice into bounded calls so peak
memory is independent of the cell's window count.

"""


def map_gliner2_label(label: str) -> PiiLabel | None:
    return GLINER2_LABEL_FOLD.get(label.strip().lower())


def subword_window_ranges(
    offsets: Sequence[tuple[int, int]],
    *,
    max_subwords: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Char ranges of overlapping <=``max_subwords``-subword windows over a doc.

    ``offsets`` are ``(char_start, char_end)`` per subword from a fast tokenizer's
    ``offset_mapping``; zero-width specials are dropped. Bounding the window by
    SUBWORD count (not words or chars) caps mDeBERTa's O(L^2) attention regardless
    of content -- a minified mega-token spreads across many windows instead of
    exploding one, and the whole doc stays covered (no char-cap gap that would drop
    the token's interior). No subwords -> no windows. <=``max_subwords`` -> one
    window. Longer -> slide by ``max_subwords - stride`` so adjacent windows share
    ``stride`` subwords of overlap.

    Returns:
        The window char ranges, covering the document with no gap. Empty when no subword has
        width; one window spanning the first subword's start to the last one's end when the
        doc fits; otherwise a sliding series whose neighbours share ``stride`` subwords.

    """
    spans = [(start, end) for start, end in offsets if end > start]
    if not spans:
        return []
    if len(spans) <= max_subwords:
        return [(spans[0][0], spans[-1][1])]
    step = max(1, max_subwords - stride)
    ranges: list[tuple[int, int]] = []
    start_idx = 0
    while start_idx < len(spans):
        end_idx = min(start_idx + max_subwords, len(spans))
        ranges.append((spans[start_idx][0], spans[end_idx - 1][1]))
        if end_idx >= len(spans):
            break
        start_idx += step
    return ranges


def entity_entries(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract the ``label -> spans`` dicts inside one text's gliner2 result.

    ``result["entities"]`` is a list holding a single OrderedDict (engine.py), but
    some versions/docs return the dict directly — accept both.

    Returns:
        The mapping entries, one per result shape: the dict itself wrapped in a list when the
        engine returned it bare, the mapping members when it returned a sequence, and empty
        for anything else. A str or bytes is rejected rather than iterated, since both are
        sequences that would otherwise decompose into characters.

    """
    raw = result.get("entities")
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [entry for entry in raw if isinstance(entry, Mapping)]
    return []


def _char_span_from_raw(
    window_text: str,
    raw_span: object,
    label: PiiLabel,
    char_offset: int,
) -> CharSpan | None:
    if not isinstance(raw_span, Mapping):
        return None
    span_map = cast("Mapping[str, Any]", raw_span)
    start = span_map.get("start")
    end = span_map.get("end")
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end <= start or end > len(window_text):
        return None
    return CharSpan(
        start=char_offset + start,
        end=char_offset + end,
        text=window_text[start:end],
        label=label,
    )


def spans_from_entities_entry(
    window_text: str,
    entry: Mapping[str, Any],
    *,
    char_offset: int,
) -> list[CharSpan]:
    """Fold one ``{native_label: [span, ...]}`` entry into supported CharSpans.

    Native labels outside the fold are dropped, so the adapter can never emit a
    label outside supported_labels.

    Returns:
        The spans whose native label folds to a supported one, in entry order. A label
        outside the fold is dropped rather than passed through, which is what makes the
        adapter's output set a subset of ``supported_labels`` by construction.

    """
    spans: list[CharSpan] = []
    for raw_label, raw_spans in entry.items():
        if not isinstance(raw_label, str):
            continue
        if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
            continue
        label = map_gliner2_label(raw_label)
        if label is None:
            continue
        for raw_span in raw_spans:
            span = _char_span_from_raw(window_text, raw_span, label, char_offset)
            if span is not None:
                spans.append(span)
    return spans


def stitch_doc_spans(
    text: str,
    windows: Sequence[tuple[int, int]],
    window_results: Sequence[Mapping[str, Any]],
) -> list[CharSpan]:
    """Reassemble one doc's spans from its per-window gliner2 results.

    Remaps every window-local span onto the original text, dedups the overlap
    region by (start, end, label), then runs clean_spans for boundary repair +
    same-label fragment merge.

    Returns:
        The document's spans in original-text coordinates, deduplicated on
        ``(start, end, label)`` so a span found in two overlapping windows appears once, then
        passed through ``clean_spans`` for boundary repair and same-label fragment merging.

    """
    collected: list[CharSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for (window_start, window_end), result in zip(windows, window_results, strict=True):
        window_text = text[window_start:window_end]
        for entry in entity_entries(result):
            for span in spans_from_entities_entry(window_text, entry, char_offset=window_start):
                key = (span.start, span.end, span.label)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(span)
    return clean_spans(text, collected)


class Gliner2Adapter:
    name = "gliner2"
    supported_labels = GLINER2_SUPPORTED_LABELS

    def __init__(self, *, batch_size: int = 64, num_workers: int = 4) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        """FLASH_DEBERTA must be set before gliner2 imports DeBERTa.

        So set it here right before the import (belt-and-suspenders with the Modal image .env).

        compile=False (eager): the eval feeds variable-length windows, and compile/CUDA-graphs re-capture per shape ->
        graph pools explode to tens of GiB (the same trap that OOM'd openmed under reduce-overhead). The eval scores
        accuracy, not throughput, so eager is the right call.

        The mDeBERTa fast tokenizer drives subword-count windowing (its offset_mapping maps each subword back to a char
        span). Same vocab the model tokenizes with, so our window subword counts match the model's.

        """
        if self._model is not None:
            return
        os.environ["FLASH_DEBERTA"] = "1"
        gliner2_module: Any = importlib.import_module("gliner2")
        hub_module: Any = importlib.import_module("huggingface_hub")
        snapshot_path = hub_module.snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        self._model = gliner2_module.GLiNER2.from_pretrained(
            snapshot_path,
            map_location="cuda",
            quantize=True,
            compile=False,
        )
        if self._tokenizer is None:
            # reason: transformers declares its public names only under `TYPE_CHECKING` and
            # reason: serves them at runtime through `_LazyModule`, so a static reader cannot
            # reason: prove the symbol is present. Verified against the pinned 5.14.1 in this
            # reason: environment: `hasattr(transformers, "AutoTokenizer")` is True.
            from transformers import (
                AutoTokenizer,  # ty: ignore[possibly-missing-import]
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                WINDOW_TOKENIZER_ID,
                revision=WINDOW_TOKENIZER_REVISION,
                use_fast=True,
            )

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        if self._model is None or self._tokenizer is None:
            msg = "Gliner2Adapter.load() must be called before predict()"
            raise RuntimeError(msg)

        doc_windows: list[list[tuple[int, int]]] = []
        flat_window_texts: list[str] = []
        flat_doc_index: list[int] = []
        chunked_docs = 0
        for doc_index, text in enumerate(texts):
            ranges = self._doc_windows(text)
            if len(ranges) > 1:
                chunked_docs += 1
            doc_windows.append(ranges)
            for window_start, window_end in ranges:
                flat_window_texts.append(text[window_start:window_end])
                flat_doc_index.append(doc_index)

        raw_results = self._batch_extract(flat_window_texts)
        if len(raw_results) != len(flat_window_texts):
            msg = f"gliner2 returned {len(raw_results)} results for {len(flat_window_texts)} windows"
            raise RuntimeError(msg)

        per_doc_results: list[list[Mapping[str, Any]]] = [[] for _ in texts]
        for doc_index, result in zip(flat_doc_index, raw_results, strict=True):
            per_doc_results[doc_index].append(result)

        predictions = [
            stitch_doc_spans(text, doc_windows[doc_index], per_doc_results[doc_index])
            for doc_index, text in enumerate(texts)
        ]
        rate = chunked_docs / len(texts) if texts else 0.0
        print(
            f"GLINER2_CHUNKED::docs={len(texts)} chunked={chunked_docs} rate={rate:.3f}",
            flush=True,
        )
        return predictions

    def _doc_windows(self, text: str) -> list[tuple[int, int]]:
        """Subword-bounded char windows for one doc via the mDeBERTa tokenizer.

        Returns:
            The window char ranges for this text, measured with the adapter's own tokenizer
            so the subword bound matches the model that will read them.

        Raises:
            RuntimeError: If ``load()`` has not run, so there is no tokenizer to measure
                with. It refuses rather than falling back to a character estimate, which
                would silently change what the window bound means.

        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            msg = "Gliner2Adapter.load() must be called before predict()"
            raise RuntimeError(msg)
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
        return subword_window_ranges(offsets, max_subwords=MAX_SUBWORDS, stride=WINDOW_STRIDE_SUBWORDS)

    def _batch_extract(self, window_texts: list[str]) -> list[Mapping[str, Any]]:
        if not window_texts:
            return []
        model = self._model
        if model is None:
            msg = "Gliner2Adapter.load() must be called before predict()"
            raise RuntimeError(msg)
        schema = {"entities": list(GLINER2_SOURCE_LABELS)}
        collected: list[Mapping[str, Any]] = []
        for start in range(0, len(window_texts), MAX_WINDOWS_PER_CALL):
            chunk = window_texts[start : start + MAX_WINDOWS_PER_CALL]
            results: Any = model.batch_extract(
                chunk,
                schema,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                include_spans=True,
                max_len=GLINER2_MAX_LEN,
            )
            collected.extend(result for result in results if isinstance(result, Mapping))
        return collected
