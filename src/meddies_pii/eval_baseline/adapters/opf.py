from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from meddies_pii.annotations.source_mapping import normalize_native_label
from meddies_pii.eval_baseline.adapters.opf_backend import (
    NativePredictFn,
    build_native_batched_predictor,
)
from meddies_pii.eval_baseline.baseline.span_cleanup import clean_spans
from meddies_pii.spans import CharSpan

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from meddies_pii.taxonomy import PiiLabel

OPF_LABEL_FOLD: Mapping[str, PiiLabel] = {
    "account_number": "id_number",
    "private_address": "address",
    "private_date": "date",
    "private_email": "email_address",
    "private_person": "human_name",
    "private_phone": "phone_number",
    "private_url": "private_url",
    "secret": "secret",
}
OPF_SUPPORTED_LABELS: frozenset[PiiLabel] = frozenset(OPF_LABEL_FOLD.values())


@dataclass(frozen=True, slots=True)
class _NativeConfig:
    batch_size: int
    triton: str
    compile_mode: str


@dataclass(frozen=True, slots=True)
class _OpfDoc:
    doc_id: str
    text: str


def map_opf_label(label: str) -> PiiLabel | None:
    return OPF_LABEL_FOLD.get(normalize_native_label(label))


def spans_from_native_spans(
    text: str,
    spans: Sequence[CharSpan],
) -> list[CharSpan]:
    mapped_spans: list[CharSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for span in spans:
        label = map_opf_label(span.label)
        if label is None:
            continue
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            continue
        key = (span.start, span.end, label)
        if key in seen:
            continue
        seen.add(key)
        mapped_spans.append(
            CharSpan(
                start=span.start,
                end=span.end,
                text=text[span.start : span.end],
                label=label,
            ),
        )
    return clean_spans(text, mapped_spans)


class OpfAdapter:
    name = "opf"
    # reason: PiiAdapter declares supported_labels as a mutable attribute, so the protocol member is
    # reason: INVARIANT and must accept a write of the full frozenset[PiiLabel]. Without this
    # reason: annotation the attribute infers the narrower set this adapter happens to fold to, and
    # reason: the adapter stops satisfying the protocol it is passed as.
    supported_labels: frozenset[PiiLabel] = OPF_SUPPORTED_LABELS

    def __init__(
        self,
        *,
        batch_size: int = 32,
        max_docs_per_call: int = 32,
        triton: str = "on",
        compile_mode: str = "none",
        predictor: NativePredictFn | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_docs_per_call = max(1, max_docs_per_call)
        self.triton = triton
        self.compile_mode = compile_mode
        self._predictor = predictor

    def load(self) -> None:
        if self._predictor is not None:
            return
        self._predictor = build_native_batched_predictor(
            _NativeConfig(
                batch_size=self.batch_size,
                triton=self.triton,
                compile_mode=self.compile_mode,
            ),
        )

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        predictor = self._predictor
        if predictor is None:
            msg = "OpfAdapter.load() must be called before predict()"
            raise RuntimeError(msg)

        docs = [_OpfDoc(doc_id=str(index), text=text) for index, text in enumerate(texts)]
        raw_by_doc: dict[str, list[CharSpan]] = {}
        for batch in _chunks(docs, self.max_docs_per_call):
            raw_by_doc.update(predictor(batch))
        return [spans_from_native_spans(text, raw_by_doc.get(str(index), [])) for index, text in enumerate(texts)]


def _chunks(docs: Sequence[_OpfDoc], size: int) -> list[Sequence[_OpfDoc]]:
    return [docs[start : start + size] for start in range(0, len(docs), max(1, size))]
