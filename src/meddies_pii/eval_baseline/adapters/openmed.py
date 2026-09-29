from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: adapters load their model stack inside `load()`, so listing an adapter costs nothing.
# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
from collections.abc import Mapping, Sequence
from itertools import starmap
from typing import Any, cast

from meddies_pii.annotations.source_mapping import map_native_label_to_pii_label
from meddies_pii.eval_baseline.baseline.span_cleanup import clean_spans
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PII_LABEL_SET, PiiLabel

MODEL_ID = "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1"
MODEL_REVISION = "df7af994d39d358e52f929ff1b3a40d894adf022"
MODEL_MAX_LENGTH = 512
"""DeBERTa-v3-large's backbone window.

The card ships model_max_length as the int64 "no-cap" sentinel (1e30), so the pipeline never chunks and a single long doc
(creddata_en had a ~7.5k-token outlier) blows up O(L^2) disentangled attention -> 176 GiB alloc -> OOM. Capping here makes
the pipeline's stride sliding-window split long docs into overlapping 512-token chunks and stitch spans back on the
original text -- bounded memory, no recall loss.

"""
OPENMED_SUPPORTED_LABELS: frozenset[PiiLabel] = PII_LABEL_SET
_OPENMED_OVERRIDES: dict[str, PiiLabel] = {
    "url": "private_url",
    "cvv": "secret",
}


def map_openmed_label(label: str) -> PiiLabel | None:
    return map_native_label_to_pii_label(label, overrides=_OPENMED_OVERRIDES)


def spans_from_pipeline_entities(
    text: str,
    entities: Sequence[Mapping[str, Any]],
) -> list[CharSpan]:
    """Repair subword fragmentation + boundary junk (clean_spans re-sorts + merges).

    Returns:
        The pipeline's entities as cleaned character spans. An entity whose label is missing,
        not a string, or outside the supported set is skipped rather than mapped to a default,
        so an unmapped vendor label never scores as a wrong prediction of a real one.
        Duplicates by start, end and label are dropped before cleaning.

    """
    spans: list[CharSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in entities:
        raw_label = entity.get("entity_group") or entity.get("entity")
        if not isinstance(raw_label, str):
            continue
        label = map_openmed_label(raw_label)
        if label is None or label not in OPENMED_SUPPORTED_LABELS:
            continue
        start = entity.get("start")
        end = entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        spans.append(CharSpan(start=start, end=end, text=text[start:end], label=label))
    return clean_spans(text, spans)


class OpenMedAdapter:
    name = "openmed"
    supported_labels = OPENMED_SUPPORTED_LABELS

    def __init__(self, *, batch_size: int = 32, stride: int = 64) -> None:
        self.batch_size = batch_size
        self.stride = stride
        self._pipe: Any | None = None

    def load(self) -> None:
        """Enable the pipeline's stride sliding-window.

        A finite max length is what turns `stride` into chunk-and-stitch (return_overflowing_tokens) instead of a no-op.
        Without it, long docs OOM (see MODEL_MAX_LENGTH).

        Eager on GPU. torch.compile(mode="reduce-overhead") uses CUDA graphs that re-capture per input shape -> 60GB+ graph
        pools -> OOM on the variable-length batched eval (32-row smoke hid it). Encoder NER is fast enough eager;
        correctness over a modest compile win.

        Model-card-recommended strategy (the model was trained label_all_tokens=False, so non-first subword preds are noise
        the card expects to be post-processed). With clean_spans it gives the cleanest boundaries — beats
        "average"/"first", which drop or over-extend spans (verified on the card's own example).

        """
        if self._pipe is not None:
            return
        import torch

        # reason: transformers declares its public names only under `TYPE_CHECKING` and serves
        # reason: them at runtime through `_LazyModule`, so a static reader cannot prove the
        # reason: symbol is present. Verified against the pinned 5.14.1 in this environment:
        # reason: `hasattr(transformers, "AutoTokenizer")` is True.
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,  # ty: ignore[possibly-missing-import]
            pipeline,
        )

        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
        # reason: `AutoTokenizer.from_pretrained` is typed as a union of backend classes plus
        # reason: None, and none of them declares `model_max_length`, though every concrete
        # reason: tokenizer carries it. Narrowing the union here would mean naming private
        # reason: transformers backend classes, which is a worse dependency than this line.
        tokenizer.model_max_length = MODEL_MAX_LENGTH  # ty: ignore[invalid-assignment]
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
        )
        model_for_pipeline: Any = model
        if torch.cuda.is_available():
            model_for_pipeline = model.to("cuda")
        token_pipeline: Any = pipeline
        self._pipe = token_pipeline(
            "token-classification",
            model=model_for_pipeline,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            batch_size=self.batch_size,
            device=device,
        )

    def predict(self, texts: list[str]) -> list[list[CharSpan]]:
        """`stride` drives the pipeline's own overflow/sliding-window handling for docs longer than the model.

        `stride` drives the pipeline's own overflow/sliding-window handling for docs longer than the model max length;
        `truncation` is NOT a valid token-classification pipeline call kwarg (the tokenizer manages it).

        Returns:
            One cleaned span list per input text, in input order. The list is always the same
            length as ``texts``, so a caller can zip the two without re-checking.

        Raises:
            RuntimeError: If ``load()`` was not called first, or if the pipeline returns a
                different number of outputs than texts given. That count is checked rather than
                assumed, because a silent misalignment would attribute every prediction to the
                wrong document and still produce a plausible-looking score.

        """
        if self._pipe is None:
            msg = "OpenMedAdapter.load() must be called before predict()"
            raise RuntimeError(msg)
        raw = self._pipe(
            texts,
            stride=self.stride,
        )
        grouped = _as_batched_entities(raw)
        if len(grouped) != len(texts):
            msg = f"openmed pipeline returned {len(grouped)} outputs for {len(texts)} texts"
            raise RuntimeError(msg)
        return list(starmap(spans_from_pipeline_entities, zip(texts, grouped, strict=True)))


def _as_batched_entities(raw: object) -> list[list[Mapping[str, Any]]]:
    if not isinstance(raw, list):
        msg = f"unexpected openmed pipeline output: {type(raw).__name__}"
        raise RuntimeError(msg)
    if not raw:
        return []
    first = raw[0]
    if isinstance(first, Mapping):
        return [list(_only_mappings(raw))]
    return [list(_only_mappings(batch)) if isinstance(batch, list) else [] for batch in raw]


def _only_mappings(values: Sequence[object]) -> list[Mapping[str, Any]]:
    return [cast("Mapping[str, Any]", value) for value in values if isinstance(value, Mapping)]
