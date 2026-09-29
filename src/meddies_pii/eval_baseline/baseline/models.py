from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meddies_pii.spans import CharSpan


@dataclass(frozen=True, slots=True)
class EvalRow:
    doc_id: str
    dataset: str
    shard: str
    text: str
    gold_spans: tuple[CharSpan, ...]
    language: str
    slices: frozenset[str] = frozenset()

    @property
    def stable_id(self) -> str:
        return f"{self.dataset}:{self.shard}:{self.doc_id}"

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
