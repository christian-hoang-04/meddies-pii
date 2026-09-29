from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from meddies_pii.eval_baseline.baseline.datasets import EvalRow
    from meddies_pii.spans import CharSpan

LengthBucket = str
SHORT_BUCKET_MAX_TOKENS = 128
MEDIUM_BUCKET_MAX_TOKENS = 384

LENGTH_BUCKETS: tuple[LengthBucket, ...] = ("short", "medium", "long")


@dataclass(frozen=True, slots=True)
class BenchmarkDoc:
    doc_id: str
    text: str
    gold_spans: tuple[CharSpan, ...]
    token_count: int
    length_bucket: LengthBucket
    text_sha256: str


@dataclass(frozen=True, slots=True)
class PinnedBenchmarkDocSet:
    name: str
    docs: tuple[BenchmarkDoc, ...]
    sha256: str
    bucket_counts: dict[str, int]

    def docs_for_bucket(self, bucket: str) -> tuple[BenchmarkDoc, ...]:
        return tuple(doc for doc in self.docs if doc.length_bucket == bucket)


def approximate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def bucket_for_token_count(token_count: int) -> LengthBucket:
    if token_count <= SHORT_BUCKET_MAX_TOKENS:
        return "short"
    if token_count <= MEDIUM_BUCKET_MAX_TOKENS:
        return "medium"
    return "long"


def build_pinned_benchmark_doc_set(
    rows: Sequence[EvalRow],
    *,
    per_bucket: int = 32,
    token_counter: Callable[[str], int] = approximate_token_count,
    name: str = "opf-varied-length",
) -> PinnedBenchmarkDocSet:
    if per_bucket <= 0:
        msg = "per_bucket must be positive"
        raise ValueError(msg)

    buckets: dict[str, list[BenchmarkDoc]] = {bucket: [] for bucket in LENGTH_BUCKETS}
    for row in rows:
        token_count = int(token_counter(row.text))
        bucket = bucket_for_token_count(token_count)
        buckets[bucket].append(
            BenchmarkDoc(
                doc_id=row.stable_id,
                text=row.text,
                gold_spans=tuple(row.gold_spans),
                token_count=token_count,
                length_bucket=bucket,
                text_sha256=row.text_sha256,
            ),
        )
    missing = [bucket for bucket, docs in buckets.items() if not docs]
    if missing:
        msg = f"benchmark doc set missing length buckets: {missing}"
        raise ValueError(msg)

    selected: list[BenchmarkDoc] = []
    for bucket in LENGTH_BUCKETS:
        selected.extend(sorted(buckets[bucket], key=lambda doc: doc.doc_id)[:per_bucket])

    counts = Counter(doc.length_bucket for doc in selected)
    return PinnedBenchmarkDocSet(
        name=name,
        docs=tuple(selected),
        sha256=benchmark_doc_set_sha256(selected),
        bucket_counts={bucket: counts.get(bucket, 0) for bucket in LENGTH_BUCKETS},
    )


def benchmark_doc_set_sha256(docs: Sequence[BenchmarkDoc]) -> str:
    payload = {
        "version": 1,
        "docs": [
            {
                "doc_id": doc.doc_id,
                "text_sha256": doc.text_sha256,
                "token_count": doc.token_count,
                "length_bucket": doc.length_bucket,
            }
            for doc in docs
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
