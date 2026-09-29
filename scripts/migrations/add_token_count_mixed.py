#!/usr/bin/env python
"""Add a per-row ``token_count_lfm25`` column to ``Meddies/meddies-pii-mixed``.

Every row gets the length (in LFM2.5 tokens) of its ``text`` field, so the corpus
can be filtered/sorted by length and the training truncation cap can be chosen from
data rather than guesswork.

Tokenizer: ``LiquidAI/LFM2.5-230M-Base``. The 230M and 350M LFM2.5 tokenizers are
byte-identical on real text (they differ only by two unused ``<think>``/``</think>``
special tokens and no id remaps), so the count is valid for BOTH models — hence the
family-scoped name ``token_count_lfm25`` rather than a per-model one. The tokenizer
loads WITHOUT ``trust_remote_code``.

Count = ``len(tokenizer(text).input_ids)`` with ``add_special_tokens=True`` — the plain
budget number, no spans, no BIOES alignment. This is deliberately NOT the manifest
builder (which judges + drops rows); here every row is counted, including ones with no
usable spans.

Provenance: the resolved tokenizer commit SHA is pinned into the push commit message so
the column is reproducible. Dry-run by default: tokenizes a small sample and prints
stats, sends NOTHING. ``--push`` maps all rows and republishes the dataset in place.
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from typing import TYPE_CHECKING

from datasets import load_dataset
from huggingface_hub import HfApi

# reason: transformers declares its public names only under TYPE_CHECKING and serves them at runtime through
# reason: _LazyModule, so no static reader can prove the symbol is present; it resolves at import time here.
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # ty: ignore[possibly-missing-import]

from meddies_pii.json_types import is_str_list

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO = "Meddies/meddies-pii-mixed"
CONFIG = "default"
SPLIT = "train"
TOKENIZER_ID = "LiquidAI/LFM2.5-230M-Base"
COLUMN = "token_count_lfm25"
BATCH = 512


def resolve_revision(revision: str | None) -> str:
    """Resolve the tokenizer repo to an immutable commit SHA for reproducibility."""
    sha = getattr(HfApi().model_info(TOKENIZER_ID, revision=revision), "sha", None)
    if not sha:
        msg = f"could not resolve a commit SHA for {TOKENIZER_ID!r}"
        raise RuntimeError(msg)
    return str(sha)


def load_tokenizer(revision: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=revision)
    if not isinstance(tok, PreTrainedTokenizerBase):
        msg = f"transformers returned no tokenizer for {TOKENIZER_ID!r} at revision {revision!r}"
        # reason: this reports a violated Transformers factory contract, not an invalid argument supplied by our caller.
        raise RuntimeError(msg)  # ruff: ignore[type-check-without-type-error]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def count_batch(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]) -> list[int]:
    encoded = tokenizer([t or "" for t in texts], add_special_tokens=True)["input_ids"]
    return [len(ids) for ids in encoded]


def print_stats(counts: list[int], cap: int = 8192) -> None:
    counts = sorted(counts)
    n = len(counts)

    def pct(p: float) -> float:
        k = (n - 1) * p / 100
        f = int(k)
        c = min(f + 1, n - 1)
        return counts[f] + (counts[c] - counts[f]) * (k - f)

    over = sum(1 for c in counts if c > cap)
    print(f"  n={n:,}  min={counts[0]}  p50={pct(50):.0f}  p95={pct(95):.0f}  p99={pct(99):.0f}  max={counts[-1]}")
    print(f"  over {cap}: {over:,}/{n:,} = {100 * over / n:.3f}%")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--push",
        action="store_true",
        help=f"Map ALL rows and republish {REPO} in place (else dry-run).",
    )
    p.add_argument(
        "--save-local",
        metavar="PATH",
        default=None,
        help="Map ALL rows and save the augmented dataset to PATH on disk. "
        "Counts everything, uploads NOTHING. Ignored when --push is set.",
    )
    p.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Pin the tokenizer to this ref/SHA (default: resolve current main).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=3000,
        help="Dry-run: how many rows to tokenize for the stats preview.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    revision = resolve_revision(args.tokenizer_revision)
    print(f"tokenizer {TOKENIZER_ID} @ {revision}")
    tokenizer = load_tokenizer(revision)

    print(f"loading {REPO} [{CONFIG}/{SPLIT}] ...", flush=True)
    ds = load_dataset(REPO, CONFIG, split=SPLIT)
    if COLUMN in ds.column_names:
        print(f"WARNING: column {COLUMN!r} already present — it will be overwritten.")

    def add_counts(batch: Mapping[str, list[object]]) -> dict[str, list[int]]:
        texts = batch["text"]
        if not is_str_list(texts):
            msg = "the 'text' column must hold strings"
            raise TypeError(msg)
        return {COLUMN: count_batch(tokenizer, texts)}

    if args.save_local and not args.push:
        print(
            f"tokenizing + adding {COLUMN!r} to all {len(ds):,} rows (LOCAL, no upload) ...",
            flush=True,
        )
        ds = ds.map(add_counts, batched=True, batch_size=BATCH, desc="token_count")
        print_stats(ds[COLUMN])
        print(f"\nsaving augmented dataset to {args.save_local} ...", flush=True)
        ds.save_to_disk(args.save_local)
        print("DONE (local only). Nothing uploaded.")
        return 0

    if not args.push:
        k = min(args.sample, len(ds))
        sample = ds.select(range(k))
        counts: list[int] = []
        texts = sample["text"]
        for i in range(0, len(texts), BATCH):
            counts.extend(count_batch(tokenizer, texts[i : i + BATCH]))
        print(f"\nDRY RUN — tokenized first {k:,} of {len(ds):,} rows:")
        print_stats(counts)
        print(f"  example {COLUMN} values: {counts[:10]}")
        print(f"\nColumn NOT added, nothing pushed. Re-run with --push to map all {len(ds):,} rows and republish.")
        return 0

    print(f"tokenizing + adding {COLUMN!r} to all {len(ds):,} rows ...", flush=True)
    ds = ds.map(add_counts, batched=True, batch_size=BATCH, desc="token_count")
    print_stats(ds[COLUMN])

    message = f"data: add {COLUMN} column (tokenizer {TOKENIZER_ID} @ {revision}, add_special_tokens=True)"
    print(f"\npushing to {REPO} [{SPLIT}] ...", flush=True)
    ds.push_to_hub(REPO, split=SPLIT, commit_message=message)
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
