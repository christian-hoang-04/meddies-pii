"""Train/eval split primitives for publishing the synthetic corpus.

Pure functions, no I/O — the HF read/push orchestration lives in
``scripts/ops/publish_corpus_to_v2.py``. Kept separate so the safety-critical
logic (dedup, leak-free holdout) is unit-tested without touching the network.
"""

from __future__ import annotations

import copy
import hashlib
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

Row = dict[str, Any]


def text_hash(text: str) -> str:
    """Return the sha256 of the exact ``text``.

    This is the same key the generation pipeline
    (`synthetic.py` dedup) and `check_eval_disjoint.py` use, so dedup and
    leakage checks agree across the codebase.

    Returns:
        The hex SHA-256 of the exact text, byte for byte with no normalization. Exactness is
        the contract: this is a dedup and leakage KEY, so two texts differing by whitespace
        are deliberately different rows.

    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_rows(rows: Iterable[Row]) -> list[Row]:
    """Keep the first row per unique exact ``text``, dropping later duplicates.

    Rows whose ``text`` is not a string can't be a training example and are
    dropped. First-wins matters at the call site: existing HF rows are passed
    before the local corpus, so a republish never replaces a curated row with
    a same-text local one.

    Returns:
        The rows in input order with later exact-text duplicates removed, and every row whose
        text is not a string dropped. First-wins is load-bearing rather than incidental --
        the call site passes curated rows first so a republish cannot replace one.

    """
    seen: set[str] = set()
    out: list[Row] = []
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            continue
        digest = text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(row)
    return out


def drop_held_out(rows: Iterable[Row], held_out_hashes: set[str]) -> list[Row]:
    """Drop every row whose exact ``text`` is in ``held_out_hashes``.

    The republish leak guard: the local corpus is a superset that re-introduces
    the rows carved into the ``eval`` holdout and the curated ``eval-challenge``
    gold, so the refreshed train must subtract every held-out ``text_hash`` to
    stay disjoint from both. An empty set is the identity. Rows whose ``text`` is
    not a string are dropped, matching ``dedup_rows`` — a non-string row can't be
    a training example, and this keeps the two filters consistent for any caller.

    Returns:
        The rows whose text hash is not in the held-out set, in input order. An empty set
        returns everything, so the guard is safe to call unconditionally.

    """
    return [row for row in rows if isinstance(row.get("text"), str) and text_hash(row["text"]) not in held_out_hashes]


def holdout_split(rows: list[Row], n: int, *, seed: int) -> tuple[list[Row], list[Row]]:
    """Carve ``n`` rows out of ``rows`` as an eval holdout.

    The holdout is disjoint from the
    returned train set by construction (the held rows are removed from train,
    not sampled independently).

    Reproducible regardless of input order: the rows are first put in a
    canonical order (by ``text_hash``) before the seeded shuffle, so the same
    row *set* + same ``seed`` always yields the same eval docs even if the union
    was assembled in a different order on a later republish. Held rows get
    ``info.split_purpose = "eval"``; train rows are returned unchanged.

    Returns:
        ``(train, held)``, disjoint by construction. The held rows are deep-copied before
        their ``split_purpose`` is set, so the caller's originals are not mutated; train rows
        are the same objects that came in.

    Raises:
        ValueError: If ``n`` is negative, or if it is not smaller than the row count -- a
            holdout that consumes everything leaves no training data, which is refused rather
            than returned as an empty train set.

    """
    if n < 0:
        msg = f"n must be >= 0, got {n}"
        raise ValueError(msg)
    if n >= len(rows):
        msg = f"holdout n={n} >= rows={len(rows)} leaves no training data"
        raise ValueError(msg)

    ordered = sorted(rows, key=lambda row: text_hash(row["text"]))
    indices = list(range(len(ordered)))
    # reason: the holdout split must be identical for anyone who publishes this corpus from the same seed, which
    # reason: is why the rows are hash-ordered first and then shuffled from an explicit seed. Reproducibility is
    # reason: the contract; this picks eval positions, never a secret, a token, or a key.
    random.Random(seed).shuffle(indices)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    eval_positions = set(indices[:n])

    train: list[Row] = []
    held: list[Row] = []
    for position, row in enumerate(ordered):
        if position in eval_positions:
            held_row = copy.deepcopy(row)
            info = held_row.get("info")
            if isinstance(info, dict):
                info["split_purpose"] = "eval"
            held.append(held_row)
        else:
            train.append(row)
    return train, held
