#!/usr/bin/env python
"""Assert the chosen HF eval configs are text-disjoint from ``meddies-pii-mixed`` train.

Phase-2 leakage gate for the LFM2.5-230M fine-tune. ``meddies-pii-mixed`` has only a
``train`` split, so eval must be borrowed from held-out data — here ``meddies-pii-v2``
``eval`` (primary) and ``eval-challenge`` (adversarial stress). Both come from the same
generated corpus pool that mixed pooled its v2 *train* from, so physical
separation does NOT guarantee content separation: a row can be byte-identical across two
repos. This proves it isn't.

The identity key is ``sha256(text)`` with no normalization — reused verbatim from
``scripts/ops/check_eval_disjoint.py`` (the "brief leakage rule") so this check agrees
with the generation pipeline's own dedup. Read-only: loads eval hashes (small), streams
mixed train, reports any collision. Exit 1 on overlap (gate).
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from datasets import load_dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

COLLISION_PREVIEW_LIMIT = 12

REPO = Path(__file__).resolve().parents[2]
MIXED = "Meddies/meddies-pii-mixed"
V2 = "Meddies/meddies-pii-v2"
EXT = "Meddies/meddies-pii-external"
AI4_LANGS = ("de", "en", "es", "fil", "fr", "id", "ja", "ko", "ms", "pt", "vi", "zh")

EVAL_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (V2, "eval", "train", "v2/eval"),
    (V2, "eval-challenge", "train", "v2/eval-challenge"),
    *((EXT, f"ai4privacy_{code}", "eval", f"ai4privacy_{code}") for code in AI4_LANGS),
    (EXT, "gretel_en", "eval", "gretel_en"),
    (EXT, "creddata_en", "eval", "creddata_en"),
    (EXT, "nemotron_en", "eval", "nemotron_en"),
)
"""Full eval suite (repo, config, split, label).

v2 eval configs store their rows under a 'train' split; external eval splits are named 'eval'. Each is kept ISOLATED —
checked per-source against mixed train so we see exactly which source (if any) leaks.

"""
PROGRESS_EVERY = 250_000


def _load_text_hash() -> Callable[[str], str]:
    """Import ``text_hash`` from the sibling gate so the hash definition is identical.

    Raises:
        ImportError: when the sibling gate script cannot be located or executed.

    """
    path = REPO / "scripts/ops/check_eval_disjoint.py"
    spec = importlib.util.spec_from_file_location("check_eval_disjoint", path)
    if spec is None or spec.loader is None:
        msg = f"cannot import the sibling gate from {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text_hash_candidate: object = getattr(module, "text_hash", None)
    if not callable(text_hash_candidate):
        msg = "sibling gate does not expose a callable text_hash"
        # reason: the sibling module imported but lacks its required capability, so this remains an import-contract error.
        raise ImportError(msg)  # ruff: ignore[type-check-without-type-error]
    hash_function = cast("Callable[[str], object]", text_hash_candidate)

    def checked_text_hash(text: str) -> str:
        digest = hash_function(text)
        if not isinstance(digest, str):
            msg = "sibling text_hash returned a non-string digest"
            raise TypeError(msg)
        return digest

    return checked_text_hash


text_hash = _load_text_hash()


def load_eval_hashes(repo: str, config: str, split: str) -> dict[str, dict[str, Any]]:
    ds = load_dataset(repo, config, split=split)
    by_hash: dict[str, dict[str, Any]] = {}
    for record in ds:
        text = record.get("text")
        if isinstance(text, str):
            info = record.get("info") or {}
            by_hash[text_hash(text)] = {
                "lang": info.get("language"),
                "id": info.get("id"),
                "preview": text[:90].replace("\n", " "),
            }
    return by_hash


def scan_mixed(gold: dict[str, dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    ds = load_dataset(MIXED, "default", split="train")
    scanned = 0
    collisions: list[dict[str, Any]] = []
    for text in ds["text"]:
        if not isinstance(text, str):
            continue
        scanned += 1
        if scanned % PROGRESS_EVERY == 0:
            print(f"  scanned {scanned:,} mixed rows ...", flush=True)
        hit = gold.get(text_hash(text))
        if hit is not None:
            collisions.append(hit)
    return scanned, collisions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Exit 1 if any eval row's text is found in mixed train (gate mode).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Tag which eval source each hash came from (last wins on cross-eval dup)."""
    args = parse_args(argv)

    all_hashes: dict[str, dict[str, Any]] = {}
    per_source: dict[str, int] = {}
    for repo, config, split, label in EVAL_SPECS:
        h = load_eval_hashes(repo, config, split)
        per_source[label] = len(h)
        print(f"{label}: {len(h):,} unique-text eval rows")
        for k, source_row in h.items():
            v = dict(source_row)
            v["eval_source"] = label
            all_hashes[k] = v

    print(f"\nscanning {MIXED} train against {len(all_hashes):,} eval hashes ...")
    scanned, collisions = scan_mixed(all_hashes)

    from collections import Counter

    hits_by_source: Counter[str] = Counter(source for c in collisions if isinstance(source := c.get("eval_source"), str))

    print("\n=== RESULT ===")
    print(f"  mixed train rows scanned: {scanned:,}")
    print(f"  eval sets checked: {len(EVAL_SPECS)}  (total {len(all_hashes):,} unique hashes)")
    print(f"  TOTAL COLLISIONS (eval text found in train): {len(collisions):,}")
    print("\n  per eval set  (rows checked -> collisions):")
    for label, n in per_source.items():
        hit = hits_by_source.get(label, 0)
        flag = "  <-- LEAK" if hit else ""
        print(f"    {label:22} {n:>7,} -> {hit:>6,}{flag}")
    if collisions:
        print("\n  sample collisions:")
        for c in collisions[:12]:
            print(f"    [{c.get('eval_source')}] {c.get('lang')} :: {c['preview']}")
        if len(collisions) > COLLISION_PREVIEW_LIMIT:
            print(f"    ... and {len(collisions) - 12:,} more")

    if collisions and args.fail_on_overlap:
        print("\nLEAKAGE DETECTED — one or more eval sets overlap mixed train.")
        return 1
    if not collisions:
        print("\nCLEAN — no eval text appears in mixed train. All eval sets usable as-is.")
    else:
        print("\nPartial overlap — leaked sets flagged above; the rest are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
