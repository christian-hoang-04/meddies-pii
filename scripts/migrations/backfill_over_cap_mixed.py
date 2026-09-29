#!/usr/bin/env python
"""Remove rows over the 8192-token cap from ``meddies-pii-mixed`` and backfill.

The mixed corpus has 443 rows whose ``token_count_lfm25`` exceeds 8192 (the training
cap) — the trainer already skips these, so they are dead weight. This replaces the 405
that are in the 15 "other" languages with fresh same-language, same-domain rows drawn
from the ORIGINAL sources (via the build's own ``iter_records``, so backfill rows are
identical in schema/provenance), keeping only candidates that are (a) ≤ 8192 tokens and
(b) not already in mixed. The 38 English/Vietnamese over-cap rows CANNOT be backfilled —
those pools were fully consumed by the build — so they are dropped without replacement.

Result: 999,962 rows, max token_count ≤ 8192, per-"other"-language counts unchanged,
English/Vietnamese reduced by 29/9. Deterministic (SEED). Dry-run by default; ``--push``
republishes ``Meddies/meddies-pii-mixed`` in place.
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import hashlib
import random
from typing import TYPE_CHECKING, Protocol, cast

# reason: transformers declares its public names only under `TYPE_CHECKING` and serves them at
# reason: runtime through `_LazyModule`, so a static reader cannot prove the symbol is present.
# reason: Verified against the pinned 5.14.1: `hasattr(transformers, "AutoTokenizer")` is True.
import build_mixed_50_20_30 as build_mixed
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer  # ty: ignore[possibly-missing-import]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO = "Meddies/meddies-pii-mixed"
TOKENIZER_ID = "LiquidAI/LFM2.5-230M-Base"
COLUMN = "token_count_lfm25"
CAP = 8192
SEED = 42
NEED = {"laos": 204, "thai": 136, "burmese": 42, "tamil": 22, "japanese": 1}
"""per-"other"-language over-cap counts to backfill (measured live)."""


class Tokenizer(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool) -> Mapping[str, Sequence[int]]: ...


def sh(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def token_len(tok: Tokenizer, text: str) -> int:
    return len(tok(text, add_special_tokens=True)["input_ids"])


def collect_candidates(lang: str, mixed_hashes: set[str], tok: Tokenizer) -> list[dict[str, object]]:
    """Same-language rows from the sources, ≤ CAP tokens, not already in mixed."""
    # reason: the backfill must draw from the SAME source registry the original build used, or the
    # reason: replacement rows differ in provenance from the ones they replace. Copying the list here
    # reason: would let the two drift silently, which is the failure this migration exists to avoid.
    sources = [s for s in build_mixed._sources() if s[3] == lang and s[0] in {build_mixed.V1, build_mixed.V2}]  # ruff: ignore[private-member-access]
    out: list[dict[str, object]] = []
    seen_local: set[str] = set()
    for src in sources:
        for rec, _b, _dom, _l in build_mixed.iter_records(*src):
            h = sh(rec["text"])
            if h in mixed_hashes or h in seen_local:
                continue
            n = token_len(tok, rec["text"])
            if n > CAP:
                continue
            seen_local.add(h)
            row = {
                "text": rec["text"],
                "label": rec["label"],
                "info": rec["info"],
                COLUMN: n,
            }
            out.append(row)
    return out


def _require_cap_contract(
    rebuilt: Dataset,
    counts: Sequence[int],
    *,
    over_remaining: int,
    expected_over: int,
) -> None:
    """Refuse to continue unless the rebuilt corpus still meets this module's stated guarantee.

    These two checks were `assert` statements, which `python -O` removes — that would let the
    migration publish a corpus violating the row count and the language contract it exists to
    enforce, with no signal at all.

    Raises:
        RuntimeError: If over-cap rows remain in an unexpected number, or if any row still over
            the cap is neither English nor Vietnamese.

    """
    from collections import Counter

    if over_remaining != expected_over:
        msg = f"over-cap rows remaining {over_remaining} != expected {expected_over}"
        raise RuntimeError(msg)
    if not over_remaining:
        return
    bad = [
        r["language"]
        for r, c in zip(rebuilt["info"], counts, strict=False)
        if c > CAP and r["language"] not in {"english", "vietnamese"}
    ]
    if bad:
        msg = f"non-en/vi rows still over cap: {Counter(bad)}"
        raise RuntimeError(msg)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--push", action="store_true", help=f"Republish {REPO} in place (else dry-run).")
    p.add_argument(
        "--save-local",
        metavar="PATH",
        default=None,
        help="Save the rebuilt dataset to disk (no upload).",
    )
    p.add_argument(
        "--drop-envi",
        action="store_true",
        help="Also drop the 38 over-cap English/Vietnamese rows (result 999,962). "
        "Default keeps them so the total stays exactly 1,000,000.",
    )
    return p.parse_args(argv)


# reason: Dataset scan, cap backfill, and writes share one sampler; splitting would desync RNG and counts.
def main(argv: Sequence[str] | None = None) -> int:  # ruff: ignore[too-many-locals,too-many-statements]
    """Over-cap rows split by language.

    The "other" languages get removed + backfilled; English/Vietnamese over-cap have NO spare pool, so (unless --drop-envi)
    they are KEPT to preserve the exact 1,000,000 count (they stay > CAP; the trainer skips them).

    every over-cap row that remains must be English or Vietnamese.

    """
    args = parse_args(argv)
    # reason: the fixed SEED is the contract — this migration must select the same rows on a re-run,
    # reason: or the 1,000,000-row count the docstring guarantees would differ between invocations.
    rng = random.Random(SEED)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    # reason: from_pretrained is declared wide enough to include None; this file already names the
    # reason: one call shape it needs as Tokenizer, so the load is narrowed to that protocol.
    tok = cast("Tokenizer", AutoTokenizer.from_pretrained(TOKENIZER_ID))

    print(f"loading {REPO} ...", flush=True)
    ds = load_dataset(REPO, "default", split="train")
    langs_all = [r["language"] for r in ds["info"]]
    over_other: list[int] = []
    over_envi: list[int] = []
    for i, n in enumerate(ds[COLUMN]):
        if n <= CAP:
            continue
        (over_other if langs_all[i] in NEED else over_envi).append(i)
    remove = over_other + (over_envi if args.drop_envi else [])
    remove_set = set(remove)
    keep_idx = [i for i in range(len(ds)) if i not in remove_set]
    print(
        f"rows={len(ds):,}  over_cap={len(over_other) + len(over_envi)}  "
        f"remove={len(remove)} (other {len(over_other)} + en/vi "
        f"{len(over_envi) if args.drop_envi else 0})  keep={len(keep_idx):,}",
    )
    if not args.drop_envi:
        print(f"  keeping {len(over_envi)} over-cap English/Vietnamese rows (no spare pool)")
    mixed_keep = ds.select(keep_idx)

    mixed_hashes = {sh(t) for t in ds["text"]}

    backfill: list[dict[str, object]] = []
    print(f"\n{'lang':10}{'need':>6}{'candidates':>12}{'taken':>8}")
    for lang, need in NEED.items():
        cands = collect_candidates(lang, mixed_hashes, tok)
        rng.shuffle(cands)
        take = cands[:need]
        backfill.extend(take)
        print(f"{lang:10}{need:>6}{len(cands):>12,}{len(take):>8}")
        if len(take) < need:
            msg = f"insufficient candidates for {lang}: {len(take)} < {need}"
            raise SystemExit(msg)

    new_rows = Dataset.from_list(backfill).cast(mixed_keep.features)
    rebuilt = concatenate_datasets([mixed_keep, new_rows])

    counts = rebuilt[COLUMN]
    max_tok = max(counts)
    over_remaining = sum(1 for c in counts if c > CAP)
    expected_over = 0 if args.drop_envi else len(over_envi)
    print("\n=== VERIFY ===")
    print(f"  total rows: {len(rebuilt):,}  (expected {len(keep_idx) + len(backfill):,})")
    print(f"  max token_count: {max_tok}")
    print(
        f"  over-cap remaining: {over_remaining}  (expected {expected_over} {'kept en/vi' if not args.drop_envi else ''})",
    )
    from collections import Counter

    langs = Counter(r["language"] for r in rebuilt["info"])
    for lang in ("english", "vietnamese", *NEED):
        print(f"    {lang:12} {langs[lang]:,}")
    _require_cap_contract(rebuilt, counts, over_remaining=over_remaining, expected_over=expected_over)

    if args.save_local:
        rebuilt.save_to_disk(args.save_local)
        print(f"\nsaved locally to {args.save_local} (no upload)")
        return 0
    if not args.push:
        print("\nDRY RUN — re-run with --push to republish, or --save-local PATH. Nothing sent.")
        return 0

    envi_note = "en/vi over-cap dropped" if args.drop_envi else f"{len(over_envi)} en/vi over-cap kept (no spare)"
    msg = (
        f"data: remove+backfill {len(over_other)} over-{CAP} 'other'-language rows "
        f"with same-language ≤{CAP} rows ({envi_note})"
    )
    print(f"\npushing {len(rebuilt):,} rows to {REPO} ...", flush=True)
    rebuilt.push_to_hub(REPO, split="train", commit_message=msg)
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
