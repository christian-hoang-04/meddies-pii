#!/usr/bin/env python
"""Trim ai4privacy IN PLACE to a 30% vi / 30% en / 40% other language mix.

ai4privacy lives on ``Meddies/meddies-pii-external`` as 12 per-language configs
(``ai4privacy_<code>``). Vietnamese is the scarce anchor (~26k rows), so we keep
ALL of it and randomly delete rows from English and from the 10 "other" languages
until the whole ai4privacy set is:

    vi 30%   en 30%   other (10 languages combined) 40%

* ``ai4privacy_vi``  -> kept whole (unchanged, not re-pushed).
* ``ai4privacy_en``  -> randomly subsampled to the SAME count as vi (both 30%).
* the 10 ``ai4privacy_<other>`` -> randomly subsampled **proportionally** to a
  combined 40% (every language survives, just smaller).

The ratio is enforced **within each split** (train and eval separately) so eval
stays representative; train/eval were already disjoint, so removing rows keeps
leakage at zero. Deletion is random with a fixed seed -> reproducible.

Overwrites the existing per-language configs IN PLACE (they become permanently
smaller). Dry-run by default: prints the full per-config plan + resulting
percentages and sends NOTHING to the hub. Re-run with ``--push`` to publish.
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
import random
import time
from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, load_dataset
from httpx import HTTPError

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ID = "Meddies/meddies-pii-external"
SPLITS = ("train", "eval")
SEED = 42

VI = "ai4privacy_vi"
EN = "ai4privacy_en"
OTHER = (
    "ai4privacy_de",
    "ai4privacy_es",
    "ai4privacy_fil",
    "ai4privacy_fr",
    "ai4privacy_id",
    "ai4privacy_ja",
    "ai4privacy_ko",
    "ai4privacy_ms",
    "ai4privacy_pt",
    "ai4privacy_zh",
)
VI_SHARE, EN_SHARE, OTHER_SHARE = 0.30, 0.30, 0.40


def subsample(ds: Dataset, keep: int, seed: int) -> Dataset:
    """Deterministically keep ``keep`` random rows (features preserved via select)."""
    n = ds.num_rows
    if keep >= n:
        return ds
    # reason: the caller's `seed` is what makes this "Deterministically keep" rather than "keep" —
    # reason: the same seed must select the same rows so a rebalanced split can be rebuilt.
    idx = sorted(random.Random(seed).sample(range(n), keep))  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    selected: object = ds.select(idx)
    if not isinstance(selected, Dataset):
        msg = "datasets.select returned a non-Dataset value"
        raise TypeError(msg)
    return selected


def plan_split(split: str) -> tuple[dict[str, Dataset], dict[str, int]]:
    """Trim one split -> ({config: trimmed Dataset for en+other}, {config: kept count})."""
    v = load_dataset(REPO_ID, VI, split=split).num_rows
    en_target = round(v * EN_SHARE / VI_SHARE)
    other_target = round(v * OTHER_SHARE / VI_SHARE)

    trimmed: dict[str, Dataset] = {}
    counts: dict[str, int] = {VI: v}

    en = load_dataset(REPO_ID, EN, split=split)
    trimmed[EN] = subsample(en, en_target, SEED)
    counts[EN] = trimmed[EN].num_rows

    other_ds = {c: load_dataset(REPO_ID, c, split=split) for c in OTHER}
    total_other = sum(d.num_rows for d in other_ds.values())
    for i, c in enumerate(OTHER):
        keep = round(other_ds[c].num_rows * other_target / total_other)
        trimmed[c] = subsample(other_ds[c], keep, SEED + i + 1)
        counts[c] = trimmed[c].num_rows
    return trimmed, counts


def build_plan() -> dict[str, dict[str, Dataset]]:
    """{config: {split: trimmed Dataset}} for the configs that change (en + 10 other)."""
    per_config: dict[str, dict[str, Dataset]] = {}
    grand = {"vi": 0, "en": 0, "other": 0}

    print(f"\n{'config':<18}{'split':<7}{'kept':>10}")
    print("-" * 36)
    for split in SPLITS:
        trimmed, counts = plan_split(split)
        grand["vi"] += counts[VI]
        grand["en"] += counts[EN]
        grand["other"] += sum(counts[c] for c in OTHER)
        print(f"{VI:<18}{split:<7}{counts[VI]:>10,}  (unchanged)")
        for c in (EN, *OTHER):
            per_config.setdefault(c, {})[split] = trimmed[c]
            print(f"{c:<18}{split:<7}{counts[c]:>10,}")

    total = sum(grand.values())
    print("-" * 36)
    print(
        f"RESULT  vi={grand['vi']:,} ({100 * grand['vi'] / total:.1f}%)  "
        f"en={grand['en']:,} ({100 * grand['en'] / total:.1f}%)  "
        f"other={grand['other']:,} ({100 * grand['other'] / total:.1f}%)  "
        f"total={total:,}",
    )
    return per_config


def publish(per_config: dict[str, dict[str, Dataset]]) -> None:
    """Overwrite each changed config in place (both splits pushed atomically).

    Retries each config on transient hub errors (504s have been observed on this
    repo). Re-running the script after a partial failure is NOT idempotent, so we
    retry hard here and raise if a config still can't be pushed.
    """
    for cfg, splits in per_config.items():
        bundle = DatasetDict({s: splits[s] for s in SPLITS if s in splits})
        for attempt in range(1, 6):
            try:
                bundle.push_to_hub(REPO_ID, config_name=cfg)
                sizes = {s: bundle[s].num_rows for s in bundle}
                print(f"  overwrote {cfg}: {sizes}", flush=True)
                break
            except HTTPError as exc:
                print(f"  retry {cfg} (attempt {attempt}): {str(exc)[:70]}", flush=True)
                time.sleep(15 * attempt)
        else:
            msg = f"failed to push {cfg} after 5 attempts"
            raise RuntimeError(msg)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push",
        action="store_true",
        help="Overwrite the ai4privacy configs in place (en + 10 other). vi is untouched.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    per_config = build_plan()
    if not args.push:
        print("\nDRY RUN — re-run with --push to overwrite ai4privacy in place. Nothing sent.")
        return 0
    print(f"\nOverwriting {len(per_config)} configs in place ...")
    publish(per_config)
    print("\nDONE. ai4privacy is now 30% vi / 30% en / 40% other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
