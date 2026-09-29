#!/usr/bin/env python
"""Restructure ``Meddies/meddies-pii-external`` into per-language configs.

The #25 build lumped ai4privacy as ``en`` / ``vi`` / ``other`` (28 languages in
one bucket) and left gretel / nemotron / creddata as single English configs.
This repartitions every source by ``info.language`` into ``<source>_<code>``
configs using **canonical Meddies codes** (``tl`` -> ``fil`` etc.), gated to the
**17 supported languages** (non-17 rows dropped), ``train`` + ``eval`` each:

* ``ai4privacy_en`` / ``ai4privacy_vi`` — already correct, kept untouched.
* ``ai4privacy_other`` — split into ``ai4privacy_<code>`` for each supported
  language present; 18 non-Meddies European languages dropped.
* ``gretel`` / ``nemotron`` / ``creddata`` — all-English -> renamed ``<source>_en``.

The 4 superseded configs (``ai4privacy_other``, ``gretel``, ``nemotron``,
``creddata``) are then deleted from the repo and removed from the dataset card.

Dry-run by default (prints the full plan + per-language counts + dropped
languages + eval/train leakage). Re-run with ``--push`` to publish, and add
``--delete-old`` to also delete the superseded configs (only after a successful
push). Nothing is sent to the hub without ``--push``.
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
from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, load_dataset

from meddies_pii.languages import is_supported_meddies_language, normalize_language
from meddies_pii.publishing.corpus_split import text_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ID = "Meddies/meddies-pii-external"
SPLITS = ("train", "eval")

OTHER_CONFIG = "ai4privacy_other"
"""ai4privacy_other -> per-language ai4privacy_<code>; these two are already correct."""
KEEP_UNTOUCHED = ("ai4privacy_en", "ai4privacy_vi")
RENAME_TO_EN = ("gretel", "nemotron", "creddata")
"""all-English single configs -> <source>_en (a rename for naming uniformity)."""
OLD_TO_DELETE = ("ai4privacy_other", "gretel", "nemotron", "creddata")


def code_or_none(language: str) -> str | None:
    """Canonical Meddies code if the language is one of the 17, else None."""
    return normalize_language(language).code if is_supported_meddies_language(language) else None


def split_other_by_code() -> tuple[dict[str, dict[str, Dataset]], dict[str, dict[str, int]]]:
    """ai4privacy_other -> {code: {split: Dataset}} plus {split: {dropped_lang: n}}.

    Plans from the ``info`` column alone (one pass, no full-row materialization),
    then ``select`` builds each per-language subset lazily over arrow.
    """
    by_code: dict[str, dict[str, Dataset]] = {}
    dropped: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        ds = load_dataset(REPO_ID, OTHER_CONFIG, split=split)
        idx_by_code: dict[str, list[int]] = {}
        drop_counts: dict[str, int] = {}
        for i, info in enumerate(ds["info"]):
            language = str((info or {}).get("language") or "")
            code = code_or_none(language)
            if code is None:
                drop_counts[language] = drop_counts.get(language, 0) + 1
            else:
                idx_by_code.setdefault(code, []).append(i)
        dropped[split] = drop_counts
        for code, indices in idx_by_code.items():
            by_code.setdefault(code, {})[split] = ds.select(indices)
    return by_code, dropped


def gate_rename_source(config: str) -> dict[str, Dataset]:
    """Load a single all-English config, gated to supported languages (drops none here)."""
    out: dict[str, Dataset] = {}
    for split in SPLITS:
        ds = load_dataset(REPO_ID, config, split=split)
        keep_idx = [
            i for i, info in enumerate(ds["info"]) if code_or_none(str((info or {}).get("language") or "")) is not None
        ]
        out[split] = ds.select(keep_idx)
    return out


def leakage(train: Dataset | None, eval_: Dataset | None) -> int:
    """Eval rows whose exact text also appears in train (should already be 0)."""
    if train is None or eval_ is None:
        return 0
    train_hashes = {text_hash(t) for t in train["text"]}
    return sum(1 for t in eval_["text"] if text_hash(t) in train_hashes)


def build_plan() -> list[tuple[str, dict[str, Dataset]]]:
    """Ordered list of (new_config_name, {split: Dataset}) to publish."""
    plan: list[tuple[str, dict[str, Dataset]]] = []
    by_code, dropped = split_other_by_code()
    for code, splits in sorted(by_code.items()):
        plan.append((f"ai4privacy_{code}", splits))
    plan.extend((f"{source}_en", gate_rename_source(source)) for source in RENAME_TO_EN)

    print(f"\n{'new config':<22}{'train':>10}{'eval':>9}{'leak':>7}")
    print("-" * 48)
    for name, splits in plan:
        tr, ev = splits.get("train"), splits.get("eval")
        print(f"{name:<22}{(tr.num_rows if tr else 0):>10,}{(ev.num_rows if ev else 0):>9,}{leakage(tr, ev):>7}")
    print("-" * 48)
    print(f"{'KEPT untouched':<22}  {', '.join(KEEP_UNTOUCHED)}")
    for split in SPLITS:
        drop = dropped.get(split, {})
        if drop:
            print(f"DROPPED non-17 ({split}): {sum(drop.values()):,} rows across {len(drop)} languages -> {sorted(drop)}")
    return plan


def publish(plan: list[tuple[str, dict[str, Dataset]]]) -> None:
    """Push both splits atomically as one config.

    A per-split push_to_hub can rewrite the config's data_files and drop the other split).

    """
    for name, splits in plan:
        bundle = DatasetDict({s: ds for s in SPLITS if (ds := splits.get(s)) is not None and ds.num_rows})
        bundle.push_to_hub(REPO_ID, config_name=name)
        sizes = {s: bundle[s].num_rows for s in bundle}
        print(f"  pushed {name}: {sizes}")


def delete_old() -> None:
    """1.

    Prune the card FIRST (both `configs` and the auto-generated `dataset_info`) so no card entry ever points at a deleted
    folder. Load fresh so the new configs pushed above are preserved.

    """
    from huggingface_hub import DatasetCard, HfApi

    api = HfApi()
    card = DatasetCard.load(REPO_ID, repo_type="dataset")
    data = card.data.to_dict()
    card.data["configs"] = [c for c in data.get("configs", []) if c.get("config_name") not in OLD_TO_DELETE]
    info = data.get("dataset_info")
    if isinstance(info, list):
        card.data["dataset_info"] = [d for d in info if d.get("config_name") not in OLD_TO_DELETE]
    card.push_to_hub(REPO_ID, repo_type="dataset")
    remaining = sorted(c.get("config_name") for c in card.data.to_dict().get("configs", []))
    print(f"  card configs now: {remaining}")
    for cfg in OLD_TO_DELETE:
        api.delete_folder(
            path_in_repo=cfg,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"remove superseded config {cfg}",
        )
        print(f"  deleted folder {cfg}/")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Publish the new configs.")
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="After a successful --push, delete the 4 superseded configs + prune the card.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan()

    if not args.push:
        print("\nDRY RUN — re-run with --push to publish. Nothing sent to the hub.")
        return 0

    print(f"\nPushing {len(plan)} new configs ...")
    publish(plan)
    if args.delete_old:
        print("\nDeleting superseded configs ...")
        delete_old()
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
