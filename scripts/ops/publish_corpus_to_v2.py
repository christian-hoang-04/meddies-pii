#!/usr/bin/env python
"""Refresh Meddies/meddies-pii-v2 train configs from the local synthetic corpus.

For each of the 17 languages: ``train = dedup(union(existing HF config, local
**/accepted.{code}.jsonl))`` with every held-out row removed. "Held out" is the
union of the ``eval`` holdout and the ``eval-challenge`` adversarial gold, both
read live from the Hub. The local corpus is a superset that re-introduces those
rows on every republish, so subtracting them is what keeps train text-disjoint
from both eval sets.

Idempotent and gold-safe: ``eval`` and ``eval-challenge`` are READ ONLY (to learn
what to exclude) and never overwritten. Only the per-language train configs are
pushed. (The one-time ``eval -> eval-challenge`` rename that created the gold
lives in git history; this script is the steady-state refresh.)

Dry-run by default: prints per-language counts, the rows dropped as held-out, and
the train ∩ (eval + eval-challenge) leakage check. Re-run with ``--push`` to
publish the train configs.
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.generation.label_corpus.generation_runs import SYNTHETIC_ROOT
from meddies_pii.jsonl import read_jsonl
from meddies_pii.languages import LANGUAGE_PROFILES, normalize_language
from meddies_pii.publishing.corpus_split import dedup_rows, drop_held_out, text_hash
from meddies_pii.publishing.huggingface import push_config

if TYPE_CHECKING:
    from collections.abc import Sequence

Row = dict[str, Any]

REPO_ID = "Meddies/meddies-pii-v2"
EVAL_CONFIG = "eval"
CHALLENGE_CONFIG = "eval-challenge"


def load_local_rows(root: Path, code: str) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(root.glob(f"**/accepted.{code}.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def load_hf_rows(repo_id: str, config: str) -> list[Row]:
    """Existing rows of a config, or [] if the config doesn't exist yet."""
    from datasets import get_dataset_config_names, load_dataset

    if config not in get_dataset_config_names(repo_id):
        return []
    return list(load_dataset(repo_id, config, split="train"))


def read_held_out(repo_id: str) -> tuple[set[str], int, int]:
    """Hash the text of every protected row: the eval holdout and the eval-challenge gold.

    Refuses if either is empty or missing — without the gold present there is no proof it stays
    out of train, and pushing blind could leak it.
    """
    eval_rows = load_hf_rows(repo_id, EVAL_CONFIG)
    gold_rows = load_hf_rows(repo_id, CHALLENGE_CONFIG)
    if not eval_rows:
        msg = f"Refusing: `{EVAL_CONFIG}` config is empty/missing."
        raise SystemExit(msg)
    if not gold_rows:
        msg = f"Refusing: `{CHALLENGE_CONFIG}` config is empty/missing."
        raise SystemExit(msg)
    hashes = {text_hash(r["text"]) for r in (*eval_rows, *gold_rows)}
    return hashes, len(eval_rows), len(gold_rows)


def build_plan(root: Path, languages: Sequence[str], held: set[str]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for lang in languages:
        code = normalize_language(lang).code
        hf_rows = load_hf_rows(REPO_ID, lang)
        local_rows = load_local_rows(root, code)
        union = dedup_rows([*hf_rows, *local_rows])
        train = drop_held_out(union, held)
        plan.append({
            "lang": lang,
            "hf": len(hf_rows),
            "local": len(local_rows),
            "union": len(union),
            "train": train,
            "dropped": len(union) - len(train),
        })
    return plan


def print_plan(plan: list[dict[str, Any]], held: set[str], n_eval: int, n_gold: int) -> int:
    print(f"\n{'language':<13}{'HF':>9}{'local':>9}{'union':>9}{'train':>9}{'dropped':>9}")
    print("-" * 58)
    train_hashes: set[str] = set()
    tot = {"hf": 0, "local": 0, "union": 0, "train": 0, "dropped": 0}
    for p in plan:
        print(f"{p['lang']:<13}{p['hf']:>9,}{p['local']:>9,}{p['union']:>9,}{len(p['train']):>9,}{p['dropped']:>9,}")
        train_hashes.update(text_hash(r["text"]) for r in p["train"])
        tot["hf"] += p["hf"]
        tot["local"] += p["local"]
        tot["union"] += p["union"]
        tot["train"] += len(p["train"])
        tot["dropped"] += p["dropped"]
    print("-" * 58)
    print(f"{'TOTAL':<13}{tot['hf']:>9,}{tot['local']:>9,}{tot['union']:>9,}{tot['train']:>9,}{tot['dropped']:>9,}")

    leaks = len(train_hashes & held)
    print(
        f"\nProtected (read-only, held out of train): "
        f"eval {n_eval:,} + eval-challenge {n_gold:,} = {len(held):,} unique hashes",
    )
    print(f"Rows subtracted from the union as held-out : {tot['dropped']:,}")
    print(f"\nLEAKAGE — train ∩ (eval + eval-challenge)  : {leaks} (must be 0)")
    return leaks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(SYNTHETIC_ROOT))
    parser.add_argument(
        "--push",
        action="store_true",
        help="Publish train configs to HF (overwrites them). Omit for dry-run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    languages = list(LANGUAGE_PROFILES)

    print(f"Reading held-out sets (eval + eval-challenge) from {REPO_ID} ...")
    held, n_eval, n_gold = read_held_out(REPO_ID)

    print(f"Building plan from {args.root} ({len(languages)} languages) ...")
    plan = build_plan(args.root, languages, held)
    leaks = print_plan(plan, held, n_eval, n_gold)

    if not args.push:
        print("\nDRY RUN — re-run with --push to publish train configs. Nothing was sent to HF.")
        return 0

    if leaks:
        msg = "Refusing to push: held-out rows leaked into train (must be 0)."
        raise SystemExit(msg)

    print(f"\nPushing {len(plan)} train configs (`{EVAL_CONFIG}` + `{CHALLENGE_CONFIG}` left untouched) ...")
    for p in plan:
        push_config(REPO_ID, p["lang"], p["train"], split="train")
        print(f"      {p['lang']:<13} {len(p['train']):>7,} rows")

    print(f"\nDONE. v2 train configs refreshed; `{EVAL_CONFIG}` (holdout) + `{CHALLENGE_CONFIG}` (gold) untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
