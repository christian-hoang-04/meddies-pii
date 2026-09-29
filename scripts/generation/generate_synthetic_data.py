#!/usr/bin/env python
"""17-language Meddies Labels generation adapters (daily weak-label + eval-gold run)."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import asyncio
import json
from typing import TYPE_CHECKING

from dotenv import load_dotenv

# reason: the redundant alias is what keeps this name on the module object. `test_registries.py`
# reason: loads this script through `importlib` and reads `module.DEFAULT_LANGUAGE_KEYS` to prove the
# reason: script and the build agree on the language set; a plain import here reads as unused and gets
# reason: stripped, and the test fails on a missing attribute.
from meddies_pii.generation.label_corpus.generation_runs import (
    DEFAULT_LANGUAGE_KEYS as DEFAULT_LANGUAGE_KEYS,  # ruff: ignore[useless-import-alias]
)
from meddies_pii.generation.label_corpus.generation_runs import (
    WEAK_LABEL_FREE_PROVIDERS,
    resolve_languages,
    run_eval_gold,
    run_weak_labels,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Run-2a Meddies Labels generation (no Modal).")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_gold = sub.add_parser("eval-gold", help="Component A eval gold (openai gpt-5.4-mini).")
    eval_gold.add_argument("--per-lang", type=int, default=200)
    eval_gold.add_argument("--languages", nargs="*", default=None, help="Subset of the 17 (default all).")
    eval_gold.add_argument(
        "--token-cap",
        type=int,
        default=None,
        help="Override the openai daily token cap for this run.",
    )

    weak_labels = sub.add_parser("weak-labels", help="private_url + non-en secret (free pools).")
    weak_labels.add_argument(
        "--private-url-per-lang",
        type=int,
        default=40,
        help="private_url samples generated per language per pass (batch size). "
        "Each provider repeats passes until its budget/RPD or --max-minutes stops it.",
    )
    weak_labels.add_argument("--secret-per-lang", type=int, default=20)
    weak_labels.add_argument("--languages", nargs="*", default=None, help="Subset of the 17 (default all).")
    weak_labels.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Generation providers (default: openrouter opencode_zen free pools). "
        "Pass 'openai' to spend paid budget on the bulk; combine with --token-cap.",
    )
    weak_labels.add_argument(
        "--token-cap",
        type=int,
        default=None,
        help="Cap openai token spend for this run. The in-memory budget resets per "
        "process, so set this to the real remaining daily allowance.",
    )
    weak_labels.add_argument(
        "--count-mode",
        choices=("total", "additional"),
        default="additional",
        help="'additional' (default) generates fresh samples each pass so the run "
        "drives toward the providers' budgets; 'total' fills each cell up to "
        "--per-lang once and resumes (leaves most quota unused).",
    )
    weak_labels.add_argument(
        "--max-rounds",
        type=int,
        default=1000,
        help="Max passes over the languages per provider. The real stop is each "
        "provider's token budget / RPD (or --max-minutes); this is just a backstop.",
    )
    weak_labels.add_argument(
        "--max-minutes",
        type=float,
        default=180.0,
        help="Wall-clock deadline for the run. Providers generate in parallel until "
        "their budget/RPD trips or this deadline — bounds runtime so uncapped/slow "
        "providers (nim, llm7) don't run indefinitely. 0 disables the deadline.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    try:
        languages = resolve_languages(args.languages)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.command == "eval-gold":
        summary = asyncio.run(run_eval_gold(args.per_lang, languages, args.token_cap))
        prefix = "EVAL_GOLD_RESULT::"
    else:
        summary = asyncio.run(
            run_weak_labels(
                args.private_url_per_lang,
                args.secret_per_lang,
                languages,
                providers=tuple(args.providers) if args.providers else WEAK_LABEL_FREE_PROVIDERS,
                openai_cap=args.token_cap,
                count_mode=args.count_mode,
                max_rounds=args.max_rounds,
                max_minutes=args.max_minutes or None,
            ),
        )
        prefix = "WEAK_LABELS_RESULT::"
    print(prefix + json.dumps(summary, ensure_ascii=False)[:60000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
