#!/usr/bin/env python
"""Generate targeted Meddies Labels synthetic rows with OpenAI-compatible providers."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import asyncio
import logging

from dotenv import load_dotenv

from meddies_pii.generation.label_corpus.catalog import (
    DOMAIN_PROFILES,
    REQUIRED_LABEL_MODES,
    SPLIT_PURPOSES,
)
from meddies_pii.generation.label_corpus.runner import (
    SyntheticGenerationRequest,
    run_synthetic_generation,
)
from meddies_pii.generation.label_corpus.synthetic import (
    DEFAULT_TARGETED_GENERATION_DIR,
)
from meddies_pii.taxonomy import PII_LABELS, require_pii_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, help="Vietnamese/vi or English/en")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--domain-profile", choices=DOMAIN_PROFILES, default="medical")
    parser.add_argument("--split-purpose", choices=SPLIT_PURPOSES, default="train")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Restrict generation to this scenario name; can be repeated.",
    )
    parser.add_argument(
        "--count-mode",
        choices=("total", "additional"),
        default="total",
        help="Interpret --target-count as an absolute accepted-file target or rows to add.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_TARGETED_GENERATION_DIR))
    parser.add_argument("--provider", default="mimo")
    parser.add_argument("--model", default=None, help="Defaults to provider env/default model")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--rpm-per-key", type=int, default=60)
    parser.add_argument(
        "--required-label",
        action="append",
        default=[],
        choices=PII_LABELS,
        help="Require this extra Meddies Labels label in every accepted sample; can be repeated.",
    )
    parser.add_argument(
        "--required-label-mode",
        choices=REQUIRED_LABEL_MODES,
        default="private_url_secret",
        help="Built-in required-label policy for the selected scenario.",
    )
    parser.add_argument("--max-attempt-multiplier", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--allow-vietnamese-without-diacritics",
        action="store_true",
        help="Disable the Vietnamese diacritic sanity check.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    summary = await run_synthetic_generation(
        SyntheticGenerationRequest(
            provider=args.provider,
            model=args.model,
            language=args.language,
            target_count=args.target_count,
            output_dir=args.output_dir,
            domain_profile=args.domain_profile,
            split_purpose=args.split_purpose,
            scenario_names=args.scenario,
            count_mode=args.count_mode,
            max_concurrency=args.max_concurrency,
            rpm_per_key=args.rpm_per_key,
            required_labels=tuple(require_pii_label(label) for label in args.required_label),
            required_label_mode=args.required_label_mode,
            max_attempt_multiplier=args.max_attempt_multiplier,
            seed=args.seed,
            require_vietnamese_marker=not args.allow_vietnamese_without_diacritics,
            log_every=args.log_every,
        ),
    )
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
