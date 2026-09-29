#!/usr/bin/env python
"""Run the planned MiMo 20k general/adversarial Meddies Labels generation job."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from meddies_pii.generation.label_corpus.runner import (
    GenerationRunError,
    GenerationRunPlan,
    build_mimo_general_segments,
    configured_base_urls,
    language_targets,
    preflight_keys,
    provider_keys,
    run_generation_plan,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/bioes-v2/synthetic/mimo_general_20k"


def mimo_key_count() -> int:
    return len(provider_keys("mimo"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--total-target",
        type=int,
        default=int(os.getenv("MIMO_GENERAL_TOTAL_TARGET", "20000")),
    )
    parser.add_argument("--provider", default="mimo")
    parser.add_argument("--model", default=os.getenv("MIMO_MODEL") or None)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("MIMO_GENERAL_MAX_CONCURRENCY", "4")),
    )
    parser.add_argument(
        "--rpm-per-key",
        type=int,
        default=int(os.getenv("MIMO_GENERAL_RPM_PER_KEY", "45")),
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=float,
        default=float(os.getenv("MIMO_GENERAL_MAX_ATTEMPT_MULTIPLIER", "3.0")),
    )
    # reason: this timestamp is a run LABEL the operator reads and matches against their own clock, not
    # reason: an instant anything computes with. A UTC default would make the id disagree with the wall
    # reason: clock of the person who launched the run.
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))  # ruff: ignore[call-datetime-now-without-tzinfo]
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-key-preflight",
        action="store_true",
        help="Use configured keys without redacted health-checking them first.",
    )
    parser.add_argument("--allow-vietnamese-without-diacritics", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    segments = build_mimo_general_segments(args.total_target)
    configured_keys = provider_keys(args.provider)
    base_urls = configured_base_urls(args.provider, len(configured_keys))
    key_preflight: list[dict[str, object]] = []
    healthy_keys = configured_keys
    healthy_base_urls = base_urls

    if not args.dry_run:
        if not configured_keys:
            msg = "No API key(s) configured for provider"
            raise SystemExit(msg)
        if not args.skip_key_preflight:
            preflight = asyncio.run(
                preflight_keys(
                    provider=args.provider,
                    model=args.model,
                    keys=configured_keys,
                    base_urls=base_urls,
                ),
            )
            key_preflight = [asdict(result) for result in preflight]
            healthy_indices = {result.key_index for result in preflight if result.ok}
            healthy_keys = [key for index, key in enumerate(configured_keys, start=1) if index in healthy_indices]
            healthy_base_urls = [base_url for index, base_url in enumerate(base_urls, start=1) if index in healthy_indices]
            if not healthy_keys:
                msg = "No healthy API keys after preflight"
                raise SystemExit(msg)

    plan = GenerationRunPlan(
        run_id=args.run_id,
        provider=args.provider,
        model=args.model,
        output_dir=output_dir,
        max_concurrency=args.max_concurrency,
        rpm_per_key=args.rpm_per_key,
        max_attempt_multiplier=args.max_attempt_multiplier,
        segments=segments,
        summary_path=output_dir / "run_summaries" / f"{args.run_id}.summary.json",
        dry_run=args.dry_run,
        api_keys=tuple(healthy_keys) if not args.dry_run else None,
        base_urls=tuple(healthy_base_urls) if not args.dry_run else None,
        require_vietnamese_marker=not args.allow_vietnamese_without_diacritics,
        metadata={
            "configured_key_count": len(configured_keys),
            "healthy_key_count": len(healthy_keys) if not args.dry_run else None,
            "configured_endpoint_count": len(set(base_urls)),
            "key_preflight": key_preflight,
            "mimo_key_count": mimo_key_count(),
            "total_target": args.total_target,
            "language_targets": language_targets(segments),
        },
    )
    try:
        summary = asyncio.run(run_generation_plan(plan))
    except GenerationRunError as exc:
        print(json.dumps(exc.summary, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
