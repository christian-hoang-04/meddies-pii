#!/usr/bin/env python
"""Daily OpenCode Zen clinical top-up runner for Meddies Labels synthetic rows."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from meddies_pii.generation.label_corpus.runner import (
    GenerationRunError,
    GenerationRunPlan,
    build_opencode_zen_daily_segments,
    provider_keys,
    resolve_max_concurrency,
    run_generation_plan,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "data/bioes-v2/synthetic/opencode_zen"


def _summary_path(output_dir: Path, run_date: str) -> Path:
    return output_dir / "daily_runs" / f"{run_date}.summary.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output dir. Default: data/bioes-v2/synthetic/opencode_zen/<run_date>/.",
    )
    parser.add_argument(
        "--accepted-per-key",
        type=int,
        default=int(os.getenv("OPENCODE_ZEN_ACCEPTED_PER_KEY", "8")),
    )
    parser.add_argument(
        "--daily-accepted-target",
        type=int,
        default=int(os.getenv("OPENCODE_ZEN_DAILY_ACCEPTED_TARGET", "0")),
    )
    parser.add_argument(
        "--expected-key-count",
        type=int,
        default=int(os.getenv("OPENCODE_ZEN_EXPECTED_KEY_COUNT", "7")),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("OPENCODE_ZEN_MAX_CONCURRENCY", "0")),
    )
    parser.add_argument(
        "--rpm-per-key",
        type=int,
        default=int(os.getenv("OPENCODE_ZEN_RPM_PER_KEY", "20")),
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=float,
        default=float(os.getenv("OPENCODE_ZEN_MAX_ATTEMPT_MULTIPLIER", "2.5")),
    )
    parser.add_argument("--model", default=os.getenv("OPENCODE_ZEN_MODEL") or None)
    # reason: the operator's LOCAL day is the intended default for a daily run. Switching to UTC would
    # reason: roll the date back for anyone east of Greenwich running before their morning cutover, so
    # reason: the run would write into yesterday's file. `--run-date` is the override when local is
    # reason: not what is wanted.
    parser.add_argument("--run-date", default=datetime.now().strftime("%Y-%m-%d"))  # ruff: ignore[call-datetime-now-without-tzinfo]
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    keys = provider_keys("opencode_zen")
    key_count = len(keys)
    if key_count <= 0 and not args.dry_run:
        msg = "No OPENCODE_ZEN_API_KEY(S) configured"
        raise SystemExit(msg)

    max_concurrency = resolve_max_concurrency(args.max_concurrency, key_count)
    daily_target = args.daily_accepted_target or key_count * args.accepted_per_key
    output_dir = (
        DEFAULT_OUTPUT_BASE / args.run_date
        if args.output_dir is None
        else args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    warnings: list[str] = []
    if key_count != args.expected_key_count:
        warnings.append(
            f"expected {args.expected_key_count} OpenCode Zen keys but found {key_count}; daily \
target derived from found keys",
        )
    plan = GenerationRunPlan(
        run_id=args.run_date,
        provider="opencode_zen",
        model=args.model,
        output_dir=output_dir,
        max_concurrency=max_concurrency,
        rpm_per_key=args.rpm_per_key,
        max_attempt_multiplier=args.max_attempt_multiplier,
        segments=build_opencode_zen_daily_segments(daily_target),
        summary_path=_summary_path(output_dir, args.run_date),
        dry_run=args.dry_run,
        warnings=tuple(warnings),
        metadata={
            "run_date": args.run_date,
            "key_count": key_count,
            "expected_key_count": args.expected_key_count,
            "daily_accepted_target": daily_target,
            "accepted_per_key": args.accepted_per_key,
            "estimated_request_budget": int(daily_target * args.max_attempt_multiplier),
            "requested_max_concurrency": args.max_concurrency,
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
