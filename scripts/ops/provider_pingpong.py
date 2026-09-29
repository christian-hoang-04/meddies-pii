#!/usr/bin/env python
"""Provider pingpong: validate every configured key + each model's real output.

For each provider it (1) preflights all configured keys (one tiny call per key,
so multi-account setups are verified per account) and (2) runs a real
weak-label generation request and inspects the accepted spans + rejection
reasons — so we confirm both "can we reach it" and "does the model produce
usable [value]<label> data".

Usage:
    uv run python scripts/ops/provider_pingpong.py            # all providers
    uv run python scripts/ops/provider_pingpong.py groq mistral
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import asyncio
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from meddies_pii.generation.label_corpus.runner import (
    SyntheticGenerationRequest,
    configured_base_urls,
    preflight_keys,
    provider_keys,
    run_synthetic_generation,
)
from meddies_pii.generation.openai_compatible.providers import (
    get_provider_spec,
    resolve_provider_model,
)
from meddies_pii.jsonl import read_jsonl
from meddies_pii.languages import normalize_language

DEFAULT_PROVIDERS = (
    "groq",
    "cerebras",
    "mistral",
    "cloudflare",
    "ovh",
    "llm7",
    "pollinations",
    "nim",
    "openrouter",
    "opencode_zen",
    "openai",
)


# reason: pingpong keeps resolve beside run; splitting would desync retries and counters.
async def pingpong(provider: str) -> None:  # ruff: ignore[complex-structure]
    """1.

    preflight every key (skip for keyless: no per-key identity, and the per-IP limit makes extra calls wasteful —
    generation is the ping).

    2. real generation request. Span/label gates live per scenario in catalog.py (private_portal_secret_focus: 10 spans / 6
    unique labels), not per request — a weak model that can't clear the bar shows accepted=0 with the reason in
    rejection_distribution, which is the diagnostic signal we want. keyless endpoints are per-IP rate-limited → serialize
    (concurrency 1).

    """
    print(f"\n{'=' * 64}\n{provider.upper()}")
    spec = get_provider_spec(provider)
    keys = provider_keys(provider)
    if not keys and spec.keyless:
        keys = [""]
    if not keys:
        print("  no keys configured — skip")
        return
    model = resolve_provider_model(provider, None)
    base_urls = configured_base_urls(provider, len(keys))
    print(f"  model={model}  keys={'keyless' if spec.keyless else len(keys)}")

    if "{account_id}" in (base_urls[0] if base_urls else ""):
        print("  ⚠ Cloudflare account id missing — set CLOUDFLARE_API_IDS in .env")
        return

    if not spec.keyless:
        results = await preflight_keys(provider=provider, model=model, keys=keys, base_urls=base_urls)
        ok = sum(1 for r in results if r.ok)
        print(f"  preflight: {ok}/{len(results)} keys OK")
        for r in results:
            if not r.ok:
                print(f"    key#{r.key_index} FAIL status={r.status} {r.error_type}: {r.error}")
        if ok == 0:
            print("  no working key — skip generation")
            return

    out = Path(tempfile.mkdtemp(prefix=f"pingpong_{provider}_"))
    try:
        summary = await run_synthetic_generation(
            SyntheticGenerationRequest(
                provider=provider,
                model=None,
                language="Vietnamese",
                target_count=2,
                output_dir=out,
                domain_profile="medical",
                split_purpose="train",
                scenario_names=("private_portal_secret_focus",),
                required_label_mode="private_url_secret",
                count_mode="total",
                max_attempt_multiplier=3.0,
                max_concurrency=1 if spec.keyless else 2,
                rpm_per_key=2 if spec.keyless else 60,
                require_vietnamese_marker=True,
            ),
        )
    # reason: the probe's product is the printed diagnosis of how a provider failed. A provider client raises
    # reason: transport, auth, quota and decoding errors, and reporting the type name is the measurement.
    except Exception as exc:  # ruff: ignore[blind-except]
        print(f"  generation ERROR: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        return

    acc = summary.get("accepted_count", 0)
    print(
        f"  generation: accepted={acc} "
        f"rejected={summary.get('rejected_count', 0)} "
        f"raw_attempts={summary.get('raw_attempt_count', 0)}",
    )
    if summary.get("label_distribution"):
        print(f"  labels: {summary['label_distribution']}")
    if summary.get("rejection_distribution"):
        print(f"  rejections: {summary['rejection_distribution']}")

    code = normalize_language("Vietnamese").code
    accepted_path = out / f"accepted.{code}.jsonl"
    if acc and accepted_path.exists():
        row = next(iter(read_jsonl(accepted_path)), None)
        if isinstance(row, dict):
            text = str(row.get("text", ""))
            labels = row.get("label", [])
            spans = [
                (s.get("category"), s.get("text"))
                for s in (labels if isinstance(labels, list) else [])
                if isinstance(s, dict)
            ]
            print(f"  sample text: {text[:240]!r}")
            print(f"  sample spans: {spans}")


async def main() -> None:
    load_dotenv()
    providers = sys.argv[1:] or list(DEFAULT_PROVIDERS)
    for provider in providers:
        try:
            await pingpong(provider)
        # reason: one provider crashing is the result being measured, not an error in the sweep; the next provider
        # reason: still runs.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"  {provider} crashed: {type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
