#!/usr/bin/env python
"""Verify whether ai4privacy openpii-1.5m is a row-superset of openpii-1m.

Method: reservoir-sample N source_text hashes from openpii-1m, then stream
openpii-1.5m and count how many of the sampled hashes appear. High overlap
(~100%) confirms the "1.5m extends 1m" superset inference; low overlap means
they are independent corpora and both could be pulled without duplication.

Run: uv run python scripts/archive/check_openpii_superset.py
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import hashlib
import random

from datasets import load_dataset

SUPERSET_OVERLAP_PERCENT = 95
DISJOINT_OVERLAP_PERCENT = 20

SAMPLE_N = 2000
ONEM_SCAN_CAP = 300_000
ONE5M_SCAN_CAP = 1_700_000
SEED = 42


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def reservoir_sample_1m() -> dict[str, str]:
    # reason: the fixed SEED is the point — this reservoir must draw the same rows on every run so
    # reason: the superset check is reproducible. A cryptographic source would make it unrepeatable.
    rng = random.Random(SEED)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split="train", streaming=True)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    for row in ds:
        if seen >= ONEM_SCAN_CAP:
            break
        text = row.get("source_text") or ""
        if not text.strip():
            continue
        lang = str(row.get("language", "?"))
        item = (_hash(text), lang)
        if len(reservoir) < SAMPLE_N:
            reservoir.append(item)
        else:
            j = rng.randint(0, seen)
            if j < SAMPLE_N:
                reservoir[j] = item
        seen += 1
    print(f"[1m] scanned {seen} rows, sampled {len(reservoir)}", flush=True)
    return dict(reservoir)


def scan_1_5m(target: dict[str, str]) -> tuple[int, int, int, int]:
    ds = load_dataset("ai4privacy/pii-masking-openpii-1.5m", split="train", streaming=True)
    remaining = set(target)
    found = 0
    scanned = 0
    vietnamese = 0
    for row in ds:
        if scanned >= ONE5M_SCAN_CAP or not remaining:
            break
        text = row.get("source_text") or ""
        if str(row.get("language", "")).lower().startswith("viet"):
            vietnamese += 1
        h = _hash(text)
        if h in remaining:
            remaining.discard(h)
            found += 1
        scanned += 1
        if scanned % 100_000 == 0:
            print(
                f"[1.5m] scanned {scanned}, found {found}/{len(target)}, vi_seen {vietnamese}",
                flush=True,
            )
    return found, len(target), scanned, vietnamese


def main() -> None:
    print("Sampling openpii-1m ...", flush=True)
    target = reservoir_sample_1m()
    print("Scanning openpii-1.5m for those rows ...", flush=True)
    found, total, scanned, vietnamese = scan_1_5m(target)
    pct = 100.0 * found / total if total else 0.0
    print("\n===== RESULT =====", flush=True)
    print(f"1m sample size:        {total}", flush=True)
    print(f"found in 1.5m:         {found} ({pct:.1f}%)", flush=True)
    print(f"1.5m rows scanned:     {scanned}", flush=True)
    print(f"1.5m Vietnamese seen:  {vietnamese}", flush=True)
    if pct >= SUPERSET_OVERLAP_PERCENT:
        print("VERDICT: 1.5m is a superset of 1m -> drop 1m, use 1.5m only.", flush=True)
    elif pct <= DISJOINT_OVERLAP_PERCENT:
        print(
            "VERDICT: largely independent -> both pull-able without duplication.",
            flush=True,
        )
    else:
        print("VERDICT: partial overlap -> needs row-level dedup on merge.", flush=True)


if __name__ == "__main__":
    main()
