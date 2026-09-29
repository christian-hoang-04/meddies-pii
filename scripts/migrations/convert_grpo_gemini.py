#!/usr/bin/env python
"""Convert the GRPO Gemini-batch hard-examples into Meddies Labels training JSONL.

Joins the batch INPUT (`gemini_batch.jsonl`, source doc in the prompt) with the
batch OUTPUT (`gemini_results.jsonl`, label -> [values]) on `key`. Core
span-reconstruction logic lives in
`meddies_pii.training.bioes.data.grpo_convert` (pure + tested); this is the I/O.
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import json
from pathlib import Path
from typing import cast

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.training.bioes.data.grpo_convert import to_record

REPO = Path(__file__).resolve().parents[2]


def _parse_results(path: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                inner = rec["response"]["candidates"][0]["content"]["parts"][0]["text"]
                out[str(rec["key"])] = json.loads(inner)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
    return out


def _parse_batch(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[str(rec["key"])] = rec["request"]["contents"][0]["parts"][0]["text"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GRPO Gemini batch hard-examples to Meddies Labels.")
    parser.add_argument("--batch", type=Path, default=REPO / "data/hard-examples/gemini_batch.jsonl")
    parser.add_argument("--results", type=Path, default=REPO / "data/hard-examples/gemini_results.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / f"data/bioes-v2/internal/grpo/grpo-gemini.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    )
    args = parser.parse_args()

    batch = _parse_batch(args.batch)
    results = _parse_results(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = dropped_no_join = dropped_no_spans = 0
    label_counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as out:
        for key, extractions in results.items():
            prompt = batch.get(key)
            if prompt is None:
                dropped_no_join += 1
                continue
            record = to_record(prompt, extractions, uid=f"grpo-{key}")
            if record is None:
                dropped_no_spans += 1
                continue
            for span in cast("list[dict[str, object]]", record["label"]):
                cat = str(span["category"])
                label_counts[cat] = label_counts.get(cat, 0) + 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(
        json.dumps(
            {
                "kept": kept,
                "dropped_no_join": dropped_no_join,
                "dropped_no_spans": dropped_no_spans,
                "label_span_counts": dict(sorted(label_counts.items())),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
