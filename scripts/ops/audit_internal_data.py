"""Audit the internal `Meddies/meddies-pii` corpus on Modal (Run 1, Wave 2).

Two entrypoints, run in order:

  # 1) schema probe — stream a few rows per config, print features + samples
  MODAL_PROFILE=openmedical uv run modal run scripts/ops/audit_internal_data.py::probe

  # 2) contamination scan — parse gold spans, run the strengthened Track-D
  #    audit gates, report per-config / per-label issue counts
  MODAL_PROFILE=openmedical uv run modal run scripts/ops/audit_internal_data.py::audit

The repo is PRIVATE → the function attaches `huggingface-secret` (lives in the
openmedical workspace). Results travel via `print(... )` lines (Modal's large-result
blob path is unimplemented in this workspace); a single log line truncates ~64KB so we
keep printed payloads small / chunked.
"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import json
from typing import TYPE_CHECKING, TypedDict

import modal

from meddies_pii.json_types import as_object_list, is_str_mapping
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

REPO = "Meddies/meddies-pii"


class AuditRecord(TypedDict):
    """One row handed to the audit gates: its id, its clean text, and its gold spans."""

    uid: str
    text: str
    gold_spans: list[dict[str, object]]


image = add_source_pythonpath(
    modal.Image
    .from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d")
    .pip_install("datasets==4.5.0", "huggingface_hub==1.19.0")
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT),
)
app = modal.App("meddies-pii-internal-audit", image=image)
_SECRET = modal.Secret.from_name("huggingface-secret")


@app.function(cpu=4.0, memory=16384, timeout=1800, secrets=[_SECRET])
def probe_schema(sample_rows: int = 2) -> None:
    """Stream a few rows per config; print features + truncated samples."""
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(REPO)
    print("INTERNAL_CONFIGS::" + json.dumps(configs, ensure_ascii=False))
    for cfg in configs:
        try:
            ds = load_dataset(REPO, cfg, split="train", streaming=True)
        # reason: a config that will not load raises whatever its loader raises — network, schema or auth — and every one
        # reason: means the same thing here: record this config as an error row and keep sweeping the others.
        except Exception as exc:  # ruff: ignore[blind-except]
            print(f"PROBE_ERROR::{cfg}::{str(exc)[:200]}")
            continue
        features = None
        samples: list[dict[str, object]] = []
        for i, row in enumerate(ds):
            if features is None:
                features = {k: type(v).__name__ for k, v in row.items()}
            trimmed = {k: (v[:240] if isinstance(v, str) else v) for k, v in row.items()}
            samples.append(trimmed)
            if i + 1 >= sample_rows:
                break
        payload = {"config": cfg, "features": features, "samples": samples}
        print("PROBE_ROW::" + json.dumps(payload, ensure_ascii=False)[:60000])


@app.local_entrypoint()
def probe(sample_rows: int = 2) -> None:
    probe_schema.remote(sample_rows)
    print("PROBE_DONE")


LANG_CONFIGS: tuple[str, ...] = (
    "vietnamese",
    "vietnamese-translated",
    "burmese",
    "chinese",
    "english",
    "filipino",
    "french",
    "german",
    "indonesian",
    "japanese",
    "korean",
    "laos",
    "malay",
    "portuguese",
    "russian",
    "spanish",
    "tamil",
    "thai",
)
"""Per-language configs carry inline `[value]<label>` in `text`, clean text in `raw`, and a `{label.

[values]}` dict in `label`. `pii-bioes` carries clean `text` + a `label` list of `{category,start,end,text}` spans. We
build the audit record `{uid, text, gold_spans}` from whichever shape the config uses.

"""


def _records_from_lang_rows(cfg: str, rows: Iterable[Mapping[str, object]]) -> list[AuditRecord]:
    """Turn an inline-tagged config into audit records.

    Offsets are best-effort (value-locate in clean text); the high-severity suspicious-gold
    detector keys on span TEXT, so the contamination rate is robust to offset approximation.
    """
    from meddies_pii.tags import find_tags, strip_pii_tags

    records: list[AuditRecord] = []
    for i, row in enumerate(rows):
        tagged = str(row.get("text") or "")
        clean = str(row.get("raw") or "") or strip_pii_tags(tagged)
        spans: list[dict[str, object]] = []
        for tagged_value, label in find_tags(tagged):
            value = str(tagged_value)
            start = clean.find(value)
            if start < 0:
                start, end = 0, min(len(value), len(clean))
            else:
                end = start + len(value)
            spans.append({"label": str(label), "start": start, "end": end, "text": value})
        records.append({"uid": f"{cfg}:{i}", "text": clean, "gold_spans": spans})
    return records


def _offset_field(span: Mapping[str, object], key: str) -> int:
    """Read one span offset, keeping `int()`'s own rejection of a value it cannot convert.

    Raises:
        TypeError: when the field holds a value `int()` would itself refuse.

    """
    value = span.get(key, 0)
    if isinstance(value, str | int | float):
        return int(value)
    msg = f"span field {key!r} holds {type(value).__name__}, which int() cannot convert"
    raise TypeError(msg)


def _records_from_bioes_rows(rows: Iterable[Mapping[str, object]]) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for i, row in enumerate(rows):
        spans: list[dict[str, object]] = [
            {
                "label": str(s.get("category")),
                "start": _offset_field(s, "start"),
                "end": _offset_field(s, "end"),
                "text": str(s.get("text", "")),
            }
            for s in as_object_list(row.get("label")) or []
            if is_str_mapping(s) and "category" in s
        ]
        records.append({
            "uid": f"pii-bioes:{i}",
            "text": str(row.get("text") or ""),
            "gold_spans": spans,
        })
    return records


@app.function(cpu=16.0, memory=65536, timeout=3600, secrets=[_SECRET])
def audit_config(cfg: str, sample: int, kind: str) -> None:
    """Stream up to `sample` rows and print a per-config contamination summary.

    The strengthened audit gates run over each row; high-severity means a mislabeled non-PII surface.
    """
    from collections import Counter

    from datasets import load_dataset

    from meddies_pii.training.bioes.eval.audit import audit_records, summarize_issues

    ds = load_dataset(REPO, cfg, split="train", streaming=True)
    buffered: list[dict[str, object]] = []
    for i, row in enumerate(ds):
        if i >= sample:
            break
        buffered.append(dict(row))

    records = _records_from_bioes_rows(buffered) if kind == "bioes" else _records_from_lang_rows(cfg, buffered)

    issues = audit_records(records)
    high = [iss for iss in issues if iss.severity == "high"]
    contam_by_label: Counter[str] = Counter(iss.label for iss in high)
    contam_by_reason: Counter[str] = Counter(iss.reason for iss in high)
    rows_with_high = len({iss.uid for iss in high})

    out = {
        "config": cfg,
        "rows_audited": len(records),
        "gold_spans_total": sum(len(r["gold_spans"]) for r in records),
        "total_issues": len(issues),
        "high_severity_issues": len(high),
        "rows_with_high_severity": rows_with_high,
        "contamination_rate_rows": round(rows_with_high / max(len(records), 1), 4),
        "high_by_label": dict(contam_by_label.most_common()),
        "high_by_reason": dict(contam_by_reason.most_common(20)),
        "issue_type_summary": summarize_issues(issues),
    }
    print("AUDIT_RESULT::" + json.dumps(out, ensure_ascii=False)[:60000])


@app.function(cpu=8.0, memory=32768, timeout=1800, secrets=[_SECRET])
def dump_examples(cfg: str, sample: int, kind: str, per_reason: int = 8) -> None:
    """Run the audit and print a few example (label, value) pairs per high-severity reason.

    Seeing the pairs is what tells true contamination apart from detector false positives.
    """
    from collections import defaultdict

    from datasets import load_dataset

    from meddies_pii.training.bioes.eval.audit import audit_records

    ds = load_dataset(REPO, cfg, split="train", streaming=True)
    buffered = [dict(row) for i, row in enumerate(ds) if i < sample]
    records = _records_from_bioes_rows(buffered) if kind == "bioes" else _records_from_lang_rows(cfg, buffered)
    examples: dict[str, list[str]] = defaultdict(list)
    for iss in audit_records(records):
        if iss.severity == "high" and len(examples[iss.reason]) < per_reason:
            examples[iss.reason].append(f"{iss.label}={iss.text[:60]!r}")
    print("EXAMPLES::" + json.dumps({"config": cfg, "examples": dict(examples)}, ensure_ascii=False)[:60000])


@app.local_entrypoint()
def examples(sample: int = 2000) -> None:
    jobs = [
        ("english", sample, "lang"),
        ("korean", sample, "lang"),
        ("vietnamese", sample, "lang"),
        ("pii-bioes", sample, "bioes"),
    ]
    for _ in dump_examples.starmap(jobs):
        pass
    print("EXAMPLES_DONE")


@app.local_entrypoint()
def audit(sample: int = 5000, bioes_sample: int = 10000) -> None:
    jobs = [(cfg, sample, "lang") for cfg in LANG_CONFIGS]
    jobs.append(("pii-bioes", bioes_sample, "bioes"))
    for _ in audit_config.starmap(jobs):
        pass
    print("AUDIT_DONE")
