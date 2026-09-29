"""Full-scale PII dataset audit on Modal.

Downloads each external source in Modal's cloud (fast), and reports per source:
raw label distribution, Meddies Labels-mapped span + doc counts, language mix, and
(Nemotron) domain mix. Drives the data-redistribution strategy with real numbers
instead of local 2k-row samples.

Run:  MODAL_PROFILE=hahuyhoang411 uv run modal run scripts/ops/modal_dataset_audit.py
Output: printed summary + JSON written back to ./_dataset_audit.json

Each container prints an AUDIT_RESULT:: line into the streamed run log; parse those lines from the log (no return values to
avoid the blob path). `only` filters SOURCES by substring (dataset id or kind), e.g. --only gretel.

"""

from __future__ import annotations

# ruff: file-ignore[implicit-namespace-package]
# reason: this module is launched as `uv run modal run <this path>` and is never imported, so it is a script
# reason: rather than a package member. An `__init__.py` would declare this directory a package it is not, and
# reason: the sibling scripts here that run under `python` say so with a shebang instead.
# ruff: file-ignore[docstring-missing-yields]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import ast
import json
import json as _json
from collections import Counter
from typing import TYPE_CHECKING

import modal

from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

image = add_source_pythonpath(
    modal.Image
    .from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d")
    .pip_install("datasets==4.5.0", "huggingface_hub==1.19.0")
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT),
)
app = modal.App("meddies-pii-dataset-audit", image=image)

SOURCES: list[tuple[str, str, str]] = [
    ("ai4privacy/pii-masking-openpii-1.5m", "train", "ai4privacy"),
    ("ai4privacy/pii-masking-openpii-1.5m", "validation", "ai4privacy"),
    ("nvidia/Nemotron-PII", "train", "nemotron"),
    ("gretelai/gretel-pii-masking-en-v1", "train", "gretel"),
    ("ai4privacy/open-pii-masking-500k-ai4privacy", "train", "ai4privacy"),
]
"""(dataset_id, split, kind) — kind selects the span/label/lang extractor."""

AI4PRIVACY_DROP = {"age", "gender", "sex", "title", "time"}
"""Mirror of bioes/data/mixed.py drop-sets (source of truth lives there)."""
NEMOTRON_DROP = {
    "age",
    "blood_type",
    "employment_status",
    "gender",
    "language",
    "occupation",
    "political_view",
    "race_ethnicity",
    "religious_belief",
    "sexuality",
    "time",
    "url",
}


def _pii_label_map(label: str, label_map: dict[str, str], allowed: set[str], drop: set[str]) -> str | None:
    key = label.strip().lower()
    if key in drop:
        return None
    if key in allowed:
        return key
    mapped = label_map.get(key)
    if mapped is None:
        return None
    out = mapped.strip("<>").lower()
    return out if out in allowed else None


def _as_item_list(value: object) -> list[object]:
    """Return the items of a span collection that a row may carry as a list or as its repr.

    The external datasets serialize these columns inconsistently: some splits hold a real
    list, others hold the ``str`` of one. Anything that survives neither reading is empty.

    Returns:
        The collection's items, or an empty list when the column holds neither form.

    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:  # ruff: ignore[blind-except]
            # reason: literal_eval raises several unrelated types on malformed input and every one of them means the
            # reason: same thing here: this row carries no readable spans.
            return []
    return list(value) if isinstance(value, list) else []


# reason: iter spans coordinates literal eval with pm; extra seams would leak shared intermediate state.
def _iter_spans(  # ruff: ignore[complex-structure,too-many-branches]
    row: Mapping[str, object],
    kind: str,
) -> Iterator[str]:
    """Yield (raw_label) strings and return (language, domain) via row inspection.

    gretel `entities` = "[{'entity': '<value>', 'types': ['<label>']}, ...]" The label is in the `types` list; `entity`
    holds the value, not the label.

    """
    if kind == "ai4privacy":
        for e in _as_item_list(row.get("privacy_mask")):
            if isinstance(e, dict):
                lab = e.get("label") or e.get("type") or e.get("entity")
                if lab:
                    yield str(lab)
    elif kind == "nemotron":
        for s in _as_item_list(row.get("spans")):
            if isinstance(s, dict):
                lab = s.get("label") or s.get("category") or s.get("type")
                if lab:
                    yield str(lab)
    elif kind == "gretel":
        for e in _as_item_list(row.get("entities")):
            if isinstance(e, dict):
                raw_types = e.get("types")
                types = [raw_types] if isinstance(raw_types, str) else _as_item_list(raw_types)
                for t in types:
                    if t:
                        yield str(t)


def _language_of(row: Mapping[str, object]) -> str:
    for k in ("language", "locale", "lang"):
        v = row.get(k)
        if v:
            return str(v).lower()
    return "?"


@app.function(cpu=16.0, memory=65536, timeout=3600)
def audit_one(dataset_id: str, split: str, kind: str) -> None:
    """Print rather than return.

    Modal's large-result blob path (BlobGet) is unimplemented in this workspace, so results travel back via streamed logs.

    """
    from datasets import load_dataset

    from meddies_pii.annotations.label_aliases import LABEL_MAP
    from meddies_pii.taxonomy import PII_LABELS

    allowed: set[str] = set(PII_LABELS)
    drop: set[str] = NEMOTRON_DROP if kind == "nemotron" else (AI4PRIVACY_DROP if kind == "ai4privacy" else set())

    ds = load_dataset(dataset_id, split=split, num_proc=16)
    schema = list(ds.features.keys())

    raw = Counter()
    mapped_spans = Counter()
    mapped_docs = Counter()
    langs = Counter()
    domains = Counter()
    n = 0
    for row in ds:
        n += 1
        langs[_language_of(row)] += 1
        if kind == "nemotron":
            dom = row.get("domain")
            if dom:
                domains[str(dom)] += 1
        doc_labels = set()
        for lab in _iter_spans(row, kind):
            raw[lab.lower()] += 1
            m = _pii_label_map(lab, LABEL_MAP, allowed, drop)
            if m:
                mapped_spans[m] += 1
                doc_labels.add(m)
        for m in doc_labels:
            mapped_docs[m] += 1

    result = {
        "dataset_id": dataset_id,
        "split": split,
        "kind": kind,
        "schema": schema,
        "n_rows": n,
        "raw_label_counts": dict(raw.most_common()),
        "pii_label_span_counts": dict(mapped_spans.most_common()),
        "pii_label_document_counts": dict(mapped_docs.most_common()),
        "language_counts": dict(langs.most_common()),
        "domain_counts": dict(domains.most_common(40)),
    }
    print("AUDIT_RESULT::" + json.dumps(result, ensure_ascii=False))


# reason: Dataset load, URL sampling, and evidence share one audit population; splitting would skew counts.
@app.function(cpu=8.0, memory=32768, timeout=900)
def dump_url_samples(dataset_id: str, kind: str, n: int = 150) -> None:  # ruff: ignore[complex-structure,too-many-branches]
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", num_proc=16)
    out: list[str] = []
    for row in ds:
        if kind == "nemotron":
            spans = row.get("spans")
            if isinstance(spans, str):
                try:
                    spans = ast.literal_eval(spans)
                # reason: `ast.literal_eval` raises several unrelated exception types on malformed input and
                # reason: every one means the same thing at this site: the row carries no readable value.
                except Exception:  # ruff: ignore[blind-except]
                    spans = []
            for s in spans or []:
                if isinstance(s, dict) and str(s.get("label", "")).lower() == "url":
                    v = s.get("text")
                    if v:
                        out.append(str(v))
        elif kind == "gretel":
            ent = row.get("entities")
            if isinstance(ent, str):
                try:
                    ent = ast.literal_eval(ent)
                # reason: `ast.literal_eval` raises several unrelated exception types on malformed input and
                # reason: every one means the same thing at this site: the row carries no readable value.
                except Exception:  # ruff: ignore[blind-except]
                    ent = []
            for e in ent or []:
                if isinstance(e, dict) and "url" in [str(t).lower() for t in (e.get("types") or [])]:
                    v = e.get("entity")
                    if v:
                        out.append(str(v))
        if len(out) >= n:
            break
    print("URL_SAMPLES::" + json.dumps(out[:n], ensure_ascii=False))


@app.local_entrypoint()
def sample_urls(dataset_id: str = "nvidia/Nemotron-PII", kind: str = "nemotron") -> None:
    dump_url_samples.remote(dataset_id, kind)


@app.local_entrypoint()
def main(only: str = "") -> None:
    selected = [s for s in SOURCES if not only or only in s[0] or only == s[2]]
    for _ in audit_one.starmap(selected):
        pass
    print("AUDIT_DONE")


@app.function(
    cpu=4.0,
    memory=16384,
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def inspect_meddies(repo: str = "Meddies/meddies-pii") -> None:

    from datasets import get_dataset_config_info, get_dataset_config_names

    configs = get_dataset_config_names(repo)
    out = {}
    for cfg in configs:
        try:
            info = get_dataset_config_info(repo, cfg)
            splits = {
                name: (s.num_examples if s.num_examples is not None else -1) for name, s in (info.splits or {}).items()
            }
            out[cfg] = splits
        # reason: a config that will not load raises whatever its loader raises — network, schema or auth — and every one
        # reason: means the same thing here: record this config as an error row and keep sweeping the others.
        except Exception as exc:  # ruff: ignore[blind-except,try-except-in-loop]
            out[cfg] = {"error": str(exc)[:120]}
    print("MEDDIES_CONFIGS::" + _json.dumps(out, ensure_ascii=False))


@app.local_entrypoint()
def meddies(repo: str = "Meddies/meddies-pii") -> None:
    inspect_meddies.remote(repo)
