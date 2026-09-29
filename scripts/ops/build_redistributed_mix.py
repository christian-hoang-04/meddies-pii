"""Redistributed-mix builder + Component B builder on Modal (Run 1, Wave 2).

Loads every source ONCE, converts through the project's real Meddies Labels converters,
re-audits the internal `pii-bioes` pool with the strengthened Track-D gates
(quarantining high-severity contamination), folds the external sources
(17-language-filtered, `source`-tagged), carves a balanced held-out, and builds the
ai4privacy `validation` Component-B eval slice.

NO license gating (2026-06-11): all sources used in full; the only gates are the
17-language allowlist + the contamination quarantine. Run 1 produces LOCAL-PREVIEW
distributions + writes train/held-out/Component-B JSONL to the artifact Volume. NO HF
push (Ha-gated).

  MODAL_PROFILE=openmedical uv run modal run scripts/ops/build_redistributed_mix.py::build

Internal `Meddies/meddies-pii` is PRIVATE → attaches `huggingface-secret` (openmedical).
Results print as `MIX_RESULT::` / `COMPB_RESULT::` lines (Modal blob-return is broken);
the JSONL payloads go to the Volume, not the logs.
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
import pathlib
from typing import TYPE_CHECKING

import modal

from meddies_pii.json_types import is_str_mapping
from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from meddies_pii.training.bioes.data.record_schema import NormalizedRecord

    type RowConverter = Callable[..., tuple[NormalizedRecord | None, Counter[str]]]

REPO = "Meddies/meddies-pii"
OPENPII = "ai4privacy/pii-masking-openpii-1.5m"
NEMOTRON = "nvidia/Nemotron-PII"
GRETEL = "gretelai/gretel-pii-masking-en-v1"
OPENPII_500K = "ai4privacy/open-pii-masking-500k-ai4privacy"
ARTIFACT_VOLUME = "meddies-pii-bioes-artifacts"
OUT_ROOT = "/artifacts/run1"

image = add_source_pythonpath(
    modal.Image
    .from_registry("python@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d")
    .pip_install(
        "datasets==4.5.0",
        "huggingface_hub==1.19.0",
        "transformers==5.11.0",
    )
    .add_local_dir("src", remote_path=MODAL_SOURCE_ROOT),
)
"""augmentation.py / splits.py import `transformers.AutoTokenizer` at module load.

So the converter + dedup chain needs transformers present.

"""
app = modal.App("meddies-pii-redistributed-mix", image=image)
_SECRET = modal.Secret.from_name("huggingface-secret")
_VOLUME = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)


def _bioes_row_to_record(row: Mapping[str, object], index: int) -> dict[str, object] | None:
    """Convert a `pii-bioes` row into the canonical Meddies Labels record.

    The shape matches `mixed.py:_record`: `{text, label:[{category,start,end,text}], info}`.
    Emitting `label` rather than `spans` keeps internal rows schema-identical to the external
    converters, so every source prepares uniformly downstream (the Component-B / external-row
    drop bug).
    """
    raw_spans = row.get("label")
    spans = [
        {
            "category": str(s.get("category")),
            "start": int(str(s.get("start", 0))),
            "end": int(str(s.get("end", 0))),
            "text": str(s.get("text", "")),
        }
        for s in (raw_spans if isinstance(raw_spans, list) else [])
        if is_str_mapping(s) and "category" in s
    ]
    text = str(row.get("text") or "")
    if not text:
        return None
    raw_info = row.get("info")
    info: dict[str, object] = dict(raw_info) if is_str_mapping(raw_info) else {}
    info.setdefault("source", "meddies-internal")
    info.setdefault("source_dataset", "Meddies/meddies-pii:pii-bioes")
    info.setdefault("id", info.get("id") or f"pii-bioes:{index}")
    return {"text": text, "label": spans, "info": info}


QUARANTINE_REASONS = frozenset({"dosage_or_unit_labeled_pii", "age_expression_labeled_pii"})
"""Reasons whose example values were verified GENUINE contamination (2026-06-11 example dump).

Dosages and ages mislabeled as PII. We quarantine ONLY on these. The broad audit high-severity set over-flags on
multilingual date surfaces (`_has_date_surface` rejects `15-MAY-1985`, Korean `2024년 10월 20일`, Vietnamese `08h30`), on
structured IDs caught by the range regex (Korean RRN `860515-1234567`), and on FHIR/JSON spans mis-extracted by `find_tags`
— auto-quarantining on those would delete ~half the corpus, biased against non-English. See the Run-1 gotcha.

"""


def _high_severity_reasons(record: Mapping[str, object]) -> list[str]:
    """Collect every high-severity audit reason for a record, for reporting.

    `audit_record` wants `label`-keyed gold spans while the canonical record stores
    `category`-keyed spans under `label`, so the remap from category to label happens here.
    """
    from meddies_pii.training.bioes.eval.audit import audit_record

    raw_spans = record.get("label")
    gold_spans = [
        {
            "label": s.get("category") or s.get("label"),
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
        }
        for s in (raw_spans if isinstance(raw_spans, list) else [])
        if is_str_mapping(s)
    ]
    raw_info = record.get("info")
    info = raw_info if is_str_mapping(raw_info) else {}
    audit_input = {
        "uid": str(info.get("id", "")),
        "text": record["text"],
        "gold_spans": gold_spans,
    }
    return [iss.reason for iss in audit_record(audit_input) if iss.severity == "high"]


# reason: build combines load dataset and from rows; splitting would misattribute row errors.
@app.function(
    cpu=16.0,
    memory=131072,
    timeout=5400,
    secrets=[_SECRET],
    volumes={"/artifacts": _VOLUME},
)
def build(  # ruff: ignore[complex-structure,too-many-arguments,too-many-locals,too-many-statements,too-many-positional-arguments]
    internal_cap: int = 40000,
    nemotron_cap: int = 50000,
    gretel_cap: int = 10000,
    openpii_500k_cap: int = 20000,
    openpii_max_scan: int = 400000,
    heldout_rows: int = 1500,
    seed: int = 17,
) -> None:
    """---- internal pii-bioes: re-audit; quarantine ONLY on precision-verified reasons.

    All high-severity reasons are TALLIED (flagged_reasons) for the review report, but only QUARANTINE_REASONS (dosage/age
    — verified genuine) actually remove a row.

    Cross-source dedup is by normalized-text hash. Per-source IDs live in disjoint namespaces, so id-dedup across sources
    is redundant — text hash is the real key.

    ---- ai4privacy 1.5m train: Track-C quota selection (17-lang, vi+APAC) ----.

    ---- helper for the streaming converters (Nemotron / gretel / 500k) ----.

    ---- natural distribution (honest, pre-balance) ----.

    ---- carve a balanced (30/30/40) held-out; train = remainder ----.

    ---- Component B: ai4privacy `validation` slice (~2k rows, lang-balanced, hash-disjoint from the whole training pool
    via seen_hashes) ----.

    """
    from collections import Counter

    from datasets import load_dataset

    from meddies_pii.training.bioes.data.augmentation import text_hash
    from meddies_pii.training.bioes.data.mixed import (
        convert_ai4privacy_row,
        convert_gretel_row,
        convert_nemotron_row,
        is_supported_meddies_language,
        summarize_records,
    )
    from meddies_pii.training.bioes.data.openpii_candidates import (
        DEFAULT_OPENPII_LANGUAGE_QUOTAS,
        select_openpii_candidates_from_rows,
    )
    from meddies_pii.training.bioes.data.splits import carve_heldout, normalize_text

    pool: list[Mapping[str, object]] = []
    quarantined = 0
    flagged_reasons: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()

    internal_seen = 0
    for i, row in enumerate(load_dataset(REPO, "pii-bioes", split="train", streaming=True)):
        if internal_seen >= internal_cap:
            break
        internal_seen += 1
        record = _bioes_row_to_record(dict(row), i)
        if record is None:
            dropped["internal:empty_text"] += 1
            continue
        reasons = _high_severity_reasons(record)
        flagged_reasons.update(reasons)
        hit = [r for r in reasons if r in QUARANTINE_REASONS]
        if hit:
            quarantined += 1
            quarantine_reasons.update(hit)
            continue
        pool.append(record)
        source_rows["meddies-internal"] += 1

    seen_hashes = {text_hash(normalize_text(str(r.get("text") or ""))) for r in pool}

    openpii_rows = load_dataset(OPENPII, split="train", streaming=True)
    selected, openpii_summary = select_openpii_candidates_from_rows(
        openpii_rows,
        quotas=DEFAULT_OPENPII_LANGUAGE_QUOTAS,
        existing_ids=set(),
        existing_text_hashes=seen_hashes,
        dataset_id=OPENPII,
        max_scan=openpii_max_scan,
    )
    for r in selected:
        pool.append(r)
        source_rows["ai4privacy-1.5m"] += 1
        seen_hashes.add(text_hash(normalize_text(str(r.get("text") or ""))))

    def _fold(
        dataset_id: str,
        split_iter: Iterable[Mapping[str, object]],
        convert: RowConverter,
        cap: int,
        tag: str,
    ) -> None:
        kept = 0
        for j, row in enumerate(split_iter):
            if kept >= cap:
                break
            record, drop = convert(dict(row), dataset_id=dataset_id, default_uid=f"{dataset_id}:{j}")
            dropped.update({f"{tag}:{k}": v for k, v in drop.items()})
            if record is None:
                continue
            lang = str((record.get("info") or {}).get("language") or "")
            if not is_supported_meddies_language(lang):
                dropped[f"{tag}:unsupported_language:{lang.lower()}"] += 1
                continue
            digest = text_hash(normalize_text(str(record.get("text") or "")))
            if digest in seen_hashes:
                dropped[f"{tag}:duplicate"] += 1
                continue
            seen_hashes.add(digest)
            pool.append(record)
            source_rows[tag] += 1
            kept += 1

    _fold(
        NEMOTRON,
        load_dataset(NEMOTRON, split="train", streaming=True),
        convert_nemotron_row,
        nemotron_cap,
        "nemotron",
    )
    _fold(
        GRETEL,
        load_dataset(GRETEL, split="train", streaming=True),
        convert_gretel_row,
        gretel_cap,
        "gretel",
    )
    _fold(
        OPENPII_500K,
        load_dataset(OPENPII_500K, split="train", streaming=True),
        convert_ai4privacy_row,
        openpii_500k_cap,
        "openpii-500k",
    )

    natural = summarize_records(pool, dropped_external_spans=dict(dropped))

    carve_note = ""
    try:
        train, heldout = carve_heldout(pool, heldout_rows=heldout_rows, seed=seed)
    except ValueError as exc:
        carve_note = f"carve_heldout failed: {exc}"
        train, heldout = pool, []

    compb_quotas = dict.fromkeys(DEFAULT_OPENPII_LANGUAGE_QUOTAS, 200)
    compb_rows, compb_summary = select_openpii_candidates_from_rows(
        load_dataset(OPENPII, split="validation", streaming=True),
        quotas=compb_quotas,
        existing_ids=set(),
        existing_text_hashes=seen_hashes,
        dataset_id=OPENPII,
        max_scan=min(openpii_max_scan, 200000),
    )

    pathlib.Path(OUT_ROOT).mkdir(exist_ok=True, parents=True)
    for name, rows in (
        ("train", train),
        ("heldout", heldout),
        ("component_b", compb_rows),
    ):
        path = f"{OUT_ROOT}/{name}.jsonl"
        with pathlib.Path(path).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _bucket_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
        from meddies_pii.training.bioes.data.splits import row_language_bucket

        return dict(Counter(row_language_bucket(r) for r in rows))

    result = {
        "pool_rows": len(pool),
        "internal_quarantined_precise": quarantined,
        "internal_quarantine_reasons": dict(quarantine_reasons.most_common()),
        "internal_flagged_all_reasons": dict(flagged_reasons.most_common()),
        "internal_kept": source_rows.get("meddies-internal", 0),
        "source_rows": dict(source_rows.most_common()),
        "natural_label_counts": natural.label_counts,
        "natural_language_bucket_counts": natural.language_bucket_counts,
        "natural_domain_bucket_counts": natural.domain_bucket_counts,
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "train_bucket_counts": _bucket_counts(train),
        "heldout_bucket_counts": _bucket_counts(heldout),
        "balance_target_30_30_40": {"vi": 0.30, "en": 0.30, "other": 0.40},
        "carve_note": carve_note,
        "openpii_selection": openpii_summary,
        "component_b_rows": len(compb_rows),
        "component_b_selection": compb_summary,
        "dropped_top": dict(dropped.most_common(30)),
        "out_dir": OUT_ROOT,
    }
    _VOLUME.commit()
    print("MIX_RESULT::" + json.dumps(result, ensure_ascii=False)[:60000])


# reason: build entry exposes internal cap/seed as its public contract; bundling would break callers.
@app.local_entrypoint()
def build_entry(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    internal_cap: int = 40000,
    nemotron_cap: int = 50000,
    gretel_cap: int = 10000,
    openpii_500k_cap: int = 20000,
    openpii_max_scan: int = 400000,
    heldout_rows: int = 1500,
    seed: int = 17,
) -> None:
    build.remote(
        internal_cap=internal_cap,
        nemotron_cap=nemotron_cap,
        gretel_cap=gretel_cap,
        openpii_500k_cap=openpii_500k_cap,
        openpii_max_scan=openpii_max_scan,
        heldout_rows=heldout_rows,
        seed=seed,
    )
    print("MIX_DONE")
