from __future__ import annotations

# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.json_types import is_object_dict
from meddies_pii.languages import language_bucket
from meddies_pii.training.bioes.data.augmentation_artifacts import text_hash
from meddies_pii.training.bioes.data.corpus_gate import evaluate_corpus
from meddies_pii.training.bioes.data.external_records import read_jsonl_records
from meddies_pii.training.bioes.data.mixed import summarize_records
from meddies_pii.training.bioes.data.mixed_sources import (
    load_local_external_rows,
    load_remote_external_rows,
    target_external_count,
)
from meddies_pii.training.bioes.data.record_io import write_jsonl
from meddies_pii.training.bioes.data.splits import normalize_text

DEFAULT_LABEL_FLOORS: dict[str, int] = {
    "company_name": 5000,
    "secret": 3000,
    "private_url": 3000,
}
"""Assembly-gate label floors (ADR 0008 §2).

Report-only unless --enforce-gate; tuned with real yield (build-first). The pre-GPU check that the weak labels
(company_name=0.00, secret=0.18 at Run-1 baseline) actually recovered.

"""

DEFAULT_ENTROPY_TARGETS: dict[str, float] = {"language": 2.5, "label": 1.8}
"""Assembly-gate entropy targets in bits (ADR 0008 §3 "highest" tier).

Enforced only under --enforce-gate; report-only otherwise. H_lang max = log2(17)≈4.09, H_label max = log2(9)≈3.17 — these
sub-maximal targets reflect the deployment prior (vi/en-heavy), not abstract uniformity.

"""


def _info_mapping(record: dict[str, object]) -> dict[str, object]:
    info = record.get("info")
    return {str(key): value for key, value in info.items()} if is_object_dict(info) else {}


def _medical_record(record: dict[str, object], row_index: int) -> dict[str, object]:
    info = _info_mapping(record)
    source = str(info.get("source") or "meddies-pii")
    language = str(info.get("language") or "UNKNOWN")
    info.setdefault("id", f"meddies-pii:{row_index}")
    info.setdefault("source_dataset", "Meddies/meddies-pii")
    info.setdefault("source", source)
    info.setdefault("language", language)
    info["domain_bucket"] = "medical"
    info.setdefault(
        "language_bucket",
        language_bucket(language=language, source=source),
    )
    out = dict(record)
    out["info"] = info
    return out


def _load_medical_rows(path: Path, limit: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    iterator = read_jsonl_records(path)
    if limit is not None:
        iterator = islice(iterator, limit)
    for idx, record in enumerate(iterator):
        rows.append(_medical_record(dict(record), idx))
    return rows


def _bucket_for_record(record: dict[str, object]) -> str:
    info = _info_mapping(record)
    bucket = str(info.get("language_bucket") or "")
    if bucket in {"vi", "en", "other"}:
        return bucket
    return language_bucket(
        language=str(info.get("language") or "UNKNOWN"),
        source=str(info.get("source") or ""),
    )


def _with_mix_id(
    record: dict[str, object],
    *,
    prefix: str,
    index: int,
    bucket: str,
) -> dict[str, object]:
    out = dict(record)
    info = _info_mapping(out)
    original_id = str(info.get("id") or f"{prefix}:{index}")
    info.setdefault("original_id", original_id)
    info["id"] = f"{prefix}:{bucket}:{index:06d}:{original_id}"
    info["language_bucket"] = bucket
    out["info"] = info
    return out


def _sample_bucket(
    pool: list[dict[str, object]],
    *,
    count: int,
    prefix: str,
    bucket: str,
    start_index: int,
) -> list[dict[str, object]]:
    if count <= 0:
        return []
    if not pool:
        msg = f"cannot sample {count} rows for empty bucket {bucket!r}"
        raise ValueError(msg)
    if count > len(pool):
        msg = f"cannot sample {count} unique rows for bucket {bucket!r}; only {len(pool)} rows available"
        raise ValueError(msg)
    chosen: list[dict[str, object]] = []
    for offset in range(count):
        record = pool[offset]
        chosen.append(
            _with_mix_id(
                record,
                prefix=prefix,
                index=start_index + offset,
                bucket=bucket,
            ),
        )
    return chosen


# reason: build targeted owns sample and with mix id together; splitting would misattribute row errors.
def _build_targeted_mix(  # ruff: ignore[complex-structure,too-many-branches,too-many-arguments,too-many-locals]
    *,
    medical_rows: list[dict[str, object]],
    external_rows: list[dict[str, object]],
    target_rows: int,
    medical_ratio: float,
    vi_ratio: float,
    en_ratio: float,
    seed: int,
    allow_underfill: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Prefer Vietnamese when extra medical capacity remains, then English, then other.

    This matches the current product priority.

    External rows overfilled one or more language targets. Reduce medical non-Vietnamese first; only reduce Vietnamese as a
    last resort.

    Returns:
        The selected rows and the per-language counts actually achieved. When targets overfill,
        the trim order is medical non-Vietnamese first and Vietnamese last, so the language the
        product cares most about is the one least likely to be cut.

    Raises:
        ValueError: If the external rows do not number exactly the external target. The two
            ratios partition a fixed row budget, so a mismatch means the caller computed a
            different split than this function is about to apply, and proceeding would produce
            a mix whose stated ratios do not describe it.

    """
    # reason: the medical/external mix must be reproducible from the seed the manifest records, or a corpus cannot
    # reason: be rebuilt from its own provenance. Reproducibility is the contract; this interleaves training rows,
    # reason: never a secret, a token, or a key.
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    medical_target = round(target_rows * medical_ratio)
    external_target = target_rows - medical_target
    if len(external_rows) != external_target:
        msg = f"expected {external_target} external rows, got {len(external_rows)}"
        raise ValueError(msg)

    desired = {
        "vi": round(target_rows * vi_ratio),
        "en": round(target_rows * en_ratio),
    }
    desired["other"] = target_rows - desired["vi"] - desired["en"]

    external_counts = Counter(_bucket_for_record(row) for row in external_rows)
    medical_targets = {
        bucket: max(0, desired[bucket] - external_counts.get(bucket, 0)) for bucket in ("vi", "en", "other")
    }
    delta = medical_target - sum(medical_targets.values())
    if delta > 0:
        for bucket in ("vi", "en", "other"):
            if delta <= 0:
                break
            medical_targets[bucket] += delta
            delta = 0
    elif delta < 0:
        for bucket in ("other", "en", "vi"):
            if delta == 0:
                break
            removable = min(medical_targets[bucket], -delta)
            medical_targets[bucket] -= removable
            delta += removable
        if delta != 0:
            msg = f"could not reconcile medical language targets: {medical_targets}"
            raise ValueError(msg)

    pools: dict[str, list[dict[str, object]]] = {"vi": [], "en": [], "other": []}
    for row in medical_rows:
        pools[_bucket_for_record(row)].append(row)
    for bucket_rows in pools.values():
        rng.shuffle(bucket_rows)

    shortfalls: dict[str, int] = {}
    if allow_underfill:
        for bucket in ("vi", "en", "other"):
            available = len(pools[bucket])
            target = medical_targets[bucket]
            if target > available:
                shortfalls[bucket] = target - available
                medical_targets[bucket] = available
    elif any(medical_targets[bucket] > len(pools[bucket]) for bucket in ("vi", "en", "other")):
        details = {
            bucket: {
                "target": medical_targets[bucket],
                "available": len(pools[bucket]),
            }
            for bucket in ("vi", "en", "other")
            if medical_targets[bucket] > len(pools[bucket])
        }
        msg = (
            "target mix requires duplicated/oversampled medical rows; "
            f"lower --target-rows or pass --allow-underfill. shortfalls={details}"
        )
        raise ValueError(
            msg,
        )

    sampled_medical: list[dict[str, object]] = []
    cursor = 0
    for bucket in ("vi", "en", "other"):
        bucket_rows = _sample_bucket(
            pools[bucket],
            count=medical_targets[bucket],
            prefix="mixed-medical",
            bucket=bucket,
            start_index=cursor,
        )
        cursor += len(bucket_rows)
        sampled_medical.extend(bucket_rows)

    external_with_ids = [
        _with_mix_id(
            row,
            prefix="mixed-general",
            index=index,
            bucket=_bucket_for_record(row),
        )
        for index, row in enumerate(external_rows)
    ]
    rows = [*sampled_medical, *external_with_ids]
    rng.shuffle(rows)
    return rows, shortfalls


def _dedupe_records_by_text(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    """Normalize.

    Strip/lowercase/collapse-whitespace) before hashing so a near-duplicate eval row cannot leak into training via a
    whitespace/case variant — same key used by openpii_candidates + splits (ADR 0008 §1/§4).

    Returns:
        The surviving rows in first-seen order, and how many were dropped. First-seen wins, so
        the earliest source in assembly order owns a duplicated document. The key is the
        normalized text, which is why this catches a near-duplicate that differs only in case
        or whitespace -- exactly the variant that would otherwise leak an eval row into
        training and inflate the score.

    """
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    dropped = 0
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            dropped += 1
            continue
        digest = text_hash(normalize_text(text))
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        out.append(row)
    return out, dropped


def _drop_empty_label_records(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    out: list[dict[str, object]] = []
    dropped = 0
    for row in rows:
        labels = row.get("label")
        if isinstance(labels, list) and labels:
            out.append(row)
        else:
            dropped += 1
    return out, dropped


# reason: medical rows and targeted mix share mixed build main's state; extraction would misattribute row errors.
def main() -> None:  # ruff: ignore[too-many-statements]
    """Local-first (ADR 0008 §1).

    Read the centralized corpus, then top up the shortfall from HuggingFace so the build is reproducible from local files
    when they cover the target and degrades gracefully when they do not.

    ADR 0008 §2/§3 assembly gate: label floors + language x label coverage + entropy on the finalized mix. Report-only
    unless --enforce-gate (build first, tune the floors/targets with real yield).

    Raises:
        SystemExit: Through ``argparse`` on a malformed command line, and with a non-zero
            status when ``--enforce-gate`` is set and the assembly gate blocks. Without that
            flag a blocked gate is reported and the build still writes, which is the intended
            default: the floors are meant to be tuned against real yield before they gate.

    """
    parser = argparse.ArgumentParser(
        description="Build a mixed Meddies Labels JSONL bundle: medical anchor plus general-domain PII.",
    )
    parser.add_argument(
        "--medical-jsonl",
        type=Path,
        default=Path(f"data/bioes-v2/base/all_lang_unique_augmented-20260530.train.{LEGACY_ARTIFACT_TOKEN}.jsonl"),
    )
    parser.add_argument("--medical-limit", type=int)
    parser.add_argument("--target-rows", type=int)
    parser.add_argument("--medical-ratio", type=float, default=0.70)
    parser.add_argument("--vi-ratio", type=float, default=0.30)
    parser.add_argument("--en-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--allow-underfill",
        action="store_true",
        help=(
            "Do not duplicate rows when target language/domain quotas exceed unique data; emit the "
            "largest no-duplicate bundle instead."
        ),
    )
    parser.add_argument(
        "--no-dedupe-text",
        action="store_true",
        help="Keep exact duplicate text rows if they occur naturally in source datasets.",
    )
    parser.add_argument(
        "--keep-empty-label-rows",
        action="store_true",
        help="Keep rows with no mapped Meddies Labels spans. Training source selection skips them by default.",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("data/bioes-v2/external"),
        help="Local-first external corpus dir read before the HF fallback (ADR 0008 §1).",
    )
    parser.add_argument("--external-count", type=int)
    parser.add_argument("--external-split", default="train")
    parser.add_argument("--max-scan-per-dataset", type=int, default=200_000)
    parser.add_argument(
        "--allow-unsupported-external-languages",
        action="store_true",
        help=(
            "Allow external rows whose language is outside the Meddies supported-language set. "
            "By default external rows are restricted to Meddies-supported languages to avoid "
            "creating a mixed-language spaghetti bucket."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"data/bioes-v2/mix/train.{LEGACY_ARTIFACT_TOKEN}.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(f"data/bioes-v2/mix/train.{LEGACY_ARTIFACT_TOKEN}.summary.json"),
    )
    parser.add_argument(
        "--cell-min",
        type=int,
        default=1,
        help="Minimum docs per language x label cell for the assembly gate (ADR 0008 §3). Default 1 (no zero cells).",
    )
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help=(
            "Make a failed assembly gate (label floors / cell coverage / entropy) exit non-zero. "
            "Default: report-only (build-first)."
        ),
    )
    args = parser.parse_args()

    medical_rows = _load_medical_rows(args.medical_jsonl, args.medical_limit)
    if args.target_rows is not None:
        target_external = args.target_rows - round(args.target_rows * args.medical_ratio)
    else:
        target_external = (
            args.external_count
            if args.external_count is not None
            else target_external_count(len(medical_rows), args.medical_ratio)
        )
    external_rows, dropped = load_local_external_rows(args.external_dir)
    if len(external_rows) < target_external:
        hf_rows, hf_dropped = load_remote_external_rows(
            target_external=target_external - len(external_rows),
            split=args.external_split,
            max_scan_per_dataset=args.max_scan_per_dataset,
            supported_languages_only=not args.allow_unsupported_external_languages,
        )
        external_rows.extend(hf_rows)
        dropped.update(hf_dropped)
    if args.target_rows is not None:
        rows, language_shortfalls = _build_targeted_mix(
            medical_rows=medical_rows,
            external_rows=external_rows,
            target_rows=args.target_rows,
            medical_ratio=args.medical_ratio,
            vi_ratio=args.vi_ratio,
            en_ratio=args.en_ratio,
            seed=args.seed,
            allow_underfill=args.allow_underfill,
        )
    else:
        rows = [*medical_rows, *external_rows]
        language_shortfalls = {}
    duplicate_text_dropped = 0
    if not args.no_dedupe_text:
        rows, duplicate_text_dropped = _dedupe_records_by_text(rows)
    empty_label_rows_dropped = 0
    if not args.keep_empty_label_rows:
        rows, empty_label_rows_dropped = _drop_empty_label_records(rows)
    write_jsonl(args.output, rows)
    gate = evaluate_corpus(
        rows,
        label_floors=DEFAULT_LABEL_FLOORS,
        cell_min=args.cell_min,
        entropy_targets=DEFAULT_ENTROPY_TARGETS,
    )
    summary = summarize_records(rows, dropped_external_spans=dropped).asdict()
    summary["duplicate_text_dropped"] = duplicate_text_dropped
    summary["empty_label_rows_dropped"] = empty_label_rows_dropped
    summary["language_shortfalls"] = language_shortfalls
    summary["assembly_gate"] = asdict(gate)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.enforce_gate and not gate.passed:
        raise SystemExit("Assembly gate BLOCKED (ADR 0008 §2/§3): " + "; ".join(gate.blocking_reasons))


if __name__ == "__main__":
    main()
