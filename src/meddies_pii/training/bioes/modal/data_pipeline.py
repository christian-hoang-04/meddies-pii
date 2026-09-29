"""bioes-v2 data pipeline on Modal — download, convert (parallel), gate (parallel).

A persistent Volume holds everything (no local storage). Two fan-outs:
- convert: per-language inline configs + nvidia (label->values) + ai4privacy-1.5m
  each converted in its own container.
- gate: each source file counted in its own container (count_file.map), then the
  per-source CorpusCounts merge instantly and evaluate — the incremental gate, so
  re-gating after adding a source only counts the new one.

Wraps the tested pure modules (inline_tags, grpo_convert.find_spans,
mixed.convert_ai4privacy_row, corpus_gate). Image stays light (datasets + httpx +
tenacity; the trivial text_hash/normalize/constants are inlined to avoid the
transformers transitive import). Profile: openmedical (private dataset).

Run: MODAL_PROFILE=openmedical modal run -m meddies_pii.training.bioes.modal.data_pipeline
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra; importing it at module load would make the package unimportable without
# reason: it.
# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
# ruff: file-ignore[implicit-namespace-package]
# reason: `modal/` is the only subpackage of `bioes/` without an `__init__.py` — assembly, data, eval,
# reason: reports and trainers all have one — so the asymmetry reads as an oversight, and adding the
# reason: file is likely inert under hatchling's src-layout discovery.
# reason: Deliberately deferred rather than fixed here: this directory holds every spend-authorization
# reason: gate, and its Modal-remote import paths have only fake-mediated local coverage, so adding
# reason: `__init__.py` is a post-merge change whose proof is a real GPU smoke run — owner ledger item.
# ruff: file-ignore[import-private-name]
# reason: `_resolve_language` and `_row_labels` are `corpus_gate`'s own resolvers, imported here so the
# reason: reported numbers line up with the gate's by construction; a second copy would be the defect.
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from modal.app import App
from modal.image import Image
from modal.secret import Secret
from modal.volume import Volume

from meddies_pii.historical_artifacts import LEGACY_ARTIFACT_TOKEN
from meddies_pii.json_types import JsonObject, as_json_object

if TYPE_CHECKING:
    from meddies_pii.training.bioes.modal.modal_types import TypedModalApp


class ConversionCounts(TypedDict):
    scanned: int
    kept: int
    dropped_lang: int


class CorpusStats(TypedDict):
    total_rows: int
    total_spans: int
    per_source: dict[str, int]
    per_language: dict[str, int]
    per_label: dict[str, int]
    lang_label_docs: dict[str, int]
    per_format: dict[str, int]
    per_edge_case: dict[str, int]
    span_density: dict[str, int]
    text_len: dict[str, int]
    format_coverage: int
    edge_coverage: int


class CountFileResult(TypedDict):
    path: str
    rows: int
    counts: JsonObject


VOLUME_NAME = "meddies-bioes-v2"
MOUNT = "/data"
REPO_ID = "Meddies/meddies-pii"
AI4PRIVACY_ID = "ai4privacy/pii-masking-openpii-1.5m"
SRC_REMOTE = "/root/src"

INLINE_CONFIGS = (
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
    "vietnamese",
    "vietnamese-translated",
)
NVIDIA_CONFIGS = ("nvidia-health", "nvidia-non-health")
GRPO_HF_CONFIGS = ("grpo-train", "grpo-hard-train")
"""HF grpo configs.

{raw_text, source, answer(label->values JSON)} — convertible by find_spans, ~7k net-new, carry non-vi/en company_name.
Never converted by the per-language path (excluded as "own converter") — folded in here.

"""

image = (
    Image
    .from_registry("python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941")
    .pip_install("datasets==4.5.0", "huggingface-hub==1.19.0", "httpx==0.28.1", "tenacity==9.1.4")
    .add_local_dir("src", remote_path=SRC_REMOTE)
)
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = cast("TypedModalApp", App("meddies-bioes-v2-data", image=image))
HF_SECRET = Secret.from_name("huggingface-secret")


def _require_json_object(value: object, *, source: str) -> JsonObject:
    json_object = as_json_object(value)
    if json_object is None:
        msg = f"{source} produced a non-JSON record"
        raise ValueError(msg)
    return json_object


def _write(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.function(volumes={MOUNT: volume}, secrets=[HF_SECRET], timeout=3600)
def download_configs() -> dict[str, int]:
    from datasets import get_dataset_config_names, load_dataset

    out = Path(MOUNT) / "hf_configs"
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for cfg in get_dataset_config_names(REPO_ID):
        ds = load_dataset(REPO_ID, cfg)
        for split in ds:
            path = out / f"{cfg}.{split}.jsonl"
            ds[split].to_json(str(path), force_ascii=False, lines=True)
            counts[f"{cfg}.{split}"] = int(ds[split].num_rows)
    volume.commit()
    return counts


@app.function(volumes={MOUNT: volume}, timeout=1800)
def convert_inline(cfg: str) -> int:
    """Convert the vietnamese-translated config without dropping unlabeled rows.

    Supply the config language as the fallback so those 10k rows land in the vi cells instead of being dropped as
    "unknown". The 16 standard configs carry their own language ("Tamil", ...) so the fallback is unused there.

    Returns:
        The number of rows written for this config. The count is of rows LANDED, so it
        excludes rows the converter rejected -- comparing it against the source row count is
        the check that the language fallback did its job rather than silently dropping the
        vietnamese-translated rows into "unknown".

    """
    sys.path.insert(0, SRC_REMOTE)
    from meddies_pii.training.bioes.data.inline_tags import convert_config_row

    default_language = "Vietnamese" if cfg == "vietnamese-translated" else cfg
    src = Path(MOUNT) / "hf_configs" / f"{cfg}.train.jsonl"
    recs: list[JsonObject] = []
    with src.open(encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = convert_config_row(
                    json.loads(line),
                    uid=f"{cfg}-{i}",
                    default_language=default_language,
                )
            except json.JSONDecodeError:
                continue
            if rec is not None:
                recs.append(_require_json_object(rec, source="inline converter"))
    _write(
        Path(MOUNT) / "internal/hf_configs" / f"{cfg}.{LEGACY_ARTIFACT_TOKEN}.jsonl",
        recs,
    )
    volume.commit()
    return len(recs)


@app.function(volumes={MOUNT: volume}, timeout=1800)
def convert_nvidia(cfg: str) -> int:
    sys.path.insert(0, SRC_REMOTE)
    from meddies_pii.training.bioes.data.grpo_convert import find_spans

    src = Path(MOUNT) / "hf_configs" / f"{cfg}.train.jsonl"
    recs: list[JsonObject] = []
    with src.open(encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                text = row.get("text") or row.get("raw") or ""
                extractions = json.loads(row.get("label") or "{}")
            except json.JSONDecodeError:
                continue
            spans = find_spans(text, extractions)
            if not spans:
                continue
            recs.append(
                _require_json_object(
                    {
                        "text": text,
                        "label": spans,
                        "info": {
                            "source": cfg,
                            "language": str(row.get("language") or "unknown"),
                            "uid": f"{cfg}-{i}",
                        },
                    },
                    source="NVIDIA converter",
                ),
            )
    _write(Path(MOUNT) / "internal/nvidia" / f"{cfg}.{LEGACY_ARTIFACT_TOKEN}.jsonl", recs)
    volume.commit()
    return len(recs)


@app.function(volumes={MOUNT: volume}, timeout=1800)
def convert_grpo_hf(cfg: str) -> int:
    sys.path.insert(0, SRC_REMOTE)
    from meddies_pii.training.bioes.data.grpo_convert import convert_grpo_hf_row

    src = Path(MOUNT) / "hf_configs" / f"{cfg}.train.jsonl"
    recs: list[JsonObject] = []
    with src.open(encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = convert_grpo_hf_row(row, uid=f"{cfg}-{i}")
            if rec is not None:
                recs.append(_require_json_object(rec, source="GRPO converter"))
    _write(Path(MOUNT) / "internal/grpo" / f"{cfg}.{LEGACY_ARTIFACT_TOKEN}.jsonl", recs)
    volume.commit()
    return len(recs)


@app.function(volumes={MOUNT: volume}, secrets=[HF_SECRET], cpu=4.0, memory=16384, timeout=7200)
def download_convert_ai4privacy(max_rows: int = 0) -> ConversionCounts:
    sys.path.insert(0, SRC_REMOTE)
    from datasets import load_dataset

    from meddies_pii.training.bioes.data.mixed import (
        convert_ai4privacy_row,
        is_supported_meddies_language,
    )

    out = Path(MOUNT) / "internal/ai4privacy"
    out.mkdir(parents=True, exist_ok=True)
    kept = scanned = dropped_lang = 0
    fh = (out / f"ai4privacy-1.5m.{LEGACY_ARTIFACT_TOKEN}.jsonl").open("w", encoding="utf-8")
    for split in ("train", "validation"):
        try:
            ds = load_dataset(AI4PRIVACY_ID, split=split, streaming=True)
        except (ValueError, FileNotFoundError):
            continue
        for row in ds:
            scanned += 1
            if max_rows and scanned > max_rows:
                break
            lang = str(row.get("language") or row.get("locale") or "")
            if not is_supported_meddies_language(lang):
                dropped_lang += 1
                continue
            rec, _ = convert_ai4privacy_row(row, dataset_id=AI4PRIVACY_ID, default_uid=f"ai4-{scanned}")
            if rec is not None:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    fh.close()
    volume.commit()
    return {"scanned": scanned, "kept": kept, "dropped_lang": dropped_lang}


# reason: resolve and row labels share corpus stats's state; extraction would split parity state.
@app.function(volumes={MOUNT: volume}, memory=16384, timeout=3600)
def corpus_stats(  # ruff: ignore[complex-structure,too-many-locals]
    path: str = f"mix/train.{LEGACY_ARTIFACT_TOKEN}.jsonl",
) -> CorpusStats:
    """Scan the assembled mix and return the full data-shape distribution.

    The distribution is
    sources x languages x labels x formats x difficulties x span-density. Drives
    the data-shape HTML report. Language/label resolution reuses the gate's
    resolvers so the numbers line up with the gate.

    Returns:
        The full cross-tabulated distribution for the scanned file. Language and label
        resolution goes through the GATE's own resolvers rather than reading the raw fields,
        which is what makes these numbers comparable with the gate's -- a local
        re-implementation would drift silently and the report would disagree with the gate
        that admitted the corpus.

    """
    sys.path.insert(0, SRC_REMOTE)
    from collections import Counter

    from meddies_pii.generation.text_formats import canonical_text_format
    from meddies_pii.training.bioes.data.corpus_gate import (
        _resolve_language,
        _row_labels,
    )

    def _span_density_bin(n: int) -> str:
        for hi, name in (
            (0, "0"),
            (1, "1"),
            (3, "2-3"),
            (6, "4-6"),
            (10, "7-10"),
            (20, "11-20"),
        ):
            if n <= hi:
                return name
        return "21+"

    def _len_bin(n: int) -> str:
        for hi, name in (
            (200, "0-200"),
            (500, "201-500"),
            (1000, "501-1k"),
            (2000, "1k-2k"),
            (4000, "2k-4k"),
        ):
            if n <= hi:
                return name
        return "4k+"

    per_source: Counter[str] = Counter()
    per_language: Counter[str] = Counter()
    per_label: Counter[str] = Counter()
    lang_label: Counter[str] = Counter()
    per_format: Counter[str] = Counter()
    per_edge: Counter[str] = Counter()
    span_density: Counter[str] = Counter()
    text_len: Counter[str] = Counter()
    total_rows = total_spans = format_known = edge_known = 0

    with (Path(MOUNT) / path).open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            total_rows += 1
            info_value = rec.get("info")
            info: Mapping[str, object] = info_value if isinstance(info_value, Mapping) else {}
            per_source[str(info.get("source") or "unknown")] += 1
            lang = _resolve_language(str(info.get("language") or "")) or "other"
            per_language[lang] += 1
            labels = _row_labels(rec)
            total_spans += len(labels)
            span_density[_span_density_bin(len(labels))] += 1
            text = rec.get("text")
            text_len[_len_bin(len(text) if isinstance(text, str) else 0)] += 1
            for label in set(labels):
                lang_label[f"{lang}|{label}"] += 1
            for label in labels:
                per_label[label] += 1
            fmt = info.get("text_format") or info.get("document_type")
            if isinstance(fmt, str) and fmt:
                per_format[canonical_text_format(fmt)] += 1
                format_known += 1
            edges = info.get("edge_cases")
            if isinstance(edges, list) and edges:
                for edge in edges:
                    if isinstance(edge, str) and edge:
                        per_edge[edge[:60]] += 1
                edge_known += 1

    return {
        "total_rows": total_rows,
        "total_spans": total_spans,
        "per_source": dict(per_source.most_common()),
        "per_language": dict(per_language.most_common()),
        "per_label": dict(per_label.most_common()),
        "lang_label_docs": dict(lang_label),
        "per_format": dict(per_format.most_common()),
        "per_edge_case": dict(per_edge.most_common(40)),
        "span_density": dict(span_density),
        "text_len": dict(text_len),
        "format_coverage": format_known,
        "edge_coverage": edge_known,
    }


@app.function(volumes={MOUNT: volume}, memory=8192, timeout=3600)
def count_file(path: str) -> CountFileResult:
    sys.path.insert(0, SRC_REMOTE)
    from meddies_pii.training.bioes.data.corpus_gate import count_corpus

    rows: list[JsonObject] = []
    with Path(path).open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    json_object = as_json_object(rec)
                    if json_object is not None:
                        rows.append(json_object)
                except json.JSONDecodeError:
                    continue
    counts = count_corpus(rows)
    return {
        "path": Path(path).name,
        "rows": len(rows),
        "counts": _require_json_object(counts.to_json(), source="corpus count"),
    }


def _source_paths(root: Path) -> list[str]:
    """Every assembled-corpus source file on the volume, in a stable order.

    Source discovery is package-owned so local and Modal assembly use the same
    source order and glob policy.

    Returns:
        Every assembled-corpus source file under the root as a string path, in the package's
        stable discovery order. Order is part of the contract rather than incidental: local
        and Modal assembly must see the same sequence, because the assembler's dedup is
        first-wins and a different order would keep a different row.

    """
    sys.path.insert(0, SRC_REMOTE)
    from meddies_pii.training.bioes.assembly import discover_assembly_source_files

    return [str(path) for path in discover_assembly_source_files(root)]


@app.function(volumes={MOUNT: volume}, timeout=600)
def list_sources() -> list[str]:
    return _source_paths(Path(MOUNT))


@app.function(volumes={MOUNT: volume}, memory=32768, timeout=7200)
def assemble_mix(waive_cells: list[list[str]] | None = None) -> JsonObject:
    """Union every source, normalized-dedup, gate, and write ``/data/mix``.

    The shared package implementation owns source discovery, normalized dedup,
    span-quality filtering, gate evaluation, and MANIFEST/mix_recipe output.
    Modal only supplies the mounted root and commits the volume.

    Returns:
        The assembly summary, as a JSON object validated on the way out. The volume is
        committed BEFORE the summary is returned, so a caller holding this result can rely on
        ``/data/mix`` and the MANIFEST already being durable on the volume.

    Raises:
        RuntimeError: When the corpus gate refuses the assembled mix. The gate's own
            ``AssemblyGateError`` is re-raised as its plain ``RuntimeError`` base, so the
            subclass is FLATTENED and a caller cannot select the gate failure by type -- match
            on the message, which is preserved verbatim, or catch ``AssemblyGateError`` at the
            package boundary instead of here. The original is chained. The raise precedes
            ``volume.commit()``, so a refused mix leaves the volume as it was.

    """
    sys.path.insert(0, SRC_REMOTE)
    from meddies_pii.training.bioes.assembly import (
        AssemblyGateError,
        assemble_bioes_v2_corpus,
    )

    waivers = frozenset((c[0], c[1]) for c in (waive_cells or []) if len(c) == 2)  # ruff: ignore[magic-value-comparison] reason: structural arity; a name restates the literal
    try:
        result = assemble_bioes_v2_corpus(Path(MOUNT), waivers=waivers)
    except AssemblyGateError as exc:
        raise RuntimeError(str(exc)) from exc

    volume.commit()
    return _require_json_object(result.to_summary(), source="mix assembly")


# reason: data pipeline exposes skip/waive as its Modal schema; bundling would break callers.
@app.local_entrypoint()
# reason: Modal entrypoint: the signature is the published CLI/remote-call contract; keyword-only would change invocation.
def main(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    skip_download: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    ai4privacy_max: int = 0,
    skip_convert: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    skip_ai4privacy: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    assemble: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
    waive: str = "",
) -> None:
    """Convert every source, then either report the pre-dedup gate or assemble.

    ``assemble`` writes the deduped ``/data/mix`` and gates THAT (authoritative —
    the count-only path below sums per-file counts and so double-counts rows that
    appear in multiple sources, e.g. the overlapping GRPO lineages). ``waive`` is
    a comma-list of ``language:label`` cells to grant an exhaustion-waiver, e.g.
    ``--waive pt:private_url``.

    ai4privacy re-streams 1.6M rows from HF (~30 min); skip when the volume already holds the converted slice and only the
    cheap sources changed.

    """
    sys.path.insert(0, "src")
    from dataclasses import asdict

    from meddies_pii.training.bioes.assembly import (
        DEFAULT_ENTROPY_TARGETS,
        DEFAULT_LABEL_FLOORS,
    )
    from meddies_pii.training.bioes.data.corpus_gate import (
        CorpusCounts,
        evaluate_counts,
        merge_counts,
    )

    if not skip_download:
        print("download:", sum(download_configs.remote().values()), "rows")
    if not skip_convert:
        inline_n = sum(convert_inline.map(INLINE_CONFIGS))
        nvidia_n = sum(convert_nvidia.map(NVIDIA_CONFIGS))
        grpo_n = sum(convert_grpo_hf.map(GRPO_HF_CONFIGS))
        summary = f"inline {inline_n:,} | nvidia {nvidia_n:,} | grpo {grpo_n:,}"
        if not skip_ai4privacy:
            ai4 = download_convert_ai4privacy.remote(ai4privacy_max)
            summary += f" | ai4privacy {ai4['kept']:,} (scanned {ai4['scanned']:,})"
        print(f"converted: {summary}")

    if assemble:
        waive_cells = [pair.split(":", 1) for pair in waive.split(",") if ":" in pair]
        manifest = assemble_mix.remote(waive_cells or None)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    paths = list_sources.remote()
    per_source = list(count_file.map(paths))
    total_rows = sum(r["rows"] for r in per_source)
    merged = merge_counts([CorpusCounts.from_json(r["counts"]) for r in per_source])
    report = evaluate_counts(
        merged,
        label_floors=DEFAULT_LABEL_FLOORS,
        cell_min=1,
        entropy_targets=DEFAULT_ENTROPY_TARGETS,
    )
    print(
        json.dumps(
            {
                "sources": len(paths),
                "total_rows_pre_dedup": total_rows,
                "per_source": {r["path"]: r["rows"] for r in per_source},
                "label_span_counts": dict(merged.label_span_counts),
                "gate_pre_dedup": asdict(report),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


@app.local_entrypoint()
def stats(
    path: str = f"mix/train.{LEGACY_ARTIFACT_TOKEN}.jsonl",
    out: str = "corpus_stats.json",
) -> None:
    """Scan the assembled mix and write the data-shape distribution JSON locally."""
    result = corpus_stats.remote(path)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}: {result['total_rows']:,} rows, {result['total_spans']:,} spans")
