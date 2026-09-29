"""Rich, drift-detecting corpus manifest for the bioes-v2 training data.

ADR 0008 §6: a lean manifest fails *silent* on drift — a stable external file
mutates and the build still claims to reproduce from a now-different source.
The rich manifest records ``sha256_16`` on every stable external file (not just
the pinned eval gold), a ``generation_run`` pointer on each synthetic entry, and
a top-level ``mix_recipe`` block when the corpus is assembled. ``verify_manifest``
recomputes the recorded hashes and fails *loud* on any mismatch, which is what
makes Decision 1's "reproducible-from-local" claim real rather than decorative.

The schema mirrors ``data/bioes-v2/MANIFEST.json`` (``schema``, ``generated``,
``labels_valid``, ``datasets[]`` with ``id`` / ``role`` / ``sha256_16``) so the
written form stays consistent with the hand-authored catalog.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meddies_pii.taxonomy import PII_LABELS

if TYPE_CHECKING:
    from collections.abc import Mapping

MANIFEST_SCHEMA = "meddies-pii/bioes-v2/manifest@1"
MANIFEST_FILENAME = "MANIFEST.json"
LABELS_VALID = PII_LABELS


def sha256_16(path: Path) -> str:
    """First 16 hex chars of the file's content SHA-256.

    Returns:
        The truncated digest. It is a drift detector for corpus bookkeeping, not an integrity
        guarantee: 16 hex characters is enough to notice a file that changed underneath the
        manifest, and short enough to stay readable in it.

    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:16]


def _normalize_entry(corpus_root: Path, dataset: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(dataset)
    current_path = entry.get("current_path")
    generation_run = entry.get("generation_run")
    should_hash = bool(entry.get("hash_file")) or generation_run is None
    sha = None
    if current_path is not None and should_hash:
        file_path = corpus_root / str(current_path)
        if file_path.is_file():
            sha = sha256_16(file_path)
    entry["sha256_16"] = sha
    if "generation_run" not in entry:
        entry["generation_run"] = generation_run
    entry.pop("hash_file", None)
    return entry


def write_manifest(
    corpus_root: Path,
    *,
    datasets: list[dict[str, Any]],
    mix_recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``corpus_root/MANIFEST.json`` and return the written dict.

    Each dataset entry gains a ``sha256_16`` (computed for stable external files —
    any entry with ``hash_file=True`` or no ``generation_run`` pointer) and a
    ``generation_run`` field (``None`` for non-synthetic entries). When the corpus
    is assembled, ``mix_recipe`` records the ratios, seed, dedup policy, and
    audit-report path.

    Returns:
        The manifest exactly as written to disk, so a caller can assert against the return
        value rather than re-reading the file it just produced.

    """
    root = Path(corpus_root)
    entries = [_normalize_entry(root, dataset) for dataset in datasets]
    generated = datetime.datetime.now(datetime.UTC).date().isoformat()
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "generated": generated,
        "labels_valid": list(LABELS_VALID),
        "datasets": entries,
    }
    if mix_recipe is not None:
        manifest["mix_recipe"] = dict(mix_recipe)
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_manifest(corpus_root: Path) -> list[str]:
    """Recompute recorded hashes; return one finding per drift, ``[]`` if clean.

    Only entries that recorded a ``sha256_16`` are checked — regenerable bulk and
    synthetic entries left unhashed are skipped. A missing file or a changed hash
    each produce one finding naming the entry's ``current_path``.

    Returns:
        One finding string per drifted entry, empty when the corpus matches the manifest. It
        returns findings rather than raising so a caller can report every drift at once; an
        entry that recorded no digest is skipped, which means an empty result proves nothing
        about unhashed entries.

    """
    root = Path(corpus_root)
    manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    findings: list[str] = []
    for entry in manifest.get("datasets", []):
        recorded = entry.get("sha256_16")
        if recorded is None:
            continue
        current_path = entry.get("current_path")
        if current_path is None:
            continue
        file_path = root / str(current_path)
        if not file_path.is_file():
            findings.append(f"{current_path}: hashed file missing (recorded {recorded})")
            continue
        actual = sha256_16(file_path)
        if actual != recorded:
            findings.append(f"{current_path}: sha256_16 drift (recorded {recorded}, found {actual})")
    return findings
