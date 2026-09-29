"""Score-time ignore-list for audited unlabeled-gold spans.

Pinned gold revisions never mutate, so a span an audit finds genuinely present in
the source text but missing from gold cannot be fixed in the dataset. A correct
detection of such a span is otherwise scored as a false positive forever. This
module lets the release gate exclude an audited span from false-positive counting
at score time — keyed to its exact audited coordinates and text, never to its text
alone, so a recurring company name is exempted only where it was actually audited.

An excluded span is removed from the false-positive bucket and its precision
denominator. It can never become a true positive: exclusion is only evaluated
against predictions that already fell outside the gold-matched set.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.eval_baseline.baseline.datasets import V2_EVAL_CONFIGS, load_v2_eval_rows
from meddies_pii.evaluation.identity import canonical_sha256, file_sha256
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PII_LABEL_SET

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_ENTRY_FIELDS = ("split", "uid", "start", "end", "text", "category")
DEFAULT_AUDIT_TIERS = frozenset({"A"})

AUDIT_POOL_PROVENANCE_NOTE = (
    "Audit pool provenance: entries were surfaced via the cue-lexicon and "
    "model-assisted channels, not by running the regex lane against itself. "
    "Cue-channel entries correlate with the org-prefix pack's own cue list, so "
    "an exempted span and a pack-detected span can share the cue that surfaced "
    "them. The standing mitigation is the per-view fp_ignored totals the gate "
    "reports alongside every verdict — a growing ignored-FP share on one view "
    "relative to the other is the signal to re-audit, not silence."
)
"""Recorded verbatim in every provenance sidecar so a reader of the pinned artifact sees the bias caveat without
cross-referencing the audit brief.
"""


@dataclass(frozen=True, slots=True)
class IgnoreListEntry:
    """One audited coordinate exempted from false-positive counting.

    Keyed by (split, uid, start, end, text, category) so a recurring name is
    exempted only at its audited mention, never wherever the same text recurs.
    """

    split: str
    uid: str
    start: int
    end: int
    text: str
    category: str

    def __post_init__(self) -> None:
        if not self.split or not self.uid or not self.text or not self.category:
            msg = "ignore-list entry requires split, uid, text, and category"
            raise ValueError(msg)
        if self.start < 0 or self.end <= self.start:
            msg = "ignore-list entry requires a positive span"
            raise ValueError(msg)

    def to_payload(self) -> dict[str, object]:
        return {
            "split": self.split,
            "uid": self.uid,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class RegexIgnoreList:
    entries: tuple[IgnoreListEntry, ...]
    sha256: str


def ignore_list_from_rows(rows: Iterable[Mapping[str, object]]) -> RegexIgnoreList:
    """Build a content-addressed ignore-list from raw entry rows.

    Refuses any row missing a key field rather than silently dropping it, so a
    truncated or malformed artifact fails loudly instead of under-exempting.

    Returns:
        The validated, content-addressed regex ignore-list.

    """
    entries = tuple(_entry_from_row(row) for row in rows)
    return RegexIgnoreList(entries=entries, sha256=_entries_sha256(entries))


def load_ignore_list_jsonl(path: str | Path) -> RegexIgnoreList:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return ignore_list_from_rows(rows)


def write_ignore_list_jsonl(path: str | Path, ignore_list: RegexIgnoreList) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(entry.to_payload(), ensure_ascii=False, sort_keys=True) + "\n" for entry in ignore_list.entries
        ),
        encoding="utf-8",
    )
    return target


def write_ignore_list_provenance(
    path: str | Path,
    *,
    ignore_list: RegexIgnoreList,
    source_proposal_path: str | Path,
    tiers: frozenset[str],
) -> Path:
    """Record where a derived ignore-list artifact came from, next to it.

    Rides into the context repo unmodified at promotion time, so it must be
    self-contained: a reader needs no other file to see the source, the tier
    filter applied, how many entries resulted, and the pool-provenance caveat.

    Returns:
        The path of the persisted provenance sidecar.

    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_proposal_path": str(source_proposal_path),
        "source_proposal_sha256": file_sha256(str(source_proposal_path)),
        "tier_filter": sorted(tiers),
        "entry_count": len(ignore_list.entries),
        "ignore_list_sha256": ignore_list.sha256,
        "pool_provenance_note": AUDIT_POOL_PROVENANCE_NOTE,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def pinned_document_uids(
    configs: Iterable[str] = V2_EVAL_CONFIGS,
) -> dict[tuple[str, str], str]:
    """Map each pinned document's dataset uid to the uid the scorer keys by.

    The scorer receives `EvalRow.stable_id` (dataset:shard:doc_id), while an
    audit proposal names the bare `doc_id`. Resolving one to the other needs the
    row's dataset and shard, so it has to come from the pinned parquets.

    Returns:
        A mapping from each pinned `(split, row_id)` pair to its scorer UID.

    """
    return {(config, row.doc_id): row.stable_id for config in configs for row in load_v2_eval_rows(config)}


def ignore_list_from_audit_proposal(
    rows: Iterable[Mapping[str, object]],
    *,
    tiers: frozenset[str] = DEFAULT_AUDIT_TIERS,
    document_uids: Mapping[tuple[str, str], str],
) -> RegexIgnoreList:
    """Convert org-gold-audit proposal rows into the ignore-list entry format.

    Proposal rows carry `split`, `uid`, `start`, `end`, `text`, `proposed_category`,
    and `tier`. Only the requested audit tiers (tier A by default) become entries;
    lower tiers stay excluded from scoring exemptions until independently promoted.

    `document_uids` maps (split, proposal uid) to the uid the scorer presents at
    lookup time. It is required rather than defaulted because a proposal uid and
    a scorer uid are both plain strings: a mis-keyed entry raises nothing, it
    simply never matches, and the run reports zero exclusions as if the audit had
    found nothing.

    Returns:
        The validated ignore-list derived from the audit proposal.

    Raises:
        ValueError: If a proposal row, decision, span, or pinned document identity is invalid.

    """
    kept = [row for row in rows if row.get("tier") in tiers]
    unresolved = sorted(
        f"{row.get('split')}:{row.get('uid')}"
        for row in kept
        if (str(row.get("split") or ""), str(row.get("uid") or "")) not in document_uids
    )
    if unresolved:
        msg = f"audit proposal names {len(unresolved)} documents outside the pinned split, first {unresolved[0]}"
        raise ValueError(msg)
    return ignore_list_from_rows(
        {
            "split": row.get("split"),
            "uid": document_uids[str(row.get("split") or ""), str(row.get("uid") or "")],
            "start": row.get("start"),
            "end": row.get("end"),
            "text": row.get("text"),
            "category": row.get("proposed_category"),
        }
        for row in kept
    )


def count_ignored_false_positives(
    predicted: Sequence[CharSpan],
    gold: Sequence[CharSpan],
    ignore_list: RegexIgnoreList | None,
    *,
    split: str,
    uid: str,
) -> int:
    """Count predicted spans that are false positives and match the ignore-list.

    Mirrors the exact-span scorer's own tp/fp partition: a predicted span is
    consumed against the gold multiset first, and only a span left unconsumed
    (a genuine false positive) is ever eligible for exclusion. A span that
    matches gold is never touched, so recall and gold counts are unaffected and
    an excluded span can never become a true positive.

    Entries are consumed as a multiset too: one audited coordinate excludes
    exactly one predicted false positive. N identical predicted spans against a
    single matching entry leave N-1 of them scored as ordinary false positives.

    Returns:
        The number of false-positive spans matched by the ignore-list.

    """
    if not ignore_list or not ignore_list.entries:
        return 0
    remaining_entries = Counter(
        (entry.start, entry.end, entry.text, entry.category)
        for entry in ignore_list.entries
        if entry.split == split and entry.uid == uid
    )
    if not remaining_entries:
        return 0
    remaining_gold = Counter((span.label, span.start, span.end) for span in gold)
    ignored = 0
    for span in predicted:
        gold_key = (span.label, span.start, span.end)
        if remaining_gold[gold_key] > 0:
            remaining_gold[gold_key] -= 1
            continue
        entry_key = (span.start, span.end, span.text, span.label)
        if remaining_entries[entry_key] > 0:
            remaining_entries[entry_key] -= 1
            ignored += 1
    return ignored


def audited_spans_for_row(ignore_list: RegexIgnoreList | None, *, split: str, uid: str) -> tuple[CharSpan, ...]:
    """Return exact audited spans belonging to one split and row.

    Returns:
        The audited spans belonging to the requested split and row.

    """
    if ignore_list is None:
        return ()
    return tuple(
        CharSpan(entry.start, entry.end, entry.text, entry.category)
        for entry in ignore_list.entries
        if entry.split == split and entry.uid == uid
    )


def _entry_from_row(row: Mapping[str, object]) -> IgnoreListEntry:
    missing = [field for field in _ENTRY_FIELDS if row.get(field) is None]
    if missing:
        msg = f"ignore-list entry is missing required fields: {missing}"
        raise ValueError(msg)
    split = row["split"]
    uid = row["uid"]
    start = row["start"]
    end = row["end"]
    text = row["text"]
    category = row["category"]
    # reason: all six fields form one ignore-list row-schema predicate with one failure contract.
    if (
        not isinstance(split, str)  # ruff: ignore[too-many-boolean-expressions]
        or not isinstance(uid, str)
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(text, str)
        or not isinstance(category, str)
    ):
        msg = "ignore-list entry fields have the wrong type"
        # reason: all invalid ignore-list record shapes and values share one schema-validation exception contract.
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    if split not in V2_EVAL_CONFIGS:
        msg = f"ignore-list entry split must be one of {V2_EVAL_CONFIGS}"
        raise ValueError(msg)
    if category not in PII_LABEL_SET:
        msg = f"ignore-list entry category is not a known PII label: {category!r}"
        raise ValueError(msg)
    return IgnoreListEntry(split=split, uid=uid, start=start, end=end, text=text, category=category)


def _entries_sha256(entries: tuple[IgnoreListEntry, ...]) -> str:
    ordered = sorted(
        entries,
        key=lambda entry: (
            entry.split,
            entry.uid,
            entry.start,
            entry.end,
            entry.text,
            entry.category,
        ),
    )
    return canonical_sha256({"schema_version": 1, "entries": [entry.to_payload() for entry in ordered]})
