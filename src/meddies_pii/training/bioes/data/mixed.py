"""Build mixed-domain PII-label JSONL records.

Meddies/meddies-pii remains the medical anchor. External PII datasets are
converted into the same PII-label label-list schema for general-domain
coverage.
"""

from __future__ import annotations

# ruff: file-ignore[useless-import-alias]
# reason: an explicit `X as X` re-export, which is this module's published surface: ruff's own
# reason: unsafe fix DELETED five of these once and broke every caller, so the alias stays.
import operator
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from meddies_pii.constants import LABEL_MAP
from meddies_pii.languages import (
    is_supported_meddies_language as is_supported_meddies_language,
)
from meddies_pii.languages import (
    language_bucket,
)
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PII_LABELS, PiiLabel, is_pii_label
from meddies_pii.training.bioes.data.external_records import (
    dedupe_non_overlapping,
    external_record,
    locate_external_span,
)
from meddies_pii.training.bioes.data.source_payloads import parse_serialized_collection

if TYPE_CHECKING:
    from meddies_pii.training.bioes.data.record_schema import NormalizedRecord

MAX_NAME_SPAN_GAP = 3

AI4PRIVACY_DROP_LABELS = frozenset({"age", "gender", "sex", "title", "time"})
AI4PRIVACY_LABEL_PREMAP = {"time": "date"}
"""ai4privacy-only premap.

TIME -> date (TIME sits in the shared AI4PRIVACY_DROP_LABELS; pre-mapping here overrides the drop for ai4privacy only).
TITLE is intentionally NOT premapped — it is handled by the name-component merge below (folded into an adjacent name span,
or dropped when standalone), matching meddies-pii-v2's "Dr. Jane Doe = one human_name span" convention.

"""

AI4PRIVACY_NAME_COMPONENT_LABELS = frozenset({"title", "givenname", "surname"})
"""ai4privacy splits a person name into separate TITLE/GIVENNAME/SURNAME spans.

We merge any adjacent run of these (gap = only spaces/period) into a single human_name span (e.g. "Dr. Jane Doe"); a run
with no given/surname (a standalone title) is dropped. USERNAME/DOCTORNAME are NOT components here — they map to human_name
as their own spans.

"""
NEMOTRON_NAME_COMPONENT_LABELS = frozenset({"first_name", "last_name"})
"""Nemotron-only label aliases the shared LABEL_MAP does not carry.

The 500k schema tags organisations with `ORGANISATIONPLACEHOLDER` (ADR 0008 §4 — the only external company_name source);
pre-mapping here keeps the alias out of the shared LABEL_MAP, which other sources route through. Nemotron splits a person
name into separate first_name/last_name spans. We merge any adjacent run into one human_name span (matching ai4privacy +
meddies-pii-v2); non-adjacent ones (separate structured fields) correctly stay split. Both are name content, so nothing is
ever dropped. user_name is NOT a component — it maps to human_name as its own span.

"""

NEMOTRON_LABEL_PREMAP = {
    "organisationplaceholder": "company_name",
    "organizationplaceholder": "company_name",
    "organisation": "company_name",
    "organization": "company_name",
}

NEMOTRON_HEALTHCARE_SOURCES = frozenset({"Healthcare", "Health", "Healthcare Providers", "Pharmaceuticals"})
"""Derive the 2-value domain_bucket (healthcare/general) from nemotron's industry `domain` tag.

These carry clinical content — patient notes/vitals/MRNs, referral letters, and clinical-trial/pharmacovigilance records
with patient PII. Note `Biotechnology` (corporate/lab/marketing), `Disability`/`Life` (insurance) and `Fitness` (wellness)
are NOT clinical and stay general.

"""

GRETEL_LABEL_PREMAP = {"time": "date"}
"""Gretel-only premap.

The shared drop-set drops bare `time`, but the project decision for gretel is to remove only `private_url` (generic `url`)
and retain everything else — so a gretel `time` value is kept as `date`. Pre-mapping here (not in the shared LABEL_MAP /
drop-set) keeps this scoped to gretel; other sources still drop bare `time`.

"""

GRETEL_HEALTHCARE_SOURCES = frozenset({"healthcare", "healthcare-administration", "pharmaceuticals-biotechnology"})
"""Derive the 2-value domain_bucket (healthcare/general) from gretel's industry `domain` tag.

These three carry clinical content — clinical notes/certificates/ reports, prescription/MRN admin, and drug
inserts/clinical-study records with patient PII; every other gretel industry maps to general.

"""


NEMOTRON_DROP_LABELS = frozenset({
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
})


@dataclass(frozen=True, slots=True)
class MixedBuildSummary:
    rows: int
    source_counts: dict[str, int]
    domain_bucket_counts: dict[str, int]
    language_bucket_counts: dict[str, int]
    label_counts: dict[str, int]
    dropped_external_spans: dict[str, int]

    def asdict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "source_counts": self.source_counts,
            "domain_bucket_counts": self.domain_bucket_counts,
            "language_bucket_counts": self.language_bucket_counts,
            "label_counts": self.label_counts,
            "dropped_external_spans": self.dropped_external_spans,
        }


def map_external_label_to_pii_label(source_label: str) -> PiiLabel | None:
    """Map an external PII label into the PII-label taxonomy.

    Returns:
        The canonical label, or None when the source label is in a drop set, has no mapping,
        or maps to something outside the taxonomy. None means DROP THIS SPAN, not that the
        label is unknown -- the drop sets and the missing-mapping case are deliberately the
        same outcome.

    """
    key = source_label.strip().lower()
    if key in AI4PRIVACY_DROP_LABELS or key in NEMOTRON_DROP_LABELS:
        return None
    if is_pii_label(key):
        return key
    mapped = LABEL_MAP.get(key)
    if mapped is None:
        return None
    label = mapped.strip("<>").lower()
    return label if is_pii_label(label) else None


def _merge_adjacent_name_spans(
    text: str,
    components: list[tuple[int, int, str]],
    content_labels: frozenset[str],
) -> tuple[list[CharSpan], Counter[str]]:
    """Merge adjacent person-name component runs into single human_name spans.

    A run separated only by spaces/periods is one name (e.g. "Dr. Jane Doe" or
    "Jane Doe"). A run containing none of ``content_labels`` (e.g. an ai4privacy
    standalone title) is dropped, not tagged. ``content_labels`` are the labels
    that carry actual name content: ai4privacy {givenname, surname} (title is a
    non-content modifier), Nemotron {first_name, last_name} (all content).

    Returns:
        The merged spans and a count of what was dropped, keyed by reason. A run carrying no
        content label produces no span and one drop entry, so a standalone title is counted
        rather than silently discarded.

    """
    merged: list[CharSpan] = []
    dropped: Counter[str] = Counter()
    if not components:
        return merged, dropped

    ordered = sorted(components, key=operator.itemgetter(0, 1))

    def flush(run: list[tuple[int, int, str]]) -> None:
        if any(label in content_labels for _, _, label in run):
            start = run[0][0]
            end = max(end for _, end, _ in run)
            merged.append(CharSpan(start=start, end=end, text=text[start:end], label="human_name"))
        else:
            for _, _, label in run:
                dropped[label] += 1

    run = [ordered[0]]
    for comp in ordered[1:]:
        gap = text[run[-1][1] : comp[0]]
        if len(gap) <= MAX_NAME_SPAN_GAP and set(gap) <= {" ", "."}:
            run.append(comp)
        else:
            flush(run)
            run = [comp]
    flush(run)
    return merged, dropped


def convert_ai4privacy_row(
    row: Mapping[str, object],
    *,
    dataset_id: str,
    default_uid: str,
) -> tuple[NormalizedRecord | None, Counter[str]]:
    """Convert one ai4privacy privacy_mask row into PII-label JSONL.

    Person-name components are deferred to the adjacency merge (incl. TITLE, which would otherwise be dropped by the shared
    drop-set).

    Returns:
        The converted row and a drop tally, or ``(None, tally)`` when the row is unusable.
        Every rejection is counted rather than raised, so one malformed row costs a tally
        entry instead of the conversion pass.

    """
    text = row.get("source_text")
    masks = row.get("privacy_mask")
    if not isinstance(text, str) or not isinstance(masks, Sequence):
        return None, Counter({"malformed_row": 1})

    other_spans: list[CharSpan] = []
    name_components: list[tuple[int, int, str]] = []
    dropped: Counter[str] = Counter()
    for mask in masks:
        if not isinstance(mask, Mapping):
            dropped["malformed_mask"] += 1
            continue
        raw_label = str(mask.get("label") or "").strip()
        raw_lower = raw_label.lower()
        located = locate_external_span(
            text=text,
            value=str(mask.get("value") or ""),
            start=mask.get("start"),
            end=mask.get("end"),
        )
        if raw_lower in AI4PRIVACY_NAME_COMPONENT_LABELS:
            if located is None:
                dropped[f"{raw_lower}:offset_mismatch"] += 1
                continue
            name_components.append((located[0], located[1], raw_lower))
            continue
        source_label = AI4PRIVACY_LABEL_PREMAP.get(raw_lower, raw_label)
        label = map_external_label_to_pii_label(source_label)
        if label is None:
            dropped[source_label.lower() or "empty_label"] += 1
            continue
        if located is None:
            dropped[f"{source_label.lower()}:offset_mismatch"] += 1
            continue
        other_spans.append(
            CharSpan(
                start=located[0],
                end=located[1],
                text=text[located[0] : located[1]],
                label=label,
            ),
        )

    merged_names, name_dropped = _merge_adjacent_name_spans(text, name_components, frozenset({"givenname", "surname"}))
    dropped.update(name_dropped)
    spans = dedupe_non_overlapping([*merged_names, *other_spans])
    if not spans:
        return None, dropped
    uid = str(row.get("uid") or default_uid)
    language = str(row.get("language") or "UNKNOWN")
    return (
        external_record(
            text=text,
            spans=spans,
            uid=f"{dataset_id}:{uid}",
            source_dataset=dataset_id,
            source=str(row.get("split") or "train"),
            language=language,
            domain_bucket="general",
            language_bucket=language_bucket(language=language, source=""),
        ),
        dropped,
    )


# reason: convert nemotron owns name spans and parse together; splitting would misattribute row errors.
def convert_nemotron_row(  # ruff: ignore[too-many-locals]
    row: Mapping[str, object],
    *,
    dataset_id: str = "nvidia/Nemotron-PII",
    default_uid: str,
) -> tuple[NormalizedRecord | None, Counter[str]]:
    """Convert one Nemotron-PII row into PII-label JSONL.

    Person-name components are deferred to the adjacency merge.

    Returns:
        The converted row and a drop tally, or ``(None, tally)`` when the text or the span
        collection will not parse. Same contract as the ai4privacy converter, so the caller
        can treat the two corpora identically.

    """
    text = row.get("text")
    if not isinstance(text, str):
        return None, Counter({"malformed_row": 1})
    raw_spans = parse_serialized_collection(row.get("spans"))
    if not isinstance(raw_spans, list):
        return None, Counter({"malformed_spans": 1})

    other_spans: list[CharSpan] = []
    name_components: list[tuple[int, int, str]] = []
    dropped: Counter[str] = Counter()
    for span in raw_spans:
        if not isinstance(span, Mapping):
            dropped["malformed_span"] += 1
            continue
        raw_label = str(span.get("label") or "").strip()
        raw_lower = raw_label.lower()
        located = locate_external_span(
            text=text,
            value=str(span.get("text") or ""),
            start=span.get("start"),
            end=span.get("end"),
        )
        if raw_lower in NEMOTRON_NAME_COMPONENT_LABELS:
            if located is None:
                dropped[f"{raw_lower}:offset_mismatch"] += 1
                continue
            name_components.append((located[0], located[1], raw_lower))
            continue
        source_label = NEMOTRON_LABEL_PREMAP.get(raw_lower, raw_label)
        label = map_external_label_to_pii_label(source_label)
        if label is None:
            dropped[source_label.lower() or "empty_label"] += 1
            continue
        if located is None:
            dropped[f"{source_label.lower()}:offset_mismatch"] += 1
            continue
        other_spans.append(
            CharSpan(
                start=located[0],
                end=located[1],
                text=text[located[0] : located[1]],
                label=label,
            ),
        )

    merged_names, name_dropped = _merge_adjacent_name_spans(text, name_components, NEMOTRON_NAME_COMPONENT_LABELS)
    dropped.update(name_dropped)
    spans = dedupe_non_overlapping([*merged_names, *other_spans])
    if not spans:
        return None, dropped
    uid = str(row.get("uid") or default_uid)
    source = str(row.get("domain") or row.get("document_type") or "nemotron")
    locale = str(row.get("locale") or "UNKNOWN")
    return (
        external_record(
            text=text,
            spans=spans,
            uid=f"{dataset_id}:{uid}",
            source_dataset=dataset_id,
            source=source,
            language=locale,
            domain_bucket=("healthcare" if source in NEMOTRON_HEALTHCARE_SOURCES else "general"),
            language_bucket=language_bucket(language=locale, source=source),
        ),
        dropped,
    )


def convert_gretel_row(
    row: Mapping[str, object],
    *,
    dataset_id: str = "gretelai/gretel-pii-masking-en-v1",
    default_uid: str,
) -> tuple[NormalizedRecord | None, Counter[str]]:
    """Convert one gretel-pii-masking-en row into PII-label JSONL.

    gretel's ``entities`` field is ``[{"entity": value, "types": [label]}, ...]``
    where the LABEL lives in ``types`` and ``entity`` holds the value. The
    corpus is English-only and ships no per-entity offsets, so spans are located
    by value via the shared ``locate_external_span`` fallback.

    Returns:
        The converted row and a drop tally, or ``(None, tally)`` when the text or the entity
        collection will not parse. Because offsets are located by value, an entity whose
        value does not occur in the text is a counted drop rather than a wrong span.

    """
    text = row.get("text")
    if not isinstance(text, str):
        return None, Counter({"malformed_row": 1})
    raw_entities = parse_serialized_collection(row.get("entities"))
    if not isinstance(raw_entities, list):
        return None, Counter({"malformed_entities": 1})

    spans: list[CharSpan] = []
    dropped: Counter[str] = Counter()
    for entity in raw_entities:
        if not isinstance(entity, Mapping):
            dropped["malformed_entity"] += 1
            continue
        types = entity.get("types")
        source_label = str(types[0]) if isinstance(types, Sequence) and types else ""
        source_label = GRETEL_LABEL_PREMAP.get(source_label.strip().lower(), source_label)
        label = map_external_label_to_pii_label(source_label)
        if label is None:
            dropped[source_label.lower() or "empty_label"] += 1
            continue
        located = locate_external_span(
            text=text,
            value=str(entity.get("entity") or ""),
            start=entity.get("start"),
            end=entity.get("end"),
        )
        if located is None:
            dropped[f"{source_label.lower()}:offset_mismatch"] += 1
            continue
        spans.append(
            CharSpan(
                start=located[0],
                end=located[1],
                text=text[located[0] : located[1]],
                label=label,
            ),
        )

    spans = dedupe_non_overlapping(spans)
    if not spans:
        return None, dropped
    uid = str(row.get("uid") or default_uid)
    source = str(row.get("domain") or row.get("document_type") or "gretel")
    return (
        external_record(
            text=text,
            spans=spans,
            uid=f"{dataset_id}:{uid}",
            source_dataset=dataset_id,
            source=source,
            language="en",
            domain_bucket=("healthcare" if source in GRETEL_HEALTHCARE_SOURCES else "general"),
            language_bucket=language_bucket(language="en", source=source),
        ),
        dropped,
    )


def summarize_records(
    records: Iterable[Mapping[str, object]],
    *,
    dropped_external_spans: Mapping[str, int] | None = None,
) -> MixedBuildSummary:
    source_counts: Counter[str] = Counter()
    domain_bucket_counts: Counter[str] = Counter()
    language_bucket_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    rows = 0
    for record in records:
        rows += 1
        raw_info = record.get("info")
        info = raw_info if isinstance(raw_info, Mapping) else {}
        source_counts[str(info.get("source_dataset") or info.get("source") or "UNKNOWN")] += 1
        domain_bucket_counts[str(info.get("domain_bucket") or "UNKNOWN")] += 1
        language_bucket_counts[str(info.get("language_bucket") or "other")] += 1
        labels = record.get("label")
        if isinstance(labels, Sequence):
            for label in labels:
                if isinstance(label, Mapping):
                    label_counts[str(label.get("category") or "UNKNOWN")] += 1
    return MixedBuildSummary(
        rows=rows,
        source_counts=dict(source_counts),
        domain_bucket_counts=dict(domain_bucket_counts),
        language_bucket_counts=dict(language_bucket_counts),
        label_counts={label: label_counts.get(label, 0) for label in PII_LABELS},
        dropped_external_spans=dict(dropped_external_spans or {}),
    )
