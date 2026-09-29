"""Repair and split helpers for targeted Meddies Labels generation rejects.

The repair surface is intentionally conservative: it only fixes explicit label
markers whose value boundaries are clear. It does not infer missing labels from
unlabeled content.
"""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: these characters sit inside patterns that must MATCH them — the symbol-substitution and
# reason: separator detectors exist to catch confusable punctuation, so normalising one to ASCII
# reason: would silently stop this module detecting that evasion.
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.generation.label_corpus.catalog import _scenario_by_name
from meddies_pii.generation.label_corpus.synthetic import language_paths, write_summary
from meddies_pii.generation.label_corpus.validate import (
    AcceptedRecord,
    accepted_record,
    validate_tagged_document,
)
from meddies_pii.historical_artifacts import LEGACY_REPAIRED_REJECTS_DATASET_ID
from meddies_pii.jsonl import read_jsonl, write_jsonl
from meddies_pii.taxonomy import PiiLabel, label_regex_alt

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

MIN_PHONE_COMPONENT_DIGITS = 3
MIN_REPAIR_VALUE_LENGTH = 2
MAX_REPAIR_VALUE_LENGTH = 260

_LABEL_ALT = label_regex_alt()

_BRACKET_SPACE_RE = re.compile(rf"\[([^\[\]\n]{{1,500}})\]\s+<({_LABEL_ALT})>", re.IGNORECASE)
_MALFORMED_CLOSE_RE = re.compile(rf"\[([^\[\]\n]{{1,500}})\]<({_LABEL_ALT})\]", re.IGNORECASE)
_QUOTED_SUFFIX_RE = re.compile(rf'"([^"\n\[\]]{{2,260}}?)<({_LABEL_ALT})>"', re.IGNORECASE)
_SPLIT_BRACKET_SUFFIX_RE = re.compile(
    rf"((?:\[[^\[\]\n]{{1,80}}\][\s./\-–—]*){{2,8}})<({_LABEL_ALT})>",
    re.IGNORECASE,
)
_LABELED_PHONE_SEQUENCE_RE = re.compile(
    r"((?:\[[^\[\]\n]{1,80}\]\s*<phone_number>[\s./\-–—]*){2,8})",
    re.IGNORECASE,
)
_PHONE_PREFIX_SPAN_RE = re.compile(
    r"(?P<prefix>(?:\+?\(?\d[\d\s()./\-–—]{1,24}))\[(?P<value>[^\[\]\n]{2,80})\]<phone_number>",
    re.IGNORECASE,
)
_URL_SUFFIX_RE = re.compile(
    rf"((?:https?|ftp)://[^\s<>\[\]\"']{{8,260}}(?:\s+(?:token|sig|signature|session|auth|patient|access_key|key|id)=[^\s<>\[\]\"']{{1,120}})?)<({_LABEL_ALT})>",
    re.IGNORECASE,
)
_PIPE_SUFFIX_RE = re.compile(
    rf"(\|\s*)([^\n\r|<>\[\]{{}}\"]{{2,180}}?)<({_LABEL_ALT})>(?=\s*(?:\||\n|\r|$))",
    re.IGNORECASE,
)
_FIELD_SUFFIX_RE = re.compile(
    rf"(^|[\n|])([^\n:<>\[\]{{}}\"]{{1,80}}:\s*)([^\n<>\[\]{{}}\"]{{2,220}}?)<({_LABEL_ALT})>(?=\s*(?:\n|\r|\||$|[),.;]))",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RejectionSplit:
    api_errors: list[dict[str, object]]
    content_rejects: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class RepairRejectedResult:
    api_errors: list[dict[str, object]]
    content_rejects: list[dict[str, object]]
    repaired_candidates: list[AcceptedRecord]
    unrepaired_content_rejects: list[dict[str, object]]


def _record_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _record_error_strings(value: object) -> list[str]:
    return [str(error) for error in value] if isinstance(value, list) else []


def split_rejected_records(records: Iterable[Mapping[str, object]]) -> RejectionSplit:
    api_errors: list[dict[str, object]] = []
    content_rejects: list[dict[str, object]] = []
    for record in records:
        errors = record.get("errors")
        if isinstance(errors, list) and "api_error" in errors:
            api_errors.append(dict(record))
        else:
            content_rejects.append(dict(record))
    return RejectionSplit(api_errors=api_errors, content_rejects=content_rejects)


# reason: repair rejected exposes records/required as its public contract; bundling would break callers.
def repair_rejected_records(  # ruff: ignore[too-many-arguments]
    records: Iterable[Mapping[str, object]],
    *,
    language: str,
    existing_text_hashes: set[str],
    model: str,
    provider: str,
    required_labels: tuple[PiiLabel, ...] = ("private_url", "secret"),
) -> RepairRejectedResult:
    split = split_rejected_records(records)
    repaired_candidates: list[AcceptedRecord] = []
    unrepaired_content_rejects: list[dict[str, object]] = []
    seen_hashes = set(existing_text_hashes)

    for record in split.content_rejects:
        content = record.get("content")
        if not isinstance(content, str) or not content.strip():
            unrepaired_content_rejects.append(record | {"repair_errors": ["empty_content"]})
            continue
        try:
            scenario = _scenario_by_name(str(record.get("scenario")))
        except ValueError:
            unrepaired_content_rejects.append(record | {"repair_errors": ["unknown_scenario"]})
            continue

        repaired_content = repair_inline_tag_format(content)
        validation = validate_tagged_document(
            repaired_content,
            language=language,
            required_labels=required_labels,
        )
        if not validation.ok:
            unrepaired_content_rejects.append(
                record
                | {
                    "repair_errors": list(validation.errors),
                    "repaired_content": repaired_content,
                    "repair_changed": repaired_content != content,
                },
            )
            continue

        text_hash = hashlib.sha256(validation.raw_text.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            unrepaired_content_rejects.append(
                record
                | {
                    "repair_errors": ["duplicate_text"],
                    "repaired_content": repaired_content,
                    "repair_changed": repaired_content != content,
                },
            )
            continue
        seen_hashes.add(text_hash)

        candidate = accepted_record(
            validation=validation,
            language=language,
            scenario=scenario,
            document_type=str(record.get("document_type") or "unknown"),
            text_format=str(record.get("text_format") or "unknown"),
            model=model,
            provider=provider,
            attempt_index=_record_int(record.get("attempt_index")),
        )
        candidate["info"].update({
            "source_dataset": LEGACY_REPAIRED_REJECTS_DATASET_ID,
            "repair_applied": True,
            "repair_changed": repaired_content != content,
            "original_errors": _record_error_strings(record.get("errors")),
        })
        repaired_candidates.append(candidate)

    return RepairRejectedResult(
        api_errors=split.api_errors,
        content_rejects=split.content_rejects,
        repaired_candidates=repaired_candidates,
        unrepaired_content_rejects=unrepaired_content_rejects,
    )


def repair_rejected_artifacts(
    output_dir: str | Path,
    language: str,
    *,
    model: str,
    provider: str,
) -> dict[str, object]:
    output_path = Path(output_dir)
    paths = language_paths(output_path, language)
    code = paths.accepted.stem.split(".")[-1]
    accepted_records = list(read_jsonl(paths.accepted)) if paths.accepted.exists() else []
    rejected_records = list(read_jsonl(paths.rejected)) if paths.rejected.exists() else []
    result = repair_rejected_records(
        rejected_records,
        language=language,
        existing_text_hashes=_text_hashes(accepted_records),
        model=model,
        provider=provider,
    )

    api_errors_path = output_path / f"api_errors.{code}.jsonl"
    content_rejected_path = output_path / f"content_rejected.{code}.jsonl"
    unrepaired_path = output_path / f"unrepaired_content_rejected.{code}.jsonl"
    repaired_path = output_path / f"repaired_candidates.{code}.jsonl"
    summary_path = output_path / f"repair_summary.{code}.json"

    write_jsonl(api_errors_path, result.api_errors)
    write_jsonl(content_rejected_path, result.content_rejects)
    write_jsonl(unrepaired_path, result.unrepaired_content_rejects)
    write_jsonl(repaired_path, [dict(candidate) for candidate in result.repaired_candidates])

    summary = {
        "language": language,
        "accepted_existing_count": len(accepted_records),
        "rejected_original_count": len(rejected_records),
        "api_error_count": len(result.api_errors),
        "content_rejected_count": len(result.content_rejects),
        "repaired_candidate_count": len(result.repaired_candidates),
        "unrepaired_content_rejected_count": len(result.unrepaired_content_rejects),
        "original_rejection_distribution": _error_counts(rejected_records),
        "unrepaired_rejection_distribution": _error_counts(result.unrepaired_content_rejects),
        "paths": {
            "api_errors": str(api_errors_path),
            "content_rejected": str(content_rejected_path),
            "unrepaired_content_rejected": str(unrepaired_path),
            "repaired_candidates": str(repaired_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(paths, language=language)
    return summary


def repair_inline_tag_format(content: str) -> str:
    """Repair explicit Meddies Labels `value<label>` formatting defects.

    This function only uses visible label markers already emitted by the model.
    It wraps the adjacent value; it never invents missing labels.

    Returns:
        The content with each malformed marker rewritten into the canonical
        ``[value]<label>`` form and the label lowercased. Nine substitutions run in a fixed
        order and each one only rewrites text around a marker the model already emitted, so a
        document with no markers comes back unchanged rather than partially annotated.

    """
    repaired = content
    repaired = _BRACKET_SPACE_RE.sub(lambda match: f"[{match.group(1)}]<{match.group(2).lower()}>", repaired)
    repaired = _MALFORMED_CLOSE_RE.sub(lambda match: f"[{match.group(1)}]<{match.group(2).lower()}>", repaired)
    repaired = _SPLIT_BRACKET_SUFFIX_RE.sub(_wrap_split_bracket_value, repaired)
    repaired = _LABELED_PHONE_SEQUENCE_RE.sub(_wrap_labeled_phone_sequence, repaired)
    repaired = _PHONE_PREFIX_SPAN_RE.sub(_wrap_phone_prefix_span, repaired)
    repaired = _QUOTED_SUFFIX_RE.sub(lambda match: f'"[{match.group(1)}]<{match.group(2).lower()}>"', repaired)
    repaired = _URL_SUFFIX_RE.sub(lambda match: f"[{match.group(1)}]<{match.group(2).lower()}>", repaired)
    repaired = _FIELD_SUFFIX_RE.sub(_wrap_field_value, repaired)
    return _PIPE_SUFFIX_RE.sub(_wrap_structured_value, repaired)


def _wrap_split_bracket_value(match: re.Match[str]) -> str:
    original = match.group(1)
    fragments = re.findall(r"\[([^\[\]\n]{1,80})\]", original)
    separator = "-" if "-" in original or "–" in original or "—" in original else " "
    value = separator.join(fragment.strip() for fragment in fragments if fragment.strip())
    if not value:
        return match.group(0)
    return f"[{value}]<{match.group(2).lower()}>"


def _wrap_labeled_phone_sequence(match: re.Match[str]) -> str:
    original = match.group(1)
    fragments = re.findall(r"\[([^\[\]\n]{1,80})\]\s*<phone_number>", original, flags=re.IGNORECASE)
    separator = "-" if "-" in original or "–" in original or "—" in original else " "
    value = separator.join(fragment.strip() for fragment in fragments if fragment.strip())
    if not value:
        return match.group(0)
    return f"[{value}]<phone_number>"


def _wrap_phone_prefix_span(match: re.Match[str]) -> str:
    prefix = match.group("prefix").strip()
    value = match.group("value").strip()
    if (
        sum(ch.isdigit() for ch in prefix) < MIN_PHONE_COMPONENT_DIGITS
        or sum(ch.isdigit() for ch in value) < MIN_PHONE_COMPONENT_DIGITS
    ):
        return match.group(0)
    cleaned_prefix = prefix.strip(" \t\r\n|")
    combined = f"{cleaned_prefix} {value}".strip()
    return f"[{combined}]<phone_number>"


def _wrap_structured_value(match: re.Match[str]) -> str:
    prefix, value, label = match.groups()
    clean_value = _clean_repair_value(value, max_words=18)
    if clean_value is None:
        return match.group(0)
    return _with_original_padding(prefix, value, label, clean_value)


def _wrap_field_value(match: re.Match[str]) -> str:
    line_start, prefix, value, label = match.groups()
    clean_value = _clean_repair_value(value, max_words=24)
    if clean_value is None:
        return match.group(0)
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    return f"{line_start}{prefix}{leading}[{clean_value}]<{label.lower()}>{trailing}"


def _with_original_padding(prefix: str, value: str, label: str, clean_value: str) -> str:
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    return f"{prefix}{leading}[{clean_value}]<{label.lower()}>{trailing}"


def _text_hashes(records: Iterable[Mapping[str, object]]) -> set[str]:
    hashes: set[str] = set()
    for record in records:
        text = record.get("text")
        if isinstance(text, str):
            hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return hashes


def _error_counts(records: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        errors = record.get("errors")
        if isinstance(errors, list):
            counts.update(str(error) for error in errors)
    return dict(sorted(counts.items()))


def _clean_repair_value(value: str, *, max_words: int) -> str | None:
    stripped = value.strip()
    if len(stripped) < MIN_REPAIR_VALUE_LENGTH or len(stripped) > MAX_REPAIR_VALUE_LENGTH:
        return None
    if stripped.count(" ") > max_words:
        return None
    if ":" in stripped:
        return None
    if "<" in stripped or ">" in stripped or "[" in stripped or "]" in stripped:
        return None
    return stripped
