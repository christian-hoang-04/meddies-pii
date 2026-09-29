from __future__ import annotations

import json
import re
from collections.abc import Mapping
from re import Match
from typing import cast

from meddies_pii.annotations.label_aliases import INVALID_LABELS, LABEL_MAP
from meddies_pii.taxonomy import PII_LABEL_SET, PII_LABELS_BRACKETED

BRACKETED_LABEL_MIN_LENGTH = 3
UNBRACKETED_LABEL_MIN_LENGTH = 4

TAG_PATTERN = re.compile(r"\[([^\]]*)\]\s*<([a-zA-Z][^>]*)>")


def strip_pii_tags(text: str) -> str:
    return TAG_PATTERN.sub(r"\1", text)


def strip_code_blocks(content: str) -> str:
    content = re.sub(r"^```[a-zA-Z]*\s*\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)
    content = content.replace("```", "")
    return content.strip()


# reason: fix labels owns replace and groups together; splitting would leak shared intermediate state.
def fix_labels(text: str) -> str:  # ruff: ignore[complex-structure]
    result = re.sub(r"\[([^\[\]]+)\)</[^>]*>?", r"\1", text)

    xml_pattern = r"<([a-zA-Z][a-zA-Z0-9_-]*)>([^<]*)</\1>"
    xml_matches = re.findall(xml_pattern, text)
    xml_placeholders: dict[str, str] = {}
    for i, (tag, content) in enumerate(xml_matches):
        placeholder = f"__XML_{i}__"
        xml_placeholders[placeholder] = f"<{tag}>{content}</{tag}>"
        result = result.replace(f"<{tag}>{content}</{tag}>", placeholder, 1)

    result = re.sub(r"</[a-zA-Z][a-zA-Z0-9_-]*>", "", result)
    result = re.sub(r"\[([^\[\]]+)\)<\([0-9,]+\)>", r"\1", result)
    result = re.sub(r"\[([^\[\]]+)\)<([a-z_]+)\)", r"[\1]<\2>", result)
    result = re.sub(r"\[([^\[\]]+)\)<\n([a-z_]+)>", r"[\1]<\2>", result)
    result = re.sub(r"\[([^\[\]]+)\)<([a-z_]+)\^[^>]+>", r"[\1]<\2>", result)
    result = re.sub(r"\[([^\[\]]+)\)<[a-z_]+\][^>]*>", r"\1", result)

    for inv in INVALID_LABELS:
        result = re.sub(rf"<{inv}>\]?", "", result)

    def fix_bracketed(match: Match[str]) -> str:
        entity = cast("str", match.group(1))
        label_text = cast("str", match.group(2))
        entity = entity.rstrip("]")
        normalized_label = label_text.lower().strip()
        if normalized_label in INVALID_LABELS:
            return entity
        if normalized_label in LABEL_MAP:
            return f"[{entity}]{LABEL_MAP[normalized_label]}"
        if normalized_label in PII_LABEL_SET:
            return f"[{entity}]<{normalized_label}>"
        for v in PII_LABEL_SET:
            if len(normalized_label) >= BRACKETED_LABEL_MIN_LENGTH and (normalized_label in v or v in normalized_label):
                return f"[{entity}]<{v}>"
        return entity

    result = re.sub(r"\[(.+?)<([a-zA-Z][a-zA-Z0-9_-]*)>", fix_bracketed, result)

    def fix_no_brackets(match: Match[str]) -> str:
        entity = cast("str", match.group(1))
        label_text = cast("str", match.group(2))
        normalized_label = label_text.lower().strip()
        if normalized_label in INVALID_LABELS:
            return entity
        if normalized_label in LABEL_MAP:
            return f"[{entity}]{LABEL_MAP[normalized_label]}"
        if normalized_label in PII_LABEL_SET:
            return f"[{entity}]<{normalized_label}>"
        for v in PII_LABEL_SET:
            if len(normalized_label) >= UNBRACKETED_LABEL_MIN_LENGTH and (normalized_label in v or v in normalized_label):
                return f"[{entity}]<{v}>"
        return entity + label_text

    result = re.sub(r"\[([^\[\]]+)\]([a-zA-Z][a-zA-Z0-9_]*)", fix_no_brackets, result)

    def strip_orphan(match: Match[str]) -> str:
        return cast("str", match.group(1)) + cast("str", match.group(2))

    result = re.sub(r"\[([^\[\]]+)\](?!\s*<[a-z_]+>)(\s+|$|[^a-zA-Z<])", strip_orphan, result)

    for placeholder, xml_tag in xml_placeholders.items():
        result = result.replace(placeholder, xml_tag)

    return result


def extract_pii(text: str) -> dict[str, list[str]]:
    if "<" in text and ">" in text:
        pattern = r"\[([^\[\]]+)\]<([a-zA-Z][a-zA-Z0-9_-]*)>"
    else:
        pattern = r"\[([^\[\]]+)\]([a-zA-Z][a-zA-Z0-9_-]*)"
    matches = re.findall(pattern, text)
    pii: dict[str, list[str]] = {}
    for entity, label in matches:
        full_label = f"<{label.lower()}>"
        if full_label in PII_LABELS_BRACKETED:
            l_key = label.lower()
            if l_key not in pii:
                pii[l_key] = []
            if entity not in pii[l_key]:
                pii[l_key].append(entity)
    return {k: pii[k] for k in sorted(pii.keys())}


def extract_entities(text: str) -> set[tuple[str, str]]:
    try:
        data = json.loads(text) if isinstance(text, str) else text
    except (json.JSONDecodeError, TypeError):
        return set()
    if not isinstance(data, Mapping):
        return set()
    entities: set[tuple[str, str]] = set()
    for label, values in data.items():
        if not isinstance(label, str) or not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str):
                entities.add((value, label))
    return entities


def find_tags(content: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2)) for match in TAG_PATTERN.finditer(content)]


def process_row(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    content_key = "text_tagged" if "text_tagged" in result else ("content" if "content" in result else "output")
    content = result.get(content_key)
    if isinstance(content, str):
        fixed_content = fix_labels(content)
        result[content_key] = fixed_content
        result["text"] = json.dumps(extract_pii(fixed_content), ensure_ascii=False)
    return result
