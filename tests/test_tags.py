import json
import os
import random
import re
from functools import cache

import pytest
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from dotenv import dotenv_values

from meddies_pii.json_types import is_str_mapping
from meddies_pii.tags import extract_pii, fix_labels, process_row
from meddies_pii.taxonomy import PII_LABELS_BRACKETED


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or dotenv_values().get("HF_TOKEN")


# reason: the download is the reason for the memo, and `cache` gives the same one-download-per-session
# reason: behaviour the module global gave, including retrying after a failure, since a raise or a
# reason: `pytest.skip` is not cached.
@cache
def load_dataset_rows() -> list[dict[str, object]]:
    token = _hf_token()
    try:
        ds = load_dataset("Meddies/vie-pii", token=token)
    except DatasetNotFoundError as exc:
        if "gated dataset" in str(exc).lower() or "authenticated" in str(exc).lower():
            pytest.skip("Meddies/vie-pii gated dataset is unavailable; set a valid HF_TOKEN to run dataset contract tests")
        raise
    rows: list[dict[str, object]] = []
    for split in ds:
        rows.extend(dict(row) for row in ds[split] if is_str_mapping(row))
    return rows


def _string_field(row: dict[str, object], key: str) -> str:
    """Return a processed-row field as the string the assertions treat it as.

    `process_row` returns `dict[str, object]`, so `text` and `output` arrive opaque and cannot
    be handed to `json.loads` or the tag regexes until the shape is stated.

    Returns:
        The named field as a string, defaulting to the empty string when it is absent.

    """
    value = row.get(key, "")
    assert isinstance(value, str), f"{key} must be a string, got {type(value).__name__}"
    return value


def test_label_coverage() -> None:
    rows = load_dataset_rows()
    found_labels = set()
    for _ in range(3):
        samples = random.sample(rows, min(100, len(rows)))
        for row in samples:
            processed = process_row(row)
            text = _string_field(processed, "text")
            pii = json.loads(text) if text else {}
            found_labels.update(f"<{label}>" for label in pii)
    assert len(found_labels) >= 5, f"Too few labels found: {found_labels}"


def test_format_validity() -> None:
    rows = load_dataset_rows()
    for i, row in enumerate(rows):
        processed = process_row(row)
        output = _string_field(processed, "output")
        if not output:
            continue
        pii_tags = re.findall(r"\[([^\[\]]+)\](<[^>]+>|[a-zA-Z_]+)", output)
        for entity, label_part in pii_tags:
            if label_part.startswith("<") and label_part.endswith(">"):
                assert label_part in PII_LABELS_BRACKETED, f"Row {i}: Invalid label {label_part}"
            else:
                msg = f"Row {i}: Malformed format [{entity}]{label_part}"
                raise AssertionError(msg)


def test_json_alphabetical() -> None:
    rows = load_dataset_rows()
    for i, row in enumerate(rows):
        processed = process_row(row)
        text = _string_field(processed, "text")
        if not text or text == "{}":
            continue
        pii = json.loads(text)
        keys = list(pii.keys())
        assert keys == sorted(keys), f"Row {i}: Keys not sorted {keys}"


def test_no_empty_text_with_pii() -> None:
    rows = load_dataset_rows()
    empty_indices = []
    for i, row in enumerate(rows):
        processed = process_row(row)
        text = _string_field(processed, "text")
        pii = json.loads(text) if text else {}
        if not pii:
            output = _string_field(processed, "output")
            if re.search(r"\[[^\[\]]+\]<[a-zA-Z_]+>", output):
                empty_indices.append(i)
    assert not empty_indices, f"Found {len(empty_indices)} rows with empty text but PII tags"


def test_xml_preservation() -> None:
    text = "The system returned <status>OK</status> for [John Doe]<name>."
    fixed = fix_labels(text)
    assert "<status>OK</status>" in fixed
    assert "[John Doe]<human_name>" in fixed


def test_fix_labels_cases() -> None:
    cases = [
        ("[Sau đại học] Quy chế", "Sau đại học Quy chế"),
        ("[John Doe]<name>", "[John Doe]<human_name>"),
        ("[John Doe]name", "[John Doe]<human_name>"),
        ("[BioNexa]company_name", "[BioNexa]<company_name>"),
        ("[2024-01-01]<date>", "[2024-01-01]<date>"),
        ("<status>OK</status>", "<status>OK</status>"),
        ("[Marriott]<company_name>", "[Marriott]<company_name>"),
        ("[test]invalid_label_xyz", "testinvalid_label_xyz"),
    ]
    for inp, expected in cases:
        result = fix_labels(inp)
        assert result == expected, f"fix_labels({inp!r}) = {result!r}, expected {expected!r}"


def test_fix_labels_does_not_promote_ambiguous_url_to_private_url() -> None:
    text = "Public [https://moh.gov.vn/guidelines/diabetes]<url> guideline"

    fixed = fix_labels(text)

    assert fixed == "Public https://moh.gov.vn/guidelines/diabetes guideline"
    assert extract_pii(fixed) == {}


def test_extract_pii() -> None:
    text = "[John]<human_name> works at [Acme]<company_name> since [2020]<date>"
    pii = extract_pii(text)
    assert pii == {"company_name": ["Acme"], "date": ["2020"], "human_name": ["John"]}


def test_no_dropped_rows() -> None:
    rows = load_dataset_rows()
    processed = [process_row(row) for row in rows]
    assert len(processed) == len(rows), "Rows were dropped during processing"
