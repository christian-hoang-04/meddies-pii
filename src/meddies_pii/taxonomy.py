from __future__ import annotations

import re
from typing import Literal, TypeGuard

PiiLabel = Literal[
    "address",
    "company_name",
    "date",
    "email_address",
    "human_name",
    "id_number",
    "phone_number",
    "private_url",
    "secret",
]

PII_LABELS: tuple[PiiLabel, ...] = (
    "address",
    "company_name",
    "date",
    "email_address",
    "human_name",
    "id_number",
    "phone_number",
    "private_url",
    "secret",
)
PII_LABEL_SET: frozenset[PiiLabel] = frozenset(PII_LABELS)
PII_LABELS_BRACKETED: frozenset[str] = frozenset(f"<{label}>" for label in PII_LABELS)
VALID_LABELS: frozenset[PiiLabel] = PII_LABEL_SET
VALID_LABELS_BRACKETED: frozenset[str] = PII_LABELS_BRACKETED


def is_pii_label(value: str) -> TypeGuard[PiiLabel]:
    return value in PII_LABEL_SET


def require_pii_label(value: str) -> PiiLabel:
    if is_pii_label(value):
        return value
    msg = f"unknown PII label: {value!r}"
    raise ValueError(msg)


def bracketed(label: PiiLabel) -> str:
    return f"<{label}>"


def label_regex_alt() -> str:
    return "|".join(re.escape(label) for label in PII_LABELS)
