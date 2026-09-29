"""Source label maps and precedence-preserving canonicalization."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from meddies_pii.annotations.label_aliases import INVALID_LABELS, LABEL_MAP
from meddies_pii.taxonomy import PiiLabel, is_pii_label

if TYPE_CHECKING:
    from collections.abc import Mapping

NEMOTRON_LABEL_MAP = {
    "first_name": "human_name",
    "last_name": "human_name",
    "middle_name": "human_name",
    "name": "human_name",
    "email": "email_address",
    "phone_number": "phone_number",
    "fax_number": "phone_number",
    "street_address": "address",
    "city": "address",
    "county": "address",
    "state": "address",
    "postcode": "address",
    "country": "address",
    "coordinate": "address",
    "company_name": "company_name",
    "date": "date",
    "date_of_birth": "date",
    "date_time": "date",
    "ssn": "id_number",
    "medical_record_number": "id_number",
    "health_plan_beneficiary_number": "id_number",
    "customer_id": "id_number",
    "employee_id": "id_number",
    "account_number": "id_number",
    "certificate_license_number": "id_number",
    "credit_debit_card": "id_number",
    "bank_routing_number": "id_number",
    "swift_bic": "id_number",
    "national_id": "id_number",
    "tax_id": "id_number",
    "vehicle_identifier": "id_number",
    "license_plate": "id_number",
    "unique_id": "id_number",
}
"""Deliberately skipped (not PII in our taxonomy).

Url, occupation, time, user_name, biometric_identifier, education_level, ipv4, ipv6, password, pin, cvv, api_key,
mac_address, http_cookie, device_identifier, employment_status, race_ethnicity, religious_belief, blood_type, gender, age,
political_view, language, sexuality.

"""

NEMOTRON_NAME_LABELS = {"first_name", "last_name", "middle_name"}

AI4PRIVACY_LABEL_MAP = {
    "GIVENNAME": "human_name",
    "SURNAME": "human_name",
    "EMAIL": "email_address",
    "TELEPHONENUM": "phone_number",
    "STREET": "address",
    "CITY": "address",
    "BUILDINGNUM": "address",
    "ZIPCODE": "address",
    "IDCARDNUM": "id_number",
    "PASSPORTNUM": "id_number",
    "DRIVERLICENSENUM": "id_number",
    "CREDITCARDNUMBER": "id_number",
    "ACCOUNTNUM": "id_number",
    "SOCIALNUM": "id_number",
    "TAXNUM": "id_number",
    "DATE": "date",
    "DATEOFBIRTH": "date",
    "ORGANISATIONPLACEHOLDER": "company_name",
}

AI4PRIVACY_NAME_LABELS = {"GIVENNAME", "SURNAME"}

GRETEL_LABEL_MAP = {
    "name": "human_name",
    "first_name": "human_name",
    "email": "email_address",
    "phone_number": "phone_number",
    "street_address": "address",
    "company": "company_name",
    "date": "date",
    "date_of_birth": "date",
    "ssn": "id_number",
    "driver_license_number": "id_number",
    "customer_id": "id_number",
    "document_id": "id_number",
    "bban": "id_number",
    "iban": "id_number",
    "credit_card_number": "id_number",
    "swift_bic_code": "id_number",
}

PII_MED_LABEL_MAP = {
    "GIVENNAME": "human_name",
    "SURNAME": "human_name",
    "EMAIL": "email_address",
    "TELEPHONENUM": "phone_number",
    "STREET": "address",
    "CITY": "address",
    "BUILDINGNUM": "address",
    "ZIPCODE": "address",
    "IDCARDNUM": "id_number",
    "CREDITCARDNUMBER": "id_number",
    "ACCOUNTNUM": "id_number",
    "SOCIALNUM": "id_number",
    "TAXNUM": "id_number",
    "DRIVERLICENSENUM": "id_number",
    "DATEOFBIRTH": "date",
}


_BIO_PREFIX_RE = re.compile(r"^(?:B|I|E|S|U)-", re.IGNORECASE)
_REUSED_SOURCE_MAPS: tuple[Mapping[str, str], ...] = (
    NEMOTRON_LABEL_MAP,
    AI4PRIVACY_LABEL_MAP,
    GRETEL_LABEL_MAP,
    PII_MED_LABEL_MAP,
)


def normalize_native_label(label: str) -> str:
    normalized = label.strip().strip("<>").replace(" ", "_")
    normalized = _BIO_PREFIX_RE.sub("", normalized)
    return normalized.strip().lower()


def coerce_pii_label(label: str) -> PiiLabel | None:
    normalized = normalize_native_label(label)
    return normalized if is_pii_label(normalized) else None


# reason: Direct labels, aliases, and credential remaps use ordered exits; merging would hide policy precedence.
def map_native_label_to_pii_label(  # ruff: ignore[too-many-return-statements]
    label: str,
    *,
    overrides: Mapping[str, PiiLabel] | None = None,
) -> PiiLabel | None:
    normalized = normalize_native_label(label)
    if not normalized or normalized == "o":
        return None
    if overrides and normalized in overrides:
        return overrides[normalized]
    if normalized in INVALID_LABELS:
        return None
    direct = coerce_pii_label(normalized)
    if direct is not None:
        return direct

    for source_map in _REUSED_SOURCE_MAPS:
        mapped = source_map.get(normalized) or source_map.get(normalized.upper()) or source_map.get(normalized.lower())
        if mapped is None:
            continue
        coerced = coerce_pii_label(mapped)
        if coerced is not None:
            return coerced

    bracketed = LABEL_MAP.get(normalized)
    if bracketed is None:
        return None
    mapped = bracketed.strip("<>").lower()
    return mapped if is_pii_label(mapped) else None
