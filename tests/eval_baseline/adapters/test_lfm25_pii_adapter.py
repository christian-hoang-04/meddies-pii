from __future__ import annotations

import pytest

from meddies_pii.eval_baseline.adapters.lfm25_pii import (
    LFM25_PII_SUPPORTED_LABELS,
    LIQUID_LABEL_FOLD,
    Lfm25PiiSpaceAdapter,
    map_hybrid_spans,
)
from meddies_pii.taxonomy import PII_LABEL_SET


class _Detector:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def detect(self, text: str) -> object:
        self.calls.append(text)
        return self.outputs[len(self.calls) - 1]


def test_liquid_label_fold_accounts_for_every_native_label_and_all_meddies_labels() -> None:
    expected_fold = {
        "identity.person_name": "human_name",
        "identity.ssn": "id_number",
        "identity.national_id": "id_number",
        "identity.passport": "id_number",
        "identity.drivers_license": "id_number",
        "identity.date_of_birth": "date",
        "identity.tax_id": "id_number",
        "contact.email": "email_address",
        "contact.phone": "phone_number",
        "contact.address": "address",
        "contact.postal_code": "address",
        "contact.ip_address": "id_number",
        "financial.credit_card": "id_number",
        "financial.iban": "id_number",
        "financial.bank_account": "id_number",
        "financial.swift_bic": "id_number",
        "financial.crypto_wallet": "id_number",
        "financial.amount": None,
        "credential.api_key": "secret",
        "credential.password": "secret",
        "credential.private_key": "secret",
        "credential.jwt": "secret",
        "credential.connection_string": "secret",
        "developer.login_credentials": "secret",
        "online.username": "human_name",
        "online.url": "private_url",
        "device.mac_address": "id_number",
        "device.imei": "id_number",
        "developer.device_id": "id_number",
        "location.gps_coordinates": "address",
        "healthcare.medical_record": "id_number",
        "healthcare.condition": None,
        "healthcare.medication": None,
        "healthcare.health_plan_id": "id_number",
        "org.company_name": "company_name",
        "special.religion": None,
        "special.political": None,
        "special.orientation": None,
        "special.health_status": None,
        "legal.case_number": "id_number",
    }

    assert expected_fold == LIQUID_LABEL_FOLD
    assert len(LIQUID_LABEL_FOLD) == 40
    assert LFM25_PII_SUPPORTED_LABELS == PII_LABEL_SET


@pytest.mark.parametrize(
    "native_label",
    [
        "financial.amount",
        "healthcare.condition",
        "healthcare.medication",
        "special.religion",
        "special.political",
        "special.orientation",
        "special.health_status",
    ],
)
def test_non_pii_native_labels_are_dropped(native_label: str) -> None:
    assert map_hybrid_spans("Alice", [{"start": 0, "end": 5, "type": native_label}]) == []


def test_adapter_calls_detector_once_per_document_in_order_and_keeps_exact_boundaries() -> None:
    detector = _Detector([
        {"spans": [{"start": 0, "end": 5, "type": "online.username"}]},
        {"spans": [{"start": 2, "end": 6, "type": "identity.person_name"}]},
    ])
    adapter = Lfm25PiiSpaceAdapter(detector)

    predictions = adapter.predict(["Alice", "xxJohn"])

    assert detector.calls == ["Alice", "xxJohn"]
    assert [[(span.start, span.end, span.text, span.label) for span in spans] for spans in predictions] == [
        [(0, 5, "Alice", "human_name")],
        [(2, 6, "John", "human_name")],
    ]


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"spans": "not-a-list"},
        {"spans": ["not-a-mapping"]},
        {"spans": [{"start": "0", "end": 5, "type": "identity.person_name"}]},
        {"spans": [{"start": 0, "end": 5, "type": 4}]},
    ],
)
def test_adapter_rejects_malformed_vendor_output(output: object) -> None:
    adapter = Lfm25PiiSpaceAdapter(_Detector([output]))

    with pytest.raises(RuntimeError, match="Liquid Space detector"):
        adapter.predict(["Alice"])


def test_map_hybrid_spans_drops_non_pii_deduplicates_and_keeps_native_boundaries() -> None:
    text = "User alice paid $50 with card 3530-2910-7196-."
    raw = [
        {"start": 5, "end": 10, "type": "online.username"},
        {"start": 16, "end": 19, "type": "financial.amount"},
        {"start": 30, "end": 45, "type": "financial.credit_card"},
        {"start": 30, "end": 45, "type": "identity.national_id"},
    ]

    spans = map_hybrid_spans(text, raw)

    assert [(span.start, span.end, span.label, span.text) for span in spans] == [
        (5, 10, "human_name", "alice"),
        (30, 45, "id_number", "3530-2910-7196-"),
    ]


def test_unknown_native_label_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unknown Liquid native label"):
        map_hybrid_spans(
            "Alice",
            [{"start": 0, "end": 5, "type": "identity.unannounced_type"}],
        )
