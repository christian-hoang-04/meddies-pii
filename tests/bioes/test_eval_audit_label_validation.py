from __future__ import annotations

import pytest

from meddies_pii.training.bioes.eval.audit import audit_record


@pytest.mark.parametrize(
    ("label", "surface", "recommended_action"),
    [
        ("date", "not a date", "review_model_false_positive"),
        ("email_address", "clinician@example.org", "review_gold_add_predicted_span"),
        ("email_address", "not-an-email", "review_model_false_positive"),
        (
            "private_url",
            "https://secure.health.local/patient/42",
            "review_gold_add_predicted_span",
        ),
        (
            "private_url",
            "https://hospital.test/clinical/overview",
            "review_model_false_positive",
        ),
        ("phone_number", "+1 202-555-0100", "review_gold_add_predicted_span"),
        ("phone_number", "12345678", "review_model_false_positive"),
        ("id_number", "AB12", "review_gold_add_predicted_span"),
        ("id_number", "ABCD", "review_model_false_positive"),
        ("company_name", "Bệnh viện X", "review_gold_add_predicted_span"),
        ("company_name", "Khoa Tim", "review_gold_add_predicted_span"),
        ("company_name", "Alpha Beta", "review_gold_add_predicted_span"),
        ("company_name", "Acme", "review_model_false_positive"),
        ("human_name", "Ada Lovelace", "review_gold_add_predicted_span"),
        ("human_name", "Doctor: Ada", "review_model_false_positive"),
        ("human_name", "Ada", "review_model_false_positive"),
        ("address", "12 Main Street", "review_gold_add_predicted_span"),
        ("address", "Tiny", "review_model_false_positive"),
        ("secret", "token=abcd", "review_gold_add_predicted_span"),
        ("secret", "abcd", "review_model_false_positive"),
        ("human_name", " ", "review_model_false_positive"),
    ],
)
def test_prediction_without_gold_is_triaged_by_label_surface(
    label: str,
    surface: str,
    recommended_action: str,
) -> None:
    issues = audit_record({
        "uid": f"{label}:{surface}",
        "text": surface,
        "gold_spans": [],
        "predicted_spans": [
            {
                "label": label,
                "start": 0,
                "end": len(surface),
                "text": surface,
            },
        ],
    })

    predictions_without_gold = [issue for issue in issues if issue.issue_type == "prediction_not_in_gold"]
    assert len(predictions_without_gold) == 1
    assert predictions_without_gold[0].recommended_action == recommended_action


@pytest.mark.parametrize(
    ("label", "surface", "reason"),
    [
        ("email_address", "Email: clinician@example.org", "span_contains_field_prefix"),
        ("date", "not a calendar value", "date_gold_surface_is_not_date_like"),
        (
            "human_name",
            "Doctor: Ada Lovelace",
            "human_name_contains_field_prefix_or_role",
        ),
    ],
)
def test_suspicious_gold_surfaces_report_the_specific_reason(
    label: str,
    surface: str,
    reason: str,
) -> None:
    issues = audit_record({
        "uid": f"gold:{label}",
        "text": surface,
        "gold_spans": [
            {
                "label": label,
                "start": 0,
                "end": len(surface),
                "text": surface,
            },
        ],
        "predicted_spans": [],
    })

    suspicious = [issue for issue in issues if issue.issue_type == "suspicious_gold_span"]
    assert len(suspicious) == 1
    assert suspicious[0].reason == reason
