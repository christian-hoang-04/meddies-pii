r"""--- Edge / near-miss contamination detectors (reference-ranges, lab values, dosages, ages-as-dates).

Non-PII surfaces wrongly tagged as PII labels. ---.

`<10` has no decimal, so the original NUMERIC_MEASUREMENT_RE missed it.

`%` is non-word, so a trailing `\b` failed to anchor `95%`; the lookahead fix catches it. SpO2/saturation percentages are
lab values, never PII.

--- Native-script date surfaces (CJK / Vietnamese / Thai / Burmese / Lao / Tamil). A genuine native-script date gold span
must NOT be flagged as `date_gold_surface_is_not_date_like`; the detector recognizes the native day+month+year structure.
---.

Empty predicted_spans so the gold span is `gold_only` and the suspicious-gold detector actually runs on this surface.

A phone number spoken as native digit-words has NO date structure (no day+month-marker+year) — it must not be accepted as a
valid date.

--- Spelled-out native dates (voice-scribe / OCR transcription). The numerals are NUMBER-WORDS, not digits, so the
digit-anchored patterns miss them; the marker-anchored detector (month name, or >=2 distinct year/month/day marker words)
must still recognize them. These exact surfaces are drawn from the pinned eval set, where they were false-flagged at 94.4%
before the fix. ---.

A native-digit time (Myanmar 09:18), not a date — labeling it `date` must still be flagged; the marker detector must not
over-accept a bare time.

"""

from __future__ import annotations

import pytest

from meddies_pii.training.bioes.eval.audit import (
    audit_record,
    audit_records,
    summarize_issues,
)


def test_audit_batch_preserves_serialized_issue_and_summary_contract() -> None:
    record = {
        "uid": "serialized-contract",
        "text": "Contact: clinician@example.org",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_records([record])

    assert [issue.to_dict() for issue in issues] == [
        {
            "uid": "serialized-contract",
            "issue_type": "unlabeled_candidate",
            "severity": "high",
            "label": "email_address",
            "start": 9,
            "end": 30,
            "text": "clinician@example.org",
            "reason": "email_pattern",
            "recommended_action": "review_regex_candidate",
            "context": "Contact: clinician@example.org",
            "evidence": {
                "candidate": {
                    "label": "email_address",
                    "start": 9,
                    "end": 30,
                    "text": "clinician@example.org",
                    "reason": "email_pattern",
                    "severity": "high",
                },
            },
        },
    ]
    assert summarize_issues(issues) == {
        "issues": 1,
        "by_type": {"unlabeled_candidate": 1},
        "by_action": {"review_regex_candidate": 1},
        "by_label": {"email_address": 1},
        "by_severity": {"high": 1},
    }


def test_flags_unlabeled_company_candidate_in_validation_text() -> None:
    record = {
        "uid": "pii-bioes:validation:3",
        "text": "Đơn vị bảo hiểm: Công ty Bảo hiểm Y tế Hải Phòng\nMã số: HD123",
        "gold_spans": [{"label": "id_number", "start": 58, "end": 63, "text": "HD123"}],
        "predicted_spans": [{"label": "id_number", "start": 58, "end": 63, "text": "HD123"}],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "unlabeled_candidate"
        and issue.label == "company_name"
        and issue.text == "Công ty Bảo hiểm Y tế Hải Phòng"
        for issue in issues
    )


def test_valid_predicted_date_missing_from_gold_is_gold_missing_candidate() -> None:
    record = {
        "uid": "row-1",
        "text": "Hạn sử dụng thẻ: 31/12/2024",
        "gold_spans": [],
        "predicted_spans": [{"label": "date", "start": 17, "end": 27, "text": "31/12/2024"}],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "prediction_not_in_gold"
        and issue.recommended_action == "review_gold_add_predicted_span"
        and issue.label == "date"
        for issue in issues
    )


def test_time_only_prediction_is_not_recommended_as_gold_date() -> None:
    record = {
        "uid": "row-time",
        "text": "Date de réception: 18/03/2025 à 08:45",
        "gold_spans": [{"label": "date", "start": 19, "end": 29, "text": "18/03/2025"}],
        "predicted_spans": [{"label": "date", "start": 32, "end": 37, "text": "08:45"}],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "prediction_not_in_gold"
        and issue.recommended_action == "review_model_false_positive"
        and issue.text == "08:45"
        for issue in issues
    )


def test_numeric_lab_range_labeled_id_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-2",
        "text": "Glucose 5.2 mmol/L REF 3.9-6.1 OK",
        "gold_spans": [{"label": "id_number", "start": 25, "end": 32, "text": "3.9-6.1"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.recommended_action == "review_gold_remove_or_relabel"
        and issue.label == "id_number"
        for issue in issues
    )


def test_pure_numeric_id_number_is_not_treated_as_lab_range() -> None:
    record = {
        "uid": "row-numeric-id",
        "text": "Patient ID: 987654 was assigned at the admission desk.",
        "gold_spans": [{"label": "id_number", "start": 12, "end": 18, "text": "987654"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(
        issue.issue_type == "suspicious_gold_span" and issue.label == "id_number" and issue.text == "987654"
        for issue in issues
    )


def test_icd_code_prediction_is_not_recommended_as_gold_id_number() -> None:
    record = {
        "uid": "row-icd",
        "text": "Diagnosis: J45.9 asthma",
        "gold_spans": [],
        "predicted_spans": [{"label": "id_number", "start": 11, "end": 16, "text": "J45.9"}],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "prediction_not_in_gold"
        and issue.recommended_action == "review_model_false_positive"
        and issue.text == "J45.9"
        for issue in issues
    )


def test_same_label_overlap_is_boundary_mismatch_not_plain_model_error() -> None:
    record = {
        "uid": "row-3",
        "text": "Adresse: 12 Rue de la Pompe, 75016 Paris",
        "gold_spans": [
            {"label": "address", "start": 9, "end": 27, "text": "12 Rue de la Pompe"},
            {"label": "address", "start": 29, "end": 40, "text": "75016 Paris"},
        ],
        "predicted_spans": [
            {
                "label": "address",
                "start": 9,
                "end": 40,
                "text": "12 Rue de la Pompe, 75016 Paris",
            },
        ],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "boundary_mismatch"
        and issue.label == "address"
        and issue.recommended_action == "review_boundary_policy"
        for issue in issues
    )


def test_department_candidate_is_medium_not_high_confidence_company() -> None:
    record = {
        "uid": "row-4",
        "text": "Bệnh nhân đến Khoa Khám bệnh theo yêu cầu vào lúc 06:30.",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    department_issues = [
        issue
        for issue in issues
        if issue.issue_type == "unlabeled_candidate"
        and issue.label == "company_name"
        and issue.text == "Khoa Khám bệnh theo yêu cầu"
    ]
    assert department_issues
    assert department_issues[0].severity == "medium"


def test_public_homepage_url_is_not_private_url_candidate() -> None:
    record = {
        "uid": "row-public-url",
        "text": "Website: www.benhvienphuclam.vn",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(issue.label == "private_url" for issue in issues)


def test_patient_portal_url_is_private_url_candidate() -> None:
    record = {
        "uid": "row-private-url",
        "text": "Portal: https://his.phuclam.vn/patients/78945",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "unlabeled_candidate" and issue.label == "private_url" and issue.severity == "high"
        for issue in issues
    )


def test_date_is_not_flagged_as_phone_candidate() -> None:
    record = {
        "uid": "row-date-phone-noise",
        "text": "Ngày sinh: 15/08/1978. Tái khám: 22/10/2024 08:00.",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(issue.label == "phone_number" for issue in issues)


def test_oid_version_number_is_not_date_candidate() -> None:
    record = {
        "uid": "row-oid",
        "text": "OID root 1.2.840.10008.5.1.4.1.1.2 appears in DICOM XML.",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(issue.label == "date" for issue in issues)


def test_date_inside_id_is_not_recommended_as_regex_gold_add() -> None:
    record = {
        "uid": "row-date-in-id",
        "text": "Invoice IFN-2025-03-18-001 was filed.",
        "gold_spans": [
            {
                "label": "id_number",
                "start": 8,
                "end": 26,
                "text": "IFN-2025-03-18-001",
            },
        ],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(
        issue.label == "date"
        and issue.recommended_action
        in {
            "review_regex_candidate",
            "review_gold_add_predicted_span",
        }
        for issue in issues
    )


def test_predicted_email_missing_from_gold_is_not_duplicated_as_regex_candidate() -> None:
    record = {
        "uid": "row-email-dup",
        "text": "Send report to reports.chennai@precisionlabs.in.",
        "gold_spans": [],
        "predicted_spans": [
            {
                "label": "email_address",
                "start": 15,
                "end": 47,
                "text": "reports.chennai@precisionlabs.in",
            },
        ],
    }

    issues = audit_record(record)
    matching = [
        issue for issue in issues if issue.label == "email_address" and issue.text == "reports.chennai@precisionlabs.in"
    ]

    assert len(matching) == 1
    assert matching[0].recommended_action == "review_gold_add_predicted_span"


def test_secret_assignment_is_high_confidence_candidate() -> None:
    record = {
        "uid": "row-secret",
        "text": "OTP đăng nhập: 834-229; x_csrf=af91bb22.",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "unlabeled_candidate"
        and issue.recommended_action == "review_regex_candidate"
        and issue.label == "secret"
        and issue.severity == "high"
        for issue in issues
    )


def test_compact_hl7_date_is_date_candidate() -> None:
    record = {
        "uid": "row-compact-date",
        "text": "HL7 DOB=19830412 for patient admission.",
        "gold_spans": [],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "unlabeled_candidate"
        and issue.recommended_action == "review_regex_candidate"
        and issue.label == "date"
        and issue.text == "19830412"
        for issue in issues
    )


def test_no_decimal_comparator_range_labeled_id_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-range-no-decimal",
        "text": "Troponin REF <10 ng/L within normal limits.",
        "gold_spans": [{"label": "id_number", "start": 13, "end": 16, "text": "<10"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "numeric_reference_range_labeled_id_number"
        and issue.label == "id_number"
        and issue.text == "<10"
        for issue in issues
    )


def test_unicode_comparator_range_labeled_id_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-range-unicode",
        "text": "Cholesterol target ≤5.2 mmol/L per guideline.",
        "gold_spans": [{"label": "id_number", "start": 19, "end": 23, "text": "≤5.2"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "numeric_reference_range_labeled_id_number"
        and issue.label == "id_number"
        and issue.text == "≤5.2"
        for issue in issues
    )


def test_lab_value_with_unit_labeled_id_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-lab-unit",
        "text": "Glucose result 5.2 mmol/L recorded today.",
        "gold_spans": [{"label": "id_number", "start": 15, "end": 25, "text": "5.2 mmol/L"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "lab_value_with_unit_labeled_pii"
        and issue.label == "id_number"
        for issue in issues
    )


def test_percent_lab_value_labeled_id_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-percent",
        "text": "Oxygen saturation 95% on room air.",
        "gold_spans": [{"label": "id_number", "start": 18, "end": 21, "text": "95%"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span" and issue.reason == "lab_value_with_unit_labeled_pii"
        for issue in issues
    )


def test_blood_pressure_unit_labeled_phone_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-bp-phone",
        "text": "Huyết áp 120 mmHg đo lúc nhập viện.",
        "gold_spans": [{"label": "phone_number", "start": 9, "end": 17, "text": "120 mmHg"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "lab_value_with_unit_labeled_pii"
        and issue.label == "phone_number"
        for issue in issues
    )


def test_dosage_string_labeled_id_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-dosage",
        "text": "Paracetamol 500mg twice daily after meals.",
        "gold_spans": [{"label": "id_number", "start": 12, "end": 17, "text": "500mg"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "dosage_or_unit_labeled_pii"
        and issue.label == "id_number"
        for issue in issues
    )


def test_tablet_count_labeled_address_is_suspicious_gold() -> None:
    record = {
        "uid": "row-tablets",
        "text": "Take 2 tablets every morning.",
        "gold_spans": [{"label": "address", "start": 5, "end": 14, "text": "2 tablets"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "dosage_or_unit_labeled_pii"
        and issue.label == "address"
        for issue in issues
    )


def test_age_expression_labeled_date_is_suspicious_gold() -> None:
    record = {
        "uid": "row-age-date",
        "text": "Bệnh nhân nam 78 tuổi nhập viện.",
        "gold_spans": [{"label": "date", "start": 14, "end": 21, "text": "78 tuổi"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "age_expression_labeled_pii"
        and issue.label == "date"
        for issue in issues
    )


def test_age_expression_labeled_id_number_is_suspicious_gold() -> None:
    record = {
        "uid": "row-age-id",
        "text": "Patient is 45 years old at intake.",
        "gold_spans": [{"label": "id_number", "start": 11, "end": 23, "text": "45 years old"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "age_expression_labeled_pii"
        and issue.label == "id_number"
        for issue in issues
    )


def test_bare_small_integer_labeled_date_is_suspicious_gold() -> None:
    record = {
        "uid": "row-bare-int-date",
        "text": "Số thứ tự khám: 45 hôm nay.",
        "gold_spans": [{"label": "date", "start": 16, "end": 18, "text": "45"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "suspicious_gold_span"
        and issue.reason == "bare_small_integer_labeled_date"
        and issue.label == "date"
        for issue in issues
    )


def test_real_mrn_id_number_is_not_flagged_as_range_or_lab() -> None:
    record = {
        "uid": "row-real-mrn",
        "text": "Medical record number MRN-4471902 issued at admission.",
        "gold_spans": [{"label": "id_number", "start": 22, "end": 33, "text": "MRN-4471902"}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(issue.issue_type == "suspicious_gold_span" and issue.text == "MRN-4471902" for issue in issues)


def test_real_date_is_not_flagged_by_age_or_bare_integer_detector() -> None:
    record = {
        "uid": "row-real-date",
        "text": "Ngày nhập viện: 31/12/2024 buổi sáng.",
        "gold_spans": [{"label": "date", "start": 16, "end": 26, "text": "31/12/2024"}],
        "predicted_spans": [{"label": "date", "start": 16, "end": 26, "text": "31/12/2024"}],
    }

    issues = audit_record(record)

    assert not any(issue.issue_type == "suspicious_gold_span" and issue.label == "date" for issue in issues)


@pytest.mark.parametrize(
    ("uid", "text", "start", "end", "surface"),
    [
        ("row-date-zh", "出生日期 2024年3月15日 入院", 5, 15, "2024年3月15日"),
        ("row-date-ko", "입원일 2024년 3월 15일 확인", 4, 16, "2024년 3월 15일"),
        (
            "row-date-vi",
            "Ngày nhập viện 15 tháng 3 năm 2024 buổi sáng",
            15,
            34,
            "15 tháng 3 năm 2024",
        ),
        ("row-date-th", "วันเกิด 15 มีนาคม 2567 ที่บันทึก", 8, 22, "15 มีนาคม 2567"),
        ("row-date-my", "မွေးနေ့ 2024 ခုနှစ် မှတ်တမ်း", 8, 19, "2024 ခုနှစ်"),
        ("row-date-lo", "ວັນເກີດ 15 ມີນາ 2024 ບັນທຶກ", 8, 20, "15 ມີນາ 2024"),
        ("row-date-ta", "பிறந்த தேதி 15 மாதம் 2024 பதிவு", 12, 25, "15 மாதம் 2024"),
    ],
)
def test_native_script_date_gold_span_is_not_suspicious(uid: str, text: str, start: int, end: int, surface: str) -> None:
    record = {
        "uid": uid,
        "text": text,
        "gold_spans": [{"label": "date", "start": start, "end": end, "text": surface}],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(issue.issue_type == "suspicious_gold_span" and issue.label == "date" for issue in issues)


def test_digit_word_phone_sequence_is_not_accepted_as_date() -> None:
    surface = "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า"
    text = f"โทร {surface} ติดต่อ"
    start = text.index(surface)
    end = start + len(surface)
    record = {
        "uid": "row-digit-word-phone",
        "text": text,
        "gold_spans": [],
        "predicted_spans": [{"label": "date", "start": start, "end": end, "text": surface}],
    }

    issues = audit_record(record)

    assert any(
        issue.issue_type == "prediction_not_in_gold"
        and issue.label == "date"
        and issue.recommended_action == "review_model_false_positive"
        for issue in issues
    )


@pytest.mark.parametrize(
    ("uid", "text", "start", "end", "surface"),
    [
        ("row-dosage-date", "Paracetamol 500mg twice daily", 12, 17, "500mg"),
        ("row-bp-date", "Huyết áp 120/80 mmHg", 9, 15, "120/80"),
        ("row-age-date", "Bệnh nhân 45 tuổi nhập viện", 10, 17, "45 tuổi"),
    ],
)
def test_clinical_measurement_is_not_accepted_as_date(uid: str, text: str, start: int, end: int, surface: str) -> None:
    record = {
        "uid": uid,
        "text": text,
        "gold_spans": [],
        "predicted_spans": [{"label": "date", "start": start, "end": end, "text": surface}],
    }

    issues = audit_record(record)

    assert not any(
        issue.issue_type == "prediction_not_in_gold"
        and issue.label == "date"
        and issue.recommended_action == "review_gold_add_predicted_span"
        for issue in issues
    )


@pytest.mark.parametrize(
    ("uid", "surface"),
    [
        ("spelled-ko", "이천이십육년 삼월 십사일"),
        ("spelled-ja-hiragana", "れいわ 6ねん しがつ じゅういちにち"),
        ("spelled-ja-imperial", "令和元年 五月 十四日"),
        ("spelled-ta", "இருபத்தி ஒன்று மார்ச் இரண்டாயிரத்து இருபத்து ஆறு"),
        ("spelled-th", "สิบเก้า มีนาคม สองพันยี่สิบห้า"),
        ("spelled-lo", "ສອງສິບສີ່ ກຸມພາ 2026"),
        ("spelled-my", "နှစ် ထောင် နှစ်ဆယ့်လေး ခုနှစ် မတ်လ ဆယ့်နှစ် ရက်"),
    ],
)
def test_spelled_out_native_date_gold_span_is_not_suspicious(uid: str, surface: str) -> None:
    text = f"ghi nhận {surface} kết thúc"
    start = text.index(surface)
    record = {
        "uid": uid,
        "text": text,
        "gold_spans": [
            {
                "label": "date",
                "start": start,
                "end": start + len(surface),
                "text": surface,
            },
        ],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert not any(issue.issue_type == "suspicious_gold_span" and issue.label == "date" for issue in issues)


def test_native_digit_time_only_labeled_date_is_suspicious() -> None:
    surface = "၀၉:၁၈"
    text = f"အချိန် {surface} မှတ်တမ်း"
    start = text.index(surface)
    record = {
        "uid": "native-time-only",
        "text": text,
        "gold_spans": [
            {
                "label": "date",
                "start": start,
                "end": start + len(surface),
                "text": surface,
            },
        ],
        "predicted_spans": [],
    }

    issues = audit_record(record)

    assert any(issue.issue_type == "suspicious_gold_span" and issue.label == "date" for issue in issues)
