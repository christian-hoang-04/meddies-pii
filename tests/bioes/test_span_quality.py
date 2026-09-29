"""Acceptance-side span-quality filter (V1+V2 structure-capture, V3 phone).

The regression guard is non-negotiable: the ACCEPTABLE hard cases (concatenation,
OCR noise, JWT cookies, written-out dates, extensions) must SURVIVE — only
structure-capture / over-long-phone spans are dropped.

`[entity]` is bracket DECORATION around a real entity — KEEP (the false reject the corpus re-measure surfaced).

body has >=2 structural markers even though it doesn't start with one.

--- ACCEPTABLE: must SURVIVE (regression guard) ---.

OCR-style fused hospital name — exactly the hard case we want.

--- V5 email-shape: a long "email" with no @ / at / dot is prose, not an email ---.

short and could be a username fragment — don't over-reach.

--- V7 entity-presence: an id_number / phone_number with NO digit is wrong-label ---.

human_name has no digit requirement — must survive.

--- V7 name-paragraph: a >60-char human_name spanning sentences is paragraph-capture ---.

a genuinely long name with no sentence punctuation survives.

--- leading-marker false-positive recovery (the case the eyeball check found) ---.

a real id with a stray <id_number> marker glued on front — strip + keep, do NOT drop on V1's leading `<`.

stripping the marker still leaves a JSON block -> still bad.

"""

from __future__ import annotations

from meddies_pii.training.bioes.data.span_quality import clean_spans, is_bad_span


def test_drops_leading_json_brace() -> None:
    assert is_bad_span(
        "company_name",
        '\n    {\n      "sequence": 1,\n      "productOrService": {"coding":',
    )


def test_drops_leading_quoted_json_key() -> None:
    assert is_bad_span("date", '"servicedDate": "2025-02-14", "sequence": 1')


def test_drops_dicom_tuple_prefix() -> None:
    assert is_bad_span("id_number", "(0008,0018) 1.2.840.10008.5.1.4")


def test_drops_json_array_prefix() -> None:
    assert is_bad_span("id_number", '[{"resourceType": "Patient", "id": "x"}]')


def test_keeps_bracket_decorated_entity() -> None:
    assert not is_bad_span("company_name", "[부산 사상 일반 산업단지 관리사무소]")


def test_drops_embedded_fhir_markers() -> None:
    assert is_bad_span(
        "address",
        'Ward 3 "system": "http://hl7.org/fhir" "code": "99213" extra',
    )


def test_drops_xml_tag_prefix() -> None:
    assert is_bad_span("human_name", "<name>Tran Bao</name> with more")


def test_drops_overlong_phone() -> None:
    assert is_bad_span("phone_number", "0" * 60)


def test_keeps_clean_company_name() -> None:
    assert not is_bad_span("company_name", "Bệnh viện Đa khoa Hà Nội")


def test_keeps_concatenation_artifact() -> None:
    assert not is_bad_span("company_name", "Bệnh việnĐa khoaXây Dựng Hà Nội")


def test_keeps_jwt_cookie_secret() -> None:
    assert not is_bad_span("secret", "key=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJwIn0; Path=/; HttpOnly; Secure")


def test_keeps_written_out_date() -> None:
    assert not is_bad_span("date", "ngày hai mươi ba tháng mười hai năm hai nghìn hai mươi tư")


def test_keeps_long_private_url() -> None:
    assert not is_bad_span(
        "private_url",
        "https://portal.greenfieldmed.com/patients/P-12345/results?token=Alpha9",
    )


def test_keeps_short_phone_extension() -> None:
    assert not is_bad_span("phone_number", "4417")


def test_keeps_obfuscated_email() -> None:
    assert not is_bad_span("email_address", "mai.nguyen at example dot vn")


def test_drops_email_without_at_or_obfuscation() -> None:
    assert is_bad_span(
        "email_address",
        "Patient was seen today and will return next week for a review",
    )


def test_keeps_short_email_fragment_without_at() -> None:
    assert not is_bad_span("email_address", "jdoe.clinic")


def test_drops_id_number_with_no_digit() -> None:
    assert is_bad_span("id_number", "Patient Identification Record Number Section")


def test_drops_phone_with_no_digit() -> None:
    assert is_bad_span("phone_number", "call the front desk during business hours")


def test_keeps_id_with_digits() -> None:
    assert not is_bad_span("id_number", "YGH-EM-882910")


def test_keeps_alpha_only_short_human_name() -> None:
    assert not is_bad_span("human_name", "Nguyễn Văn An")


def test_drops_human_name_paragraph() -> None:
    assert is_bad_span(
        "human_name",
        "Dr. Smith examined the patient. Dr. Jones will follow up tomorrow morning.",
    )


def test_keeps_titled_name_short() -> None:
    assert not is_bad_span("human_name", "Dr. Smith Jr.")


def test_keeps_long_multi_word_name_no_sentences() -> None:
    assert not is_bad_span("human_name", "Alexander Bartholomew Reginald Kensington Smythe the Third")


def test_keeps_entity_with_stray_leading_marker() -> None:
    assert not is_bad_span("id_number", "<id_number>MRN998231")
    assert not is_bad_span("date", "<date>2024-01-18")
    assert not is_bad_span("email_address", "<email_address>a@b.org")


def test_still_drops_structure_after_marker() -> None:
    assert is_bad_span("id_number", '<id_number>{"resourceType": "Patient"}')


def test_clean_spans_strips_leading_marker_and_fixes_offset() -> None:
    spans = [
        {
            "category": "id_number",
            "start": 10,
            "end": 30,
            "text": "<id_number>MRN998231",
        },
    ]
    kept, dropped = clean_spans(spans)
    assert dropped == 0
    assert kept[0]["text"] == "MRN998231"
    assert kept[0]["start"] == 21
    assert kept[0]["end"] == 30
