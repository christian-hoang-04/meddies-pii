from __future__ import annotations

import unicodedata
from dataclasses import fields, replace
from typing import get_args, get_type_hints

import pytest

from meddies_pii.eval_baseline.regex_release.regex_fixtures import (
    AUDITED_ENGINE_AND_AUTH_FIXTURES,
    AUDITED_ORG_KEEP_TEXTS,
    AUDITED_ORG_NEGATIVE_TEXTS,
    POSITIVE_SELECTION_FIXTURES,
    RegexExpectedFixture,
)
from meddies_pii.regex_runtime import regex_manifest
from meddies_pii.regex_runtime.packs import ALL_PACKS, Cue, CuePack, pack_pattern
from meddies_pii.regex_runtime.packs.vi import VI_ORG_PACK
from meddies_pii.regex_runtime.postprocess import apply_regex_postprocess
from meddies_pii.regex_runtime.rules import (
    REGEX_RULES,
    RegexRule,
    regex_candidates,
)
from meddies_pii.spans import CharSpan

NON_BREAKING_HYPHEN = "\N{NON-BREAKING HYPHEN}"


def _org_spans(text: str, *, language: str | None = None) -> list[tuple[str, str]]:
    return [(span.text, span.label) for span in apply_regex_postprocess(text, (), language=language).spans]


@pytest.mark.parametrize(
    ("language", "text"),
    [(language, text) for language, texts in AUDITED_ORG_NEGATIVE_TEXTS.items() for text in texts],
)
def test_audited_organization_false_positives_stay_locked_out(language: str, text: str) -> None:
    assert _org_spans(text, language=language) == []


@pytest.mark.parametrize(
    ("language", "text"),
    [(language, text) for language, texts in AUDITED_ORG_KEEP_TEXTS.items() for text in texts],
)
def test_audited_unlabeled_organization_names_remain_captured(language: str, text: str) -> None:
    assert _org_spans(text, language=language) == [(text, "company_name")]


@pytest.mark.parametrize(
    "fixture",
    AUDITED_ENGINE_AND_AUTH_FIXTURES,
    ids=lambda fixture: fixture.fixture_id,
)
def test_audited_engine_and_authoritative_cases_match_their_disposition(
    fixture: RegexExpectedFixture,
) -> None:
    result = apply_regex_postprocess(fixture.text, (), language=fixture.language)

    assert tuple((span.text, span.label) for span in result.spans) == fixture.expected


@pytest.mark.parametrize(
    "text",
    [
        "Variant detected on Chr 17.",
        "Plan: Labor Induction tomorrow.",
        "Diagnosis: Hospital Acquired Pneumonia.",
        "Assessment: Hospital Type 2 Diabetes.",
    ],
)
def test_known_document_language_limits_organization_packs(text: str) -> None:
    assert _org_spans(text, language="en") == []


def test_unknown_document_language_fails_closed_for_organization_packs() -> None:
    assert _org_spans("Hospital Problem List") == []
    assert _org_spans("Email: ha@example.com") == [("ha@example.com", "email_address")]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("HOSPITAL MÁS CERCANO", "ms"),
        ("HOSPITAL KERAJAAN", "es"),
        ("KLINIK TERDEKAT", "de"),
    ],
)
def test_shared_cues_union_every_owner_packs_blocked_tokens(text: str, language: str) -> None:
    assert _org_spans(text, language=language) == []


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Hospital Course-Summary", "en"),
        (f"Hospital Course{NON_BREAKING_HYPHEN}Summary", "en"),
        ("Hospital Day-3", "en"),
        ("Hôpital de Jour", "fr"),
        ("Clínica da Unidade Básica", "pt"),
        ("Hospital Clinical Course", "en"),
    ],
)
def test_blocked_tokens_anywhere_in_the_candidate_keep_the_phrase_negative(text: str, language: str) -> None:
    assert _org_spans(text, language=language) == []


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Praxis Freitag Berlin", "de"),
        ("Praxis August Müller", "de"),
        ("Hospital Medicine Associates", "ms"),
    ],
)
def test_language_scoping_recovers_organization_shaped_names(text: str, language: str) -> None:
    assert _org_spans(text, language=language) == [(text, "company_name")]


def test_every_pack_compiles_into_a_rule_at_its_declared_tier() -> None:
    pack_rules = {rule.name: rule for rule in REGEX_RULES if rule.name.startswith("pack:")}

    assert set(pack_rules) == {f"pack:{pack.pack_id}" for pack in ALL_PACKS}
    assert all(rule.tier == "CONTEXT" for rule in pack_rules.values())
    assert all(rule.text_view == "composed" for rule in pack_rules.values())


def test_a_suffix_cue_captures_the_proper_name_run_and_cue() -> None:
    suffix = replace(VI_ORG_PACK, cues=(Cue("Medical Center", "post"),))
    match = pack_pattern(suffix).search("Seen at Sunnyvale Community Medical Center.")

    assert match is not None
    assert match.group("value") == "Sunnyvale Community Medical Center"


def test_pack_types_expose_only_declared_engine_states() -> None:
    assert set(get_args(get_type_hints(Cue)["position"])) == {"pre", "post"}
    assert "closing" not in {field.name for field in fields(Cue)}
    assert "cross_language" not in {field.name for field in fields(CuePack)}
    assert "known_language_only" not in {field.name for field in fields(CuePack)}
    assert "blocked_values" not in {field.name for field in fields(CuePack)}
    assert "cross_language" not in {field.name for field in fields(RegexRule)}
    assert "known_language_only" not in {field.name for field in fields(RegexRule)}


def test_a_pack_carries_cues_from_more_than_one_script() -> None:
    mixed = replace(VI_ORG_PACK, cues=(*VI_ORG_PACK.cues, Cue("Clinic", "pre")))
    match = pack_pattern(mixed).search("Referred to Clinic Hoa Sen today")

    assert match is not None
    assert match.group("value") == "Clinic Hoa Sen"


def test_a_decomposed_document_captures_at_offsets_into_its_own_bytes() -> None:
    text = unicodedata.normalize("NFD", "Chuyển đến Bệnh viện Nhi Đồng 1 để khám.")
    expected = unicodedata.normalize("NFD", "Bệnh viện Nhi Đồng 1")

    (span,) = apply_regex_postprocess(text, (), language="vi").spans

    assert span.label == "company_name"
    assert text[span.start : span.end] == expected


def test_one_decomposed_word_inside_a_composed_document_still_captures() -> None:
    text = "Chuyển đến Bệnh viện " + unicodedata.normalize("NFD", "Nhi Đồng") + " 1."
    expected = "Bệnh viện " + unicodedata.normalize("NFD", "Nhi Đồng") + " 1"

    (span,) = apply_regex_postprocess(text, (), language="vi").spans

    assert text[span.start : span.end] == expected


def test_manifest_hashes_pack_content_so_a_pack_edit_changes_the_address() -> None:
    components = regex_manifest()["component_sha256"]
    assert isinstance(components, dict)

    assert len(components["packs"]) == 64


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Chuyển đến Bệnh viện Đa khoa Sông Xanh để điều trị.",
            "Bệnh viện Đa khoa Sông Xanh",
        ),
        ("**Tên cơ sở:** Bệnh Viện Đa Khoa Trà Ôn", "Bệnh Viện Đa Khoa Trà Ôn"),
        (
            "Cơ sở chuyển đến: Bệnh viện Ung bướu Thành phố Hồ Chí Minh",
            "Bệnh viện Ung bướu Thành phố Hồ Chí Minh",
        ),
        ("Điều trị tại Bệnh viện 108 từ tháng trước.", "Bệnh viện 108"),
        ("Trung tâm Y tế huyện Tương Dương", "Trung tâm Y tế huyện Tương Dương"),
        ("Mua thuốc tại Nhà thuốc Long Châu.", "Nhà thuốc Long Châu"),
    ],
)
def test_prefix_cue_captures_the_institution_name_including_its_prefix(text: str, expected: str) -> None:
    assert _org_spans(text, language="vi") == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Verlegung in die Klinik Sonnenhain GmbH.", "Klinik Sonnenhain GmbH"),
        ("Aufnahme im Klinikum Stadtmitte erfolgte gestern.", "Klinikum Stadtmitte"),
        ("Befund vom MVZ Labor Leipzig liegt vor.", "MVZ Labor Leipzig"),
        ("Termin in der Praxis Dr. Meier vereinbart.", "Praxis Dr. Meier"),
        ("Entlassung aus dem Krankenhaus St. Josef.", "Krankenhaus St. Josef"),
    ],
)
def test_german_prefix_cue_captures_the_institution_name(text: str, expected: str) -> None:
    assert _org_spans(text, language="de") == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Transféré à l'Hôpital Saint-Louis hier.", "Hôpital Saint-Louis"),
        ("Suivi assuré par la Clinique du Soleil.", "Clinique du Soleil"),
        ("Adressé au Centre Hospitalier Val-Doire.", "Centre Hospitalier Val-Doire"),
        (
            "Consultation au Centre de Santé du Haut Languedoc.",
            "Centre de Santé du Haut Languedoc",
        ),
        ("Prélèvement au Laboratoire Sainte-Anne.", "Laboratoire Sainte-Anne"),
    ],
)
def test_french_prefix_cue_captures_the_institution_name(text: str, expected: str) -> None:
    assert _org_spans(text, language="fr") == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Ingresó en el Hospital Central de Referencia.",
            "Hospital Central de Referencia",
        ),
        ("Control posterior en la Clínica del Valle.", "Clínica del Valle"),
        (
            "Muestras enviadas al Laboratorio Clínico San José.",
            "Laboratorio Clínico San José",
        ),
        (
            "Atendido en el Centro Médico Aurora del Norte.",
            "Centro Médico Aurora del Norte",
        ),
        ("Póliza emitida por Clínica Valle Claro S.A.", "Clínica Valle Claro"),
    ],
)
def test_spanish_prefix_cue_captures_the_institution_name(text: str, expected: str) -> None:
    assert _org_spans(text, language="es") == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Encaminhado ao Hospital Santa Íris.", "Hospital Santa Íris"),
        ("Seguimento na Clínica Aurora do Vale.", "Clínica Aurora do Vale"),
        ("Exames colhidos no Laboratório VidaNova.", "Laboratório VidaNova"),
        ("Consulta na Clínica de Medicina Interna.", "Clínica de Medicina Interna"),
    ],
)
def test_portuguese_prefix_cue_captures_the_institution_name(text: str, expected: str) -> None:
    assert _org_spans(text, language="pt") == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dirujuk ke Rumah Sakit Sehat Selalu.", "Rumah Sakit Sehat Selalu"),
        (
            "Kontrol di Rumah Sakit Ibu dan Anak Bunda Sehat.",
            "Rumah Sakit Ibu dan Anak Bunda Sehat",
        ),
        ("Pasien dirawat di RSU Nusantara Sehat.", "RSU Nusantara Sehat"),
        ("Obat ditebus di Apotek Melati Indah.", "Apotek Melati Indah"),
    ],
)
def test_indonesian_prefix_cue_captures_the_institution_name(text: str, expected: str) -> None:
    assert _org_spans(text, language="id") == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dirujuk ke Hospital Damai Seroja.", "Hospital Damai Seroja"),
        ("Rawatan susulan di Klinik Sihat Harmoni.", "Klinik Sihat Harmoni"),
        ("Pemeriksaan di Pusat Perubatan Seri Murni.", "Pusat Perubatan Seri Murni"),
        ("Ubat diambil di Farmasi Seroja Maya.", "Farmasi Seroja Maya"),
    ],
)
def test_malay_prefix_cue_captures_the_institution_name(text: str, expected: str) -> None:
    assert _org_spans(text, language="ms") == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("laboratorium Anda", "id"),
        ("clinique OCR", "fr"),
        ("clínica OCR", "es"),
        ("klinik OCR", "ms"),
    ],
)
def test_lowercase_latin_cues_do_not_capture_one_artifact_token(text: str, language: str) -> None:
    assert _org_spans(text, language=language) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Isinugod sa Ospital ng Maynila.", "Ospital ng Maynila"),
        ("Nagpakonsulta sa Klinika ng San Isidro.", "Klinika ng San Isidro"),
        ("Bumili ng gamot sa Botika ng Bayan.", "Botika ng Bayan"),
    ],
)
def test_filipino_prefix_cue_captures_the_institution_name(text: str, expected: str) -> None:
    assert _org_spans(text, language="fil") == [(expected, "company_name")]


def test_english_suffix_cue_captures_the_institution_name_and_cue() -> None:
    text = "Follow-up with PCP at Sunnyvale Community Clinic."

    assert _org_spans(text, language="en") == [("Sunnyvale Community Clinic", "company_name")]


@pytest.mark.parametrize("legal_form", ["Inc", "LLC", "Ltd"])
def test_english_suffix_cue_keeps_a_terminal_legal_form(legal_form: str) -> None:
    expected = f"LabAccess Healthcare {legal_form}"

    assert _org_spans(f"{expected}.", language="en") == [(expected, "company_name")]


def test_english_suffix_cue_captures_corpus_backed_health_compounds() -> None:
    assert _org_spans("Campus Health Services", language="en") == [("Campus Health Services", "company_name")]
    assert _org_spans("Northbridge Health Systems", language="en") == [("Northbridge Health Systems", "company_name")]


def test_english_suffix_cue_never_emits_before_an_unconsumed_proper_token() -> None:
    assert _org_spans("LabAccess Healthcare Incorporated", language="en") == []


def test_japanese_scriptless_suffix_cue_captures_the_glued_name() -> None:
    assert _org_spans("紹介先は京都中央総合病院です。", language="ja") == [("京都中央総合病院", "company_name")]


def test_chinese_scriptless_suffix_cue_captures_the_glued_name() -> None:
    assert _org_spans("拟转至明澄医院。", language="zh") == [("明澄医院", "company_name")]


def test_korean_scriptless_suffix_cue_captures_the_glued_name() -> None:
    assert _org_spans("서울종합병원에서 진료받았다.", language="ko") == [("서울종합병원", "company_name")]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("서울가상메디병원", "ko"),
        ("医療法人はるか病院", "ja"),
        ("至仁医院", "zh"),
    ],
)
def test_scriptless_boundaries_do_not_split_name_internal_particles(text: str, language: str) -> None:
    assert _org_spans(text, language=language) == [(text, "company_name")]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("保险公司", "zh"),
        ("บริษัทประกัน", "th"),
        ("คลินิกส่งต่อ", "th"),
        ("பொது மருத்துவமனை", "ta"),
    ],
)
def test_scriptless_generic_compounds_stay_unredacted(text: str, language: str) -> None:
    assert _org_spans(text, language=language) == []


def test_japanese_suffix_cue_blocks_the_end_adjacent_generic_token() -> None:
    assert _org_spans("患者のかかりつけクリニック", language="ja") == []


def test_tamil_hyphen_does_not_restart_after_blocked_left_context() -> None:
    assert _org_spans("இந்த-அறிக்கை பரிந்துரைத்த மருத்துவமனை", language="ta") == []


def test_thai_scriptless_prefix_cue_captures_the_glued_name() -> None:
    assert _org_spans("ส่งต่อโรงพยาบาลสุขุมวิททันที", language="th") == [("โรงพยาบาลสุขุมวิท", "company_name")]


def test_lao_scriptless_prefix_cue_captures_the_glued_name() -> None:
    assert _org_spans("ໂຮງພະຍາບານສຸກສາວັດ.", language="lo") == [("ໂຮງພະຍາບານສຸກສາວັດ", "company_name")]


def test_tamil_scriptless_suffix_cue_captures_one_adjacent_name_token() -> None:
    assert _org_spans("மீனல் மருத்துவமனை.", language="ta") == [("மீனல் மருத்துவமனை", "company_name")]


def test_tamil_scriptless_suffix_keeps_bounded_internal_name_tokens() -> None:
    text = "தமிழ்நாடு அரசு மருத்துவமனை"

    assert _org_spans(text, language="ta") == [(text, "company_name")]


def test_thai_scriptless_prefix_keeps_a_trailing_local_numeral() -> None:
    text = "โรงพยาบาลเมตตาพระราม ๒"

    assert _org_spans(text, language="th") == [(text, "company_name")]


def test_russian_abbreviation_captures_only_the_quoted_name() -> None:
    assert _org_spans("Направлен в ГБУЗ «Клиника Северный Берег».", language="ru") == [
        ("Клиника Северный Берег", "company_name"),
    ]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("株式会社青葉", "ja"),
        ("云澜公司", "zh"),
        ("주식회사한빛", "ko"),
        ("บริษัทสุขภาพดี", "th"),
    ],
)
def test_scriptless_packs_honor_each_declared_cue_position(text: str, language: str) -> None:
    assert _org_spans(text, language=language) == [(text, "company_name")]


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("Acme Pharmacy", "en", "Acme Pharmacy"),
        ("Northwell Vista Health", "en", "Northwell Vista Health"),
        ("Harborview Medical Center", "en", "Harborview Medical Center"),
        ("青葉みらいクリニック", "ja", "青葉みらいクリニック"),
        ("深圳市中心医院", "zh", "深圳市中心医院"),
        ("华康仁和门诊部", "zh", "华康仁和门诊部"),
        ("서울대병원", "ko", "서울대병원"),
        ("한빛메디컬센터", "ko", "한빛메디컬센터"),
        ("โรงพยาบาลรามา", "th", "โรงพยาบาลรามา"),
        ("คลินิกสุขภาพสยาม", "th", "คลินิกสุขภาพสยาม"),
        ("ГКБ «Северный Мост»", "ru", "Северный Мост"),
        ("ஆரோக்கியம் மருத்துவமனை", "ta", "ஆரோக்கியம் மருத்துவமனை"),
        ("ໂຮງພະຍາບານສຸຂະພາບສີດາ", "lo", "ໂຮງພະຍາບານສຸຂະພາບສີດາ"),
        ("ຄລີນິກກາງເມືອງ", "lo", "ຄລີນິກກາງເມືອງ"),
    ],
)
def test_new_packs_capture_locked_corpus_shapes(text: str, language: str, expected: str) -> None:
    assert _org_spans(text, language=language) == [(expected, "company_name")]


def test_bare_rs_never_captures_even_in_an_indonesian_document() -> None:
    assert _org_spans("MRI menunjukkan RS Grade 2 changes.", language="id") == []
    assert _org_spans("Kontrol lanjutan di RS Sardjito.", language="id") == []


def test_multicharacter_indonesian_rs_cues_remain_supported() -> None:
    assert _org_spans("Pasien dirawat di RSU Nusantara Sehat.", language="id") == [("RSU Nusantara Sehat", "company_name")]


def test_two_coordinated_institutions_stay_two_spans() -> None:
    text = "Pusat Perubatan Seri Murni dan Klinik Sihat Harmoni."

    assert _org_spans(text, language="ms") == [
        ("Pusat Perubatan Seri Murni", "company_name"),
        ("Klinik Sihat Harmoni", "company_name"),
    ]


def test_a_name_initial_continues_the_capture_but_never_ends_it() -> None:
    text = "Termin in der Praxis Dr. J. Meier vereinbart."

    assert _org_spans(text, language="de") == [("Praxis Dr. J. Meier", "company_name")]


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("Bệnh viện K", "vi", "Bệnh viện K"),
        ("Bệnh viện K Trung Ương", "vi", "Bệnh viện K Trung Ương"),
        ("Bệnh viện E", "vi", "Bệnh viện E"),
        ("Hôpital St. Joseph", "fr", "Hôpital St. Joseph"),
        ("Clinique Ste. Marie", "fr", "Clinique Ste. Marie"),
        (
            "Praxis Dr. A. B. C. D. E. F. G. Meier",
            "de",
            "Praxis Dr. A. B. C. D. E. F. G. Meier",
        ),
        (
            "Clínica Valle Claro S. A. Paciente estable.",
            "es",
            "Clínica Valle Claro",
        ),
    ],
)
def test_single_letter_and_dotted_name_tokens_keep_exact_boundaries(text: str, language: str, expected: str) -> None:
    assert _org_spans(text, language=language) == [(expected, "company_name")]


@pytest.mark.parametrize(
    ("fixture_id", "expected"),
    [
        ("vi-single-letter-hospital-k-before-period", "Bệnh viện K"),
        ("vi-single-letter-hospital-e-before-period", "Bệnh viện E"),
    ],
)
def test_single_letter_hospital_names_end_before_sentence_punctuation(fixture_id: str, expected: str) -> None:
    fixture = next(fixture for fixture in POSITIVE_SELECTION_FIXTURES if fixture.fixture_id == fixture_id)

    assert _org_spans(fixture.text, language=fixture.language) == [(expected, "company_name")]


def test_single_letter_hospital_name_does_not_bridge_into_the_next_sentence() -> None:
    text = "Bệnh viện K. Bệnh nhân ổn định."

    assert _org_spans(text, language="vi") == [("Bệnh viện K", "company_name")]


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("Hôpital A. B. Patient stable.", "fr", []),
        (
            "Clínica Valle Claro S. L. Paciente estable.",
            "es",
            [("Clínica Valle Claro", "company_name")],
        ),
        (
            "Clínica Valle Claro S. A. Paciente estable.",
            "es",
            [("Clínica Valle Claro", "company_name")],
        ),
    ],
)
def test_dotted_runs_never_bridge_into_following_prose(text: str, language: str, expected: list[tuple[str, str]]) -> None:
    assert _org_spans(text, language=language) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("công ty ABC. Để", "công ty ABC"),
        ("công ty bảo hiểm ABC. Tôi", "công ty bảo hiểm ABC"),
        ("Công ty TNHH ABC. Mọi", "Công ty TNHH ABC"),
    ],
)
def test_uppercase_acronym_before_a_period_ends_the_organization(text: str, expected: str) -> None:
    assert _org_spans(text, language="vi") == [(expected, "company_name")]


def test_an_organization_capture_never_truncates_immediately_before_a_period() -> None:
    text = "Hôpital St. Joseph"

    assert all(
        candidate_end == len(text) or text[candidate_end] != "."
        for candidate_end in (candidate.span.end for candidate in regex_candidates(text, language="fr"))
    )


def test_a_trailing_legal_form_is_left_out_rather_than_cut_to_its_initial() -> None:
    text = "Suivi par le Laboratoire Nova Aurora S.A."

    assert _org_spans(text, language="fr") == [("Laboratoire Nova Aurora", "company_name")]


def test_capture_stops_at_a_field_boundary_rather_than_running_into_the_next_line() -> None:
    text = "**Tên cơ sở:** Bệnh viện Mỹ Tú\n**Địa chỉ:** Số 8, Đường Nguyễn Trãi"

    assert _org_spans(text, language="vi") == [("Bệnh viện Mỹ Tú", "company_name")]


@pytest.mark.parametrize(
    "text",
    [
        ("Công ty Điện tử ABC"),
        ("Bệnh viện Phổi Trung ương"),
        ("Bệnh viện Đa khoa Trung ương Hà Nội"),
    ],
)
def test_vietnamese_name_continuations_do_not_truncate_real_names(text: str) -> None:
    assert _org_spans(text, language="vi") == [(text, "company_name")]


def test_organization_inside_a_model_address_span_yields_to_the_model() -> None:
    text = "Địa chỉ: 12 Trường Chinh, Quận Tân Bình"
    address = CharSpan(9, len(text), text[9:], "address")

    result = apply_regex_postprocess(text, (address,), language="vi")

    assert result.spans == (address,)
    assert [conflict.reason for conflict in result.conflicts] == ["different_label_preserves_model"]


def test_partial_model_organization_span_widens_to_the_cued_capture() -> None:
    text = "Chuyển đến Bệnh viện Đa khoa Sông Xanh để điều trị."
    partial = CharSpan(29, 38, "Sông Xanh", "company_name")

    result = apply_regex_postprocess(text, (partial,), language="vi")

    assert result.spans == (CharSpan(11, 38, "Bệnh viện Đa khoa Sông Xanh", "company_name"),)
