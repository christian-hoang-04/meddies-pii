from __future__ import annotations

import unicodedata
from typing import Literal

import pytest

from meddies_pii.regex_runtime.language import detect_language


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("ja", "患者は青葉みらい病院を受診し、担当医から治療計画の説明を受けました。"),
        ("zh", "患者今天前往深圳市中心医院接受检查并讨论后续治疗方案。"),
        ("ko", "환자는 오늘 서울중앙병원에서 검사를 받고 치료 계획을 상담했습니다."),
        ("th", "ผู้ป่วยเข้ารับการตรวจที่โรงพยาบาลและได้รับคำแนะนำเรื่องการรักษา"),
        ("lo", "ຄົນເຈັບໄດ້ໄປກວດຢູ່ໂຮງພະຍາບານແລະໄດ້ຮັບຄຳແນະນຳການຮັກສາ"),
        ("my", "လူနာသည်ဆေးရုံတွင်စစ်ဆေးမှုခံယူပြီးကုသမှုအစီအစဉ်ကိုဆွေးနွေးခဲ့သည်"),
        (
            "ta",
            "நோயாளி மருத்துவமனையில் பரிசோதனை செய்து சிகிச்சை திட்டத்தை மருத்துவருடன் விவாதித்தார்",
        ),
        (
            "ru",
            "Пациент прошел обследование в городской больнице и обсудил лечение \N{CYRILLIC SMALL LETTER ES} врачом.",
        ),
    ],
)
def test_non_latin_script_identifies_the_language(language: str, text: str) -> None:
    assert detect_language(text) == language


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("患者は東京中央総合病院で検査を受けて治療計画を確認しました。", "ja"),
        ("환자는 서울 병원에서 漢字 검사 결과와 치료 계획을 상담했습니다.", "ko"),
        ("患者今天前往中心医院接受检查并讨论后续治疗方案。", "zh"),
    ],
)
def test_cjk_marker_precedence_resolves_mixed_text(text: str, language: str) -> None:
    assert detect_language(text) == language


def test_han_dominant_chinese_with_three_incidental_kana_remains_undetected() -> None:
    text = (
        "患者今天前往深圳市中心医院接受检查并讨论后续治疗方案。"
        "患者今天前往深圳市中心医院接受检查并讨论后续治疗方案。あいう"
    )

    assert detect_language(text) is None


def test_twenty_seven_han_and_three_kana_remains_undetected() -> None:
    assert detect_language("患者今天前往深圳市中心医院接受检查并讨论后續治療方案病あいう") is None


def test_han_does_not_break_a_japanese_korean_tiebreak() -> None:
    assert detect_language("漢" * 14 + "あいう" + "가나다라마") is None


def test_shared_han_does_not_break_a_qualified_japanese_korean_tiebreak() -> None:
    text = "漢" * 7 + "あ" * 7 + "가" * 6

    assert detect_language(text) is None


def test_dominant_hangul_wins_over_non_dominant_han_with_incidental_kana() -> None:
    text = "가" * 13 + "漢" * 6 + "あ"

    assert detect_language(text) == "ko"


def test_han_dominant_japanese_with_two_kana_remains_undetected() -> None:
    text = "患者東京中央総合病院診療計画確認検査結果担当医説明治療継続方針病棟退院後連携体制あい"

    assert detect_language(text) is None


def test_han_dominant_text_with_two_kana_does_not_fall_through_to_english() -> None:
    text = "漢" * 14 + "あい the and of with is was"

    assert detect_language(text) is None


def test_kana_rich_japanese_remains_detected() -> None:
    text = "患者は青葉みらい病院を受診し、担当医から治療計画の説明を受けました。"

    assert detect_language(text) == "ja"


def test_pure_han_chinese_remains_detected() -> None:
    text = "患者今天前往深圳市中心医院接受检查并讨论后续治疗方案。"

    assert detect_language(text) == "zh"


def test_isolated_han_character_does_not_override_dominant_latin_text() -> None:
    text = "The patient record uses the 漢 marker only as a reference code."

    assert detect_language(text) is None


def test_one_incidental_kana_character_does_not_override_english() -> None:
    text = "The patient is stable and ready for treatment with the care team あ today."

    assert detect_language(text) == "en"


def test_one_incidental_hangul_character_does_not_override_english() -> None:
    text = "The patient is stable and ready for treatment with the care team 가 today."

    assert detect_language(text) == "en"


def test_one_vietnamese_name_does_not_override_english() -> None:
    text = "The patient Nguyễn is stable and ready for treatment with the care team today."

    assert detect_language(text) == "en"


def test_equally_qualified_non_latin_scripts_fail_closed() -> None:
    assert detect_language("กขฃคฅฆงจฉชАБВГДЕЁЖЗИ") is None


def test_hangul_block_and_incidental_kana_fail_closed() -> None:
    assert detect_language("あいう漢字仮名가나다라마바사아자차카타파") is None


def test_mixed_script_document_does_not_fall_through_to_latin_scoring() -> None:
    text = "Patient record fields o e do com are stored with မြန်မာဆေးရုံ data."

    assert detect_language(text) is None


@pytest.mark.parametrize("normalization", ["NFC", "NFD"])
def test_vietnamese_diacritics_identify_vietnamese(normalization: Literal["NFC", "NFD"]) -> None:
    text = "Bệnh nhân được chuyển đến bệnh viện để tiếp tục điều trị và theo dõi."

    assert detect_language(unicodedata.normalize(normalization, text)) == "vi"


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("de", "Der Patient ist stabil und benötigt die Behandlung nicht mehr."),
        ("fr", "Le patient est stable et vous pouvez poursuivre avec le traitement."),
        ("es", "El paciente está estable y usted puede continuar con el tratamiento."),
        ("pt", "O paciente está estável e você pode continuar com o tratamento."),
        (
            "id",
            "Pasien yang stabil tidak dirawat karena dokter memastikan bahwa ia pulih.",
        ),
        (
            "ms",
            "Pesakit ialah stabil kerana rawatan telah diberikan daripada klinik itu.",
        ),
        ("fil", "Ang pasyente ay dinala sa klinika ng bayan para sa mga pagsusuri."),
        ("en", "The patient is stable and ready for the next stage of treatment."),
    ],
)
def test_latin_function_words_identify_the_language(language: str, text: str) -> None:
    assert detect_language(text) == language


@pytest.mark.parametrize(
    "text",
    [
        "Hospital Problem List",
        "Patient stable today after review",
        "the patient et le traitement",
    ],
)
def test_short_or_tied_latin_text_remains_undetected(text: str) -> None:
    assert detect_language(text) is None


def test_repeated_single_letter_url_tokens_do_not_infer_portuguese() -> None:
    text = "https://resultados.clinica-arboleda.test/ordenes/O-91rn7?token=tok_8O5l-zz&source=app"

    assert detect_language(text) is None


def test_incidental_short_tokens_do_not_infer_portuguese() -> None:
    text = "Patient record fields o e do com are stored as identifiers."

    assert detect_language(text) is None


def test_short_english_organization_name_remains_undetected() -> None:
    assert detect_language("Lung Center of the Philippines") is None
