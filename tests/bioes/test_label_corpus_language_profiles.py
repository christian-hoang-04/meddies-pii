from __future__ import annotations

import pytest

from meddies_pii.generation.label_corpus.validate import (
    _marker_re_for,
    _min_raw_length_for,
)
from meddies_pii.languages import LANGUAGE_PROFILES, SUPPORTED_LANGUAGES, normalize_language


def test_language_profiles_cover_every_supported_language() -> None:
    profile_names = {profile.name for profile in LANGUAGE_PROFILES.values()}
    assert profile_names == set(SUPPORTED_LANGUAGES)
    assert len(LANGUAGE_PROFILES) == len(SUPPORTED_LANGUAGES) == 17


@pytest.mark.parametrize("language_name", SUPPORTED_LANGUAGES)
def test_each_supported_language_resolves_to_profile(language_name: str) -> None:
    profile = normalize_language(language_name)
    assert profile.name == language_name
    assert profile.code
    assert profile.country_hint
    assert profile.public_negative_hint
    assert profile.adversarial_hint


def test_profile_keys_match_lowercased_names() -> None:
    for key, profile in LANGUAGE_PROFILES.items():
        assert key == profile.name.lower()


def test_language_codes_are_unique() -> None:
    codes = [profile.code for profile in LANGUAGE_PROFILES.values()]
    assert len(codes) == len(set(codes))


@pytest.mark.parametrize(
    ("code", "native", "foreign"),
    [
        ("th", "สวัสดีโรงพยาบาล", "hospital"),
        ("lo", "ໂຮງໝໍສະບາຍດີ", "hospital"),
        ("my", "ဆေးရုံမင်္ဂလာပါ", "hospital"),
        ("ta", "மருத்துவமனை வணக்கம்", "hospital"),
        ("ru", "больница приветствие", "hospital"),
        ("ko", "병원 안녕하세요", "hospital"),
        ("zh", "病院 上海医院", "hospital"),
        ("ja", "びょういん こんにちは", "hospital"),
    ],
)
def test_marker_matches_native_script_and_rejects_latin(code: str, native: str, foreign: str) -> None:
    marker = _marker_re_for(code)
    assert marker is not None
    assert marker.search(native) is not None
    assert marker.search(foreign) is None


def test_cjk_marker_matches_han_and_rejects_latin() -> None:
    marker = _marker_re_for("zh")
    assert marker is not None
    assert marker.search("病院") is not None
    assert marker.search("hospital") is None


def test_japanese_marker_matches_kana_but_not_plain_latin() -> None:
    marker = _marker_re_for("ja")
    assert marker is not None
    assert marker.search("カタカナ") is not None
    assert marker.search("ひらがな") is not None
    assert marker.search("hospital") is None


def test_latin_script_languages_have_no_strict_marker() -> None:
    for code in ("en", "fr", "de", "es", "pt", "id", "fil", "ms"):
        assert _marker_re_for(code) is None


def test_cjk_char_gate_is_lower_than_default() -> None:
    default = _min_raw_length_for("en")
    for code in ("zh", "ja", "ko"):
        assert _min_raw_length_for(code) < default
    assert _min_raw_length_for("vi") == default
    assert _min_raw_length_for("th") == default
