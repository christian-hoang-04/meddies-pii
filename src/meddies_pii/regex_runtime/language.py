"""Deterministically infer a supported language from text.

Tier 1 requires script-specific absolute counts and document share. Japanese needs
enough kana, while Han-dominant text with kana that falls below that bar is ambiguous
and fails closed instead of becoming Chinese. Tier 2 requires several Vietnamese
markers and their share. Tier 3 scores small Latin-language function-word sets and
requires both enough evidence and a clear margin. Short or ambiguous text remains
undetected.
"""

from __future__ import annotations

import unicodedata
from operator import itemgetter

MIN_LETTERS = 20
MIN_SCRIPT_LETTERS = 5
MIN_SCRIPT_SHARE = 0.30
MIN_KANA_LETTERS = 3
MIN_JAPANESE_KANA_SHARE = MIN_SCRIPT_SHARE
MIN_VIETNAMESE_DIACRITICS = 3
MIN_VIETNAMESE_SHARE = 0.15


def _in_ranges(character: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in ranges)


_KANA_RANGES = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D))
_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x323AF),
)
_HANGUL_RANGES = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7AF),
    (0xD7B0, 0xD7FF),
)

_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "th": ((0x0E00, 0x0E7F),),
    "lo": ((0x0E80, 0x0EFF),),
    "my": ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)),
    "ta": ((0x0B80, 0x0BFF),),
    "ru": ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
}

_VIETNAMESE_DISTINCT_LETTERS = frozenset("ăĂđĐơƠưƯ")
_VIETNAMESE_TONES = frozenset("\u0300\u0301\u0303\u0309\u0323")
_VIETNAMESE_VOWEL_MARKS = frozenset("\u0302\u0306\u031b")

_LATIN_FUNCTION_WORDS: dict[str, tuple[str, ...]] = {
    "de": ("der", "die", "das", "und", "nicht", "ist", "im"),
    "fr": ("le", "les", "et", "vous", "avec", "dans", "est"),
    "es": ("el", "los", "y", "usted", "del", "con", "para"),
    "pt": ("o", "os", "e", "você", "do", "com", "uma"),
    "id": ("yang", "tidak", "bahwa", "karena", "dengan", "adalah"),
    "ms": ("ialah", "bahawa", "kerana", "daripada", "kepada", "boleh", "telah"),
    "fil": ("ang", "ng", "mga", "sa", "ay", "para"),
    "en": ("the", "and", "of", "with", "is", "was", "for"),
}
_MIN_LATIN_SCORE = 3
_LATIN_SCORE_MARGIN = 3
_MIN_LATIN_WORDS = 6
_MIN_SUBSTANTIAL_LATIN_WORDS = 2
_MIN_SUBSTANTIAL_LATIN_WORD_LENGTH = 3


def _has_vietnamese_diacritic(character: str) -> bool:
    if character in _VIETNAMESE_DISTINCT_LETTERS:
        return True
    marks = frozenset(unicodedata.normalize("NFD", character)[1:])
    return "\u0323" in marks or bool(marks & _VIETNAMESE_TONES and marks & _VIETNAMESE_VOWEL_MARKS)


def _words(text: str) -> tuple[str, ...]:
    words: list[str] = []
    current: list[str] = []
    for character in text.casefold():
        if character.isalpha():
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return tuple(words)


# reason: language precedence is one ordered fail-closed policy; extraction risks changing fall-through order.
def detect_language(text: str) -> str | None:  # ruff: ignore[complex-structure,too-many-return-statements,too-many-locals]
    """Return a supported language code when deterministic evidence is sufficient.

    Returns:
        The supported language code, or `None` when deterministic evidence is insufficient.

    """
    normalized = unicodedata.normalize("NFC", text)
    letters = tuple(character for character in normalized if character.isalpha())
    if len(letters) < MIN_LETTERS:
        return None

    kana_count = sum(_in_ranges(character, _KANA_RANGES) for character in letters)
    han_count = sum(_in_ranges(character, _HAN_RANGES) for character in letters)
    hangul_count = sum(_in_ranges(character, _HANGUL_RANGES) for character in letters)
    japanese_qualified = kana_count >= MIN_KANA_LETTERS and _has_script_share(
        kana_count,
        letters,
        minimum_share=MIN_JAPANESE_KANA_SHARE,
    )
    han_qualified = han_count >= MIN_SCRIPT_LETTERS and _has_script_share(han_count, letters)
    hangul_qualified = hangul_count >= MIN_SCRIPT_LETTERS and _has_script_share(hangul_count, letters)
    counts = {
        language: sum(_in_ranges(character, ranges) for character in letters)
        for language, ranges in _SCRIPT_RANGES.items()
    }
    qualified_non_han_counts = [
        count
        for count in ([hangul_count] if hangul_qualified else []) + list(counts.values())
        if count >= MIN_SCRIPT_LETTERS and _has_script_share(count, letters)
    ]
    strongest_non_han_count = max(qualified_non_han_counts, default=0)
    han_is_dominant_or_near_tied = han_count >= strongest_non_han_count - len(letters) * MIN_SCRIPT_SHARE
    if han_qualified and kana_count and not japanese_qualified and han_is_dominant_or_near_tied:
        return None
    if kana_count >= MIN_KANA_LETTERS and hangul_qualified and not japanese_qualified:
        return None

    script_candidates: list[tuple[str, int]] = []
    if japanese_qualified:
        script_candidates.append(("ja", kana_count))
    if hangul_qualified:
        script_candidates.append(("ko", hangul_count))
    if han_qualified and not kana_count:
        counts["zh"] = han_count
    script_candidates.extend(
        (language, count)
        for language, count in counts.items()
        if count >= MIN_SCRIPT_LETTERS and _has_script_share(count, letters)
    )
    if script_candidates:
        ranked_scripts = sorted(script_candidates, key=itemgetter(1), reverse=True)
        language, count = ranked_scripts[0]
        if len(ranked_scripts) > 1 and count - ranked_scripts[1][1] <= len(letters) * MIN_SCRIPT_SHARE:
            return None
        return language
    vietnamese_count = sum(_has_vietnamese_diacritic(character) for character in letters)
    if vietnamese_count >= MIN_VIETNAMESE_DIACRITICS and _has_script_share(
        vietnamese_count,
        letters,
        minimum_share=MIN_VIETNAMESE_SHARE,
    ):
        return "vi"

    words = _words(normalized)
    if len(words) < _MIN_LATIN_WORDS:
        return None
    distinct_words = frozenset(words)
    scores = {
        candidate: len(distinct_words.intersection(stopwords)) for candidate, stopwords in _LATIN_FUNCTION_WORDS.items()
    }
    ranked = sorted(scores.items(), key=itemgetter(1), reverse=True)
    (best_language, best_score), (_, runner_up_score) = ranked[:2]
    substantial_words = {
        word
        for word in distinct_words.intersection(_LATIN_FUNCTION_WORDS[best_language])
        if len(word) >= _MIN_SUBSTANTIAL_LATIN_WORD_LENGTH
    }
    if (
        best_score >= _MIN_LATIN_SCORE
        and best_score - runner_up_score >= _LATIN_SCORE_MARGIN
        and len(substantial_words) >= _MIN_SUBSTANTIAL_LATIN_WORDS
    ):
        return best_language
    return None


def _has_script_share(count: int, letters: tuple[str, ...], *, minimum_share: float = MIN_SCRIPT_SHARE) -> bool:
    return count / len(letters) >= minimum_share
