from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: these characters sit inside patterns that must MATCH them — the symbol-substitution and
# reason: separator detectors exist to catch confusable punctuation, so normalising one to ASCII
# reason: would silently stop this module detecting that evasion.
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from meddies_pii.spans import CharSpan

SliceFilterReport = dict[str, dict[str, Any]]

MIN_STRUCTURE_LINES = 2
MIN_KEY_VALUE_PAIRS = 3

ADVERSARIAL_SLICE_NAMES: tuple[str, ...] = (
    "all",
    "line_breaks",
    "spacing",
    "at_dot_obfuscation",
    "symbol_substitution",
    "emoji_word_replacement",
    "emoji_symbol_replacement",
    "phonetic_alphabet",
    "digit_words",
    "minimal_name_fragment",
    "repeated_surface",
    "structure_heavy",
)

_NATIVE_AT = (
    r"a\s*c[oò]ng|a\s*m[oó]c|小老鼠|艾特|골뱅이|собач(?:ка|ий)?|アットマーク|アット"
    r"|ก้นหอย|แอท|arobase|klammeraffe|arroba"
)
"""Native spoken/written 'at'/'dot' markers for voice-scribe/OCR text across the 17 supported languages.

Vi/zh/ko/ru/ja shown; English keeps the original forms).

17-lang spoken '@': vi zh ko ru ja th fr de es pt + variants.

"""
_NATIVE_DOT = r"ch[aấ]m(?:\s*com)?|點|点|점|точка|ドット|จุด|புள்ளி|ຈຸດ"
"""17-lang spoken '.': vi zh ko ru ja th fr de es pt id ms ta lo + variants.

Latin dot-words (point/punkt/punto/ponto/titik) dropped: they match ordinary text ('appointment' contains 'point').
Native-script markers are unambiguous.

"""
_AT_DOT_RE = re.compile(
    r"(?:\[(?:at|dot)\]|\((?:at|dot)\)|\b\w+\s+at\s+\w+\s+dot\s+\w+"
    r"|" + _NATIVE_AT + r"|" + _NATIVE_DOT + r")",
    re.IGNORECASE | re.UNICODE,
)
_SYMBOL_SUBSTITUTION_RE = re.compile(r"[＠﹫．｡。•·●○▪▫◦⁃–—−＿*#]")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_PHONETIC_ALPHABET_RE = re.compile(
    r"\b(?:alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|india|juliett|"
    r"kilo|lima|mike|november|oscar|papa|quebec|romeo|sierra|tango|uniform|"
    r"victor|whiskey|xray|yankee|zulu)\b"
    r"(?:[ ._-]+\b(?:alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|"
    r"india|juliett|kilo|lima|mike|november|oscar|papa|quebec|romeo|"
    r"sierra|tango|uniform|victor|whiskey|xray|yankee|zulu)\b){2,}",
    re.IGNORECASE,
)
_LATIN_DIGIT = r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"
"""Spoken digit sequences (3+).

English + native number-words for the primary deployment languages, so voice-transcribed phone/ID surfaces slice correctly.

"""
_FR_DIGIT = r"(?:zéro|un|deux|trois|quatre|cinq|six|sept|huit|neuf)"
_DE_DIGIT = r"(?:null|eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun)"
_ES_DIGIT = r"(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)"
_PT_DIGIT = r"(?:zero|um|dois|três|quatro|cinco|seis|sete|oito|nove)"
_ID_DIGIT = r"(?:nol|kosong|satu|dua|tiga|empat|lima|enam|tujuh|delapan|lapan|sembilan)"
_VI_DIGIT = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|linh|lẻ)"
_RU_DIGIT = r"(?:ноль|один|два|три|четыре|пять|шесть|семь|восемь|девять)"
_KO_DIGIT = r"(?:영|공|일|이|삼|사|오|육|칠|팔|구)"
_ZH_DIGIT = r"[零〇一二三四五六七八九]"
_ALPHA_DIGIT_WORD = (
    r"(?:" + f"{_LATIN_DIGIT}|{_VI_DIGIT}|{_RU_DIGIT}|{_FR_DIGIT}|{_DE_DIGIT}|{_ES_DIGIT}|{_PT_DIGIT}|{_ID_DIGIT}" + r")"
)
"""Union of the word-boundary-using (alphabetic-script) digit words, for the mixed-form branch below.

CJK digits have no word boundary and are covered by their own same-script runs.

"""
_DIGIT_WORD_RE = re.compile(
    r"(?:"
    r"\b" + _LATIN_DIGIT + r"\b(?:[\s,-]+\b" + _LATIN_DIGIT + r"\b){2,}"
    r"|\b" + _VI_DIGIT + r"\b(?:[\s,-]+\b" + _VI_DIGIT + r"\b){2,}"
    r"|\b" + _RU_DIGIT + r"\b(?:[\s,-]+\b" + _RU_DIGIT + r"\b){2,}"
    r"|" + _KO_DIGIT + r"(?:[\s,-]*" + _KO_DIGIT + r"){2,}"
    r"|" + _ZH_DIGIT + r"(?:[\s,、-]*" + _ZH_DIGIT + r"){2,}"
    r"|\b" + _FR_DIGIT + r"\b(?:[\s,-]+\b" + _FR_DIGIT + r"\b){2,}"
    r"|\b" + _DE_DIGIT + r"\b(?:[\s,-]+\b" + _DE_DIGIT + r"\b){2,}"
    r"|\b" + _ES_DIGIT + r"\b(?:[\s,-]+\b" + _ES_DIGIT + r"\b){2,}"
    r"|\b" + _PT_DIGIT + r"\b(?:[\s,-]+\b" + _PT_DIGIT + r"\b){2,}"
    r"|\b" + _ID_DIGIT + r"\b(?:[\s,-]+\b" + _ID_DIGIT + r"\b){2,}"
    r"|\b" + _ALPHA_DIGIT_WORD + r"\b[\s,-]+\d"
    r"|\d[\s,-]+\b" + _ALPHA_DIGIT_WORD + r"\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)
"""mixed form.

A spelled digit-word adjacent to an Arabic numeral, either order ("012 ba bốn", "one 2 three") — a phone/id written
half-numeral, half-spoken, which the same-script word runs above miss.

"""
_SPACING_RE = re.compile(
    r"(?:\b(?:tok|sk|api|key)[_\-]?\s+[A-Za-z0-9]|"
    r"[A-Za-z0-9]{2,}\s+[._\-]\s*[A-Za-z0-9]{2,}|"
    r"@\s*\n|\.\s*\n)",
)
_MINIMAL_NAME_FRAGMENT_RE = re.compile(r"\b[A-Z]\.\s*[A-Z][a-z]{1,20}\b")
_KEY_VALUE_RE = re.compile(r"(?m)\b[\w -]{2,32}\s*[:=]\s*\S+")


# reason: classify coordinates search with add; extra seams would fork shared metric totals.
def classify_adversarial_slices(  # ruff: ignore[complex-structure]
    text: str,
    spans: Sequence[CharSpan] = (),
) -> tuple[str, ...]:
    """Assign source-derived adversarial slice labels.

    The labels are intentionally based only on source text and gold spans. They
    must not depend on model predictions; otherwise slice reports can leak the
    outcome and become another form of reward hacking.

    Returns:
        Every slice label the row belongs to, always including ``"all"``, so a row is never
        in zero slices. Membership is derived ONLY from the source text and gold spans --
        adding a predictions-derived label here would make every slice metric self-confirming.

    """
    present = {"all"}
    if "\n" in text or "\r" in text:
        present.add("line_breaks")
    if _SPACING_RE.search(text):
        present.add("spacing")
    if _AT_DOT_RE.search(text):
        present.add("at_dot_obfuscation")
    if _SYMBOL_SUBSTITUTION_RE.search(text):
        present.add("symbol_substitution")
    if _EMOJI_RE.search(text):
        present.add("emoji_word_replacement")
        present.add("emoji_symbol_replacement")
    if _PHONETIC_ALPHABET_RE.search(text):
        present.add("phonetic_alphabet")
    if _DIGIT_WORD_RE.search(text):
        present.add("digit_words")
    if _MINIMAL_NAME_FRAGMENT_RE.search(text):
        present.add("minimal_name_fragment")

    normalized_surfaces = [" ".join(span.text.casefold().split()) for span in spans if span.text and span.text.strip()]
    surface_counts = Counter(normalized_surfaces)
    if any(count > 1 for count in surface_counts.values()):
        present.add("repeated_surface")

    line_count = text.count("\n") + text.count("\r")
    key_value_count = len(_KEY_VALUE_RE.findall(text))
    # reason: classify keeps text/line count in one gate; helper predicates would scatter the rule.
    if (
        "\t" in text  # ruff: ignore[too-many-boolean-expressions]
        or "|" in text
        or "{" in text
        or "}" in text
        or line_count >= MIN_STRUCTURE_LINES
        or key_value_count >= MIN_KEY_VALUE_PAIRS
    ):
        present.add("structure_heavy")

    return tuple(name for name in ADVERSARIAL_SLICE_NAMES if name in present)


def new_slice_filter_report() -> SliceFilterReport:
    return {
        name: {
            "attempted_docs": 0,
            "source_accepted_docs": 0,
            "source_filtered_docs": 0,
            "source_rejection_reasons": {},
            "preparation_attempted_docs": 0,
            "prepared_docs": 0,
            "preparation_filtered_docs": 0,
            "preparation_rejection_reasons": {},
        }
        for name in ADVERSARIAL_SLICE_NAMES
    }


def _increment_reason(counter: MutableMapping[str, int], reason: str) -> None:
    counter[reason] = int(counter.get(reason, 0)) + 1


def record_slice_event(
    report: SliceFilterReport | None,
    slices: Sequence[str],
    *,
    stage: str,
    reason: str | None = None,
) -> None:
    """Record source/preparation filtering counts for each row slice.

    A ``None`` report and a slice absent from the report are both no-ops, so callers can
    instrument unconditionally without checking whether reporting is enabled.

    Raises:
        ValueError: On an unrecognised ``stage``. The stage set is closed deliberately: a
            typo'd stage would otherwise silently record nothing and the slice counts would
            under-report with no signal that a stage went missing.

    """
    if report is None:
        return
    for slice_name in slices:
        if slice_name not in report:
            continue
        bucket = report[slice_name]
        if stage == "attempted":
            bucket["attempted_docs"] = int(bucket.get("attempted_docs", 0)) + 1
        elif stage == "source_accepted":
            bucket["source_accepted_docs"] = int(bucket.get("source_accepted_docs", 0)) + 1
        elif stage == "source_rejected":
            bucket["source_filtered_docs"] = int(bucket.get("source_filtered_docs", 0)) + 1
            _increment_reason(bucket["source_rejection_reasons"], reason or "unknown")
        elif stage == "preparation_attempted":
            bucket["preparation_attempted_docs"] = int(bucket.get("preparation_attempted_docs", 0)) + 1
        elif stage == "prepared":
            bucket["prepared_docs"] = int(bucket.get("prepared_docs", 0)) + 1
        elif stage == "preparation_rejected":
            bucket["preparation_filtered_docs"] = int(bucket.get("preparation_filtered_docs", 0)) + 1
            _increment_reason(bucket["preparation_rejection_reasons"], reason or "unknown")
        else:
            msg = f"Unsupported slice filter stage: {stage!r}"
            raise ValueError(msg)
