from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: this module's confusables ARE its subject matter — the adversarial PII corpus names the
# reason: evasion it must generate, so normalising them to ASCII would delete the technique itself.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random

    from meddies_pii.generation.label_corpus.catalog import Scenario


EDGE_CASE_RATE = 0.3
EDGE_CASES: tuple[str, ...] = (
    (
        "Write one email's '@' and '.' as this language's native spoken words (Vietnamese 'a "
        "còng'/'chấm com', Chinese '小老鼠'/'點', Japanese 'アットマーク'/'ドット', Korean '골뱅이'/'점', or a "
        "generic ' at '/' dot '), keeping the [value]<email_address> tag around the whole spoken "
        "form."
    ),
    (
        "Spell one phone or id_number's digits as this language's number-words, realistically "
        "mixing numerals and number-words rather than converting them all (e.g. English '012 "
        "three four five 67 eight nine', or the native equivalent; a clean full spell-out like "
        "'oh-four-one-two' is the simpler variant), tagging the full span by what the number is — "
        "a contact number is <phone_number>, an identifier is <id_number>."
    ),
    (
        "Spell one id_number, code, or token with a phonetic / NATO or this language's spelling "
        "alphabet (e.g. 'Alpha-Four-Bravo' or the native equivalent), tagging the full "
        "spelled-out span."
    ),
    (
        "Split one phone / email / URL / id_number across a line break inside the tagged value, "
        "the way a wrapped fax, column, or form field breaks it."
    ),
    (
        "Put odd internal spacing inside one value (e.g. '04 12 3 4 56 78', 'j smith @ example . "
        "com'), keeping the tag around the whole spaced span."
    ),
    (
        "Run one value directly into an adjacent value or word with no separator (e.g. "
        "'NguyenVanA0912345678', 'P-1042status=active'), and give EACH value its own exact "
        "[value]<label> span with no shared or overlapping characters."
    ),
    (
        "Truncate one value mid-string the way a column width, screenshot, or log limit clips it "
        "(e.g. 'patient.name@exa...', 'P-100482' cut to 'P-1004...'), tagging only the visible "
        "remaining surface."
    ),
    (
        "Replace the separators in one value with full-width or look-alike punctuation (＠ for @, "
        "． ｡ 。 for the dot, – — for hyphens), keeping the tag around the whole value."
    ),
    (
        "Swap one or two Latin letters in a value for unicode confusables that render almost "
        "identically (Cyrillic 'о а е р с х' or Greek look-alikes inside a Latin name / email / "
        "id_number), keeping the tag around the whole value."
    ),
    (
        "Add OCR character confusion (l<->1, O<->0, 5<->S, rn<->m, B<->8) inside one id_number / "
        "email / URL, as a scanner would misread it."
    ),
    (
        "Use an emoji in place of a separator or label word near one value (e.g. name 📧 email, ☎ "
        "before a phone, 🔑 before a token), tagging only the real PII value and not the emoji."
    ),
    "Use a minimal name fragment for one human_name (given-name only, initials, or a one-token surname).",
    (
        "Partially redact one value so only some real characters survive (e.g. phone "
        "'098****4567', email 'j***@example.com', id '****-1042'), and tag the entire "
        "partially-masked value because it is still PII."
    ),
)
ADVERSARIAL_EDGE_CASE_RANGE = (4, 5)


def sample_edge_cases(rng: random.Random, scenario: Scenario) -> tuple[str, ...]:
    if scenario.adversarial:
        count = min(rng.randint(*ADVERSARIAL_EDGE_CASE_RANGE), len(EDGE_CASES))
        return tuple(rng.sample(EDGE_CASES, count))
    if rng.random() < EDGE_CASE_RATE:
        return (rng.choice(EDGE_CASES),)
    return ()
