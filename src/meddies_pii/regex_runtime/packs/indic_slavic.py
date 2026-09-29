"""Tamil and Russian organization-pack declarations."""

from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: the Russian pack declares exact Cyrillic organization cues and a non-breaking name
# reason: separator. ASCII substitutions would change which source text the pack matches.
from meddies_pii.regex_runtime.packs.pack import Cue, CuePack

TA_ORG_PACK = CuePack(
    pack_id="ta_org_scriptless",
    version=1,
    language="ta",
    label="company_name",
    tier="CONTEXT",
    cues=(Cue("மருத்துவமனை", "post"),),
    head_letters="",
    body_letters="",
    name_continuations=(),
    blocked_adjacent_tokens=(
        "பொது",
        "மற்றும்",
        "பெறுநர்",
        "அனுப்புநர்",
        "மேலும்",
        "பொதுத்",
        "பதிவு",
        "நீங்கள்",
        "நினைவில்",
        "நகர",
        "எனது",
        "இந்த",
        "அறிகுறிகள்",
    ),
    max_name_tokens=0,
    min_adjacent_digits=0,
    capture_mode="scriptless",
    script_ranges=((0x0B80, 0x0BFF),),
    max_name_chars=16,
    script_name_separator=True,
    max_script_name_tokens=3,
)


_LOWERCASE = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_UPPERCASE = _LOWERCASE.upper()

RU_ORG_PACK = CuePack(
    pack_id="ru_org_quoted",
    version=1,
    language="ru",
    label="company_name",
    tier="CONTEXT",
    cues=tuple(Cue(text, "pre") for text in ("ГБУЗ", "ГКБ", "ООО", "АО")),
    head_letters=_UPPERCASE,
    body_letters=_LOWERCASE + _UPPERCASE + "-‑",
    name_continuations=(),
    blocked_adjacent_tokens=(),
    max_name_tokens=8,
    min_adjacent_digits=2,
    capture_mode="quoted",
    opening_delimiter="«",
    closing_delimiter="»",
)
