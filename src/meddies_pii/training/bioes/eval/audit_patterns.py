from __future__ import annotations

# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: these characters sit inside patterns that must MATCH them — the symbol-substitution and
# reason: separator detectors exist to catch confusable punctuation, so normalising one to ASCII
# reason: would silently stop this module detecting that evasion.
import re

MIN_DISTINCT_DATE_MARKERS = 2
MAX_DATE_SURFACE_LENGTH = 64

DATE_RE = re.compile(
    r"(?iu)(?:"
    r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:\d{2}|\d{4})\b"
    r"|\b(?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])\b"
    r")",
)
ISO_DATETIME_RE = re.compile(
    r"(?iu)\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b",
)
DOT_DATE_RE = re.compile(r"(?iu)\b(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.(?:\d{2}|\d{4})\b")
COMPACT_DATE_RE = re.compile(
    r"(?iu)\b(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
    r"(?:(?:[01]\d|2[0-3])(?:[0-5]\d){1,2})?\b",
)
TEXTUAL_DATE_RE = re.compile(
    r"(?iu)\b(?:"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|"
    r"october|november|december"
    r")\.?\s+\d{1,2},?\s+\d{4}\b",
)
LOCALIZED_TEXTUAL_DATE_RE = re.compile(
    r"(?iu)\b\d{1,2}\s+(?:"
    r"jan|january|januari|"
    r"feb|february|februari|"
    r"mar|march|maret|mac|"
    r"apr|april|"
    r"may|mei|"
    r"jun|june|juni|"
    r"jul|july|juli|"
    r"aug|august|agustus|ogo|ogos|"
    r"sep|sept|september|"
    r"oct|okt|october|oktober|"
    r"nov|november|"
    r"dec|des|dis|december|"
    r"tháng\s+\d{1,2}"
    r")\.?(?:\s+năm)?\s+\d{4}\b",
)
_CJK_NUM = r"[零〇一二三四五六七八九\d]"
"""Native-script date surfaces for the non-Latin deployment languages.

Every alternative is STRUCTURE-ANCHORED: it requires a day/month/year marker (a calendar word or month name), never a bare
run of native digits. A phone or ID spoken/written as native digit-words therefore never matches — only genuine
day+month+year structure does. Unicode digit ranges are verified per script: zh/ja CJK numerals (U+96F6 etc.); th
U+0E50-59; my U+1040-49; lo U+0ED0-D9; ta U+0BE6-EF (the special day/month/year symbols U+0BF3-F5 are also markers).

"""
_NATIVE_DATE_CJK = (
    r"(?:令和|平成|昭和|大正|明治)?"
    + _CJK_NUM
    + r"{1,4}年[\s,-]*"
    + _CJK_NUM
    + r"{1,2}月(?:[\s,-]*"
    + _CJK_NUM
    + r"{1,2}[日号號])?"
)
"""zh/ja: YYYY年MM月DD(日|号|號) big-endian, optional ja imperial-era prefix."""
_NATIVE_DATE_KO = r"\d{1,4}년[\s,-]*\d{1,2}월(?:[\s,-]*\d{1,2}일)?"
"""ko: YYYY년 MM월 DD일 (Arabic digits in calendar dates)."""
_TH_DIGIT = "\u0e50-\u0e59"
r"""Native digit ranges written as \uXXXX escapes.

Not the literal native-zero glyph (which trips ruff RUF001: ambiguous-with-Latin-o) inside a class.

"""
_TH_D = rf"(?:[{_TH_DIGIT}]{{1,2}}|\d{{1,2}})"
_TH_Y = rf"(?:[{_TH_DIGIT}]{{2,4}}|\d{{2,4}})"
_TH_MONTH = (
    r"มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|"
    r"กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม"
)
_NATIVE_DATE_TH = (
    _TH_D
    + r"[\s,-]+(?:"
    + _TH_MONTH
    + r")[\s,-]+(?:ปี[\s,-]+)?"
    + _TH_Y
    + r"|"
    + _TH_D
    + r"[\s,-]+เดือน[\s,-]+"
    + _TH_D
    + r"[\s,-]+ปี[\s,-]+"
    + _TH_Y
)
_MY_D = r"(?:[\u1040-\u1049]{1,2}|\d{1,2})"
_MY_Y = r"(?:[\u1040-\u1049]{4}|\d{4})"
_NATIVE_DATE_MY = _MY_D + r"[\s,-]+လ[\s,-]+" + _MY_Y + r"[\s,-]+(?:ခု)?နှစ်" + r"|" + _MY_Y + r"[\s,-]+(?:ခု)?နှစ်"
"""my collision.

နှစ် means both 'two' and 'year'; anchored only as a year marker after a 4-digit block or a လ(month)+year pair — never as a
standalone digit.

"""
_LO_D = r"(?:[\u0ed0-\u0ed9]{1,2}|\d{1,2})"
_LO_Y = r"(?:[\u0ed0-\u0ed9]{2,4}|\d{2,4})"
_LO_MONTH = (
    r"ມັງກອນ|ກຸມພາ|ມີນາ|ເມສາ|ພຶດສະພາ|ມິຖຸນາ|ມີຖຸນາ|"
    r"ກໍລະກົດ|ສິງຫາ|ກັນຍາ|ຕຸລາ|ພະຈິກ|ທັນວາ"
)
_NATIVE_DATE_LO = (
    _LO_D
    + r"[\s,-]+(?:"
    + _LO_MONTH
    + r")[\s,-]+(?:ປີ[\s,-]+)?"
    + _LO_Y
    + r"|"
    + _LO_D
    + r"[\s,-]+ເດືອນ[\s,-]+"
    + _LO_D
    + r"[\s,-]+ປີ[\s,-]+"
    + _LO_Y
)
_TA_D = r"(?:[\u0be6-\u0bef]{1,2}|\d{1,2})"
_TA_Y = r"(?:[\u0be6-\u0bef]{2,4}|\d{2,4})"
_TA_MONTH = (
    r"சித்திரை|வைகாசி|ஆனி|ஆடி|ஆவணி|புரட்டாசி|"
    r"ஐப்பசி|கார்த்திகை|மார்கழி|தை|மாசி|பங்குனி|மாதம்"
)
_NATIVE_DATE_TA = (
    _TA_D
    + r"[\s,-]+(?:"
    + _TA_MONTH
    + r")[\s,-]+(?:ஆண்டு[\s,-]+)?"
    + _TA_Y
    + r"|"
    + r"(?:[\u0be6-\u0bef]{1,4}|\d{1,4})[\s,-]+ஆண்டு"
    + r"|[\u0bf3\u0bf4\u0bf5]"
)
NATIVE_DATE_RE = re.compile(
    r"(?u)(?:"
    + _NATIVE_DATE_CJK
    + r"|"
    + _NATIVE_DATE_KO
    + r"|"
    + _NATIVE_DATE_TH
    + r"|"
    + _NATIVE_DATE_MY
    + r"|"
    + _NATIVE_DATE_LO
    + r"|"
    + _NATIVE_DATE_TA
    + r")",
)
TIME_ONLY_RE = re.compile(r"(?iu)^\d{1,2}:\d{2}(?::\d{2})?$")
TIME_CANDIDATE_RE = re.compile(r"(?iu)\b\d{1,2}:\d{2}(?::\d{2})?\b")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
PHONE_CUE_RE = re.compile(
    r"(?iu)(?:sđt|sdt|điện thoại|dien thoai|tel|phone|fax|mobile|"
    r"liên hệ|lien he|contact|hotline|zalo)",
)
PHONE_STRONG_SURFACE_RE = re.compile(
    r"(?iu)^(?:"
    r"\+\d[\d .()-]{7,}\d"
    r"|0[235789][\d .()-]{7,}\d"
    r")$",
)
MEASUREMENT_UNITS = r"mg/dl|mmol/l|meq/l|bpm|mmhg|%"
DOSAGE_UNITS = r"mg|mcg|g|ml|iu|units?|tablets?|caps?|viên|ống|gói"
_RANGE_COMPARATOR = r"[<>≤≥]\s*\d+(?:\.\d+)?"
"""Three reference-range/measurement surfaces.

Comparator (<10, >5, ≤5.2), bare decimal (5.2), and numeric range (60-110, 3.9-6.1).

"""
_RANGE_DECIMAL = r"\d+\.\d+"
_RANGE_INTERVAL = r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?"
NUMERIC_MEASUREMENT_RE = re.compile(rf"(?iu)^(?:{_RANGE_COMPARATOR}|{_RANGE_DECIMAL}|{_RANGE_INTERVAL})$")
LAB_VALUE_UNIT_RE = re.compile(rf"(?iu)\d+(?:[.,]\d+)?\s*(?:{MEASUREMENT_UNITS})(?=\W|$)")
r"""Trailing lookahead (not `\b`).

A word boundary fails after `%` (non-word → end is no boundary), so `95%` would slip through. `(?=\W|$)` matches
end-or-non-word and still anchors word-char units like `mmol/L`, `mmHg`.

"""
DOSAGE_UNIT_RE = re.compile(rf"(?iu)^\d+(?:[.,]\d+)?\s*(?:{DOSAGE_UNITS})\b")
AGE_EXPRESSION_RE = re.compile(r"(?iu)^\d{1,3}\s*(?:years?\s*old|yrs?\s*old|yo|tuổi|tuoi)\b")
BARE_SMALL_INT_RE = re.compile(r"^\d{1,3}$")
ICD_CODE_RE = re.compile(r"^[A-Z]\d{2}(?:\.\d+)?$")
IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)$",
)
SECRET_RE = re.compile(
    r"(?iu)\b(?:"
    r"api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|"
    r"csrf|x_csrf|session|cookie|password|pwd|secret"
    r")\s*[:=]\s*[A-Za-z0-9._~+/=-]{4,}\b",
)
OTP_RE = re.compile(
    r"(?iu)\b(?:otp|pin|mã\s+otp|ma\s+otp|mã\s+pin|ma\s+pin)"
    r"(?:\s+\S+){0,3}\s*[:=]?\s*\d{4,8}(?:[- ]\d{2,8})?\b",
)
FIELD_PREFIX_RE = re.compile(
    r"(?iu)^(?:"
    r"địa chỉ|dia chi|ngày|ngay|thời gian|thoi gian|"
    r"người tiếp nhận|nguoi tiep nhan|bác sĩ|bac si|"
    r"mã|ma|email|sđt|sdt|tel|phone"
    r")\s*:",
)
STRONG_ORG_RE = re.compile(
    r"(?iu)\b(?:"
    r"Công ty(?:\s+(?:TNHH|CP|Cổ phần|MTV))?"
    r"|Bệnh viện"
    r"|Phòng khám"
    r"|Trung tâm"
    r"|Nhà thuốc"
    r"|Khách sạn"
    r"|Bảo hiểm Y tế"
    r")\b[^\n\r\]\[`:,;()]{2,90}",
)
DEPARTMENT_RE = re.compile(
    r"\bKhoa\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỵ]"
    r"[^\n\r\]\[`:,;()]{2,72}",
)

TRAILING_CHARS = " .,:;)]}>`'\"，。"
ORG_STOP_RE = re.compile(r"(?iu)\s+(?:vào lúc|ngày\s+\d|để\s+|mang theo|không\s+|khong\s+)")
PRIVATE_URL_MARKERS = (
    "patient",
    "portal",
    "record",
    "ehr",
    "emr",
    "pacs",
    "dicom",
    "referral",
    "reset",
    "token",
    "password",
    "signed",
    "session",
    "account",
    "temporary",
    "private",
    "secure",
)
PUBLIC_URL_NEGATIVES = (
    "w3.org/",
    "schema.org/",
    "example.com",
)

_MY_MONTH = (
    r"တန်ခူး|ကဆုန်|နယုန်|ဝါဆို|ဝါခေါင်|တော်သလင်း|"
    r"သီတင်းကျွတ်|တန်ဆောင်မုန်း|နတ်တော်|ပြာသို|တပို့တွဲ|တပေါင်း|မတ်လ"
)
_TA_MONTH_TRANSLIT = (
    r"ஜனவரி|பிப்ரவரி|மார்ச்|ஏப்ரல்|மே|ஜூன்|ஜூலை|ஆகஸ்ட்|"
    r"செப்டம்பர்|அக்டோபர்|நவம்பர்|டிசம்பர்"
)
_NATIVE_MONTH_NAME_RE = re.compile(
    r"(?u)(?:" + _TH_MONTH + r"|" + _LO_MONTH + r"|" + _TA_MONTH + r"|" + _MY_MONTH + r"|" + _TA_MONTH_TRANSLIT + r")",
)
_DATE_MARKER_RE = re.compile(
    r"(?u)(?:令和|平成|昭和|大正|明治|れいわ|へいせい|しょうわ"
    r"|ခုနှစ်|วันที่|ວັນທີ|ねん|がつ|にち"
    r"|年|月|日|号|號|년|월|일|ปี|เดือน|ປີ|ເດືອນ|ရက်|လ|ஆண்டு|மாதம்|தேதி|நாள்)",
)
"""Unambiguous year/month/day markers; >=2 DISTINCT in a span -> date structure.

Multi-char markers precede single chars so the alternation prefers the longer.

"""
_ANY_DATE_DIGIT = r"[0-9\u0e50-\u0e59\u1040-\u1049\u0ed0-\u0ed9\u0be6-\u0bef]"
"""Native + ASCII digit ISO/separated dates (e.g.

Myanmar digits 1992-04-18).

"""
_NATIVE_NUMERIC_DATE_RE = re.compile(
    rf"(?u){_ANY_DATE_DIGIT}{{2,4}}[-/.]{_ANY_DATE_DIGIT}{{1,2}}"
    rf"[-/.]{_ANY_DATE_DIGIT}{{1,4}}",
)


def _has_native_date_surface(value: str) -> bool:
    if _NATIVE_MONTH_NAME_RE.search(value) or _NATIVE_NUMERIC_DATE_RE.search(value):
        return True
    distinct_markers = {match.group(0) for match in _DATE_MARKER_RE.finditer(value)}
    return len(distinct_markers) >= MIN_DISTINCT_DATE_MARKERS


def _has_date_surface(value: str) -> bool:
    return any(
        pattern.search(value) is not None
        for pattern in (
            ISO_DATETIME_RE,
            DATE_RE,
            DOT_DATE_RE,
            COMPACT_DATE_RE,
            TEXTUAL_DATE_RE,
            LOCALIZED_TEXTUAL_DATE_RE,
            NATIVE_DATE_RE,
        )
    ) or _has_native_date_surface(value)


def _is_valid_date_surface(value: str) -> bool:
    stripped = value.strip()
    return (
        bool(stripped)
        and len(stripped) <= MAX_DATE_SURFACE_LENGTH
        and not TIME_ONLY_RE.fullmatch(stripped)
        and _has_date_surface(stripped)
    )
