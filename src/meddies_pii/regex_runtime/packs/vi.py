"""Vietnamese institutional-prefix inventory for CONTEXT-tier organization capture.

The cue inventory is the one curated for the Vietnamese facility probe
(``scripts/vi_facility_probe/vi_names.py``): the institutional prefixes that open a
hospital, clinic, pharmacy, school, institute, or company name. There is no name
gazetteer -- the cue plus the shape of the run that follows it is the whole signal,
so a name the corpus has never seen is captured on the same terms as a known one.

``NAME_CONTINUATIONS`` are the lowercase generic words that sit *inside* a real
name (``Bệnh viện Đa khoa ...``, ``Trung tâm Y tế dự phòng tỉnh ...``). Every word
outside that list ends the capture, which is what keeps a bare cue in prose
(``chuyển bệnh viện tuyến trên``) from capturing anything.
"""

from __future__ import annotations

import unicodedata

from meddies_pii.regex_runtime.packs.pack import Cue, CuePack

_TONE_MARKS = ("", "̀", "́", "̃", "̉", "̣")
_TONE_BEARING_VOWELS = "aăâeêioôơuưy"


def _vietnamese_lowercase_letters() -> str:
    letters = set("abcdefghijklmnopqrstuvwxyzăâêôơưđ")
    letters.update(unicodedata.normalize("NFC", vowel + tone) for vowel in _TONE_BEARING_VOWELS for tone in _TONE_MARKS)
    return "".join(sorted(letter for letter in letters if len(letter) == 1))


_LOWERCASE_LETTERS = _vietnamese_lowercase_letters()
_UPPERCASE_LETTERS = "".join(sorted({letter.upper() for letter in _LOWERCASE_LETTERS}))

CUES: tuple[Cue, ...] = (
    Cue("Trung tâm Kiểm soát bệnh tật", "pre"),
    Cue("Trung tâm Chẩn đoán Y khoa", "pre"),
    Cue("Trung tâm Y tế dự phòng", "pre"),
    Cue("Trung tâm Y tế", "pre"),
    Cue("Trung tâm", "pre"),
    Cue("Bệnh viện", "pre"),
    Cue("Phòng khám", "pre"),
    Cue("Trạm Y tế", "pre"),
    Cue("Nhà thuốc", "pre"),
    Cue("Quầy thuốc", "pre"),
    Cue("Trường Phổ thông Dân tộc Nội trú", "pre"),
    Cue("Trường Trung học phổ thông", "pre"),
    Cue("Trường Trung học cơ sở", "pre"),
    Cue("Trường Cao đẳng", "pre"),
    Cue("Trường Trung cấp", "pre"),
    Cue("Trường Đại học", "pre"),
    Cue("Trường Phổ thông", "pre"),
    Cue("Trường Tiểu học", "pre"),
    Cue("Trường Mẫu giáo", "pre"),
    Cue("Trường Mầm non", "pre"),
    Cue("Trường THPT", "pre"),
    Cue("Trường THCS", "pre"),
    Cue("Trường", "pre"),
    Cue("Đại học", "pre"),
    Cue("Học viện", "pre"),
    Cue("Công ty Cổ phần", "pre"),
    Cue("Công ty TNHH", "pre"),
    Cue("Công ty", "pre"),
    Cue("Viện", "pre"),
)
"""Institutional prefixes. Case is irrelevant at match time, so the list carries one spelling per prefix; the compiler
orders them longest-first.
"""

NAME_CONTINUATIONS: tuple[str, ...] = (
    "bướu",
    "bảo",
    "chuyên",
    "chấn",
    "chẩn",
    "chỉnh",
    "chủng",
    "chức",
    "cơ",
    "cổ",
    "cứu",
    "da",
    "dưỡng",
    "dược",
    "dịch",
    "dục",
    "dự",
    "dựng",
    "hiểm",
    "huyết",
    "huyện",
    "hàm",
    "hình",
    "học",
    "họng",
    "hồi",
    "hữu",
    "khoa",
    "khu",
    "khẩu",
    "khỏe",
    "khớp",
    "kiểm",
    "lao",
    "liễu",
    "liệu",
    "máu",
    "mũi",
    "mầm",
    "mắt",
    "mặt",
    "mỹ",
    "nghiên",
    "nghiệp",
    "nghệ",
    "nghị",
    "ngoại",
    "ngữ",
    "nhi",
    "nhiệt",
    "nhánh",
    "nhân",
    "non",
    "năng",
    "nội",
    "phát",
    "phòng",
    "phường",
    "phẩm",
    "phố",
    "phổ",
    "phổi",
    "phụ",
    "phục",
    "quân",
    "quận",
    "quốc",
    "răng",
    "sinh",
    "soát",
    "sĩ",
    "sản",
    "sở",
    "sức",
    "tai",
    "thiết",
    "thuốc",
    "thành",
    "thông",
    "thương",
    "thường",
    "thần",
    "thẩm",
    "thận",
    "thị",
    "tim",
    "tiết",
    "tiểu",
    "triển",
    "truyền",
    "trị",
    "tuỷ",
    "tâm",
    "tư",
    "tật",
    "tế",
    "tễ",
    "tỉnh",
    "tử",
    "ung",
    "vùng",
    "vật",
    "vực",
    "xuyên",
    "xuất",
    "xã",
    "y",
    "đa",
    "điều",
    "đông",
    "đới",
    "đức",
    "ương",
)
"""Lowercase words that continue a name rather than ending it: service-tier qualifiers, clinical specialties,
administrative-unit heads, and the generic institutional vocabulary of the health, education, and trade sectors. These
are category words, never name cores -- an institution's identifying core is capitalized in every register this pack
targets, so it is matched by shape.
"""

BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    "gần",
    "của",
    "hợp",
    "khác",
    "này",
    "nào",
    "phí",
    "sau",
    "tuyến",
    "tư nhân",
    "địa",
)
"""Words that turn the cue into prose. ``trường hợp`` is a case, not a school; ``bệnh viện tuyến trên`` and ``bệnh viện
địa phương`` name a tier, not an institution; ``nhà thuốc gần nhất`` names proximity, not a pharmacy.
"""

VI_ORG_PACK = CuePack(
    pack_id="vi_org_prefix",
    version=1,
    language="vi",
    label="company_name",
    tier="CONTEXT",
    cues=CUES,
    head_letters=_UPPERCASE_LETTERS,
    body_letters=_UPPERCASE_LETTERS + _LOWERCASE_LETTERS,
    name_continuations=NAME_CONTINUATIONS,
    blocked_adjacent_tokens=BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=8,
    min_adjacent_digits=2,
    name_terminal_continuations=("ương",),
)
