"""Japanese, Chinese, and Korean organization-pack declarations."""

from __future__ import annotations

from meddies_pii.regex_runtime.packs.pack import Cue, CuePack

JA_ORG_PACK = CuePack(
    pack_id="ja_org_scriptless",
    version=1,
    language="ja",
    label="company_name",
    tier="CONTEXT",
    cues=(
        Cue("総合病院", "post"),
        Cue("クリニック", "post"),
        Cue("病院", "post"),
        Cue("株式会社", "pre"),
    ),
    head_letters="",
    body_letters="",
    name_continuations=(),
    blocked_adjacent_tokens=("紹介元", "受診", "の", "かかりつけ"),
    max_name_tokens=0,
    min_adjacent_digits=0,
    capture_mode="scriptless",
    script_ranges=((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF)),
    left_boundary_particles=("紹介先は", "入院した"),
    right_boundary_particles=("から", "です", "で", "に", "へ"),
    max_name_chars=24,
)


ZH_ORG_PACK = CuePack(
    pack_id="zh_org_scriptless",
    version=1,
    language="zh",
    label="company_name",
    tier="CONTEXT",
    cues=tuple(
        Cue(text, "post")
        for text in (
            "社区卫生服务中心",
            "互联网医院",
            "检验中心",
            "医疗中心",
            "康复医院",
            "门诊部",
            "医院",
            "诊所",
            "公司",
        )
    ),
    head_letters="",
    body_letters="",
    name_continuations=(),
    blocked_adjacent_tokens=(),
    max_name_tokens=0,
    min_adjacent_digits=0,
    capture_mode="scriptless",
    script_ranges=((0x3400, 0x4DBF), (0x4E00, 0x9FFF)),
    left_boundary_particles=("拟转至", "这里是", "患者在", "转至"),
    right_boundary_particles=("接受", "发布"),
    max_name_chars=24,
    blocked_terminal_tokens=(
        "保险",
        "就诊",
        "转诊",
        "到达",
        "前往",
        "联系",
        "用于",
        "开药",
        "核对",
        "合作",
        "出院",
        "公立",
        "会诊",
        "示例",
    ),
)


KO_ORG_PACK = CuePack(
    pack_id="ko_org_scriptless",
    version=1,
    language="ko",
    label="company_name",
    tier="CONTEXT",
    cues=(
        Cue("종합병원", "post"),
        Cue("메디컬센터", "post"),
        Cue("병원", "post"),
        Cue("의원", "post"),
        Cue("주식회사", "pre"),
    ),
    head_letters="",
    body_letters="",
    name_continuations=(),
    blocked_adjacent_tokens=(),
    max_name_tokens=0,
    min_adjacent_digits=0,
    capture_mode="scriptless",
    script_ranges=((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)),
    right_boundary_particles=(
        "에서",
        "에는",
        "으로",
        "에게",
        "은",
        "는",
        "이",
        "가",
        "에",
        "의",
    ),
    max_name_chars=20,
    blocked_terminal_tokens=("협력", "진료", "연계"),
)
