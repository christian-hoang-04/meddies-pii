"""Locked high-precision regex candidates for the paired ablation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from meddies_pii.regex_runtime.packs import (
    ALL_PACKS,
    CuePack,
    pack_pattern,
    pack_payload,
)
from meddies_pii.spans import CharSpan
from meddies_pii.taxonomy import PiiLabel, is_pii_label

if TYPE_CHECKING:
    from collections.abc import Callable

_NAME_SEPARATOR_PATTERN = r"[ \t\u00a0\-\N{NON-BREAKING HYPHEN}]+"

RegexTier = Literal["AUTH", "CONTEXT", "SNAP"]
TextView = Literal["normalized", "composed"]

TIER_PRECEDENCE: tuple[RegexTier, ...] = ("AUTH", "CONTEXT", "SNAP")


@dataclass(frozen=True, slots=True)
class NormalizedText:
    text: str
    original_start: tuple[int, ...]
    original_end: tuple[int, ...]

    def original_bounds(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end <= start or end > len(self.text):
            msg = f"invalid normalized interval: [{start}, {end})"
            raise ValueError(msg)
        return self.original_start[start], self.original_end[end - 1]


def normalize_with_offsets(text: str) -> NormalizedText:
    """NFKC-casefold each codepoint while retaining exact source bounds.

    Returns:
        The NFKC-casefolded text and its exact source-interval map.

    """
    if text.isascii():
        return NormalizedText(text.casefold(), tuple(range(len(text))), tuple(range(1, len(text) + 1)))
    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, character in enumerate(text):
        replacement = unicodedata.normalize("NFKC", character).casefold()
        normalized.extend(replacement)
        starts.extend(index for _ in replacement)
        ends.extend(index + 1 for _ in replacement)
    return NormalizedText("".join(normalized), tuple(starts), tuple(ends))


def compose_with_offsets(text: str) -> NormalizedText:
    """NFC-compose each base-plus-marks cluster while retaining exact source bounds.

    Scans and legacy EMR exports emit decomposed Vietnamese, where ``ồ`` is two
    codepoints. A pattern written against composed letters silently matches a prefix
    of such a word and truncates the span, so the case-preserving view composes
    first. Composition shortens the text, which is why the interval map is required:
    normalizing without one would report offsets into a string the caller never had.

    Returns:
        The NFC-composed text and its exact source-interval map.

    """
    if text.isascii():
        return NormalizedText(text, tuple(range(len(text))), tuple(range(1, len(text) + 1)))
    composed: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        cluster_end = index + 1
        while cluster_end < len(text) and unicodedata.combining(text[cluster_end]):
            cluster_end += 1
        replacement = unicodedata.normalize("NFC", text[index:cluster_end])
        composed.extend(replacement)
        starts.extend(index for _ in replacement)
        ends.extend(cluster_end for _ in replacement)
        index = cluster_end
    return NormalizedText("".join(composed), tuple(starts), tuple(ends))


@dataclass(frozen=True, slots=True)
class RegexRule:
    """One compiled candidate source.

    ``tier`` fixes how the candidate may act on a model span: AUTH is authoritative
    for validator-gated shapes, CONTEXT is cue-gated capture that is authoritative
    for its own label but never overrides AUTH, and SNAP never adds a span — it only
    expands a model span it overlaps. ``text_view`` selects the string the pattern
    runs against: shape rules match the casefolded normalization, cue rules that key
    on capitalization must match the raw source.
    """

    name: str
    label: PiiLabel
    pattern: re.Pattern[str]
    value_group: str
    validator: Callable[[str], bool]
    tier: RegexTier = "AUTH"
    text_view: TextView = "normalized"
    language: str | None = None


@dataclass(frozen=True, slots=True)
class RegexCandidate:
    span: CharSpan
    tier: RegexTier
    rule_name: str


_EMAIL = re.compile(
    r"(?<!\w)(?=[a-z0-9_+.-]{1,64}@)"
    r"(?P<value>[a-z0-9_][a-z0-9_+-]*(?:\.[a-z0-9_][a-z0-9_+-]*)*"
    r"@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+)(?![\w.-])",
)
"""The local part is a dot-atom: atoms of [a-z0-9_+-] joined by single dots, each atom opening on a word character. The
grammar, not a list of banned characters, is what makes a leading, trailing or doubled dot unrepresentable —
enumerating delimiters kept reopening the same leftward bleed, which _valid_email then blessed into a widened model
span. The alphabet is the pragmatic set: `_ + -` appear in real mailboxes (underscore usernames, plus-addressing,
hyphenated names) while `! # $ % ' *` are RFC-legal, absent from clinical-corpus addresses, and each one generated a
defect here. This AUTH lane trades recall on exotic addresses for that precision.

The bounded lookahead is a scan guard that prunes or re-anchors: `_valid_email` rejects a local part longer than 64
characters, while scanning may restart after punctuation and capture a valid suffix. That suffix capture is intended,
consistent with the dot-atom precedent where `a..b@host.test` captures `b@host.test`. Without the guard, every position
after a `+`, `-` or `.` starts a scan that runs to the end of the token, which makes a long punctuation run quadratic.
`test_email_length_bound_matches_the_validator` pins the pattern and validator bounds.
"""
_PHONE = re.compile(
    r"\b(?:điện\s*thoại|dien\s*thoai|sđt|sdt|phone|tel(?:ephone)?|hotline|"
    r"liên\s*hệ)\s*(?:[:#-]\s*)?"
    r"(?<!\d)(?P<value>(?:\+?84|0)[ .-]?[35789]\d"
    r"(?:[ .-]?\d){7})(?!\d)",
)
_CCCD = re.compile(
    r"\b(?:cccd|căn\s*cước(?:\s*công\s*dân)?|số\s*định\s*danh)\s*"
    r"(?:[:#-]\s*)?(?P<value>\d{12})(?!\d)",
)
_PRIVATE_URL = re.compile(
    r"(?P<value>https://[a-z0-9.-]+(?::\d{2,5})?"
    r"(?:/(?:[a-z0-9._~!$&')*+,;=:@%/-]|\((?!https://))*)?"
    r"\?(?:(?:[a-z0-9._~!$')*+,;:@%/?=-]|\((?!https://))+&)*"
    r"(?:access_token|auth|key|secret|signature|sig|token)="
    r"(?:[a-z0-9._~!$')*+,;:@%/?=-]|\((?!https://)){8,}"
    r"(?:&[a-z0-9_.-]+=(?:[a-z0-9._~!$')*+,;:@%/?=-]|\((?!https://))*)*"
    r"(?<=[a-z0-9_~=-]))",
)
"""Parentheses remain valid URL characters. An opening parenthesis immediately before another ``https://`` is a malformed
Markdown boundary, not query data; refusing only that sequence splits a doubled link without dropping valid values.

Cross a trailing `&` only into a further name=value parameter, so an ampersand-joined phrase after the link stays out of
the span. A parameter value is bounded by the next `&name=` or by the span end, so it needs no shape of its own; the `&`
and `=` the name class excludes are what force one parse per parameter. One lookbehind then decides where the whole span
may end, which is why a value of pure punctuation no longer breaks the chain.
"""
_SECRET = re.compile(
    r"\b(?:api[ _-]?key|access[ _-]?token|password|secret|token)\s*"
    r"[:=]\s*(?P<value>[a-z0-9][a-z0-9._~+/-]{6,}[a-z0-9_~+/=-])",
)
_MAX_EMAIL_LENGTH = 254
_MAX_EMAIL_LOCAL_LENGTH = 64
_MIN_SECRET_LENGTH = 8
_MIN_AUTHORITATIVE_NUMERIC_NAME_LENGTH = 3


def _valid_email(value: str) -> bool:
    local, domain = value.rsplit("@", 1)
    return (
        len(value) <= _MAX_EMAIL_LENGTH
        and len(local) <= _MAX_EMAIL_LOCAL_LENGTH
        and "." in domain
        and bool(_EMAIL.fullmatch(value))
    )


def _valid_phone(value: str) -> bool:
    compact = re.sub(r"[ .-]", "", value)
    if compact.startswith("+84"):
        compact = "0" + compact[3:]
    elif compact.startswith("84"):
        compact = "0" + compact[2:]
    return bool(re.fullmatch(r"0[35789]\d{8}", compact))


def _valid_cccd(value: str) -> bool:
    return bool(re.fullmatch(r"(?:00[1-9]|0[1-8]\d|09[0-6])\d{9}", value))


def _valid_private_url(value: str) -> bool:
    return bool(_PRIVATE_URL.fullmatch(value))


def _valid_secret(value: str) -> bool:
    return (
        len(value) >= _MIN_SECRET_LENGTH
        and bool(re.fullmatch(r"[a-z0-9][a-z0-9._~+/-]{6,}[a-z0-9_~+/=-]", value))
        and any(character.isalpha() for character in value)
        and any(character.isdigit() for character in value)
    )


def _compile_pack(pack: CuePack) -> RegexRule:
    """Turn a data-only pack into a rule.

    The validator is the pack's own pattern anchored end to end. That is what the
    merge path asks before widening a model span: the widened text must still read
    as one complete cued mention, so a union can never stretch a capture past the
    name it was gated on.

    Returns:
        The executable regex rule compiled from the data-only pack.

    Raises:
        ValueError: If the pack declares a label outside the Meddies PII taxonomy.

    """
    if not is_pii_label(pack.label):
        msg = f"pack {pack.pack_id} declares unknown label {pack.label!r}"
        raise ValueError(msg)
    if pack.tier not in TIER_PRECEDENCE:
        msg = f"pack {pack.pack_id} declares unknown tier {pack.tier!r}"
        raise ValueError(msg)
    blocked_tokens = _shared_blocked_tokens(pack)
    blocked_terminal_tokens = _shared_blocked_terminal_tokens(pack)
    blocked_singleton_tokens = _shared_blocked_singleton_tokens(pack)
    pattern = pack_pattern(pack)

    def accepts(value: str) -> bool:
        if pack.capture_mode == "quoted":
            pattern_match = any(
                pattern.fullmatch(f"{cue.text} {pack.opening_delimiter}{value}{pack.closing_delimiter}")
                for cue in pack.cues
            )
        else:
            pattern_match = bool(pattern.fullmatch(value))
        return pattern_match and not _contains_blocked_name_tokens(
            pack,
            value,
            blocked_tokens,
            blocked_terminal_tokens,
            blocked_singleton_tokens,
        )

    return RegexRule(
        name=f"pack:{pack.pack_id}",
        label=pack.label,
        pattern=pattern,
        value_group="value",
        validator=accepts,
        tier=pack.tier,
        text_view="composed",
        language=pack.language,
    )


# reason: capture mode, cue position, and token adjacency form one ordered fail-closed acceptance policy.
def _contains_blocked_name_tokens(  # ruff: ignore[complex-structure,too-many-branches,too-many-return-statements]
    pack: CuePack,
    value: str,
    blocked_tokens: tuple[str, ...],
    blocked_terminal_tokens: tuple[str, ...],
    blocked_singleton_tokens: tuple[str, ...],
) -> bool:
    normalized_value = " ".join(re.split(_NAME_SEPARATOR_PATTERN, value.casefold()))
    if re.match(r"(?i:copyright) (?:19|20)\d{2}\b", normalized_value) or re.match(r"(?:19|20)\d{2}\b", normalized_value):
        return True
    if pack.require_capitalized_cue and not value[0].isupper():
        return True
    if pack.capture_mode == "scriptless":
        name_core, cue_position = _scriptless_name_core(pack, value)
        if pack.script_name_separator:
            name_tokens = tuple(re.split(_NAME_SEPARATOR_PATTERN, name_core))
            return any(blocked.casefold() in name_tokens for blocked in blocked_tokens)
        adjacent_is_blocked = name_core.startswith if cue_position == "pre" else name_core.endswith
        return any(adjacent_is_blocked(blocked.casefold()) for blocked in blocked_tokens) or any(
            name_core.endswith(blocked.casefold()) for blocked in blocked_terminal_tokens
        )
    if pack.capture_mode == "quoted":
        name_start = 0
        name_end = len(value)
        cue_end = -1
    else:
        cue_end = max(
            (
                match.end()
                for cue in pack.cues
                if (
                    match := re.match(
                        "(?i:" + r"[ \t\u00a0]+".join(re.escape(word) for word in cue.text.split()) + ")",
                        value,
                    )
                )
            ),
            default=0,
        )
        name_start = cue_end
        name_end = len(value)
    if cue_end == 0:
        terminal = ""
        if pack.suffix_terminal_tokens:
            terminal = (
                r"(?:[ \t\u00a0]+(?i:"
                + "|".join(
                    re.escape(token)
                    for token in sorted(
                        pack.suffix_terminal_tokens,
                        key=lambda item: (-len(item), item),
                    )
                )
                + "))?"
            )
        suffix_start = min(
            (
                match.start()
                for cue in pack.cues
                if cue.position == "post"
                and (
                    match := re.search(
                        "(?i:" + r"[ \t\u00a0]+".join(re.escape(word) for word in cue.text.split()) + ")" + terminal + "$",
                        value,
                    )
                )
            ),
            default=len(value),
        )
        if suffix_start == len(value):
            return True
        name_start = 0
        name_end = suffix_start
    name_tokens = tuple(
        token.rstrip(".").casefold() for token in re.split(_NAME_SEPARATOR_PATTERN, value[name_start:name_end].strip())
    )
    continuation_tokens = {token.rstrip(".").casefold() for token in pack.name_continuations}
    semantic_name_tokens = tuple(
        token for token in name_tokens if token and token not in continuation_tokens and not token.isdigit()
    )
    has_authoritative_numeric_name = any(
        token.isdigit() and len(token) >= _MIN_AUTHORITATIVE_NUMERIC_NAME_LENGTH for token in name_tokens
    )
    if not semantic_name_tokens and not has_authoritative_numeric_name:
        return True
    if len(semantic_name_tokens) == 1 and semantic_name_tokens[0] in {
        token.casefold() for token in blocked_singleton_tokens
    }:
        return True
    for blocked in blocked_tokens:
        blocked_parts = tuple(part.casefold() for part in re.split(_NAME_SEPARATOR_PATTERN, blocked))
        width = len(blocked_parts)
        if any(name_tokens[index : index + width] == blocked_parts for index in range(len(name_tokens) - width + 1)):
            return True
    for blocked in blocked_terminal_tokens:
        blocked_parts = tuple(part.casefold() for part in re.split(_NAME_SEPARATOR_PATTERN, blocked))
        if name_tokens[-len(blocked_parts) :] == blocked_parts:
            return True
    return False


def _scriptless_name_core(pack: CuePack, value: str) -> tuple[str, str]:
    for cue in pack.cues:
        if cue.position == "pre" and value.startswith(cue.text):
            return value[len(cue.text) :].strip().casefold(), cue.position
        if cue.position == "post" and value.endswith(cue.text):
            return value[: -len(cue.text)].strip().casefold(), cue.position
    return value.casefold(), "pre"


def _shared_blocked_tokens(pack: CuePack) -> tuple[str, ...]:
    owned_cues = {cue.text.casefold() for cue in pack.cues}
    return tuple(
        sorted(
            {
                token
                for owner in ALL_PACKS
                if owned_cues.intersection(cue.text.casefold() for cue in owner.cues)
                for token in owner.blocked_adjacent_tokens
            },
            key=lambda token: (token.casefold(), token),
        ),
    )


def _shared_blocked_terminal_tokens(pack: CuePack) -> tuple[str, ...]:
    owned_cues = {cue.text.casefold() for cue in pack.cues}
    return tuple(
        sorted(
            {
                token
                for owner in ALL_PACKS
                if owned_cues.intersection(cue.text.casefold() for cue in owner.cues)
                for token in owner.blocked_terminal_tokens
            },
            key=lambda token: (token.casefold(), token),
        ),
    )


def _shared_blocked_singleton_tokens(pack: CuePack) -> tuple[str, ...]:
    owned_cues = {cue.text.casefold() for cue in pack.cues}
    return tuple(
        sorted(
            {
                token
                for owner in ALL_PACKS
                if owned_cues.intersection(cue.text.casefold() for cue in owner.cues)
                for token in owner.blocked_singleton_tokens
            },
            key=lambda token: (token.casefold(), token),
        ),
    )


REGEX_RULES: tuple[RegexRule, ...] = (
    RegexRule("email", "email_address", _EMAIL, "value", _valid_email),
    RegexRule("vietnam_phone", "phone_number", _PHONE, "value", _valid_phone),
    RegexRule("cccd", "id_number", _CCCD, "value", _valid_cccd),
    RegexRule("private_access_url", "private_url", _PRIVATE_URL, "value", _valid_private_url),
    RegexRule("explicit_secret", "secret", _SECRET, "value", _valid_secret),
    *(_compile_pack(pack) for pack in ALL_PACKS),
)

_POLICY: dict[str, object] = {
    "schema_version": 3,
    "normalization": "per-codepoint NFKC then casefold with source interval map",
    "boundary": "rule regex lookarounds plus full-match validator",
    "pack_applicability": (
        "language-independent authoritative rules always run; organization packs "
        "run only when their declared document language is known"
    ),
    "regex_overlap": (
        "tier precedence AUTH > CONTEXT > SNAP; within a tier the earlier span by "
        "(start, longest, label) wins and later overlapping candidates are dropped"
    ),
    "model_overlap": (
        "collapse exact duplicate; add non-overlap; same-label union only when the "
        "family validator accepts; different-label model span preserved and conflict "
        "recorded; structured labels emit non-overlapping remainders of at least 3 "
        "chars; other labels, including company_name, discard the candidate"
    ),
    "rules": [
        {
            "name": rule.name,
            "label": rule.label,
            "pattern": rule.pattern.pattern,
            "flags": rule.pattern.flags,
            "value_group": rule.value_group,
            "validator": getattr(rule.validator, "__name__", rule.name),
            "tier": rule.tier,
            "text_view": rule.text_view,
            "language": rule.language,
        }
        for rule in REGEX_RULES
    ],
}


def regex_manifest_inputs() -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return policy and expanded pack payloads for provenance hashing.

    Returns:
        The frozen policy payload and each pack's effective policy payload.

    """
    return _POLICY, [
        pack_payload(
            pack,
            blocked_tokens=_shared_blocked_tokens(pack),
            blocked_terminal_tokens=_shared_blocked_terminal_tokens(pack),
            blocked_singleton_tokens=_shared_blocked_singleton_tokens(pack),
        )
        for pack in ALL_PACKS
    ]


def union_accepted(label: str, text: str, language: str | None = None) -> bool:
    """Report whether any rule of ``label`` accepts ``text`` as one complete value.

    The merge path calls this before widening a model span to a regex span of the
    same label. Each rule is asked against the text view it was written for, so a
    case-preserving cue rule is not handed casefolded text it can never accept.

    Returns:
        `True` when a rule for the label accepts the complete text union.

    """
    if not is_pii_label(label):
        return False
    for rule in REGEX_RULES:
        if rule.label != label or not _rule_applies(rule, language):
            continue
        if rule.validator(_text_view(rule.text_view, text).text):
            return True
    return False


def _text_view(view: TextView, text: str) -> NormalizedText:
    if view == "composed":
        return compose_with_offsets(text)
    return normalize_with_offsets(text)


def regex_candidates(text: str, language: str | None = None) -> tuple[RegexCandidate, ...]:
    views: dict[TextView, NormalizedText] = {
        "normalized": normalize_with_offsets(text),
        "composed": compose_with_offsets(text),
    }
    candidates: list[RegexCandidate] = []
    for rule in REGEX_RULES:
        if not _rule_applies(rule, language):
            continue
        view = views[rule.text_view]
        for match in rule.pattern.finditer(view.text):
            value_start, value_end = match.span(rule.value_group)
            if not rule.validator(view.text[value_start:value_end]):
                continue
            start, end = view.original_bounds(value_start, value_end)
            candidates.append(
                RegexCandidate(
                    CharSpan(start, end, text[start:end], rule.label),
                    rule.tier,
                    rule.name,
                ),
            )
    return _resolve_regex_overlaps(candidates)


def _rule_applies(rule: RegexRule, language: str | None) -> bool:
    return rule.language is None or (language is not None and rule.language == language)


def _resolve_regex_overlaps(
    candidates: list[RegexCandidate],
) -> tuple[RegexCandidate, ...]:
    """Keep one candidate per overlapping region, highest tier first.

    A lower tier never displaces a region already claimed by a higher one, so a
    cue-gated CONTEXT capture cannot cut into a validated AUTH shape and a SNAP
    shape cannot cut into either. Inside a tier the earlier, then longer, span wins.

    Returns:
        The non-overlapping candidates selected by tier and span precedence.

    """
    accepted: list[RegexCandidate] = []
    for tier in TIER_PRECEDENCE:
        claimed = sorted(accepted, key=lambda candidate: candidate.span.start)
        claimed_index = 0
        accepted_in_tier: list[RegexCandidate] = []
        ordered = sorted(
            (candidate for candidate in candidates if candidate.tier == tier),
            key=lambda candidate: (
                candidate.span.start,
                -(candidate.span.end - candidate.span.start),
                candidate.span.label,
            ),
        )
        for candidate in ordered:
            while claimed_index < len(claimed) and claimed[claimed_index].span.end <= candidate.span.start:
                claimed_index += 1
            overlaps_higher_tier = claimed_index < len(claimed) and _overlaps(claimed[claimed_index].span, candidate.span)
            overlaps_same_tier = bool(accepted_in_tier) and _overlaps(accepted_in_tier[-1].span, candidate.span)
            if overlaps_higher_tier or overlaps_same_tier:
                continue
            accepted_in_tier.append(candidate)
        accepted.extend(accepted_in_tier)
    return tuple(
        sorted(
            accepted,
            key=lambda candidate: (
                candidate.span.start,
                candidate.span.end,
                candidate.span.label,
            ),
        ),
    )


def _overlaps(left: CharSpan, right: CharSpan) -> bool:
    return left.start < right.end and right.start < left.end
