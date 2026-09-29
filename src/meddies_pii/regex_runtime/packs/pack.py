"""Pattern-pack format: a cue-gated capture rule expressed entirely as data.

A pack carries the inventory for one language and one label. Compiling it produces
the regex; adding the next language is a new pack instance, not new code. The
compiled pattern is derived from the pack fields alone, so hashing the fields
content-addresses the behavior.

Where the institution type sits relative to the name is a property of the *cue*,
not of the language: Vietnamese writes ``Bệnh viện X``, Japanese writes ``X病院``,
and Spanish, German, and Filipino carry cues of both kinds at once. Every cue
declares its own ``position``. A pack's cue list may also mix scripts, since documents
in a language routinely carry the Latin-script form of a registered name.

The cased engine matches either cue orientation:

    CUE  (name token)*  PROPER
    PROPER  (name token)*  CUE  (legal form)?

Name tokens are proper tokens (a cased word, including an undotted single letter, or
a short number), dotted abbreviations, or one of the pack's ``name_continuations``.
The run must end on a proper token, so a dotted abbreviation never becomes a
truncated final span. A bare cue in prose captures nothing. Tokens are separated by
spaces only: a newline, a comma, or any other punctuation ends the capture at a
field boundary.

An undotted single letter may end a name (``Bệnh viện K``). A dotted token carries
a name (``Praxis Dr. J. Meier``) but never ends one. A run of dotted initials is a
boundary unless a dotted title introduces the initials and a surname closes them.

The compiler rejects a short bare number adjacent to the cue. The candidate
validator applies the pack's blocked tokens to the whole matched name, including
hyphen-joined tokens, rather than only the first token after the cue.

Scriptless packs declare Unicode ranges and boundary particles. Their capture stops
at a script change, punctuation, or a declared particle. Quoted packs use the same
proper-token grammar but return only the text inside their delimiters.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

_NON_BREAKING_HYPHEN = "\N{NON-BREAKING HYPHEN}"

CuePosition = Literal["pre", "post"]
CaptureMode = Literal["cased", "scriptless", "quoted"]

# reason: this is a regular-expression fragment for whitespace between name tokens, not a credential.
_TOKEN_SEPARATOR = "[ \\t\\u00a0]+"  # ruff: ignore[hardcoded-password-string]
_MAX_NONFINAL_CONTINUATION_LENGTH = 3


@dataclass(frozen=True, slots=True)
class Cue:
    text: str
    position: CuePosition


@dataclass(frozen=True, slots=True)
class CuePack:
    pack_id: str
    version: int
    language: str
    label: str
    tier: str
    cues: tuple[Cue, ...]
    head_letters: str
    body_letters: str
    name_continuations: tuple[str, ...]
    blocked_adjacent_tokens: tuple[str, ...]
    max_name_tokens: int
    min_adjacent_digits: int
    blocked_terminal_tokens: tuple[str, ...] = ()
    blocked_singleton_tokens: tuple[str, ...] = ()
    suffix_terminal_tokens: tuple[str, ...] = ()
    name_terminal_continuations: tuple[str, ...] = ()
    capture_mode: CaptureMode = "cased"
    script_ranges: tuple[tuple[int, int], ...] = ()
    left_boundary_particles: tuple[str, ...] = ()
    right_boundary_particles: tuple[str, ...] = ()
    max_name_chars: int = 0
    script_name_separator: bool = False
    max_script_name_tokens: int = 1
    allow_trailing_number: bool = False
    opening_delimiter: str = ""
    closing_delimiter: str = ""
    require_capitalized_cue: bool = False


# reason: capture-mode branches compile one interacting regex policy whose ordering must remain visible in one place.
def pack_pattern(  # ruff: ignore[complex-structure,too-many-statements,too-many-locals]
    pack: CuePack,
) -> re.Pattern[str]:
    """Compile the pack's cue-gated capture pattern.

    A dotted word is one token and must be followed by the final proper token. A dotted-initial run is a boundary, not
    a bridge into the next sentence.

    Returns:
        The compiled cue-gated regular expression for the pack.

    Raises:
        ValueError: If the pack declares an unsupported capture mode or invalid cue configuration.

    """
    if pack.capture_mode == "scriptless":
        return _scriptless_pattern(pack)
    head = re.escape(pack.head_letters)
    body = re.escape(pack.body_letters)
    nonfinal_guard = ""
    nonfinal_continuations = tuple(
        continuation
        for continuation in pack.name_continuations
        if len(continuation.rstrip(".")) <= _MAX_NONFINAL_CONTINUATION_LENGTH
    )
    if nonfinal_continuations:
        nonfinal = _word_alternation(nonfinal_continuations)
        nonfinal_guard = rf"(?!(?:{nonfinal})(?![{body}0-9]))"
    dotted_initial = rf"[{head}]\."
    single_letter_proper = (
        rf"[{head}](?!\.(?:{_TOKEN_SEPARATOR})?{dotted_initial}"
        rf"(?:{_TOKEN_SEPARATOR}|$))"
    )
    uppercase_acronym_before_period = rf"[{head}]{{2,3}}(?=\.)"
    letter_proper = (
        rf"(?:{uppercase_acronym_before_period}|{single_letter_proper}|"
        rf"[{head}][{body}0-9]{{1,2}}(?!\.)|"
        rf"[{head}][{body}0-9]{{3,}})"
    )
    proper_branches = [letter_proper, rf"\d{{1,4}}[{body}]?"]
    if pack.name_terminal_continuations:
        proper_branches.append(rf"(?:{_word_alternation(pack.name_terminal_continuations)})")
    proper = nonfinal_guard + rf"(?:{'|'.join(proper_branches)})" + rf"(?![{body}0-9])"
    uppercase_acronym = rf"[{head}]{{2,3}}\."
    dotted = (
        rf"(?!{uppercase_acronym})"
        rf"(?!{dotted_initial}(?:{_TOKEN_SEPARATOR})?{dotted_initial}"
        rf"(?:{_TOKEN_SEPARATOR}|$))"
        rf"[{head}][{body}0-9]{{1,2}}\."
    )
    branches = [dotted]
    if pack.name_continuations:
        continuation = _word_alternation(pack.name_continuations)
        branches.append(rf"(?i:{continuation})(?![{body}0-9])")
    branches.append(proper)
    token = rf"(?:{'|'.join(branches)})"
    generic_name = (
        rf"(?:{_TOKEN_SEPARATOR}{token}){{0,{pack.max_name_tokens - 1}}}"
        rf"{_TOKEN_SEPARATOR}{proper}"
    )
    name_branches: list[str] = []
    dotted_titles = tuple(
        continuation
        for continuation in pack.name_continuations
        if continuation.endswith(".") and len(continuation.rstrip(".")) > 1
    )
    if dotted_titles:
        title = _word_alternation(dotted_titles)
        name_branches.append(
            rf"{_TOKEN_SEPARATOR}(?i:{title})(?![{body}0-9])"
            rf"(?:{_TOKEN_SEPARATOR}{dotted_initial})"
            rf"{{1,{pack.max_name_tokens - 2}}}"
            rf"{_TOKEN_SEPARATOR}{proper}",
        )
    name_branches.append(generic_name)
    if pack.capture_mode == "quoted":
        if not pack.opening_delimiter or not pack.closing_delimiter:
            msg = f"pack {pack.pack_id} needs both quote delimiters"
            raise ValueError(msg)
        cues = tuple(cue.text for cue in pack.cues if cue.position == "pre")
        if len(cues) != len(pack.cues):
            msg = f"pack {pack.pack_id} quoted cues must be prefixes"
            raise ValueError(msg)
        quoted_name = (
            rf"(?:{proper}|{proper}(?:{_TOKEN_SEPARATOR}{token})"
            rf"{{0,{pack.max_name_tokens - 2}}}{_TOKEN_SEPARATOR}{proper})"
        )
        return re.compile(
            rf"\b(?i:{_word_alternation(cues)})(?![{body}0-9])"
            rf"{_TOKEN_SEPARATOR}{re.escape(pack.opening_delimiter)}"
            rf"(?P<value>{quoted_name}){re.escape(pack.closing_delimiter)}",
        )
    capture_branches: list[str] = []
    prefix_cues = _cue_texts(pack, "pre")
    if prefix_cues:
        capture_branches.append(
            rf"(?i:{_word_alternation(prefix_cues)})(?![{body}0-9])"
            rf"(?!{_TOKEN_SEPARATOR}\d{{1,{pack.min_adjacent_digits - 1}}}(?!\d))"
            rf"(?:{'|'.join(name_branches)})",
        )
    suffix_cues = _cue_texts(pack, "post")
    if suffix_cues:
        terminal = ""
        if pack.suffix_terminal_tokens:
            terminal = (
                rf"(?:{_TOKEN_SEPARATOR}(?i:"
                rf"{_word_alternation(pack.suffix_terminal_tokens)})"
                rf"(?![{body}0-9]))?"
            )
        capture_branches.append(
            rf"{proper}(?:{_TOKEN_SEPARATOR}{token})"
            rf"{{0,{pack.max_name_tokens - 1}}}{_TOKEN_SEPARATOR}"
            rf"(?i:{_word_alternation(suffix_cues)})(?![{body}0-9]){terminal}"
            rf"(?!{_TOKEN_SEPARATOR}{proper})",
        )
    if not capture_branches:
        msg = f"pack {pack.pack_id} declares no cased capture cues"
        raise ValueError(msg)
    return re.compile(rf"\b(?P<value>(?:{'|'.join(capture_branches)}))")


def _scriptless_pattern(pack: CuePack) -> re.Pattern[str]:
    if not pack.script_ranges or pack.max_name_chars < 1:
        msg = f"pack {pack.pack_id} needs script ranges and a positive name bound"
        raise ValueError(msg)
    script = "".join(f"\\U{start:08x}-\\U{end:08x}" for start, end in pack.script_ranges)
    left_boundaries = [rf"(?<![{script}\-{_NON_BREAKING_HYPHEN}])"]
    left_boundaries.extend(rf"(?<={re.escape(particle)})" for particle in pack.left_boundary_particles)
    right_boundary = rf"(?=$|[^{script}]"
    if pack.right_boundary_particles:
        right_boundary += rf"|(?:{_literal_alternation(pack.right_boundary_particles)})"
    right_boundary += ")"
    branches: list[str] = []
    prefix_cues = _cue_texts(pack, "pre")
    if prefix_cues:
        script_character = _script_character(script, pack.right_boundary_particles)
        separator = _TOKEN_SEPARATOR if pack.script_name_separator else ""
        name_run = _script_name_run(pack, script_character)
        trailing_number = rf"(?:{_TOKEN_SEPARATOR}\d{{1,4}})?" if pack.allow_trailing_number else ""
        branches.append(
            rf"(?:{_literal_alternation(prefix_cues)})"
            rf"{separator}"
            rf"{name_run}{trailing_number}",
        )
    suffix_cues = _cue_texts(pack, "post")
    if suffix_cues:
        script_character = _script_character(script, pack.left_boundary_particles)
        separator = _TOKEN_SEPARATOR if pack.script_name_separator else ""
        name_run = _script_name_run(pack, script_character)
        branches.append(
            rf"{name_run}"
            rf"{separator}"
            rf"(?:{_literal_alternation(suffix_cues)})",
        )
    if not branches:
        msg = f"pack {pack.pack_id} declares no scriptless capture cues"
        raise ValueError(msg)
    return re.compile(rf"(?:{'|'.join(left_boundaries)})(?P<value>(?:{'|'.join(branches)}))" + right_boundary)


def pack_payload(
    pack: CuePack,
    *,
    blocked_tokens: tuple[str, ...] | None = None,
    blocked_terminal_tokens: tuple[str, ...] | None = None,
    blocked_singleton_tokens: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Return the canonical, hashable content of a pack, including its compiled pattern.

    Returns:
        The canonical, hashable pack payload with its effective blocked-token policy.

    """
    effective_blocked_tokens = blocked_tokens or pack.blocked_adjacent_tokens
    effective_blocked_terminal_tokens = blocked_terminal_tokens or pack.blocked_terminal_tokens
    effective_blocked_singletons = blocked_singleton_tokens or pack.blocked_singleton_tokens
    return {
        **asdict(pack),
        "effective_blocked_tokens": effective_blocked_tokens,
        "effective_blocked_terminal_tokens": effective_blocked_terminal_tokens,
        "effective_blocked_singleton_tokens": effective_blocked_singletons,
        "compiled_pattern": pack_pattern(pack).pattern,
    }


def _cue_texts(pack: CuePack, position: Literal["pre", "post"]) -> tuple[str, ...]:
    """Return cues usable at one side of a cased name run.

    Returns:
        The cue texts declared for the requested side of the name run.

    """
    return tuple(cue.text for cue in pack.cues if cue.position == position)


def _word_alternation(phrases: tuple[str, ...]) -> str:
    """Alternate the phrases longest-first with whitespace-flexible word gaps.

    Returns:
        A longest-first regex alternation with whitespace-flexible word gaps.

    """
    ordered = sorted(phrases, key=lambda phrase: (-len(phrase), phrase))
    return "|".join(_TOKEN_SEPARATOR.join(re.escape(word) for word in phrase.split()) for phrase in ordered)


def _literal_alternation(phrases: tuple[str, ...]) -> str:
    return "|".join(re.escape(phrase) for phrase in sorted(phrases, key=lambda item: (-len(item), item)))


def _script_character(script: str, stop_particles: tuple[str, ...]) -> str:
    character = rf"[{script}]"
    if not stop_particles:
        return character
    return rf"(?!(?:{_literal_alternation(stop_particles)})){character}"


def _script_name_run(pack: CuePack, script_character: str) -> str:
    token = rf"(?:{script_character}){{1,{pack.max_name_chars}}}"
    if not pack.script_name_separator:
        return token + "?"
    separator = rf"(?:{_TOKEN_SEPARATOR}|[-{_NON_BREAKING_HYPHEN}])"
    return token + rf"(?:{separator}{token})" + rf"{{0,{pack.max_script_name_tokens - 1}}}"
