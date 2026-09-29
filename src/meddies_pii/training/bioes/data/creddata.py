r"""Convert Samsung/CredData credential annotations into Meddies Labels span records.

CredData annotates credentials by ``(LineStart/End, ValueStart/End)`` over code
files. This converter resolves each TRUE (``GroundTruth == 'T'``) credential's
``Category`` to a Meddies Labels label, locates it in the file, and emits a windowed
code snippet with the credential as a span. False/Unknown (``F``/``X``) rows and
non-PII categories stay in the snippet text untagged — natural hard negatives.

Offset convention matches CredData's own code (obfuscate_creds.py / review_data.py):
lines are 1-indexed inclusive, columns are 0-indexed Python-slice; the file text is
normalized ``\\r\\n``/``\\r`` -> ``\\n`` BEFORE splitting on ``\\n``; a multi-line
value runs from ``ValueStart`` on the first line to ``ValueEnd`` on the last.

Category -> Meddies Labels (priority, highest wins):
  credential -> ``secret``  >  URL Credentials -> ``private_url``
  >  UUID / AWS Client ID / *app id -> ``id_number``  >  drop (untagged).
``CryptographyKey == 'Public'`` -> drop (a public key is not a secret).
"""

from __future__ import annotations

import operator
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from meddies_pii.languages import language_bucket
from meddies_pii.spans import CharSpan
from meddies_pii.training.bioes.data.external_records import (
    dedupe_non_overlapping,
    external_record,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from meddies_pii.training.bioes.data.record_schema import NormalizedRecord

DATASET_ID = "Samsung/CredData"

_DROP_RULES = frozenset({"nonce", "salt", "aws s3 bucket", "firebase domain", "public key"})
"""Non-credential categories → no Meddies Labels home (kept in text, untagged)."""
_PRIVATE_URL_RULES = frozenset({"url credentials"})
_ID_NUMBER_RULES = frozenset({"uuid", "aws client id"})

_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s'\"<>\\)\]}]+")
"""A URL with a scheme.

Used to expand a URL-embedded credential span to the whole (access-bearing) URL for private_url. Stops at
whitespace/quotes/closing brackets.

"""


def _is_identifier(rule: str) -> bool:
    return rule in _ID_NUMBER_RULES or rule.endswith("app id")


# reason: Credential category, kind, and context jointly determine one PII label; splitting would reorder policy.
def creddata_category_to_pii_label(category: str, crypto_key: str, ground_truth: str) -> str | None:  # ruff: ignore[complex-structure,too-many-return-statements]
    """Resolve a CredData ``Category`` (colon-separated rules) to a Meddies Labels label.

    Returns ``None`` for rows that should stay untagged (F/X, public keys, and the
    non-credential drop set).

    URL Credentials wins: the credential lives inside a URL, so the converter tags the whole (access-bearing) URL as
    private_url — falling back to secret if no enclosing URL is actually found around the credential.

    Returns:
        The Meddies Labels label for the row, or ``None`` for a row that must stay UNTAGGED.
        ``None`` is a positive decision rather than a failure to classify: a ground truth
        other than T, a public crypto key, and the non-credential drop set all mean the row
        is a true negative that belongs in the snippet with no span on it. A caller that
        treats ``None`` as "skip this row entirely" loses the negatives the corpus needs.

    """
    if (ground_truth or "").strip().upper() != "T":
        return None
    if (crypto_key or "").strip().lower() == "public":
        return None
    rules = [r.strip().lower() for r in (category or "").split(":") if r.strip()]
    if not rules:
        return None

    def kind(rule: str) -> str:
        if rule in _DROP_RULES:
            return "drop"
        if rule in _PRIVATE_URL_RULES:
            return "url"
        if _is_identifier(rule):
            return "id"
        return "secret"

    kinds = {kind(r) for r in rules}
    if "url" in kinds:
        return "private_url"
    if "secret" in kinds:
        return "secret"
    if "id" in kinds:
        return "id_number"
    return None


@dataclass(frozen=True, slots=True)
class CredMetaRow:
    """The subset of a CredData meta CSV row the converter needs."""

    line_start: int
    line_end: int
    value_start: int
    value_end: int
    ground_truth: str
    crypto_key: str
    category: str

    @classmethod
    def from_csv(cls, row: Mapping[str, Any]) -> CredMetaRow:
        """F (false) rows often omit a precise value position (empty / -1).

        Tolerate it — those rows are skipped (not tagged) downstream anyway.

        Returns:
            The row with a missing or unparseable value position normalized to -1 rather than
            rejected. -1 is the sentinel the converter reads as "line granularity": it spans
            the whole line block instead of a precise column range, which is what multi-line
            PEM and BASE64 private keys need.

        """
        return cls(
            line_start=int(row["LineStart"]),
            line_end=int(row["LineEnd"]),
            value_start=int(row.get("ValueStart") or -1),
            value_end=int(row.get("ValueEnd") or -1),
            ground_truth=str(row.get("GroundTruth") or ""),
            crypto_key=str(row.get("CryptographyKey") or ""),
            category=str(row.get("Category") or ""),
        )


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _enclosing_url(line: str, col_start: int, col_end: int) -> tuple[int, int] | None:
    """(start, end) of a URL on this line that contains [col_start, col_end), if any.

    Returns:
        The column bounds of the first URL on the line that fully ENCLOSES the credential
        span, or ``None`` when no URL contains it. Containment is required in both
        directions, so a URL merely adjacent to the credential does not match and the caller
        correctly falls back to tagging the credential itself as a secret.

    """
    for m in _URL_RE.finditer(line):
        if m.start() <= col_start and m.end() >= col_end:
            return m.start(), m.end()
    return None


def _line_start_offsets(lines: list[str]) -> list[int]:
    r"""offsets[i] = char offset where line i (0-indexed) starts in the \\n-joined text.

    Returns:
        One offset per line PLUS a trailing entry one past the end, so the list is one longer
        than ``lines`` and ``offsets[i + 1]`` is always safe to read for a valid line index.
        Each line is counted as its own length plus one for the separator, so the offsets are
        correct only against text already normalized to ``\\n`` by ``_normalize``.

    """
    offsets: list[int] = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


# reason: convert creddata combines normalize and line offsets; splitting would misattribute row errors.
def convert_creddata_file(  # ruff: ignore[complex-structure,too-many-branches,too-many-arguments,too-many-locals,too-many-statements]
    file_text: str,
    meta_rows: Iterable[CredMetaRow],
    *,
    repo: str,
    file_id: str,
    dataset_id: str = DATASET_ID,
    window_lines: int = 12,
    max_window_chars: int = 12_000,
) -> tuple[list[NormalizedRecord], Counter[str]]:
    """Convert one CredData file + its meta rows into windowed Meddies Labels records.

    One record per cluster of nearby credentials; F/X and non-PII rows are left in
    the snippet untagged. ``dropped`` counts rows skipped on errors (out-of-range
    offsets), not the intentionally-untagged negatives.

    Rows without a precise value column (mostly multi-line PEM/BASE64 private keys, marked at line granularity) → span the
    whole line block: ValueStart missing → start of the first line, ValueEnd missing → end of the last line.

    Trim surrounding whitespace (e.g. indentation on a whole-line span).

    URL Credentials: the credential lives in a URL. Expand the span to the whole enclosing URL → private_url; if no URL is
    actually found, it's just a credential → secret.

    Cluster credentials whose context windows would overlap → one record each.

    Shrink context to honor the char cap, but never drop the credential lines.

    2-domain corpus: CredData is code/credentials, no clinical content, so every record is general (not healthcare).

    Returns:
        The windowed records and a counter of rows dropped on ERRORS, keyed by category and
        cause (``line_oob``, ``offset_oob``). The counter deliberately excludes the
        intentionally-untagged negatives -- F/X rows and public keys stay in the snippet with
        no span and are not dropped -- so a rising ``dropped`` total means the meta rows and
        the file text disagree, never that the file held few credentials.

    """
    norm = _normalize(file_text)
    lines = norm.split("\n")
    offsets = _line_start_offsets(lines)
    n_lines = len(lines)
    dropped: Counter[str] = Counter()

    creds: list[tuple[int, int, str, int, int]] = []
    for row in meta_rows:
        label = creddata_category_to_pii_label(row.category, row.crypto_key, row.ground_truth)
        if label is None:
            continue
        if not 1 <= row.line_start <= row.line_end <= n_lines:
            dropped[f"{row.category.lower()}:line_oob"] += 1
            continue
        value_start = max(row.value_start, 0)
        value_end = row.value_end if row.value_end >= 0 else len(lines[row.line_end - 1])
        start = offsets[row.line_start - 1] + value_start
        end = offsets[row.line_end - 1] + value_end
        if not 0 <= start < end <= len(norm):
            dropped[f"{row.category.lower()}:offset_oob"] += 1
            continue
        while start < end and norm[start] in " \t\r\n":
            start += 1
        while end > start and norm[end - 1] in " \t\r\n":
            end -= 1
        if start >= end:
            dropped[f"{row.category.lower()}:empty_after_trim"] += 1
            continue
        if label == "private_url":
            line_off = offsets[row.line_start - 1]
            url = _enclosing_url(lines[row.line_start - 1], start - line_off, end - line_off)
            if url is not None:
                start, end = line_off + url[0], line_off + url[1]
            else:
                label = "secret"
        creds.append((start, end, label, row.line_start, row.line_end))

    if not creds:
        return [], dropped

    creds.sort(key=operator.itemgetter(3, 0))
    clusters: list[tuple[list[tuple[int, int, str, int, int]], int, int]] = []
    current = [creds[0]]
    lo, hi = creds[0][3], creds[0][4]
    for cred in creds[1:]:
        if cred[3] <= hi + 2 * window_lines:
            current.append(cred)
            hi = max(hi, cred[4])
        else:
            clusters.append((current, lo, hi))
            current, lo, hi = [cred], cred[3], cred[4]
    clusters.append((current, lo, hi))

    def window_bounds(lo: int, hi: int, wl: int) -> tuple[int, int]:
        first = max(1, lo - wl)
        last = min(n_lines, hi + wl)
        return offsets[first - 1], offsets[last - 1] + len(lines[last - 1])

    records: list[NormalizedRecord] = []
    for cluster, lo, hi in clusters:
        wl = window_lines
        snip_start, snip_end = window_bounds(lo, hi, wl)
        while snip_end - snip_start > max_window_chars and wl > 0:
            wl //= 2
            snip_start, snip_end = window_bounds(lo, hi, wl)
        snippet = norm[snip_start:snip_end]
        spans = dedupe_non_overlapping([
            CharSpan(
                start=start - snip_start,
                end=end - snip_start,
                text=norm[start:end],
                label=label,
            )
            for start, end, label, _ls, _le in cluster
        ])
        if not spans:
            continue
        records.append(
            external_record(
                text=snippet,
                spans=spans,
                uid=f"{dataset_id}:{file_id}:{lo}",
                source_dataset=dataset_id,
                source=repo,
                language="en",
                domain_bucket="general",
                language_bucket=language_bucket(language="en", source=repo),
            ),
        )
    return records, dropped
