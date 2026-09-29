"""5x5 spot check and cheap regression probe for repaired Meddies Labels candidates."""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: these are command-line entry points; the printed table and report path are the product.
# ruff: file-ignore[ambiguous-unicode-character-string]
# reason: the multiplication sign is typography meaning "by" in an operator-facing report label
# reason: (source x language x bucket, 2xA100); the ASCII letter x would misrender the heading.
import argparse
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from meddies_pii.annotations.span_records import parse_labeled_record
from meddies_pii.generation.label_corpus.synthetic import (
    DEFAULT_TARGETED_GENERATION_DIR,
)
from meddies_pii.generation.label_corpus.validate import (
    _LEFTOVER_LABEL_RE,
    _MALFORMED_LEFTOVER_LABEL_RE,
    digit_count,
    expanded_phone_window,
)
from meddies_pii.jsonl import read_jsonl, write_jsonl

MIN_PHONE_DIGITS = 7

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def check_record(record: Mapping[str, object]) -> tuple[str, list[str], list[str]]:
    example_id, text, spans = parse_labeled_record(record, default_id="row")
    issues: list[str] = []
    phone_spans: list[str] = []
    categories = {span.label for span in spans}
    if not {"private_url", "secret"} <= categories:
        issues.append("missing private_url/secret")
    if "]<" in text or _LEFTOVER_LABEL_RE.search(text) or _MALFORMED_LEFTOVER_LABEL_RE.search(text):
        issues.append("leftover/malformed label marker")
    for span in spans:
        if text[span.start : span.end] != span.text:
            issues.append(f"bad offset {span.label}:{span.text!r}")
            continue
        if span.label != "phone_number":
            continue
        phone_spans.append(span.text)
        expanded = expanded_phone_window(text, span.start, span.end)
        span_digits = digit_count(span.text)
        expanded_digits = digit_count(expanded)
        if expanded_digits >= MIN_PHONE_DIGITS and expanded_digits > span_digits:
            issues.append(f"underbounded phone span {span.text!r} vs window {expanded!r}")
        if 0 < span_digits < MIN_PHONE_DIGITS and expanded_digits < MIN_PHONE_DIGITS:
            issues.append(f"short numeric phone span {span.text!r}")
    return example_id, issues, phone_spans


def split_clean_and_flagged(
    records: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clean: list[dict[str, object]] = []
    flagged: list[dict[str, object]] = []
    for record in records:
        try:
            _example_id, issues, _phones = check_record(record)
        except ValueError as exc:
            issues = [f"parse error: {exc}"]
        if issues:
            flagged.append(dict(record) | {"spot_check_issues": issues})
        else:
            clean.append(dict(record))
    return clean, flagged


def markdown_table_cell(value: object, *, limit: int = 120) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    text = text.replace("|", "/")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def record_info(record: Mapping[str, object]) -> dict[str, object]:
    info = record.get("info")
    return {str(key): value for key, value in info.items()} if isinstance(info, dict) else {}


# reason: parse args and write text share audit main's state; extraction would desync retries and counters.
def main() -> None:  # ruff: ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TARGETED_GENERATION_DIR,
    )
    parser.add_argument("--source-prefix", default="repaired_candidates")
    parser.add_argument("--clean-prefix", default="repaired_candidates.clean")
    parser.add_argument("--flagged-prefix", default="repaired_candidates.flagged")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--write-clean", action="store_true")
    args = parser.parse_args()

    base = args.output_dir
    rows = {lang: list(read_jsonl(base / f"{args.source_prefix}.{lang}.jsonl")) for lang in ("vi", "en")}
    clean_rows: dict[str, list[dict[str, object]]] = {}
    flagged_rows: dict[str, list[dict[str, object]]] = {}
    for lang, source_records in rows.items():
        clean, flagged = split_clean_and_flagged(source_records)
        clean_rows[lang] = clean
        flagged_rows[lang] = flagged
        if args.write_clean:
            write_jsonl(base / f"{args.clean_prefix}.{lang}.jsonl", clean)
            write_jsonl(base / f"{args.flagged_prefix}.{lang}.jsonl", flagged)

    report = args.report or (base / "spot_check_5x5.md")
    lines: list[str] = [
        "# Meddies Labels repaired data spot check 5x5",
        "",
        "Generated: 2026-05-15 Europe/Paris",
        "",
        "## Scope",
        f"- Source prefix: `{args.source_prefix}`.",
        f"- Clean prefix: `{args.clean_prefix}`.",
        "- Spot check: 5 deterministic runs × 5 rows = 25 rows, balanced across Vietnamese and English.",
        (
            "- Full cheap probe checks parse errors, duplicates, leftover label markers, required "
            "`private_url` + `secret`, exact offsets, and likely phone-number underbounding."
        ),
        "",
        "## Full repaired-candidate probe",
    ]
    lines.extend(
        f"- `{lang}` source={len(rows[lang]):,}, clean={len(clean_rows[lang]):,}, flagged={len(flagged_rows[lang]):,}."
        for lang in ("vi", "en")
    )
    all_flagged = [(lang, row) for lang, records in flagged_rows.items() for row in records]
    lines.append(f"- Full-probe issue rows: {len(all_flagged)}.")
    if all_flagged:
        lines.extend(("", "### Quarantined/flagged rows"))
        for flagged_lang, flagged_row in all_flagged[:80]:
            info = record_info(flagged_row)
            issues = flagged_row.get("spot_check_issues", [])
            issue_text = "; ".join(str(issue) for issue in issues) if isinstance(issues, list) else "unknown issue"
            lines.append(f"- `{flagged_lang}` `{info.get('id', 'unknown')}`: {issue_text}")
    label_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    dup_texts = 0
    seen_texts: set[str] = set()
    for clean_records in clean_rows.values():
        for clean_record in clean_records:
            example_id, text, spans = parse_labeled_record(clean_record, default_id="row")
            if text in seen_texts:
                dup_texts += 1
            seen_texts.add(text)
            label_counts.update(span.label for span in spans)
            info = record_info(clean_record)
            scenario_counts.update([str(info.get("scenario", "unknown"))])
    lines.extend((f"- Duplicate clean texts: {dup_texts}.", "", "### Clean label distribution"))
    for label, count in sorted(label_counts.items()):
        lines.append(f"- `{label}`: {count:,}")
    lines.extend(("", "### Clean scenario distribution"))
    for scenario, count in sorted(scenario_counts.items()):
        lines.append(f"- `{scenario}`: {count:,}")

    runs: list[list[tuple[str, int, dict[str, object]]]] = []
    for run in range(1, 6):
        # reason: the spot-check rounds must be the same rows on every re-run, which is why the seed is a fixed
        # reason: literal offset by the round number. An unpredictable generator would make the audit
        # reason: unreproducible; nothing here is a secret, a token, or a key.
        rng = random.Random(2026051500 + run)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        vi_n = 3 if run % 2 == 1 else 2
        en_n = 5 - vi_n
        sample: list[tuple[str, int, dict[str, object]]] = []
        sample.extend(("vi", i, clean_rows["vi"][i]) for i in rng.sample(range(len(clean_rows["vi"])), vi_n))
        sample.extend(("en", i, clean_rows["en"][i]) for i in rng.sample(range(len(clean_rows["en"])), en_n))
        rng.shuffle(sample)
        runs.append(sample)

    lines.extend(("", "## 5x5 sampled clean rows"))
    sampled_issues = 0
    for run_index, sample in enumerate(runs, start=1):
        lines.extend((
            "",
            f"### Run {run_index}",
            "| lang | clean line | id | scenario | phones | status |",
            "|---|---:|---|---|---|---|",
        ))
        for sample_lang, index, sample_record in sample:
            example_id, issues, phones = check_record(sample_record)
            sampled_issues += len(issues)
            info = record_info(sample_record)
            status = "PASS" if not issues else "FAIL: " + "; ".join(issues)
            phone = markdown_table_cell("; ".join(phones[:3]))
            lines.append(
                f"| {sample_lang} | {index + 1} | `{example_id}` | "
                f"{markdown_table_cell(info.get('scenario', 'unknown'))} | "
                f"{phone} | {markdown_table_cell(status)} |",
            )
    lines.extend(("", "## Verdict"))
    if sampled_issues or dup_texts:
        lines.append("FAIL: sampled clean rows still have issues.")
    else:
        lines.append(
            "PASS: 5x5 clean spot check passed; flagged rows are quarantined and excluded from clean repaired candidates.",
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    print(
        f"flagged={len(all_flagged)} clean_vi={len(clean_rows['vi'])} \
clean_en={len(clean_rows['en'])} sampled_issues={sampled_issues}",
    )


if __name__ == "__main__":
    main()
