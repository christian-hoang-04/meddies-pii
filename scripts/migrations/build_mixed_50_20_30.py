#!/usr/bin/env python
"""Build a combined 50/20/30 (en/vi/other) Meddies Labels training set and publish it to a NEW HF repo.

The set is pooled from v1 + v2 + external-trio + ai4privacy.

Sources (TRAIN splits only; eval sets left untouched):
  * v1  (Meddies/meddies-pii)          — 17 per-language configs, inline ``[value]<label>``
        format -> converted via ``convert_config_row``; all clinical (medical).
  * v2  (Meddies/meddies-pii-v2)       — 17 per-language configs, already char-spans;
        domain from ``info.domain_bucket`` (medical/general/code_logs).
  * trio (…-external gretel/nemotron/creddata_en) — English char-spans; domain from info.
  * ai4privacy (…-external ai4privacy_<code>)     — char-spans, all general, 12 langs.

Everything is normalized to ``{text, label:[{category,start,end,text}],
info:{language, domain_bucket, source_dataset, uid}}`` (domain_bucket collapsed to
``medical`` | ``general``).

Ratio: en 40% / vi 30% / other 30%, computed over the pooled TRAIN data (vi=30%
uses ALL available Vietnamese, the binding bucket).
  * The binding bucket (smallest avail/target) sets the total.
  * en / vi are trimmed general-first (all clinical kept); ``other`` is trimmed
    EQUAL-per-language, clinical-first.
Random deletion is seeded (reproducible). Dry-run by default: prints the exact plan
and sends NOTHING to the hub. ``--push --repo <id>`` publishes.
"""

from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-yields]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import random
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

from meddies_pii.training.bioes.data.inline_tags import convert_config_row

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

type TallyKey = tuple[str, str]
type QuotaKey = tuple[str | tuple[str, str], str]
type Record = dict[str, Any]

V1, V2, EXT = (
    "Meddies/meddies-pii",
    "Meddies/meddies-pii-v2",
    "Meddies/meddies-pii-external",
)
SEED = 42
SHARES = {"en": 0.40, "vi": 0.30, "other": 0.30}

LANG17 = [
    "vietnamese",
    "english",
    "french",
    "german",
    "spanish",
    "laos",
    "thai",
    "burmese",
    "indonesian",
    "filipino",
    "malay",
    "tamil",
    "portuguese",
    "russian",
    "chinese",
    "japanese",
    "korean",
]
OTHER15 = [lang for lang in LANG17 if lang not in {"vietnamese", "english"}]
AI4_CODE2LANG = {
    "en": "english",
    "vi": "vietnamese",
    "de": "german",
    "es": "spanish",
    "fil": "filipino",
    "fr": "french",
    "id": "indonesian",
    "ja": "japanese",
    "ko": "korean",
    "ms": "malay",
    "pt": "portuguese",
    "zh": "chinese",
}


def bucket(language: str) -> str:
    return "vi" if language == "vietnamese" else "en" if language == "english" else "other"


def norm_domain(x: object) -> str:
    x = str(x).lower()
    return "medical" if x in {"medical", "healthcare"} else "general"


_VN_RE = re.compile(
    r"[đĐơƠưƯăĂâÂêÊôÔ"
    r"àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)
"""vietnamese-translated is a v1 config of GENERAL-domain docs, mostly Vietnamese with some English.

Its `language` field is None, so detect per row (Vietnamese- only characters that other Latin scripts don't use).

"""


def is_vietnamese(text: str) -> bool:
    return bool(_VN_RE.search(text or ""))


def _sources() -> list[tuple[str, str, str, str, str | None, str]]:
    """(repo, config, source_dataset, language, domain_override, kind).

    extra v1 config: general-domain, mostly-Vietnamese, language detected per row.

    """
    s: list[tuple[str, str, str, str, str | None, str]] = []
    for lang in LANG17:
        s.extend(((V1, lang, "meddies-pii-v1", lang, "medical", "v1"), (V2, lang, "meddies-pii-v2", lang, None, "span")))
    s.append((
        V1,
        "vietnamese-translated",
        "meddies-pii-v1",
        "vietnamese",
        "general",
        "v1trans",
    ))
    s.extend(
        (EXT, cfg, cfg.replace("_en", ""), "english", None, "span") for cfg in ("gretel_en", "nemotron_en", "creddata_en")
    )
    for code, lang in AI4_CODE2LANG.items():
        s.append((EXT, f"ai4privacy_{code}", "ai4privacy", lang, "general", "span"))
    return s


# reason: iter records exposes repo/kind as its public contract; bundling would break callers.
def iter_records(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    repo: str,
    config: str,
    source_dataset: str,
    language: str,
    domain_override: str | None,
    kind: str,
) -> Iterator[tuple[dict[str, Any], str, str, str]]:
    """Yield (record, bucket, domain, language) for every usable TRAIN row."""
    ds = load_dataset(repo, config, split="train")
    b = bucket(language)
    if kind == "v1":
        for i, row in enumerate(ds):
            rec = convert_config_row(row, uid=f"{source_dataset}:{config}:{i}", default_language=language)
            if rec is None:
                continue
            info = {
                "language": language,
                "domain_bucket": "medical",
                "source_dataset": source_dataset,
                "uid": rec["info"]["uid"],
            }
            yield (
                {"text": rec["text"], "label": rec["label"], "info": info},
                b,
                "medical",
                language,
            )
        return
    if kind == "v1trans":
        for i, row in enumerate(ds):
            rec = convert_config_row(row, uid=f"{source_dataset}:{config}:{i}", default_language=language)
            if rec is None:
                continue
            lang = "vietnamese" if is_vietnamese(rec["text"]) else "english"
            info = {
                "language": lang,
                "domain_bucket": "general",
                "source_dataset": source_dataset,
                "uid": rec["info"]["uid"],
            }
            yield (
                {"text": rec["text"], "label": rec["label"], "info": info},
                bucket(lang),
                "general",
                lang,
            )
        return
    texts, labels, infos = ds["text"], ds["label"], ds["info"]
    for i in range(len(ds)):
        if not labels[i]:
            continue
        domain = domain_override or norm_domain((infos[i] or {}).get("domain_bucket"))
        info = {
            "language": language,
            "domain_bucket": domain,
            "source_dataset": source_dataset,
            "uid": str((infos[i] or {}).get("id") or f"{config}:{i}"),
        }
        yield {"text": texts[i], "label": labels[i], "info": info}, b, domain, language


def tally() -> tuple[Counter[TallyKey], Counter[TallyKey]]:
    """Count usable rows per (bucket, domain) and per (language, domain) for 'other'."""
    bd: Counter[TallyKey] = Counter()
    other_ld: Counter[TallyKey] = Counter()
    for src in _sources():
        for _rec, b, domain, lang in iter_records(*src):
            bd[b, domain] += 1
            if b == "other":
                other_ld[lang, domain] += 1
    return bd, other_ld


def build_plan(bd: Counter[TallyKey], other_ld: Counter[TallyKey], total_target: int | None = None) -> dict[str, Any]:
    """size-first mode: keep ALL English + ALL Vietnamese, fill 'other' to hit N."""
    avail = {b: bd[b, "medical"] + bd[b, "general"] for b in ("en", "vi", "other")}
    if total_target is not None:
        total = total_target
        tgt = {"en": avail["en"], "vi": avail["vi"]}
        tgt["other"] = total - tgt["en"] - tgt["vi"]
    else:
        total = min(int(avail[b] / SHARES[b]) for b in avail)
        tgt = {"en": round(SHARES["en"] * total), "vi": round(SHARES["vi"] * total)}
        tgt["other"] = total - tgt["en"] - tgt["vi"]

    def keep_general_first(b: str) -> tuple[int, int]:
        clin = bd[b, "medical"]
        keep_clin = min(clin, tgt[b])
        keep_gen = max(0, tgt[b] - keep_clin)
        return keep_clin, keep_gen

    per = tgt["other"] // len(OTHER15)
    rem = tgt["other"] - per * len(OTHER15)
    other_keep: dict[str, tuple[int, int]] = {}
    for j, lang in enumerate(OTHER15):
        t = per + (1 if j < rem else 0)
        clin = other_ld[lang, "medical"]
        kc = min(clin, t)
        other_keep[lang] = (kc, max(0, t - kc))
    return {
        "total": total,
        "avail": avail,
        "tgt": tgt,
        "en": keep_general_first("en"),
        "vi": keep_general_first("vi"),
        "other_keep": other_keep,
        "bd": bd,
    }


def print_plan(plan: dict[str, Any]) -> None:
    bd, tgt = plan["bd"], plan["tgt"]
    print(f"\nBinding total = {plan['total']:,}   avail={plan['avail']}")
    print(f"\n{'bucket':<8}{'avail':>10}{'target':>10}{'clin-keep':>11}{'gen-keep':>10}")
    for b in ("en", "vi"):
        kc, kg = plan[b]
        print(f"{b:<8}{plan['avail'][b]:>10,}{tgt[b]:>10,}{kc:>11,}{kg:>10,}")
    okc = sum(c for c, _ in plan["other_keep"].values())
    okg = sum(g for _, g in plan["other_keep"].values())
    print(f"{'other':<8}{plan['avail']['other']:>10,}{tgt['other']:>10,}{okc:>11,}{okg:>10,}")
    print("\n'other' equal-per-language keep:")
    for lang, (c, g) in plan["other_keep"].items():
        print(f"   {lang:<12}{c + g:>8,}  (clin {c:,}{', gen ' + format(g, ',') if g else ''})")
    kept = tgt["en"] + tgt["vi"] + tgt["other"]
    clin_kept = plan["en"][0] + plan["vi"][0] + okc
    print(
        f"\nRESULT total={kept:,}  en={100 * tgt['en'] / kept:.1f}%  "
        f"vi={100 * tgt['vi'] / kept:.1f}%  other={100 * tgt['other'] / kept:.1f}%",
    )
    all_clin = sum(bd[b, "medical"] for b in ("en", "vi", "other"))
    print(f"clinical kept={clin_kept:,} / {all_clin:,}  -> clinical DELETED={all_clin - clin_kept:,}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--push", action="store_true", help="Publish to --repo (else dry-run).")
    p.add_argument("--repo", help="Target HF dataset repo id (required with --push).")
    p.add_argument(
        "--total",
        type=int,
        help="Size-first mode: target total rows (keep all en+vi, fill other). Overrides the SHARES ratio.",
    )
    return p.parse_args(argv)


def select_and_push(plan: dict[str, Any], repo: str) -> None:
    """reservoir-sample per (bucket|lang, domain) so we hold only the kept subset."""
    # reason: the fixed SEED is the contract — the published 50/20/30 mix must be reconstructible
    # reason: from this script alone, which a cryptographic source would make impossible.
    rng = random.Random(SEED)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    quotas: dict[QuotaKey, int] = {}
    for b in ("en", "vi"):
        quotas[b, "medical"], quotas[b, "general"] = plan[b]
    for lang, (c, g) in plan["other_keep"].items():
        quotas[("other", lang), "medical"] = c
        quotas[("other", lang), "general"] = g
    seen: Counter[QuotaKey] = Counter()
    reservoir: dict[QuotaKey, list[Record]] = defaultdict(list)
    for src in _sources():
        for rec, b, domain, lang in iter_records(*src):
            key = (b, domain) if b != "other" else (("other", lang), domain)
            q = quotas.get(key, 0)
            if q == 0:
                continue
            seen[key] += 1
            if len(reservoir[key]) < q:
                reservoir[key].append(rec)
            else:
                j = rng.randint(0, seen[key] - 1)
                if j < q:
                    reservoir[key][j] = rec
    rows = [r for rs in reservoir.values() for r in rs]
    rng.shuffle(rows)
    print(f"pushing {len(rows):,} rows to {repo} ...", flush=True)
    Dataset.from_list(rows).push_to_hub(repo, split="train")
    print("DONE.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print("tallying pool (loads + converts all TRAIN sources)...", flush=True)
    bd, other_ld = tally()
    plan = build_plan(bd, other_ld, args.total)
    print_plan(plan)
    if not args.push:
        print("\nDRY RUN — re-run with --push --repo <id> to publish. Nothing sent.")
        return 0
    if not args.repo:
        msg = "--push requires --repo <hf-dataset-id>"
        raise SystemExit(msg)
    select_and_push(plan, args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
