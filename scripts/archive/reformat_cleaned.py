#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[docstring-missing-exception]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[import-outside-top-level]
# reason: Modal function bodies import inside the container, where the machine-learning stack exists; the client running
# reason: this script does not have it.
# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
import os
from typing import TYPE_CHECKING

from meddies_pii.tags import TAG_PATTERN

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

ALL_CONFIGS = [
    "burmese",
    "chinese",
    "english",
    "filipino",
    "french",
    "german",
    "indonesian",
    "japanese",
    "korean",
    "laos",
    "malay",
    "portuguese",
    "russian",
    "spanish",
    "tamil",
    "thai",
    "vietnamese",
    "nvidia-health",
    "nvidia-non-health",
]

RENAME = {"content": "text", "text": "label"}
NVIDIA_RENAME = {"text_tagged": "text", "text": "label"}
VIE_PII_RENAME = {"output": "text", "text": "label"}


def strip_pii_tags(text: str) -> str:
    return TAG_PATTERN.sub(r"\1", text)


def make_reformat_fn(rename_map: Mapping[str, str]) -> Callable[[dict[str, object]], dict[str, object]]:
    def reformat_row(row: dict[str, object]) -> dict[str, object]:
        source_key = next(k for k, v in rename_map.items() if v == "text")
        tagged = row.get(source_key, "")
        if not isinstance(tagged, str):
            msg = f"row field {source_key!r} must hold text, got {type(tagged).__name__}"
            raise TypeError(msg)
        out = dict(row)
        for old_key in rename_map:
            out.pop(old_key, None)
        out["text"] = tagged
        out["raw"] = strip_pii_tags(tagged)
        out["label"] = row.get(next(k for k, v in rename_map.items() if v == "label"), "{}")
        return out

    return reformat_row


# reason: process config exposes source repo/rename map as its public contract; bundling would break callers.
def process_config(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]
    source_repo: str,
    config: str | None,
    target_repo: str,
    target_config: str,
    token: str,
    rename_map: Mapping[str, str],
) -> int:
    """Only remove the columns we're renaming, keep all others."""
    from datasets import load_dataset

    ds = load_dataset(source_repo, config, token=token, split="train")
    reformat_fn = make_reformat_fn(rename_map)

    cols_to_remove = [c for c in rename_map if c in ds.column_names]
    reformatted = ds.map(reformat_fn, remove_columns=cols_to_remove)

    reformatted.push_to_hub(target_repo, config_name=target_config, token=token)
    print(
        f"  {source_repo}/{config or 'default'} -> {target_config} ({len(reformatted)} rows, cols: \
{reformatted.column_names})",
    )

    return len(reformatted)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", nargs="+", default=ALL_CONFIGS)
    parser.add_argument("--source", default="Meddies/meddies-pii-cleaned")
    parser.add_argument("--target", default="Meddies/meddies-pii")
    parser.add_argument("--skip", nargs="+", default=[], help="Configs to skip")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Required to log in to Hugging Face and push reformatted configs.",
    )
    args = parser.parse_args(argv)
    if not args.publish:
        parser.error("this legacy script publishes to Hugging Face; pass --publish to execute")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Vietnamese also published as the default config for older consumers."""
    args = parse_args(argv)
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()

    token = os.environ["HF_TOKEN"]
    login(token=token)

    configs = [c for c in args.configs if c not in args.skip]
    print(f"Processing {len(configs)} configs + vietnamese-translated -> {args.target}\n")

    total = 0
    failures: list[str] = []
    for config in configs:
        try:
            rename_map = NVIDIA_RENAME if "nvidia" in config else RENAME
            total += process_config(args.source, config, args.target, config, token, rename_map)
        # reason: one config failing is recorded and the remaining configs still run. The conversion path reaches the hub,
        # reason: the dataset loaders and the tokenizers, so no nameable exception set covers it without dropping a real
        # reason: failure on the floor.
        except Exception as e:  # ruff: ignore[blind-except,try-except-in-loop]
            print(f"  {config}: ERROR - {e}")
            failures.append(f"{config}: {e}")

    try:
        total += process_config(args.source, "vietnamese", args.target, "default", token, RENAME)
    # reason: one config failing is recorded and the remaining configs still run. The conversion path reaches the hub,
    # reason: the dataset loaders and the tokenizers, so no nameable exception set covers it without dropping a real
    # reason: failure on the floor.
    except Exception as e:  # ruff: ignore[blind-except]
        print(f"  default: ERROR - {e}")
        failures.append(f"default: {e}")

    try:
        total += process_config(
            "Meddies/vie-pii",
            None,
            args.target,
            "vietnamese-translated",
            token,
            VIE_PII_RENAME,
        )
    # reason: one config failing is recorded and the remaining configs still run. The conversion path reaches the hub,
    # reason: the dataset loaders and the tokenizers, so no nameable exception set covers it without dropping a real
    # reason: failure on the floor.
    except Exception as e:  # ruff: ignore[blind-except]
        print(f"  vietnamese-translated: ERROR - {e}")
        failures.append(f"vietnamese-translated: {e}")

    print(f"\nDone. {total} total rows pushed.")
    if failures and not args.allow_partial:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
