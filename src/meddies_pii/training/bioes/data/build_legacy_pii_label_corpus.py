from __future__ import annotations

# ruff: file-ignore[print]
# reason: results and progress travel back through the streamed run log, because Modal's large-result blob path is
# reason: unimplemented in this workspace.
import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from meddies_pii.historical_artifacts import (
    legacy_audit_jsonl_locator,
    legacy_jsonl_locator,
    legacy_summary_json_locator,
)
from meddies_pii.languages import MEDDIES_PII_LANGUAGE_CONFIGS
from meddies_pii.training.bioes.data.legacy_to_pii_labels import (
    write_legacy_conversion_artifacts,
)


def _language_for_config(config: str) -> str:
    if config == "vietnamese-translated":
        return "Vietnamese"
    return config.replace("-", " ").title()


def _load_rows(
    dataset_id: str,
    *,
    config: str,
    split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_id, config, split=split)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        record = dict(row)
        record.setdefault("source", config)
        record.setdefault("language", _language_for_config(config))
        rows.append(record)
    return rows


def _load_rows_for_configs(
    dataset_id: str,
    *,
    configs: list[str],
    split: str,
    limit_per_config: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        rows.extend(
            _load_rows(
                dataset_id,
                config=config,
                split=split,
                limit=limit_per_config,
            ),
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Meddies Labels span JSONL plus migration audit sidecars.")
    parser.add_argument("--dataset-id", default="Meddies/meddies-pii")
    parser.add_argument(
        "--config",
        default="train",
        help=(
            "Dataset config/subset. Comma-separate multiple configs to build a combined pool. "
            "Use 'all-languages' for every language config exposed by Meddies/meddies-pii."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum rows per config before migration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed") / legacy_jsonl_locator("train"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/processed") / legacy_audit_jsonl_locator("train"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/processed") / legacy_summary_json_locator("train"),
    )
    args = parser.parse_args()

    configs = [item.strip() for item in args.config.split(",") if item.strip()]
    if configs == ["all-languages"]:
        configs = list(MEDDIES_PII_LANGUAGE_CONFIGS)
    rows = _load_rows_for_configs(
        args.dataset_id,
        configs=configs,
        split=args.split,
        limit_per_config=args.limit,
    )
    summary = write_legacy_conversion_artifacts(
        rows,
        output_path=args.output,
        audit_path=args.audit,
        summary_path=args.summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
