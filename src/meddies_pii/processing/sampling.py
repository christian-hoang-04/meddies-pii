from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from meddies_pii.annotations.inline_records import parse_inline_record
from meddies_pii.jsonl import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def sample_and_check(file_path: str, sample_size: int = 5000) -> None:
    input_path = Path(file_path)
    if not input_path.exists():
        logger.error("File not found: %s", file_path)
        return

    logger.info("Reading %s...", file_path)

    error_line_indices: list[int] = []
    valid_records: list[dict[str, Any]] = []

    for index, record in enumerate(read_jsonl(input_path)):
        if "error" in record:
            error_line_indices.append(index)
            continue
        valid_records.append(record)

    if error_line_indices:
        logger.info(
            'Found %s lines with "error"; remove these before publishing.',
            len(error_line_indices),
        )

    if not valid_records:
        logger.warning("No valid records found.")
        return

    logger.info("Total valid records: %s", len(valid_records))

    actual_sample_size = min(sample_size, len(valid_records))
    sampled_records = random.sample(valid_records, actual_sample_size)

    logger.info("--- Quality check on %s samples ---", actual_sample_size)

    passed_count = 0
    failure_details: list[dict[str, Any]] = []

    for index, record in enumerate(sampled_records):
        parsed, errors = parse_inline_record(record, line_number=index + 1)
        if parsed is not None and not errors:
            passed_count += 1
            continue
        preview = record.get("content") or record.get("text_tagged") or record.get("output") or ""
        failure_details.append({
            "index": index,
            "errors": list(errors),
            "content_preview": str(preview)[:100] + "...",
        })

    pass_rate = passed_count / actual_sample_size * 100
    logger.info("Passed: %s/%s (%.1f%%)", passed_count, actual_sample_size, pass_rate)

    if failure_details:
        logger.info("Failure examples:")
        for failure in failure_details[:5]:
            logger.info(
                "- Sample %s: %s",
                failure["index"],
                ", ".join(failure["errors"]),
            )
            logger.info("  Preview: %s", failure["content_preview"])

    output_path = input_path.parent / f"sample_{input_path.name}"
    write_jsonl(output_path, sampled_records)

    logger.info("Samples saved to %s for manual inspection.", output_path)
