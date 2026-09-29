#!/usr/bin/env python
from __future__ import annotations

# ruff: file-ignore[print]
# reason: this is a command-line script; its standard output is the product.
import argparse
from pathlib import Path

from meddies_pii.training.bioes.reports.inference_preview import (
    DEFAULT_INFERENCE_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULT_JSON,
    write_inference_preview_report,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a BIOES inference preview HTML report from persisted JSON.")
    parser.add_argument("--preview-json", type=Path, default=DEFAULT_INFERENCE_JSON)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    html_path = write_inference_preview_report(
        preview_json=args.preview_json,
        result_json=args.result_json,
        output_dir=args.output_dir,
    )
    print(f"HTML preview: file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
