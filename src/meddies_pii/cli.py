# ruff: file-ignore[import-outside-top-level]
# reason: subcommands import their own dependencies, so starting the CLI does not pay for all of them.
# ruff: file-ignore[print]
# reason: this is the command-line entry point; its standard output is the product.
import argparse
import asyncio
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from meddies_pii.annotations.inline_records import ValidationResult, validate_inline_jsonl
from meddies_pii.generation.gemini.inference import (
    GeminiClient,
    cancel_job,
    delete_job,
    download_batch_results,
    get_client,
    list_jobs,
    merge_results,
    monitor_job,
    prepare_hf_review,
    prepare_jsonl,
    save_results_to_json,
    submit_batch_job,
)
from meddies_pii.generation.label_corpus.catalog import DOMAIN_PROFILES, SPLIT_PURPOSES
from meddies_pii.languages import SUPPORTED_LANGUAGES
from meddies_pii.offline_demo import write_demo_output
from meddies_pii.processing.sampling import sample_and_check
from meddies_pii.taxonomy import PII_LABELS

DEFAULT_SAMPLE_PATH = Path("examples/sample.inline.jsonl")
COMMAND_FAILURE_EXIT_CODE = 1
INTERRUPTED_EXIT_CODE = 130


logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _exit_operation_failure(operation: str, error: Exception) -> NoReturn:
    """Report an operational failure and terminate with the documented status 1.

    Raises:
        SystemExit: Always, with the documented failure status, chained from the original
            error. This never returns -- it is the single exit point for an operational
            failure, so every command reports one status for the same class of problem instead
            of each choosing its own.

    """
    logger.error("%s failed: %s", operation, error)
    raise SystemExit(COMMAND_FAILURE_EXIT_CODE) from error


def _run_operation(operation: str, action: Callable[[], object]) -> None:
    try:
        action()
    except KeyboardInterrupt:
        logger.warning("%s interrupted by user", operation)
        raise SystemExit(INTERRUPTED_EXIT_CODE) from None
    # reason: this is the CLI's single operation boundary and `action` is an arbitrary caller-supplied
    # reason: callable, so the breadth IS the contract: any failure becomes a clean exit, never a traceback.
    except Exception as error:  # ruff: ignore[blind-except]
        _exit_operation_failure(operation, error)


def run_generate(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    from meddies_pii.generation.label_corpus.runner import (
        LanguageGenerationPlan,
        SyntheticGenerationRequest,
        run_language_generation,
    )

    load_dotenv()

    scenarios = tuple(s.strip() for s in args.scenarios.split(",") if s.strip()) if args.scenarios else None
    plan = LanguageGenerationPlan(
        request_template=SyntheticGenerationRequest(
            provider=args.provider,
            model=args.model,
            language=args.language,
            target_count=args.count,
            output_dir=args.output_dir,
            domain_profile=args.domain,
            split_purpose=args.split,
            scenario_names=scenarios,
            count_mode=args.count_mode,
            max_concurrency=args.concurrency,
            rpm_per_key=args.rpm,
        ),
        supported_languages=tuple(SUPPORTED_LANGUAGES),
    )

    logger.info(
        "Generating %s docs per language for %s (domain=%s) -> %s",
        args.count,
        ", ".join(plan.languages),
        args.domain,
        args.output_dir,
    )
    try:
        result = asyncio.run(run_language_generation(plan))
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        raise SystemExit(INTERRUPTED_EXIT_CODE) from None

    for success in result.successes:
        logger.info(
            "Accepted %s/%s for %s",
            success.accepted_count,
            args.count,
            success.language,
        )
    for failure in result.failures:
        if failure.kind == "configuration":
            logger.error("Configuration error: %s", failure.message)
        elif failure.kind == "generation":
            logger.error("Generation error for %s: %s", failure.language, failure.message)
        else:
            logger.error(
                "Unexpected error during generation for %s: %s",
                failure.language,
                failure.message,
            )

    if result.failures:
        successful_languages = [item.language for item in result.successes]
        failed_languages = [item.language for item in result.failures]
        successful_text = ", ".join(successful_languages) or "none"
        failed_text = ", ".join(failed_languages)
        logger.error(
            "Generation failed: %s succeeded (%s); %s failed (%s). "
            "Generated artifacts, including partial failed-language progress, were retained for retry.",
            len(successful_languages),
            successful_text,
            len(failed_languages),
            failed_text,
        )
        raise SystemExit(COMMAND_FAILURE_EXIT_CODE)


def run_sample(args: argparse.Namespace) -> None:
    sample_and_check(args.file, args.count)


def run_inference(args: argparse.Namespace) -> None:
    no_client_actions: dict[str, Callable[[], object]] = {
        "inf-prepare": lambda: prepare_jsonl(args.input, args.output, args.column, args.limit),
        "inf-prepare-hf": lambda: prepare_hf_review(
            args.config,
            args.output,
            args.repo_id,
            args.raw_col,
            args.label_col,
            allow_external_provider=args.allow_external_provider,
            data_classification=args.data_classification,
        ),
    }

    no_client_action = no_client_actions.get(args.command)
    if no_client_action is not None:
        _run_operation(f"Inference {args.command}", no_client_action)
        return

    use_vertex = args.vertex_ai
    if hasattr(args, "job_name") and args.job_name and args.job_name.startswith("projects/") and not use_vertex:
        use_vertex = True

    try:
        client = get_client(use_vertex_ai=use_vertex)
    # reason: provider client construction fails through the vendor SDK with auth, transport and config
    # reason: errors this module cannot enumerate; a CLI must turn every one into a clean exit.
    except Exception as error:  # ruff: ignore[blind-except]
        _exit_operation_failure(f"Inference {args.command}", error)

    client_actions: dict[str, Callable[[], object]] = {
        "inf-submit": lambda: submit_batch_job(
            client,
            args.src,
            args.model,
            is_vertex=use_vertex,
            allow_external_provider=args.allow_external_provider,
            data_classification=args.data_classification,
        ),
        "inf-monitor": lambda: monitor_job(client, args.job_name),
        "inf-list": lambda: list_jobs(client, args.limit),
        "inf-download": lambda: _handle_download(client, args),
        "inf-cancel": lambda: cancel_job(client, args.job_name),
        "inf-delete": lambda: delete_job(client, args.job_name),
    }

    client_action = client_actions.get(args.command)
    if client_action is not None:
        _run_operation(f"Inference {args.command}", client_action)


def _handle_download(client: GeminiClient, args: argparse.Namespace) -> None:
    results = download_batch_results(client, args.job_name)
    if results:
        if args.merge_input:
            merge_results(results, args.merge_input)
        else:
            save_results_to_json(results, args.job_name)


def run_correct(args: argparse.Namespace) -> None:
    from meddies_pii.generation.correction import CorrectionRequest, correct_hf_data

    try:
        request = CorrectionRequest(
            repo=args.repo,
            limit=args.limit,
            output=args.output,
        )
        asyncio.run(correct_hf_data(request))
    except KeyboardInterrupt:
        logger.warning("Correction interrupted by user")
        raise SystemExit(INTERRUPTED_EXIT_CODE) from None
    # reason: the correction run spans network, dataset and model code paths, so the breadth IS the
    # reason: contract at this CLI boundary: any failure becomes a clean exit, never a traceback.
    except Exception as error:  # ruff: ignore[blind-except]
        _exit_operation_failure("Correction", error)


# reason: every subcommand handler takes the parsed namespace so the dispatch chain at :444 can call
# reason: them uniformly; this one needs nothing out of it, and renaming only this member would
# reason: misdescribe the family's shared shape.
def run_labels(args: argparse.Namespace) -> None:  # ruff: ignore[unused-function-argument]
    print(f"Meddies Labels ({len(PII_LABELS)}):")
    for label in PII_LABELS:
        print(f"- {label}")


def run_validate(args: argparse.Namespace) -> None:
    result = validate_inline_jsonl(args.path)
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    _print_validation_summary(result)


def run_demo(args: argparse.Namespace) -> None:
    result = validate_inline_jsonl(args.sample)
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    demo = write_demo_output(result, args.output_dir)
    _print_validation_summary(result)
    print(f"Wrote demo output: {demo.output_dir}")
    print(f"- {demo.plain_path}")
    print(f"- {demo.spans_path}")
    print(f"- {demo.report_path}")


def run_pool_status(args: argparse.Namespace) -> None:
    from meddies_pii.generation.pool_monitor import (
        format_pool_status,
        load_usage_record,
    )

    record = load_usage_record(args.usage_dir, args.date)
    if record is None:
        where = f" for {args.date}" if args.date else ""
        print(
            f"No usage record found{where} in {args.usage_dir}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    text, exit_code = format_pool_status(record)
    print(text)
    raise SystemExit(exit_code)


def _print_validation_summary(result: ValidationResult) -> None:
    print(f"OK: {len(result.records)} records, {result.span_count} spans, {len(result.labels)} labels")
    print("Labels: " + ", ".join(result.labels))


# reason: cli main keeps parse args beside run labels; splitting would fragment diagnostics.
def main() -> None:  # ruff: ignore[too-many-locals,too-many-statements]
    setup_logging()
    parser = argparse.ArgumentParser(description="Meddies PII Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("labels", help="Print the current Meddies Labels")
    validate_parser = subparsers.add_parser("validate", help="Validate local inline-tagged JSONL")
    validate_parser.add_argument("path", help="Path to inline-tagged JSONL")

    demo_parser = subparsers.add_parser("demo", help="Run the offline bundled demo")
    demo_parser.add_argument(
        "--sample",
        default=DEFAULT_SAMPLE_PATH,
        help=f"Inline JSONL sample path (default: {DEFAULT_SAMPLE_PATH})",
    )
    demo_parser.add_argument(
        "--output-dir",
        default="demo-output",
        help="Local output directory for demo artifacts",
    )

    gen_parser = subparsers.add_parser("generate", help="Generate synthetic PII data")
    gen_parser.add_argument("--count", type=int, default=1, help="Target accepted documents per language")
    gen_parser.add_argument(
        "--count-mode",
        choices=["total", "additional"],
        default="total",
        help="'total': stop at --count accepted; 'additional': add --count more",
    )
    gen_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory (writes accepted.<lang>.jsonl, rejected.*, summary.*)",
    )
    gen_parser.add_argument("--domain", choices=DOMAIN_PROFILES, default="medical", help="Domain profile")
    gen_parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario names (default: all enabled for the domain)",
    )
    gen_parser.add_argument("--split", choices=SPLIT_PURPOSES, default="train", help="Split purpose")
    gen_parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent API calls")
    gen_parser.add_argument("--rpm", type=int, default=60, help="Max requests per minute per key")
    gen_parser.add_argument(
        "--provider",
        type=str,
        default="mimo",
        help="OpenAI-compatible generation provider",
    )
    gen_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (provider default if omitted)",
    )
    gen_parser.add_argument(
        "--language",
        type=str,
        choices=[*list(SUPPORTED_LANGUAGES), "all"],
        default="Vietnamese",
        help="Target language (use 'all' for every supported language)",
    )

    sample_parser = subparsers.add_parser("sample", help="Sample and check JSONL file quality")
    sample_parser.add_argument("file", help="Path to the JSONL file")
    sample_parser.add_argument("--count", type=int, default=5000, help="Number of samples")

    inf_parser = subparsers.add_parser("inference", help="Batch generation (tagging) using Gemini")
    inf_subparsers = inf_parser.add_subparsers(dest="inf_command", required=True)

    inf_parser.add_argument("--vertex-ai", action="store_true", help="Use Vertex AI")

    prepare_p = inf_subparsers.add_parser("prepare", help="Prepare JSONL for batch")
    prepare_p.add_argument("--input", required=True)
    prepare_p.add_argument("--output", default="batch_requests.jsonl")
    prepare_p.add_argument("--column", default="")
    prepare_p.add_argument("--limit", type=int, default=0)

    prepare_hf_p = inf_subparsers.add_parser(
        "prepare-hf",
        help="Prepare batch JSONL from HuggingFace dataset for label review",
    )
    prepare_hf_p.add_argument("--config", required=True, help="HF dataset config name")
    prepare_hf_p.add_argument("--output", required=True, help="Output JSONL path")
    prepare_hf_p.add_argument("--repo-id", default="Meddies/meddies-pii")
    prepare_hf_p.add_argument("--raw-col", default="raw")
    prepare_hf_p.add_argument("--label-col", default="label")
    prepare_hf_p.add_argument(
        "--allow-external-provider",
        action="store_true",
        help="Allow writing this dataset into Gemini batch-review requests",
    )
    prepare_hf_p.add_argument(
        "--data-classification",
        help="Data classification for external-provider review",
    )

    submit_p = inf_subparsers.add_parser("submit", help="Submit batch job")
    submit_p.add_argument("--src", default="batch_requests.jsonl")
    submit_p.add_argument("--model", default="gemini-3.1-pro-preview")
    submit_p.add_argument(
        "--allow-external-provider",
        action="store_true",
        help="Allow submitting this batch file to Gemini",
    )
    submit_p.add_argument(
        "--data-classification",
        help="Data classification for external-provider submission",
    )

    monitor_p = inf_subparsers.add_parser("monitor", help="Monitor job")
    monitor_p.add_argument("--job-name", required=True)

    list_p = inf_subparsers.add_parser("list", help="List jobs")
    list_p.add_argument("--limit", type=int, default=10)

    download_p = inf_subparsers.add_parser("download", help="Download results")
    download_p.add_argument("--job-name", required=True)
    download_p.add_argument("--merge-input")

    cancel_p = inf_subparsers.add_parser("cancel", help="Cancel job")
    cancel_p.add_argument("--job-name", required=True)

    delete_p = inf_subparsers.add_parser("delete", help="Delete job")
    delete_p.add_argument("--job-name", required=True)

    correct_p = subparsers.add_parser("correct", help="Correct HuggingFace dataset with OpenAI-compatible provider")
    correct_p.add_argument("--repo", default="Meddies/vie-pii", help="HuggingFace repository name")
    correct_p.add_argument("--limit", type=int, default=100, help="Number of items to process")
    correct_p.add_argument("--output", default="corrected_data.jsonl", help="Output JSONL file")

    pool_status_p = subparsers.add_parser(
        "pool-status",
        help="Verify/monitor per-account utilization vs real caps (read-only)",
    )
    pool_status_p.add_argument(
        "--usage-dir",
        default="data/bioes-v2/synthetic/usage",
        help="Directory of <date>.json usage records",
    )
    pool_status_p.add_argument(
        "--date",
        default=None,
        help="Specific date YYYY-MM-DD (default: latest record)",
    )

    args = parser.parse_args()

    if args.command == "labels":
        run_labels(args)
    elif args.command == "validate":
        run_validate(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "sample":
        run_sample(args)
    elif args.command == "inference":
        args.command = f"inf-{args.inf_command}"
        run_inference(args)
    elif args.command == "correct":
        run_correct(args)
    elif args.command == "pool-status":
        run_pool_status(args)


if __name__ == "__main__":
    main()
