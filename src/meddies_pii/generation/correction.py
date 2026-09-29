"""Label correction over a HuggingFace dataset via OpenAICompatibleClient."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from meddies_pii.generation.datasets_adapter import load_streaming_train
from meddies_pii.generation.openai_compatible.client import OpenAICompatibleClient
from meddies_pii.generation.prompts import SYSTEM_PROMPT
from meddies_pii.json_types import JsonObject, as_json_object
from meddies_pii.jsonl import write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

logger = logging.getLogger(__name__)


MIN_SUBSTANTIVE_TEXT_LENGTH = 100


@dataclass(frozen=True)
class CorrectionRequest:
    """Validated correction inputs accepted from the CLI boundary."""

    repo: str
    limit: int
    output: str

    def __post_init__(self) -> None:
        if not self.repo.strip():
            msg = "repo must not be blank"
            raise ValueError(msg)
        if self.limit <= 0:
            msg = "limit must be positive"
            raise ValueError(msg)
        if not self.output.strip():
            msg = "output must not be blank"
            raise ValueError(msg)


@dataclass(frozen=True)
class CorrectionOutcome:
    records: list[JsonObject]
    used_real_data: bool
    successful_count: int
    failed_count: int
    skipped_count: int


class DataCorrector:
    """Re-tag HuggingFace dataset rows via OpenAICompatibleClient."""

    def __init__(self) -> None:
        self._client = OpenAICompatibleClient.from_env()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    async def stream_and_correct(self, repo: str = "Meddies/vie-pii", limit: int = 100) -> CorrectionOutcome:
        logger.info("Processing %s items from %s", limit, repo)

        data_iterable: Iterable[object]
        try:
            data_iterable = load_streaming_train(repo)
            use_real_data = True
        except ImportError:
            logger.warning("datasets library not available, using sample data")
            data_iterable = [
                {
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": (
                                                "### Validation Report\n\n**Report Title:** "
                                                "Validation of Biotechnological Process for "
                                                "[BioNexa]<company_name>\n\n**Executive "
                                                "Summary:**\nThis validation report, "
                                                "prepared by "
                                                "employee id [MKT-3281]<id_number>, "
                                                "documents comprehensive validation study "
                                                "conducted on "
                                                "the biotechnological process at "
                                                "[BioNexa]<company_name>. The study was "
                                                "performed under "
                                                "the certificate license number "
                                                "[FL-NUR-871204]<id_number> and adheres to "
                                                "the standards "
                                                "outlined in the report accessible at "
                                                "[https://biotechjournals.com/reports?document_type=Validation+Report&"
                                                "year=2023&author=Dr.+Jane+Doe]url. "
                                                "The validation process was completed on "
                                                "[15/11/2023]<date>, ensuring all "
                                                "procedures were "
                                                "meticulously followed."
                                            ),
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                },
            ] * limit
            use_real_data = False

        corrected_data: list[JsonObject] = []
        processed_count = 0
        failed_count = 0
        skipped_count = 0

        for index, raw_item in enumerate(data_iterable):
            if index >= limit:
                break

            item = as_json_object(raw_item)
            if item is None:
                logger.warning("Non-JSON object at item %s, skipping", index)
                skipped_count += 1
                continue
            text = self.extract_text_from_response(item)
            if not text:
                logger.warning("No text found in item %s, skipping", index)
                skipped_count += 1
                continue

            logger.info("Processing item %s/%s", index + 1, limit)
            try:
                corrected_text = await self._client.chat(SYSTEM_PROMPT, text, temperature=0.7, max_tokens=8192)
                corrected_data.append({
                    "original": item,
                    "extracted_text": text,
                    "corrected_text": corrected_text,
                })
                processed_count += 1
                logger.info("Successfully corrected item %s", index + 1)
            except Exception as error:
                logger.exception("Failed to correct item %s", index + 1)
                failed_count += 1
                corrected_data.append({
                    "original": item,
                    "extracted_text": text,
                    "error": str(error),
                })

        logger.info(
            "Correction outcome: %s succeeded, %s failed, %s skipped",
            processed_count,
            failed_count,
            skipped_count,
        )
        return CorrectionOutcome(
            records=corrected_data,
            used_real_data=use_real_data,
            successful_count=processed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
        )

    @staticmethod
    def extract_text_from_response(item: JsonObject) -> str:
        """Extract text only after validating the provider-shaped JSON record.

        Returns:
            The response text. A plain string response is returned as-is; otherwise the
            provider's nested candidate/content/parts shape is walked, each level checked
            before it is indexed. An empty string means the record carried no text in any
            recognized position, which the caller treats as a failed item rather than content.

        """
        response = item.get("response")
        if isinstance(response, str):
            return response

        response_object = as_json_object(response)
        if response_object is not None:
            candidates = response_object.get("candidates")
            if isinstance(candidates, list) and candidates:
                candidate = as_json_object(candidates[0])
                content = candidate.get("content") if candidate is not None else None
                content_object = as_json_object(content)
                parts = content_object.get("parts") if content_object else None
                if isinstance(parts, list) and parts:
                    first_part = as_json_object(parts[0])
                    text = first_part.get("text") if first_part is not None else None
                    if isinstance(text, str):
                        return text
            response_text = response_object.get("text")
            if isinstance(response_text, str):
                return response_text

        for value in item.values():
            if isinstance(value, str) and len(value) > MIN_SUBSTANTIVE_TEXT_LENGTH:
                return value
        return ""

    # reason: part of DataCorrector's observed surface — `tests/test_public_cli.py:519` substitutes a
    # reason: stand-in declaring this same instance method, so the shape is what a substitute mirrors.
    def save_results(  # ruff: ignore[no-self-use]
        self,
        corrected_data: list[JsonObject],
        output_path: str = "corrected_data.jsonl",
    ) -> None:
        logger.info("Saving %s items to %s", len(corrected_data), output_path)
        write_jsonl(output_path, corrected_data)


async def correct_hf_data(request: CorrectionRequest) -> None:
    async with DataCorrector() as corrector:
        outcome = await corrector.stream_and_correct(repo=request.repo, limit=request.limit)
        corrector.save_results(outcome.records, request.output)

    if outcome.successful_count == 0:
        logger.error(
            "Correction completed with no successful rows: %s succeeded, %s failed, %s skipped. Diagnostics saved to %s",
            outcome.successful_count,
            outcome.failed_count,
            outcome.skipped_count,
            request.output,
        )
        msg = "Correction produced no successful rows"
        raise RuntimeError(msg)

    logger.info(
        "Correction completed: %s succeeded, %s failed, %s skipped",
        outcome.successful_count,
        outcome.failed_count,
        outcome.skipped_count,
    )
    if not outcome.used_real_data:
        logger.warning("Used sample data (install 'datasets' library for real HuggingFace data)")
    logger.info("Results saved to %s", request.output)
