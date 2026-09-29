from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the dependency is optional at runtime and deferred to the call that needs it.
import logging
from typing import Any

logger = logging.getLogger(__name__)


def push_config(
    repo_id: str,
    config_name: str,
    rows: list[dict[str, Any]],
    *,
    split: str = "train",
    private: bool | None = None,
) -> int:
    """Push rows as a named config/split to a multi-config HF dataset repo.

    datasets handles parquet conversion and dataset-card config registration.
    `private` is omitted unless set so a push to an existing repo never touches
    its visibility. Auth resolves through the huggingface_hub chain (HF_TOKEN env,
    else cached `hf auth login`); no dotenv side effect, so a stale .env HF_TOKEN
    can't shadow a valid cached login.

    Returns:
        The number of rows pushed. ``private`` is passed only when explicitly set, so a push to
        an existing repo leaves its visibility exactly as it was rather than resetting it to a
        default the caller never chose.

    """
    from datasets import Dataset

    extra: dict[str, Any] = {} if private is None else {"private": private}
    Dataset.from_list(rows).push_to_hub(repo_id, config_name=config_name, split=split, **extra)
    logger.info("Pushed %s rows to %s/%s (split=%s).", len(rows), repo_id, config_name, split)
    return len(rows)
