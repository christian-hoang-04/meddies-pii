"""Characterization tests for the OpenAI-compatible provider catalog."""

from __future__ import annotations

import os
from unittest.mock import patch

from meddies_pii.generation.label_corpus.provider_access import (
    provider_keys as preflight_provider_keys,
)
from meddies_pii.generation.openai_compatible.providers import (
    get_provider_spec,
    provider_keys,
    resolve_provider_base_urls,
    resolve_provider_model,
)


def test_provider_resolution_keeps_alias_and_environment_precedence() -> None:
    with patch.dict(
        os.environ,
        {
            "OPENCODE_ZEN_BASE_URLS": "https://one.example/v1,https://two.example/v1",
            "OPENCODE_ZEN_MODEL": "environment-model",
        },
        clear=True,
    ):
        spec = get_provider_spec("zen")

        assert spec.name == "OpenCode Zen"
        assert resolve_provider_base_urls("zen", 2) == [
            "https://one.example/v1",
            "https://two.example/v1",
        ]
        assert resolve_provider_model("zen") == "environment-model"
        assert resolve_provider_model("zen", "explicit-model") == "explicit-model"


def test_provider_key_precedence_is_shared_with_preflight() -> None:
    with patch.dict(
        os.environ,
        {
            "MIMO_API_KEYS": "  ",
            "MIMO_API_KEY": " first-key, , second-key ",
        },
        clear=True,
    ):
        assert provider_keys("mimo") == ["first-key", "second-key"]
        assert preflight_provider_keys("mimo") == ["first-key", "second-key"]
