from __future__ import annotations

import random

from meddies_pii.generation.label_corpus import catalog, generation_runs, runner
from meddies_pii.generation.label_corpus.edge_cases import sample_edge_cases


def test_edge_case_sampling_uses_catalog_scenarios_without_a_catalog_import_cycle() -> None:
    scenario = next(item for item in catalog.SCENARIOS if item.adversarial)

    # reason: seed 7 is known to draw a non-empty set for an adversarial scenario; an unseeded source
    # reason: would make this assertion probabilistic and let the test flake.
    assert sample_edge_cases(random.Random(7), scenario)  # ruff: ignore[suspicious-non-cryptographic-random-usage]


def test_generation_runs_builds_an_empty_ledger_before_requests() -> None:
    assert generation_runs.build_account_ledger().snapshot() == {}


def test_runner_returns_no_base_urls_without_accounts() -> None:
    assert runner.configured_base_urls("mimo", 0) == []


def test_summary_views_drop_rows_without_changing_the_result_shape() -> None:
    summary: dict[str, dict[str, object]] = {"english": {"accepted_count": 1, "rows": [{"sensitive": "row"}]}}

    assert generation_runs.strip_rows(summary) == {"english": {"accepted_count": 1}}
