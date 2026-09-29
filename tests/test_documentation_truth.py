from __future__ import annotations

import re
from pathlib import Path

import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_MODULES = (
    "baseline",
    "adapters",
    "regex_release",
    "opf_benchmark",
    "pii350_release",
)
OPF_TEST_PATHS = (
    "tests/eval_baseline/opf_benchmark/test_opf_benchmark.py",
    "tests/eval_baseline/adapters/test_opf_adapter.py",
)


def test_evaluation_architecture_describes_the_enforced_identity_contract() -> None:
    architecture = (REPOSITORY_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    identity_source = (REPOSITORY_ROOT / "src" / "meddies_pii" / "evaluation" / "identity.py").read_text(encoding="utf-8")
    required_identity_fields = {
        "model": "model artifact",
        "vendor_inference_source": "vendor inference source",
        "local_adapter_source": "local adapter source",
        "applied_label_prediction_contract": "applied-label prediction contract",
        "decoder_contract": "decoder contract",
        "resolved_runtime_environment": "resolved runtime environment",
        "scorer_contract": "scorer contract",
        "supported_labels": "supported labels",
        "result_schema": "result schema",
    }

    for field, documentation_term in required_identity_fields.items():
        assert f"{field}:" in identity_source
        assert documentation_term in architecture
    assert "dataset, shard, fixture rows, and row count" in architecture
    assert "persists the evaluation contract" in architecture
    assert "accepts a result only when its persisted identity matches" in architecture
    assert "do not record or verify the complete model" not in architecture


def test_type_gate_docs_match_the_configured_source_scope() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    mypy = project["tool"]["mypy"]
    basedpyright = project["tool"]["basedpyright"]

    assert mypy["strict"] is True
    assert "exclude" not in mypy
    assert basedpyright["exclude"] == [".venv"]

    expected_statement = (
        "Strict MyPy checks every module under `src/meddies_pii` with no MyPy exclusions. "
        "BasedPyright runs in standard mode across the same source and excludes only `.venv`."
    )
    for documentation_path in ("README.md", "CONTRIBUTING.md"):
        documentation = (REPOSITORY_ROOT / documentation_path).read_text(encoding="utf-8")
        assert expected_statement in documentation
        assert "Narrow optional-runtime seams are excluded" not in documentation
        assert "narrow exclusions" not in documentation


def test_contributing_lint_commands_match_the_enforced_strict_policy() -> None:
    contributing = (REPOSITORY_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    strict_command = "uv run ruff check --config ruff-strict.toml ."

    assert contributing.count(strict_command) == 2
    assert "uv run ruff check src scripts tests" not in contributing


def test_full_lintmax_gate_is_pinned_in_ci_and_contributor_pr_checklist() -> None:
    pinned_command = (
        "uvx --from 'git+https://github.com/hahuyhoang411/lintmax-py.git@"
        "4cac0c5b770b8f9cdb5a1d84186554ffbd0eb2a7' lintmax-py check ."
    )
    unpinned_command = "uvx --from lintmax-py lintmax-py check ."
    quality_workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    contributor_pr_checklist = (
        (REPOSITORY_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").split("## Pull requests", maxsplit=1)[1]
    )

    assert pinned_command in quality_workflow
    assert pinned_command in contributor_pr_checklist
    assert unpinned_command not in quality_workflow
    assert unpinned_command not in contributor_pr_checklist


def test_evaluation_documentation_paths_match_the_canonical_tree() -> None:
    evaluation_root = REPOSITORY_ROOT / "src" / "meddies_pii" / "eval_baseline"
    for module in EVALUATION_MODULES:
        assert (evaluation_root / module).is_dir()

    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (REPOSITORY_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "eval_baseline/{baseline,adapters,regex_release,opf_benchmark,pii350_release}" in architecture
    for module in EVALUATION_MODULES:
        assert f"    {module}/" in readme

    for path in OPF_TEST_PATHS:
        assert (REPOSITORY_ROOT / path).is_file()


def _collapsed(text: str) -> str:
    """Collapse runs of spaces so a table row matches whatever column padding the formatter chose.

    Returns:
        The text with every space run reduced to one space.

    """
    return re.sub(r" +", " ", text)


def test_verification_map_matches_the_split_quality_and_security_workflows() -> None:
    architecture = _collapsed((REPOSITORY_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8"))
    security_workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
    quality_workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")

    assert "uv audit --locked" in security_workflow
    assert "bandit -r src/meddies_pii scripts" in security_workflow
    assert "uv audit --locked" not in quality_workflow

    assert (
        "| Quality CI | `.github/workflows/quality.yml` on pull requests and pushes "
        "to `main` | The quality workflow runs the lock, lint, test, coverage, type, "
        "build, and package checks. |"
    ) in architecture
    assert (
        "| Security CI | `.github/workflows/security.yml` on pull requests, pushes "
        "to `main`, and weekly schedule | The security workflow owns `uv audit --locked` "
        "and runs Bandit against `src/meddies_pii` and `scripts/` at high severity "
        "and confidence. |"
    ) in architecture
    assert "CodeQL" not in architecture
