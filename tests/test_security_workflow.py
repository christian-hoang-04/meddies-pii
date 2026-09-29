from __future__ import annotations

import re
from pathlib import Path
from typing import cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODEQL_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/codeql.yml"
SECURITY_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/security.yml"
QUALITY_WORKFLOW = REPOSITORY_ROOT / ".github/workflows/quality.yml"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
CHECKOUT_ACTION_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_PYTHON_ACTION_SHA = "5fda3b95a4ea91299a34e894583c3862153e4b97"
SETUP_UV_ACTION_SHA = "c771a70e6277c0a99b617c7a806ffedaca235ff9"
UPLOAD_ARTIFACT_ACTION_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"


def _step(workflow: str, name: str) -> str:
    matches = re.findall(
        rf"^      - name: {re.escape(name)}\n(.*?)(?=^      - |\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(matches) != 1:
        msg = f"expected exactly one {name!r} step, found {len(matches)}"
        raise ValueError(msg)
    return cast("str", matches[0])


def _workflow(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_security_workflow_replaces_unauthorized_codeql() -> None:
    assert not CODEQL_WORKFLOW.exists()
    assert SECURITY_WORKFLOW.is_file()


def test_workflows_use_reviewed_pins_and_preserve_setup_uv_cache_pruning() -> None:
    for workflow_path in (QUALITY_WORKFLOW, SECURITY_WORKFLOW):
        workflow = _workflow(workflow_path)

        assert f"uses: actions/checkout@{CHECKOUT_ACTION_SHA} # v7.0.1" in workflow
        assert f"uses: actions/setup-python@{SETUP_PYTHON_ACTION_SHA} # v7.0.0" in workflow
        assert f"uses: astral-sh/setup-uv@{SETUP_UV_ACTION_SHA} # v9.0.0" in workflow
        setup_uv_step = _step(workflow, "Install uv")
        assert "enable-cache: true" in setup_uv_step
        assert "prune-cache: true" in setup_uv_step
        assert 'version: "0.12.0"' in setup_uv_step

    quality_workflow = _workflow(QUALITY_WORKFLOW)
    assert f"uses: actions/upload-artifact@{UPLOAD_ARTIFACT_ACTION_SHA} # v7.0.1" in _step(
        quality_workflow,
        "Upload coverage report",
    )


def test_security_workflow_has_least_privilege_schedule_and_pinned_actions() -> None:
    workflow = _workflow(SECURITY_WORKFLOW)

    assert "name: Security" in workflow
    assert "  push:\n    branches: [main]" in workflow
    assert "  pull_request:\n    branches: [main]" in workflow
    assert '  schedule:\n    - cron: "17 3 * * 1"' in workflow
    assert "permissions:\n  contents: read\n" in workflow
    assert "actions:" not in workflow.split("jobs:\n", maxsplit=1)[0]


def test_security_workflow_owns_dependency_audit_and_high_high_bandit_gate() -> None:
    workflow = _workflow(SECURITY_WORKFLOW)
    quality_workflow = _workflow(QUALITY_WORKFLOW)

    assert "uv audit --locked" in _step(workflow, "Audit locked dependencies")
    assert "uv sync --frozen --only-group security" in _step(workflow, "Install security dependencies")
    bandit_step = _step(workflow, "Run Bandit")
    assert bandit_step.strip() == (
        "run: uv run --no-sync bandit -r src/meddies_pii scripts --severity-level high --confidence-level high"
    )
    assert "uv audit --locked" not in quality_workflow
    for bypass in ("--exit-zero", "# nosec", "--skip", "--exclude", "baseline"):
        assert bypass not in workflow


def test_security_dependency_group_pins_bandit() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert re.search(r"(?ms)^security = \[\n\s*\"bandit==1\.9\.4\",\n\]", pyproject)
