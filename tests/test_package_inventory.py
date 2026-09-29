"""Tests for the release package artifact inventory gate."""

from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_gate() -> ModuleType:
    path = REPO_ROOT / "scripts/quality/check_package_inventory.py"
    spec = importlib.util.spec_from_file_location("package_inventory_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sdist(path: Path, members: tuple[str, ...] = ()) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            info = tarfile.TarInfo(member)
            info.size = 0
            archive.addfile(info)


def _write_wheel(path: Path, members: tuple[str, ...] = ()) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "")


def _release_artifacts(tmp_path: Path) -> Path:
    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    _write_sdist(artifacts / "meddies_pii-0.1.0.tar.gz", ("src/meddies_pii/__init__.py",))
    _write_wheel(artifacts / "meddies_pii-0.1.0-py3-none-any.whl", ("meddies_pii/__init__.py",))
    return artifacts


def test_accepts_one_readable_wheel_and_sdist_without_forbidden_repository_files(
    tmp_path: Path,
) -> None:
    gate = _load_gate()

    errors = gate.validate_package_inventory(_release_artifacts(tmp_path))

    assert errors == []


@pytest.mark.parametrize(
    ("artifact_name", "forbidden_member"),
    [
        (
            "meddies_pii-0.1.0.tar.gz",
            "meddies_pii-0.1.0/experiments/archive/old_experiment.py",
        ),
        (
            "meddies_pii-0.1.0-py3-none-any.whl",
            "experiments/archive/old_experiment.py",
        ),
        (
            "meddies_pii-0.1.0-py3-none-any.whl",
            "experiments/current_experiment.py",
        ),
        ("meddies_pii-0.1.0.tar.gz", "meddies_pii-0.1.0/coverage.json"),
        ("meddies_pii-0.1.0.tar.gz", "meddies_pii-0.1.0/coverage.xml"),
        ("meddies_pii-0.1.0-py3-none-any.whl", "coverage.json"),
        ("meddies_pii-0.1.0-py3-none-any.whl", "coverage.xml"),
    ],
)
def test_rejects_forbidden_members_in_either_release_artifact(
    tmp_path: Path,
    artifact_name: str,
    forbidden_member: str,
) -> None:
    gate = _load_gate()
    artifacts = _release_artifacts(tmp_path)
    artifact = artifacts / artifact_name
    if artifact.suffix == ".whl":
        _write_wheel(artifact, (forbidden_member,))
    else:
        _write_sdist(artifact, (forbidden_member,))

    errors = gate.validate_package_inventory(artifacts)

    assert errors == [f"{artifact.name} contains forbidden member: {forbidden_member}"]


@pytest.mark.parametrize(
    ("setup", "expected_error"),
    [
        (
            lambda artifacts: (artifacts / "meddies_pii-0.1.0-py3-none-any.whl").unlink(),
            "expected exactly one wheel (*.whl), found 0",
        ),
        (
            lambda artifacts: _write_wheel(artifacts / "another-0.1.0-py3-none-any.whl"),
            "expected exactly one wheel (*.whl), found 2",
        ),
        (
            lambda artifacts: (artifacts / "meddies_pii-0.1.0.tar.gz").unlink(),
            "expected exactly one source distribution (*.tar.gz), found 0",
        ),
    ],
)
def test_rejects_missing_or_ambiguous_release_artifacts(
    tmp_path: Path,
    setup: Callable[[Path], object],
    expected_error: str,
) -> None:
    gate = _load_gate()
    artifacts = _release_artifacts(tmp_path)
    setup(artifacts)

    errors = gate.validate_package_inventory(artifacts)

    assert errors == [expected_error]


def test_rejects_non_directory_artifact_path(tmp_path: Path) -> None:
    gate = _load_gate()
    artifact_path = tmp_path / "not-a-directory"
    artifact_path.write_text("release artifacts belong in a directory\n")

    errors = gate.validate_package_inventory(artifact_path)

    assert errors == [f"artifact path is not a directory: {artifact_path}"]


@pytest.mark.parametrize(
    ("artifact_name", "contents", "expected_error"),
    [
        (
            "meddies_pii-0.1.0.tar.gz",
            b"not a tarball",
            "cannot read source distribution meddies_pii-0.1.0.tar.gz",
        ),
        (
            "meddies_pii-0.1.0-py3-none-any.whl",
            b"not a wheel",
            "cannot read wheel meddies_pii-0.1.0-py3-none-any.whl",
        ),
    ],
)
def test_rejects_malformed_release_artifacts(
    tmp_path: Path,
    artifact_name: str,
    contents: bytes,
    expected_error: str,
) -> None:
    gate = _load_gate()
    artifacts = _release_artifacts(tmp_path)
    (artifacts / artifact_name).write_bytes(contents)

    errors = gate.validate_package_inventory(artifacts)

    assert len(errors) == 1
    assert errors[0].startswith(expected_error)


def test_cli_reports_missing_artifact_directory_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gate = _load_gate()
    missing = tmp_path / "missing-dist"

    exit_code = gate.main([str(missing)])

    assert exit_code == 1
    assert capsys.readouterr().err == (f"package inventory gate error: artifact directory does not exist: {missing}\n")
