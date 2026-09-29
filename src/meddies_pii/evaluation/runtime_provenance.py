"""Immutable source closure identities for executable evaluation runtimes."""

from __future__ import annotations

from pathlib import Path

from meddies_pii.evaluation.identity import (
    ArtifactIdentity,
    file_sha256,
    payload_artifact,
    source_artifact,
)

_IGNORED_SOURCE_PATH_PARTS = frozenset({".git", "__pycache__"})


def source_tree_artifact(reference: str, root: str | Path) -> ArtifactIdentity:
    """Identify stable source-tree bytes while excluding generated Python cache files.

    Returns:
        An identity over the sorted list of every non-generated file's relative path, size and
        digest. ``.git`` and ``__pycache__`` are excluded because they are byte-unstable across
        machines that hold identical source, and including them would make the same tree hash
        differently on each host.

    Raises:
        ValueError: If the root is not a directory, or if every file under it was excluded. An
            empty file list would hash to a stable digest over nothing, which would then match
            any other empty tree and read as a successful pin.

    """
    root_path = Path(root)
    if not root_path.is_dir():
        msg = f"runtime source tree is missing: {root_path}"
        raise ValueError(msg)
    files = [
        {
            "path": path.relative_to(root_path).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(str(path)),
        }
        for path in sorted(candidate for candidate in root_path.rglob("*") if candidate.is_file())
        if not _is_generated_source_file(path, root_path)
    ]
    if not files:
        msg = f"runtime source tree has no files: {root_path}"
        raise ValueError(msg)
    return payload_artifact(reference, files)


def evaluation_runtime_source_artifact(
    package_root: str | Path,
    runner_source: str | Path,
) -> ArtifactIdentity:
    """Bind the mounted package and its runner to one evaluation contract artifact.

    Returns:
        One identity over both the package tree and the runner file. Binding them together
        means a result cannot be attributed to a package version while the runner that produced
        it has changed -- either side moving yields a different contract artifact.

    """
    package = source_tree_artifact("meddies-pii-runtime-package", package_root)
    runner = source_artifact("meddies-pii-runtime-runner", str(runner_source))
    return payload_artifact(
        "meddies-pii-evaluation-runtime-source",
        {
            "package": package.to_payload(),
            "runner": runner.to_payload(),
        },
    )


def _is_generated_source_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return path.suffix == ".pyc" or bool(_IGNORED_SOURCE_PATH_PARTS.intersection(relative.parts))
