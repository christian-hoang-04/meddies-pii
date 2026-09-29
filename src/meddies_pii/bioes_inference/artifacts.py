"""Per-file hydration is deliberate.

Snapshot metadata is not part of the runtime contract, and hf_hub_download cannot fetch undeclared repo files.

"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the redaction runtime is an optional extra, and several backends are resolved by name at call time.
import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol

from .file_identity import file_sha256

if TYPE_CHECKING:
    from .detector import BioesSpanDetector

FileRole = Literal["model", "tokenizer"]
HydrationFailureCode = Literal[
    "cache_unavailable",
    "dependency_missing",
    "download_failed",
    "missing_file",
    "path_escape",
    "size_mismatch",
    "digest_mismatch",
    "file_unreadable",
]

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class HubFileDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class PinnedFileSpec:
    relative_path: str
    size_bytes: int
    sha256: str
    role: FileRole

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or "\\" in self.relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != self.relative_path
        ):
            msg = "relative_path must be a normalized relative repository path"
            raise ValueError(msg)
        if self.size_bytes <= 0:
            msg = "size_bytes must be positive"
            raise ValueError(msg)
        if _SHA256_RE.fullmatch(self.sha256) is None:
            msg = "sha256 must be a lowercase SHA-256 digest"
            raise ValueError(msg)
        if self.role not in {"model", "tokenizer"}:
            msg = "role must be model or tokenizer"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PinnedArtifactSpec:
    artifact_id: str
    repo_id: str
    revision: str
    files: tuple[PinnedFileSpec, ...]

    def __post_init__(self) -> None:
        if _SAFE_ID_RE.fullmatch(self.artifact_id) is None:
            msg = "artifact_id must be a safe identifier"
            raise ValueError(msg)
        if _REPO_ID_RE.fullmatch(self.repo_id) is None:
            msg = "repo_id must be an owner/name identifier"
            raise ValueError(msg)
        if _COMMIT_RE.fullmatch(self.revision) is None:
            msg = "revision must be a 40-character lowercase commit SHA"
            raise ValueError(msg)
        if not self.files:
            msg = "artifact spec must declare at least one file"
            raise ValueError(msg)
        paths = tuple(file.relative_path for file in self.files)
        if len(set(paths)) != len(paths):
            msg = "artifact file paths must be unique"
            raise ValueError(msg)


PUBLIC_Q8_ARTIFACT_SPEC = PinnedArtifactSpec(
    artifact_id="meddies_pii_v2_q8",
    repo_id="Meddies/meddies-pii-v2-onnx",
    revision="47b041b16ec3ecdf75871d42e38951f7f6d65a31",
    files=(
        PinnedFileSpec(
            relative_path="onnx/model.q8.onnx",
            size_bytes=592_471_465,
            sha256=("896b7cc8a9621c9b11e54a7893071091acf05c0055a2e7df7e35220587c73fe3"),
            role="model",
        ),
        PinnedFileSpec(
            relative_path="tokenizer.json",
            size_bytes=4_733_016,
            sha256=("4905ab82b2cfc25e0c88adc8f4eeffe759c57c5626312b30b0aaeaf8ad3379bc"),
            role="tokenizer",
        ),
        PinnedFileSpec(
            relative_path="tokenizer_config.json",
            size_bytes=526,
            sha256=("1c02b0dd850ea012fa8824ae9facdf4cfb367463d3b33134e567b9f434cc9241"),
            role="tokenizer",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class HydratedFileIdentity:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str
    role: FileRole


@dataclass(frozen=True, slots=True)
class HydratedArtifactIdentity:
    artifact_id: str
    repo_id: str
    revision: str
    root: Path
    tokenizer_directory: Path
    files: tuple[HydratedFileIdentity, ...]

    @property
    def model_path(self) -> Path:
        model_files = tuple(file.path for file in self.files if file.role == "model")
        if len(model_files) != 1:
            msg = "hydrated artifact must have exactly one model file"
            raise RuntimeError(msg)
        return model_files[0]

    @property
    def tokenizer_root(self) -> Path:
        if not any(file.role == "tokenizer" for file in self.files):
            msg = "hydrated artifact has no tokenizer files"
            raise RuntimeError(msg)
        return self.tokenizer_directory


class ArtifactHydrationError(RuntimeError):
    """Supply-chain failure carrying only identifiers, numbers, and digests."""

    # reason: Artifact code, expected/observed size and hash, and cause digest are the error contract.
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        code: HydrationFailureCode,
        spec: PinnedArtifactSpec,
        relative_path: str | None = None,
        expected_size: int | None = None,
        observed_size: int | None = None,
        expected_sha256: str | None = None,
        observed_sha256: str | None = None,
        cause_type: str | None = None,
        cause_digest: str | None = None,
    ) -> None:
        self.code = code
        self.artifact_id = spec.artifact_id
        self.repo_id = spec.repo_id
        self.revision = spec.revision
        self.relative_path = relative_path
        self.expected_size = expected_size
        self.observed_size = observed_size
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256
        self.cause_type = cause_type
        self.cause_digest = cause_digest
        super().__init__(self._safe_message())

    def _safe_message(self) -> str:
        fields = (
            ("code", self.code),
            ("artifact", self.artifact_id),
            ("repo", self.repo_id),
            ("revision", self.revision),
            ("file", self.relative_path),
            ("expected_size", self.expected_size),
            ("observed_size", self.observed_size),
            ("expected_sha", _prefix(self.expected_sha256)),
            ("observed_sha", _prefix(self.observed_sha256)),
            ("cause_type", self.cause_type),
            ("cause_digest", _prefix(self.cause_digest)),
        )
        return "artifact_hydration_failed " + " ".join(f"{key}={value}" for key, value in fields if value is not None)


def hydrate_artifact(
    spec: PinnedArtifactSpec,
    root: Path,
    *,
    downloader: HubFileDownloader | None = None,
) -> HydratedArtifactIdentity:
    """Download and verify a pinned file set without importing any model runtime.

    Returns:
        The hydrated identity: artifact id, repo id, revision, root, tokenizer directory and the
        verified per-file identities. Every file is downloaded and byte-verified before it appears
        here. This function raises nothing of its own — refusals come from the helpers it delegates
        to (`_prepare_root`, `_download_declared_file`, `_verify_file`,
        `_build_clean_tokenizer_directory`) and propagate unchanged, so a returned identity means
        the whole pinned set passed.

    """
    artifact_root, download_root, tokenizer_root = _prepare_root(spec, root)
    download = downloader or _hf_hub_download
    hydrated: list[HydratedFileIdentity] = []
    for file_spec in spec.files:
        downloaded = _download_declared_file(spec, file_spec, download_root, downloader=download)
        hydrated.append(_verify_file(spec, file_spec, downloaded, download_root))
    clean_tokenizer_files = _build_clean_tokenizer_directory(
        spec,
        tokenizer_root,
        tuple(hydrated),
    )
    clean_tokenizer_by_path = {file.relative_path: file for file in clean_tokenizer_files}
    return HydratedArtifactIdentity(
        artifact_id=spec.artifact_id,
        repo_id=spec.repo_id,
        revision=spec.revision,
        root=artifact_root,
        tokenizer_directory=tokenizer_root,
        files=tuple(clean_tokenizer_by_path.get(file.relative_path, file) for file in hydrated),
    )


def create_default_detector(
    cache_dir: Path,
    *,
    threads: int = 6,
) -> BioesSpanDetector:
    """Hydrate the pinned public q8 artifact and compose one lazy CPU detector.

    Returns:
        A `BioesSpanDetector` bound to the hydrated q8 artifact. Composition is lazy: the model
        runtime is not imported here.

    Raises:
        ValueError: if `threads` is not a positive integer.
        RuntimeError: if the default artifact does not declare exactly one model file, which would
            make the detector's model path ambiguous.

    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        msg = "threads must be a positive integer"
        raise ValueError(msg)

    from .contracts import ExpectedFileIdentity
    from .detector import BioesSpanDetector, OpenVinoBackend

    artifact = hydrate_artifact(PUBLIC_Q8_ARTIFACT_SPEC, cache_dir)
    model_files = tuple(file for file in PUBLIC_Q8_ARTIFACT_SPEC.files if file.role == "model")
    if len(model_files) != 1:
        msg = "default artifact must declare exactly one model file"
        raise RuntimeError(msg)
    model_file = model_files[0]
    return BioesSpanDetector(
        model_path=artifact.model_path,
        tokenizer_path=artifact.tokenizer_root,
        expected_model_size_bytes=model_file.size_bytes,
        expected_model_sha256=model_file.sha256,
        expected_tokenizer_files={
            file.relative_path: ExpectedFileIdentity(
                size_bytes=file.size_bytes,
                sha256=file.sha256,
            )
            for file in PUBLIC_Q8_ARTIFACT_SPEC.files
            if file.role == "tokenizer"
        },
        backend=OpenVinoBackend(threads=threads),
    )


def _prepare_root(spec: PinnedArtifactSpec, root: Path) -> tuple[Path, Path, Path]:
    # reason: prepare root's try keeps resolve with wrapped; splitting would mix coordinate frames.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        root.mkdir(parents=True, exist_ok=True)
        cache_root = root.resolve(strict=True)
        artifact_candidate = cache_root / spec.artifact_id / spec.revision
        if not artifact_candidate.resolve(strict=False).is_relative_to(cache_root):
            # reason: this security refusal must bypass the OS-error wrapper while the same try owns path creation.
            raise ArtifactHydrationError(code="path_escape", spec=spec)  # ruff: ignore[raise-within-try]
        artifact_candidate.mkdir(parents=True, exist_ok=True)
        artifact_root = artifact_candidate.resolve(strict=True)
        download_candidate = artifact_root / "files"
        if not download_candidate.resolve(strict=False).is_relative_to(artifact_root):
            # reason: this security refusal must bypass the OS-error wrapper while the same try owns path creation.
            raise ArtifactHydrationError(code="path_escape", spec=spec)  # ruff: ignore[raise-within-try]
        download_candidate.mkdir(parents=True, exist_ok=True)
        download_root = download_candidate.resolve(strict=True)
    except ArtifactHydrationError:
        raise
    except (OSError, RuntimeError) as error:
        failure_code: HydrationFailureCode = "cache_unavailable"
        raise _wrapped_failure(failure_code, spec, error) from None
    return artifact_root, download_root, artifact_root / "tokenizer"


def _build_clean_tokenizer_directory(
    spec: PinnedArtifactSpec,
    tokenizer_root: Path,
    hydrated: tuple[HydratedFileIdentity, ...],
) -> tuple[HydratedFileIdentity, ...]:
    tokenizer_files = tuple(file for file in hydrated if file.role == "tokenizer")
    if not tokenizer_files:
        return ()

    try:
        staging_root = Path(tempfile.mkdtemp(prefix=".tokenizer-", dir=tokenizer_root.parent))
    except OSError as error:
        cache_failure_code: HydrationFailureCode = "cache_unavailable"
        raise _wrapped_failure(cache_failure_code, spec, error) from None

    file_specs = {file.relative_path: file for file in spec.files}
    clean_files: list[HydratedFileIdentity] = []
    # reason: build clean's try keeps verify file with wrapped; splitting would split cleanup from writes.
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        for hydrated_file in tokenizer_files:
            destination = staging_root / hydrated_file.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hydrated_file.path, destination)
            clean_files.append(
                _verify_file(
                    spec,
                    file_specs[hydrated_file.relative_path],
                    destination,
                    staging_root,
                ),
            )

        if tokenizer_root.is_symlink() or tokenizer_root.is_file():
            tokenizer_root.unlink()
        elif tokenizer_root.exists():
            shutil.rmtree(tokenizer_root)
        staging_root.replace(tokenizer_root)
    except ArtifactHydrationError:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    except OSError as error:
        shutil.rmtree(staging_root, ignore_errors=True)
        file_failure_code: HydrationFailureCode = "file_unreadable"
        raise _wrapped_failure(file_failure_code, spec, error) from None

    return tuple(
        HydratedFileIdentity(
            relative_path=file.relative_path,
            path=tokenizer_root / file.relative_path,
            size_bytes=file.size_bytes,
            sha256=file.sha256,
            role=file.role,
        )
        for file in clean_files
    )


def _download_declared_file(
    spec: PinnedArtifactSpec,
    file_spec: PinnedFileSpec,
    root: Path,
    *,
    downloader: HubFileDownloader,
) -> Path:
    try:
        downloaded = downloader(
            repo_id=spec.repo_id,
            filename=file_spec.relative_path,
            revision=spec.revision,
            local_dir=str(root),
        )
        return Path(downloaded)
    except ArtifactHydrationError:
        raise
    # reason: Hugging Face download clients expose transport, auth, cache, and dependency failures through many types.
    except Exception as error:  # ruff: ignore[blind-except]
        code: HydrationFailureCode = "download_failed"
        if isinstance(error, ModuleNotFoundError) and error.name == "huggingface_hub":
            code = "dependency_missing"
        raise _wrapped_failure(code, spec, error, relative_path=file_spec.relative_path) from None


def _verify_file(
    spec: PinnedArtifactSpec,
    file_spec: PinnedFileSpec,
    downloaded: Path,
    root: Path,
) -> HydratedFileIdentity:
    try:
        resolved = downloaded.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ArtifactHydrationError(
            code="missing_file",
            spec=spec,
            relative_path=file_spec.relative_path,
        ) from None
    if not resolved.is_relative_to(root):
        raise ArtifactHydrationError(
            code="path_escape",
            spec=spec,
            relative_path=file_spec.relative_path,
        )
    if not resolved.is_file():
        raise ArtifactHydrationError(
            code="missing_file",
            spec=spec,
            relative_path=file_spec.relative_path,
        )
    try:
        observed_size = resolved.stat().st_size
    except OSError:
        raise ArtifactHydrationError(
            code="file_unreadable",
            spec=spec,
            relative_path=file_spec.relative_path,
        ) from None
    if observed_size != file_spec.size_bytes:
        raise ArtifactHydrationError(
            code="size_mismatch",
            spec=spec,
            relative_path=file_spec.relative_path,
            expected_size=file_spec.size_bytes,
            observed_size=observed_size,
        )
    try:
        observed_sha256 = file_sha256(resolved)
    except OSError:
        raise ArtifactHydrationError(
            code="file_unreadable",
            spec=spec,
            relative_path=file_spec.relative_path,
        ) from None
    if observed_sha256 != file_spec.sha256:
        raise ArtifactHydrationError(
            code="digest_mismatch",
            spec=spec,
            relative_path=file_spec.relative_path,
            expected_sha256=file_spec.sha256,
            observed_sha256=observed_sha256,
        )
    return HydratedFileIdentity(
        relative_path=file_spec.relative_path,
        path=resolved,
        size_bytes=observed_size,
        sha256=observed_sha256,
        role=file_spec.role,
    )


def _hf_hub_download(*, repo_id: str, filename: str, revision: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
    )


def _wrapped_failure(
    code: HydrationFailureCode,
    spec: PinnedArtifactSpec,
    error: Exception,
    *,
    relative_path: str | None = None,
) -> ArtifactHydrationError:
    cause_type = type(error).__name__
    safe_cause_type = cause_type if cause_type.isidentifier() else "Exception"
    cause_digest = hashlib.sha256(f"{cause_type}:{error}".encode()).hexdigest()
    return ArtifactHydrationError(
        code=code,
        spec=spec,
        relative_path=relative_path,
        cause_type=safe_cause_type,
        cause_digest=cause_digest,
    )


def _prefix(value: str | None) -> str | None:
    return value[:12] if value is not None else None
