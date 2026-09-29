"""Shared Modal scaffolding for the BIOES training probes and full runs.

Every probe in this package declares the same four things: a source-mounted
image, the Hugging Face cache and artifact volumes, a JSONL event log on the
artifact volume, and an isolated child process that does the paid GPU work.
Only the packages, the GPU options and the child body differ per probe, so
those stay in the probe module and everything else lives here.
"""

from __future__ import annotations

# ruff: file-ignore[print]
# reason: this module runs inside a Modal container; its standard output is the operator's streamed run log.
# ruff: file-ignore[implicit-namespace-package]
# reason: `modal/` is the only subpackage of `bioes/` without an `__init__.py` — assembly, data, eval,
# reason: reports and trainers all have one — so the asymmetry reads as an oversight, and adding the
# reason: file is likely inert under hatchling's src-layout discovery.
# reason: Deliberately deferred rather than fixed here: this directory holds every spend-authorization
# reason: gate, and its Modal-remote import paths have only fake-mediated local coverage, so adding
# reason: `__init__.py` is a post-merge change whose proof is a real GPU smoke run — owner ledger item.
import json
import os

# reason: probe parents launch an allowlisted in-package module through sys.executable and list-form argv.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, Any, Protocol

import modal

from meddies_pii.modal_runtime import (
    MODAL_SOURCE_ROOT,
    add_source_pythonpath,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

CACHE_MOUNT = "/cache"
CACHE_ROOT = "/cache/hf"
ARTIFACT_MOUNT = "/artifacts"
CACHE_VOLUME_NAME = "huggingface-cache"
ARTIFACT_VOLUME_NAME = "meddies-pii-bioes-artifacts"
# reason: this is the public name of a Modal secret object; it contains no secret value.
HF_SECRET_NAME = "huggingface-secret"  # ruff: ignore[hardcoded-password-string]

VolumeMounts = dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]


class ChildProcess(Protocol):
    """The subset of ``Popen`` the streaming helper and its test fakes share."""

    stdout: IO[str] | None
    stderr: IO[str] | None

    def wait(self) -> int: ...


class TrainableModule(Protocol):
    """What the gradient and inventory helpers need of a tagger: its trainable parameter pairs."""

    def named_parameters(self) -> Iterable[tuple[str, Any]]: ...


def source_mounted_image(image: modal.Image) -> modal.Image:
    """Export PYTHONPATH before mounting, so the layer order stays cache-stable.

    Returns:
        The same image with PYTHONPATH set and ``src`` mounted, in that order. Mounting first
        would invalidate the cached layer every time the source tree changed.

    """
    return add_source_pythonpath(image).add_local_dir("src", remote_path=MODAL_SOURCE_ROOT)


def hf_cache_environment(
    *,
    root: str = CACHE_ROOT,
    hub_cache: str | None = None,
    offline: bool | None = True,
    disable_hub_telemetry: bool = False,
    disable_unsloth_statistics: bool = True,
) -> dict[str, str]:
    """Build one probe's Hugging Face cache contract.

    ``offline=None`` omits the offline flags entirely. A prewarm image must not
    inherit a training image's offline contract, and an absent flag differs from
    an explicit ``"0"`` for callers that assert the exact environment.

    Returns:
        The environment mapping: the four cache paths always, the three offline flags only
        when ``offline`` is not None, and each telemetry opt-out only when its flag is set.

    """
    hub = hub_cache if hub_cache is not None else root
    environment = {
        "HF_HOME": root,
        "HF_HUB_CACHE": hub,
        "HF_DATASETS_CACHE": f"{root}/datasets",
        "TRANSFORMERS_CACHE": hub,
    }
    if offline is not None:
        flag = "1" if offline else "0"
        environment["HF_HUB_OFFLINE"] = flag
        environment["HF_DATASETS_OFFLINE"] = flag
        environment["TRANSFORMERS_OFFLINE"] = flag
    if disable_hub_telemetry:
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if disable_unsloth_statistics:
        environment["UNSLOTH_DISABLE_STATISTICS"] = "1"
    return environment


def hf_cache_volume() -> modal.Volume:
    """Never create this one: a missing read cache means the prewarm never ran.

    Returns:
        The existing cache volume. Lookup is ``create_if_missing=False`` on purpose, so an
        absent volume fails here rather than silently giving a probe an empty cache to
        re-download into at GPU rates.

    """
    return modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=False)


def artifact_volume(name: str = ARTIFACT_VOLUME_NAME) -> modal.Volume:
    return modal.Volume.from_name(name, create_if_missing=True)


def hf_secret() -> modal.Secret:
    return modal.Secret.from_name(HF_SECRET_NAME)


def volume_mounts(cache: modal.Volume, artifacts: modal.Volume) -> VolumeMounts:
    return {CACHE_MOUNT: cache, ARTIFACT_MOUNT: artifacts}


def require_modal_profile(profile: str, subject: str) -> None:
    if os.environ.get("MODAL_PROFILE") != profile:
        msg = f"{subject} requires MODAL_PROFILE={profile}"
        raise RuntimeError(msg)


def append_event(
    root: Path,
    event: Mapping[str, Any],
    *,
    commit: Callable[[], None] | None = None,
    fsync: bool = False,
) -> None:
    """Append one JSONL record and echo it, so paid progress survives a crash."""
    root.mkdir(parents=True, exist_ok=True)
    record = json.dumps(dict(event), sort_keys=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(record + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    print(record, flush=True)
    if commit is not None:
        commit()


def stream_child_output(
    process: ChildProcess,
    *,
    stdout_sink: Callable[[str], None],
    stderr_sink: Callable[[str], None],
    capture_stdout: bool = False,
) -> tuple[int, str, str]:
    """Forward both child streams live, then return once both have drained.

    Reading the streams only after ``wait`` would withhold every paid GPU
    progress line until the child exits, and a full pipe buffer would deadlock.

    Returns:
        The child's exit status and its captured stdout and stderr. stdout is captured only
        when ``capture_stdout`` is set; otherwise that string is empty and the lines were
        forwarded to the sink alone.

    """
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def forward_stdout() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            if capture_stdout:
                stdout_lines.append(line)
            stdout_sink(line)

    def forward_stderr() -> None:
        if process.stderr is None:
            return
        for line in process.stderr:
            stderr_lines.append(line)
            stderr_sink(line)

    stdout_thread = threading.Thread(target=forward_stdout, daemon=True)
    stderr_thread = threading.Thread(target=forward_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join(timeout=5.0)
    stderr_thread.join(timeout=5.0)
    return returncode, "".join(stdout_lines), "".join(stderr_lines)


def print_stdout_sink(line: str) -> None:
    """Preserve the child's JSONL bytes so Modal's log boundary keeps each record."""
    print(line, end="", flush=True)


def print_stderr_sink(line: str) -> None:
    print(line, end="", file=sys.stderr, flush=True)


def write_child_spec(path: Path, spec: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(spec), sort_keys=True), encoding="utf-8")


def launch_child_process(module: str, spec_path: Path, result_path: Path) -> subprocess.Popen[str]:
    """Run the probe body in a fresh interpreter with line-buffered pipes.

    Returns:
        The started process, still running, with both pipes open in text mode. The caller
        owns draining it -- ``stream_child_output`` is what does that safely.

    """
    # reason: callers supply fixed in-package module names; sys.executable and all flags are fixed, with no shell.
    return subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            sys.executable,
            "-m",
            module,
            "--child-spec",
            str(spec_path),
            "--child-result",
            str(result_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )


def child_argv_paths(argv: Sequence[str]) -> tuple[str, str] | None:
    """Return the child spec/result paths when the module was re-entered as a child.

    Returns:
        The ``(spec, result)`` paths, or None when neither flag is present, which is how a
        module distinguishes being run as the Modal parent from being re-entered as its own
        subprocess.

    """
    if "--child-spec" not in argv and "--child-result" not in argv:
        return None
    spec_index = argv.index("--child-spec")
    result_index = argv.index("--child-result")
    return argv[spec_index + 1], argv[result_index + 1]
