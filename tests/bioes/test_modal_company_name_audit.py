from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[no-self-use]
# reason: stateless test doubles retain the bound method shape of the executor interfaces they replace.
import json
from hashlib import sha256
from typing import TYPE_CHECKING, Self

import pytest

from meddies_pii.training.bioes.modal import company_name_audit as modal_audit
from meddies_pii.training.bioes.trainers import company_name_audit as audit

if TYPE_CHECKING:
    from pathlib import Path


def test_cpu_modal_audit_uses_exact_pins_and_no_gpu() -> None:
    assert "gpu" not in modal_audit.CPU_OPTIONS
    assert modal_audit.CPU_OPTIONS["cpu"] == 12.0
    assert modal_audit.CPU_OPTIONS["memory"] == (32 * 1024, 64 * 1024)
    assert modal_audit.FULL_AUDIT_WORKERS == 12
    assert modal_audit.PACKED_DATASET_REVISION == audit.PACKED_DATASET_REVISION
    assert modal_audit.EVAL_DATASET_REVISION == audit.EVAL_DATASET_REVISION
    assert modal_audit.STOP_COMMAND == ("MODAL_PROFILE=meddies-pii uv run modal app stop meddies-bioes-company-name-audit")


def test_remote_guard_requires_confirmation() -> None:
    with pytest.raises(RuntimeError, match="confirmation"):
        modal_audit._remote_guard("")
    modal_audit._remote_guard(audit.COMPANY_NAME_AUDIT_CONFIRMATION)


def test_progress_writer_prints_and_persists_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    writer = modal_audit.ArtifactWriter(tmp_path, commit=None)
    writer.event({"event": "shard_complete", "shard_index": 1})
    assert json.loads(capsys.readouterr().out)["event"] == "shard_complete"
    assert json.loads((tmp_path / "events.jsonl").read_text())["shard_index"] == 1


def test_render_path_is_pure_and_does_not_start_modal_work() -> None:
    rendered = modal_audit.render_dry_run("meddies-pii")
    assert rendered == audit.render_modal_command("meddies-pii")


def test_streaming_audit_uses_12_workers_and_keeps_prefix_ordered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    label2id, id2label = audit.authoritative_label_maps()
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    paths = {"packed/data/first.parquet": first, "packed/data/second.parquet": second}
    shards = tuple(
        {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    )
    rows = {
        str(first): [
            {
                "labels": [label2id["O"]],
                "row_ranges": [{"uid": "a", "start": 0, "end": 1}],
            },
        ],
        str(second): [
            {
                "labels": [label2id["S-company_name"]],
                "row_ranges": [{"uid": "b", "start": 0, "end": 1}],
            },
        ],
    }
    worker_counts: list[int] = []

    class ImmediateFuture:
        def __init__(self, value: audit.PackedCompanyNameAudit) -> None:
            self.value = value

        def result(self) -> audit.PackedCompanyNameAudit:
            return self.value

    class ImmediateExecutor:
        def __init__(self, *, max_workers: int) -> None:
            worker_counts.append(max_workers)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, _callback: object, path: str) -> ImmediateFuture:
            return ImmediateFuture(audit.audit_packed_units(rows[path], label2id=label2id, id2label=id2label))

    monkeypatch.setattr(modal_audit, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(
        modal_audit,
        "wait",
        lambda pending, **_kwargs: (set(pending), set()),
    )
    events: list[dict[str, object]] = []
    prefix, full = modal_audit._stream_packed_audit(
        shards,
        download=lambda shard: paths[str(shard["path"])],
        row_iter=lambda path: iter(rows[path]),
        emit=lambda event: events.append(dict(event)),
        prefix_units=1,
        full_units=2,
    )
    assert prefix.company_name_span_count == 0
    assert full.company_name_span_count == 1
    assert worker_counts == [12]
    assert [event["event"] for event in events] == [
        "prefix_shard_complete",
        "shard_complete",
        "shard_complete",
    ]
    assert events[0]["shard_index"] == 1
    assert events[0]["prefix_packed_units_processed"] == 1
    assert events[0]["stop_command"] == modal_audit.STOP_COMMAND
