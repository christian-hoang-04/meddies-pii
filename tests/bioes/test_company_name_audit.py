from __future__ import annotations

# reason: this import-isolation test runs sys.executable with a literal script as list-form argv and no shell.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import textwrap

import pytest

from meddies_pii.training.bioes.trainers import company_name_audit as audit


def _maps() -> tuple[dict[str, int], dict[int, str]]:
    labels = audit.authoritative_label_vocabulary()
    label2id = {label: index for index, label in enumerate(labels)}
    return label2id, {index: label for label, index in label2id.items()}


def _unit(labels: list[int], *, with_source: bool = False) -> dict[str, object]:
    row_range: dict[str, object] = {"uid": "row-1", "start": 0, "end": len(labels)}
    if with_source:
        row_range.update({"source_uid": "source-1", "source_index": 7})
    return {"labels": labels, "row_ranges": [row_range], "row_uids": ["row-1"]}


def test_counter_reconstructs_bies_spans_not_tag_fragments() -> None:
    label2id, id2label = _maps()
    labels = [
        label2id["B-company_name"],
        label2id["I-company_name"],
        label2id["E-company_name"],
        label2id["S-company_name"],
        label2id["S-company_name"],
    ]
    report = audit.audit_packed_units([_unit(labels)], label2id=label2id, id2label=id2label, required_units=1)
    assert report.company_name_token_count == 5
    assert report.company_name_span_count == 3
    assert report.company_name_packed_unit_count == 1
    assert report.company_name_original_row_range_count == 1
    assert report.malformed_transition_count == 0
    assert report.source_membership_status == "unavailable"


def test_counter_records_malformed_transitions_and_rejects_short_prefix() -> None:
    label2id, id2label = _maps()
    malformed = [label2id["I-company_name"], label2id["E-company_name"]]
    report = audit.audit_packed_units([_unit(malformed)], label2id=label2id, id2label=id2label, required_units=1)
    assert report.company_name_span_count == 0
    assert report.malformed_transition_count == 2
    with pytest.raises(RuntimeError, match="prefix geometry"):
        audit.audit_packed_units([_unit(malformed)], label2id=label2id, id2label=id2label, required_units=2)


def test_counter_fails_closed_on_missing_or_inconsistent_company_label() -> None:
    label2id, id2label = _maps()
    missing = dict(label2id)
    del missing["S-company_name"]
    with pytest.raises(RuntimeError, match="label2id"):
        audit.audit_packed_units([], label2id=missing, id2label=id2label)
    inconsistent = dict(id2label)
    inconsistent[label2id["B-company_name"]] = "B-human_name"
    with pytest.raises(RuntimeError, match="id2label"):
        audit.audit_packed_units([], label2id=label2id, id2label=inconsistent)


def test_counter_does_not_infer_source_membership_from_original_row_identity() -> None:
    label2id, id2label = _maps()
    report = audit.audit_packed_units(
        [_unit([label2id["S-company_name"]], with_source=True)],
        label2id=label2id,
        id2label=id2label,
    )
    assert report.source_membership_status == "unavailable"
    assert "labels" in report.observed_schema_fields
    assert "source_uid" in report.observed_row_range_fields


def test_merge_preserves_exact_counts_without_retaining_packed_units() -> None:
    label2id, id2label = _maps()
    first = audit.audit_packed_units(
        [_unit([label2id["S-company_name"]])],
        label2id=label2id,
        id2label=id2label,
    )
    second = audit.audit_packed_units(
        [_unit([label2id["B-company_name"], label2id["E-company_name"]])],
        label2id=label2id,
        id2label=id2label,
    )
    merged = audit.merge_packed_company_name_audits([first, second])
    assert merged.processed_packed_units == 2
    assert merged.company_name_token_count == 3
    assert merged.company_name_span_count == 2
    assert merged.source_membership_status == "unavailable"


def test_eval_counter_uses_authoritative_record_parser_and_exact_inventory() -> None:
    records = [
        {
            "info": {"id": "eval-1"},
            "text": "Acme Corp called",
            "label": [{"category": "company_name", "start": 0, "end": 9}],
        },
        {"info": {"id": "eval-2"}, "text": "No entity", "label": []},
    ]
    report = audit.audit_eval_records(records, expected_rows=2)
    assert report.company_name_document_count == 1
    assert report.company_name_span_count == 1
    with pytest.raises(RuntimeError, match="inventory"):
        audit.audit_eval_records(records, expected_rows=1)


def test_rendered_command_and_execution_confirmation_are_fail_closed() -> None:
    command = audit.render_modal_command()
    assert "MODAL_PROFILE=meddies-pii" in command
    assert audit.COMPANY_NAME_AUDIT_CONFIRMATION in command
    with pytest.raises(RuntimeError, match="confirmation"):
        audit.require_execution("")
    audit.require_execution(audit.COMPANY_NAME_AUDIT_CONFIRMATION)


def test_audit_modules_import_without_torch_or_full_run_path() -> None:
    """Match the Modal CPU image: source code present, torch deliberately absent."""
    # reason: sys.executable and the inline source below are test literals passed as separate argv operands.
    process = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import builtins
                import sys

                original_import = builtins.__import__

                def block_torch(name, *args, **kwargs):
                    if name == "torch" or name.startswith("torch."):
                        raise ModuleNotFoundError("No module named 'torch'")
                    return original_import(name, *args, **kwargs)

                builtins.__import__ = block_torch
                import meddies_pii.training.bioes.trainers.company_name_audit
                import meddies_pii.training.bioes.modal.company_name_audit

                forbidden = (
                    "torch",
                    "meddies_pii.training.bioes.trainers.full_run",
                    "meddies_pii.training.bioes.trainers.base_selection",
                )
                assert not any(
                    module == prefix or module.startswith(prefix + ".")
                    for module in sys.modules
                    for prefix in forbidden
                )
                print("lightweight-import-ok")
                """,
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert process.stdout.strip() == "lightweight-import-ok"


def test_audit_pins_state_the_same_datasets_the_audit_functions_default_to() -> None:
    assert audit.AUDIT_PINS["packed"] == {
        "id": audit.PACKED_DATASET_ID,
        "revision": audit.PACKED_DATASET_REVISION,
        "expected_units": audit.PACKED_UNIT_COUNT,
    }
    assert audit.AUDIT_PINS["evaluation"]["id"] == audit.EVAL_DATASET_ID
    assert audit.AUDIT_PINS["evaluation"]["revision"] == audit.EVAL_DATASET_REVISION
    assert audit.AUDIT_PINS["evaluation"]["expected_rows"] == audit.EVAL_ROWS
