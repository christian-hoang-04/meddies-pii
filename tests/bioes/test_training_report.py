from __future__ import annotations

import json
from typing import TYPE_CHECKING

from meddies_pii.spans import CharSpan
from meddies_pii.training.bioes.report import (
    TrainingReportOptions,
    build_training_report,
    highlight_spans,
    scan_training_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _write_jsonl(path: Path, records: Sequence[object]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_highlight_spans_escapes_html_and_wraps_label() -> None:
    text = "Patient <Alice> & Bob"
    span = CharSpan(start=8, end=15, text="<Alice>", label="human_name")

    html = highlight_spans(text, [span])

    assert "&lt;Alice&gt;" in html
    assert "Patient " in html
    assert 'class="pii label-human-name"' in html
    assert "<Alice>" not in html


def test_scan_training_jsonl_can_include_all_rows(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    _write_jsonl(
        train_jsonl,
        [
            {
                "text": "Patient Alice uses https://portal.example/patients/1?token=abc.",
                "label": [
                    {
                        "category": "human_name",
                        "start": 8,
                        "end": 13,
                    },
                    {
                        "category": "private_url",
                        "start": 19,
                        "end": 62,
                    },
                ],
                "info": {"id": "r1"},
            },
            {
                "text": "Secret sk-live-123 belongs to Bob.",
                "label": [
                    {"category": "secret", "start": 7, "end": 18},
                    {"category": "human_name", "start": 30, "end": 33},
                ],
                "info": {"id": "r2"},
            },
        ],
    )

    scan = scan_training_jsonl(
        TrainingReportOptions(
            train_jsonl=train_jsonl,
            output_html=tmp_path / "report.html",
            include_all_rows=True,
            rows_per_label=4,
        ),
    )

    assert scan.rows_seen == 2
    assert len(scan.row_previews) == 2
    assert scan.label_counts["human_name"] == 2
    assert scan.label_counts["private_url"] == 1
    assert scan.label_counts["secret"] == 1


def test_build_training_report_renders_stack_real_examples_and_modal(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    summary_json = tmp_path / "summary.json"
    audit_jsonl = tmp_path / "audit.jsonl"
    adversarial_jsonl = tmp_path / "adversarial.jsonl"
    modal_json = tmp_path / "modal.json"
    todo_md = tmp_path / "todo.md"
    progress_md = tmp_path / "progress.md"
    output_html = tmp_path / "report.html"

    records = [
        {
            "text": "Patient Alice uses https://portal.example/patients/1?token=abc.",
            "label": [
                {"category": "human_name", "start": 8, "end": 13},
                {"category": "private_url", "start": 19, "end": 62},
            ],
            "info": {"id": "r1"},
        },
    ]
    _write_jsonl(train_jsonl, records)
    _write_jsonl(adversarial_jsonl, records)
    summary_json.write_text(
        json.dumps({
            "rows": 1,
            "kept_rows": 1,
            "quarantined_rows": 0,
            "label_counts": {
                "address": 0,
                "company_name": 0,
                "date": 0,
                "email_address": 0,
                "human_name": 1,
                "id_number": 0,
                "phone_number": 0,
                "private_url": 1,
                "secret": 0,
            },
            "decision_counts": {"keep:private_url_context": 1},
        }),
        encoding="utf-8",
    )
    _write_jsonl(
        audit_jsonl,
        [
            {
                "uid": "r1",
                "status": "kept",
                "decisions": [
                    {
                        "decision": "keep",
                        "rule": "private_url_context",
                        "old_label": "url",
                        "new_label": "private_url",
                        "text": "https://portal.example/patients/1?token=abc",
                    },
                ],
            },
        ],
    )
    modal_json.write_text(
        json.dumps({
            "provenance": {"modal_url": "https://modal.example/run"},
            "result": {
                "backend": "hf",
                "config": {
                    "model_id": "LiquidAI/LFM2.5-350M-Base",
                    "dataset_id": "Meddies/meddies-pii",
                    "max_length": 512,
                },
                "steps": 3,
                "train_examples": 2,
                "first_train_loss": 4.5,
                "final_train_loss": 3.9,
                "train_loss_delta": -0.6,
                "train_loss_values": [4.5, 3.9],
            },
        }),
        encoding="utf-8",
    )
    todo_md.write_text("upload bundle before full training", encoding="utf-8")
    progress_md.write_text("smoke passed", encoding="utf-8")

    path = build_training_report(
        TrainingReportOptions(
            train_jsonl=train_jsonl,
            summary_json=summary_json,
            audit_jsonl=audit_jsonl,
            adversarial_jsonl=adversarial_jsonl,
            modal_result_json=modal_json,
            todo_md=todo_md,
            progress_md=progress_md,
            output_html=output_html,
            include_all_rows=True,
            rows_per_label=2,
        ),
    )

    html = path.read_text(encoding="utf-8")
    assert "Training stack" in html
    assert "LiquidAI/LFM2.5-350M-Base" in html
    assert "fastino/gliner2-multi-v1" in html
    assert "Real training examples by label" in html
    assert "https://portal.example/patients/1?token=abc" in html
    assert "keep:private_url_context" in html
    assert "https://modal.example/run" in html
    assert "upload bundle before full training" in html


def test_training_report_reconstructs_openmedical_command_from_config(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    modal_json = tmp_path / "modal.json"
    output_html = tmp_path / "report.html"

    _write_jsonl(
        train_jsonl,
        [
            {
                "text": "Patient Alice uses portal token abc.",
                "label": [{"category": "human_name", "start": 8, "end": 13}],
                "info": {"id": "r1"},
            },
        ],
    )
    modal_json.write_text(
        json.dumps({
            "provenance": {
                "command_kwargs": {
                    "gpu": "H100",
                    "steps": 5,
                    "max_length": 4096,
                    "train_limit": 64,
                    "eval_limit": 8,
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "packing": True,
                    "train_config": "pii-bioes",
                    "eval_config": "pii-bioes",
                    "eval_dataset_split": "validation",
                },
                "out": "project-manager/preview/result.json",
                "raw_log": "project-manager/preview/raw_logs/result.log",
            },
            "result": {
                "config": {
                    "backend": "unsloth",
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "packing": True,
                },
                "packing_attention_probe": {
                    "passed": True,
                    "max_abs_diff": 0.0,
                    "compared_tokens": 12,
                },
                "train_examples": 64,
                "train_packed_examples": 11,
            },
        }),
        encoding="utf-8",
    )

    path = build_training_report(
        TrainingReportOptions(
            train_jsonl=train_jsonl,
            modal_result_json=modal_json,
            output_html=output_html,
        ),
    )

    html = path.read_text(encoding="utf-8")
    assert "MODAL_PROFILE=openmedical" in html
    assert "--batch-size 2" in html
    assert "--gradient-accumulation-steps 1" in html
    assert "--raw-log project-manager/preview/raw_logs/result.log" in html
    assert "--out project-manager/preview/result.json" in html
    assert "not_attached_for_config_batch_2" in html
