from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType


def _load_preview_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "reports" / "render_bioes_training_setup_preview.py"
    spec = importlib.util.spec_from_file_location("render_bioes_training_setup_preview_for_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_training_setup_preview_writes_launch_artifacts(tmp_path: Path) -> None:
    module = _load_preview_module()
    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    split_summary_json = tmp_path / "split.summary.json"
    dataset_summary_json = tmp_path / "dataset.summary.json"
    smoke_json = tmp_path / "smoke.json"
    output_dir = tmp_path / "report"

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
    _write_jsonl(validation_jsonl, records)
    split_summary_json.write_text(
        json.dumps({
            "train": {"rows": 173652},
            "validation": {"rows": 500},
            "split_validation": {
                "train_validation_id_overlap": 0,
                "train_validation_text_overlap": 0,
                "train_rows_over_max_length": 0,
                "validation_rows_over_max_length": 0,
                "max_length": 4096,
            },
        }),
        encoding="utf-8",
    )
    dataset_summary_json.write_text(
        json.dumps({
            "all_rows_summary": {
                "rows": 174152,
                "label_span_counts": {
                    "human_name": 629328,
                    "private_url": 24196,
                },
                "domain_bucket_counts": {"medical": 126913, "general": 44999},
            },
        }),
        encoding="utf-8",
    )
    smoke_json.write_text(
        json.dumps({
            "result": {
                "train_examples": 1024,
                "train_packed_examples": 254,
                "train_packing_utilization": 0.7960849071112205,
                "cuda_memory": {
                    "max_reserved_gb": 14.63812096,
                    "device": "NVIDIA H100 80GB HBM3",
                },
                "artifact_persisted": True,
            },
        }),
        encoding="utf-8",
    )

    report = module.render_html(
        train_jsonl=train_jsonl,
        validation_jsonl=validation_jsonl,
        split_summary_json=split_summary_json,
        dataset_summary_json=dataset_summary_json,
        smoke_result_json=smoke_json,
        output_dir=output_dir,
        sample_count=1,
    )

    html = report.read_text(encoding="utf-8")
    command = (output_dir / "training-command.sh").read_text(encoding="utf-8")
    hparams = json.loads((output_dir / "hyperparameters.json").read_text())

    assert report == output_dir / "index.html"
    assert "Meddies PII — BIOES full training setup" in html
    assert "Not launched yet" in html
    assert "r128 / alpha256" in html
    assert "Train cap" in html
    assert "Budget-bounded chunk" in html
    assert "Patient " in html
    assert 'class="pii label-human-name"' in html
    assert "meddies-pii-bioes-artifacts" in html
    assert "--max-length 8192" in command
    assert "modal run --detach --timestamps" in command
    assert "--batch-size 128" in command
    assert "--lora-rank 128" in command
    assert "--lora-alpha 256" in command
    assert "--train-limit 173652" in command
    assert "--steps 120" in command
    assert "--checkpoint-every-steps 10" in command
    assert "--logging-steps 1" in command
    assert "--epochs" not in command
    assert "120step_ckpt10" in hparams["artifacts"]["artifact_root"]
    assert hparams["training"]["max_length"] == 8192
    assert hparams["training"]["epochs"] is None
    assert hparams["training"]["steps"] == 120
    assert hparams["training"]["checkpoint_every_steps"] == 10
    assert hparams["data"]["train_limit_semantics"] == ("upper_bound_before_quality_filtering")
    assert hparams["lora"]["rank"] == 128
    assert hparams["lora"]["alpha"] == 256
    assert hparams["lora"]["target_groups"] == {
        "attention": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "mlp": ["in_proj", "w1", "w2", "w3"],
    }
