from __future__ import annotations

# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast, override

import pytest
import torch

from meddies_pii.annotations.bioes.encoding import TokenizedExample
from meddies_pii.annotations.tagged_text import ParsedTaggedDocument
from meddies_pii.training.bioes.data.artifacts import ProbeArtifacts
from meddies_pii.training.bioes.data.preparation import PreparationStats, PreparedRow
from meddies_pii.training.bioes.trainers import trainer
from meddies_pii.training.bioes.trainers.checkpointing import LoadedTrainingCheckpoint
from meddies_pii.training.bioes.trainers.packing_runtime import PreparedTrainingUnits

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedTokenizerBase

    from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger

_SMOKE_ROW_RAW = "patient"


def make_prepared_row(uid: str) -> PreparedRow:
    """Build a real `PreparedRow` for a test that hands rows to production training code.

    The trainer and the packing runtime take `Sequence[PreparedRow]` and reach through
    `row.tokenized.input_ids` with `dataclasses.replace`, which no Protocol can express. A double
    narrower than the record stands in for a shape the callee never accepts, so this constructs the
    record itself: two tokens, empty spans, and no label JSON.
    """
    return PreparedRow(
        uid=uid,
        raw=_SMOKE_ROW_RAW,
        provided_label_json={},
        parsed_label_json={},
        parsed=ParsedTaggedDocument(
            text=_SMOKE_ROW_RAW,
            normalized_text=_SMOKE_ROW_RAW,
            had_label_repairs=False,
            raw=_SMOKE_ROW_RAW,
            spans=(),
        ),
        tokenized=TokenizedExample(
            raw=_SMOKE_ROW_RAW,
            spans=(),
            input_ids=[1, 2],
            attention_mask=[1, 1],
            labels=[0, 0],
            offset_mapping=[(0, 3), (4, 7)],
        ),
    )


class _SmokeTagger(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.backbone = torch.nn.Linear(1, 1)

    @override
    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        input_ids = batch.get("input_ids")
        logits = (
            torch.zeros((*input_ids.shape, 1), device=input_ids.device)
            if input_ids is not None
            else torch.zeros((1, 1, 1))
        )
        return {"loss": self.weight.square(), "logits": logits}


def _smoke_metric(f1: float) -> dict[str, object]:
    block = {
        "tp": 1.0,
        "pred_total": 1.0,
        "gold_total": 1.0,
        "precision": f1,
        "recall": f1,
        "f1": f1,
    }
    return {
        **block,
        "typed": block,
        "untyped": block,
        "untyped_minus_typed": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "slices": {},
        "containment_span": {
            **block,
            "typed": block,
            "untyped": block,
            "untyped_minus_typed": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "slices": {},
        },
    }


@pytest.fixture
def prepared_row_factory() -> Callable[[str], PreparedRow]:
    """Expose `make_prepared_row` to sibling test modules through pytest's own sharing mechanism."""
    return make_prepared_row


@pytest.fixture
def install_smoke_training_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[list[int], Callable[[float], dict[str, object]]]]:
    """Install deterministic external boundaries around public smoke training."""

    def install(
        *,
        resumed: bool = False,
    ) -> tuple[list[int], Callable[[float], dict[str, object]]]:
        model = _SmokeTagger()
        rows = [make_prepared_row("train-1"), make_prepared_row("train-2")]
        metrics = iter([_smoke_metric(0.25), _smoke_metric(0.75)])
        checkpoints: list[int] = []
        artifacts = ProbeArtifacts(
            # reason: the double exposes only the members this call path uses; test-side narrowing is the
            # reason: owner-ruled shape for a stand-in that cannot be a real HiddenStateTokenTagger.
            tagger=cast("HiddenStateTokenTagger", model),
            # reason: the double exposes only the members this call path uses; test-side narrowing is the
            # reason: owner-ruled shape for a stand-in that cannot be a real PreTrainedTokenizerBase.
            tokenizer=cast("PreTrainedTokenizerBase", SimpleNamespace(pad_token_id=0)),
            label_vocab=("O",),
            label_to_id={"O": 0},
        )

        monkeypatch.setattr(trainer, "_build_artifacts", lambda _config: artifacts)
        monkeypatch.setattr(trainer, "_load_rows", lambda *_args, **_kwargs: [{"uid": "source"}])
        monkeypatch.setattr(
            trainer,
            "_select_source_rows",
            lambda source_rows, **_kwargs: (
                source_rows,
                PreparationStats(candidates=len(source_rows), accepted=len(source_rows)),
            ),
        )
        monkeypatch.setattr(
            trainer,
            "_prepare_rows",
            lambda *_args, **_kwargs: (
                rows,
                PreparationStats(candidates=2, accepted=2),
            ),
        )
        monkeypatch.setattr(trainer, "_load_targeted_eval_selection", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            trainer,
            "_optimizer_for_config",
            lambda model, *_args, **_kwargs: torch.optim.SGD(model.parameters(), lr=0.1),
        )
        monkeypatch.setattr(trainer, "_evaluate", lambda *_args, **_kwargs: next(metrics))
        monkeypatch.setattr(
            trainer,
            "prepare_training_units",
            lambda *_args, **_kwargs: PreparedTrainingUnits(rows, None, None, None, None),
        )
        monkeypatch.setattr(
            trainer,
            "_save_classifier_state",
            lambda *_args, **_kwargs: "classifier.pt",
        )
        monkeypatch.setattr(
            trainer,
            "_save_backbone_adapter",
            lambda *_args, **_kwargs: "adapter",
        )
        monkeypatch.setattr(trainer, "_row_hash", lambda row: f"hash:{row.uid}")
        monkeypatch.setattr(trainer, "_package_versions", lambda: {"torch": "fake"})
        monkeypatch.setattr(
            trainer,
            "_cuda_memory_report",
            lambda device: {"device": device, "available": False},
        )
        monkeypatch.setattr(
            trainer,
            "_save_training_checkpoint",
            lambda *, completed_steps, **_kwargs: checkpoints.append(completed_steps) or f"checkpoint-{completed_steps}",
        )
        if resumed:
            monkeypatch.setattr(
                trainer,
                "_load_training_checkpoint",
                lambda **_kwargs: LoadedTrainingCheckpoint("checkpoint-2", 2, [0.9, 0.8]),
            )
        return checkpoints, _smoke_metric

    return install
