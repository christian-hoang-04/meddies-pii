from __future__ import annotations

from typing import TYPE_CHECKING, override

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
from meddies_pii.annotations.bioes import TokenizedExample
from meddies_pii.annotations.tagged_text import ParsedTaggedDocument
from meddies_pii.training.bioes.data.preparation import PreparedRow
from meddies_pii.training.bioes.trainers.contamination import (
    packed_row_token_range,
    run_packed_attention_contamination_probe,
)
from meddies_pii.training.bioes.trainers.packing import (
    StatefulPacker,
    collate_packed_units,
    pack_prepared_rows,
)

if TYPE_CHECKING:
    from torch import Tensor


class DummyTokenizer:
    pad_token_id = 0


def _row(uid: str, input_ids: list[int], labels: list[int]) -> PreparedRow:
    raw = f"raw-{uid}"
    return PreparedRow(
        uid=uid,
        raw=raw,
        provided_label_json={},
        parsed_label_json={},
        parsed=ParsedTaggedDocument(
            text=raw,
            normalized_text=raw,
            had_label_repairs=False,
            raw=raw,
            spans=(),
        ),
        tokenized=TokenizedExample(
            raw=raw,
            spans=(),
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
            offset_mapping=[(idx, idx + 1) for idx in range(len(input_ids))],
        ),
    )


def test_pack_prepared_rows_emits_unsloth_seq_lengths_and_reset_positions() -> None:
    units = pack_prepared_rows(
        [_row("a", [10, 11, 12], [1, 2, 3]), _row("b", [20, 21], [4, 5])],
        max_length=8,
        pad_token_id=0,
    )

    assert len(units) == 1
    unit = units[0]
    assert unit.row_uids == ("a", "b")
    assert unit.input_ids == (10, 11, 12, 20, 21, 0, 0, 0)
    assert unit.labels == (1, 2, 3, 4, 5, -100, -100, -100)
    assert unit.seq_lengths == (3, 2, 3)
    assert unit.position_ids == (0, 1, 2, 0, 1, 0, 1, 2)
    assert unit.real_token_count == 5
    assert unit.padded_token_count == 8


def test_collate_packed_units_has_no_attention_mask_and_preserves_boundaries() -> None:
    units = pack_prepared_rows(
        [
            _row("a", [10, 11, 12], [1, 2, 3]),
            _row("b", [20, 21], [4, 5]),
            _row("c", [30, 31, 32, 33], [6, 7, 8, 9]),
        ],
        max_length=8,
        pad_token_id=0,
    )

    batch = collate_packed_units(units, device="cpu")

    assert set(batch) == {"input_ids", "labels", "packed_seq_lengths", "position_ids"}
    assert tuple(batch["input_ids"].shape) == (2, 8)
    assert tuple(batch["labels"].shape) == (2, 8)
    assert batch["packed_seq_lengths"].tolist() == [3, 2, 3, 4, 4]
    assert batch["packed_seq_lengths"].sum().item() == batch["input_ids"].numel()
    assert batch["position_ids"].tolist() == [
        [0, 1, 2, 0, 1, 0, 1, 2],
        [0, 1, 2, 3, 0, 1, 2, 3],
    ]


def test_pack_prepared_rows_can_insert_fixed_boundary_guard_segments() -> None:
    units = pack_prepared_rows(
        [_row("a", [10, 11], [1, 2]), _row("b", [20, 21], [3, 4])],
        max_length=8,
        pad_token_id=0,
        boundary_token_count=2,
        boundary_token_id=99,
    )

    unit = units[0]
    assert unit.input_ids == (10, 11, 99, 99, 20, 21, 0, 0)
    assert unit.labels == (1, 2, -100, -100, 3, 4, -100, -100)
    assert unit.seq_lengths == (2, 2, 2, 2)
    assert unit.position_ids == (0, 1, 0, 1, 0, 1, 0, 1)
    assert packed_row_token_range(unit, row_uid="a") == (0, 2)
    assert packed_row_token_range(unit, row_uid="b") == (4, 6)


def test_stateful_packer_chunk_boundary_matches_one_shot_output() -> None:
    rows = [
        _row("a", [1, 2, 3], [1, 1, 1]),
        _row("b", [4, 5], [2, 2]),
        _row("c", [6, 7, 8], [3, 3, 3]),
    ]
    expected = pack_prepared_rows(rows, max_length=6, pad_token_id=0, boundary_token_count=1)
    packer = StatefulPacker(max_length=6, pad_token_id=0, boundary_token_count=1)
    actual = [*packer.push(rows[:2]), *packer.push(rows[2:]), *packer.finish()]
    assert actual == expected


def test_packed_attention_contamination_probe_catches_prefix_leakage() -> None:
    from torch import nn

    class ContaminatingBackbone(nn.Module):
        @override
        def forward(self, input_ids: Tensor, **_kwargs: object) -> dict[str, Tensor]:
            hidden = input_ids.float().cumsum(dim=1).unsqueeze(-1).repeat(1, 1, 4)
            return {"last_hidden_state": hidden}

    class ToyTagger(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = ContaminatingBackbone()
            self.classifier = nn.Linear(4, 3, bias=False)
            nn.init.ones_(self.classifier.weight)

        @override
        def forward(self, **batch: Tensor) -> dict[str, Tensor]:
            hidden = self.backbone(**batch)["last_hidden_state"]
            return {"logits": self.classifier(hidden)}

    clean = pack_prepared_rows(
        [_row("prefix", [1, 2], [0, 0]), _row("target", [3, 4], [1, 1])],
        max_length=4,
        pad_token_id=0,
    )[0]
    mutated = pack_prepared_rows(
        [_row("prefix", [9, 9], [0, 0]), _row("target", [3, 4], [1, 1])],
        max_length=4,
        pad_token_id=0,
    )[0]

    result = run_packed_attention_contamination_probe(
        ToyTagger(),
        clean_unit=clean,
        mutated_unit=mutated,
        target_row_uid="target",
        device="cpu",
        atol=0.0,
        rtol=0.0,
    )

    assert not result.passed
    assert result.max_abs_diff > 0


def test_packed_attention_contamination_probe_accepts_block_isolated_logits() -> None:
    import torch
    from torch import nn

    class PackedAwareBackbone(nn.Module):
        @override
        def forward(self, input_ids: Tensor, packed_seq_lengths: Tensor, **_kwargs: object) -> dict[str, Tensor]:
            hidden = torch.zeros((*input_ids.shape, 4), dtype=torch.float32, device=input_ids.device)
            flat_ids = input_ids.reshape(-1)
            flat_hidden = hidden.reshape(-1, 4)
            offset = 0
            for length in packed_seq_lengths.tolist():
                segment = flat_ids[offset : offset + length].float()
                flat_hidden[offset : offset + length] = segment.sum()
                offset += length
            return {"last_hidden_state": hidden}

    class ToyTagger(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = PackedAwareBackbone()
            self.classifier = nn.Linear(4, 3, bias=False)
            nn.init.ones_(self.classifier.weight)

        @override
        def forward(self, **batch: Tensor) -> dict[str, Tensor]:
            hidden = self.backbone(**batch)["last_hidden_state"]
            return {"logits": self.classifier(hidden)}

    clean = pack_prepared_rows(
        [_row("prefix", [1, 2], [0, 0]), _row("target", [3, 4], [1, 1])],
        max_length=4,
        pad_token_id=0,
    )[0]
    mutated = pack_prepared_rows(
        [_row("prefix", [9, 9], [0, 0]), _row("target", [3, 4], [1, 1])],
        max_length=4,
        pad_token_id=0,
    )[0]

    result = run_packed_attention_contamination_probe(
        ToyTagger(),
        clean_unit=clean,
        mutated_unit=mutated,
        target_row_uid="target",
        device="cpu",
        atol=0.0,
        rtol=0.0,
    )

    assert result.passed
    assert result.max_abs_diff == 0


def test_stateful_packer_finish_is_one_shot_and_snapshot_restores_carry() -> None:
    packer = StatefulPacker(max_length=5, pad_token_id=0)
    assert packer.push([_row("a", [1, 2], [1, 1])]) == []
    restored = StatefulPacker.from_snapshot(packer.snapshot())
    assert [
        *restored.push([_row("b", [3, 4, 5], [2, 2, 2])]),
        *restored.finish(),
    ] == pack_prepared_rows(
        [_row("a", [1, 2], [1, 1]), _row("b", [3, 4, 5], [2, 2, 2])],
        max_length=5,
        pad_token_id=0,
    )
    packer.finish()
    import pytest

    with pytest.raises(RuntimeError, match="finish"):
        packer.finish()
    with pytest.raises(RuntimeError, match="cannot push"):
        packer.push([])
