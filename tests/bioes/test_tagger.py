from __future__ import annotations

from typing import override

import torch
from torch import nn

from meddies_pii.training.bioes.data.tagger import HiddenStateTokenTagger


class RecordingBackbone(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)
        self.last_kwargs: dict[str, object] | None = None

    @override
    def forward(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        hidden = self.embedding(kwargs["input_ids"])
        if kwargs.get("output_hidden_states"):
            return {"hidden_states": (hidden,)}
        return {"last_hidden_state": hidden}


def test_hidden_state_tagger_limits_causal_lm_logits_when_requesting_hidden_states() -> None:
    backbone = RecordingBackbone()
    tagger = HiddenStateTokenTagger(
        backbone=backbone,
        hidden_size=4,
        num_labels=3,
        request_hidden_states=True,
    )

    result = tagger(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
    )

    assert tuple(result["logits"].shape) == (1, 3, 3)
    assert backbone.last_kwargs is not None
    assert backbone.last_kwargs["logits_to_keep"] == 1


def test_hidden_state_tagger_does_not_pass_logits_limit_to_encoder_backbone() -> None:
    backbone = RecordingBackbone()
    tagger = HiddenStateTokenTagger(
        backbone=backbone,
        hidden_size=4,
        num_labels=3,
        request_hidden_states=False,
    )

    result = tagger(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
    )

    assert tuple(result["logits"].shape) == (1, 3, 3)
    assert backbone.last_kwargs is not None
    assert "logits_to_keep" not in backbone.last_kwargs


def test_hidden_state_tagger_forwards_uncontaminated_packing_metadata() -> None:
    backbone = RecordingBackbone()
    tagger = HiddenStateTokenTagger(
        backbone=backbone,
        hidden_size=4,
        num_labels=3,
        request_hidden_states=True,
    )
    packed_seq_lengths = torch.tensor([2, 1], dtype=torch.int32)
    position_ids = torch.tensor([[0, 1, 0]], dtype=torch.int32)

    result = tagger(
        input_ids=torch.tensor([[1, 2, 3]]),
        labels=torch.tensor([[0, 1, -100]]),
        packed_seq_lengths=packed_seq_lengths,
        position_ids=position_ids,
    )

    assert tuple(result["logits"].shape) == (1, 3, 3)
    assert result["loss"] is not None
    assert backbone.last_kwargs is not None
    assert "attention_mask" not in backbone.last_kwargs
    assert backbone.last_kwargs["packed_seq_lengths"] is packed_seq_lengths
    assert backbone.last_kwargs["position_ids"] is position_ids


def test_hidden_state_tagger_can_isolate_packed_segments_before_backbone() -> None:
    """If the backbone had seen the packed row as one sequence.

    Token 3's hidden value would include prefix tokens 1 and 2. Isolation resets it to 3.

    """

    class ContaminatingBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[int, ...]] = []

        @override
        def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
            ids = kwargs["input_ids"]
            self.calls.append(tuple(ids.flatten().tolist()))
            hidden = ids.float().cumsum(dim=1).unsqueeze(-1).repeat(1, 1, 4)
            return {"last_hidden_state": hidden}

    backbone = ContaminatingBackbone()
    tagger = HiddenStateTokenTagger(
        backbone=backbone,
        hidden_size=4,
        num_labels=3,
        request_hidden_states=False,
        packed_segment_isolation=True,
    )

    result = tagger(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        labels=torch.tensor([[0, 0, 1, 1]]),
        packed_seq_lengths=torch.tensor([2, 2], dtype=torch.int32),
        position_ids=torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
    )

    assert backbone.calls == [(1, 2), (3, 4)]
    assert result["logits"].shape == (1, 4, 3)
