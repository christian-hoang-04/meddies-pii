"""Hidden-state token tagger — a BIOES classifier head over a backbone.

Reads the backbone's last hidden state and projects it to the BIOES label
space via a single linear layer. Loss uses cross-entropy with
``IGNORE_INDEX`` marking padding positions.
"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the training stack is an optional extra; importing it at module load would make the package unimportable without
# reason: it.
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    cast,
    override,
    runtime_checkable,
)

from meddies_pii.annotations.bioes import IGNORE_INDEX

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.nn import Module as _ModuleBase
else:
    try:
        from torch.nn import Module as _ModuleBase
    except ImportError:

        class _ModuleBase:
            pass


class Backbone(Protocol):
    def __call__(self, **kwargs: Tensor | bool | int) -> object: ...


class TokenTaggerOutput(TypedDict):
    loss: Tensor | None
    logits: Tensor


@runtime_checkable
class _LastHiddenStateOutput(Protocol):
    last_hidden_state: object


@runtime_checkable
class _HiddenStatesOutput(Protocol):
    hidden_states: object


def _torch_tensor(value: object) -> Tensor | None:
    import torch

    return value if isinstance(value, torch.Tensor) else None


def extract_sequence_output(outputs: object) -> Tensor:
    if isinstance(outputs, Mapping):
        output_mapping = cast("Mapping[object, object]", outputs)
        last_hidden_state = _torch_tensor(output_mapping.get("last_hidden_state"))
        if last_hidden_state is not None:
            return last_hidden_state
        hidden_states = output_mapping.get("hidden_states")
    else:
        last_hidden_state = (
            _torch_tensor(outputs.last_hidden_state) if isinstance(outputs, _LastHiddenStateOutput) else None
        )
        if last_hidden_state is not None:
            return last_hidden_state
        hidden_states = outputs.hidden_states if isinstance(outputs, _HiddenStatesOutput) else None

    if isinstance(hidden_states, Sequence) and hidden_states:
        hidden_state_sequence = cast("Sequence[object]", hidden_states)
        last_hidden_state = _torch_tensor(hidden_state_sequence[-1])
        if last_hidden_state is not None:
            return last_hidden_state
    msg = "No hidden-state output found in backbone result"
    raise ValueError(msg)


class HiddenStateTokenTagger(_ModuleBase):
    # reason: HiddenStateTokenTagger exposes backbone/packed as its HiddenStateTokenTagge; bundling would break callers.
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        backbone: Backbone,
        hidden_size: int,
        num_labels: int,
        *,
        request_hidden_states: bool,
        classifier_dtype: torch.dtype | None = None,
        dropout: float = 0.1,
        packed_segment_isolation: bool = False,
    ) -> None:
        try:
            from torch import nn
        except ImportError as exc:
            msg = "torch is required to instantiate HiddenStateTokenTagger"
            raise ImportError(msg) from exc
        super().__init__()
        self.backbone = backbone
        self.request_hidden_states = request_hidden_states
        self.packed_segment_isolation = packed_segment_isolation
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=classifier_dtype)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **packing_kwargs: Tensor,
    ) -> TokenTaggerOutput:
        from torch.nn import functional

        packed_seq_lengths = packing_kwargs.get("packed_seq_lengths")
        if self.packed_segment_isolation and packed_seq_lengths is not None:
            hidden = self._isolated_packed_hidden(
                input_ids=input_ids,
                labels=labels,
                position_ids=packing_kwargs.get("position_ids"),
                packed_seq_lengths=packed_seq_lengths,
            )
        else:
            outputs = self._backbone_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **packing_kwargs,
            )
            hidden = extract_sequence_output(outputs)
        classifier_dtype = self.classifier.weight.dtype
        if getattr(hidden, "dtype", None) != classifier_dtype:
            hidden = hidden.to(classifier_dtype)
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return {"loss": loss, "logits": logits}

    def _backbone_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **packing_kwargs: Tensor,
    ) -> object:
        """CausalLM backbones such as LFM2 still compute vocabulary logits even though this tagger only.

        CausalLM backbones such as LFM2 still compute vocabulary logits even though this tagger only consumes the hidden
        states. Keep a single-token LM projection to avoid allocating [batch, sequence, vocab] logits for long-context
        BIOES batches.

        Returns:
            The backbone output, from which the caller reads hidden states. Caching is off and
            the LM projection is kept to a single token, so a long-context BIOES batch never
            allocates the ``[batch, sequence, vocab]`` logits tensor it would not read.

        """
        backbone_kwargs: dict[str, Tensor | bool | int] = {
            "input_ids": input_ids,
            "use_cache": False,
            "return_dict": True,
            "output_hidden_states": self.request_hidden_states,
        }
        if attention_mask is not None:
            backbone_kwargs["attention_mask"] = attention_mask
        backbone_kwargs.update(packing_kwargs)
        if self.request_hidden_states:
            backbone_kwargs["logits_to_keep"] = 1
        return self.backbone(**backbone_kwargs)

    def _isolated_packed_hidden(
        self,
        *,
        input_ids: Tensor,
        labels: Tensor | None,
        position_ids: Tensor | None,
        packed_seq_lengths: Tensor,
    ) -> Tensor:
        import torch

        batch_size, seq_length = input_ids.shape
        flat_input_ids = input_ids.reshape(-1)
        flat_labels = labels.reshape(-1) if labels is not None else None
        flat_position_ids = position_ids.reshape(-1) if position_ids is not None else None
        segment_lengths = [int(length.item()) for length in packed_seq_lengths.detach().cpu().reshape(-1).unbind()]
        if sum(segment_lengths) != int(flat_input_ids.numel()):
            msg = "sum(packed_seq_lengths) must match flattened input_ids length"
            raise ValueError(msg)

        pieces: list[Tensor] = []
        offset = 0
        for length in segment_lengths:
            end = offset + length
            segment_labels = flat_labels[offset:end] if flat_labels is not None else None
            if segment_labels is not None and bool(torch.all(segment_labels == IGNORE_INDEX).detach().cpu()):
                pieces.append(self.classifier.weight.new_zeros((1, length, self.classifier.in_features)))
            else:
                segment_kwargs: dict[str, Tensor] = {}
                if flat_position_ids is not None:
                    segment_kwargs["position_ids"] = flat_position_ids[offset:end].unsqueeze(0)
                outputs = self._backbone_forward(
                    input_ids=flat_input_ids[offset:end].unsqueeze(0),
                    attention_mask=torch.ones((1, length), dtype=torch.long, device=input_ids.device),
                    **segment_kwargs,
                )
                pieces.append(extract_sequence_output(outputs))
            offset = end

        hidden = torch.cat(pieces, dim=1)
        return hidden.reshape(batch_size, seq_length, self.classifier.in_features)
