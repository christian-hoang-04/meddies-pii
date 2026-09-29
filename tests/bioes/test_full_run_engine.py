from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, override

import pytest

from meddies_pii.training.bioes.trainers.full_run import (
    EVAL_DATASET_REVISION,
    PACKED_DATASET_REVISION,
    PACKED_MANIFEST_SHA256,
    PACKED_UNIT_COUNT,
)
from meddies_pii.training.bioes.trainers.full_run_engine import (
    RuntimeResumeState,
    RunWriter,
    checkpoint_metadata,
    next_step_reason,
    validate_resume_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from typing import Self


def _state() -> RuntimeResumeState:
    return RuntimeResumeState(
        candidate_key="encoder350",
        config_digest="digest",
        packed_revision=PACKED_DATASET_REVISION,
        packed_manifest_sha256=PACKED_MANIFEST_SHA256,
        eval_revision=EVAL_DATASET_REVISION,
        packed_cursor=1920,
        optimizer_step=10,
    )


def test_deadline_stops_before_the_next_optimizer_step() -> None:
    assert (
        next_step_reason(
            elapsed_seconds=8_399.0,
            predicted_next_step_seconds=1.0,
            training_deadline_seconds=8_400.0,
            epoch_complete=False,
        )
        == "deadline_reached"
    )


def test_epoch_complete_wins_over_deadline() -> None:
    assert (
        next_step_reason(
            elapsed_seconds=1_000_000.0,
            predicted_next_step_seconds=1.0,
            training_deadline_seconds=8_400.0,
            epoch_complete=True,
        )
        == "epoch_complete"
    )


def test_resume_rejects_wrong_pins() -> None:
    state = _state()
    validate_resume_state(state, candidate_key="encoder350", config_digest="digest")
    with pytest.raises(RuntimeError, match="packed pin"):
        validate_resume_state(
            replace(state, packed_revision="wrong"),
            candidate_key="encoder350",
            config_digest="digest",
        )
    with pytest.raises(RuntimeError, match="cursor"):
        validate_resume_state(
            replace(state, packed_cursor=PACKED_UNIT_COUNT + 1),
            candidate_key="encoder350",
            config_digest="digest",
        )


def test_writer_persists_each_event_and_atomic_checkpoint_metadata(tmp_path: Path) -> None:
    writer = RunWriter(tmp_path / "run")
    writer.append({"event": "optimizer_step", "step": 1})
    path = writer.atomic_json(
        "checkpoints/step-00000001/metadata.json",
        checkpoint_metadata(_state(), reason="training"),
    )
    assert json.loads(writer.events_path.read_text()) == {
        "event": "optimizer_step",
        "step": 1,
    }
    assert json.loads(path.read_text())["packed_cursor"] == 1920
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_writer_fsyncs_every_event_without_explicit_volume_commit(tmp_path: Path) -> None:
    commits: list[str] = []
    writer = RunWriter(tmp_path / "run", commit=lambda: commits.append("commit"))

    writer.append({"event": "optimizer_step", "step": 1})

    assert commits == []
    assert json.loads(writer.events_path.read_text()) == {
        "event": "optimizer_step",
        "step": 1,
    }
    writer.atomic_json("result.json", {"status": "ok"})
    assert commits == ["commit"]


def test_resume_restores_default_adapter_without_changing_optimizer_parameter_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA-built torch without a driver raises on the state query."""
    from types import SimpleNamespace

    import torch
    from peft import PeftType

    from meddies_pii.training.bioes.trainers import full_run
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _resume_if_requested,
    )

    class _Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
            self.peft_config = {"default": SimpleNamespace(peft_type=PeftType.LORA)}
            self.active_adapter = "default"

        def load_adapter(self, *_args: object, **_kwargs: object) -> str:
            self.resume_adapter = torch.nn.Parameter(torch.ones(1))
            return "resume"

        def set_adapter(self, adapter_name: str) -> None:
            self.active_adapter = adapter_name

    class _Tagger(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _Backbone()
            self.classifier = torch.nn.Linear(1, 37)

    contract = full_run.render_full_run("encoder350")
    state = RuntimeResumeState(
        candidate_key="encoder350",
        config_digest=contract["config_digest"],
        packed_revision=full_run.PACKED_DATASET_REVISION,
        packed_manifest_sha256=full_run.PACKED_MANIFEST_SHA256,
        eval_revision=full_run.EVAL_DATASET_REVISION,
        packed_cursor=192,
        optimizer_step=1,
    )
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "adapter").mkdir()
    (root / "metadata.json").write_text(json.dumps(asdict(state)), encoding="utf-8")
    tagger = _Tagger()
    optimizer = torch.optim.AdamW(tagger.parameters(), lr=1e-4)
    classifier_weight = tagger.classifier.weight.detach().clone()
    torch.save(tagger.classifier.state_dict(), root / "classifier.pt")
    with torch.no_grad():
        tagger.classifier.weight.zero_()
    default_adapter_id = id(tagger.backbone.lora_A["default"].weight)
    torch.save(optimizer.state_dict(), root / "optimizer.pt")
    torch.save(
        {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        root / "rng.pt",
    )
    original_load = torch.load

    # reason: this wrapper forwards straight to torch.load, whose stub is an overload set over
    # reason: path, map_location and weights_only; no annotation on an opaque forward satisfies it.
    def cpu_load(
        *args,  # ruff: ignore[missing-type-args]
        **kwargs,  # ruff: ignore[missing-type-kwargs]
    ) -> object:
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    import peft
    from safetensors import torch as safetensors_torch

    def load_adapter_state(_path: str) -> dict[str, torch.Tensor]:
        return {"lora_A.weight": torch.full((1, 1), 3.0)}

    def load_into_default(
        model: _Backbone,
        adapter_state: Mapping[str, torch.Tensor],
        *,
        adapter_name: str,
    ) -> SimpleNamespace:
        assert adapter_name == "default"
        # reason: ty resolves Tensor.data.copy_ to a Tensor rather than to the bound method, the same
        # reason: stub limitation this file already carries at the three tagger.backbone assertions.
        model.lora_A["default"].weight.data.copy_(  # ty: ignore[call-non-callable]
            adapter_state["lora_A.weight"],
        )
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    monkeypatch.setattr(torch, "load", cpu_load)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _state: None)
    monkeypatch.setattr(safetensors_torch, "load_file", load_adapter_state)
    monkeypatch.setattr(peft, "set_peft_model_state_dict", load_into_default)

    _resume_if_requested(
        str(root),
        tagger=tagger,
        optimizer=optimizer,
        candidate_key="encoder350",
        config_digest=contract["config_digest"],
    )

    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    active_trainable_ids = {id(parameter) for parameter in tagger.backbone.parameters() if parameter.requires_grad}
    assert tagger.backbone.active_adapter == "default"
    assert id(tagger.backbone.lora_A["default"].weight) == default_adapter_id
    # reason: ModuleDict.__getitem__ returns Module, whose __getattr__ is typed Tensor | Module |
    # reason: Parameter, so .weight is a union and ty rejects the call on its non-callable variant.
    # reason: The real object is a Linear, whose .weight is a Parameter. Stub limitation, not a defect.
    assert tagger.backbone.lora_A["default"].weight.item() == 3.0  # ty: ignore[call-non-callable]
    assert torch.equal(tagger.classifier.weight, classifier_weight)
    assert active_trainable_ids <= optimizer_parameter_ids


def test_resume_rejects_missing_or_unexpected_adapter_tensors() -> None:
    from types import SimpleNamespace

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _require_exact_adapter_load,
    )

    with pytest.raises(RuntimeError, match=r"missing=.*adapter"):
        _require_exact_adapter_load(SimpleNamespace(missing_keys=["adapter.lora_A.weight"], unexpected_keys=[]))
    with pytest.raises(RuntimeError, match=r"unexpected=.*adapter"):
        _require_exact_adapter_load(SimpleNamespace(missing_keys=[], unexpected_keys=["adapter.extra.weight"]))


def test_adapter_only_missing_key_filter_allows_frozen_body_but_not_lora_or_extra() -> None:
    from types import SimpleNamespace

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _require_exact_adapter_load,
    )

    keys = ("model.layers.0.lora_A.default.weight",)
    _require_exact_adapter_load(
        SimpleNamespace(missing_keys=["model.layers.0.self_attn.q_proj.weight"], unexpected_keys=[]),
        checkpoint_model_keys=keys,
        active_trainable_adapter_keys=keys,
    )
    with pytest.raises(RuntimeError, match=r"missing=.*lora_A"):
        _require_exact_adapter_load(
            SimpleNamespace(missing_keys=list(keys), unexpected_keys=[]),
            checkpoint_model_keys=keys,
            active_trainable_adapter_keys=keys,
        )
    with pytest.raises(RuntimeError, match=r"unexpected=.*extra"):
        _require_exact_adapter_load(
            SimpleNamespace(missing_keys=[], unexpected_keys=["adapter.extra.weight"]),
            checkpoint_model_keys=keys,
            active_trainable_adapter_keys=keys,
        )


def test_adapter_only_restore_allows_frozen_peft_missing_keys_and_proves_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PEFT reports frozen base keys as missing when a LoRA-only file is loaded."""
    from types import SimpleNamespace

    import torch
    from peft import PeftType
    from safetensors import torch as safetensors_torch

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _restore_pii_training_payloads,
    )

    class _Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = torch.nn.Parameter(torch.full((1,), 17.0), requires_grad=False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
            self.peft_config = {"default": SimpleNamespace(peft_type=PeftType.LORA)}
            self.active_adapter = "default"

        def set_adapter(self, adapter_name: str) -> None:
            self.active_adapter = adapter_name

    class _Tagger(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _Backbone()
            self.classifier = torch.nn.Linear(1, 37)

    root = tmp_path / "checkpoint"
    (root / "adapter").mkdir(parents=True)
    tagger = _Tagger()
    optimizer = torch.optim.AdamW(tagger.parameters(), lr=1e-4)
    classifier_state = tagger.classifier.state_dict()
    torch.save(classifier_state, root / "classifier.pt")
    torch.save(optimizer.state_dict(), root / "optimizer.pt")
    torch.save({"rank_rng_states": []}, root / "rng.pt")
    checkpoint_adapter_state = {
        "lora_A.weight": torch.full((1, 1), 3.0),
        "lora_B.weight": torch.full((1, 1), 5.0),
    }

    original_load = torch.load

    # reason: this wrapper forwards straight to torch.load, whose stub is an overload set over
    # reason: path, map_location and weights_only; no annotation on an opaque forward satisfies it.
    def cpu_load(
        *args,  # ruff: ignore[missing-type-args]
        **kwargs,  # ruff: ignore[missing-type-kwargs]
    ) -> object:
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    def load_into_default(
        model: _Backbone,
        state: Mapping[str, torch.Tensor],
        *,
        adapter_name: str,
    ) -> SimpleNamespace:
        assert adapter_name == "default"
        # reason: ty resolves Tensor.data.copy_ to a Tensor rather than to the bound method, the same
        # reason: stub limitation this file already carries at the three tagger.backbone assertions.
        model.lora_A["default"].weight.data.copy_(state["lora_A.weight"])  # ty: ignore[call-non-callable]
        model.lora_B["default"].weight.data.copy_(state["lora_B.weight"])  # ty: ignore[call-non-callable]
        return SimpleNamespace(missing_keys=["body"], unexpected_keys=[])

    import peft

    monkeypatch.setattr(torch, "load", cpu_load)
    monkeypatch.setattr(safetensors_torch, "load_file", lambda _path: checkpoint_adapter_state)
    monkeypatch.setattr(peft, "set_peft_model_state_dict", load_into_default)

    _restore_pii_training_payloads(root, tagger=tagger, optimizer=optimizer)

    assert tagger.backbone.active_adapter == "default"
    # reason: ModuleDict.__getitem__ returns Module, whose __getattr__ is typed Tensor | Module |
    # reason: Parameter, so .weight is a union and ty rejects the call on its non-callable variant.
    # reason: The real object is a Linear, whose .weight is a Parameter. Stub limitation, not a defect.
    assert torch.equal(
        tagger.backbone.lora_A["default"].weight.detach(),  # ty: ignore[call-non-callable]
        checkpoint_adapter_state["lora_A.weight"],
    )
    assert torch.equal(
        tagger.backbone.lora_B["default"].weight.detach(),  # ty: ignore[call-non-callable]
        checkpoint_adapter_state["lora_B.weight"],
    )
    assert tagger.backbone.body.item() == 17.0
    assert torch.equal(tagger.classifier.weight, classifier_state["weight"])


def test_pii_body_attestation_rejects_missing_or_mismatched_encoder_tensors() -> None:
    import torch

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        pii_body_state_attestation,
        require_pii_body_state_attestation,
    )

    class _FullBody(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            self.second = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

    class _MissingBody(torch.nn.Module):
        def __init__(self, source: _FullBody) -> None:
            super().__init__()
            self.first = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            self.first.load_state_dict(source.first.state_dict())

    expected_body = _FullBody()
    expected = pii_body_state_attestation(expected_body)
    assert require_pii_body_state_attestation(expected, expected_body) == expected
    with pytest.raises(RuntimeError, match="do not match"):
        require_pii_body_state_attestation(expected, _MissingBody(expected_body))

    mismatched = _FullBody()
    mismatched.load_state_dict(expected_body.state_dict())
    with torch.no_grad():
        mismatched.second.weight.add_(1)
    with pytest.raises(RuntimeError, match="do not match"):
        require_pii_body_state_attestation(expected, mismatched)


def test_pii_checkpoint_attestation_normalizes_unsloth_namespace_but_checks_values(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        lfm2_checkpoint_body_state_attestation,
        pii_tensor_state_attestation,
        require_lfm2_checkpoint_body_state_attestation,
    )

    weight = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    assert pii_tensor_state_attestation({"lfm2.weight": weight}) == (
        pii_tensor_state_attestation({"base_model.model.lfm2.weight": weight})
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    save_file({"lfm2.weight": weight}, str(snapshot / "model.safetensors"))
    expected = lfm2_checkpoint_body_state_attestation(str(snapshot))
    assert require_lfm2_checkpoint_body_state_attestation(expected, str(snapshot)) == expected

    save_file({"lfm2.weight": weight + 1}, str(snapshot / "model.safetensors"))
    with pytest.raises(RuntimeError, match="do not match"):
        require_lfm2_checkpoint_body_state_attestation(expected, str(snapshot))


def test_unsloth_encoder_lora_uses_feature_extraction_not_causal_generation() -> None:
    from types import SimpleNamespace

    from meddies_pii.training.bioes.trainers import full_run
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _attach_unsloth_feature_extraction_lora,
    )

    class _FastModel:
        kwargs: dict[str, object]

        @classmethod
        def get_peft_model(cls, model: SimpleNamespace, **kwargs: object) -> SimpleNamespace:
            cls.kwargs = kwargs
            if kwargs.get("task_type") != "FEATURE_EXTRACTION":
                msg = "prepare_inputs_for_generation"
                raise AttributeError(msg)
            model.peft_config = {"default": SimpleNamespace(task_type=kwargs["task_type"])}
            return model

    model = SimpleNamespace()
    adapted = _attach_unsloth_feature_extraction_lora(
        _FastModel,
        model,
        task_type="FEATURE_EXTRACTION",
        contract=full_run.render_full_run("encoder350"),
    )
    assert adapted is model
    assert _FastModel.kwargs["task_type"] == "FEATURE_EXTRACTION"
    assert _FastModel.kwargs["r"] == 64
    assert _FastModel.kwargs["lora_alpha"] == 128
    assert _FastModel.kwargs["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "in_proj",
        "w1",
        "w2",
        "w3",
    ]

    class _WrongTaskFastModel(_FastModel):
        @classmethod
        @override
        def get_peft_model(cls, model: SimpleNamespace, **kwargs: object) -> SimpleNamespace:
            model.peft_config = {"default": SimpleNamespace(task_type="CAUSAL_LM")}
            return model

    with pytest.raises(RuntimeError, match="not FEATURE_EXTRACTION"):
        _attach_unsloth_feature_extraction_lora(
            _WrongTaskFastModel,
            SimpleNamespace(),
            task_type="FEATURE_EXTRACTION",
            contract=full_run.render_full_run("encoder350"),
        )


@pytest.mark.parametrize("profile", ["meddies-pii", "anhthunguyenump"])
def test_pii350_encoder_path_uses_the_immutable_contract_lora_mechanics(
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise PII350's real encoder-loader seam without importing Unsloth."""
    import sys
    from types import ModuleType, SimpleNamespace

    from meddies_pii.training.bioes.data import tagger as tagger_module
    from meddies_pii.training.bioes.trainers import (
        base230_unsloth_probe,
        full_run,
        full_run_runtime,
    )

    calls: list[dict[str, object]] = []

    class _Body:
        config = SimpleNamespace(hidden_size=1024, use_cache=True)

    class _FastModel:
        @classmethod
        def from_pretrained(cls, **_kwargs: object) -> tuple[SimpleNamespace, SimpleNamespace]:
            return (
                SimpleNamespace(lfm2=_Body(), _unsloth_trust_remote_code=True),
                # reason: these are the tokenizer's sentinel strings, not credentials. The rule keys on
                # reason: a name containing `token`, and a stand-in for a tokenizer has to carry the
                # reason: names the real one does.
                SimpleNamespace(pad_token_id=None, eos_token="<eos>", pad_token=None),  # ruff: ignore[hardcoded-password-func-arg]
            )

        @classmethod
        def get_peft_model(cls, model: SimpleNamespace, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            model.peft_config = {"default": SimpleNamespace(task_type=kwargs["task_type"])}
            return model

    class _Tagger:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def cuda(self) -> Self:
            return self

        def train(self) -> Self:
            return self

    unsloth = ModuleType("unsloth")
    peft = ModuleType("peft")
    transformers = ModuleType("transformers")
    # reason: ModuleType declares no attribute beyond the module dunders, so ty cannot type an
    # reason: assignment onto a stand-in module even though every read through it resolves at runtime
    # reason: via ModuleType.__getattr__. Building these as real modules is the point -- the code under
    # reason: test imports them by name out of sys.modules -- so there is no annotation that admits the
    # reason: write. Recorded in the burn-down report as the ModuleType-global case.
    unsloth.FastModel = _FastModel  # ty: ignore[unresolved-attribute]
    peft.TaskType = SimpleNamespace(FEATURE_EXTRACTION="FEATURE_EXTRACTION")  # ty: ignore[unresolved-attribute]
    transformers.AutoModelForMaskedLM = object  # ty: ignore[unresolved-attribute]
    transformers.AutoModelForTokenClassification = object  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(base230_unsloth_probe, "resolve_candidate_snapshot", lambda *_: "/snapshot")
    monkeypatch.setattr(
        full_run_runtime,
        "require_lfm2_checkpoint_body_state_attestation",
        lambda *_: {},
    )
    monkeypatch.setattr(full_run_runtime, "_runtime_attestation", lambda **_: {})
    monkeypatch.setattr(tagger_module, "HiddenStateTokenTagger", _Tagger)

    contract = full_run.render_pii350_milestone_run_contract(profile)
    full_run_runtime._unsloth_encoder_tagger(
        "pii350",
        contract=contract,
        expected_encoder_checkpoint_attestation={},
    )

    assert len(calls) == 1
    mechanics = contract["training"]["mechanics"]["lora"]
    assert calls[0] == {
        "r": 128,
        "target_modules": mechanics["target_modules"],
        "lora_alpha": 256,
        "lora_dropout": mechanics["dropout"],
        "bias": mechanics["bias"],
        "use_gradient_checkpointing": "unsloth",
        "use_rslora": mechanics["use_rslora"],
        "use_dora": mechanics["use_dora"],
        "random_state": mechanics["random_state"],
        "task_type": "FEATURE_EXTRACTION",
    }


def test_resume_moves_optimizer_state_to_each_parameter_device() -> None:
    from types import SimpleNamespace

    import torch

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _move_optimizer_state_to_parameter_devices,
    )

    parameter = torch.nn.Parameter(torch.empty(1, device="meta"))
    optimizer = SimpleNamespace(state={parameter: {"exp_avg": torch.ones(1)}})

    _move_optimizer_state_to_parameter_devices(optimizer)

    assert optimizer.state[parameter]["exp_avg"].device.type == "meta"


def test_runtime_label_vocabulary_mismatch_fails_closed_and_persists_exact_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meddies_pii.training.bioes.trainers import full_run, full_run_runtime

    contract = full_run.render_full_run("base230")
    labels = full_run_runtime._validate_runtime_label_vocabulary(contract)
    writer = RunWriter(tmp_path / "run")
    full_run_runtime._persist_label_vocabulary(writer, labels)
    assert json.loads((writer.root / "label_vocabulary.json").read_text()) == {
        "labels": contract["label_vocabulary"],
    }

    monkeypatch.setattr(full_run_runtime, "_runtime_label_vocabulary", lambda: ["O"])
    with pytest.raises(RuntimeError, match="label vocabulary mismatch"):
        full_run_runtime._validate_runtime_label_vocabulary(contract)


def test_trainable_lora_proof_requires_every_target_and_fp32_adapters() -> None:
    from types import SimpleNamespace

    from meddies_pii.training.bioes.trainers.config import LORA_TARGET_MODULES
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _validate_trainable_lora_parameters,
    )

    class Model:
        def __init__(self, names: list[str]) -> None:
            self.names = names

        def named_parameters(self) -> list[tuple[str, SimpleNamespace]]:
            return [(name, SimpleNamespace(requires_grad=True, dtype="torch.float32")) for name in self.names]

    all_targets = [f"base_model.{target}.lora_A.default.weight" for target in LORA_TARGET_MODULES]
    assert _validate_trainable_lora_parameters(Model(all_targets), LORA_TARGET_MODULES) == {
        "lora_trainable_parameter_dtypes": ["torch.float32"],
        "resolved_lora_modules": sorted(LORA_TARGET_MODULES),
    }
    with pytest.raises(RuntimeError, match="no trainable LoRA"):
        _validate_trainable_lora_parameters(Model([]), LORA_TARGET_MODULES)
    with pytest.raises(RuntimeError, match=r"lack adapters: \['in_proj'\]"):
        _validate_trainable_lora_parameters(
            Model([target for target in all_targets if ".in_proj." not in target]),
            LORA_TARGET_MODULES,
        )


def test_optimizer_progress_reports_remaining_epoch_and_deadline_projections() -> None:
    from meddies_pii.training.bioes.trainers.full_run_runtime import _optimizer_progress

    progress = _optimizer_progress(
        cursor=240,
        batch_size=240,
        durations=[4.0, 6.0, 8.0],
        elapsed_seconds=60.0,
    )
    assert progress == {
        "packed_epoch_progress_fraction": 240 / 120_036,
        "packed_units_remaining": 119_796,
        "training_deadline_seconds_remaining": 8_340.0,
        "rolling_median_step_seconds": 6.0,
        "projected_epoch_eta_seconds": 3_000.0,
        "projected_steps_before_deadline": 1_390,
    }


def test_runtime_attestation_requires_observable_training_state_and_reports_attention() -> None:
    from types import SimpleNamespace

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _observed_attention_implementation,
        _validate_observed_training_state,
    )

    model = SimpleNamespace(
        is_gradient_checkpointing=True,
        config=SimpleNamespace(use_cache=False, _attn_implementation="sdpa"),
    )
    assert _validate_observed_training_state(model) == {
        "observed_gradient_checkpointing": True,
        "observed_use_cache": False,
    }
    assert _observed_attention_implementation(model) == "sdpa"
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        _validate_observed_training_state(SimpleNamespace(config=SimpleNamespace(use_cache=False)))
    with pytest.raises(RuntimeError, match="disable use_cache"):
        _validate_observed_training_state(SimpleNamespace(is_gradient_checkpointing=True, config=SimpleNamespace()))


def test_unsloth_encoder_body_preserves_true_wrapper_remote_code_provenance() -> None:
    from types import SimpleNamespace

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _propagate_unsloth_remote_code_provenance,
    )

    body = SimpleNamespace()
    _propagate_unsloth_remote_code_provenance(SimpleNamespace(_unsloth_trust_remote_code=True), body)
    assert body._unsloth_trust_remote_code is True


@pytest.mark.parametrize("provenance", [None, False, 1, "true"])
def test_unsloth_encoder_body_rejects_missing_false_or_non_bool_provenance(
    provenance: object,
) -> None:
    from types import SimpleNamespace

    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _propagate_unsloth_remote_code_provenance,
    )

    with pytest.raises(RuntimeError, match="provenance"):
        _propagate_unsloth_remote_code_provenance(
            SimpleNamespace(_unsloth_trust_remote_code=provenance),
            SimpleNamespace(),
        )


def test_runtime_package_attestation_fails_closed_for_wrong_candidate_image() -> None:
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _require_exact_runtime_package_versions,
    )

    packages = ("unsloth==2026.7.4", "unsloth_zoo==2026.7.4")
    assert _require_exact_runtime_package_versions(packages, {"unsloth": "2026.7.4", "unsloth_zoo": "2026.7.4"}) == {
        "unsloth": "2026.7.4",
        "unsloth_zoo": "2026.7.4",
    }
    with pytest.raises(RuntimeError, match="attestation mismatch"):
        _require_exact_runtime_package_versions(packages, {"unsloth": "2026.5.2", "unsloth_zoo": "2026.5.1"})


def test_runtime_adamw_parameters_are_validated_from_the_immutable_contract() -> None:
    from meddies_pii.training.bioes.trainers import full_run
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _contract_adamw_parameters,
    )

    contract = full_run.render_pii350_milestone_run_contract("meddies-pii")
    assert asdict(_contract_adamw_parameters(contract)) == {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "fused": False,
    }
    contract["optimizer"]["lr"] = 2e-4
    with pytest.raises(RuntimeError, match="disagree"):
        _contract_adamw_parameters(contract)
    contract = full_run.render_pii350_milestone_run_contract("meddies-pii")
    contract["optimizer"]["schedule"] = "linear"
    with pytest.raises(RuntimeError, match="constant"):
        _contract_adamw_parameters(contract)


def test_wsd_scheduler_contract_trace_and_fixed_step_stop_are_exact() -> None:
    import torch

    from meddies_pii.training.bioes.trainers import full_run
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _apply_optimizer_step,
        _build_scheduler,
        _contract_scheduler_parameters,
        _require_fresh_scheduler_run,
    )

    scout = full_run.render_pii350_scout_contract("wsd3e-4")
    scheduler_parameters = _contract_scheduler_parameters(scout)
    assert scheduler_parameters is not None
    assert asdict(scheduler_parameters) == {
        "peak_lr": 3e-4,
        "total_steps": 60,
        "warmup_steps": 6,
        "stable_steps": 48,
        "decay_steps": 6,
        "warmup_type": "linear",
        "decay_type": "cosine",
        "min_lr_ratio": 0.0,
    }

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=scheduler_parameters.peak_lr)
    scheduler = _build_scheduler(optimizer, scheduler_parameters)
    trace = [_apply_optimizer_step(optimizer, scheduler) for _ in range(60)]
    assert trace[0]["scheduler"] == "wsd"
    assert trace[0]["lr"] == 0.0
    assert trace[0]["next_lr"] == pytest.approx(5e-5)
    assert trace[5]["lr"] == pytest.approx(2.5e-4)
    assert trace[5]["next_lr"] == pytest.approx(3e-4)
    assert trace[6]["lr"] == pytest.approx(3e-4)
    assert trace[6]["next_lr"] == pytest.approx(3e-4)
    assert trace[53]["lr"] == pytest.approx(3e-4)
    assert trace[53]["next_lr"] == pytest.approx(3e-4)
    assert trace[54]["lr"] == 3e-4
    assert trace[54]["next_lr"] == pytest.approx(2.799038105676658e-4)
    assert trace[59]["lr"] == pytest.approx(2.009618943233419e-5)
    assert trace[59]["next_lr"] == 0.0

    constant = full_run.render_pii350_scout_contract("lr3e-4")
    assert _contract_scheduler_parameters(constant) is None
    constant_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=3e-4)
    assert _apply_optimizer_step(constant_optimizer, None) == {"lr": 3e-4}
    _require_fresh_scheduler_run(None, resume_checkpoint="checkpoint")
    with pytest.raises(RuntimeError, match="cannot resume"):
        _require_fresh_scheduler_run(scheduler_parameters, resume_checkpoint="checkpoint")

    assert (
        next_step_reason(
            elapsed_seconds=0.0,
            predicted_next_step_seconds=1.0,
            training_deadline_seconds=10.0,
            epoch_complete=False,
            optimizer_step=59,
            optimizer_step_cap=60,
        )
        == "training"
    )
    assert (
        next_step_reason(
            elapsed_seconds=0.0,
            predicted_next_step_seconds=1.0,
            training_deadline_seconds=10.0,
            epoch_complete=False,
            optimizer_step=60,
            optimizer_step_cap=60,
        )
        == "optimizer_step_cap_reached"
    )

    invalid = full_run.render_pii350_scout_contract("wsd3e-4")
    invalid["optimizer"]["stable_steps"] = 47
    with pytest.raises(RuntimeError, match="disagree"):
        _contract_scheduler_parameters(invalid)

    invalid = full_run.render_pii350_scout_contract("wsd3e-4")
    invalid["optimizer"]["stable_steps"] = 47
    invalid["training"]["mechanics"]["scheduler"]["stable_steps"] = 47
    with pytest.raises(RuntimeError, match="sum"):
        _contract_scheduler_parameters(invalid)

    invalid = full_run.render_pii350_scout_contract("wsd3e-4")
    invalid["optimizer"]["schedule"] = "linear"
    with pytest.raises(RuntimeError, match="supported"):
        _contract_scheduler_parameters(invalid)


def _cuda_build_without_driver() -> bool:
    import torch

    return torch.version.cuda is not None and not torch.cuda.is_available()


@pytest.mark.skipif(
    _cuda_build_without_driver(),
    reason="faking cuda availability makes the optimizer health check "
    "initialize CUDA for real; CUDA-built torch without a driver raises",
)
@pytest.mark.parametrize("arm", ["wsd3e-4", "lr4e-4"])
def test_60_step_scout_runtime_persists_one_terminal_checkpoint_and_evaluates_once(
    arm: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from meddies_pii.training.bioes.trainers import full_run, full_run_runtime

    class _Backbone(torch.nn.Module):
        @staticmethod
        def save_pretrained(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)

    class _Tagger(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _Backbone()
            self.classifier = torch.nn.Linear(1, 37)

        @override
        def forward(self, **_batch: object) -> dict[str, torch.Tensor]:
            return {"loss": self.classifier.weight.square().mean()}

    class _Sampler:
        @staticmethod
        def start() -> None:
            return None

        @staticmethod
        def stop() -> None:
            return None

        @staticmethod
        def latest() -> dict[str, int]:
            return {"peak_nvml_bytes": 0}

    tagger = _Tagger()
    writer = RunWriter(tmp_path / "run")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    monkeypatch.setattr(
        full_run_runtime,
        "_build_tagger",
        lambda *_args, **_kwargs: (tagger, object(), {"runtime_backend": "unsloth"}),
    )
    monkeypatch.setattr(full_run_runtime, "_validate_runtime_label_vocabulary", lambda _contract: [])
    monkeypatch.setattr(full_run_runtime, "_persist_label_vocabulary", lambda *_args: None)
    monkeypatch.setattr(full_run_runtime, "_iter_packed_rows", lambda _paths: iter(()))
    monkeypatch.setattr(
        full_run_runtime,
        "_physical_batches",
        lambda _iterator, _batch_size: iter([[object()] * 128 for _ in range(61)]),
    )
    monkeypatch.setattr(full_run_runtime, "_packed_batch", lambda _rows, _device: ({}, 128))
    monkeypatch.setattr(full_run_runtime, "_packed_order_attestation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(full_run_runtime, "_NvmlSampler", _Sampler)
    evaluations: list[str] = []

    def evaluate_once(*_args: object, **_kwargs: object) -> dict[str, int]:
        evaluations.append("final")
        return {"rows": 1700}

    monkeypatch.setattr(full_run_runtime, "_final_evaluate", evaluate_once)

    result = full_run_runtime._execute_full_candidate(
        full_run.render_pii350_scout_contract(arm),
        shard_paths=(),
        writer=writer,
    )

    events = [json.loads(line) for line in writer.events_path.read_text().splitlines()]
    optimizer_events = [event for event in events if event["event"] == "optimizer_step"]
    assert [event["step"] for event in optimizer_events] == list(range(1, 61))
    if arm == "lr4e-4":
        assert all("scheduler" not in event and "next_lr" not in event for event in optimizer_events)
    assert result["lifecycle_state"] == "optimizer_step_cap_reached"
    assert result["optimizer_steps"] == 60
    assert result["packed_cursor"] == 7680
    assert evaluations == ["final"]
    assert [event["event"] for event in events].count("final_evaluation_started") == 1
    assert not any(event["event"].startswith("milestone_evaluation") for event in events)
    assert sum(event["event"] == "checkpoint_started" and event["step"] == 60 for event in events) == 1
    assert sum(event["event"] == "checkpoint_complete" and event["step"] == 60 for event in events) == 1
    cap_event = next(event for event in events if event["event"] == "optimizer_step_cap_reached")
    assert {
        key: cap_event[key]
        for key in (
            "event",
            "step",
            "packed_cursor",
            "training_deadline_seconds",
            "optimizer_step_cap",
        )
    } == {
        "event": "optimizer_step_cap_reached",
        "step": 60,
        "packed_cursor": 7680,
        "training_deadline_seconds": 6874.0,
        "optimizer_step_cap": 60,
    }
    metadata = json.loads((writer.root / "checkpoints" / "step-00000060" / "metadata.json").read_text())
    assert metadata["lifecycle_state"] == "optimizer_step_cap_reached"


def test_base230_comparison_runtime_attestation_uses_its_july_contract_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the batch-320 app's 2026.7.4 image rather than Base230 defaults."""
    import importlib.metadata
    from types import SimpleNamespace

    import torch

    from meddies_pii.training.bioes.trainers import full_run, full_run_runtime
    from meddies_pii.training.bioes.trainers.config import LORA_TARGET_MODULES

    comparison = full_run.render_m230_comparison_run_contract("base230")
    observed = {
        package.split("==", maxsplit=1)[0]: package.split("==", maxsplit=1)[1]
        for package in comparison["runtime"]["packages"]
    }

    class FastLanguageModel:
        pass

    class Model:
        config = SimpleNamespace(use_cache=False, _attn_implementation="unsloth_managed")
        is_gradient_checkpointing = True

        @staticmethod
        def named_parameters() -> list[tuple[str, SimpleNamespace]]:
            return [
                (
                    f"base_model.{target}.lora_A.default.weight",
                    SimpleNamespace(requires_grad=True, dtype=torch.float32),
                )
                for target in LORA_TARGET_MODULES
            ]

    tagger = SimpleNamespace(
        packed_segment_isolation=True,
        request_hidden_states=True,
        classifier=torch.nn.Linear(1, 37),
    )
    monkeypatch.setattr(importlib.metadata, "version", observed.__getitem__)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "H100")

    result = full_run_runtime._runtime_attestation(
        contract=comparison,
        fast_model=FastLanguageModel,
        model=Model(),
        tagger=tagger,
        candidate_key="base230",
    )
    assert result["installed_package_versions"] == observed

    monkeypatch.setitem(observed, "unsloth", "2026.5.2")
    with pytest.raises(RuntimeError, match="attestation mismatch"):
        full_run_runtime._runtime_attestation(
            contract=comparison,
            fast_model=FastLanguageModel,
            model=Model(),
            tagger=tagger,
            candidate_key="base230",
        )


def test_milestone_attestation_requires_exact_step_cursor_and_non_stopping_evaluation() -> None:
    from meddies_pii.training.bioes.trainers import full_run
    from meddies_pii.training.bioes.trainers.full_run_runtime import (
        _is_milestone_step,
        _require_milestone_state,
    )

    contract = full_run.render_pii350_milestone_run_contract("anhthunguyenump")
    milestone = contract["training"]["milestone"]
    state = RuntimeResumeState(
        candidate_key="pii350",
        config_digest=contract["config_digest"],
        packed_revision=full_run.PACKED_DATASET_REVISION,
        packed_manifest_sha256=full_run.PACKED_MANIFEST_SHA256,
        eval_revision=full_run.EVAL_DATASET_REVISION,
        packed_cursor=7_680,
        optimizer_step=48,
    )
    _require_milestone_state(contract, state)
    assert state.optimizer_step % contract["training"]["checkpoint_every_optimizer_steps"] != 0
    assert _is_milestone_step(contract, state) is True
    with pytest.raises(RuntimeError, match="cursor"):
        _require_milestone_state(contract, replace(state, packed_cursor=7_520))
    with pytest.raises(RuntimeError, match="step"):
        _require_milestone_state(contract, replace(state, optimizer_step=47))
    assert milestone["continues_training"] is True


def test_milestone_evaluation_restores_training_and_emits_completion_or_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meddies_pii.training.bioes.trainers import full_run, full_run_runtime

    contract = full_run.render_pii350_milestone_run_contract("meddies-pii")
    state = RuntimeResumeState(
        candidate_key="pii350",
        config_digest=contract["config_digest"],
        packed_revision=full_run.PACKED_DATASET_REVISION,
        packed_manifest_sha256=full_run.PACKED_MANIFEST_SHA256,
        eval_revision=full_run.EVAL_DATASET_REVISION,
        packed_cursor=7_680,
        optimizer_step=40,
    )

    class _Tagger:
        training = True

        # reason: _evaluate_milestone calls tagger.train(was_training) positionally, so the double must accept it.
        def train(
            self,
            enabled: bool = True,  # ruff: ignore[boolean-type-hint-positional-argument,boolean-default-value-positional-argument]
        ) -> None:
            self.training = enabled

    writer = RunWriter(tmp_path / "run")

    class _EvaluationTagger:
        training: bool

    def evaluate_for_test(
        tagger: _EvaluationTagger,
        _tokenizer: object,
        *,
        progress: Callable[[Mapping[str, object]], None],
        progress_event: str = "final_evaluation_progress",
    ) -> dict[str, object]:
        tagger.training = False
        progress({"event": progress_event, "processed_rows": 1700})
        return {"rows": 1700, "exact_span": {"f1": 1.0}}

    monkeypatch.setattr(full_run_runtime, "_final_evaluate", evaluate_for_test)
    tagger = _Tagger()
    result = full_run_runtime._evaluate_milestone(contract, state, tagger, object(), writer)
    assert result["rows"] == 1700
    assert tagger.training is True
    events = [json.loads(line) for line in writer.events_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "milestone_evaluation_started",
        "milestone_evaluation_progress",
        "milestone_evaluation_completed",
    ]
    assert not any(event["event"].startswith("final_evaluation_") for event in events)
    assert events[-1]["metrics"]["exact_span"]["f1"] == 1.0

    failed_writer = RunWriter(tmp_path / "failed-run")
    monkeypatch.setattr(
        full_run_runtime,
        "_final_evaluate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken eval")),
    )
    with pytest.raises(RuntimeError, match="broken eval"):
        full_run_runtime._evaluate_milestone(contract, state, _Tagger(), object(), failed_writer)
    failed_events = [json.loads(line) for line in failed_writer.events_path.read_text().splitlines()]
    assert failed_events[-1]["event"] == "milestone_evaluation_error"
