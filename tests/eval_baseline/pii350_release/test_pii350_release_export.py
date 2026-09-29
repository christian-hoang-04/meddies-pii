from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch lands, or an optional wheel is skipped, before the symbol is bound.
# ruff: file-ignore[docstring-missing-returns]
# reason: documentation debt accepted here: these are operational scripts, archived experiments, and tests, not the
# reason: shipped package. A generated `Returns:` line would restate the summary without adding information, so the gap
# reason: stays visible instead.
# ruff: file-ignore[no-self-use]
# reason: stateless test doubles retain the bound method shape of import and extractor interfaces they replace.
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from meddies_pii.annotations.bioes import (
    ENTITY_LABELS,
    build_bioes_label_space,
    decode_bioes_from_offsets,
    viterbi_decode_logits,
)


def _release() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "ops" / "export_pii350_release.py"
    spec = importlib.util.spec_from_file_location("pii350_release_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bioes_label_map_round_trips_to_the_fixed_nine_labels() -> None:
    """Every entity contributes exactly the four BIOES prefixes."""
    release = _release()

    id_to_tag = release.bioes_label_map()

    assert len(id_to_tag) == 37
    assert id_to_tag[0] == "O"
    assert release.entity_label_for_tag("O") is None
    recovered = sorted({entity for tag in id_to_tag.values() if (entity := release.entity_label_for_tag(tag)) is not None})
    assert recovered == sorted(ENTITY_LABELS)
    assert len(recovered) == 9
    for entity in recovered:
        assert {f"B-{entity}", f"I-{entity}", f"E-{entity}", f"S-{entity}"} <= set(id_to_tag.values())
    config = release.release_label_config()
    assert config["num_labels"] == 37
    assert config["entity_labels"] == sorted(ENTITY_LABELS)
    assert config["label2id"][config["id2label"]["36"]] == 36
    with pytest.raises(ValueError, match="not a BIOES label"):
        release.entity_label_for_tag("X-human_name")


def _write_checkpoint_manifest(root: Path, embedded_digest: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "committed": True,
            "files": [{"path": "metadata.json", "bytes": 2, "sha256": "0" * 64}],
            "checkpoint_digest": embedded_digest,
        }),
        encoding="utf-8",
    )


def test_packaging_refuses_a_checkpoint_whose_digest_does_not_match(
    tmp_path: Path,
) -> None:
    release = _release()
    cache_root = tmp_path / "cache"
    artifact_root = cache_root / release.STEP250_ARTIFACT_PATH
    _write_checkpoint_manifest(artifact_root, "c" * 64)

    with pytest.raises(RuntimeError, match="does not match requested checkpoint"):
        release.verified_checkpoint_root(cache_root)


def test_packaging_binds_the_pinned_step250_identity(tmp_path: Path) -> None:
    """The pinned digest reaches the manifest gate, which then fails on the files."""
    release = _release()
    cache_root = tmp_path / "cache"
    artifact_root = cache_root / release.STEP250_ARTIFACT_PATH
    _write_checkpoint_manifest(artifact_root, release.STEP250_CHECKPOINT_DIGEST)

    with pytest.raises(RuntimeError, match="failed hash verification"):
        release.verified_checkpoint_root(cache_root)
    assert release.STEP250_REVISION == "ceef989794477051a6c17b06d75e17c3b442041c"
    assert len(release.STEP250_CHECKPOINT_DIGEST) == 64


class _BlockInstalledMeddiesPii:
    """Make `meddies_pii` unimportable so a leaked import cannot pass silently.

    The release payload runs on machines that do not have this repo installed.
    Without this block the vendored modules happily resolve `meddies_pii.*` from
    the local site-packages and a broken rewrite still reads as green.
    """

    def find_spec(self, fullname: str, _path: object = None, _target: object = None) -> None:
        if fullname == "meddies_pii" or fullname.startswith("meddies_pii."):
            msg = f"meddies_pii is blocked for the standalone payload check: {fullname}"
            raise ModuleNotFoundError(msg)


def _import_vendored(
    tmp_path: Path,
    release: ModuleType,
    module_names: tuple[str, ...],
) -> dict[str, ModuleType]:
    """Materialise the shipped decode package and import it standalone."""
    payload = release.vendored_decode_payload()
    for relative, content in payload.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    package = release.VENDORED_DECODE_PACKAGE
    cached = {
        name: module
        for name, module in sys.modules.items()
        if name in {package, "meddies_pii"} or name.startswith((f"{package}.", "meddies_pii."))
    }
    for name in cached:
        del sys.modules[name]
    blocker = _BlockInstalledMeddiesPii()
    sys.meta_path.insert(0, blocker)
    sys.path.insert(0, str(tmp_path))
    try:
        return {name: importlib.import_module(f"{package}.{name}") for name in module_names}
    finally:
        sys.path.remove(str(tmp_path))
        sys.meta_path.remove(blocker)
        for name in [name for name in sys.modules if name == package or name.startswith(f"{package}.")]:
            del sys.modules[name]
        sys.modules.update(cached)


# reason: Vocabulary, logits, offsets, and three decoders share one fixture for exact release-parity comparison.
def test_shipped_decode_matches_the_evaluated_decode_on_synthetic_logits(  # ruff: ignore[too-many-locals]
    tmp_path: Path,
) -> None:
    """The release postprocess entry point agrees with both."""
    import torch

    release = _release()
    vocabulary = build_bioes_label_space(ENTITY_LABELS)
    id_to_label = dict(enumerate(vocabulary))
    text = "Anna Lee emailed anna@example.com about the file."
    offsets = [(0, 4), (5, 8), (9, 16), (17, 32), (33, 38), (39, 43), (44, 48)]
    single = vocabulary.index("S-human_name")
    begin = vocabulary.index("B-human_name")
    end = vocabulary.index("E-human_name")
    email = vocabulary.index("S-email_address")
    preferred = [begin, end, 0, email, 0, 0, 0]
    logits = torch.full((len(offsets), len(vocabulary)), -6.0)
    for position, label_id in enumerate(preferred):
        logits[position][label_id] = 9.0
    logits[0][single] = 4.0

    expected_tags = viterbi_decode_logits(logits, id_to_label, offsets)
    expected = list(decode_bioes_from_offsets(text, offsets, expected_tags, id_to_label))
    assert expected, "fixture must decode at least one span for the test to bite"

    vendored = _import_vendored(tmp_path, release, ("viterbi", "bioes_spans"))
    shipped_tags = vendored["viterbi"].viterbi_decode_logits(logits, id_to_label, offsets)
    shipped = list(vendored["bioes_spans"].decode_bioes_from_offsets(text, offsets, shipped_tags, id_to_label))

    assert shipped_tags == expected_tags
    assert [(span.start, span.end, span.label) for span in shipped] == [
        (span.start, span.end, span.label) for span in expected
    ]
    through_release = release.decode_spans_from_logits(text, logits, offsets, id_to_label)
    assert [(span.start, span.end, span.label) for span in through_release] == [
        (span.start, span.end, span.label) for span in expected
    ]


def test_vendoring_only_rewrites_the_import_prefixes() -> None:
    release = _release()
    payload = release.vendored_decode_payload()
    root = release.package_source_root()

    for relative, vendored_name in release.VENDORED_DECODE_MODULES:
        original = (root / relative).read_text(encoding="utf-8")
        shipped = payload[f"{release.VENDORED_DECODE_PACKAGE}/{vendored_name}"].decode("utf-8")
        assert shipped == release.rewrite_vendored_source(original)
        differing = [
            line
            for original_line, line in zip(original.splitlines(), shipped.splitlines(), strict=True)
            if original_line != line
        ]
        assert all(line.startswith("from ") for line in differing)
    assert "meddies_pii." not in payload[f"{release.VENDORED_DECODE_PACKAGE}/bioes_spans.py"].decode("utf-8")


class _FakeHub:
    """The identically named dataset repo must never receive release files."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.objects: dict[str, bytes] = {}
        self._commits = 0

    def upload_file(
        self,
        *,
        path_or_fileobj: object,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> SimpleNamespace:
        assert repo_type == "model"
        assert repo_id.startswith("Meddies/meddies-pii-v2")
        assert commit_message
        payload = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(str(path_or_fileobj)).read_bytes()
        self.order.append(path_in_repo)
        self.objects[path_in_repo] = payload
        self._commits += 1
        return SimpleNamespace(oid=f"{self._commits:040x}")


def test_release_upload_commits_the_manifest_last(tmp_path: Path) -> None:
    release = _release()
    payload_root = tmp_path / "payload"
    (payload_root / "meddies_pii_decode").mkdir(parents=True)
    (payload_root / "config.json").write_text('{"a": 1}', encoding="utf-8")
    (payload_root / "model.safetensors").write_bytes(b"weights")
    (payload_root / "meddies_pii_decode" / "viterbi.py").write_bytes(b"decode")
    hub = _FakeHub()

    receipt = release.upload_release_tree(
        payload_root,
        repo_id=release.RELEASE_MODEL_REPO,
        kind=release.RELEASE_KIND_TORCH,
        api=hub,
    )

    assert hub.order[-1] == "manifest.json"
    assert len(hub.order) == 4
    assert receipt["repo_id"] == "Meddies/meddies-pii-v2"
    assert receipt["repo_type"] == "model"
    assert receipt["private"] is True
    manifest = receipt["manifest"]
    assert [entry["path"] for entry in manifest["files"]] == sorted(hub.order[:-1])
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    from meddies_pii.evaluation.identity import canonical_sha256

    assert manifest["manifest_digest"] == canonical_sha256(body)
    assert manifest["checkpoint_digest"] == release.STEP250_CHECKPOINT_DIGEST


def test_release_upload_refuses_an_empty_payload(tmp_path: Path) -> None:
    release = _release()
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RuntimeError, match="release payload is empty"):
        release.upload_release_tree(
            empty,
            repo_id=release.RELEASE_MODEL_REPO,
            kind=release.RELEASE_KIND_TORCH,
            api=_FakeHub(),
        )


def test_manifest_verification_catches_a_tampered_file(tmp_path: Path) -> None:
    release = _release()
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    (payload_root / "config.json").write_text('{"a": 1}', encoding="utf-8")
    hub = _FakeHub()
    receipt = release.upload_release_tree(
        payload_root,
        repo_id=release.RELEASE_MODEL_REPO,
        kind=release.RELEASE_KIND_TORCH,
        api=hub,
    )

    assert release.verify_manifest_tree(payload_root, receipt["manifest"]) == ["config.json"]
    (payload_root / "config.json").write_text('{"a": 2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed hash verification"):
        release.verify_manifest_tree(payload_root, receipt["manifest"])
    (payload_root / "config.json").unlink()
    with pytest.raises(RuntimeError, match="release file is absent"):
        release.verify_manifest_tree(payload_root, receipt["manifest"])


def test_release_keeps_adapter_form_as_the_verified_default() -> None:
    """Merging bakes W+BA and rounds the LoRA delta.

    The merged file ships as a convenience form, so the release kind, dtype, and gate stay adapter-form.

    """
    release = _release()

    assert release.RUNTIME_DTYPE == "bfloat16"
    assert release.RELEASE_KIND_TORCH == "adapter_form_release"
    assert release.RELEASE_KIND_ONNX == "adapter_form_onnx"
    assert release.MERGED_WEIGHTS_FILENAME == "model.safetensors"
    assert release.MERGED_WEIGHTS_FORMAT == "meddies-pii-v2-merged"


def test_r1_receipt_stages_between_distinct_roots_before_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = _release()
    local_root = tmp_path / "local"
    remote_root = tmp_path / "remote-release"
    monkeypatch.setattr(release, "RELEASE_MOUNT", str(remote_root))

    class FakeVolume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setattr(release, "release_volume", volume)
    source = release.SourceArtifactIdentity(
        repository="modal-volume://meddies-pii350-release",
        revision="f" * 64,
        path="meddies-pii-v2-onnx/model.onnx",
        bytes=1,
        sha256="1" * 64,
    )
    pending = release.build_r1_preflight(
        source=source,
        exporter_commit="a" * 40,
        estimated_spend_usd=0.01,
        approval_link="evidence://ha/r1-pending-20260806",
        approval_status="pending",
    )
    pending_path = release.write_r1_preflight_receipt(pending, local_root / "preflight")
    with pytest.raises(RuntimeError, match="not approved"):
        release.stage_r1_preflight_receipt.local(pending_path.read_bytes(), pending["receipt_sha256"])
    assert not remote_root.exists()

    tampered = json.loads(pending_path.read_text(encoding="utf-8"))
    tampered["modal"]["memory_mib"] = 65_536
    with pytest.raises(RuntimeError, match="receipt content"):
        release.stage_r1_preflight_receipt.local(json.dumps(tampered).encode("utf-8"), pending["receipt_sha256"])
    assert not remote_root.exists()

    approved = release.build_r1_preflight(
        source=source,
        exporter_commit="a" * 40,
        estimated_spend_usd=0.01,
        approval_link="evidence://ha/r1-approved-20260806",
        approval_status="approved",
    )
    approved_path = release.write_r1_preflight_receipt(approved, local_root / "preflight")
    assert release._stage_r1_command(approved_path, approved["receipt_sha256"]) == (
        "MODAL_PROFILE=huyhoang041100 uv run modal run "
        "scripts/ops/export_pii350_release.py --action stage_r1_preflight_receipt "
        f"--local-preflight-receipt-path {approved_path} "
        f"--expected-receipt-sha256 {approved['receipt_sha256']}"
    )
    staged = release.stage_r1_preflight_receipt.local(approved_path.read_bytes(), approved["receipt_sha256"])
    remote_receipt = remote_root / staged["remote_path"]
    assert approved_path.is_file()
    assert remote_receipt.is_file()
    assert not approved_path.is_relative_to(remote_root)
    assert not remote_receipt.is_relative_to(local_root)
    assert staged["model_bytes_transferred"] == 0
    assert release._export_r1_command(staged["remote_path"], staged["receipt_sha256"]) == (
        "MODAL_PROFILE=huyhoang041100 uv run modal run "
        "scripts/ops/export_pii350_release.py --action export_r1_q8_head_preserved "
        f"--preflight-receipt-path {staged['remote_path']} "
        f"--expected-receipt-sha256 {staged['receipt_sha256']} --no-upload"
    )

    def fake_export(
        source_path: Path,
        output_path: Path,
        manifest_path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert source_path == remote_root / source.path
        assert output_path.is_relative_to(remote_root)
        assert manifest_path.is_relative_to(remote_root)
        return {
            "source": approved["source"],
            "output": {
                "path": output_path.name,
                "bytes": 123,
                "sha256": "2" * 64,
            },
            "manifest_digest": "3" * 64,
            "classifier_proof": {"node": "/classifier/MatMul"},
            "nodes": {
                "included": ["/backbone/MatMul"],
                "excluded": ["/classifier/MatMul"],
            },
            "determinism": {"rerun_identical": True},
        }

    monkeypatch.setattr(release, "export_head_preserved_q8", fake_export)
    exported = release.export_r1_q8_head_preserved.local(staged["remote_path"], staged["receipt_sha256"])
    assert exported["status"] == "exported_r1_q8_head_preserved"
    assert exported["preflight_receipt_sha256"] == approved["receipt_sha256"]
    assert volume.commits == 2
    with pytest.raises(RuntimeError, match="SHA-256 is invalid"):
        release.export_r1_q8_head_preserved.local("approved", "approved")


def test_required_payload_files_name_the_merged_weights_and_the_licence() -> None:
    """The verbatim evaluated bytes remain required alongside the merged form."""
    release = _release()

    plan = release.required_payload_files()

    assert release.MERGED_WEIGHTS_FILENAME in plan
    assert release.LICENSE_FILENAME in plan
    assert "adapter/adapter_model.safetensors" in plan
    assert "adapter/adapter_config.json" in plan
    assert "classifier.pt" in plan
    assert "config.json" in plan
    assert release.REMOTE_CODE_FILENAME in plan
    assert f"{release.VENDORED_DECODE_PACKAGE}/viterbi.py" in plan
    assert len(set(plan)) == len(plan)


def test_payload_completeness_refuses_a_tree_missing_the_merged_weights(
    tmp_path: Path,
) -> None:
    release = _release()
    payload_root = tmp_path / "payload"
    plan = release.required_payload_files()
    for relative in plan:
        target = payload_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"present")

    assert release.verify_payload_complete(payload_root) == plan

    (payload_root / release.MERGED_WEIGHTS_FILENAME).unlink()
    (payload_root / release.LICENSE_FILENAME).unlink()
    with pytest.raises(RuntimeError, match="missing required files") as failure:
        release.verify_payload_complete(payload_root)
    assert release.MERGED_WEIGHTS_FILENAME in str(failure.value)
    assert release.LICENSE_FILENAME in str(failure.value)


def test_published_parity_record_carries_the_merged_delta_without_gating_it() -> None:
    """A nonzero merged delta is published, never raised on.

    Adapter form is the gated release and the merged file is the recorded convenience path.

    The record cannot silently omit the delta it exists to disclose.

    """
    release = _release()
    probe = {
        "rows": 200,
        "mismatched_rows": 0,
        "identical": True,
        "batched_reference_mismatched_rows": 9,
        "merged_mismatched_rows": 11,
    }

    record = release.release_parity_record(probe)

    assert record == {
        "rows": 200,
        "reference_shape": "batch_of_1",
        "mismatched_rows": 0,
        "batched_reference_mismatched_rows": 9,
        "merged_mismatched_rows": 11,
    }
    assert record["merged_mismatched_rows"] == 11
    with pytest.raises(KeyError, match="merged_mismatched_rows"):
        release.release_parity_record({key: value for key, value in probe.items() if key != "merged_mismatched_rows"})


def _loadremote_code_module_source(tmp_path: Path, release: ModuleType) -> ModuleType:
    """Materialise the shipped loader beside its vendored decode and import it.

    The template puts its own directory on sys.path; tmp_path dies with the test, so leaving either that entry or its
    modules behind poisons later ones.

    """
    for relative, content in release.vendored_decode_payload().items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    module_path = tmp_path / release.REMOTE_CODE_FILENAME
    module_path.write_text(release.remote_code_module_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("meddies_pii_v2_loader_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    package = release.VENDORED_DECODE_PACKAGE
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = [entry for entry in sys.path if entry != str(tmp_path)]
        for name in [name for name in sys.modules if name == package or name.startswith(f"{package}.")]:
            del sys.modules[name]
    return module


def test_shipped_loader_rejects_an_unknown_weights_form(tmp_path: Path) -> None:
    """tmp_path holds no config.json.

    So a ValueError here also proves the form is validated before the loader touches the release tree.

    """
    release = _release()
    loader = _loadremote_code_module_source(tmp_path, release)

    with pytest.raises(ValueError, match="weights must be"):
        loader.MeddiesPiiExtractor.from_pretrained(str(tmp_path), weights="fp8")


def test_merged_weights_round_trip_into_the_namespaces_the_loader_reads(
    tmp_path: Path,
) -> None:
    """The published loader must read back exactly what packaging wrote.

    So the split is checked with the shipped helper rather than a restatement of it.

    The head's Linear shape is recoverable from the tensor alone, which is how the merged branch rebuilds it without
    reading hidden_size from config.

    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    release = _release()
    tagger = SimpleNamespace(
        backbone=torch.nn.Linear(4, 3, dtype=torch.bfloat16),
        classifier=torch.nn.Linear(3, 2, dtype=torch.bfloat16),
    )
    target = tmp_path / release.MERGED_WEIGHTS_FILENAME

    written = release._save_merged_weights(tagger, target)

    with safe_open(str(target), framework="pt") as handle:
        metadata = handle.metadata()
        keys = sorted(handle.keys())
    assert metadata["format"] == release.MERGED_WEIGHTS_FORMAT
    assert metadata["dtype"] == release.RUNTIME_DTYPE
    assert keys == [
        "classifier.bias",
        "classifier.weight",
        "encoder.bias",
        "encoder.weight",
    ]
    assert written == {"tensors": 4, "bytes": target.stat().st_size}

    loader = _loadremote_code_module_source(tmp_path / "loader", release)
    state = load_file(str(target))
    encoder = loader._namespace(state, release.MERGED_ENCODER_PREFIX)
    head = loader._namespace(state, release.MERGED_CLASSIFIER_PREFIX)
    assert sorted(encoder) == ["bias", "weight"]
    assert sorted(head) == ["bias", "weight"]
    assert torch.equal(encoder["weight"], tagger.backbone.weight)
    assert torch.equal(head["bias"], tagger.classifier.bias)
    assert head["weight"].shape == (2, 3)


def test_remote_code_merged_branch_skips_peft_and_the_base_weights() -> None:
    """The merged branch builds the skeleton from config alone.

    From_pretrained would pull the multi-hundred-megabyte base weights the merged file replaces.

    """
    release = _release()
    module = release.remote_code_module_source()

    assert 'weights == "merged"' in module
    assert "AutoConfig.from_pretrained" in module
    assert "from_config" in module
    assert f'load_file(str(root / "{release.MERGED_WEIGHTS_FILENAME}"))' in module
    assert release.MERGED_ENCODER_PREFIX in module
    assert release.MERGED_CLASSIFIER_PREFIX in module
    assert "strict=True" in module
    compile(module, "modeling_meddies_pii.py", "exec")


def test_remote_code_loads_base_and_applies_the_adapter_without_merging() -> None:
    """Merging happens once at packaging time.

    The load path never merges: the adapter form applies the adapter live, the merged form reads baked weights.

    The forward pass must mirror the evaluated tagger's backbone call exactly.

    The base weights are referenced by pinned revision, never redistributed.

    """
    release = _release()
    module = release.remote_code_module_source()

    assert "merge_and_unload" not in module
    assert 'weights="adapter"' in module
    assert "PeftModel.from_pretrained" in module
    assert "class MeddiesPiiExtractor" in module
    for kwarg in ("use_cache=False", "return_dict=True", "output_hidden_states=False"):
        assert kwarg in module
    assert "last_hidden_state" in module
    assert f"from {release.VENDORED_DECODE_PACKAGE}.viterbi import" in module
    assert 'release["base_model_id"]' in module
    assert 'revision=release["base_model_revision"]' in module
    compile(module, "modeling_meddies_pii.py", "exec")


def test_published_merged_spans_load_the_merged_form_on_cuda(tmp_path: Path) -> None:
    """Verification must exercise the merged branch.

    Not re-run the adapter default, and must leave dtype to the config so the arithmetic matches packaging.

    """
    release = _release()
    from meddies_pii.spans import CharSpan

    calls: list[dict[str, object]] = []

    class _Extractor:
        @classmethod
        def from_pretrained(
            cls,
            path: object,
            dtype: object = None,
            device: str = "cpu",
            weights: str = "adapter",
        ) -> _Extractor:
            calls.append({
                "path": path,
                "dtype": dtype,
                "device": device,
                "weights": weights,
            })
            return cls()

        def extract(self, text: str) -> list[CharSpan]:
            return [CharSpan(0, len(text), text, "human_name")]

    loader = SimpleNamespace(MeddiesPiiExtractor=_Extractor)

    spans = release.published_merged_spans(loader, tmp_path, ["Anna", "Lee Minh"])

    assert calls == [
        {
            "path": str(tmp_path),
            "dtype": None,
            "device": "cuda",
            "weights": "merged",
        },
    ]
    assert [[(span.start, span.end) for span in row] for row in spans] == [
        [(0, 4)],
        [(0, 8)],
    ]


def test_merged_parity_reconciles_against_the_recorded_delta() -> None:
    release = _release()
    config = {"meddies_release": {"parity": {"merged_mismatched_rows": 11}}}

    record = release.reconcile_merged_parity(config, {"rows": 200, "mismatched_rows": 11})

    assert record == {
        "rows": 200,
        "expected_mismatched_rows": 11,
        "observed_mismatched_rows": 11,
    }


def test_merged_parity_fails_closed_when_the_published_weights_drift() -> None:
    """Same shapes, dtype, and device on both sides, so the counts are deterministic.

    An unequal count means the published bytes are not the ones packaging measured.

    A config carrying no recorded delta must not pass by defaulting to zero.

    """
    release = _release()
    config = {"meddies_release": {"parity": {"merged_mismatched_rows": 11}}}

    with pytest.raises(RuntimeError, match="reproduce the recorded span delta") as drift:
        release.reconcile_merged_parity(config, {"rows": 200, "mismatched_rows": 12})
    assert "11" in str(drift.value)
    assert "12" in str(drift.value)

    with pytest.raises(KeyError, match="merged_mismatched_rows"):
        release.reconcile_merged_parity({"meddies_release": {"parity": {}}}, {"rows": 200, "mismatched_rows": 0})


def test_onnxpostprocess_module_source_is_importable_source() -> None:
    release = _release()
    module = release.postprocess_module_source()

    assert f"from {release.VENDORED_DECODE_PACKAGE}.viterbi import" in module
    compile(module, "meddies_pii_postprocess.py", "exec")
    compile(release.usage_snippet_source(), "usage_onnxruntime.py", "exec")


def test_span_comparison_reports_mismatched_rows() -> None:
    release = _release()
    from meddies_pii.spans import CharSpan

    left = [[CharSpan(0, 4, "Anna", "human_name")], []]
    same = [[CharSpan(0, 4, "Anna", "human_name")], []]
    different = [[CharSpan(0, 3, "Ann", "human_name")], []]

    assert release.compare_span_sets(left, same)["identical"] is True
    result = release.compare_span_sets(left, different)
    assert result["identical"] is False
    assert result["mismatched_rows"] == 1
    assert result["mismatches"][0]["row"] == 0
