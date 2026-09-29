"""Hash-addressed provenance for the production regex runtime."""

from __future__ import annotations

import hashlib
import inspect
import json
from importlib import import_module
from typing import TYPE_CHECKING

from meddies_pii.regex_runtime import language, postprocess
from meddies_pii.regex_runtime import rules as regex_rules

if TYPE_CHECKING:
    from types import ModuleType


def regex_manifest() -> dict[str, object]:
    """Return the canonical, hash-addressed regex policy.

    Returns:
        The canonical regex policy payload with component and overall SHA-256 digests.

    """
    runtime = import_module("meddies_pii.regex_runtime")
    policy, pack_payloads = regex_rules.regex_manifest_inputs()
    rule_payload = policy["rules"]
    validator_payload = {rule.name: inspect.getsource(rule.validator) for rule in regex_rules.REGEX_RULES}
    components = {
        "rules": _canonical_sha256(rule_payload),
        "validators": _canonical_sha256(validator_payload),
        "normalization": _canonical_sha256(inspect.getsource(regex_rules.normalize_with_offsets)),
        "offset_map": _canonical_sha256(inspect.getsource(regex_rules.NormalizedText.original_bounds)),
        "boundary_policy": _canonical_sha256(policy["boundary"]),
        "applicability_policy": _canonical_sha256(policy["pack_applicability"]),
        "overlap_policy": _canonical_sha256({
            "regex": policy["regex_overlap"],
            "model": policy["model_overlap"],
        }),
        "packs": _canonical_sha256(pack_payloads),
        "rules_executable": _source_sha256(regex_rules),
        "postprocess_executable": _source_sha256(postprocess),
        "language_executable": _source_sha256(language),
        "runtime_executable": _source_sha256(runtime),
    }
    payload = {**policy, "component_sha256": components}
    return {**payload, "sha256": _canonical_sha256(payload)}


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _source_sha256(module: ModuleType) -> str:
    return hashlib.sha256(inspect.getsource(module).encode("utf-8")).hexdigest()
