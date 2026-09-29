from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import importlib.util
from pathlib import Path
from typing import cast, get_args


def test_pii_label_registry_drives_package_surfaces() -> None:
    from meddies_pii.annotations.bioes import ENTITY_LABELS
    from meddies_pii.constants import PII_LABELS as CONSTANTS_LABELS
    from meddies_pii.taxonomy import PII_LABEL_SET, PII_LABELS, PiiLabel
    from meddies_pii.training.bioes.data import corpus_manifest
    from meddies_pii.training.bioes.data.span_quality import LEADING_LABELS

    assert PII_LABELS == (
        "address",
        "company_name",
        "date",
        "email_address",
        "human_name",
        "id_number",
        "phone_number",
        "private_url",
        "secret",
    )
    assert CONSTANTS_LABELS == PII_LABELS
    assert get_args(PiiLabel) == PII_LABELS
    assert frozenset(PII_LABELS) == PII_LABEL_SET
    assert tuple(sorted(PII_LABELS)) == ENTITY_LABELS
    assert tuple(corpus_manifest.LABELS_VALID) == PII_LABELS
    assert LEADING_LABELS == PII_LABELS


def test_language_registry_covers_public_names_aliases_and_grid_codes() -> None:
    from meddies_pii.constants import SUPPORTED_LANGUAGES
    from meddies_pii.languages import (
        GRID_LANGUAGE_CODES,
        LANGUAGE_PROFILES,
        is_supported_meddies_language,
        language_bucket,
        normalize_language,
    )

    assert tuple(profile.name for profile in LANGUAGE_PROFILES.values()) == (SUPPORTED_LANGUAGES)
    assert tuple(profile.code for profile in LANGUAGE_PROFILES.values()) == GRID_LANGUAGE_CODES
    assert len(GRID_LANGUAGE_CODES) == 17
    assert normalize_language("Vietnamese").code == "vi"
    assert normalize_language("vie").code == "vi"
    assert normalize_language("jpn").code == "ja"
    assert is_supported_meddies_language("fil")
    assert language_bucket(language="UNKNOWN", source="vietnamese-translated") == "vi"
    assert language_bucket(language="uk", source="") == "en"
    assert language_bucket(language="zh", source="") == "other"


def test_script_language_surfaces_are_registry_derived() -> None:
    from meddies_pii.languages import LANGUAGE_PROFILES, MEDDIES_PII_LANGUAGE_CONFIGS
    from meddies_pii.training.bioes.data.build_legacy_pii_label_corpus import (
        MEDDIES_PII_LANGUAGE_CONFIGS as BUILD_LANGUAGE_CONFIGS,
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generation" / "generate_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_data_for_registry_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    default_language_keys = cast("tuple[str, ...]", module.DEFAULT_LANGUAGE_KEYS)
    resolve_languages = module.resolve_languages

    assert default_language_keys == tuple(LANGUAGE_PROFILES)
    assert resolve_languages(None) == tuple(LANGUAGE_PROFILES)
    assert resolve_languages(["English", "jpn"]) == ("english", "japanese")
    assert BUILD_LANGUAGE_CONFIGS == MEDDIES_PII_LANGUAGE_CONFIGS
