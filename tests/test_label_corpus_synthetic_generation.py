"""Closes the 17-language gap.

Prove generate_until GENERATES + ACCEPTS a non-Latin (Thai) document end-to-end — the native-script marker check
(generalized from the vi-only guard) must pass for Thai raw text.

ADR 0008 §5: the dedup seed must also see previously-migrated pm_accepted.* rows, otherwise a cross-run duplicate of an
already-accepted text re-enters the corpus invisibly.

3 cells/round: english (private_url only) + vietnamese (private_url + secret). Simulate the daily token budget running out
after 5 generating calls.

The core of the parallel design: when one provider exhausts its budget it stops immediately, and the others keep generating
— a slow/exhausted provider never gates the rest.

A slow/stuck provider (here groq sleeping 30s) must not hang the run: the hard wall-clock deadline cancels the in-flight
worker so the run still finishes promptly. This is what makes keeping a 1-rpm provider (llm7) safe.

"""

from __future__ import annotations

# ruff: file-ignore[import-outside-top-level]
# reason: the import is deferred so a patch is in place first, and so collection does not pay for the heavy dependency.
import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, ClassVar, cast

import httpx
import pytest

from meddies_pii.exceptions import DailyBudgetExceeded
from meddies_pii.generation.label_corpus.catalog import (
    SCENARIOS,
    generation_profile_for_domain,
    required_labels_for_mode,
    scenario_pool_for_profile,
)
from meddies_pii.generation.label_corpus.generation_runs import (
    WEAK_LABEL_FREE_PROVIDERS,
    build_budget,
    run_eval_gold,
    run_weak_labels,
)
from meddies_pii.generation.label_corpus.prompts import (
    targeted_system_prompt,
    targeted_user_prompt,
)
from meddies_pii.generation.label_corpus.runner import (
    SyntheticGenerationRequest,
    run_synthetic_generation,
)
from meddies_pii.generation.label_corpus.synthetic import (
    LabelCorpusGenerator,
    _existing_text_hashes,
)
from meddies_pii.generation.label_corpus.validate import (
    accepted_record,
    validate_tagged_document,
)
from meddies_pii.historical_artifacts import (
    LEGACY_LABEL_POLICY,
    LEGACY_SYNTHETIC_DATASET_IDS,
)
from meddies_pii.json_types import is_str_mapping
from meddies_pii.languages import normalize_language

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from meddies_pii.generation.label_corpus.catalog import Scenario
    from meddies_pii.generation.label_corpus.runner import ClientFactory
    from meddies_pii.generation.openai_compatible.client import OpenAICompatibleClient


def _mapping_field(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a summary field the caller reads by key, asserting it really is a nested mapping.

    `run_synthetic_generation` and `run_eval_gold` both return `dict[str, object]`, so the
    nested `summaries` block and the recorded call kwargs arrive as `object`.

    Returns:
        The named field, narrowed to a mapping the caller can read by string key.

    """
    value = payload[key]
    assert is_str_mapping(value), f"{key} must be a mapping, got {type(value).__name__}"
    return value


def _scenario(name: str) -> Scenario:
    return next(s for s in SCENARIOS if s.name == name)


def test_validate_accepts_private_url_secret_pii_label_spans() -> None:
    content = """Bệnh viện [Phòng khám An Tâm]<company_name>
Bệnh nhân [Nguyễn Thị Mai]<human_name>, MRN [MRN-VN-884422]<id_number>, sinh ngày [12/04/1982]<date>.
Địa chỉ [18/7 Nguyễn Trãi, phường Bến Thành, Quận 1, TP.HCM]<address>; điện thoại [0903 445 778]<phone_number>.
Email [mai.nguyen at example dot vn]<email_address> nhận link kết quả [https://portal.example.vn/patients/MRN-VN-884422/lab-results?token=abc-123]<private_url>.
OTP đăng nhập [834-229]<secret>, cookie phiên [session=vn_9f88aa77]<secret>.
Tái khám [15/05/2026 08:30]<date> với bác sĩ [Trần Quốc Bảo]<human_name> tại [Bệnh viện Hòa Bình]<company_name>.
Thông tin công khai: hướng dẫn Bộ Y tế cập nhật ngày 01/01/2024 tại \
https://example.gov.vn/guidelines không cần gắn nhãn."""

    result = validate_tagged_document(
        content,
        language="Vietnamese",
        required_labels=("private_url", "secret"),
    )

    assert result.ok, result.errors
    assert {span.label for span in result.spans} >= {"private_url", "secret"}
    assert all(result.raw_text[span.start : span.end] == span.text for span in result.spans)
    assert "]<" not in result.raw_text


def test_validate_rejects_missing_private_url_and_secret() -> None:
    content = """Patient [Maya Green]<human_name> has MRN [MRN-123]<id_number> and appointment [05/13/2026]<date>.
Address [44 Lake Road, Boston, MA]<address>; phone [617-555-0199]<phone_number>.
Email [maya.green@example.com]<email_address> checked in at [Harbor Clinic]<company_name>.
Public guideline date 2024-01-01 at https://example.org/guidelines is untagged."""

    result = validate_tagged_document(
        content,
        language="English",
        required_labels=("private_url", "secret"),
    )

    assert not result.ok
    assert any(error.startswith("missing_required_labels:") for error in result.errors)


def test_validate_supports_private_url_only_general_scenario() -> None:
    content = """Support ticket [SUP-44219]<id_number> for [Maya Green]<human_name>.
Email [maya.green at example dot com]<email_address>; phone [617 555 0199]<phone_number>.
Shipping address [44 Lake Road, Boston, MA 02111]<address>.
Private invoice link [https://account.example.test/users/maya/invoices/44219?token=abc]<private_url>.
Public status page https://status.example.test and public date 2026-01-01 are untagged."""

    result = validate_tagged_document(
        content,
        language="English",
        required_labels=("private_url",),
    )

    assert result.ok, result.errors
    labels = {span.label for span in result.spans}
    assert "private_url" in labels
    assert "secret" not in labels


def test_required_label_modes_do_not_force_private_url_secret_together() -> None:
    scenario = _scenario("general_consumer_admin_support")

    assert required_labels_for_mode(scenario, "private_url_only") == (
        "human_name",
        "email_address",
        "id_number",
        "private_url",
    )
    assert required_labels_for_mode(scenario, "secret_only") == (
        "human_name",
        "email_address",
        "id_number",
        "secret",
    )
    assert required_labels_for_mode(scenario, "none") == ()


def test_validation_can_skip_scenario_required_labels_for_neither_mode() -> None:
    content = """Application audit line for [Maya Green]<human_name>.
Account [acct-4911]<id_number> changed email to [maya.green@example.com]<email_address>.
Recovery phone [617-555-0199]<phone_number> and billing address [44 Lake Road, Boston]<address> were confirmed.
Public status URL https://status.example.test and release date 2026-01-01 are intentionally untagged."""

    result = validate_tagged_document(
        content,
        language="English",
        required_labels=required_labels_for_mode(_scenario("code_log_security_account_recovery"), "none"),
    )

    assert result.ok, result.errors


def test_private_url_only_prompt_does_not_force_secret() -> None:
    scenario = _scenario("general_consumer_admin_support")
    prompt = targeted_user_prompt(
        profile=normalize_language("English"),
        scenario=scenario,
        document_type="support ticket",
        text_format="plain",
        min_spans=5,
        min_unique_labels=5,
        required_labels=required_labels_for_mode(scenario, "private_url_only"),
    )

    assert "<private_url>" in prompt
    assert "Always include at least one `<private_url>` and one `<secret>`" not in prompt
    assert "Required labels for this sample" in prompt
    assert "Required-label coverage before you answer" in prompt
    assert "private_url=present" in prompt
    assert "secret=present" not in prompt
    assert "Do not output a checklist" in prompt


def test_validate_rejects_leftover_label_markers_in_raw_text() -> None:
    """The stray `<address>` marker.

    Model forgot the `[...]`) is now stripped by the parser instead of poisoning the raw text — the rich doc is KEPT.

    """
    content = """Bệnh nhân [Nguyễn Thị Mai]<human_name>, MRN [MRN-VN-884422]<id_number>, sinh ngày [12/04/1982]<date>.
Địa chỉ 18/7 Nguyễn Trãi, TP.HCM<address>; điện thoại [0903 445 778]<phone_number>.
Email [mai.nguyen at example dot vn]<email_address> nhận link [https://portal.example.vn/patients/MRN-VN-884422/lab?token=abc]<private_url>.
OTP [834-229]<secret>, cookie [session=vn_9f88aa77]<secret>, nơi khám [Bệnh viện Hòa Bình]<company_name>.
Tái khám [15/05/2026 08:30]<date> với bác sĩ [Trần Quốc Bảo]<human_name>."""

    result = validate_tagged_document(
        content,
        language="Vietnamese",
        required_labels=("private_url", "secret"),
    )

    assert result.ok
    assert "leftover_label_marker" not in result.errors
    assert "<address>" not in result.raw_text


def test_validate_rejects_malformed_label_markers_in_raw_text() -> None:
    content = """Bệnh nhân [Nguyễn Thị Mai]<human_name>, MRN [MRN-VN-884422]<id_number>, sinh ngày [12/04/1982]<date>.
Địa chỉ [18/7 Nguyễn Trãi, TP.HCM]<address>; điện thoại [0903 445 778]<phone_number>.
Email [mai.nguyen at example dot vn]<email_address> nhận link [https://portal.example.vn/patients/MRN-VN-884422/lab?token=abc]<private_url>.
OTP [834-229]<secret>, cookie [session=vn_9f88aa77]<secret>, nơi khám [Bệnh viện Hòa Bình]<company_name>.
Tái khám [15/05/2026]<date] với bác sĩ [Trần Quốc Bảo]<human_name>."""

    result = validate_tagged_document(
        content,
        language="Vietnamese",
        required_labels=("private_url", "secret"),
    )

    assert not result.ok
    assert "malformed_leftover_label_marker" in result.errors


def test_validate_rejects_short_phone_fragments() -> None:
    """A 4-digit callback extension is a real hard case we WANT.

    The doc is kept (issue recorded but non-fatal) and the edge-case phone span is preserved.

    """
    content = """Support case [CASE-9901]<id_number> for [Maya Green]<human_name>.
Email [maya.green@example.com]<email_address> and callback extension [4417]<phone_number> were recorded.
Address [44 Lake Road, Boston]<address>.
Private account URL [https://account.example.test/recovery/CASE-9901?token=abc]<private_url>.
Temporary password [Temp-9901!]<secret>.
Public status page https://status.example.test is untagged."""

    result = validate_tagged_document(
        content,
        language="English",
        required_labels=("private_url", "secret"),
    )

    assert result.ok
    assert "short_phone_span" in result.errors
    assert any(span.text == "4417" for span in result.spans)


def test_validate_rejects_underbounded_phone_spans() -> None:
    """An under-bounded phone span is a minor boundary quirk, not a reason to drop the whole doc.

    Kept (non-fatal), issue recorded.

    """
    content = """Support case [CASE-9901]<id_number> for [Maya Green]<human_name>.
Email [maya.green@example.com]<email_address> and callback [916.442.8031]<phone_number>.3 were recorded.
Address [44 Lake Road, Boston]<address>.
Private account URL [https://account.example.test/recovery/CASE-9901?token=abc]<private_url>.
Temporary password [Temp-9901!]<secret>.
Public status page https://status.example.test is untagged."""

    result = validate_tagged_document(
        content,
        language="English",
        required_labels=("private_url", "secret"),
    )

    assert result.ok
    assert "underbounded_phone_span" in result.errors


def test_accepted_record_uses_pii_label_span_contract() -> None:
    content = """Patient [Maya Green]<human_name> MRN [MRN-123]<id_number> DOB [02/03/1980]<date>.
Address [44 Lake Road, Boston, MA 02111]<address>; phone [617-555-0199]<phone_number>; email [maya \
at example dot com]<email_address>.
Clinic [Harbor Clinic]<company_name> sent private result [https://portal.example.test/patients/MRN-123/results?token=abc]<private_url>.
Use OTP [443-221]<secret> and session cookie [session=abc123]<secret>. Follow-up with [Noah \
Patel]<human_name> on [05/13/2026 09:00]<date>.
Public article copyright 2024 at https://example.org/article is not patient PII."""
    validation = validate_tagged_document(
        content,
        language="English",
        required_labels=("private_url", "secret"),
    )
    assert validation.ok, validation.errors

    record = accepted_record(
        validation=validation,
        language="English",
        scenario=_scenario("private_portal_secret_focus"),
        document_type="portal note",
        text_format="plain",
        model="MiMo-V2.5-Pro",
        provider="mimo",
        attempt_index=7,
    )

    assert set(record) == {"text", "label", "info"}
    assert record["info"]["language_bucket"] == "en"
    assert record["info"]["label_policy"] == LEGACY_LABEL_POLICY
    assert record["label"][0].keys() == {"category", "start", "end", "text"}


def test_accepted_record_can_mark_general_challenge_metadata() -> None:
    content = """Support case [CASE-9901]<id_number> for [Maya Green]<human_name>.
Email [maya@example.com]<email_address>, phone [617-555-0199]<phone_number>, address [44 Lake Road, Boston]<address>.
Private account URL [https://account.example.test/cases/CASE-9901?token=abc]<private_url>.
Public status page https://status.example.test and public company blog date 2026-01-01 are untagged contrast examples."""
    validation = validate_tagged_document(
        content,
        language="English",
        required_labels=("private_url",),
    )
    assert validation.ok, validation.errors

    record = accepted_record(
        validation=validation,
        language="English",
        scenario=_scenario("general_consumer_admin_support"),
        document_type="support ticket",
        text_format="plain",
        model="MiMo-V2.5-Pro",
        provider="mimo",
        attempt_index=7,
        source_dataset=LEGACY_SYNTHETIC_DATASET_IDS["general"],
        domain_bucket="general",
        split_purpose="challenge",
    )

    assert record["info"]["source_dataset"] == LEGACY_SYNTHETIC_DATASET_IDS["general"]
    assert record["info"]["domain_bucket"] == "general"
    assert record["info"]["domain_profile"] == "general"
    assert record["info"]["split_purpose"] == "challenge"


def test_generation_profile_maps_domain_and_split_to_metadata() -> None:
    general = generation_profile_for_domain("general", split_purpose="train")
    code_logs = generation_profile_for_domain("code_logs", split_purpose="train")
    challenge = generation_profile_for_domain("general", split_purpose="challenge")

    assert general.source_dataset == LEGACY_SYNTHETIC_DATASET_IDS["general"]
    assert general.domain_bucket == "general"
    assert "general_consumer_admin_support" in general.scenario_names
    assert "public_negative_url_date_contrast" in general.scenario_names
    assert "public_negative_contrast" not in general.scenario_names
    assert code_logs.source_dataset == LEGACY_SYNTHETIC_DATASET_IDS["code_logs"]
    assert code_logs.domain_bucket == "code_logs"
    assert "structured_general_payload" in code_logs.scenario_names
    assert "structured_payload" not in code_logs.scenario_names
    assert "adversarial_obfuscation" not in code_logs.scenario_names
    assert challenge.source_dataset == LEGACY_SYNTHETIC_DATASET_IDS["challenge"]
    assert challenge.split_purpose == "challenge"


def test_scenario_pool_restricts_generation_to_requested_profile_scenarios() -> None:
    general = generation_profile_for_domain("general", split_purpose="train")

    pool = scenario_pool_for_profile(general, ("general_consumer_admin_support",))

    assert [scenario.name for scenario in pool] == ["general_consumer_admin_support"]
    with pytest.raises(ValueError, match="not enabled"):
        scenario_pool_for_profile(general, ("private_portal_secret_focus",))


def test_targeted_prompt_names_private_url_and_public_negative_rule() -> None:
    system = targeted_system_prompt(normalize_language("English"))
    assert "<private_url>" in system
    assert "<secret>" in system
    assert "Public hospital homepages" in system
    assert "Dates are PII only when tied to a patient/person" in system
    assert "Good format examples" in system
    assert "Bad format examples" in system
    assert "Phone numbers must be tagged as the full phone number" in system


def test_general_profile_prompts_do_not_leak_medical_public_negative_rules() -> None:
    general = generation_profile_for_domain("general", split_purpose="train")
    scenario = _scenario("general_consumer_admin_support")

    system = targeted_system_prompt(
        normalize_language("English"),
        domain_hint=general.domain_hint,
        public_url_hint=general.public_url_hint,
    )
    user = targeted_user_prompt(
        profile=normalize_language("English"),
        scenario=scenario,
        document_type="customer support ticket",
        text_format="plain support note",
        min_spans=5,
        min_unique_labels=5,
        required_labels=required_labels_for_mode(scenario, "private_url_only"),
        public_negative_hint=general.public_negative_hint,
        adversarial_hint=general.adversarial_hint,
        realism_hint=general.realism_hint,
    )

    assert "Public hospital homepages" not in system
    assert "public status pages" in system
    assert "public guideline" not in user.lower()
    assert "status page" in user.lower()


def test_generate_until_wires_general_profile_metadata_and_prompt(tmp_path: Path) -> None:
    class FakeClient:
        calls: ClassVar[list[tuple[str, str]]] = []

        # reason: `synthetic.py:390` calls `chat(system, user, temperature=..., max_tokens=...)`, so these two names are
        # reason: written at the call site while the leading pair is passed positionally.
        async def chat(
            self,
            system: str,
            user: str,
            *,
            temperature: float,  # ruff: ignore[unused-method-argument]
            max_tokens: int,  # ruff: ignore[unused-method-argument]
        ) -> str:
            self.calls.append((system, user))
            return """Support case [CASE-9901]<id_number> for [Maya Green]<human_name> after a billing login issue.
Email [maya.green at example dot com]<email_address>, recovery phone [617 555 0199]<phone_number>, \
and address [44 Lake Road, Boston, MA]<address> were confirmed by the account owner.
The private invoice URL is [https://account.example.test/users/maya/invoices/CASE-9901?token=abc]<private_url>.
Public status page https://status.example.test and public company blog date 2026-01-01 are untagged \
because they are not person/account PII."""

    fake_client = FakeClient()
    generator = LabelCorpusGenerator(
        cast("OpenAICompatibleClient", fake_client),
        provider="mimo",
        model="fake-model",
        max_concurrency=1,
        seed=1,
    )

    summary = asyncio.run(
        generator.generate_until(
            language="English",
            target_count=1,
            output_dir=tmp_path,
            domain_profile="general",
            scenario_names=("general_consumer_admin_support",),
            required_label_mode="private_url_only",
        ),
    )

    accepted_path = tmp_path / "accepted.en.jsonl"
    record = json.loads(accepted_path.read_text(encoding="utf-8").strip())
    system_prompt, user_prompt = fake_client.calls[0]

    assert summary["accepted_count"] == 1
    assert record["info"]["source_dataset"] == LEGACY_SYNTHETIC_DATASET_IDS["general"]
    assert record["info"]["domain_bucket"] == "general"
    assert record["info"]["domain_profile"] == "general"
    assert "Public hospital homepages" not in system_prompt
    assert "public guideline" not in user_prompt.lower()
    raw_record = json.loads((tmp_path / "raw.en.jsonl").read_text(encoding="utf-8"))
    assert raw_record["required_label_mode"] == "private_url_only"
    assert raw_record["required_labels"] == [
        "human_name",
        "email_address",
        "id_number",
        "private_url",
    ]
    assert 5 <= raw_record["min_spans"] <= 20
    assert raw_record["min_unique_labels"] == 5


def test_generate_until_uses_larger_token_budget_for_opencode_deepseek(
    tmp_path: Path,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.max_tokens_seen: list[int] = []

        # reason: `synthetic.py:390` calls `chat(system, user, temperature=..., max_tokens=...)`, so these two names are
        # reason: written at the call site while the leading pair is passed positionally.
        async def chat(
            self,
            _system: str,
            _user: str,
            *,
            temperature: float,  # ruff: ignore[unused-method-argument]
            max_tokens: int,
        ) -> str:
            self.max_tokens_seen.append(max_tokens)
            return """Support case [CASE-9901]<id_number> for [Maya Green]<human_name>.
Email [maya.green@example.com]<email_address> and phone [617-555-0199]<phone_number> were confirmed.
Private account URL [https://account.example.test/recovery/CASE-9901?token=abc]<private_url>.
Temporary password [Temp-9901!]<secret>.
Public status page https://status.example.test is untagged."""

    fake_client = FakeClient()
    generator = LabelCorpusGenerator(
        cast("OpenAICompatibleClient", fake_client),
        provider="opencode_zen",
        model="deepseek-v4-flash-free",
        max_concurrency=1,
    )

    summary = asyncio.run(
        generator.generate_until(
            language="English",
            target_count=1,
            output_dir=tmp_path,
            domain_profile="general",
            scenario_names=("general_consumer_admin_support",),
            required_label_mode="private_url_secret",
            max_attempt_multiplier=1,
        ),
    )

    assert summary["accepted_count"] == 1
    assert fake_client.max_tokens_seen == [8192]


def test_generate_until_stops_on_fatal_api_error(tmp_path: Path) -> None:
    class UnauthorizedClient:
        # reason: `synthetic.py:390` calls `chat(system, user, temperature=..., max_tokens=...)`, so these two names are
        # reason: written at the call site while the leading pair is passed positionally.
        @staticmethod
        async def chat(
            _system: str,
            _user: str,
            *,
            temperature: float,  # ruff: ignore[unused-static-method-argument]
            max_tokens: int,  # ruff: ignore[unused-static-method-argument]
        ) -> str:
            request = httpx.Request("POST", "https://example.test/chat/completions")
            response = httpx.Response(401, request=request)
            msg = "unauthorized"
            raise httpx.HTTPStatusError(msg, request=request, response=response)

    generator = LabelCorpusGenerator(
        cast("OpenAICompatibleClient", UnauthorizedClient()),
        provider="mimo",
        model="mimo-v2.5-pro",
        max_concurrency=1,
    )

    with pytest.raises(RuntimeError, match="Fatal generation API error"):
        asyncio.run(
            generator.generate_until(
                language="English",
                target_count=10,
                output_dir=tmp_path,
                domain_profile="general",
                scenario_names=("general_consumer_admin_support",),
                required_label_mode="private_url_secret",
                max_attempt_multiplier=10,
            ),
        )

    raw_rows = [json.loads(line) for line in (tmp_path / "raw.en.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected_rows = [
        json.loads(line) for line in (tmp_path / "rejected.en.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(raw_rows) == 1
    assert rejected_rows[0]["fatal_api_error"] is True
    assert rejected_rows[0]["status_code"] == 401


def test_generate_until_treats_400_as_attempt_level_api_error(tmp_path: Path) -> None:
    class BadRequestThenValidClient:
        def __init__(self) -> None:
            self.calls = 0

        # reason: `synthetic.py:390` calls `chat(system, user, temperature=..., max_tokens=...)`, so these two names are
        # reason: written at the call site while the leading pair is passed positionally.
        async def chat(
            self,
            _system: str,
            _user: str,
            *,
            temperature: float,  # ruff: ignore[unused-method-argument]
            max_tokens: int,  # ruff: ignore[unused-method-argument]
        ) -> str:
            self.calls += 1
            if self.calls == 1:
                request = httpx.Request("POST", "https://example.test/chat/completions")
                response = httpx.Response(400, request=request)
                msg = "bad request"
                raise httpx.HTTPStatusError(msg, request=request, response=response)
            return """Support case [CASE-9901]<id_number> for [Maya Green]<human_name>.
Email [maya.green@example.com]<email_address> and phone [617-555-0199]<phone_number> were confirmed.
Private account URL [https://account.example.test/recovery/CASE-9901?token=abc]<private_url>.
Temporary password [Temp-9901!]<secret>.
Public status page https://status.example.test is untagged."""

    client = BadRequestThenValidClient()
    generator = LabelCorpusGenerator(
        cast("OpenAICompatibleClient", client),
        provider="mimo",
        model="mimo-v2.5-pro",
        max_concurrency=1,
    )

    summary = asyncio.run(
        generator.generate_until(
            language="English",
            target_count=1,
            output_dir=tmp_path,
            domain_profile="general",
            scenario_names=("general_consumer_admin_support",),
            required_label_mode="private_url_secret",
            max_attempt_multiplier=3,
        ),
    )

    raw_rows = [json.loads(line) for line in (tmp_path / "raw.en.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected_rows = [
        json.loads(line) for line in (tmp_path / "rejected.en.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["accepted_count"] == 1
    assert len(raw_rows) == 2
    assert rejected_rows[0]["fatal_api_error"] is False
    assert rejected_rows[0]["status_code"] == 400


def test_generate_until_accepts_non_latin_thai_document(tmp_path: Path) -> None:
    """The accepted record retains Thai script (native-script marker passed end-to-end)."""

    class FakeClient:
        # reason: `synthetic.py:390` calls `chat(system, user, temperature=..., max_tokens=...)`, so these two names are
        # reason: written at the call site while the leading pair is passed positionally.
        @staticmethod
        async def chat(
            _system: str,
            _user: str,
            *,
            temperature: float,  # ruff: ignore[unused-static-method-argument]
            max_tokens: int,  # ruff: ignore[unused-static-method-argument]
        ) -> str:
            return (
                "เคสสนับสนุน [CASE-9901]<id_number> สำหรับ [มานี ใจดี]<human_name> "
                "หลังปัญหาการเข้าสู่ระบบ\n"
                "อีเมล [mani.jaidee@example.com]<email_address> โทรศัพท์ "
                "[617-555-0199]<phone_number> และที่อยู่ "
                "[44 ถนนสุขุมวิท กรุงเทพ]<address> ได้รับการยืนยันแล้ว\n"
                "ลิงก์ใบแจ้งหนี้ส่วนตัวคือ "
                "[https://account.example.test/users/mani/invoices/CASE-9901?token=abc]<private_url>\n"
                "รหัสผ่านชั่วคราว [Temp-9901!]<secret>\n"
                "หน้าสถานะสาธารณะ https://status.example.test ไม่ได้ติดแท็ก"
            )

    generator = LabelCorpusGenerator(
        cast("OpenAICompatibleClient", FakeClient()),
        provider="openrouter",
        model="moonshotai/kimi-k2.6:free",
        max_concurrency=1,
    )

    summary = asyncio.run(
        generator.generate_until(
            language="Thai",
            target_count=1,
            output_dir=tmp_path,
            domain_profile="general",
            scenario_names=("general_consumer_admin_support",),
            required_label_mode="private_url_secret",
        ),
    )

    assert summary["accepted_count"] == 1
    accepted_path = tmp_path / "accepted.th.jsonl"
    record = json.loads(accepted_path.read_text(encoding="utf-8").strip())
    assert record["info"]["language"] == "Thai"
    blob = json.dumps(record, ensure_ascii=False)
    assert any("฀" <= ch <= "๿" for ch in blob)


def test_existing_text_hashes_seeds_from_migrated_pm_accepted_siblings(
    tmp_path: Path,
) -> None:
    accepted_path = tmp_path / "accepted.en.jsonl"
    pm_accepted_path = tmp_path / "pm_accepted.en.jsonl"
    root_text = "Patient [Maya Green]<human_name> visited the clinic."
    migrated_text = "Support case [CASE-9901]<id_number> for [Lan Pham]<human_name>."
    accepted_path.write_text(json.dumps({"text": root_text}, ensure_ascii=False) + "\n", encoding="utf-8")
    pm_accepted_path.write_text(json.dumps({"text": migrated_text}, ensure_ascii=False) + "\n", encoding="utf-8")

    hashes = _existing_text_hashes(accepted_path)

    assert hashlib.sha256(root_text.encode("utf-8")).hexdigest() in hashes
    assert hashlib.sha256(migrated_text.encode("utf-8")).hexdigest() in hashes


def test_run_synthetic_generation_owns_provider_model_and_generator_contract(
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class FakeClientContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def fake_client_factory(**kwargs: object) -> FakeClientContext:
        calls["client_kwargs"] = kwargs
        return FakeClientContext()

    class FakeGenerator:
        def __init__(self, _client: object, **kwargs: object) -> None:
            calls["generator_kwargs"] = kwargs

        @staticmethod
        async def generate_until(**kwargs: object) -> dict[str, object]:
            calls["generate_kwargs"] = kwargs
            return {"accepted_count": 3, "paths": {"summary": str(tmp_path / "s.json")}}

    request = SyntheticGenerationRequest(
        provider="mimo",
        model=None,
        language="English",
        target_count=3,
        output_dir=tmp_path,
        domain_profile="general",
        scenario_names=("general_consumer_admin_support",),
        count_mode="total",
        max_concurrency=2,
        rpm_per_key=10,
        required_labels=("secret",),
        required_label_mode="private_url_secret",
        max_attempt_multiplier=1.5,
        seed=7,
        log_every=1,
        require_vietnamese_marker=False,
    )

    summary = asyncio.run(
        run_synthetic_generation(
            request,
            # reason: the fake context yields a stand-in the replaced generator never uses as a client;
            # reason: casting states that at the boundary rather than widening the runner's own contract.
            client_factory=cast("ClientFactory", fake_client_factory),
            generator_cls=FakeGenerator,
        ),
    )

    assert summary["accepted_count"] == 3
    assert calls["client_kwargs"] == {
        "provider": "mimo",
        "model": "mimo-v2-flash",
        "rpm_per_key": 10,
        "daily_budget": None,
        "account_ledger": None,
    }
    assert calls["generator_kwargs"] == {
        "provider": "mimo",
        "model": "mimo-v2-flash",
        "max_concurrency": 2,
        "seed": 7,
    }
    generate_kwargs = _mapping_field(calls, "generate_kwargs")
    assert generate_kwargs["target_count"] == 3
    assert generate_kwargs["required_labels"] == ("secret",)


def test_eval_gold_stops_cleanly_on_budget_exceeded(tmp_path: Path) -> None:
    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        if kwargs["language"] == "french":
            msg = "cap reached"
            raise DailyBudgetExceeded(msg)
        return {"accepted_count": 1, "rows": [{"private": "omitted"}]}

    summary = asyncio.run(
        run_eval_gold(
            per_lang=1,
            languages=("english", "french", "german"),
            out_root=tmp_path,
            generate_one=fake_generate_one,
        ),
    )

    assert summary["per_lang"] == 1
    assert summary["languages"] == ["english", "french", "german"]
    summaries = _mapping_field(summary, "summaries")
    assert summaries["english"] == {"accepted_count": 1}
    assert summaries["french"] == {
        "skipped": "budget_exceeded",
        "provider": "openai",
    }
    assert "german" not in summaries


def test_weak_labels_routes_explicit_providers(tmp_path: Path) -> None:
    """Every language now gets private_url + secret.

    The old english-only- private_url skip is gone (english is now a 20k priority language): 2 languages x 2 passes = 4.

    Labels are namespaced per provider + account + round so concurrent per-account workers don't clash on the summaries
    dict.

    """
    seen_providers: list[object] = []

    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        seen_providers.append(kwargs["provider"])
        return {"accepted_count": 1, "rows": [{"omit": "this"}]}

    summary = asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("english", "vietnamese"),
            run_date="2026-06-15",
            providers=("openai",),
            openai_cap=1000,
            max_rounds=1,
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )

    assert set(seen_providers) == {"openai"}
    assert len(seen_providers) == 4
    summaries = _mapping_field(summary, "summaries")
    assert "openai:acct1:english:private_url:r0" in summaries
    assert "openai:acct1:english:secret:r0" in summaries
    assert "openai:acct1:vietnamese:secret:r0" in summaries


def test_weak_labels_defaults_to_free_provider_pool(tmp_path: Path) -> None:
    seen_providers: list[object] = []

    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        seen_providers.append(kwargs["provider"])
        return {"accepted_count": 1, "rows": [{"omit": "this"}]}

    asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("english", "vietnamese"),
            run_date="2026-06-15",
            max_rounds=1,
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )

    assert set(seen_providers) <= set(WEAK_LABEL_FREE_PROVIDERS)
    assert "openai" not in seen_providers


def test_build_budget_keeps_openai_grant_at_full_cap() -> None:
    """OpenAI is a single-org grant calibrated to sit just under the 2.5M/day ceiling.

    The free-pool 0.9 headroom must not shrink it. Regression guard for the silent 2.4M -> 2.16M cut when build_budget
    began headroom'ing every provider. Spending exactly the cap is allowed; one token over trips.

    """
    from datetime import date

    budget = build_budget()
    day = date(2026, 6, 19)
    budget.check_and_add("openai", 2_400_000, today=day)
    with pytest.raises(DailyBudgetExceeded):
        budget.check_and_add("openai", 1, today=day)


def test_weak_labels_threads_count_mode_and_defaults_to_additional(tmp_path: Path) -> None:
    """The daily run defaults to additive generation so each run adds fresh samples toward the providers'.

    The daily run defaults to additive generation so each run adds fresh samples toward the providers' budgets instead of
    resuming to a fixed target.

    """
    seen_modes: list[object] = []

    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        seen_modes.append(kwargs.get("count_mode"))
        return {"accepted_count": 1, "rows": [{"omit": "this"}]}

    asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("vietnamese",),
            run_date="2026-06-15",
            providers=("groq",),
            max_rounds=1,
            count_mode="additional",
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )
    assert seen_modes == ["additional", "additional"]

    seen_modes.clear()
    asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("vietnamese",),
            run_date="2026-06-15",
            providers=("groq",),
            max_rounds=1,
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )
    assert seen_modes == ["additional", "additional"]


def test_weak_labels_loops_rounds_until_budget_exhausted(tmp_path: Path) -> None:
    """Looped past the first 3-cell pass.

    Then stopped on the budget rather than running all 10 rounds (30 calls). The cap is the limiter, not max_rounds.

    """
    generating_budget = 5
    calls = {"n": 0}

    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**_kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        calls["n"] += 1
        if calls["n"] > generating_budget:
            msg = "cap reached"
            raise DailyBudgetExceeded(msg)
        return {"accepted_count": calls["n"], "rows": [{"omit": "this"}]}

    asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("english", "vietnamese"),
            run_date="2026-06-15",
            providers=("openai",),
            count_mode="additional",
            max_rounds=10,
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )

    assert calls["n"] > 3
    assert calls["n"] < 30


def test_weak_labels_max_rounds_controls_passes(tmp_path: Path) -> None:
    """2 passes over vietnamese's 2 cells (private_url + secret) = 4 calls."""
    calls = {"n": 0}

    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**_kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        calls["n"] += 1
        return {"accepted_count": 1, "rows": [{"omit": "this"}]}

    asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("vietnamese",),
            run_date="2026-06-15",
            providers=("openai",),
            count_mode="additional",
            max_rounds=2,
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )

    assert calls["n"] == 4


def test_weak_labels_provider_stops_on_budget_without_blocking_others(tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    # reason: `generation_runs.py:172` and `:202` call this injected hook as `await generate_one(...)`, so dropping
    # reason: `async` would leave the run awaiting a value that is not awaitable.
    async def fake_generate_one(**kwargs: object) -> dict[str, object]:  # ruff: ignore[unused-async]
        provider = str(kwargs["provider"])
        calls[provider] = calls.get(provider, 0) + 1
        if provider == "groq":
            msg = "groq budget exhausted"
            raise DailyBudgetExceeded(msg)
        return {"accepted_count": 1, "rows": [{"omit": "this"}]}

    asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("vietnamese",),
            run_date="2026-06-15",
            providers=("groq", "cerebras"),
            max_rounds=3,
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )

    assert calls["groq"] == 1
    assert calls["cerebras"] == 6


def test_weak_labels_hard_deadline_cancels_stuck_provider(tmp_path: Path) -> None:
    """The run returned (did not block on the 30s sleep) — deadline cancellation fired.

    groq started exactly one (stuck) cell; cerebras finished its pass.

    """
    started = {"slow": 0, "fast": 0}

    async def fake_generate_one(**kwargs: object) -> dict[str, object]:
        if kwargs["provider"] == "groq":
            started["slow"] += 1
            await asyncio.sleep(30)
            return {"accepted_count": 1, "rows": []}
        started["fast"] += 1
        return {"accepted_count": 1, "rows": [{"omit": "this"}]}

    summary = asyncio.run(
        run_weak_labels(
            private_url_per_lang=1,
            secret_per_lang=1,
            languages=("vietnamese",),
            run_date="2026-06-15",
            providers=("groq", "cerebras"),
            max_rounds=1,
            max_minutes=0.02,
            out_root=tmp_path,
            generate_one=fake_generate_one,
            account_keys=lambda _provider: ["k"],
        ),
    )

    assert started["slow"] == 1
    assert started["fast"] == 2
    assert summary["run_date"] == "2026-06-15"
