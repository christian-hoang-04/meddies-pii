"""Generation catalog for the Meddies Labels synthetic generator — the control surface.

Pure declarative data plus the resolution helpers over it: the scenarios (intent +
required labels), the document types / text formats each domain profile may draw
from, and how a domain string maps to a profile. This is the single place to add,
adjust, or rate-control WHAT gets generated. Prompt phrasing lives in prompts.py;
the accept/reject gate in validate.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from meddies_pii.historical_artifacts import LEGACY_SYNTHETIC_DATASET_IDS
from meddies_pii.taxonomy import PII_LABELS, PiiLabel, is_pii_label

if TYPE_CHECKING:
    import random
    from collections.abc import Iterable, Sequence

REQUIRED_PRIVATE_LABELS: tuple[PiiLabel, ...] = ("private_url", "secret")


REQUIRED_LABEL_MODES: tuple[str, ...] = (
    "scenario_default",
    "private_url_secret",
    "private_url_only",
    "secret_only",
    "none",
)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    description: str
    required_labels: tuple[PiiLabel, ...]
    min_unique_labels: int
    min_spans: int
    adversarial: bool = False


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    domain_profile: str
    split_purpose: str
    source_dataset: str
    domain_bucket: str
    scenario_names: tuple[str, ...]
    document_types: tuple[str, ...]
    text_formats: tuple[str, ...]
    domain_hint: str
    realism_hint: str
    public_url_hint: str | None = None
    public_negative_hint: str | None = None
    adversarial_hint: str | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="all_labels_dense",
        description=(
            "Dense administrative-medical document. Include every one of the 9 Meddies labels at least once. "
            "Use patient, clinician, relative, hospital/clinic/insurer, address, dates, phone, email, identifiers, "
            "one private patient portal URL, and one secret credential/token."
        ),
        required_labels=PII_LABELS,
        min_unique_labels=9,
        min_spans=12,
    ),
    Scenario(
        name="private_portal_secret_focus",
        description=(
            "Patient portal / lab-result access workflow. Make private_url and secret prominent: signed result link, "
            "portal deep link, reset token, OTP, session cookie, or API key. Add several normal PII labels too."
        ),
        required_labels=("private_url", "secret", "human_name", "id_number"),
        min_unique_labels=6,
        min_spans=10,
    ),
    Scenario(
        name="adversarial_obfuscation",
        description=(
            "Noisy/adversarial PII as it appears in real voice-scribe, OCR, chat, and messy admin text. "
            "Several values are mangled into hard-to-parse surface forms (apply each in THIS document's "
            "language's native conventions, not only English), while 1-2 values stay completely clean for "
            "contrast. Keep every [value]<label> tag wrapping the FULL obfuscated surface, and tag by WHAT "
            "the value is, not how it is written."
        ),
        required_labels=("private_url", "secret", "email_address", "phone_number"),
        min_unique_labels=6,
        min_spans=10,
        adversarial=True,
    ),
    Scenario(
        name="public_negative_contrast",
        description=(
            "Contrast positives vs negatives. Include patient/person PII with tags, but also include "
            "public guideline dates, "
            "copyright dates, or public hospital/article URLs untagged. Do not tag public/non-person dates or URLs."
        ),
        required_labels=("private_url", "secret", "date", "human_name"),
        min_unique_labels=6,
        min_spans=10,
    ),
    Scenario(
        name="structured_payload",
        description=(
            "Structured data such as JSON, FHIR-like JSON, HL7-ish text, XML, table, or portal audit log. "
            "Put labels inside string values only; leave lab values, dosages, vitals, ages, and medical codes untagged."
        ),
        required_labels=("private_url", "secret", "id_number", "company_name"),
        min_unique_labels=6,
        min_spans=10,
    ),
    Scenario(
        name="general_consumer_admin_support",
        description=(
            "General-domain consumer/admin/support document. Use account support, billing, shipping, insurance, "
            "school, telecom, or e-commerce context. Include common PII and public-negative URLs/dates where useful; "
            "do not require both private_url and secret unless the requested label mode asks for them."
        ),
        required_labels=("human_name", "email_address", "id_number"),
        min_unique_labels=5,
        min_spans=5,
    ),
    Scenario(
        name="hr_employer_school_insurance_telecom_admin",
        description=(
            "General HR, employer, school, insurance, telecom, banking-adjacent, or civic admin document. "
            "Use employee/student/customer IDs, names, addresses, contacts, dates, organizations, and realistic forms."
        ),
        required_labels=("human_name", "id_number", "email_address"),
        min_unique_labels=5,
        min_spans=6,
    ),
    Scenario(
        name="code_log_security_account_recovery",
        description=(
            "Code, log, support-ticket, account-recovery, or security incident text. Use JSON, .env, traceback, "
            "audit log, cookie/header, reset workflow, API key, auth token, or session context."
        ),
        required_labels=("secret", "id_number"),
        min_unique_labels=4,
        min_spans=5,
    ),
    Scenario(
        name="adversarial_formatting",
        description=(
            "General adversarial PII formatting without assuming a medical setting. Several values appear in "
            "messy, hard-to-parse surface forms while 1-2 stay clean for contrast; tag the full value by what "
            "it is, not how it is written."
        ),
        required_labels=("email_address", "phone_number", "id_number"),
        min_unique_labels=5,
        min_spans=6,
        adversarial=True,
    ),
    Scenario(
        name="structured_general_payload",
        description=(
            "Structured general-domain payload such as JSON, YAML, key-value logs, CSV-like rows, headers, "
            "or account audit records. Label string values only and leave public/system fields untagged."
        ),
        required_labels=("id_number", "email_address"),
        min_unique_labels=4,
        min_spans=5,
    ),
    Scenario(
        name="one_hop_deferred_clue",
        description=(
            "One-hop/deferred-clue example where surrounding text explains why a value is private, such as "
            "a link, token, ID, or date belonging to a named person/account rather than being public."
        ),
        required_labels=("human_name", "id_number"),
        min_unique_labels=5,
        min_spans=6,
    ),
    Scenario(
        name="public_negative_url_date_contrast",
        description=(
            "General-domain contrast example. Include tagged personal/account PII plus untagged public URLs, "
            "public documentation links, status pages, public article dates, or copyright dates."
        ),
        required_labels=("human_name", "date", "email_address"),
        min_unique_labels=5,
        min_spans=6,
    ),
)


DOCUMENT_TYPES: tuple[str, ...] = (
    "discharge summary",
    "outpatient note",
    "lab result notification",
    "patient portal access log",
    "insurance claim note",
    "referral letter",
    "pharmacy refill request",
    "appointment reminder",
    "telehealth triage note",
    "medical bill / invoice",
    "FHIR-like Patient + Encounter resource",
    "HL7-style admission/update message",
)

TEXT_FORMATS: tuple[str, ...] = (
    "plain clinical note",
    "Markdown sections",
    "messy nurse/admin note",
    "table-like rows",
    "JSON-like object",
    "FHIR-like JSON",
    "HL7-like pipe-delimited text",
    "patient portal audit log",
    "voice-scribe transcript",
    "OCR text with light noise",
    "clinical dialogue transcript",
)

GENERAL_DOCUMENT_TYPES: tuple[str, ...] = (
    "customer support ticket",
    "billing dispute",
    "shipping update",
    "school enrollment note",
    "telecom account note",
    "insurance claim message",
    "HR onboarding form",
    "account recovery email",
)

CODE_LOG_DOCUMENT_TYPES: tuple[str, ...] = (
    "application audit log",
    ".env-style configuration snippet",
    "support ticket with request headers",
    "account recovery trace",
    "JSON security event",
    "API error report",
)

GENERAL_TEXT_FORMATS: tuple[str, ...] = (
    "plain support note",
    "email thread",
    "Markdown sections",
    "table-like rows",
    "form fields",
    "chat transcript",
)

CODE_LOG_TEXT_FORMATS: tuple[str, ...] = (
    "JSON-like object",
    ".env key-value block",
    "HTTP headers",
    "application log lines",
    "YAML-like config",
    "support ticket with code block",
)


MEDICAL_SCENARIO_NAMES: tuple[str, ...] = (
    "all_labels_dense",
    "private_portal_secret_focus",
    "adversarial_obfuscation",
    "public_negative_contrast",
    "structured_payload",
)

GENERAL_SCENARIO_NAMES: tuple[str, ...] = (
    "general_consumer_admin_support",
    "hr_employer_school_insurance_telecom_admin",
    "public_negative_url_date_contrast",
    "one_hop_deferred_clue",
)

CODE_LOG_SCENARIO_NAMES: tuple[str, ...] = (
    "code_log_security_account_recovery",
    "structured_general_payload",
    "adversarial_formatting",
    "one_hop_deferred_clue",
)

DOMAIN_PROFILES: tuple[str, ...] = ("medical", "general", "code_logs", "mixed")
SPLIT_PURPOSES: tuple[str, ...] = ("train", "challenge")


def generation_profile_for_domain(
    domain_profile: str,
    *,
    split_purpose: str = "train",
) -> GenerationProfile:
    domain = domain_profile.strip().lower()
    purpose = split_purpose.strip().lower()
    if domain not in DOMAIN_PROFILES:
        msg = f"domain_profile must be one of {DOMAIN_PROFILES}; got {domain_profile!r}"
        raise ValueError(msg)
    if purpose not in SPLIT_PURPOSES:
        msg = f"split_purpose must be one of {SPLIT_PURPOSES}; got {split_purpose!r}"
        raise ValueError(msg)

    source_dataset = LEGACY_SYNTHETIC_DATASET_IDS[domain]
    if purpose == "challenge":
        source_dataset = LEGACY_SYNTHETIC_DATASET_IDS["challenge"]

    if domain == "medical":
        return GenerationProfile(
            domain_profile=domain,
            split_purpose=purpose,
            source_dataset=source_dataset,
            domain_bucket="medical",
            scenario_names=MEDICAL_SCENARIO_NAMES,
            document_types=DOCUMENT_TYPES,
            text_formats=TEXT_FORMATS,
            domain_hint="healthcare, hospitals, clinics, insurers, patient portals, and administrative forms",
            realism_hint=(
                "Keep the document realistic for healthcare/admin operations, but do not include real "
                "people or real credentials."
            ),
        )
    if domain == "general":
        return GenerationProfile(
            domain_profile=domain,
            split_purpose=purpose,
            source_dataset=source_dataset,
            domain_bucket="general",
            scenario_names=GENERAL_SCENARIO_NAMES,
            document_types=GENERAL_DOCUMENT_TYPES,
            text_formats=GENERAL_TEXT_FORMATS,
            domain_hint=(
                "general consumer, HR, school, telecom, insurance, billing, e-commerce, and account support contexts"
            ),
            realism_hint=(
                "Keep the document realistic for general account/admin operations, but do not include "
                "real people or real credentials."
            ),
            public_url_hint=(
                "Public homepages, documentation pages, public status pages, product pages, and articles are not PII."
            ),
            public_negative_hint=(
                "Include one untagged public homepage, documentation URL, status page, article date, "
                "or copyright date, but do not tag it because it is not person/account PII."
            ),
            adversarial_hint=(
                "Use general-domain adversarial PII forms in some values: at-dot email, line-broken phone/account URL, "
                "spaced IDs, symbol substitution, or token-like strings only when they naturally belong."
            ),
        )
    if domain == "code_logs":
        return GenerationProfile(
            domain_profile=domain,
            split_purpose=purpose,
            source_dataset=source_dataset,
            domain_bucket="code_logs",
            scenario_names=CODE_LOG_SCENARIO_NAMES,
            document_types=CODE_LOG_DOCUMENT_TYPES,
            text_formats=CODE_LOG_TEXT_FORMATS,
            domain_hint=(
                "application logs, account recovery, support tickets, API traces, cookies, headers, and "
                "configuration snippets"
            ),
            realism_hint=(
                "Keep the document realistic for software/security operations, but do not include real "
                "credentials or real endpoints."
            ),
            public_url_hint=(
                "Public documentation URLs, status pages, package pages, public repo links, "
                "and product homepages are not PII."
            ),
            public_negative_hint=(
                "Include one untagged public docs/status/package/repo URL or public release date, "
                "but do not tag it because it is not person/account PII."
            ),
            adversarial_hint=(
                "Use software/log adversarial PII forms in some values: wrapped headers, line-broken URLs, "
                "masked account IDs, env-style secrets only when required, or token fragments split by punctuation."
            ),
        )
    return GenerationProfile(
        domain_profile=domain,
        split_purpose=purpose,
        source_dataset=source_dataset,
        domain_bucket="general",
        scenario_names=(*GENERAL_SCENARIO_NAMES, *CODE_LOG_SCENARIO_NAMES),
        document_types=(*GENERAL_DOCUMENT_TYPES, *CODE_LOG_DOCUMENT_TYPES),
        text_formats=(*GENERAL_TEXT_FORMATS, *CODE_LOG_TEXT_FORMATS),
        domain_hint="mixed general-admin, support, account recovery, software log, and security contexts",
        realism_hint=(
            "Keep the document realistic for general or software/admin operations, but do not include "
            "real people or real credentials."
        ),
        public_url_hint=(
            "Public documentation URLs, status pages, product pages, repo links, public homepages, "
            "and articles are not PII."
        ),
        public_negative_hint=(
            "Include one untagged public documentation/status/product/homepage URL or public date, "
            "but do not tag it because it is not person/account PII."
        ),
        adversarial_hint=(
            "Use general/software adversarial PII forms in some values: at-dot email, line-broken phone/URL, "
            "spaced IDs, wrapped headers, or token fragments only when they naturally belong."
        ),
    )


def _scenario_by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    msg = f"Unknown scenario: {name}"
    raise ValueError(msg)


def scenario_pool_for_profile(
    generation_profile: GenerationProfile,
    requested_scenarios: Sequence[str] | None = None,
) -> tuple[Scenario, ...]:
    enabled = set(generation_profile.scenario_names)
    requested = tuple(requested_scenarios or generation_profile.scenario_names)
    out: list[Scenario] = []
    for scenario_name in requested:
        normalized = scenario_name.strip()
        if normalized not in enabled:
            msg = (
                f"Scenario {normalized!r} is not enabled for domain profile "
                f"{generation_profile.domain_profile!r}; enabled={sorted(enabled)}"
            )
            raise ValueError(
                msg,
            )
        out.append(_scenario_by_name(normalized))
    if not out:
        msg = "At least one scenario must be enabled for generation"
        raise ValueError(msg)
    return tuple(out)


def _dedupe_labels(labels: Iterable[PiiLabel]) -> tuple[PiiLabel, ...]:
    out: list[PiiLabel] = []
    seen: set[PiiLabel] = set()
    for raw_label in labels:
        label = raw_label.strip().lower()
        if not is_pii_label(label):
            msg = f"Unsupported Meddies Labels required label: {raw_label!r}"
            raise ValueError(msg)
        if label in seen:
            continue
        out.append(label)
        seen.add(label)
    return tuple(out)


def required_labels_for_mode(
    scenario: Scenario,
    mode: str,
    *,
    extra_required_labels: Sequence[PiiLabel] = (),
) -> tuple[PiiLabel, ...]:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "scenario_default":
        base: tuple[PiiLabel, ...] = scenario.required_labels
    elif normalized_mode == "private_url_secret":
        base = (*scenario.required_labels, "private_url", "secret")
    elif normalized_mode == "private_url_only":
        base = (*scenario.required_labels, "private_url")
    elif normalized_mode == "secret_only":
        base = (*scenario.required_labels, "secret")
    elif normalized_mode == "none":
        base = ()
    else:
        msg = f"required label mode must be one of {REQUIRED_LABEL_MODES}; got {mode!r}"
        raise ValueError(msg)
    return _dedupe_labels((*base, *extra_required_labels))


SPAN_TARGET_RANGE = (3, 20)


def sample_span_target(rng: random.Random, scenario: Scenario, required_labels: Sequence[PiiLabel]) -> int:
    """Per-document target span count for density diversity.

    Drawn from SPAN_TARGET_RANGE, but never below what the scenario needs to stay
    coherent (its unique-label minimum) or below the number of required labels, so
    the requested labels still fit.

    Returns:
        The largest of the sampled draw, the scenario's unique-label minimum and the required
        label count. Raising the floor rather than resampling means the density spread stays
        whatever the range gives, and only documents that would have been too sparse to hold
        their own labels get pushed up.

    """
    base = rng.randint(*SPAN_TARGET_RANGE)
    return max(base, scenario.min_unique_labels, len(required_labels))
