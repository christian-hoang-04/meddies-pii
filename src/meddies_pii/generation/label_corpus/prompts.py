"""Prompt templates for the Meddies Labels synthetic generator.

The system + user prompts sent to the LLM. Edit phrasing here; WHAT can be
generated (scenarios, document types, text formats) lives in catalog.py.
"""

from __future__ import annotations

# ruff: file-ignore[hardcoded-password-string]
# reason: `secret` and `private_url` are PII LABEL names in this project's taxonomy, and this
# reason: variable holds a prompt instruction telling the generator to emit spans carrying those
# reason: labels. The string is corpus-generation copy; it carries no credential.
from typing import TYPE_CHECKING

from meddies_pii.taxonomy import PII_LABELS, PiiLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.generation.label_corpus.catalog import Scenario
    from meddies_pii.languages import LanguageProfile


def targeted_system_prompt(
    profile: LanguageProfile,
    *,
    domain_hint: str | None = None,
    public_url_hint: str | None = None,
) -> str:
    labels = "\n".join(f"- `<{label}>`" for label in PII_LABELS)
    domain = domain_hint or profile.country_hint
    public_url_rule = (
        public_url_hint
        or "Public hospital homepages, public guidelines, public articles, and documentation URLs are not PII."
    )
    return f"""You generate fully synthetic PII training documents for a Meddies Labels BIOES span detector.

Output only the document. No preface, no explanation, no markdown code fence unless the requested \
document itself is structured text.

Language: {profile.name}.
Domain: {domain}.

Allowed positive labels, and only these labels:
{labels}

Annotation format is exactly `[specific synthetic value]<label>`.
Every positive label marker must be immediately preceded by a square-bracketed value.
Never write `value<label>` or dangling `<label>` markers.
For JSON, XML, HL7, tables, or logs, the string field value itself must contain `[value]<label>`.
Good format examples:
- `[Maya Green]<human_name>`
- `[maya.green@example.com]<email_address>`
- `[https://portal.example.test/patients/P-104/results?token=tok_8f91]<private_url>`
- `[otp-482-119]<secret>`
Bad format examples:
- `Maya Green<human_name>` because the value is not bracketed.
- `<human_name>Maya Green` because the label appears before the value.
- `[Maya Green] <human_name>` because a space splits the value and label.
- `[Maya Green]<human_name` because the closing `>` is missing.
Human names are always PII. Dates are PII only when tied to a patient/person/encounter/credential; \
public guideline/article/copyright dates are not PII.
Private URLs are patient portals, signed links, private records/results/claims/messages, or URLs \
with patient/session/token/auth parameters. {public_url_rule}
Secrets are passwords, OTPs, PINs, API keys, bearer/access/auth/session/CSRF tokens, cookies, reset links/codes.
Phone numbers must be tagged as the full phone number, including country/area code and extension \
when present. Do not tag a 4-digit extension or suffix as a standalone `<phone_number>`.

Tag the MINIMAL entity value only — never a whole sentence, title, heading, descriptive clause, or \
document label. `company_name` is the institution name itself (the clinic/hospital/insurer name), \
not a sentence describing it.
When digits are spoken aloud or OCR-transcribed as words, still tag by WHAT the number is: a \
contact phone number is `<phone_number>`; a patient/record/claim/insurance/account identifier is \
`<id_number>`. Tag the full spoken span but do not absorb trailing narration words.
Never tag vitals, lab measurements, dosages, medical terms, diagnoses, procedure names, medication \
names, age, sex/gender, or public/non-person facts.
All names, addresses, IDs, URLs, emails, tokens, dates, and institutions must be fictional; use \
reserved example domains (example.com, example.org, example.net, example.test) for emails and URLs.
Make every phone number and identifier look realistic and varied; never use lazy sequential or \
repeated digit runs such as 0987654321, 0123456789, or 1111111111."""


# reason: targeted user exposes profile/edge cases as its public contract; bundling would break callers.
def targeted_user_prompt(  # ruff: ignore[too-many-arguments]
    *,
    profile: LanguageProfile,
    scenario: Scenario,
    document_type: str,
    text_format: str,
    min_spans: int,
    min_unique_labels: int,
    required_labels: Sequence[PiiLabel] | None = None,
    public_negative_hint: str | None = None,
    adversarial_hint: str | None = None,
    realism_hint: str | None = None,
    edge_cases: Sequence[str] = (),
) -> str:
    resolved_required_labels: tuple[PiiLabel, ...] = (
        tuple(required_labels) if required_labels is not None else tuple(scenario.required_labels)
    )
    if resolved_required_labels:
        required = ", ".join(f"<{label}>" for label in resolved_required_labels)
        required_instruction = f"Required labels for this sample: {required}."
    else:
        required_instruction = (
            "No specific label is globally required for this sample; still include realistic Meddies Labels PII spans."
        )
    label_set = set(resolved_required_labels)
    if {"private_url", "secret"} <= label_set:
        private_secret_instruction = "Always include at least one `<private_url>` and one `<secret>`."
    elif "private_url" in label_set:
        private_secret_instruction = (
            "Include at least one `<private_url>`. Do not add `<secret>` unless it naturally belongs in the document."
        )
    elif "secret" in label_set:
        private_secret_instruction = (
            "Include at least one `<secret>`. Do not add `<private_url>` unless it naturally belongs in the document."
        )
    else:
        private_secret_instruction = (
            "Do not force `<private_url>` or `<secret>`; use them only if they naturally belong in the scenario."
        )
    public_negative_rule = public_negative_hint or profile.public_negative_hint
    adversarial_rule = adversarial_hint or profile.adversarial_hint
    realism = realism_hint or (
        "Keep the document realistic for healthcare/admin operations, but do not include real people or real credentials."
    )
    if not edge_cases:
        edge_case_line = ""
    elif len(edge_cases) == 1:
        edge_case_line = (
            f"\n15. Apply ONE surface perturbation to at most 1-2 values, leaving the rest clean: {edge_cases[0]}"
        )
    else:
        perturbations = "\n".join(f"   - {edge}" for edge in edge_cases)
        edge_case_line = (
            f"\n15. Apply EACH of these {len(edge_cases)} surface perturbations, each to a "
            "DIFFERENT value, and still leave 1-2 values completely clean for contrast:\n"
            f"{perturbations}"
        )
    return f"""Generate one synthetic {profile.name} {document_type} in {text_format} format.

Scenario: {scenario.name}
{scenario.description}

Hard constraints:
1. Use at least {min_spans} tagged PII spans.
2. Use at least {min_unique_labels} unique Meddies Labels labels.
3. {required_instruction}
4. {private_secret_instruction}
5. {public_negative_rule}
6. {adversarial_rule}
7. {realism}
8. Output only the document content. Do not explain the labels.
9. Before final output, self-check that every required label appears and every label uses `[value]<label>` exactly.
10. Do not output a checklist, legend, label inventory, or bare label such as `<human_name>` without a value.
11. If you use a phone number, bracket the whole number; do not tag short suffixes, extensions, or \
fragments as `<phone_number>`.
12. If you use `<private_url>`, bracket the full private URL including patient/account/session/token/auth query parameters.
13. If you use `<secret>`, bracket only the credential value itself: OTP, password, cookie, bearer \
token, API key, reset code, or session token.
14. Do not name or describe the document type or format in the output (never write "this is an \
ambient voice scribe transcript" or similar); produce the document content directly.
{edge_case_line}

Required-label coverage before you answer: {", ".join(f"{label}=present" for label in resolved_required_labels) or "none"}.
Minimum coverage before you answer: at least {min_spans} bracketed tags and at least {min_unique_labels} unique labels.

Remember: label syntax must be `[value]<label>` and labels must be one of: \
{", ".join(f"<{label}>" for label in PII_LABELS)}."""
