"""Batch translation of tagged clinical text into Vietnamese via a local vLLM model.

Translates ``[value]<tag>`` documents to Vietnamese and localizes Western PII
values to plausible Vietnamese equivalents, preserving the tag structure and
Markdown layout. ``vllm`` is an optional GPU-only dependency imported lazily
inside the functions, so this module loads (and ``clean_output`` runs) without
it; the GPU paths raise ``ModuleNotFoundError`` if vllm is absent.

NOTE: ``SYSTEM_PROMPT``'s KEEP-tag list is the generic ~70-label taxonomy this
tool was originally built against (ai4privacy-style), NOT the current Meddies Labels
labels (``meddies_pii.taxonomy.PII_LABELS``). Align the KEEP list with the
9-label set before using this on the 9-label corpus.
"""

from __future__ import annotations

# ruff: file-ignore[type-check-without-type-error]
# reason: every guard here reports an environment or contract failure - a missing asset, an unverified
# reason: checkpoint, a wrong profile, a malformed launch contract - so TypeError would misdescribe it. The
# reason: same function raises this type from non-isinstance guards too; splitting on the guard shape would
# reason: make one failure class signal two exception types.
import importlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from meddies_pii.json_types import as_json_object

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DEFAULT_MODEL = "Qwen/Qwen3-8B"


class _ChatMessage(TypedDict):
    role: str
    content: str


class _Tokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[_ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


class _Completion(Protocol):
    text: str


class _GenerationOutput(Protocol):
    outputs: Sequence[_Completion]


class _VllmModel(Protocol):
    def get_tokenizer(self) -> _Tokenizer: ...

    def generate(self, prompts: list[str], *, sampling_params: object, use_tqdm: bool) -> Sequence[_GenerationOutput]: ...


@runtime_checkable
class _VllmModule(Protocol):
    LLM: Callable[..., _VllmModel]
    SamplingParams: Callable[..., object]


def _load_vllm() -> _VllmModule:
    """Load and validate the optional vLLM seam without leaking unknown types.

    Returns:
        The imported module, narrowed to the protocol this module calls, so every later use
        is checked rather than reaching through an opaque import.

    Raises:
        RuntimeError: If the installed vllm does not expose the expected LLM API. It fails
            at the seam rather than at the first attribute access inside a GPU run.

    """
    module = importlib.import_module("vllm")
    if not isinstance(module, _VllmModule):
        msg = "vllm does not expose the expected LLM API"
        raise RuntimeError(msg)
    return module


def _record_text(record: object) -> str:
    """Read the one supported JSONL field from untrusted decoded input.

    Returns:
        The ``text_tagged`` string, or ``""`` when the record is not a JSON object or the
        field is absent or not a string. Every rejection collapses to the empty string, so a
        malformed record is skipped downstream rather than raising mid-file.

    """
    record_object = as_json_object(record)
    if record_object is None:
        return ""
    value = record_object.get("text_tagged")
    return value if isinstance(value, str) else ""


SYSTEM_PROMPT = """
You are a Vietnamese localization expert. Translate the user content to natural, formal Vietnamese \
and localize any Western details to Vietnamese equivalents. Final output must be only the \
translated content, with the original structure/Markdown preserved. No explanations or extra text.

  Work in this order:
  1) Keep: For tags in the KEEP list, preserve the tag name but translate/replace only the content inside the brackets.
  2) Translate everything else to Vietnamese, keep units and medical values accurate, use active voice.

  Tag format examples: [123 Main St.]address, [John]first_name
  KEEP tags: ['street_address','mac_address','company_name','email','first_name','last_name',\
'phone_number','certificate_license_number','health_plan_beneficiary_number',\
'medical_record_number','national_id','ssn','bank_routing_number','vehicle_identifier',\
'credit_debit_card','date','device_identifier','date_of_birth','license_plate','ip_address',\
'biometric_identifier','account_number','fax_number','blood_type','time','user_name','country',\
'employee_id','pin','date_time','religious_belief','tax_id','postcode','customer_id','api_key',\
'language','cvv','race_ethnicity','swift_bic','password','url','sexuality','age','ipv4',\
'education_level','occupation','employment_status','county','political_view','http_cookie',\
'coordinate','state','city','ipv6','gender','unique_id']

  Localization rules:
  - Keep Markdown layout (headings, lists, line breaks).
  - Use formal, clear Vietnamese; always prefer active voice (avoid passive constructions).
  - address -> replace with a plausible Vietnamese address.
  - company_name -> replace with a Vietnamese hospital/department/company.
  - email -> keep exactly as provided.
  - first_name -> replace with a common Vietnamese first name (e.g., Nam, Linh, Anh, Hoa).
  - last_name -> replace with a common Vietnamese surname (e.g., Nguyen, Tran, Le, Pham).
  - phone_number -> use Vietnamese-style numbers; intl: +84 + number (no leading 0); domestic: 0 + area code + number.
  - date/date_of_birth -> use Vietnamese date style (e.g., 15/07/2024 or "ngay 15 thang 7 nam \
2024"); keep numeric accuracy.
  - device_identifier/vehicle_identifier/ip_address -> swap with plausible Vietnamese-style \
identifiers of the same format type.
  - id numbers:
    * MRN: 2 letters from {DN, CH, HC, HT, XB, TN, HS, SV, TE} + 13 digits (e.g., DN1234567890123).
    * SSN: keep 9-digit ###-##-#### format but treat as Vietnamese ID.
    * Bank account: 9-14 digits, plausible VN bank number.
    * License/certificate: 12-digit VN driver license pattern (e.g., 7901195002222: province code \
79, gender/year code, 7 random digits).

  Reminders:
  - Do not invent new tags or change tag names; only edit the bracketed content for KEEP tags.
  - No explanations or extra text.
"""


def clean_output(text: str) -> str:
    """Strip ``<think>...</think>`` reasoning blocks and surrounding whitespace.

    Returns:
        The text with every reasoning block removed and the result stripped. The match is
        non-greedy, case-insensitive and spans newlines, so several blocks in one response
        are each removed rather than everything between the first and last being eaten.

    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# reason: translate texts exposes texts/top p as its public contract; bundling would break callers.
def translate_texts(  # ruff: ignore[too-many-arguments]
    texts: list[str],
    *,
    model_id: str = DEFAULT_MODEL,
    max_model_len: int = 4000,
    temperature: float = 0.0,
    max_tokens: int = 128_000,
    top_p: float = 0.9,
) -> list[str]:
    """Translate tagged texts to Vietnamese with a local vLLM model.

    Loads the model and runs one batched ``generate`` call. Requires the optional
    ``vllm`` package and a GPU.

    Returns:
        One cleaned translation per input text, in input order. Each is the first candidate
        of its generation passed through ``clean_output``, so reasoning blocks never reach
        the caller.

    """
    vllm = _load_vllm()
    llm = vllm.LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
    )
    sampling = vllm.SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in texts
    ]
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
    return [clean_output(output.outputs[0].text) for output in outputs]


def translate_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_id: str = DEFAULT_MODEL,
) -> None:
    """Translate every ``text_tagged`` record in a JSONL file, one JSON per output."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records:
        return

    texts = [_record_text(record) for record in records]
    translations = translate_texts(texts, model_id=model_id)

    for idx, (source_text, translated) in enumerate(zip(texts, translations, strict=True)):
        payload = {"text_tagged": source_text, "translated": translated}
        out_path = output_dir / f"{idx:04d}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
