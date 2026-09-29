"""OpenAI-compatible provider catalog and environment resolution.

This module owns the provider specifications and the precedence rules that turn
provider-specific environment variables into keys, endpoints, and models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from meddies_pii.exceptions import ConfigurationError


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    default_model: str
    key_env_vars: tuple[str, ...]
    base_url_env_vars: tuple[str, ...] = ()
    base_urls_env_vars: tuple[str, ...] = ()
    model_env_vars: tuple[str, ...] = ()
    account_id_env_vars: tuple[str, ...] = ()
    """Providers that put an account id in the URL path (Cloudflare).

    The id(s) are read from these env vars and formatted into a ``{account_id}`` placeholder in ``base_url``, aligned
    one-per-key like ``base_urls_env_vars``.

    """
    extra_chat_payload: tuple[tuple[str, object], ...] = ()
    max_tokens_param: str = "max_tokens"
    """OpenAI reasoning models reject `max_tokens`; they need `max_completion_tokens`."""
    keyless: bool = False
    """Anonymous endpoints (OVH/LLM7/Pollinations): no key, no Authorization header.

    These are per-IP rate-limited, so keep concurrency low when using them.

    """


_DEFAULT_PROVIDER = "mimo"
_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
_MIMO_DEFAULT_MODEL = "mimo-v2-flash"
_OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/v1"
_OPENCODE_ZEN_DEFAULT_MODEL = "deepseek-v4-flash-free"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_MODEL = "google/gemma-4-31b-it:free"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_DEFAULT_MODEL = "qwen/qwen3-32b"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_CEREBRAS_DEFAULT_MODEL = "gpt-oss-120b"
_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
_MISTRAL_DEFAULT_MODEL = "mistral-medium-2505"
"""Dated versions carry the high per-model TPM; "*-latest" aliases are throttled."""
_CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
_CLOUDFLARE_DEFAULT_MODEL = "@cf/openai/gpt-oss-120b"
"""gpt-oss-120b over glm-4.7-flash and llama-3.3-70b-fp8-fast.

Glm emits markdown tables + mislabels (public URL as private_url), and llama-fp8-fast has the worst Cloudflare output
neuron rate (204,805/M, 3x gpt-oss). gpt-oss-120b produces clean [value]<label> output (same model Cerebras runs well) at a
reasonable ~105 neurons/accepted sample.

"""
_OVH_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
_OVH_DEFAULT_MODEL = "Qwen3.5-397B-A17B"
_LLM7_BASE_URL = "https://api.llm7.io/v1"
_LLM7_DEFAULT_MODEL = "qwen3-235b"
_POLLINATIONS_BASE_URL = "https://text.pollinations.ai/openai/v1"
_POLLINATIONS_DEFAULT_MODEL = "openai-fast"
_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_NIM_DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
"""llama-3.3-70b over minimax-m3.

Minimax is a reasoning model (burns tokens) and was DEGRADED/unavailable when probed; llama-3.3-70b is stable +
non-reasoning.

"""
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OPENAI_DEFAULT_MODEL = "gpt-5.4-mini"
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
"""Google AI Studio via its OpenAI-compatible surface.

The client appends /chat/completions, giving .../v1beta/openai/chat/completions (verified live).

"""
_GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
_GLM_CN_BASE_URL = "https://api.z.ai/api/paas/v4"
_GLM_CN_DEFAULT_MODEL = "glm-4.5-flash"
_SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"
_SAMBANOVA_DEFAULT_MODEL = "gpt-oss-120b"
_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
_MINIMAX_DEFAULT_MODEL = "MiniMax-M3"

_PROVIDERS: dict[str, ProviderSpec] = {
    "mimo": ProviderSpec(
        name="MIMO",
        base_url=_MIMO_BASE_URL,
        default_model=_MIMO_DEFAULT_MODEL,
        key_env_vars=("MIMO_API_KEYS", "MIMO_API_KEY", "XIAOMI_MIMO_API_KEY"),
        base_url_env_vars=("MIMO_BASE_URL", "XIAOMI_MIMO_BASE_URL"),
        base_urls_env_vars=("MIMO_BASE_URLS", "XIAOMI_MIMO_BASE_URLS"),
        model_env_vars=("MIMO_MODEL", "XIAOMI_MIMO_MODEL"),
    ),
    "opencode_zen": ProviderSpec(
        name="OpenCode Zen",
        base_url=_OPENCODE_ZEN_BASE_URL,
        default_model=_OPENCODE_ZEN_DEFAULT_MODEL,
        key_env_vars=("OPENCODE_ZEN_API_KEYS", "OPENCODE_ZEN_API_KEY"),
        base_url_env_vars=("OPENCODE_ZEN_BASE_URL",),
        base_urls_env_vars=("OPENCODE_ZEN_BASE_URLS",),
        model_env_vars=("OPENCODE_ZEN_MODEL",),
        extra_chat_payload=(("include_reasoning", False),),
    ),
    "openrouter": ProviderSpec(
        name="OpenRouter",
        base_url=_OPENROUTER_BASE_URL,
        default_model=_OPENROUTER_DEFAULT_MODEL,
        key_env_vars=("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
        base_url_env_vars=("OPENROUTER_BASE_URL",),
        base_urls_env_vars=("OPENROUTER_BASE_URLS",),
        model_env_vars=("OPENROUTER_MODEL",),
        extra_chat_payload=(("reasoning", {"effort": "none"}),),
    ),
    "groq": ProviderSpec(
        name="Groq",
        base_url=_GROQ_BASE_URL,
        default_model=_GROQ_DEFAULT_MODEL,
        key_env_vars=("GROQ_API_KEYS", "GROQ_API_KEY"),
        base_url_env_vars=("GROQ_BASE_URL",),
        base_urls_env_vars=("GROQ_BASE_URLS",),
        model_env_vars=("GROQ_MODEL",),
        extra_chat_payload=(("reasoning_effort", "none"),),
    ),
    "ovh": ProviderSpec(
        name="OVH AI Endpoints",
        base_url=_OVH_BASE_URL,
        default_model=_OVH_DEFAULT_MODEL,
        key_env_vars=("OVH_API_KEYS", "OVH_API_KEY"),
        base_url_env_vars=("OVH_BASE_URL",),
        base_urls_env_vars=("OVH_BASE_URLS",),
        model_env_vars=("OVH_MODEL",),
        keyless=True,
    ),
    "llm7": ProviderSpec(
        name="LLM7",
        base_url=_LLM7_BASE_URL,
        default_model=_LLM7_DEFAULT_MODEL,
        key_env_vars=("LLM7_API_KEYS", "LLM7_API_KEY"),
        base_url_env_vars=("LLM7_BASE_URL",),
        base_urls_env_vars=("LLM7_BASE_URLS",),
        model_env_vars=("LLM7_MODEL",),
        keyless=True,
        extra_chat_payload=(("chat_template_kwargs", {"enable_thinking": False}),),
    ),
    "pollinations": ProviderSpec(
        name="Pollinations",
        base_url=_POLLINATIONS_BASE_URL,
        default_model=_POLLINATIONS_DEFAULT_MODEL,
        key_env_vars=("POLLINATIONS_API_KEYS", "POLLINATIONS_API_KEY"),
        base_url_env_vars=("POLLINATIONS_BASE_URL",),
        base_urls_env_vars=("POLLINATIONS_BASE_URLS",),
        model_env_vars=("POLLINATIONS_MODEL",),
        keyless=True,
        extra_chat_payload=(("reasoning_effort", "low"),),
    ),
    "nim": ProviderSpec(
        name="NVIDIA NIM",
        base_url=_NIM_BASE_URL,
        default_model=_NIM_DEFAULT_MODEL,
        key_env_vars=(
            "NIM_API_KEYS",
            "NIM_API_KEY",
            "NVIDIA_API_KEYS",
            "NVIDIA_API_KEY",
        ),
        base_url_env_vars=("NIM_BASE_URL",),
        base_urls_env_vars=("NIM_BASE_URLS",),
        model_env_vars=("NIM_MODEL",),
    ),
    "cloudflare": ProviderSpec(
        name="Cloudflare Workers AI",
        base_url=_CLOUDFLARE_BASE_URL,
        default_model=_CLOUDFLARE_DEFAULT_MODEL,
        key_env_vars=("CLOUDFLARE_API_KEYS", "CLOUDFLARE_API_KEY"),
        base_url_env_vars=("CLOUDFLARE_BASE_URL",),
        base_urls_env_vars=("CLOUDFLARE_BASE_URLS",),
        account_id_env_vars=("CLOUDFLARE_API_IDS", "CLOUDFLARE_API_ID"),
        model_env_vars=("CLOUDFLARE_MODEL",),
        extra_chat_payload=(("reasoning_effort", "low"),),
    ),
    "mistral": ProviderSpec(
        name="Mistral",
        base_url=_MISTRAL_BASE_URL,
        default_model=_MISTRAL_DEFAULT_MODEL,
        key_env_vars=("MISTRAL_API_KEYS", "MISTRAL_API_KEY"),
        base_url_env_vars=("MISTRAL_BASE_URL",),
        base_urls_env_vars=("MISTRAL_BASE_URLS",),
        model_env_vars=("MISTRAL_MODEL",),
    ),
    "cerebras": ProviderSpec(
        name="Cerebras",
        base_url=_CEREBRAS_BASE_URL,
        default_model=_CEREBRAS_DEFAULT_MODEL,
        key_env_vars=("CEREBRAS_API_KEYS", "CEREBRAS_API_KEY"),
        base_url_env_vars=("CEREBRAS_BASE_URL",),
        base_urls_env_vars=("CEREBRAS_BASE_URLS",),
        model_env_vars=("CEREBRAS_MODEL",),
        extra_chat_payload=(("reasoning_effort", "low"),),
    ),
    "openai": ProviderSpec(
        name="OpenAI",
        base_url=_OPENAI_BASE_URL,
        default_model=_OPENAI_DEFAULT_MODEL,
        key_env_vars=("OPENAI_API_KEYS", "OPENAI_API_KEY"),
        base_url_env_vars=("OPENAI_BASE_URL",),
        base_urls_env_vars=("OPENAI_BASE_URLS",),
        model_env_vars=("OPENAI_MODEL",),
        extra_chat_payload=(("reasoning_effort", "none"),),
        max_tokens_param="max_completion_tokens",
    ),
    "gemini": ProviderSpec(
        name="Google AI Studio (Gemini)",
        base_url=_GEMINI_BASE_URL,
        default_model=_GEMINI_DEFAULT_MODEL,
        key_env_vars=("AI_STUDIO_API_KEYS", "GEMINI_API_KEY"),
        base_url_env_vars=("AI_STUDIO_BASE_URL",),
        base_urls_env_vars=("AI_STUDIO_BASE_URLS",),
        model_env_vars=("AI_STUDIO_MODEL", "GEMINI_MODEL"),
        extra_chat_payload=(("reasoning_effort", "none"),),
    ),
    "glm_cn": ProviderSpec(
        name="Z.AI (GLM)",
        base_url=_GLM_CN_BASE_URL,
        default_model=_GLM_CN_DEFAULT_MODEL,
        key_env_vars=("GLM_CN_API_KEYS", "GLM_CN_API_KEY"),
        base_url_env_vars=("GLM_CN_BASE_URL",),
        base_urls_env_vars=("GLM_CN_BASE_URLS",),
        model_env_vars=("GLM_CN_MODEL",),
        extra_chat_payload=(("thinking", {"type": "disabled"}),),
    ),
    "sambanova": ProviderSpec(
        name="SambaNova Cloud",
        base_url=_SAMBANOVA_BASE_URL,
        default_model=_SAMBANOVA_DEFAULT_MODEL,
        key_env_vars=("SAMBANOVA_API_KEYS", "SAMBANOVA_API_KEY"),
        base_url_env_vars=("SAMBANOVA_BASE_URL",),
        base_urls_env_vars=("SAMBANOVA_BASE_URLS",),
        model_env_vars=("SAMBANOVA_MODEL",),
    ),
    "minimax": ProviderSpec(
        name="MiniMax",
        base_url=_MINIMAX_BASE_URL,
        default_model=_MINIMAX_DEFAULT_MODEL,
        key_env_vars=("MINIMAX_API_KEYS", "MINIMAX_API_KEY"),
        base_url_env_vars=("MINIMAX_BASE_URL",),
        base_urls_env_vars=("MINIMAX_BASE_URLS",),
        model_env_vars=("MINIMAX_MODEL",),
        extra_chat_payload=(("thinking", {"type": "disabled"}),),
    ),
}
"""Reasoning-off form confirmed against OpenRouter API parameter docs reasoning-off via effort:none.

Research 2026-06-11: `enabled:false` 400s on mandatory-reasoning models). kimi-k2.6:free does not exist ->
gemma-4-31b:free, OpenRouter announcement, 2026-06-11. Some `:free` models force reasoning on and reject this; the field is
best-effort.

Limits are per-model and org-level (verified: each model returns its own independent x-ratelimit counters; Groq docs say
"organization level, not per key"). Rotate GROQ_MODEL to stack each model's quota; multiple keys in ONE org do NOT multiply
(only separate accounts do). qwen3 disables thinking via reasoning_effort "none" (verified clean).

OVH's gateway 400s on chat_template_kwargs (verified), so thinking can't be disabled via payload here — rely on robust
content extraction. Anonymous tier is 2 req/min/IP.

qwen3-235b is a thinking model; ask the upstream to skip it.

openai-fast is GPT-OSS 20B — minimise reasoning to save tokens.

minimax-m3 rejects reasoning_effort (400, verified) and is a reasoning model that spends tokens thinking — for bulk data
set NIM_MODEL to a non-reasoning instruct model. Free tier is 40 RPM, no rate-limit headers.

The account id lives in the URL path. Either set CLOUDFLARE_API_IDS (account id[s], formatted into the base_url) or
override the full URL with CLOUDFLARE_BASE_URL(S).

gpt-oss-120b is a reasoning model; keep thinking minimal so neurons feed the answer, not hidden reasoning.
reasoning_effort=low is accepted here (verified live); the glm models reject it (400) and need chat_template_kwargs instead
— so this param is model-specific.

gpt-oss-120b is a reasoning model; keep thinking minimal so the daily token budget (1M TPD) feeds completions, not hidden
reasoning.

gpt-5.4-mini live-verified (2026-06-11): thinking-off is reasoning_effort "none" (NOT "minimal" — unsupported), and it
needs max_completion_tokens.

Free tier is per PROJECT, and each AI_STUDIO key is a separate project, so multiple keys stack quota (unlike Groq's
single-org keys). Limits are small and per-minute (see ACCOUNT_RATE_LIMITS). gemini-3.5-flash is a thinking model;
reasoning_effort "none" disables thinking over the OpenAI-compat surface (verified accepted, clean [value]<label> output).
A restricted project returns 403 "denied access" -> the client trips that account out of rotation for the run (no
re-hammering = ban-safe).

z.ai international (no China real-name verification). glm-4.5-flash is the FREE model; the listed flagships
(glm-4.5/4.6/4.7/5.x) need a paid balance (429 "insufficient balance"). Flash defaults thinking ON; z.ai disables it with a
nested {"thinking": {"type": "disabled"}} body field (NOT the OpenAI reasoning_effort). glm-4.5-flash is the weakest of the
pool (lower private_url/secret yield) but free + uncapped -> high serial volume; its prose+JSON output matches the corpus's
intended format mix. Verified 2026-06-30.

Free tier is active only while NO payment card is linked. Limits are PER MODEL (~20 RPM / 20 RPD / 200K TPD each) -> rotate
SAMBANOVA_MODEL across gpt-oss-120b / DeepSeek-V3.1 / DeepSeek-V3.2 to stack quota + add diversity. gpt-oss-120b (our
proven model) verified 2026-06-30: 18 accepted, 0 rejected, all 9 labels. (Meta-Llama-3.3-70B, the first default, was too
weak.).

MiniMax-M3 (frontier, 1M ctx) is a reasoning model -> emits <think> blocks; disable with the nested {"thinking": {"type":
"disabled"}} body field (the reasoning_effort / chat_template_kwargs forms do NOT suppress it). Verified clean output
2026-06-30. NOTE: the current MINIMAX_API_KEYS are publicly shared keys -> ephemeral (re-drain to 429 "Token Plan usage
limit reached"); swap in an own-account key for reliability.

"""

_PROVIDER_ALIASES = {
    "opencode": "opencode_zen",
    "zen": "opencode_zen",
    "opencode-zen": "opencode_zen",
}

SUPPORTED_PROVIDER_NAMES = tuple(_PROVIDERS)
_DEFAULT_BASE_URL = _MIMO_BASE_URL
_DEFAULT_MODEL = _MIMO_DEFAULT_MODEL


def normalize_provider_name(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def get_provider_spec(provider: str) -> ProviderSpec:
    normalized = normalize_provider_name(provider)
    try:
        return _PROVIDERS[normalized]
    except KeyError as exc:
        msg = f"Unknown generation provider: {provider}"
        raise ConfigurationError(
            msg,
            context={"supported_providers": list(SUPPORTED_PROVIDER_NAMES)},
        ) from exc


def provider_keys(provider: str) -> list[str]:
    """Read provider keys using the catalog's environment precedence.

    Returns:
        The keys from the first environment variable in the catalog's precedence order that
        holds any, blanks dropped. First non-empty wins outright rather than the lists being
        merged, so a legacy variable left set cannot quietly add keys to a newer one. An empty
        list means no key is configured, which the caller distinguishes from a keyless provider.

    """
    for env_var in get_provider_spec(provider).key_env_vars:
        keys = _split_csv(os.getenv(env_var, ""))
        if keys:
            return keys
    return []


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_first_env(env_vars: tuple[str, ...]) -> str | None:
    for env_var in env_vars:
        value = os.getenv(env_var, "").strip()
        if value:
            return value
    return None


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        msg = "Provider base URL cannot be empty."
        raise ConfigurationError(msg)
    return normalized


def _resolve_base_urls(
    *,
    base_url: str,
    base_urls: list[str] | tuple[str, ...] | None,
    key_count: int,
) -> list[str]:
    if key_count <= 0:
        msg = "Cannot resolve base URLs without API keys."
        raise ConfigurationError(msg)
    if base_urls is None:
        return [_normalize_base_url(base_url)] * key_count

    clean_urls = [_normalize_base_url(url) for url in base_urls if url.strip()]
    if len(clean_urls) == 1:
        return clean_urls * key_count
    if len(clean_urls) != key_count:
        msg = "Provider base URL count must be 1 or match API key count."
        raise ConfigurationError(
            msg,
            context={"base_url_count": len(clean_urls), "api_key_count": key_count},
        )
    return clean_urls


def resolve_provider_base_urls(provider: str, key_count: int) -> list[str]:
    """Resolve provider endpoint URLs, optionally one per API key.

    `*_BASE_URLS` is comma-aligned with `*_API_KEYS`. A single URL still means
    "use this endpoint for all keys", preserving the original behavior. Explicit
    URLs win; otherwise account ids (Cloudflare) are formatted into the base_url
    `{account_id}` placeholder, one per id, comma-aligned with the keys.

    Returns:
        One endpoint URL per key, in key order, so index ``i`` of this list is the endpoint for
        key ``i``. Four sources are tried in precedence order -- the plural URL variable, the
        singular one, account ids formatted into the spec's placeholder, then the spec default
        -- and the first that yields anything wins. A single URL is broadcast to every key.

    """
    spec = get_provider_spec(provider)
    raw_base_urls = _read_first_env(spec.base_urls_env_vars)
    if raw_base_urls:
        return _resolve_base_urls(
            base_url=spec.base_url,
            base_urls=_split_csv(raw_base_urls),
            key_count=key_count,
        )
    single = _read_first_env(spec.base_url_env_vars)
    if single:
        return _resolve_base_urls(base_url=single, base_urls=None, key_count=key_count)
    raw_ids = _read_first_env(spec.account_id_env_vars)
    if raw_ids:
        urls = [spec.base_url.format(account_id=i) for i in _split_csv(raw_ids)]
        return _resolve_base_urls(base_url=spec.base_url, base_urls=urls, key_count=key_count)
    return _resolve_base_urls(base_url=spec.base_url, base_urls=None, key_count=key_count)


def resolve_provider_model(provider: str, model: str | None = None) -> str:
    """Resolve the effective model for a provider.

    Explicit CLI/API overrides win. Otherwise, use only that provider's model
    env vars before falling back to the provider default. This keeps targeted
    generation metadata aligned with the actual client request.

    Returns:
        The model name the request will actually carry: an explicit override once stripped,
        otherwise the first of that provider's own model variables that is set, otherwise the
        spec default. Reading only this provider's variables keeps a model set for one provider
        out of another's request, which would otherwise surface as a 404 mid-run.

    """
    explicit_model = (model or "").strip()
    if explicit_model:
        return explicit_model
    spec = get_provider_spec(provider)
    return _read_first_env(spec.model_env_vars) or spec.default_model
