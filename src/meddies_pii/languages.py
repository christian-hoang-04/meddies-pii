from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal[
    "vi",
    "fr",
    "de",
    "es",
    "lo",
    "th",
    "my",
    "id",
    "fil",
    "ms",
    "ta",
    "pt",
    "ru",
    "zh",
    "ja",
    "ko",
    "en",
]
LanguageBucket = Literal["vi", "en", "other"]


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    name: str
    code: LanguageCode
    country_hint: str
    public_negative_hint: str
    adversarial_hint: str
    aliases: tuple[str, ...] = ()

    @property
    def config_slug(self) -> str:
        return self.name.lower()


def _public_negative_hint(language: str) -> str:
    return (
        f"Include one untagged public {language}-language guideline/article date or public hospital "
        "homepage URL, but do not tag it because it is not patient/person PII."
    )


def _adversarial_hint(language: str) -> str:
    return (
        f"Obfuscate using {language}'s OWN native conventions, not just English: native number-words for spoken "
        f"phone/ID digits, the native spoken words for email '@' and '.', the native spelling-alphabet; plus "
        "full-width/look-alike symbols, an emoji separator, and a line break inside one value. Vary the technique "
        "across different PII values and keep [value]<label> wrapping the full obfuscated surface."
    )


# reason: profile keeps name/aliases at its adapter seam; bundling would hide required inputs.
def _profile(  # ruff: ignore[too-many-arguments]
    *,
    name: str,
    code: LanguageCode,
    country_hint: str,
    public_negative_hint: str | None = None,
    adversarial_hint: str | None = None,
    aliases: tuple[str, ...] = (),
) -> LanguageProfile:
    return LanguageProfile(
        name=name,
        code=code,
        country_hint=country_hint,
        public_negative_hint=public_negative_hint or _public_negative_hint(name),
        adversarial_hint=adversarial_hint or _adversarial_hint(name),
        aliases=aliases,
    )


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "vietnamese": LanguageProfile(
        name="Vietnamese",
        code="vi",
        country_hint=(
            "Vietnamese healthcare, hospitals, clinics, insurers, patient portals, and administrative forms in Vietnam"
        ),
        public_negative_hint=(
            "Include one untagged public guideline/article date or public hospital homepage URL, "
            "but do not tag it because it is not patient/person PII."
        ),
        adversarial_hint=(
            "Use Vietnamese native obfuscation: phone/ID digits as Vietnamese number-words (không/một/hai/ba/...), "
            "email '@' as 'a còng'/'a móc' and '.' as 'chấm'/'chấm com', native spelling; plus full-width symbols, "
            "an emoji separator, and a line break inside one value. Keep [value]<label> on the full surface."
        ),
        aliases=("vi", "vie", "vn"),
    ),
    "french": _profile(
        name="French",
        code="fr",
        country_hint=(
            "French healthcare: hôpitaux, cliniques, Assurance Maladie / CPAM, mutuelles, and patient "
            "portals like Doctolib and Mon espace santé in France"
        ),
        aliases=("fr", "fra", "fre"),
    ),
    "german": _profile(
        name="German",
        code="de",
        country_hint=(
            "German healthcare: Krankenhäuser, Arztpraxen, gesetzliche/private Krankenkassen, and "
            "patient portals (elektronische Patientenakte) in Germany"
        ),
        aliases=("de", "deu", "ger"),
    ),
    "spanish": _profile(
        name="Spanish",
        code="es",
        country_hint=(
            "Spanish-language healthcare: hospitales, clínicas, Seguridad Social / aseguradoras, and "
            "patient portals across Spain and Latin America"
        ),
        aliases=("es", "spa"),
    ),
    "laos": _profile(
        name="Laos",
        code="lo",
        country_hint=(
            "Lao healthcare: ໂຮງໝໍ (hospitals), clinics, the Ministry of Health, social health "
            "insurance, and patient records in Laos"
        ),
        public_negative_hint=_public_negative_hint("Lao"),
        adversarial_hint=_adversarial_hint("Lao"),
        aliases=("lo", "lao"),
    ),
    "thai": _profile(
        name="Thai",
        code="th",
        country_hint=(
            "Thai healthcare: โรงพยาบาล (hospitals), clinics, the National Health Security Office "
            "(สปสช.), social security insurance, and patient portals in Thailand"
        ),
        aliases=("th", "tha"),
    ),
    "burmese": _profile(
        name="Burmese",
        code="my",
        country_hint=(
            "Myanmar healthcare: ဆေးရုံ (hospitals), clinics, the Ministry of Health, and patient records in Myanmar"
        ),
        aliases=("my", "mya"),
    ),
    "indonesian": _profile(
        name="Indonesian",
        code="id",
        country_hint=(
            "Indonesian healthcare: rumah sakit, puskesmas, klinik, BPJS Kesehatan, and patient "
            "portals (Mobile JKN) in Indonesia"
        ),
        aliases=("id", "ind"),
    ),
    "filipino": _profile(
        name="Filipino",
        code="fil",
        country_hint=(
            "Philippine healthcare: ospital, health centers, PhilHealth, HMOs, and patient portals in the Philippines"
        ),
        aliases=("fil", "tl", "tgl"),
    ),
    "malay": _profile(
        name="Malay",
        code="ms",
        country_hint=(
            "Malaysian healthcare: hospital, klinik kesihatan, the Ministry of Health (KKM), "
            "insurers/takaful, and patient portals (MySejahtera) in Malaysia"
        ),
        aliases=("ms", "msa"),
    ),
    "tamil": _profile(
        name="Tamil",
        code="ta",
        country_hint=(
            "Tamil healthcare: மருத்துவமனை (hospitals), clinics, government health schemes, insurers, "
            "and patient records in Tamil Nadu and Sri Lanka"
        ),
        aliases=("ta", "tam"),
    ),
    "portuguese": _profile(
        name="Portuguese",
        code="pt",
        country_hint=(
            "Portuguese-language healthcare: hospitais, clínicas, SUS / planos de saúde, and patient "
            "portals across Brazil and Portugal"
        ),
        aliases=("pt", "por"),
    ),
    "russian": _profile(
        name="Russian",
        code="ru",
        country_hint=(
            "Russian healthcare: больницы, поликлиники, the OMS compulsory medical insurance fund, "
            "and patient portals (Госуслуги) in Russia"
        ),
        aliases=("ru", "rus"),
    ),
    "chinese": _profile(
        name="Chinese",
        code="zh",
        country_hint=(
            "Chinese healthcare: 医院 (hospitals), 社区卫生服务中心 (clinics), 医保 (medical insurance), and "
            "patient portals across mainland China"
        ),
        aliases=("zh", "zho", "chi"),
    ),
    "japanese": _profile(
        name="Japanese",
        code="ja",
        country_hint=(
            "Japanese healthcare: 病院 (hospitals), 診療所 (clinics), 健康保険 (health insurance), and patient "
            "portals in Japan"
        ),
        aliases=("ja", "jpn"),
    ),
    "korean": _profile(
        name="Korean",
        code="ko",
        country_hint=(
            "Korean healthcare: 병원 (hospitals), 의원 (clinics), 국민건강보험 (National Health Insurance), and "
            "patient portals in South Korea"
        ),
        aliases=("ko", "kor"),
    ),
    "english": LanguageProfile(
        name="English",
        code="en",
        country_hint=(
            "English-language healthcare, hospital, insurer, pharmacy, patient portal, and employer/clinic documentation"
        ),
        public_negative_hint=(
            "Include one untagged public guideline publication date or public article/copyright date, "
            "and one public homepage/documentation URL, but do not tag them."
        ),
        adversarial_hint=(
            "Use adversarial PII forms: at-dot email obfuscation, full-width/look-alike symbol substitution, an emoji "
            "separator, line-broken phone/private URL, digit-words, and NATO/phonetic spelling around one identifier."
        ),
        aliases=("en", "eng", "english", "us", "uk", "intl"),
    ),
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(profile.name for profile in LANGUAGE_PROFILES.values())
GRID_LANGUAGE_CODES: tuple[LanguageCode, ...] = tuple(profile.code for profile in LANGUAGE_PROFILES.values())
LANGUAGE_CONFIG_SLUGS: tuple[str, ...] = tuple(sorted(profile.config_slug for profile in LANGUAGE_PROFILES.values()))
MEDDIES_PII_LANGUAGE_CONFIGS: tuple[str, ...] = (
    *LANGUAGE_CONFIG_SLUGS,
    "vietnamese-translated",
)
LANGUAGE_ALIASES: dict[str, str] = {key: key for key in LANGUAGE_PROFILES}
LANGUAGE_ALIASES.update({profile.name.lower(): key for key, profile in LANGUAGE_PROFILES.items()})
LANGUAGE_ALIASES.update({alias: key for key, profile in LANGUAGE_PROFILES.items() for alias in profile.aliases})


def normalize_language(language: str) -> LanguageProfile:
    key = language.strip().lower()
    key = LANGUAGE_ALIASES.get(key, key)
    try:
        return LANGUAGE_PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(profile.name for profile in LANGUAGE_PROFILES.values())
        msg = f"Unsupported Meddies PII language: {language}. Supported: {supported}"
        raise ValueError(msg) from exc


def language_bucket(*, language: str, source: str = "") -> LanguageBucket:
    normalized = language.strip().lower()
    source_normalized = source.strip().lower()
    if normalized in {"vi", "vie", "vietnamese"} or source_normalized in {
        "vietnamese",
        "vietnamese-translated",
    }:
        return "vi"
    if normalized in {"en", "eng", "english", "us", "uk", "intl"}:
        return "en"
    return "other"


def is_supported_meddies_language(language: str) -> bool:
    normalized = language.strip().lower()
    return normalized in LANGUAGE_ALIASES
