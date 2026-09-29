"""Latin-script organization-pack declarations."""

from __future__ import annotations

from meddies_pii.regex_runtime.packs.pack import Cue, CuePack
from meddies_pii.regex_runtime.packs.shared import (
    CAPITALIZED_TEMPORAL_TOKENS,
    ENGLISH_HEADING_TOKENS,
    ENGLISH_TERMINAL_HEADING_TOKENS,
    LATIN_BODY,
    LATIN_UPPERCASE,
)

DE_CUES: tuple[Cue, ...] = (
    Cue("Universitätsklinikum", "pre"),
    Cue("Universitätsklinik", "pre"),
    Cue("Klinikum", "pre"),
    Cue("Klinik", "pre"),
    Cue("Krankenhaus", "pre"),
    Cue("Medizinisches Versorgungszentrum", "pre"),
    Cue("MVZ", "pre"),
    Cue("Gemeinschaftspraxis", "pre"),
    Cue("Facharztpraxis", "pre"),
    Cue("Zahnarztpraxis", "pre"),
    Cue("Praxis", "pre"),
    Cue("Apotheke", "pre"),
    Cue("Labor", "pre"),
    Cue("Pflegeheim", "pre"),
    Cue("Seniorenheim", "pre"),
    Cue("Gesundheitszentrum", "pre"),
)

DE_NAME_CONTINUATIONS: tuple[str, ...] = (
    "an",
    "der",
    "des",
    "für",
    "von",
    "zum",
    "zur",
    "Dr.",
    "Prof.",
    "St.",
    "Sankt",
)
"""``Dr.`` and ``St.`` end in a period, which is not a token separator, so without them a practice named after a person
captures as ``Praxis Dr`` -- a truncated span the union lane would then have to widen.
"""

DE_BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    *CAPITALIZED_TEMPORAL_TOKENS,
    *ENGLISH_HEADING_TOKENS,
    "eingewiesen",
    "Innere",
    "Medizin",
)
"""The two shared lists are the whole blocked inventory here: the temporal one was written for exactly this language's
opening-hours lines, and German adds no category word of its own that a capitalized token could follow.
"""

DE_ORG_PACK = CuePack(
    pack_id="de_org_prefix",
    version=1,
    language="de",
    label="company_name",
    tier="CONTEXT",
    cues=DE_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=DE_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=DE_BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=10,
    min_adjacent_digits=2,
    require_capitalized_cue=True,
)


ES_CUES: tuple[Cue, ...] = (
    Cue("Hospital Universitario", "pre"),
    Cue("Hospital", "pre"),
    Cue("Policlínica", "pre"),
    Cue("Clínica", "pre"),
    Cue("Sanatorio", "pre"),
    Cue("Centro Médico", "pre"),
    Cue("Centro de Salud", "pre"),
    Cue("Consultorio", "pre"),
    Cue("Laboratorio", "pre"),
    Cue("Farmacia", "pre"),
    Cue("Instituto", "pre"),
)

ES_NAME_CONTINUATIONS: tuple[str, ...] = (
    "de",
    "del",
    "la",
    "las",
    "los",
    "Dr.",
    "Dra.",
)

ES_BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    *ENGLISH_HEADING_TOKENS,
    "más",
    "cercano",
    "cercana",
    "local",
    "público",
    "pública",
    "privado",
    "privada",
    "correspondiente",
    "barrio",
    "OCR",
    "Ministerio",
    "Asociada",
    "Asignada",
)
"""Adjectives that qualify a facility rather than naming one."""

ES_ORG_PACK = CuePack(
    pack_id="es_org_prefix",
    version=1,
    language="es",
    label="company_name",
    tier="CONTEXT",
    cues=ES_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=ES_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=ES_BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=8,
    min_adjacent_digits=2,
    blocked_terminal_tokens=ENGLISH_TERMINAL_HEADING_TOKENS,
    blocked_singleton_tokens=("Referencia",),
    require_capitalized_cue=True,
)


FR_CUES: tuple[Cue, ...] = (
    Cue("Centre Hospitalier Universitaire", "pre"),
    Cue("Centre Hospitalier Régional", "pre"),
    Cue("Centre Hospitalier", "pre"),
    Cue("Centre de Santé", "pre"),
    Cue("Centre Médical", "pre"),
    Cue("Groupe Hospitalier", "pre"),
    Cue("Maison de Santé", "pre"),
    Cue("Hôpital", "pre"),
    Cue("Polyclinique", "pre"),
    Cue("Clinique", "pre"),
    Cue("Cabinet Médical", "pre"),
    Cue("Laboratoire", "pre"),
    Cue("Pharmacie", "pre"),
    Cue("Institut", "pre"),
    Cue("CHU", "pre"),
    Cue("CHR", "pre"),
    Cue("EHPAD", "pre"),
)

FR_NAME_CONTINUATIONS: tuple[str, ...] = (
    "de",
    "du",
    "des",
    "la",
    "le",
    "les",
    "sur",
    "sous",
)

FR_BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    *ENGLISH_HEADING_TOKENS,
    "plus",
    "proche",
    "référent",
    "voisin",
    "local",
    "locale",
    "public",
    "publique",
    "privé",
    "privée",
    "général",
    "générale",
    "jour",
    "OCR",
)
"""Adjectives that qualify a facility rather than naming one."""

FR_ORG_PACK = CuePack(
    pack_id="fr_org_prefix",
    version=1,
    language="fr",
    label="company_name",
    tier="CONTEXT",
    cues=FR_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=FR_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=FR_BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=8,
    min_adjacent_digits=2,
    require_capitalized_cue=True,
)


PT_CUES: tuple[Cue, ...] = (
    Cue("Hospital Universitário", "pre"),
    Cue("Hospital", "pre"),
    Cue("Policlínica", "pre"),
    Cue("Clínica", "pre"),
    Cue("Centro Médico", "pre"),
    Cue("Centro de Saúde", "pre"),
    Cue("Consultório", "pre"),
    Cue("Laboratório", "pre"),
    Cue("Farmácia", "pre"),
    Cue("Instituto", "pre"),
)

PT_NAME_CONTINUATIONS: tuple[str, ...] = (
    "de",
    "da",
    "do",
    "das",
    "dos",
    "Dr.",
    "Dra.",
)

PT_BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    *ENGLISH_HEADING_TOKENS,
    "mais",
    "próximo",
    "próxima",
    "local",
    "público",
    "pública",
    "privado",
    "privada",
    "referência",
    "unidade",
    "básica",
    "OCR",
    "Dr",
    "Retirada",
    "Hospitalar",
    "Origem",
    "Responsável",
    "Geral",
    "Associada",
)
"""Adjectives that qualify a facility rather than naming one."""

PT_ORG_PACK = CuePack(
    pack_id="pt_org_prefix",
    version=1,
    language="pt",
    label="company_name",
    tier="CONTEXT",
    cues=PT_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=PT_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=PT_BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=8,
    min_adjacent_digits=2,
    blocked_terminal_tokens=ENGLISH_TERMINAL_HEADING_TOKENS,
    require_capitalized_cue=True,
)


ID_CUES: tuple[Cue, ...] = (
    Cue("Rumah Sakit Ibu dan Anak", "pre"),
    Cue("Rumah Sakit Umum Daerah", "pre"),
    Cue("Rumah Sakit Umum", "pre"),
    Cue("Rumah Sakit", "pre"),
    Cue("RSUD", "pre"),
    Cue("RSIA", "pre"),
    Cue("RSU", "pre"),
    Cue("Puskesmas", "pre"),
    Cue("Poliklinik", "pre"),
    Cue("Klinik", "pre"),
    Cue("Apotek", "pre"),
    Cue("Laboratorium", "pre"),
)

ID_NAME_CONTINUATIONS: tuple[str, ...] = ()
"""No name-internal generic words: the one candidate, the conjunction ``dan``, merges two coordinated institution names
into a single span.
"""

ID_BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    *ENGLISH_HEADING_TOKENS,
    "terdekat",
    "rujukan",
    "swasta",
    "pemerintah",
    "setempat",
    "tersebut",
    "lain",
    "ini",
    "itu",
    "Anda",
    "Umum",
    "Pasien",
    "Tujuan",
)
"""Words that make the cue a category rather than a name."""

ID_ORG_PACK = CuePack(
    pack_id="id_org_prefix",
    version=1,
    language="id",
    label="company_name",
    tier="CONTEXT",
    cues=ID_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=ID_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=ID_BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=8,
    min_adjacent_digits=2,
    require_capitalized_cue=True,
)

MS_CUES: tuple[Cue, ...] = (
    Cue("Hospital", "pre"),
    Cue("Pusat Perubatan", "pre"),
    Cue("Pusat Kesihatan", "pre"),
    Cue("Klinik Kesihatan", "pre"),
    Cue("Klinik Pakar", "pre"),
    Cue("Poliklinik", "pre"),
    Cue("Klinik", "pre"),
    Cue("Farmasi", "pre"),
    Cue("Makmal", "pre"),
)

MS_NAME_CONTINUATIONS: tuple[str, ...] = ()
"""No name-internal generic words: the one candidate, the conjunction ``dan``, merges two coordinated institution names
into a single span.
"""

MS_BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    *ENGLISH_HEADING_TOKENS,
    "berdekatan",
    "berhampiran",
    "terdekat",
    "kerajaan",
    "swasta",
    "pesakit",
    "rujukan",
    "lain",
    "ini",
    "itu",
    "OCR",
    "Pengirim",
    "Ortopedik",
    "Kesihatan",
    "Bertugas",
    "Contoh",
)
"""Words that make the cue a category rather than a name."""

MS_ORG_PACK = CuePack(
    pack_id="ms_org_prefix",
    version=1,
    language="ms",
    label="company_name",
    tier="CONTEXT",
    cues=MS_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=MS_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=MS_BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=8,
    min_adjacent_digits=2,
    blocked_terminal_tokens=ENGLISH_TERMINAL_HEADING_TOKENS,
    require_capitalized_cue=True,
)


FIL_CUES: tuple[Cue, ...] = (
    Cue("Pagamutan", "pre"),
    Cue("Klinika", "pre"),
    Cue("Botika", "pre"),
    Cue("Laboratoryo", "pre"),
)

FIL_NAME_CONTINUATIONS: tuple[str, ...] = (
    "ng",
    "sa",
    "ni",
)

FIL_BLOCKED_ADJACENT_TOKENS: tuple[str, ...] = (
    *ENGLISH_HEADING_TOKENS,
    "na",
    "malapit",
    "pinakamalapit",
    "publiko",
    "pribado",
    "ito",
    "iyon",
    "Telepono",
    "Address",
)
"""Words that make the cue a category rather than a name."""

FIL_ORG_PACK = CuePack(
    pack_id="fil_org_prefix",
    version=1,
    language="fil",
    label="company_name",
    tier="CONTEXT",
    cues=FIL_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=FIL_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=FIL_BLOCKED_ADJACENT_TOKENS,
    max_name_tokens=8,
    min_adjacent_digits=2,
    require_capitalized_cue=True,
)

FIL_OSPITAL_ORG_PACK = CuePack(
    pack_id="fil_org_ospital_prefix",
    version=1,
    language="fil",
    label="company_name",
    tier="CONTEXT",
    cues=(Cue("Ospital", "pre"),),
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=FIL_NAME_CONTINUATIONS,
    blocked_adjacent_tokens=FIL_BLOCKED_ADJACENT_TOKENS,
    blocked_singleton_tokens=("Bayan",),
    max_name_tokens=8,
    min_adjacent_digits=2,
    require_capitalized_cue=True,
)


EN_CUES: tuple[Cue, ...] = (
    Cue("Health Services", "post"),
    Cue("Health Systems", "post"),
    Cue("Medical Center", "post"),
    Cue("Healthcare", "post"),
    Cue("Laboratory", "post"),
    Cue("Pharmacy", "post"),
    Cue("Hospital", "post"),
    Cue("Clinic", "post"),
    Cue("Health", "post"),
)

EN_ORG_PACK = CuePack(
    pack_id="en_org_suffix",
    version=1,
    language="en",
    label="company_name",
    tier="CONTEXT",
    cues=EN_CUES,
    head_letters=LATIN_UPPERCASE,
    body_letters=LATIN_BODY,
    name_continuations=(),
    blocked_adjacent_tokens=("Discharged", "Referred", "The"),
    blocked_singleton_tokens=(
        "Referring",
        "Primary",
        "Public",
        "Updated",
        "Sending",
        "Professional",
        "Preferred",
        "Performing",
        "Other",
        "Initial",
        "Home",
        "General",
        "For",
        "Call",
        "CDC",
        "Attending",
        "Admitting",
    ),
    max_name_tokens=6,
    min_adjacent_digits=2,
    suffix_terminal_tokens=("Inc", "LLC", "Ltd"),
)
