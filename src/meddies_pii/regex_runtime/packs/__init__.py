"""Versioned pattern packs consumed by the regex lane.

Adding a language means adding a pack to ``ALL_PACKS``; the compiler and the merge
path stay untouched.
"""

from __future__ import annotations

# ruff: file-ignore[non-empty-init-module]
# reason: this package root is the stable registry and re-export facade consumed by the manifest compiler.
from meddies_pii.regex_runtime.packs.cjk import (
    JA_ORG_PACK,
    KO_ORG_PACK,
    ZH_ORG_PACK,
)
from meddies_pii.regex_runtime.packs.indic_slavic import RU_ORG_PACK, TA_ORG_PACK
from meddies_pii.regex_runtime.packs.latin import (
    DE_ORG_PACK,
    EN_ORG_PACK,
    ES_ORG_PACK,
    FIL_ORG_PACK,
    FIL_OSPITAL_ORG_PACK,
    FR_ORG_PACK,
    ID_ORG_PACK,
    MS_ORG_PACK,
    PT_ORG_PACK,
)
from meddies_pii.regex_runtime.packs.pack import (
    Cue,
    CuePack,
    pack_pattern,
    pack_payload,
)
from meddies_pii.regex_runtime.packs.sea import (
    LO_ORG_PACK,
    MY_ORG_PACK,
    TH_ORG_PACK,
)
from meddies_pii.regex_runtime.packs.vi import VI_ORG_PACK

ALL_PACKS: tuple[CuePack, ...] = (
    VI_ORG_PACK,
    DE_ORG_PACK,
    ES_ORG_PACK,
    FR_ORG_PACK,
    ID_ORG_PACK,
    MS_ORG_PACK,
    FIL_ORG_PACK,
    FIL_OSPITAL_ORG_PACK,
    PT_ORG_PACK,
    EN_ORG_PACK,
    JA_ORG_PACK,
    ZH_ORG_PACK,
    KO_ORG_PACK,
    TH_ORG_PACK,
    LO_ORG_PACK,
    MY_ORG_PACK,
    TA_ORG_PACK,
    RU_ORG_PACK,
)

__all__ = ["ALL_PACKS", "Cue", "CuePack", "pack_pattern", "pack_payload"]
