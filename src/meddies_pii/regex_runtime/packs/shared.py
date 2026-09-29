"""Letter inventories shared by the Latin-script packs.

Every Latin-script pack keys its capture on letter case, so each one needs the same
two strings: the characters a proper token may open with, and the characters it may
continue with. Deriving both from one lowercase alphabet keeps them consistent --
a letter present in the body but absent from the head silently truncates a name.

The alphabet is the union across the Latin-script languages of the corpus rather
than a per-language subset. A pack only loses precision from a letter its language
never writes if that letter opens a word, which by definition it does not.
"""

from __future__ import annotations

import string

_ASCII_LOWERCASE = string.ascii_lowercase

_ACCENTED_LOWERCASE = "áàâäãåæçéèêëíìîïñóòôöõøœúùûüýÿšžßđ"
"""Accented lowercase letters of the corpus's Latin-script languages. The sharp s is here for the body only: its uppercase
form is two characters, so it never enters the head.
"""

LATIN_LOWERCASE = "".join(sorted(set(_ASCII_LOWERCASE + _ACCENTED_LOWERCASE)))
LATIN_UPPERCASE = "".join(sorted({letter.upper() for letter in LATIN_LOWERCASE if len(letter.upper()) == 1}))

NAME_INTERNAL_PUNCTUATION = "-" + chr(0x2011)
"""Hyphens join the parts of one name token (``Saint-Louis``, ``Pont-Neuf``). Both the plain and the non-breaking form
appear in the corpus, and a token split on either ends the capture early, so both continue a token.
"""

LATIN_BODY = LATIN_UPPERCASE + LATIN_LOWERCASE + NAME_INTERNAL_PUNCTUATION

CAPITALIZED_TEMPORAL_TOKENS: tuple[str, ...] = (
    "Montag",
    "Dienstag",
    "Mittwoch",
    "Donnerstag",
    "Samstag",
    "Sonntag",
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "September",
    "Oktober",
    "November",
    "Dezember",
)
"""German capitalizes weekdays and months, so these tokens guard opening-hours and appointment prose such as ``Klinik
Montag``. ``Freitag`` and ``August`` stay out: the locked multi-token names ``Praxis Freitag Berlin`` and ``Praxis
August Müller`` provide stronger name evidence than a bare temporal token. Shared-cue compilation gives every
``Klinik`` owner the same effective list.
"""


ENGLISH_HEADING_TOKENS: tuple[str, ...] = (
    "Course",
    "Day",
    "Days",
    "Admission",
    "Admissions",
    "Discharge",
    "Stay",
    "Ward",
    "Room",
    "Bed",
    "Visit",
    "Record",
    "Records",
    "Number",
    "Policy",
    "Protocol",
)
"""``Hospital`` is a cue in three of these inventories and an ordinary English word in every English document, where it
opens a section heading far more often than a name: ``Hospital Course``, ``Hospital Day 3``. English carries no pack of
its own -- its institution cues trail the name -- so the guard has to live on the packs that own the cue, and it is
shared because any of them may fire on English text.
"""

ENGLISH_TERMINAL_HEADING_TOKENS: tuple[str, ...] = ("Medicine",)
"""``Hospital Medicine`` is a service heading, while ``Hospital Medicine Associates`` has a further name core.
Terminal-only blocking preserves both.
"""
