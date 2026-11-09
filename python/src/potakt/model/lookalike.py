"""Look-alike ids: Greek capitals that look like Latin ones (ADR-0008 D9, TASK-005e1).

Ids are compared exactly; this module only explains a miss. In the spirit of the confusable
"skeleton" of Unicode TR39 (restricted to the 14 Greek capitals that have a Latin look-alike), two
ids are look-alikes when their skeletons are equal. Standard library only.
"""

import unicodedata
from collections.abc import Collection

# Greek capital letter -> its Latin look-alike (U+0391 U+0392 U+0395 U+0396 U+0397 U+0399 U+039A
# U+039C U+039D U+039F U+03A1 U+03A4 U+03A5 U+03A7 -> A B E Z H I K M N O P T Y X).
_LATIN_LOOKALIKES = {
    "\N{GREEK CAPITAL LETTER ALPHA}": "A",
    "\N{GREEK CAPITAL LETTER BETA}": "B",
    "\N{GREEK CAPITAL LETTER EPSILON}": "E",
    "\N{GREEK CAPITAL LETTER ZETA}": "Z",
    "\N{GREEK CAPITAL LETTER ETA}": "H",
    "\N{GREEK CAPITAL LETTER IOTA}": "I",
    "\N{GREEK CAPITAL LETTER KAPPA}": "K",
    "\N{GREEK CAPITAL LETTER MU}": "M",
    "\N{GREEK CAPITAL LETTER NU}": "N",
    "\N{GREEK CAPITAL LETTER OMICRON}": "O",
    "\N{GREEK CAPITAL LETTER RHO}": "P",
    "\N{GREEK CAPITAL LETTER TAU}": "T",
    "\N{GREEK CAPITAL LETTER UPSILON}": "Y",
    "\N{GREEK CAPITAL LETTER CHI}": "X",
}
_SKELETON = str.maketrans(_LATIN_LOOKALIKES)


def skeleton(text: str) -> str:
    """``text`` with each Greek capital that has a Latin look-alike replaced by that letter."""
    return text.translate(_SKELETON)


def lookalike(unknown: str, known: Collection[str]) -> str | None:
    """The one id of ``known`` that is a look-alike of ``unknown`` (``unknown`` itself if it is
    known), or ``None`` when no id, or more than one, has its skeleton."""
    if unknown in known:
        return unknown
    target = skeleton(unknown)
    matches = {candidate for candidate in known if skeleton(candidate) == target}
    return matches.pop() if len(matches) == 1 else None


def lookalike_hint(unknown: str, known: Collection[str]) -> str:
    """``"; did you mean '<id>'? (<NAME> where the data has <NAME>, ...)"`` for an unknown id
    with exactly one look-alike in ``known``, naming each differing character; ``""`` otherwise.
    """
    match = lookalike(unknown, known)
    if match is None or match == unknown:
        return ""
    pairs = ", ".join(
        f"{unicodedata.name(written)} where the data has {unicodedata.name(expected)}"
        for written, expected in zip(unknown, match, strict=True)
        if written != expected
    )
    return f"; did you mean {match!r}? ({pairs})"
