"""Question normalisation.

Shared by the S1 exact FAQ stage, the cache key, and unmatched-question
clustering. All three must agree on what "the same question" means, so the
normaliser lives in one place.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_CONTRACTIONS = {
    "what's": "what is",
    "whats": "what is",
    "how's": "how is",
    "hows": "how is",
    "it's": "it is",
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "shouldn't": "should not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "you're": "you are",
    "youre": "you are",
    "there's": "there is",
    "that's": "that is",
    "thats": "that is",
    "let's": "let us",
    "i'd": "i would",
    "i'll": "i will",
}

# Units get canonicalised so "70 kgs", "70kg" and "70 kilograms" collapse.
_UNITS = [
    (r"\bkilograms?\b|\bkgs\b", "kg"),
    (r"\bpounds?\b|\blbs\b", "lb"),
    (r"\bcentimet(?:er|re)s?\b|\bcms\b", "cm"),
    (r"\bmillilit(?:er|re)s?\b", "ml"),
    (r"\bkilocalories?\b|\bcalories?\b|\bkcals\b", "kcal"),
    (r"\bgrams?\b|\bgms\b", "g"),
    (r"\bmilligrams?\b", "mg"),
    (r"\bmicrograms?\b|\bmcgs\b", "ug"),
    (r"\bhours?\b|\bhrs\b", "h"),
    (r"\bminutes?\b|\bmins\b", "min"),
]

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s%+./-]")
_NUM_SPACE_UNIT = re.compile(r"(\d)\s+(kg|lb|cm|ml|kcal|g|mg|ug|h|min)\b")


def normalise_question(text: str) -> str:
    """Lowercase, strip punctuation, expand contractions, canonicalise units."""
    if not text:
        return ""

    s = unicodedata.normalize("NFKC", text).lower().strip()

    for contraction, expansion in _CONTRACTIONS.items():
        s = re.sub(rf"\b{re.escape(contraction)}\b", expansion, s)

    for pattern, replacement in _UNITS:
        s = re.sub(pattern, replacement, s)

    s = _PUNCT.sub(" ", s)
    s = _NUM_SPACE_UNIT.sub(r"\1\2", s)
    s = _WS.sub(" ", s).strip()
    return s


def normalise_ingredient(token: str) -> str:
    """Ingredient tokens keep digits and hyphens (E-numbers, CAS, 'peg-40')."""
    if not token:
        return ""
    s = unicodedata.normalize("NFKD", token).lower().strip()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s\-/]", " ", s)
    s = _WS.sub(" ", s).strip()
    return s


def sha256_hex(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")  # unit separator: prevents ("ab","c") colliding with ("a","bc")
    return h.hexdigest()


def norm_hash(text: str, locale: str = "en") -> str:
    return sha256_hex(normalise_question(text), locale)


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    if not text or len(text) <= limit:
        return text or ""
    return text[: max(0, limit - len(suffix))] + suffix
