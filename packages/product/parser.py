"""Ingredient panel parsing (arch.md 8.3 steps 1-2).

A dedicated normaliser, not a regex in the request path. Two things the old
code got wrong and this fixes:

  * parentheticals were discarded. On an INCI panel "(and)" joins co-supplied
    components and "(nano)" is a regulatory qualifier that changes the hazard
    assessment — they are kept as qualifiers, not dropped.
  * declaration order was not preserved through dedupe. Order carries
    concentration meaning on an INCI panel; losing it loses the only
    concentration signal the label gives.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MAX_TOKENS = 60

# Splitting on commas alone breaks "CI 77491 (Iron Oxides), Aqua" badly, so
# separators are only honoured at paren depth zero (see _split_top_level).
_SEPARATORS = ",;•·|\n"

_LEADING_NUMBER = re.compile(r"^\s*\(?\d+[\).]\s*")
_PERCENTAGE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")
_QUANTITY = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|µg|g|kg|ml|l|iu|ppm)\b", re.IGNORECASE)
_ASTERISK = re.compile(r"[*†‡]+")
_WHITESPACE = re.compile(r"\s+")

# Qualifiers worth keeping: they change how the chemical is assessed.
_KNOWN_QUALIFIERS = re.compile(
    r"^(nano|and|organic|certified\s+organic|fair\s+trade|natural|synthetic|"
    r"from\s+.+|derived\s+from\s+.+|as\s+preservative|preservative|antioxidant|"
    r"emulsifier|stabiliser|stabilizer|colou?r|flavou?r|acidity\s+regulator|"
    r"anti[-\s]?caking\s+agent|humectant|thickener|raising\s+agent)$",
    re.IGNORECASE,
)

# Trailing prose that is not an ingredient.
_NOISE = re.compile(
    r"^(may\s+contain.*|contains?\s+(?:less\s+than|no)\s+.*|and\s+other\s+ingredients?|"
    r"other\s+ingredients?|list\s+subject\s+to\s+change|for\s+external\s+use.*|"
    r"see\s+(?:carton|pack|label).*|\W*)$",
    re.IGNORECASE,
)

# "MAY CONTAIN" opens the allergen trace section on food packs, and on
# cosmetics it opens the colourant list. Both are declared separately.
#
# The optional "traces of" has to be part of the marker, not left behind: split
# on "may contain" alone and "May contain traces of Peanuts" yields a token
# called "traces of Peanuts", which then fails to resolve and — worse — would
# not match a declared peanut allergy by word boundary.
_TRACE_MARKER = re.compile(
    r"\bmay\s+contain(?:\s+traces?\s+of)?\b|\bcontains\s+traces?\s+of\b|\+/-",
    re.IGNORECASE,
)


@dataclass
class ParsedToken:
    raw: str
    text: str
    position: int
    qualifiers: list[str] = field(default_factory=list)
    is_trace: bool = False


@dataclass
class ParsedPanel:
    tokens: list[ParsedToken] = field(default_factory=list)
    truncated: bool = False
    had_header: bool = False
    dropped: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.tokens)


def _split_top_level(text: str) -> list[str]:
    """Split on separators that sit outside brackets.

    "CI 77491 (Iron Oxides, Titanium Dioxide), Aqua" must yield two tokens, not
    three — the inner comma belongs to the parenthetical.
    """
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0

    for char in text:
        if char in "([{":
            depth += 1
            buffer.append(char)
        elif char in ")]}":
            depth = max(0, depth - 1)
            buffer.append(char)
        elif char in _SEPARATORS and depth == 0:
            parts.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)

    if buffer:
        parts.append("".join(buffer))
    return parts


def _extract_qualifiers(text: str) -> tuple[str, list[str]]:
    """Pull parentheticals out, keeping the recognised ones as qualifiers.

    An unrecognised parenthetical is usually the common name for the INCI term
    ("Tocopherol (Vitamin E)"). It is kept as a qualifier too, because the
    resolver can fall back to matching on it when the INCI term misreads.
    """
    qualifiers: list[str] = []

    def take(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if inner:
            qualifiers.append(inner)
        return " "

    stripped = re.sub(r"\(([^()]*)\)", take, text)
    while "(" in stripped and ")" in stripped:
        new = re.sub(r"\(([^()]*)\)", take, stripped)
        if new == stripped:
            break
        stripped = new

    return stripped, qualifiers


_OPENERS = {"(": ")", "[": "]"}


def _split_compound(raw_part: str) -> Optional[tuple[str, str]]:
    """(parent, inner) when this token is "Name (a, b, c)", else None.

    Depth-counted rather than regex-matched, because the brackets nest. The
    real Maggi panel reads:

        Masala Tastemaker (hydrolysed groundnut protein, mixed spices,
        Noodle powder, ..., Thickeners (508 & 412), acidity regulators
        (501i & 500i), Sugar, ...)

    A regex whose inner group excluded parentheses could not match that at
    all, so the compound was never expanded and the groundnut stayed hidden.
    """
    text = raw_part.strip()
    start = next((i for i, ch in enumerate(text) if ch in _OPENERS), -1)
    if start <= 0:
        return None

    closer = _OPENERS[text[start]]
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char in _OPENERS:
            depth += 1
        elif char in (")", "]"):
            depth -= 1
            if depth == 0:
                # The bracket must close at the very end; anything after it
                # means this is not a simple "Name (list)" shape.
                if text[index + 1:].strip():
                    return None
                if char != closer:
                    return None
                return text[:start].strip(), text[start + 1:index]
    return None


def _expand_compound(raw_part: str) -> list[str]:
    """Split a compound ingredient into itself plus its sub-ingredients.

    Real labels declare allergens inside these brackets — the seasoning,
    tastemaker or flavour blend. Keeping only the outer name lost them
    entirely: a Maggi noodles panel reduced to "Masala Tastemaker" with
    "hydrolysed groundnut protein" nowhere in the parsed output, so a declared
    peanut allergy could not fire no matter how good the matching was.

    The parent is kept as well as the parts. It is a real ingredient in its own
    right, and dropping it would lose the entry the label actually lists.
    """
    split = _split_compound(raw_part)
    if split is None:
        return [raw_part]

    parent, inner = split
    parts = [p.strip() for p in _split_top_level(inner) if p.strip()]

    # A single part is not a list; leave it to the qualifier path so
    # "Tocopherol (Vitamin E)" and "Thickeners (508 & 412)" keep their shape.
    if len(parts) < 2:
        return [raw_part]

    # Sub-lists nest one more level ("Thickeners (508 & 412)" inside the
    # tastemaker). Expanding recursively keeps every declared substance
    # reachable however deeply the manufacturer buried it.
    expanded: list[str] = [parent] if parent else []
    for part in parts:
        nested = _expand_compound(part)
        expanded.extend(nested if nested != [part] else [part])
    return expanded


def clean_token(raw: str) -> Optional[tuple[str, list[str]]]:
    """Normalise one token. Returns None when it is not an ingredient."""
    if not raw:
        return None

    text = _ASTERISK.sub(" ", raw)
    text = _LEADING_NUMBER.sub("", text)
    text, qualifiers = _extract_qualifiers(text)
    text = _PERCENTAGE.sub(" ", text)
    text = _QUANTITY.sub(" ", text)
    text = text.replace(":", " ").strip(" .-–—_[]{}")
    text = _WHITESPACE.sub(" ", text).strip()

    if not text or len(text) < 2:
        return None
    if _NOISE.match(text):
        return None
    # A token that is all digits is a colour index number without its "CI"
    # prefix, or OCR noise. Neither resolves on its own.
    if text.replace(" ", "").isdigit():
        return None
    # Panels do not have 12-word ingredients; this is a sentence that leaked in.
    if len(text.split()) > 12:
        return None

    # Qualifiers were dropped above four words, which silently deleted
    # "Masala Tastemaker (contains hydrolysed groundnut protein)" — the
    # parenthetical on a compound ingredient is where allergens are declared,
    # so length is the worst possible reason to discard it. The cap now only
    # excludes prose that has leaked in from elsewhere on the pack.
    kept = [
        q for q in qualifiers
        if _KNOWN_QUALIFIERS.match(q.strip()) or len(q.split()) <= 12
    ]
    return text, kept


def parse_panel(panel_text: str, *, had_header: bool = False, max_tokens: int = MAX_TOKENS) -> ParsedPanel:
    """Segment a panel into ordered, deduplicated tokens."""
    result = ParsedPanel(had_header=had_header)

    if not panel_text or not panel_text.strip():
        return result

    # Everything after a trace marker is declared separately, and mixing the
    # two would put "may contain peanuts" in the ingredient list as fact.
    trace_split = _TRACE_MARKER.split(panel_text, maxsplit=1)
    main_text = trace_split[0]
    trace_text = trace_split[1] if len(trace_split) > 1 else ""

    seen: set[str] = set()
    position = 0

    for source_text, is_trace in ((main_text, False), (trace_text, True)):
        for outer_part in _split_top_level(source_text):
            if not outer_part.strip():
                continue

            # A compound ingredient carries its own ingredient list in
            # brackets. Both the compound and its parts are real ingredients,
            # so both become tokens — the parts are where allergens are
            # actually declared.
            for raw_part in _expand_compound(outer_part):
                cleaned = clean_token(raw_part)
                if cleaned is None:
                    if raw_part.strip():
                        result.dropped.append(raw_part.strip()[:80])
                    continue

                text_value, qualifiers = cleaned
                key = text_value.lower()
                if key in seen:
                    continue
                seen.add(key)

                if len(result.tokens) >= max_tokens:
                    result.truncated = True
                    return result

                result.tokens.append(
                    ParsedToken(
                        raw=raw_part.strip(),
                        text=text_value,
                        position=position,
                        qualifiers=qualifiers,
                        is_trace=is_trace,
                    )
                )
                position += 1

    return result
