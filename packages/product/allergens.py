"""Allergen term expansion.

A user declares "peanuts". The label says "hydrolysed groundnut protein". Those
are the same legume, and a word-boundary match on "peanut" finds nothing in
"groundnut" — so the scan came back clean for someone with a peanut allergy.
That was found on a real product: Maggi Masala Noodles, whose tastemaker
contains hydrolysed groundnut protein.

This matters most in exactly the market this app targets. Indian labels use
"groundnut" for peanut, "kaju" for cashew, "badam" for almond, "til" for
sesame, "sarson" for mustard, "maida"/"atta"/"suji" for wheat. None of those
strings contain the English allergen name.

Two properties this module must have:

  1. EXPANSION IS ADDITIVE. The declared term is always matched as well; the
     synonyms only add ways to fire. Nothing here can make an allergen stop
     matching.
  2. EXCLUSIONS PREVENT FALSE ALARMS, NOT REAL ONES. "Butter" means dairy on
     most labels, but "shea butter", "cocoa butter" and "peanut butter" are not
     dairy at all. Firing a milk alert on cocoa butter teaches people to ignore
     the alerts, which costs more safety than it buys. Exclusions are therefore
     narrow and each one is justified.

The cross-reactant table in the database stays as it is: that is for
biologically *related* allergens (latex/banana, birch/apple) and is keyed on
chemical_id. This module handles the different problem of the same substance
under a different name, and works on raw label text, so it fires on
ingredients the Chemical KB has never heard of.
"""

from __future__ import annotations

import re
from typing import Iterable

# family -> (declared-name triggers, label synonyms, exclusion contexts)
#
# "triggers" are what a user might type in their profile. "synonyms" are what
# a label might print. Both are matched on word boundaries after normalisation.
_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "peanut": {
        "triggers": ("peanut", "peanuts", "groundnut", "groundnuts", "arachis", "moongphali"),
        "synonyms": (
            "peanut", "peanuts", "groundnut", "groundnuts", "ground nut",
            "arachis", "arachis hypogaea", "arachis oil", "monkey nut",
            "earthnut", "earth nut", "mandelona", "beer nut", "moongphali",
            "mungfali", "hydrolysed groundnut protein", "hydrolyzed groundnut protein",
        ),
        "excludes": (),
    },
    "tree_nut": {
        "triggers": ("tree nut", "tree nuts", "nuts", "nut allergy"),
        "synonyms": (
            "almond", "almonds", "badam", "cashew", "cashews", "kaju",
            "walnut", "walnuts", "akhrot", "pecan", "pecans",
            "pistachio", "pistachios", "pista", "hazelnut", "hazelnuts",
            "filbert", "macadamia", "brazil nut", "brazil nuts",
            "pine nut", "pine nuts", "chilgoza", "praline", "marzipan",
            "nut butter", "mixed nuts",
        ),
        "excludes": (),
    },
    "milk": {
        "triggers": ("milk", "dairy", "lactose", "casein"),
        "synonyms": (
            "milk", "dairy", "casein", "caseinate", "sodium caseinate",
            "calcium caseinate", "whey", "whey protein", "lactose",
            "lactalbumin", "lactoglobulin", "milk solids", "milk powder",
            "skimmed milk", "skim milk", "buttermilk", "butterfat",
            "butter oil", "ghee", "cream", "curd", "paneer", "khoya",
            "cheese", "yoghurt", "yogurt", "dahi", "malai",
        ),
        # Plant fats named "butter", and the cocoa/shea/nut butters. None of
        # these contain dairy; flagging them is a pure false positive.
        "excludes": ("shea", "cocoa", "cacao", "peanut", "almond", "cashew", "nut", "seed"),
    },
    "egg": {
        "triggers": ("egg", "eggs"),
        "synonyms": (
            "egg", "eggs", "albumin", "albumen", "ovalbumin", "ovomucoid",
            "ovoglobulin", "lysozyme", "livetin", "vitellin", "globulin",
            "meringue", "mayonnaise", "egg white", "egg yolk", "anda",
        ),
        "excludes": ("plant", "vegan", "eggplant"),
    },
    "soy": {
        "triggers": ("soy", "soya", "soybean"),
        "synonyms": (
            "soy", "soya", "soybean", "soya bean", "soy protein",
            "soy lecithin", "soya lecithin", "edamame", "tofu", "tempeh",
            "miso", "natto", "tamari", "textured vegetable protein",
            "hydrolysed vegetable protein", "hydrolyzed vegetable protein",
        ),
        "excludes": (),
    },
    "wheat": {
        "triggers": ("wheat", "gluten", "coeliac", "celiac"),
        "synonyms": (
            "wheat", "gluten", "wheat flour", "wheat gluten", "atta", "maida",
            "suji", "sooji", "semolina", "rava", "durum", "spelt", "farina",
            "seitan", "bulgur", "couscous", "farro", "kamut", "triticale",
            "barley", "rye", "malt", "malt extract", "brewer's yeast",
        ),
        "excludes": ("gluten free", "gluten-free", "buckwheat"),
    },
    "sesame": {
        "triggers": ("sesame", "til"),
        "synonyms": (
            "sesame", "sesamum", "sesame seed", "sesame oil", "til",
            "gingelly", "gingelly oil", "benne", "tahini", "tahina", "halva",
        ),
        "excludes": (),
    },
    "fish": {
        "triggers": ("fish",),
        "synonyms": (
            "fish", "anchovy", "anchovies", "cod", "salmon", "tuna",
            "sardine", "sardines", "mackerel", "haddock", "pollock",
            "fish sauce", "fish oil", "worcestershire", "surimi",
            "bombay duck", "machli",
        ),
        "excludes": ("shellfish",),
    },
    "shellfish": {
        "triggers": ("shellfish", "crustacean", "prawn", "shrimp", "crab", "lobster"),
        "synonyms": (
            "shellfish", "crustacean", "prawn", "prawns", "shrimp", "shrimps",
            "jhinga", "crab", "lobster", "krill", "crayfish", "langoustine",
            "mollusc", "mollusk", "oyster", "mussel", "clam", "scallop",
            "squid", "calamari", "octopus", "snail", "abalone",
        ),
        "excludes": (),
    },
    "mustard": {
        "triggers": ("mustard", "sarson"),
        "synonyms": ("mustard", "mustard seed", "sarson", "rai", "kasundi", "brassica"),
        "excludes": (),
    },
    "celery": {
        "triggers": ("celery", "celeriac"),
        "synonyms": ("celery", "celeriac", "celery seed", "celery salt"),
        "excludes": (),
    },
    "lupin": {
        "triggers": ("lupin", "lupine"),
        "synonyms": ("lupin", "lupine", "lupin flour"),
        "excludes": (),
    },
    "sulphite": {
        "triggers": ("sulphite", "sulfite", "sulphur dioxide", "sulfur dioxide"),
        "synonyms": (
            "sulphite", "sulfite", "sulphur dioxide", "sulfur dioxide",
            "sodium metabisulphite", "sodium metabisulfite",
            "potassium metabisulphite", "potassium metabisulfite",
            "sodium bisulphite", "sodium bisulfite",
            "e220", "e221", "e222", "e223", "e224", "e225", "e226", "e227", "e228",
        ),
        "excludes": (),
    },
}


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (value or "").lower()).strip()


def _mentions(haystack: str, needle: str) -> bool:
    """Word-boundary containment, so 'nut' does not fire inside 'coconut'."""
    if not needle:
        return False
    return re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", haystack) is not None


def _singular(value: str) -> str:
    """Crude de-pluralisation for trigger matching only.

    Users write "sulphites" and "nuts"; the table lists the singular. Word
    boundaries mean "sulphites" does not contain "sulphite", so without this
    a declared allergy silently expanded to nothing. Applied only to the
    user's own wording, never to label text, so an over-eager strip can add a
    match but never remove one.
    """
    if len(value) > 3 and value.endswith("es") and value[-3] in "shxz":
        return value[:-2]
    if len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def families_for(declared: str) -> list[str]:
    """Which allergen families a declared allergy belongs to.

    A declaration can name a family directly ("tree nuts") or one member of it
    ("cashew"), and both must expand to the whole family — someone allergic to
    cashew who writes "cashew" still needs "kaju" matched.
    """
    needle = _normalise(declared)
    if not needle:
        return []

    # Match on the wording given and on its singular form.
    forms = {needle, " ".join(_singular(w) for w in needle.split())}

    hits: list[str] = []
    for family, spec in _FAMILIES.items():
        if any(_mentions(form, t) for form in forms for t in spec["triggers"]):
            hits.append(family)
            continue
        # Declaring a single member expands to its family's other names.
        if any(_mentions(form, s) for form in forms for s in spec["synonyms"]):
            hits.append(family)
    return hits


def expand(declared: str) -> list[str]:
    """All label terms that should fire for this declared allergy.

    The declared term itself is always included, so an allergy this table does
    not know still behaves exactly as it did before.
    """
    terms = {_normalise(declared)} - {""}
    for family in families_for(declared):
        terms.update(_normalise(s) for s in _FAMILIES[family]["synonyms"])
    return sorted(t for t in terms if t)


def _excluded(haystack: str, family: str) -> bool:
    return any(_mentions(haystack, e) for e in _FAMILIES[family]["excludes"])


def match(declared: str, ingredient_text: str) -> tuple[bool, str]:
    """Does this ingredient trigger this declared allergy?

    Returns (matched, matched_term). The term is reported so the answer can say
    *why* — "groundnut is peanut" is information the user needs, not an
    implementation detail.
    """
    haystack = _normalise(ingredient_text)
    if not haystack:
        return False, ""

    declared_norm = _normalise(declared)
    if declared_norm and _mentions(haystack, declared_norm):
        return True, declared_norm

    for family in families_for(declared):
        if _excluded(haystack, family):
            continue
        for synonym in _FAMILIES[family]["synonyms"]:
            term = _normalise(synonym)
            if term and _mentions(haystack, term):
                return True, term

    return False, ""


def known_families() -> Iterable[str]:
    return _FAMILIES.keys()
