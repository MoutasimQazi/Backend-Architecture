"""Personal risk matching (arch.md 8.6).

Rule match against `UserContext`: declared allergies with cross-reactant
expansion, conditions, pregnancy status, age band, dietary restrictions.

This is the part where a miss is a real harm, so it is deliberately biased
toward flagging: a substring match on a declared allergy fires even when the
ingredient did not resolve to a chemical id. An unresolved token that contains
the word the user is allergic to still gets flagged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from packages.domain.enums import HazardLevel, Severity, Verdict
from packages.domain.models import (
    HazardFinding,
    PersonalFlag,
    ResolvedIngredient,
    UserContext,
)

logger = logging.getLogger(__name__)

# Condition -> substances worth flagging, with the reason the user sees.
CONDITION_RULES: dict[str, list[tuple[str, str, Severity]]] = {
    "hypertension": [
        ("sodium", "high sodium intake raises blood pressure", Severity.MODERATE),
        ("liquorice", "liquorice can raise blood pressure", Severity.HIGH),
        ("licorice", "licorice can raise blood pressure", Severity.HIGH),
    ],
    "diabetes": [
        ("glucose syrup", "a rapidly absorbed sugar", Severity.MODERATE),
        ("high fructose corn syrup", "a rapidly absorbed sugar", Severity.MODERATE),
        ("maltodextrin", "raises blood glucose faster than table sugar", Severity.MODERATE),
        ("dextrose", "a rapidly absorbed sugar", Severity.MODERATE),
    ],
    "ckd": [
        ("potassium", "potassium load matters in reduced kidney function", Severity.HIGH),
        ("phosphate", "phosphate additives are a concern in kidney disease", Severity.HIGH),
    ],
    "celiac": [
        ("wheat", "contains gluten", Severity.CRITICAL),
        ("barley", "contains gluten", Severity.CRITICAL),
        ("rye", "contains gluten", Severity.CRITICAL),
        ("malt", "usually barley-derived, so contains gluten", Severity.HIGH),
    ],
    "g6pd": [
        ("menthol", "can trigger haemolysis in G6PD deficiency", Severity.HIGH),
    ],
}

PREGNANCY_AVOID: list[tuple[str, str, Severity]] = [
    ("retinol", "retinoids are advised against in pregnancy", Severity.HIGH),
    ("retinyl palmitate", "retinoids are advised against in pregnancy", Severity.HIGH),
    ("retinoic acid", "retinoids are contraindicated in pregnancy", Severity.CRITICAL),
    ("tretinoin", "retinoids are contraindicated in pregnancy", Severity.CRITICAL),
    ("salicylic acid", "high-dose salicylates are advised against in pregnancy", Severity.MODERATE),
    ("caffeine", "intake is usually limited in pregnancy", Severity.LOW),
    ("hydroquinone", "significant systemic absorption; avoided in pregnancy", Severity.HIGH),
]

DIET_RULES: dict[str, list[tuple[str, str]]] = {
    "vegan": [
        ("gelatin", "animal-derived"),
        ("carmine", "insect-derived"),
        ("cochineal", "insect-derived"),
        ("shellac", "insect-derived"),
        ("lanolin", "sheep-derived"),
        ("beeswax", "bee-derived"),
        ("honey", "bee-derived"),
        ("casein", "milk-derived"),
        ("whey", "milk-derived"),
        ("lactose", "milk-derived"),
    ],
    "vegetarian": [
        ("gelatin", "animal-derived"),
        ("carmine", "insect-derived"),
        ("cochineal", "insect-derived"),
    ],
    "halal": [
        ("gelatin", "unless certified halal, gelatin may be porcine"),
        ("alcohol denat", "contains denatured alcohol"),
        ("ethanol", "contains alcohol"),
    ],
}


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (value or "").lower()).strip()


def _mentions(haystack: str, needle: str) -> bool:
    """Word-boundary containment.

    Substring matching would fire 'soy' inside 'soybean oil' correctly but also
    'nut' inside 'coconut' — a false allergy alert that trains users to ignore
    the real ones.
    """
    if not needle:
        return False
    return re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", haystack) is not None


@dataclass
class PersonalAssessment:
    flags: list[PersonalFlag]
    verdict: Verdict
    verdict_reason: str


class PersonalRiskMatcher:
    def __init__(self, context: Optional[UserContext], cross_reactants: Optional[dict] = None) -> None:
        self.context = context
        self.cross_reactants = cross_reactants or {}

    def assess(
        self,
        ingredients: list[ResolvedIngredient],
        findings: list[HazardFinding],
    ) -> PersonalAssessment:
        flags: list[PersonalFlag] = []

        if self.context is not None:
            flags.extend(self._allergy_flags(ingredients))
            flags.extend(self._condition_flags(ingredients))
            flags.extend(self._pregnancy_flags(ingredients))
            flags.extend(self._diet_flags(ingredients))

        verdict, reason = self._verdict(ingredients, findings, flags)
        return PersonalAssessment(flags=flags, verdict=verdict, verdict_reason=reason)

    # -- individual matchers ----------------------------------------------

    def _allergy_flags(self, ingredients: list[ResolvedIngredient]) -> list[PersonalFlag]:
        allergies = [a for a in (self.context.medical.allergies or []) if a]
        if not allergies:
            return []

        flags: list[PersonalFlag] = []
        for ingredient in ingredients:
            haystack = _normalise(
                " ".join([ingredient.raw_token, ingredient.display_name or "", *ingredient.qualifiers])
            )

            for allergy in allergies:
                needle = _normalise(allergy)
                if not needle:
                    continue

                if _mentions(haystack, needle):
                    flags.append(
                        PersonalFlag(
                            chemical_id=ingredient.chemical_id or "",
                            display_name=ingredient.display_name or ingredient.raw_token,
                            reason=f"you have declared an allergy to {allergy}",
                            severity=Severity.CRITICAL,
                            source_of_rule="declared_allergy",
                        )
                    )
                    continue

                # Cross-reactant expansion: latex/banana, birch/apple and the
                # fragrance-allergen families are the cases users cannot be
                # expected to know themselves.
                for related in self.cross_reactants.get(needle, []):
                    if ingredient.chemical_id and related.get("chemical_id") == ingredient.chemical_id:
                        flags.append(
                            PersonalFlag(
                                chemical_id=ingredient.chemical_id,
                                display_name=ingredient.display_name or ingredient.raw_token,
                                reason=f"can cross-react with your declared {allergy} allergy",
                                severity=Severity(related.get("severity", "moderate")),
                                source_of_rule="cross_reactant",
                            )
                        )
        return flags

    def _condition_flags(self, ingredients: list[ResolvedIngredient]) -> list[PersonalFlag]:
        conditions = [_normalise(c) for c in (self.context.medical.conditions or [])]
        if not conditions:
            return []

        flags: list[PersonalFlag] = []
        for condition_key, rules in CONDITION_RULES.items():
            if not any(condition_key in c for c in conditions):
                continue
            for ingredient in ingredients:
                haystack = _normalise(f"{ingredient.raw_token} {ingredient.display_name or ''}")
                for needle, reason, severity in rules:
                    if _mentions(haystack, needle):
                        flags.append(
                            PersonalFlag(
                                chemical_id=ingredient.chemical_id or "",
                                display_name=ingredient.display_name or ingredient.raw_token,
                                reason=f"{reason} — relevant to your {condition_key}",
                                severity=severity,
                                source_of_rule=f"condition:{condition_key}",
                            )
                        )
        return flags

    def _pregnancy_flags(self, ingredients: list[ResolvedIngredient]) -> list[PersonalFlag]:
        status = _normalise(self.context.profile.pregnancy_status or "")
        if status not in ("pregnant", "trying", "breastfeeding", "nursing"):
            return []

        flags: list[PersonalFlag] = []
        for ingredient in ingredients:
            haystack = _normalise(f"{ingredient.raw_token} {ingredient.display_name or ''}")
            for needle, reason, severity in PREGNANCY_AVOID:
                if _mentions(haystack, needle):
                    flags.append(
                        PersonalFlag(
                            chemical_id=ingredient.chemical_id or "",
                            display_name=ingredient.display_name or ingredient.raw_token,
                            reason=reason,
                            severity=severity,
                            source_of_rule="pregnancy",
                        )
                    )
        return flags

    def _diet_flags(self, ingredients: list[ResolvedIngredient]) -> list[PersonalFlag]:
        preferences = [_normalise(p) for p in (self.context.profile.preferences or [])]
        if not preferences:
            return []

        flags: list[PersonalFlag] = []
        for diet, rules in DIET_RULES.items():
            if not any(diet in p for p in preferences):
                continue
            for ingredient in ingredients:
                haystack = _normalise(f"{ingredient.raw_token} {ingredient.display_name or ''}")
                for needle, reason in rules:
                    if _mentions(haystack, needle):
                        flags.append(
                            PersonalFlag(
                                chemical_id=ingredient.chemical_id or "",
                                display_name=ingredient.display_name or ingredient.raw_token,
                                reason=f"{reason} — you have set a {diet} preference",
                                severity=Severity.MODERATE,
                                source_of_rule=f"diet:{diet}",
                            )
                        )
        return flags

    # -- verdict ----------------------------------------------------------

    def _verdict(
        self,
        ingredients: list[ResolvedIngredient],
        findings: list[HazardFinding],
        flags: list[PersonalFlag],
    ) -> tuple[Verdict, str]:
        """The rules produce the verdict. arch.md 8.6: the LLM only selects the
        phrase the rules already chose, and guard_out checks it matched."""
        if not ingredients:
            return Verdict.INSUFFICIENT_DATA, "no ingredients could be read"

        resolved = [i for i in ingredients if i.resolved]
        coverage = len(resolved) / len(ingredients)

        if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in flags):
            return (
                Verdict.NOT_RECOMMENDED,
                "an ingredient conflicts with your declared allergies or health profile",
            )

        if any(f.banned_in for f in findings):
            return Verdict.NOT_RECOMMENDED, "contains a substance prohibited by a regulator"

        if any(f.hazard_level is HazardLevel.HIGH for f in findings):
            return Verdict.NOT_RECOMMENDED, "contains a substance with a high hazard classification"

        # Coverage gates optimism, not pessimism: a clean result over a panel we
        # mostly could not read is not evidence of safety.
        if coverage < 0.5:
            return (
                Verdict.INSUFFICIENT_DATA,
                f"only {len(resolved)} of {len(ingredients)} ingredients could be identified",
            )

        if flags or any(f.hazard_level is HazardLevel.MODERATE for f in findings):
            return Verdict.USE_WITH_CAUTION, "some ingredients carry cautions worth knowing about"

        if any(f.hazard_level is HazardLevel.LOW for f in findings):
            return Verdict.USE_WITH_CAUTION, "contains mild irritants or declarable allergens"

        return Verdict.GENERALLY_SUITABLE, "nothing in the panel conflicts with your profile"
