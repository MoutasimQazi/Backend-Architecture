"""Hazard assessment: rules, not a model (arch.md 8.5).

The LLM never assigns a hazard level. A declarative rule matches assertions in
the Chemical KB and produces a `HazardFinding` carrying the rule ids that fired,
so any verdict can be reproduced and audited later.

Rule shape, stored as JSON in `hazard_rule`:

    condition: {
      "all": [                       # every clause must match ("any" also works)
        {"domain": "hazard", "key": "iarc_group", "in": ["1", "2A"]},
        {"domain": "regulatory", "key": "annex_ii", "exists": true}
      ]
    }
    effect: {
      "hazard_level": "high",
      "endocrine_flag": true,
      "caveat": "Concentration is not stated on the panel.",
      "reason": "IARC group 1 or 2A carcinogen"
    }

Comparison operators are a closed set. There is no expression evaluation here
on purpose: rules are authored through an admin UI, and an authoring UI that
can run arbitrary code in the request path is a vulnerability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from packages.domain.enums import HazardLevel
from packages.domain.models import EvidenceRef, HazardFinding, ResolvedIngredient

logger = logging.getLogger(__name__)

RULES_VERSION = "v1"

_LEVEL_ORDER = {
    HazardLevel.UNKNOWN: -1,
    HazardLevel.NONE: 0,
    HazardLevel.LOW: 1,
    HazardLevel.MODERATE: 2,
    HazardLevel.HIGH: 3,
}


def _max_level(a: HazardLevel, b: HazardLevel) -> HazardLevel:
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b


@dataclass
class _Clause:
    domain: Optional[str] = None
    key: Optional[str] = None
    equals: Optional[str] = None
    in_values: Optional[list[str]] = None
    exists: Optional[bool] = None
    contains: Optional[str] = None
    jurisdiction: Optional[str] = None

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "_Clause":
        return cls(
            domain=raw.get("domain"),
            key=raw.get("key"),
            equals=raw.get("equals"),
            in_values=[str(v) for v in raw["in"]] if isinstance(raw.get("in"), list) else None,
            exists=raw.get("exists"),
            contains=raw.get("contains"),
            jurisdiction=raw.get("jurisdiction"),
        )

    def matches(self, assertions: list[dict[str, Any]]) -> bool:
        candidates = assertions
        if self.domain:
            candidates = [a for a in candidates if a.get("domain") == self.domain]
        if self.key:
            candidates = [a for a in candidates if a.get("key_name") == self.key]
        if self.jurisdiction:
            candidates = [
                a
                for a in candidates
                if a.get("jurisdiction") in (None, "INTL", self.jurisdiction)
            ]

        if self.exists is not None:
            return bool(candidates) is self.exists

        if not candidates:
            return False

        if self.equals is not None:
            return any(str(a.get("value")) == self.equals for a in candidates)

        if self.in_values is not None:
            return any(str(a.get("value")) in self.in_values for a in candidates)

        if self.contains is not None:
            needle = self.contains.lower()
            return any(needle in str(a.get("value") or "").lower() for a in candidates)

        # A clause with only domain/key is an existence test.
        return True


def _evaluate(condition: dict[str, Any], assertions: list[dict[str, Any]]) -> bool:
    if not condition:
        return False

    if "all" in condition:
        clauses = condition["all"]
        return bool(clauses) and all(
            _Clause.parse(c).matches(assertions) for c in clauses if isinstance(c, dict)
        )

    if "any" in condition:
        clauses = condition["any"]
        return any(
            _Clause.parse(c).matches(assertions) for c in clauses if isinstance(c, dict)
        )

    if "none" in condition:
        clauses = condition["none"]
        return not any(
            _Clause.parse(c).matches(assertions) for c in clauses if isinstance(c, dict)
        )

    return _Clause.parse(condition).matches(assertions)


class HazardRulesEngine:
    def __init__(
        self,
        rules: list[dict[str, Any]],
        *,
        jurisdiction: str = "IN",
        product_class: Optional[str] = None,
    ) -> None:
        self.rules = rules
        self.jurisdiction = jurisdiction
        self.product_class = product_class

    def assess(
        self,
        ingredient: ResolvedIngredient,
        dossier: dict[str, Any],
        assertions: list[dict[str, Any]],
        evidence: Optional[list[EvidenceRef]] = None,
        *,
        total_ingredients: int = 0,
    ) -> HazardFinding:
        """Assess one resolved ingredient. Deterministic given the same inputs."""
        display = (
            ingredient.display_name
            or dossier.get("display_name")
            or dossier.get("inci_name")
            or ingredient.raw_token
        )

        finding = HazardFinding(
            chemical_id=ingredient.chemical_id or "",
            display_name=display,
            hazard_level=HazardLevel.UNKNOWN if not assertions else HazardLevel.NONE,
            evidence=evidence or [],
            rules_version=RULES_VERSION,
        )

        # Facts read straight off the assertions — no rule needed for these.
        for assertion in assertions:
            domain, key, value = (
                assertion.get("domain"),
                assertion.get("key_name"),
                assertion.get("value"),
            )

            if domain == "hazard" and key == "ghs_code" and value:
                finding.ghs_codes.append(str(value))
            elif domain == "hazard" and key == "iarc_group" and value:
                finding.iarc_group = str(value)
            elif domain == "endocrine" and value:
                finding.endocrine_flag = True
                finding.endocrine_lists.append(str(value))
            elif domain == "allergen":
                finding.allergen_flag = True
            elif domain == "regulatory" and key in ("banned", "annex_ii"):
                jurisdiction = assertion.get("jurisdiction") or "INTL"
                if jurisdiction not in finding.banned_in:
                    finding.banned_in.append(jurisdiction)
            elif domain == "regulatory" and key in ("restricted", "annex_iii", "limit"):
                jurisdiction = assertion.get("jurisdiction") or "INTL"
                if jurisdiction not in finding.restricted_in:
                    finding.restricted_in.append(jurisdiction)

        finding.ghs_codes = sorted(set(finding.ghs_codes))
        finding.endocrine_lists = sorted(set(finding.endocrine_lists))

        # Rules set the level. Highest wins; every rule that fires is recorded.
        for rule in self.rules:
            try:
                if not _evaluate(rule.get("condition") or {}, assertions):
                    continue
            except Exception:
                logger.exception("rule %s failed to evaluate; skipping", rule.get("rule_id"))
                continue

            effect = rule.get("effect") or {}
            finding.rule_ids.append(f"{rule.get('rule_id')}@{rule.get('version', 1)}")

            level = effect.get("hazard_level")
            if level:
                try:
                    finding.hazard_level = _max_level(finding.hazard_level, HazardLevel(level))
                except ValueError:
                    logger.warning("rule %s has invalid hazard_level %r", rule.get("rule_id"), level)

            if effect.get("endocrine_flag"):
                finding.endocrine_flag = True
            if effect.get("allergen_flag"):
                finding.allergen_flag = True
            if effect.get("caveat") and not finding.concentration_caveat:
                finding.concentration_caveat = str(effect["caveat"])

        # arch.md 8.5: a position-based concentration caveat. INCI order is
        # descending by weight above 1%, so a hazard flagged in the first third
        # of a long panel is present at a meaningfully higher concentration.
        if (
            total_ingredients >= 6
            and ingredient.position < max(3, total_ingredients // 3)
            and _LEVEL_ORDER[finding.hazard_level] >= _LEVEL_ORDER[HazardLevel.MODERATE]
            and not finding.concentration_caveat
        ):
            finding.concentration_caveat = (
                f"Declared {_ordinal(ingredient.position + 1)} of {total_ingredients}, "
                "so it is among the higher-concentration components."
            )

        return finding


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Starter rule set. Seeded into `hazard_rule` so it is versioned and editable
# rather than compiled in; this is the initial content, not the mechanism.
DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "iarc_known_carcinogen",
        "priority": 10,
        "description": "IARC group 1 or 2A — known or probable human carcinogen",
        "condition": {"domain": "hazard", "key": "iarc_group", "in": ["1", "2A"]},
        "effect": {
            "hazard_level": "high",
            "reason": "classified by IARC as a known or probable human carcinogen",
        },
    },
    {
        "rule_id": "iarc_possible_carcinogen",
        "priority": 20,
        "description": "IARC group 2B — possible human carcinogen",
        "condition": {"domain": "hazard", "key": "iarc_group", "in": ["2B"]},
        "effect": {
            "hazard_level": "moderate",
            "reason": "classified by IARC as a possible human carcinogen",
        },
    },
    {
        "rule_id": "banned_in_jurisdiction",
        "priority": 5,
        "description": "Prohibited for this product class in at least one jurisdiction",
        "condition": {"any": [
            {"domain": "regulatory", "key": "banned", "exists": True},
            {"domain": "regulatory", "key": "annex_ii", "exists": True},
        ]},
        "effect": {"hazard_level": "high", "reason": "prohibited by a regulator"},
    },
    {
        "rule_id": "restricted_with_limit",
        "priority": 30,
        "description": "Permitted only below a concentration limit",
        "condition": {"any": [
            {"domain": "regulatory", "key": "restricted", "exists": True},
            {"domain": "regulatory", "key": "annex_iii", "exists": True},
        ]},
        "effect": {
            "hazard_level": "moderate",
            "caveat": "Permitted only below a concentration limit, which the label does not state.",
            "reason": "use is restricted to a maximum concentration",
        },
    },
    {
        "rule_id": "endocrine_disruptor_listed",
        "priority": 25,
        "description": "On an endocrine-disruptor list (EU EDC / TEDX)",
        "condition": {"domain": "endocrine", "key": "list_membership", "exists": True},
        "effect": {
            "hazard_level": "moderate",
            "endocrine_flag": True,
            "reason": "appears on a recognised endocrine-disruptor list",
        },
    },
    {
        "rule_id": "svhc_listed",
        "priority": 15,
        "description": "ECHA Substance of Very High Concern",
        "condition": {"domain": "hazard", "key": "svhc", "exists": True},
        "effect": {
            "hazard_level": "high",
            "reason": "listed by ECHA as a Substance of Very High Concern",
        },
    },
    {
        "rule_id": "eu26_fragrance_allergen",
        "priority": 40,
        "description": "One of the EU 26 declarable fragrance allergens",
        "condition": {"domain": "allergen", "key": "eu26", "exists": True},
        "effect": {
            "hazard_level": "low",
            "allergen_flag": True,
            "reason": "a declarable fragrance allergen — relevant to sensitised skin only",
        },
    },
    {
        "rule_id": "ghs_severe_health_hazard",
        "priority": 18,
        "description": "GHS carcinogenicity, mutagenicity or reproductive toxicity",
        "condition": {
            "domain": "hazard",
            "key": "ghs_code",
            "in": ["H340", "H341", "H350", "H351", "H360", "H361", "H362"],
        },
        "effect": {"hazard_level": "high", "reason": "carries a GHS CMR hazard statement"},
    },
    {
        "rule_id": "ghs_irritant",
        "priority": 60,
        "description": "GHS skin/eye irritation",
        "condition": {
            "domain": "hazard",
            "key": "ghs_code",
            "in": ["H315", "H317", "H318", "H319", "H320"],
        },
        "effect": {"hazard_level": "low", "reason": "may irritate skin or eyes"},
    },
]
