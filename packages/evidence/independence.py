"""Independence scoring (arch.md 9.3).

This replaces the current rule — *drop any paper that declares a grant* — which
inverts the signal it was reaching for. Publicly funded research (NIH, EU
Horizon, ICMR, Wellcome) is exactly the independent evidence the requirement
wants; unfunded work skews toward lower-powered and non-peer-reviewed output.

The score is **shown, not used to filter**. Industry-funded studies are
labelled and ranked lower, never hidden — that is both more defensible and more
useful than silently discarding half the literature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FunderClass(str, Enum):
    PUBLIC = "public"
    CHARITABLE = "charitable"
    NONE = "none"
    INDUSTRY = "industry"
    TRADE_ASSOCIATION = "trade_association"
    UNKNOWN = "unknown"


# Weighted contributions to the score. Public funding is the strongest positive
# signal available; a manufacturer-employed author is the strongest negative.
FUNDER_WEIGHT: dict[FunderClass, float] = {
    FunderClass.PUBLIC: 1.00,
    FunderClass.CHARITABLE: 0.90,
    FunderClass.NONE: 0.65,
    FunderClass.UNKNOWN: 0.50,
    FunderClass.TRADE_ASSOCIATION: 0.25,
    FunderClass.INDUSTRY: 0.15,
}

_PUBLIC_PATTERNS = [
    r"\bnih\b", r"national institutes? of health", r"\bnci\b", r"\bniehs\b",
    r"\bnsf\b", r"\bcdc\b", r"\bfda\b", r"\bepa\b",
    r"european commission", r"horizon 2020", r"horizon europe", r"\berc\b",
    r"medical research council", r"\bmrc\b", r"\bnihr\b", r"\bukri\b",
    r"\bicmr\b", r"\bdbt\b", r"\bdst\b", r"council of scientific",
    r"\bcihr\b", r"\bnhmrc\b", r"\bdfg\b", r"\binserm\b", r"\bcnrs\b",
    r"ministry of", r"department of health", r"national research foundation",
    r"\bwho\b", r"world health organi[sz]ation",
]

_CHARITABLE_PATTERNS = [
    r"wellcome", r"gates foundation", r"cancer research uk", r"british heart foundation",
    r"american heart association", r"american cancer society", r"\bcrf\b",
    r"leverhulme", r"howard hughes", r"charitable trust", r"\bfoundation\b",
]

_TRADE_PATTERNS = [
    r"trade association", r"industry council", r"manufacturers association",
    r"\bilsi\b", r"international life sciences", r"cosmetics europe",
    r"personal care products council", r"\bcir\b", r"beverage association",
    r"sugar association", r"dairy council", r"\bcefic\b",
]

_INDUSTRY_PATTERNS = [
    r"\binc\b", r"\bltd\b", r"\bllc\b", r"\bgmbh\b", r"\bs\.?a\.?s?\b",
    r"pharmaceutical[s]? (?:inc|ltd|company|corp)", r"\bcorp\b", r"\bcorporation\b",
    r"unilever", r"\bl'?or[ée]al\b", r"procter", r"nestl[ée]", r"pepsi", r"coca[- ]cola",
    r"johnson\s*&\s*johnson", r"pfizer", r"novartis", r"bayer", r"basf", r"dow chemical",
    r"monsanto", r"syngenta", r"colgate", r"beiersdorf", r"shiseido", r"estee lauder",
]


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_funder(funders: list[str]) -> FunderClass:
    """Classify a funding statement.

    Order matters: a paper funded by both NIH and a manufacturer is classified
    industry, because the conflict is what the reader needs to weigh. Being
    generous about public funding while ignoring co-funding would launder
    exactly the studies this score exists to flag.
    """
    if not funders:
        return FunderClass.NONE

    joined = " ; ".join(f for f in funders if f)
    if not joined.strip():
        return FunderClass.NONE

    if _matches(joined, _INDUSTRY_PATTERNS):
        return FunderClass.INDUSTRY
    if _matches(joined, _TRADE_PATTERNS):
        return FunderClass.TRADE_ASSOCIATION
    if _matches(joined, _PUBLIC_PATTERNS):
        return FunderClass.PUBLIC
    if _matches(joined, _CHARITABLE_PATTERNS):
        return FunderClass.CHARITABLE

    return FunderClass.UNKNOWN


@dataclass
class IndependenceScore:
    value: float = 0.5
    funder_class: FunderClass = FunderClass.UNKNOWN
    declared_coi: Optional[bool] = None
    sponsor_role: Optional[str] = None
    registry_status: Optional[str] = None
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """What the user is shown, since the score is displayed not applied."""
        if self.value >= 0.8:
            return "independent"
        if self.value >= 0.6:
            return "likely independent"
        if self.value >= 0.4:
            return "unclear"
        return "industry-linked"


def score_independence(
    funders: Optional[list[str]] = None,
    *,
    declared_coi: Optional[bool] = None,
    author_affiliations: Optional[list[str]] = None,
    sponsor_role: Optional[str] = None,
    registry_status: Optional[str] = None,
) -> IndependenceScore:
    """arch.md 9.3:

        independence = f(funder_class, declared_COI, author_affiliation,
                         sponsor_role, registry_status)
    """
    funder_class = classify_funder(funders or [])
    score = FUNDER_WEIGHT[funder_class]
    reasons: list[str] = [f"funding: {funder_class.value}"]

    # A manufacturer-employed author is a conflict even when the funding
    # statement is clean.
    if author_affiliations and _matches(" ; ".join(author_affiliations), _INDUSTRY_PATTERNS):
        score -= 0.25
        reasons.append("author affiliated with a commercial manufacturer")

    if declared_coi is True:
        score -= 0.15
        reasons.append("authors declared a conflict of interest")
    elif declared_coi is False:
        score += 0.05
        reasons.append("no conflict declared")

    if sponsor_role:
        role = sponsor_role.lower()
        if any(w in role for w in ("design", "analysis", "writing", "decision")):
            score -= 0.20
            reasons.append("sponsor involved in design, analysis or writing")
        elif "none" in role or "no role" in role:
            score += 0.10
            reasons.append("sponsor had no role in the research")

    # Pre-registration is the single best guard against outcome switching.
    if registry_status:
        status = registry_status.lower()
        if "prospective" in status or "pre-registered" in status or "preregistered" in status:
            score += 0.15
            reasons.append("prospectively registered")
        elif "retrospective" in status:
            score -= 0.05
            reasons.append("retrospectively registered")

    return IndependenceScore(
        value=round(max(0.0, min(1.0, score)), 3),
        funder_class=funder_class,
        declared_coi=declared_coi,
        sponsor_role=sponsor_role,
        registry_status=registry_status,
        reasons=reasons,
    )
