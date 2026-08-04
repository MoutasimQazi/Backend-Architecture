"""Source tiering and ranking (arch.md 9.2).

The allowlist is a table with an owner and a review date — not a hardcoded set.
What lives here is the *default* content for that table plus the ranking maths:

    rank = tier x independence x study_design x recency
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from packages.domain.enums import SourceTier

TIER_WEIGHT: dict[SourceTier, float] = {
    SourceTier.T1_GOVERNMENT: 1.00,
    SourceTier.T2_SYSTEMATIC: 0.90,
    SourceTier.T3_PRIMARY: 0.65,
    SourceTier.T4_SECONDARY: 0.35,
    SourceTier.BLOCKED: 0.0,
}

DESIGN_WEIGHT: dict[str, float] = {
    "meta-analysis": 1.00,
    "systematic-review": 0.95,
    "rct": 0.85,
    "clinical-trial": 0.75,
    "cohort": 0.65,
    "case-control": 0.55,
    "cross-sectional": 0.45,
    "review": 0.40,
    "animal": 0.35,
    "in-vitro": 0.30,
    "case-report": 0.25,
}

# Seeds `source_allowlist`. Owned and review-dated in the table, not here.
DEFAULT_ALLOWLIST: list[tuple[str, SourceTier]] = [
    # T1 — government / intergovernmental
    ("who.int", SourceTier.T1_GOVERNMENT),
    ("nih.gov", SourceTier.T1_GOVERNMENT),
    ("nlm.nih.gov", SourceTier.T1_GOVERNMENT),
    ("ncbi.nlm.nih.gov", SourceTier.T1_GOVERNMENT),
    ("cdc.gov", SourceTier.T1_GOVERNMENT),
    ("fda.gov", SourceTier.T1_GOVERNMENT),
    ("epa.gov", SourceTier.T1_GOVERNMENT),
    ("efsa.europa.eu", SourceTier.T1_GOVERNMENT),
    ("echa.europa.eu", SourceTier.T1_GOVERNMENT),
    ("ec.europa.eu", SourceTier.T1_GOVERNMENT),
    ("nhs.uk", SourceTier.T1_GOVERNMENT),
    ("canada.ca", SourceTier.T1_GOVERNMENT),
    ("tga.gov.au", SourceTier.T1_GOVERNMENT),
    ("icmr.gov.in", SourceTier.T1_GOVERNMENT),
    ("fssai.gov.in", SourceTier.T1_GOVERNMENT),
    ("foscos.fssai.gov.in", SourceTier.T1_GOVERNMENT),
    ("iarc.who.int", SourceTier.T1_GOVERNMENT),
    ("monographs.iarc.who.int", SourceTier.T1_GOVERNMENT),
    ("oehha.ca.gov", SourceTier.T1_GOVERNMENT),
    # T2 — systematic evidence
    ("cochranelibrary.com", SourceTier.T2_SYSTEMATIC),
    ("cochrane.org", SourceTier.T2_SYSTEMATIC),
    ("crd.york.ac.uk", SourceTier.T2_SYSTEMATIC),
    ("nice.org.uk", SourceTier.T2_SYSTEMATIC),
    # T3 — peer-reviewed primary
    ("pubmed.ncbi.nlm.nih.gov", SourceTier.T3_PRIMARY),
    ("europepmc.org", SourceTier.T3_PRIMARY),
    ("ebi.ac.uk", SourceTier.T3_PRIMARY),
    ("openalex.org", SourceTier.T3_PRIMARY),
    ("crossref.org", SourceTier.T3_PRIMARY),
    ("semanticscholar.org", SourceTier.T3_PRIMARY),
    ("doi.org", SourceTier.T3_PRIMARY),
    ("pubchem.ncbi.nlm.nih.gov", SourceTier.T3_PRIMARY),
]

# arch.md 9.2 "blocked": content farms, supplement retailers, unattributed
# health blogs. Never surfaced, at any rank.
DEFAULT_BLOCKLIST: list[str] = [
    "healthline.com",
    "webmd.com",
    "draxe.com",
    "mercola.com",
    "naturalnews.com",
    "goop.com",
    "iherb.com",
    "bodybuilding.com",
]


def tier_for_domain(domain: str, allowlist: Optional[dict[str, str]] = None) -> SourceTier:
    """Resolve a domain to a tier. Unknown domains are T4, never blocked —
    blocking is an explicit decision recorded in the table."""
    if not domain:
        return SourceTier.T4_SECONDARY

    host = domain.lower().strip().lstrip(".")

    if allowlist:
        if host in allowlist:
            return SourceTier(allowlist[host])
        for known, tier in allowlist.items():
            if host.endswith("." + known):
                return SourceTier(tier)
        return SourceTier.T4_SECONDARY

    for blocked in DEFAULT_BLOCKLIST:
        if host == blocked or host.endswith("." + blocked):
            return SourceTier.BLOCKED

    for known, tier in DEFAULT_ALLOWLIST:
        if host == known or host.endswith("." + known):
            return tier

    return SourceTier.T4_SECONDARY


def tier_weight(tier: SourceTier) -> float:
    return TIER_WEIGHT.get(tier, 0.35)


def recency_weight(year: Optional[int], half_life_years: float = 8.0) -> float:
    """Decay by age, floored at 0.3.

    Floored deliberately: a 1970s IARC monograph is still the authority on its
    substance, and decaying it to nothing would rank a recent blog above it.
    """
    if not year:
        return 0.6
    current = datetime.now(timezone.utc).year
    age = max(0, current - year)
    return round(max(0.3, 0.5 ** (age / half_life_years)), 3)


def rank_score(
    tier: SourceTier,
    independence: float,
    study_design: Optional[str],
    year: Optional[int],
) -> float:
    """arch.md 9.3: rank by tier x independence x design x recency.

    Multiplicative, so a weakness anywhere pulls the whole score down — an
    industry-funded in-vitro study in a low-tier venue should not out-rank a
    public-funded meta-analysis on the strength of being recent.
    """
    if tier is SourceTier.BLOCKED:
        return 0.0
    design = DESIGN_WEIGHT.get((study_design or "").lower(), 0.5)
    return round(
        tier_weight(tier) * max(0.05, independence) * design * recency_weight(year), 5
    )
