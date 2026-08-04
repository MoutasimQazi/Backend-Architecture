"""Evidence: source tiering, independence scoring, ranking, synthesis."""

from packages.evidence.independence import (
    FunderClass,
    IndependenceScore,
    classify_funder,
    score_independence,
)
from packages.evidence.tiers import DESIGN_WEIGHT, tier_for_domain, tier_weight

__all__ = [
    "DESIGN_WEIGHT",
    "FunderClass",
    "IndependenceScore",
    "classify_funder",
    "score_independence",
    "tier_for_domain",
    "tier_weight",
]
