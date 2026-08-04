"""LCEL-style chains: one model call each, with a declared policy and cost.

arch.md 13 — small models do small jobs. Routing, classification, extraction,
verification and memory run on the small model; only the personal agent and the
product explanation touch the large one.
"""

from packages.chains.base import Chain, ChainResult, Masker, extract_json
from packages.chains.classify import RouterClassifier
from packages.chains.embeddings import embed_query, embed_many
from packages.chains.explain import ProductExplainer, ProductExplanation
from packages.chains.memory import FactExtractor, MemorySummariser
from packages.chains.personal import AgentResult, PersonalAgent, render_context_summary
from packages.chains.providers import (
    Completion,
    DataPolicy,
    ModelClass,
    ProviderError,
    available_providers,
    complete,
    embed,
    is_configured,
)
from packages.chains.verify import ClaimVerifier, VerificationResult

__all__ = [
    "AgentResult",
    "Chain",
    "ChainResult",
    "ClaimVerifier",
    "Completion",
    "DataPolicy",
    "FactExtractor",
    "Masker",
    "MemorySummariser",
    "ModelClass",
    "PersonalAgent",
    "ProductExplainer",
    "ProductExplanation",
    "ProviderError",
    "RouterClassifier",
    "VerificationResult",
    "available_providers",
    "complete",
    "embed",
    "embed_many",
    "embed_query",
    "extract_json",
    "is_configured",
    "render_context_summary",
]
