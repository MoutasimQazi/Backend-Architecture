"""S4 — the small-LLM router classifier (arch.md 5.3).

Only sees the ambiguous band: questions S0-S3 could not settle. That is the
whole point of the cascade — this is the first stage that costs money, and by
the time a message reaches it the free stages have already handled the
greetings, the exact FAQ hits and the cache repeats.

Runs on the small model, with no user data in the prompt: classifying "how is
my recovery trending" needs the shape of the question, not the numbers.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from packages.chains.base import Chain
from packages.chains.providers import DataPolicy, ModelClass
from packages.domain.enums import RouteLabel

logger = logging.getLogger(__name__)


class Classification(BaseModel):
    label: Literal["SMALLTALK", "FAQ", "PERSONAL", "RESEARCH", "PRODUCT", "RESTAURANT"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


SYSTEM = """You classify messages sent to a health and fitness assistant.

Pick exactly one label:

SMALLTALK  — greetings, thanks, acknowledgements, chit-chat with no question.
FAQ        — a general question about fitness, nutrition or health that has a
             standard answer and does not depend on this user's own data.
PERSONAL   — needs this user's own logged data to answer: their metrics,
             trends, weight, sleep, workouts, lab results, or advice explicitly
             about their situation.
RESEARCH   — asks what the evidence or studies say about a substance,
             supplement, ingredient or health claim.
PRODUCT    — about a specific packaged product, its ingredients, or whether a
             named product is safe to use.
RESTAURANT — about a named restaurant, cafe or food outlet: hygiene, safety,
             inspections or reputation.

Rules:
- First-person references to own data ("my", "I", "am I") mean PERSONAL.
- "Is X safe" about a substance is RESEARCH; about a named branded product it
  is PRODUCT.
- If two labels fit, choose the one that needs more specific data.
- Set confidence below 0.6 when genuinely uncertain. A low confidence is
  useful; a confident wrong answer is not."""


class RouterClassifier(Chain):
    name = "router_classify"
    model_class = ModelClass.SMALL
    data_policy = DataPolicy.PUBLIC
    temperature = 0.0
    max_tokens = 200
    system_prompt = SYSTEM

    def classify(self, message: str) -> Optional[tuple[RouteLabel, float, str]]:
        """Returns (label, confidence, rationale), or None if the call failed."""
        if not message or not message.strip():
            return None

        # A long message is a paste, not a question; the first part carries the
        # intent and the rest just costs tokens.
        excerpt = message.strip()[:600]

        result = self.run_structured(f"Message: {excerpt}", Classification)
        if not result.ok or result.value is None:
            logger.info("S4 classifier unavailable (%s); router falls back", result.error)
            return None

        try:
            label = RouteLabel(result.value.label)
        except ValueError:
            logger.warning("S4 returned unknown label %r", result.value.label)
            return None

        return label, result.value.confidence, result.value.rationale
