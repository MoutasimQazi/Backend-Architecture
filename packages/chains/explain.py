"""The LLM explanation layer over scan findings (arch.md 8.6).

The model's *only* job here is prose. It is given the structured findings the
rules already produced and asked to explain them in plain language. It does not
choose the verdict, assign a hazard level, or introduce an ingredient — and
`guard_out` re-checks all three afterwards.

The verdict is passed in and echoed back rather than requested, so there is no
path by which a persuasive-sounding model output becomes a safety judgement.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from packages.chains.base import Chain
from packages.chains.providers import DataPolicy, ModelClass
from packages.domain.models import ProductAnalysis

logger = logging.getLogger(__name__)


class IngredientNote(BaseModel):
    name: str
    note: str = Field(max_length=400)


class ProductExplanation(BaseModel):
    overall: str = Field(max_length=1200)
    ingredient_notes: list[IngredientNote] = Field(default_factory=list)


SYSTEM = """You explain product ingredient analysis to a non-expert.

You are given findings that a deterministic rules engine has already produced.
Your job is wording, nothing else.

Absolute rules:
- Only discuss ingredients present in the findings. Never introduce another
  ingredient, even one you would expect in this kind of product.
- Never state a hazard level, verdict or risk the findings do not contain.
- Never soften or contradict a finding. If something is flagged, say so.
- The verdict is already decided and is given to you. Do not restate it as
  your own conclusion or argue with it.
- Where a finding says a concentration limit exists but the label does not
  state the concentration, say exactly that. Do not guess the amount.
- Plain language. No marketing tone, no reassurance that the findings do not
  support, no alarm the findings do not support.
- Personal flags matter most to this reader: lead with them when present.

Write British English, second person, calm and factual."""


class ProductExplainer(Chain):
    name = "product_explain"
    model_class = ModelClass.LARGE
    # Allergies and conditions travel with the findings, so this is user data.
    data_policy = DataPolicy.PSEUDONYMOUS
    temperature = 0.2
    max_tokens = 1200
    system_prompt = SYSTEM

    def explain(self, analysis: ProductAnalysis) -> Optional[ProductExplanation]:
        if not analysis.ingredients:
            return None

        payload = self._findings_payload(analysis)
        result = self.run_structured(
            "Findings:\n" + json.dumps(payload, indent=2), ProductExplanation
        )

        if not result.ok or result.value is None:
            logger.info("product explanation unavailable (%s)", result.error)
            return None

        return self._strip_hallucinated(result.value, analysis)

    def _findings_payload(self, analysis: ProductAnalysis) -> dict:
        hazards = {h.chemical_id: h for h in analysis.hazards}
        flags: dict[str, list[str]] = {}
        for flag in analysis.personal_flags:
            flags.setdefault(flag.chemical_id, []).append(flag.reason)

        ingredients = []
        for ingredient in analysis.ingredients:
            if not ingredient.resolved:
                # Unrecognised tokens are excluded from the prompt entirely:
                # given a name and nothing else the model would fill the gap
                # from memory, which is exactly the hallucination this design
                # exists to prevent. They are reported by a separate block.
                continue

            hazard = hazards.get(ingredient.chemical_id or "")
            ingredients.append(
                {
                    "name": ingredient.display_name or ingredient.raw_token,
                    "position": ingredient.position + 1,
                    "hazard_level": str(hazard.hazard_level.value) if hazard else "unknown",
                    "iarc_group": hazard.iarc_group if hazard else None,
                    "endocrine_lists": hazard.endocrine_lists if hazard else [],
                    "allergen": hazard.allergen_flag if hazard else False,
                    "restricted_in": hazard.restricted_in if hazard else [],
                    "banned_in": hazard.banned_in if hazard else [],
                    "concentration_caveat": hazard.concentration_caveat if hazard else None,
                    "personal_flags": flags.get(ingredient.chemical_id or "", []),
                }
            )

        return {
            "verdict_already_decided": str(analysis.verdict.value),
            "product": analysis.identity.name if analysis.identity else None,
            "total_ingredients": len(analysis.ingredients),
            "unrecognised_count": analysis.unresolved_count,
            "ingredients": ingredients,
        }

    @staticmethod
    def _strip_hallucinated(
        explanation: ProductExplanation, analysis: ProductAnalysis
    ) -> ProductExplanation:
        """Drop per-ingredient notes naming something not in the findings.

        `guard_out` would catch this too, but dropping it here means the user
        never sees it — a guard that only flags after the fact still shipped
        the wrong text.
        """
        known = {
            (i.display_name or i.raw_token).lower()
            for i in analysis.ingredients
            if i.resolved
        }
        kept = []
        for note in explanation.ingredient_notes:
            if note.name.lower() in known:
                kept.append(note)
            else:
                logger.warning("explanation named unknown ingredient %r; dropped", note.name)
        explanation.ingredient_notes = kept
        return explanation
