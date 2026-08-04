"""Conversation memory (arch.md 12).

Two chains:

  * **Rolling summary** — durable facts only, updated asynchronously after the
    turn so it never sits on the critical path.
  * **Fact extraction** — typed rows the user can see, query and delete, rather
    than prose prepended to a prompt. This is what makes "what do you remember
    about me" answerable and "forget that" actually possible.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from packages.chains.base import Chain
from packages.chains.providers import DataPolicy, ModelClass

logger = logging.getLogger(__name__)

# Closed vocabulary: an open one drifts into hundreds of near-duplicate kinds
# that nothing can query.
MEMORY_KINDS = (
    "goal",
    "dietary_restriction",
    "disliked_exercise",
    "preferred_exercise",
    "constraint",
    "injury",
    "schedule",
    "equipment",
    "motivation",
)


class ExtractedFact(BaseModel):
    kind: Literal[
        "goal",
        "dietary_restriction",
        "disliked_exercise",
        "preferred_exercise",
        "constraint",
        "injury",
        "schedule",
        "equipment",
        "motivation",
    ]
    value: str = Field(max_length=200)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class FactExtraction(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


SUMMARY_SYSTEM = """You maintain a compact long-term memory of a fitness app user.

Merge the previous summary with the new conversation into an updated summary.

Keep: goals, constraints, injuries, dietary restrictions, equipment, schedule,
recurring concerns, and preferences they have stated.

Drop: small talk, one-off questions, anything already in their structured
profile (weight, age), and anything they only asked about rather than stated
about themselves.

Never invent. If the conversation adds nothing durable, return the previous
summary unchanged.

Under 200 words. Plain sentences, no headings or bullets."""


FACT_SYSTEM = """You extract durable facts a user has stated about themselves.

Only extract what the user asserted about their own situation. Do not extract:
- questions they asked ("is creatine good?" is not a preference)
- anything the assistant said
- transient states ("I'm tired today")
- numbers already tracked by the app (weight, sleep hours)

Each fact must be a short, self-contained phrase that will still make sense in
six months, with no pronouns.

Return an empty list when nothing durable was stated. That is the common case
and it is the correct answer."""


class MemorySummariser(Chain):
    name = "memory_summarise"
    model_class = ModelClass.SMALL
    data_policy = DataPolicy.PSEUDONYMOUS
    temperature = 0.2
    max_tokens = 400
    system_prompt = SUMMARY_SYSTEM

    def summarise(
        self, turns: list[dict[str, str]], previous: Optional[str] = None
    ) -> Optional[str]:
        if not turns:
            return previous

        transcript = "\n".join(
            f"{t.get('role', '?')}: {str(t.get('content') or '')[:800]}" for t in turns[-20:]
        )

        prompt = ""
        if previous:
            prompt += f"PREVIOUS SUMMARY:\n{previous}\n\n"
        prompt += f"NEW CONVERSATION:\n{transcript}"

        result = self.run_text(prompt)
        if not result.ok:
            logger.info("memory summarisation unavailable (%s)", result.error)
            return previous

        text = (result.value or "").strip()
        return text or previous


class FactExtractor(Chain):
    name = "memory_extract"
    model_class = ModelClass.SMALL
    data_policy = DataPolicy.PSEUDONYMOUS
    temperature = 0.0
    max_tokens = 500
    system_prompt = FACT_SYSTEM

    def extract(self, turns: list[dict[str, str]]) -> list[ExtractedFact]:
        user_turns = [t for t in turns if t.get("role") == "user"]
        if not user_turns:
            return []

        transcript = "\n".join(str(t.get("content") or "")[:600] for t in user_turns[-10:])
        result = self.run_structured(f"User messages:\n{transcript}", FactExtraction)

        if not result.ok or result.value is None:
            return []

        # A "fact" with no content is noise the model produced to fill the schema.
        return [f for f in result.value.facts if f.value.strip() and f.confidence >= 0.5]
