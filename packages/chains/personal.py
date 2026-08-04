"""The personal agent — the PERSONAL branch of the graph.

A bounded tool-calling loop over the user-scoped toolset. The system prompt is
rendered from the typed `UserContext` at the last moment (arch.md principle 4)
and carries only a *summary*; anything specific the model wants, it fetches
through a tool.

Two limits that matter: the round cap stops a model looping on tools forever,
and the context summary is capped so a user with three years of history does
not silently blow the prompt budget.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from packages.chains.base import Masker
from packages.chains.providers import (
    Completion,
    DataPolicy,
    ModelClass,
    ProviderError,
    complete,
)
from packages.domain.models import UserContext
from packages.tools.health_tools import MAX_TOOL_ROUNDS, TOOL_SCHEMAS, HealthToolset

logger = logging.getLogger(__name__)

SYSTEM = """You are a health and fitness assistant inside a mobile app.

You have tools that read this user's own logged data. Use them whenever a
question depends on their numbers — never guess or invent a value, and never
state a figure you have not fetched.

How to answer:
- Lead with the answer, then the reasoning. Two or three short paragraphs.
- Quote specific numbers you fetched, with their dates or windows.
- If a tool returns no data, say so plainly and say what would fix it. Do not
  substitute population averages for their data without saying that is what
  you are doing.
- Trends need context: a change is only meaningful against a baseline.
- You are not their doctor. For anything clinical — symptoms, medication,
  lab values outside normal range — give general information and recommend
  they speak to a qualified clinician. Never diagnose.
- Do not moralise about their choices. They asked a question; answer it.

British English, second person, direct and warm without being effusive."""


@dataclass
class AgentResult:
    text: str = ""
    tools_used: list[str] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)
    data_fetched: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.text) and self.error is None

    @property
    def total_usd(self) -> float:
        return round(sum(c.usd for c in self.completions), 8)

    def evidence_strings(self) -> list[str]:
        """Tool results as evidence, for the claim verifier."""
        return [f"{name}: {json.dumps(value, default=str)}" for name, value in self.data_fetched.items()]


def render_context_summary(context: Optional[UserContext], max_chars: int = 1500) -> str:
    """A summary, not the dataset (arch.md 6.2)."""
    if context is None:
        return "No profile data is available for this user."

    parts: list[str] = []

    profile = context.profile
    bits = []
    if profile.display_name:
        bits.append(f"name {profile.display_name}")
    if profile.age_band:
        bits.append(f"age {profile.age_band}")
    if profile.sex:
        bits.append(profile.sex)
    if profile.height_cm:
        bits.append(f"{profile.height_cm:g} cm")
    if profile.weight_kg:
        bits.append(f"{profile.weight_kg:g} kg")
    if profile.goals:
        bits.append("goals: " + ", ".join(profile.goals))
    if profile.preferences:
        bits.append("preferences: " + ", ".join(profile.preferences))
    if profile.pregnancy_status:
        bits.append(f"pregnancy status: {profile.pregnancy_status}")
    if bits:
        parts.append("PROFILE — " + "; ".join(bits) + ".")

    if context.vitals.latest:
        latest = ", ".join(
            f"{p.metric} {p.value:g}{p.unit or ''}"
            for p in context.vitals.latest[:8]
            if p.value is not None
        )
        if latest:
            parts.append(f"LATEST METRICS — {latest}.")

    if context.derived.trends:
        trends = ", ".join(f"{k} {v}" for k, v in list(context.derived.trends.items())[:6])
        parts.append(f"RECENT TRENDS — {trends}.")

    medical = context.medical
    if medical.conditions or medical.allergies:
        med_bits = []
        if medical.conditions:
            med_bits.append("conditions: " + ", ".join(medical.conditions))
        if medical.allergies:
            med_bits.append("allergies: " + ", ".join(medical.allergies))
        parts.append("MEDICAL — " + "; ".join(med_bits) + ".")

    # What was withheld matters as much as what was included: without it the
    # model reads absent data as absent activity.
    withheld = [name for name, meta in context.meta.items() if meta.withheld]
    if withheld:
        parts.append(
            f"WITHHELD — the user has not granted access to: {', '.join(sorted(withheld))}. "
            "Do not speculate about these; say you cannot see them."
        )

    summary = "\n".join(parts) if parts else "This user has no data synced yet."
    return summary[:max_chars]


class PersonalAgent:
    def __init__(self, toolset: HealthToolset, masker: Optional[Masker] = None) -> None:
        self.toolset = toolset
        self.masker = masker or Masker()

    def run(
        self,
        question: str,
        context: Optional[UserContext] = None,
        history: Optional[list[dict[str, str]]] = None,
        memory_summary: Optional[str] = None,
    ) -> AgentResult:
        result = AgentResult()

        system = SYSTEM + "\n\n" + render_context_summary(context)
        if memory_summary:
            system += f"\n\nWHAT YOU REMEMBER — {memory_summary[:600]}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in (history or [])[-6:]:
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                messages.append({"role": role, "content": str(turn["content"])[:2000]})
        messages.append({"role": "user", "content": question})

        for round_index in range(MAX_TOOL_ROUNDS):
            try:
                completion = complete(
                    messages,
                    model_class=ModelClass.LARGE,
                    temperature=0.3,
                    max_tokens=1200,
                    tools=TOOL_SCHEMAS,
                )
            except ProviderError as exc:
                logger.warning("personal agent failed: %s", exc)
                result.error = str(exc)
                return result

            result.completions.append(completion)

            # The OpenAI SDK exposes tool calls on the message object, which
            # `complete()` flattens away. Re-issue without tools when the model
            # has stopped calling them.
            raw_calls = self._tool_calls(completion)
            if not raw_calls:
                result.text = completion.text.strip()
                result.tools_used = list(self.toolset.call_log)
                return result

            messages.append(
                {
                    "role": "assistant",
                    "content": completion.text or "",
                    "tool_calls": raw_calls,
                }
            )

            for call in raw_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}

                output = self.toolset.execute(name, args)
                result.data_fetched[f"{name}({json.dumps(args, sort_keys=True)})"] = output
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(output, default=str)[:4000],
                    }
                )

        # Out of rounds: ask once more without tools so the model commits to an
        # answer from what it already fetched, rather than returning nothing.
        try:
            final = complete(messages, model_class=ModelClass.LARGE, temperature=0.3, max_tokens=1000)
            result.completions.append(final)
            result.text = final.text.strip()
        except ProviderError as exc:
            result.error = str(exc)

        result.tools_used = list(self.toolset.call_log)
        return result

    @staticmethod
    def _tool_calls(completion: Completion) -> list[dict[str, Any]]:
        raw = getattr(completion, "tool_calls", None)
        return raw or []
