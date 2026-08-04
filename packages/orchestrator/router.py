"""The router cascade (arch.md 5.3).

Cheapest first; each stage can decide or defer. S0-S2 need no model call at
all, which is the point: today every "hi" costs a frontier-model call (P5).

S4 (small-LLM classifier for the ambiguous band) is the only stage that spends
money, and it only sees questions the free stages could not settle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from packages.chains.classify import RouterClassifier
from packages.chains.providers import is_configured
from packages.common.text import normalise_question
from packages.config import get_settings
from packages.domain.enums import FaqCategory, RouteLabel, RouteStage, SafetyFlagKind
from packages.domain.models import FaqHit, RouteDecision, SafetyFlag
from packages.retrievers.faq import FaqRetriever
from packages.storage.repositories.faq import FaqRepository

logger = logging.getLogger(__name__)

# A trailing "there", "everyone", "a lot" etc. keeps it small talk. Without
# them "hey there" and "thanks a lot" fell through to the expensive branch,
# which is exactly the cost the cascade exists to avoid (arch.md P5).
_GREETING = re.compile(
    r"^(hi|hey|hello|yo|hola|namaste|good\s+(morning|afternoon|evening)|"
    r"thanks?|thank\s+you|thx|ty|ok|okay|cool|nice|great|got\s+it|"
    r"bye|goodbye|see\s+you|good\s+night)"
    r"(\s+(there|everyone|all|mate|buddy|again|so\s+much|a\s+lot|man))?"
    r"\b[\s!.?]*$"
)

_RESTAURANT = re.compile(
    r"\b(restaurant|cafe|café|diner|eatery|hotel|dhaba|outlet|branch|food\s+court)\b"
)
_RESTAURANT_INTENT = re.compile(
    r"\b(safe|hygien\w*|clean|inspection|review|rating|licen[cs]e|fssai|complaint|food\s*poison\w*)\b"
)

_RESEARCH = re.compile(
    r"\b(study|studies|research|evidence|paper|meta[-\s]?analysis|systematic\s+review|"
    r"trial|clinical|is\s+\w+\s+(safe|harmful|toxic|carcinogenic)|side\s+effects?)\b"
)

_PERSONAL = re.compile(
    r"\b(my|mine|i\s+(am|have|feel|weigh|slept|ate|did)|me\b|our)\b|"
    r"\b(how\s+(am|did)\s+i|should\s+i)\b"
)

# arch.md 4.3: guard_in triages these. Matching here only sets the route to
# UNSAFE; the safety response itself is a deterministic template, never a model
# call, so a crisis cannot be answered by a hallucination.
_SELF_HARM = re.compile(
    r"\b(kill\s+myself|suicide|suicidal|end\s+my\s+life|self[-\s]?harm|cut\s+myself|"
    r"want\s+to\s+die|no\s+reason\s+to\s+live)\b"
)
_EMERGENCY = re.compile(
    r"\b(chest\s+pain|can'?t\s+breathe|cannot\s+breathe|severe\s+bleeding|unconscious|"
    r"stroke|heart\s+attack|anaphyla\w*|overdose|poisoned)\b"
)
_DISORDERED_EATING = re.compile(
    r"\b(purge|purging|make\s+myself\s+(sick|vomit)|starve\s+myself|not\s+eating\s+for\s+days|"
    r"laxative\s+(abuse|to\s+lose)|anorexi\w*|bulimi\w*)\b"
)


@dataclass
class RouterOutcome:
    decision: RouteDecision
    faq_hits: list[FaqHit]
    safety_flags: list[SafetyFlag]


class Router:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.faq_repo = FaqRepository(session)
        self.retriever = FaqRetriever(session)
        # Hits that were retrieved but did not clear the threshold. Still worth
        # handing to the branch that runs instead.
        self._deferred_hits: list[FaqHit] = []

    def route(
        self,
        text: str,
        *,
        locale: str = "en",
        has_attachments: bool = False,
    ) -> RouterOutcome:
        normalised = normalise_question(text)
        flags = self._safety_triage(text)
        self._deferred_hits = []

        if any(f.blocking for f in flags):
            return RouterOutcome(
                decision=RouteDecision(
                    label=RouteLabel.UNSAFE,
                    confidence=1.0,
                    stage=RouteStage.S0_RULES,
                    rationale="safety trigger matched",
                ),
                faq_hits=[],
                safety_flags=flags,
            )

        s0 = self._stage_rules(normalised, has_attachments=has_attachments)
        if s0 is not None:
            return RouterOutcome(decision=s0, faq_hits=[], safety_flags=flags)

        s1 = self._stage_exact(text, locale)
        if s1 is not None:
            decision, hits = s1
            return RouterOutcome(decision=decision, faq_hits=hits, safety_flags=flags)

        s2 = self._stage_hybrid(text, locale)
        if s2 is not None:
            decision, hits = s2
            return RouterOutcome(decision=decision, faq_hits=hits, safety_flags=flags)

        # S3 (semantic cache) is probed by the pipeline right after routing.
        s4 = self._stage_llm(text)
        if s4 is not None:
            return RouterOutcome(
                decision=s4, faq_hits=self._deferred_hits, safety_flags=flags
            )

        return RouterOutcome(
            decision=self._stage_heuristic(normalised),
            faq_hits=self._deferred_hits,
            safety_flags=flags,
        )

    # -- stages -----------------------------------------------------------

    def _safety_triage(self, text: str) -> list[SafetyFlag]:
        from packages.domain.enums import Severity

        lowered = (text or "").lower()
        flags: list[SafetyFlag] = []

        if _SELF_HARM.search(lowered):
            flags.append(
                SafetyFlag(
                    kind=SafetyFlagKind.SELF_HARM,
                    detail="self-harm language detected",
                    severity=Severity.CRITICAL,
                    blocking=True,
                )
            )
        if _EMERGENCY.search(lowered):
            flags.append(
                SafetyFlag(
                    kind=SafetyFlagKind.EMERGENCY_SYMPTOM,
                    detail="acute symptom language detected",
                    severity=Severity.CRITICAL,
                    blocking=True,
                )
            )
        if _DISORDERED_EATING.search(lowered):
            flags.append(
                SafetyFlag(
                    kind=SafetyFlagKind.DISORDERED_EATING,
                    detail="disordered-eating language detected",
                    severity=Severity.HIGH,
                    blocking=True,
                )
            )
        return flags

    def _stage_rules(self, normalised: str, *, has_attachments: bool) -> Optional[RouteDecision]:
        if has_attachments:
            return RouteDecision(
                label=RouteLabel.PRODUCT,
                confidence=0.99,
                stage=RouteStage.S0_RULES,
                rationale="attachments present",
                category=FaqCategory.PRODUCT,
            )

        if not normalised:
            return RouteDecision(
                label=RouteLabel.SMALLTALK,
                confidence=0.9,
                stage=RouteStage.S0_RULES,
                rationale="empty message",
            )

        if _GREETING.match(normalised):
            return RouteDecision(
                label=RouteLabel.SMALLTALK,
                confidence=0.95,
                stage=RouteStage.S0_RULES,
                rationale="greeting or acknowledgement",
            )

        if _RESTAURANT.search(normalised) and _RESTAURANT_INTENT.search(normalised):
            return RouteDecision(
                label=RouteLabel.RESTAURANT,
                confidence=0.85,
                stage=RouteStage.S0_RULES,
                rationale="place entity with review intent",
            )

        return None

    def _stage_exact(self, text: str, locale: str) -> Optional[tuple[RouteDecision, list[FaqHit]]]:
        match = self.faq_repo.find_exact(text, locale)
        if not match:
            return None
        faq_id, surface = match
        item = self.faq_repo.get(faq_id)
        hit = FaqHit(faq_id=faq_id, score=1.0, matched_surface=surface, stage=RouteStage.S1_EXACT, item=item)
        return (
            RouteDecision(
                label=RouteLabel.FAQ,
                confidence=1.0,
                stage=RouteStage.S1_EXACT,
                rationale="normalised-string hash hit",
                category=item.category if item else None,
                fallbacks=[RouteLabel.PERSONAL],
            ),
            [hit],
        )

    def _stage_hybrid(self, text: str, locale: str) -> Optional[tuple[RouteDecision, list[FaqHit]]]:
        """arch.md 5.2 stage 2-3: hybrid candidates, fused and reranked.

        Returns a decision only when the reranked score clears the category's
        threshold. Below it the hits are still handed back as context — a
        near-miss FAQ is useful to the personal agent even when it is not a
        confident enough answer on its own.
        """
        candidates, _ = self.retriever.retrieve(text, locale, top_k=5)
        if not candidates:
            return None

        top = candidates[0]
        category = top.item.category if top.item else None
        threshold = self.settings.router.tau_faq_for(str(category.value) if category else None)

        hits = [
            FaqHit(
                faq_id=c.faq_id,
                score=c.score,
                matched_surface=c.matched_surface,
                stage=RouteStage.S2_EMBEDDING,
                item=c.item,
            )
            for c in candidates
        ]

        if top.score < threshold:
            self._deferred_hits = hits
            return None

        arms = "both arms" if top.both_arms else "one arm"
        return (
            RouteDecision(
                label=RouteLabel.FAQ,
                confidence=top.score,
                stage=RouteStage.S2_EMBEDDING,
                rationale=f"hybrid {top.score:.2f} >= tau {threshold:.2f} ({arms})",
                category=category,
                fallbacks=[RouteLabel.PERSONAL],
            ),
            hits,
        )

    def _stage_llm(self, text: str) -> Optional[RouteDecision]:
        """S4 — the only stage that spends money, and only on the ambiguous band."""
        if not is_configured():
            return None

        outcome = RouterClassifier().classify(text)
        if outcome is None:
            return None

        label, confidence, rationale = outcome

        # A low-confidence classification is not better than the heuristics; it
        # just costs more. Fall through and let them decide.
        if confidence < 0.5:
            logger.info("S4 low confidence (%.2f) for %r; using heuristics", confidence, text[:60])
            return None

        # FAQ needs a retrieved item to answer from. S2 already looked and did
        # not clear the bar, so honour that rather than routing to an empty branch.
        if label is RouteLabel.FAQ:
            label = RouteLabel.PERSONAL
            rationale = f"S4 said FAQ but no item cleared tau; {rationale}"

        return RouteDecision(
            label=label,
            confidence=confidence,
            stage=RouteStage.S4_LLM,
            rationale=rationale[:400],
            fallbacks=[RouteLabel.PERSONAL],
        )

    def _stage_heuristic(self, normalised: str) -> RouteDecision:
        """Last resort: keyword shape. Reached when S4 is unconfigured, failed,
        or was not confident enough."""
        if _RESEARCH.search(normalised):
            return RouteDecision(
                label=RouteLabel.RESEARCH,
                confidence=0.55,
                stage=RouteStage.FALLBACK,
                rationale="research intent keywords",
                fallbacks=[RouteLabel.PERSONAL],
            )
        if _PERSONAL.search(normalised):
            return RouteDecision(
                label=RouteLabel.PERSONAL,
                confidence=0.55,
                stage=RouteStage.FALLBACK,
                rationale="first-person reference",
            )
        return RouteDecision(
            label=RouteLabel.PERSONAL,
            confidence=0.4,
            stage=RouteStage.FALLBACK,
            rationale="no stage decided; defaulting to the personal agent",
        )
