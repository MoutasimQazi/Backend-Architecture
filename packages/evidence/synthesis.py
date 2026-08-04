"""Grounded synthesis across sources (arch.md 9.4).

Extract → summarise per source → synthesise across sources, with each output
sentence carrying the chunks that support it. Conflicting findings are reported
as disagreement, not averaged away — "studies disagree" is the honest answer
when they do, and averaging it into a confident middle is how health
misinformation gets manufactured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from packages.chains.base import Chain
from packages.chains.providers import DataPolicy, ModelClass, is_configured
from packages.chains.verify import ClaimVerifier
from packages.domain.models import Citation, EvidenceRef
from packages.evidence.tiers import rank_score
from packages.storage.repositories.evidence import EvidenceRepository

logger = logging.getLogger(__name__)

MAX_SOURCES = 8


class SynthesisOutput(BaseModel):
    answer: str = Field(max_length=2000)
    consensus: Literal["strong", "moderate", "limited", "conflicting", "insufficient"]
    disagreement: str = Field(default="", max_length=600)
    caveats: list[str] = Field(default_factory=list)


SYSTEM = """You answer health questions strictly from supplied sources.

Rules:
- Use only the sources given. If they do not answer the question, say so and
  set consensus to "insufficient". That is a complete, correct answer.
- Where sources disagree, report the disagreement in the `disagreement` field
  and set consensus to "conflicting". Never average conflicting findings into
  a confident middle position.
- Weight by study design and independence, both of which are given. A
  meta-analysis outweighs one small trial; an industry-funded study on its own
  sponsor's product deserves explicit mention.
- Never state a number that is not in the sources.
- No treatment advice and no dosing. Describe what the evidence shows and
  recommend a clinician for anything individual.
- Plain British English, second person, three short paragraphs at most."""


@dataclass
class SynthesisResult:
    answer: str = ""
    consensus: str = "insufficient"
    disagreement: str = ""
    caveats: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    sources_used: int = 0
    unsupported_sentences: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.answer) and self.error is None


class EvidenceSynthesiser(Chain):
    name = "evidence_synthesise"
    model_class = ModelClass.LARGE
    data_policy = DataPolicy.PUBLIC  # the question, not the user
    temperature = 0.2
    max_tokens = 1400
    system_prompt = SYSTEM

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.repo = EvidenceRepository(session)

    def answer(self, question: str, *, verify: bool = True) -> SynthesisResult:
        if not is_configured():
            return SynthesisResult(error="no model provider configured")

        refs = self.repo.search(question, limit=MAX_SOURCES * 2)
        if not refs:
            return SynthesisResult(
                consensus="insufficient",
                error="no matching evidence in the knowledge base",
            )

        # arch.md 9.3: rank by tier x independence x design x recency.
        ranked = sorted(
            refs,
            key=lambda r: rank_score(r.tier, r.independence or 0.5, r.study_design, r.year),
            reverse=True,
        )[:MAX_SOURCES]

        chunks = self.repo.chunks_for([r.source_id for r in ranked])
        sources_payload = []
        for index, ref in enumerate(ranked, start=1):
            body = " ".join(chunks.get(ref.source_id, []))[:2500]
            sources_payload.append(
                {
                    "id": index,
                    "source_id": ref.source_id,
                    "title": ref.title,
                    "year": ref.year,
                    "tier": str(ref.tier.value),
                    "study_design": ref.study_design,
                    "independence": ref.independence,
                    "content": body or (ref.title or ""),
                }
            )

        result = self.run_structured(
            f"Question: {question}\n\nSources:\n{json.dumps(sources_payload, indent=2)}",
            SynthesisOutput,
        )

        if not result.ok or result.value is None:
            return SynthesisResult(error=result.error or "synthesis failed")

        output = result.value
        synthesis = SynthesisResult(
            answer=output.answer,
            consensus=output.consensus,
            disagreement=output.disagreement,
            caveats=output.caveats,
            sources_used=len(ranked),
            citations=[
                Citation(
                    citation_id=ref.source_id,
                    source=ref.title or ref.source_id,
                    tier=ref.tier,
                    url=ref.url,
                    title=ref.title,
                    retrieved_at=ref.retrieved_at,
                    supports_block_ids=["research_1"],
                )
                for ref in ranked
            ],
        )

        # arch.md 9.4: a verification pass checks entailment; unsupported
        # sentences are reported rather than shipped as established fact.
        if verify:
            evidence_texts = [s["content"] for s in sources_payload if s["content"]]
            report = ClaimVerifier().verify(output.answer, evidence_texts)
            if report.checked and report.unsupported:
                synthesis.unsupported_sentences = report.unsupported
                synthesis.caveats.append(
                    f"{len(report.unsupported)} statement(s) could not be traced to a cited source."
                )
                if synthesis.consensus in ("strong", "moderate"):
                    synthesis.consensus = "limited"

        return synthesis
