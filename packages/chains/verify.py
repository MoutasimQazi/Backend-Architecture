"""Claim verification (arch.md 4.4, 9.4, and the guard_out contract in 4.3).

Every sentence the assistant asserts should be entailed by something it
retrieved. This runs the check and reports which sentences are not supported;
`guard_out` decides what to do with them — drop, or downgrade to "limited
evidence" — because that decision is policy, not model output.

Deliberately a separate small-model call rather than folded into generation:
a model asked to both write and self-certify does neither reliably.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

from packages.chains.base import Chain
from packages.chains.providers import DataPolicy, ModelClass, is_configured

logger = logging.getLogger(__name__)


class SentenceVerdict(BaseModel):
    index: int
    status: Literal["supported", "partial", "unsupported"]
    reason: str = Field(default="", max_length=200)


class VerificationReport(BaseModel):
    verdicts: list[SentenceVerdict] = Field(default_factory=list)


SYSTEM = """You check whether statements are supported by given evidence.

For each numbered sentence, decide:
  supported   — the evidence directly states or clearly implies it
  partial     — partly supported, or supported with a qualification the
                sentence omits
  unsupported — the evidence does not establish it, or contradicts it

Judge only against the evidence provided. Your own knowledge is irrelevant
here, even when the sentence is true in general — an unsupported true
statement is still unsupported.

General framing, hedges and questions ("this varies between people", "worth
discussing with your doctor") count as supported: they assert no fact.

Be strict about numbers. A sentence citing a figure the evidence does not
contain is unsupported."""

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class VerificationResult:
    supported: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    checked: bool = False
    error: Optional[str] = None

    @property
    def all_supported(self) -> bool:
        return self.checked and not self.unsupported

    @property
    def confidence_penalty(self) -> float:
        """How far to knock down the answer's confidence."""
        total = len(self.supported) + len(self.partial) + len(self.unsupported)
        if not total or not self.checked:
            return 0.0
        return round((len(self.unsupported) + 0.5 * len(self.partial)) / total, 3)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


class ClaimVerifier(Chain):
    name = "claim_verify"
    model_class = ModelClass.SMALL
    data_policy = DataPolicy.PSEUDONYMOUS
    temperature = 0.0
    max_tokens = 800
    system_prompt = SYSTEM

    def verify(self, claim_text: str, evidence: list[str]) -> VerificationResult:
        sentences = split_sentences(claim_text)
        if not sentences:
            return VerificationResult(checked=False)

        if not evidence:
            # Nothing to check against. Reporting everything as unsupported
            # would be as misleading as reporting it supported.
            return VerificationResult(checked=False, error="no evidence supplied")

        if not is_configured():
            return VerificationResult(checked=False, error="no model provider configured")

        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
        evidence_block = "\n".join(f"- {e[:1200]}" for e in evidence[:20])

        result = self.run_structured(
            f"Evidence:\n{evidence_block}\n\nSentences:\n{numbered}",
            VerificationReport,
        )

        if not result.ok or result.value is None:
            return VerificationResult(checked=False, error=result.error)

        out = VerificationResult(checked=True)
        seen: set[int] = set()

        for verdict in result.value.verdicts:
            if not 0 <= verdict.index < len(sentences):
                continue
            seen.add(verdict.index)
            sentence = sentences[verdict.index]
            if verdict.status == "supported":
                out.supported.append(sentence)
            elif verdict.status == "partial":
                out.partial.append(sentence)
            else:
                out.unsupported.append(sentence)

        # A sentence the verifier skipped has not been checked. Treat it as
        # partial rather than assuming it passed.
        for index, sentence in enumerate(sentences):
            if index not in seen:
                out.partial.append(sentence)

        if out.unsupported:
            logger.info(
                "verification: %d/%d sentences unsupported",
                len(out.unsupported),
                len(sentences),
            )
        return out
