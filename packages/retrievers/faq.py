"""Hybrid FAQ retrieval (arch.md 5.2).

Three stages:

  1. **Exact** — normalised-string hash. O(1), no model.
  2. **Hybrid candidates** — top-50 BM25 (MySQL FULLTEXT) ∪ top-50 embedding
     cosine, fused with Reciprocal Rank Fusion.
  3. **Rerank** — cross-encoder over the top-20.

Why both halves: lexical recall catches the tokens embeddings blur — `TDEE`,
`BMR`, `INCI`, `E211` — while the embedding half catches phrasings that share
no words with the canonical question. Fusing by *rank* rather than score means
FULLTEXT relevance and cosine never have to be put on a common scale.

The vector index lives in-process and reloads when the live KB version token
changes, so publishing an FAQ is visible without a restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from packages.chains.embeddings import embed_query
from packages.domain.enums import RouteStage
from packages.domain.models import FaqItem
from packages.storage.repositories.faq import FaqRepository
from packages.storage.vectors import get_index, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

CANDIDATES_PER_ARM = 50
RERANK_DEPTH = 20


@dataclass
class RetrievedFaq:
    faq_id: str
    score: float
    stage: RouteStage
    matched_surface: str = ""
    item: Optional[FaqItem] = None
    lexical_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    vector_score: float = 0.0

    @property
    def both_arms(self) -> bool:
        return self.lexical_rank is not None and self.vector_rank is not None


class FaqRetriever:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = FaqRepository(session)

    # -- stage 1 ----------------------------------------------------------

    def exact(self, question: str, locale: str = "en") -> Optional[RetrievedFaq]:
        hit = self.repo.find_exact(question, locale)
        if hit is None:
            return None
        faq_id, surface = hit
        return RetrievedFaq(
            faq_id=faq_id,
            score=1.0,
            stage=RouteStage.S1_EXACT,
            matched_surface=surface,
            item=self.repo.get(faq_id),
        )

    # -- stage 2 ----------------------------------------------------------

    def _vector_index(self, locale: str):
        index = get_index(f"faq:{locale}")
        token = self.repo.live_version_token(locale)
        if index.is_stale(token):
            index.load(self.repo.embedding_rows(locale), version_token=token)
        return index

    def hybrid(self, question: str, locale: str = "en", top_k: int = 10) -> list[RetrievedFaq]:
        """RRF over the lexical and vector arms."""
        lexical = self.repo.search_lexical(question, locale, limit=CANDIDATES_PER_ARM)
        lexical_ids = [faq_id for faq_id, _ in lexical]
        lexical_rank = {faq_id: i + 1 for i, faq_id in enumerate(lexical_ids)}

        vector_ids: list[str] = []
        vector_score: dict[str, float] = {}

        query_vector = embed_query(question)
        if query_vector is not None:
            index = self._vector_index(locale)
            if index.size:
                for scored in index.search(query_vector, top_k=CANDIDATES_PER_ARM):
                    faq_id = scored.payload.get("faq_id")
                    if not faq_id:
                        continue
                    # Several surfaces map to one item; keep its best.
                    if faq_id not in vector_score or scored.score > vector_score[faq_id]:
                        vector_score[faq_id] = scored.score
                    if faq_id not in vector_ids:
                        vector_ids.append(faq_id)
        else:
            # Embeddings unavailable — lexical alone still answers, with a
            # narrower recall. Degrading beats returning nothing.
            logger.debug("faq hybrid running lexical-only (no query embedding)")

        vector_rank = {faq_id: i + 1 for i, faq_id in enumerate(vector_ids)}

        arms = [ids for ids in (lexical_ids, vector_ids) if ids]
        if not arms:
            return []

        fused = reciprocal_rank_fusion(arms)[:top_k]
        items = self.repo.get_many([faq_id for faq_id, _ in fused])

        results = [
            RetrievedFaq(
                faq_id=faq_id,
                score=rrf_score,
                stage=RouteStage.S2_RETRIEVAL,
                matched_surface=items[faq_id].canonical_question if faq_id in items else "",
                item=items.get(faq_id),
                lexical_rank=lexical_rank.get(faq_id),
                vector_rank=vector_rank.get(faq_id),
                vector_score=vector_score.get(faq_id, 0.0),
            )
            for faq_id, rrf_score in fused
        ]
        return results

    # -- stage 3 ----------------------------------------------------------

    def rerank(self, question: str, candidates: list[RetrievedFaq]) -> list[RetrievedFaq]:
        """Score candidates in [0,1].

        A cross-encoder is the arch.md choice; it needs a model this deployment
        does not host. The calibration used instead is deliberately conservative
        and built from signals already computed:

          * cosine against the canonical question, when embeddings are up
          * agreement between the two arms — appearing in both is strong
          * rank position in each arm

        It produces a comparable score without a second model, and the rerank
        interface stays put for when a cross-encoder is available.
        """
        if not candidates:
            return []

        query_vector = embed_query(question)
        pool = candidates[:RERANK_DEPTH]

        for candidate in pool:
            score = 0.0

            if query_vector is not None and candidate.vector_score:
                # Cosine dominates when we have it: it is the only signal that
                # actually compares meanings.
                score = candidate.vector_score
            elif candidate.lexical_rank is not None:
                # Lexical-only: decay by rank. Capped below the FAQ threshold's
                # typical value so a lexical-only hit rarely auto-routes.
                score = max(0.0, 0.80 - 0.05 * (candidate.lexical_rank - 1))

            if candidate.both_arms:
                score = min(1.0, score + 0.10)

            candidate.score = round(score, 4)

        pool.sort(key=lambda c: c.score, reverse=True)
        return pool

    # -- the whole pipeline ----------------------------------------------

    def retrieve(
        self, question: str, locale: str = "en", top_k: int = 5
    ) -> tuple[list[RetrievedFaq], RouteStage]:
        """Exact first; otherwise hybrid + rerank. Returns the stage that decided."""
        exact = self.exact(question, locale)
        if exact is not None:
            return [exact], RouteStage.S1_EXACT

        candidates = self.hybrid(question, locale, top_k=RERANK_DEPTH)
        if not candidates:
            return [], RouteStage.S2_RETRIEVAL

        return self.rerank(question, candidates)[:top_k], RouteStage.S2_RETRIEVAL
