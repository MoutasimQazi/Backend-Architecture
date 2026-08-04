"""Embedding service.

Feeds three things: the hybrid FAQ retriever (arch.md 5.2), the L3 semantic
cache (7.1) and the resolver's nearest-neighbour stage (8.3). All three read
vectors out of MySQL `VARBINARY` columns and score in-process — there is no
pgvector on this host.

Query embeddings are cached in-process. The same question asked twice in a
minute should not pay for two embedding calls, and the router embeds every
message that reaches S2.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from packages.cache.tiers import LruCache
from packages.chains.providers import ProviderError, embed
from packages.common.text import normalise_question, sha256_hex
from packages.config import get_settings
from packages.storage.vectors import pack

logger = logging.getLogger(__name__)

# Query embeddings only — small, and keyed on normalised text.
_query_cache = LruCache(max_entries=2048, ttl_seconds=900)

BATCH_SIZE = 96


def vector_search_enabled() -> bool:
    """Whether retrieval has a vector leg at all. Cheap; read it per call."""
    return get_settings().models.vector_search_enabled


def embed_query(text_value: str) -> Optional[list[float]]:
    """Embed one query string, cached. Returns None if embeddings are down —
    callers fall back to lexical retrieval rather than failing the turn."""
    # Checked before any work: on a lexical-only deployment this is every
    # single turn, and it should cost nothing and say nothing.
    if not vector_search_enabled():
        return None

    normalised = normalise_question(text_value)
    if not normalised:
        return None

    key = sha256_hex(normalised, get_settings().models.embedding_model)
    cached = _query_cache.get(key)
    if cached is not None:
        return cached

    try:
        vectors = embed([normalised])
    except ProviderError as exc:
        logger.warning("query embedding unavailable: %s", exc)
        return None

    if not vectors:
        return None

    _query_cache.put(key, vectors[0])
    return vectors[0]


def embed_many(texts: list[str]) -> list[Optional[list[float]]]:
    """Batch, chunked to the provider's practical limit."""
    if not texts:
        return []

    if not vector_search_enabled():
        return [None] * len(texts)

    out: list[Optional[list[float]]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start : start + BATCH_SIZE]
        try:
            out.extend(embed(chunk))
        except ProviderError as exc:
            logger.warning("batch embedding failed for %d texts: %s", len(chunk), exc)
            out.extend([None] * len(chunk))
    return out


# -- backfills --------------------------------------------------------------

# Reported instead of {"failed": N}: with vector search off there is nothing
# to do and nothing wrong, and a worker logging thousands of "failures" every
# run is how real failures stop being noticed.
_DISABLED_RESULT = {"pending": 0, "embedded": 0, "failed": 0, "disabled": 1}


def backfill_faq_surfaces(session: Session, limit: int = 500) -> dict[str, int]:
    """Embed FAQ surface forms that have none.

    Idempotent and resumable: it only selects rows where `embedding IS NULL`,
    so an interrupted run continues where it stopped.
    """
    if not vector_search_enabled():
        return dict(_DISABLED_RESULT)

    settings = get_settings().models
    rows = session.execute(
        sql("SELECT id, surface_text FROM faq_surface WHERE embedding IS NULL LIMIT :lim"),
        {"lim": limit},
    ).all()

    if not rows:
        return {"pending": 0, "embedded": 0, "failed": 0}

    texts = [normalise_question(r[1]) for r in rows]
    vectors = embed_many(texts)

    embedded = failed = 0
    for (surface_id, _), vector in zip(rows, vectors):
        if vector is None:
            failed += 1
            continue
        session.execute(
            sql(
                "UPDATE faq_surface SET embedding = :e, embedding_model = :m, embedding_dim = :d "
                "WHERE id = :sid"
            ),
            {
                "e": pack(vector),
                "m": settings.embedding_model,
                "d": len(vector),
                "sid": surface_id,
            },
        )
        embedded += 1

    remaining = session.execute(
        sql("SELECT COUNT(*) FROM faq_surface WHERE embedding IS NULL")
    ).scalar()

    logger.info("faq backfill: %d embedded, %d failed, %d left", embedded, failed, remaining)
    return {"pending": int(remaining or 0), "embedded": embedded, "failed": failed}


def backfill_chemical_synonyms(session: Session, limit: int = 500) -> dict[str, int]:
    """Embed chemical synonyms for the resolver's nearest-neighbour stage."""
    if not vector_search_enabled():
        return dict(_DISABLED_RESULT)

    settings = get_settings().models
    rows = session.execute(
        sql("SELECT id, synonym FROM chemical_synonym WHERE embedding IS NULL LIMIT :lim"),
        {"lim": limit},
    ).all()

    if not rows:
        return {"pending": 0, "embedded": 0, "failed": 0}

    vectors = embed_many([r[1] for r in rows])

    embedded = failed = 0
    for (row_id, _), vector in zip(rows, vectors):
        if vector is None:
            failed += 1
            continue
        session.execute(
            sql("UPDATE chemical_synonym SET embedding = :e, embedding_model = :m WHERE id = :rid"),
            {"e": pack(vector), "m": settings.embedding_model, "rid": row_id},
        )
        embedded += 1

    remaining = session.execute(
        sql("SELECT COUNT(*) FROM chemical_synonym WHERE embedding IS NULL")
    ).scalar()
    return {"pending": int(remaining or 0), "embedded": embedded, "failed": failed}


def backfill_evidence_chunks(session: Session, limit: int = 200) -> dict[str, int]:
    if not vector_search_enabled():
        return dict(_DISABLED_RESULT)

    settings = get_settings().models
    rows = session.execute(
        sql("SELECT id, chunk_text FROM evidence_chunk WHERE embedding IS NULL LIMIT :lim"),
        {"lim": limit},
    ).all()

    if not rows:
        return {"pending": 0, "embedded": 0, "failed": 0}

    # Chunks can be long; the embedding endpoint has a token ceiling.
    vectors = embed_many([r[1][:6000] for r in rows])

    embedded = failed = 0
    for (row_id, _), vector in zip(rows, vectors):
        if vector is None:
            failed += 1
            continue
        session.execute(
            sql("UPDATE evidence_chunk SET embedding = :e, embedding_model = :m WHERE id = :rid"),
            {"e": pack(vector), "m": settings.embedding_model, "rid": row_id},
        )
        embedded += 1

    remaining = session.execute(
        sql("SELECT COUNT(*) FROM evidence_chunk WHERE embedding IS NULL")
    ).scalar()
    return {"pending": int(remaining or 0), "embedded": embedded, "failed": failed}
