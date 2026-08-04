"""Process-local cache for the Chemical Knowledge Base.

Every ingredient panel is mostly the same substances. "Aqua", "Glycerin",
"Parfum", "Citric Acid", "Tocopherol" appear on a large share of labels, so
without a cache each scan re-runs the same synonym-hash lookups and re-fetches
the same dossiers and assertions from MySQL. The work is identical every time
and the answer only changes when the ETL writes.

Scope is one process. That is deliberate rather than a limitation:

  * The KB is small and slow-moving — the whole working set fits in memory,
    and a shared cache would add a network hop to save a local dictionary
    lookup.
  * There is no Redis on this host by design (see 001_init.sql), and putting
    KB rows into the `cache_entry` table would mean a MySQL round trip to
    avoid a MySQL round trip.

Correctness comes from the version token. Every cached value is stored under a
key that includes a token derived from the KB's own state, so an ETL run that
adds a chemical or publishes a dossier makes every stale entry unreachable
rather than merely old. That matters here more than in an ordinary cache: a
hazard assertion that has been corrected must not keep being served, because
the output is a safety verdict.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.cache.tiers import LruCache
from packages.config import get_settings

logger = logging.getLogger(__name__)

# Dossiers and assertions are small dicts; a few thousand is a modest footprint
# and covers far more distinct chemicals than any real KB holds today.
_dossier_cache = LruCache(max_entries=4096, ttl_seconds=3600)
_assertion_cache = LruCache(max_entries=4096, ttl_seconds=3600)
_synonym_cache = LruCache(max_entries=8192, ttl_seconds=3600)

# Re-deriving the version token per lookup would defeat the point — it is a
# query itself. It is re-read at most this often.
_VERSION_TTL_SECONDS = 30

_version_token: Optional[str] = None
_version_checked_at: float = 0.0
_version_lock = threading.Lock()


def version_token(session: Session, *, force: bool = False) -> str:
    """A token that changes whenever the KB changes.

    COUNT plus MAX(updated_at) catches all three ways the KB moves: a new
    chemical, an edited dossier, and a draft being published. Both columns are
    indexed, so this is cheap, and it is re-read at most every 30 seconds.
    """
    global _version_token, _version_checked_at

    now = time.monotonic()
    with _version_lock:
        if not force and _version_token is not None and now - _version_checked_at < _VERSION_TTL_SECONDS:
            return _version_token

    try:
        row = session.execute(
            text(
                "SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), '0') AS t FROM chemical"
            )
        ).mappings().first()
        token = f"{get_settings().kb_version}:{row['n']}:{row['t']}" if row else "unknown"
    except Exception:
        # A cache that cannot verify freshness must not serve stale safety
        # data, so an unreadable token becomes a unique one: everything misses
        # and goes to the database.
        logger.exception("could not read KB version token; bypassing chemical cache")
        token = f"error:{time.time()}"

    with _version_lock:
        _version_token = token
        _version_checked_at = now
    return token


def _get_or_load(
    cache: LruCache,
    session: Session,
    prefix: str,
    keys: list[str],
    loader: Callable[[list[str]], dict[str, Any]],
) -> dict[str, Any]:
    """Serve what is cached, load only the rest, remember the answer.

    Misses are cached too, as an explicit ``None``. An ingredient the KB has
    never heard of is the single most repeated lookup in the system — every
    scan of every product containing it asks again — and without negative
    caching each one is a query that finds nothing.
    """
    if not keys:
        return {}

    token = version_token(session)
    out: dict[str, Any] = {}
    missing: list[str] = []

    for key in keys:
        cached = cache.get(f"{prefix}:{token}:{key}")
        if cached is None:
            missing.append(key)
        elif cached != "__absent__":
            out[key] = cached

    if missing:
        loaded = loader(missing)
        for key in missing:
            value = loaded.get(key)
            cache.put(f"{prefix}:{token}:{key}", "__absent__" if value is None else value)
            if value is not None:
                out[key] = value

    return out


def cached_dossiers(
    session: Session, chemical_ids: list[str], loader: Callable[[list[str]], dict[str, Any]]
) -> dict[str, Any]:
    return _get_or_load(_dossier_cache, session, "dos", chemical_ids, loader)


def cached_assertions(
    session: Session, chemical_ids: list[str], loader: Callable[[list[str]], dict[str, Any]]
) -> dict[str, Any]:
    return _get_or_load(_assertion_cache, session, "asr", chemical_ids, loader)


def cached_synonyms(
    session: Session, hashes: list[str], loader: Callable[[list[str]], dict[str, Any]]
) -> dict[str, Any]:
    return _get_or_load(_synonym_cache, session, "syn", hashes, loader)


def invalidate() -> None:
    """Drop everything. Called after an ETL write so the same process does not
    keep serving what it just replaced, without waiting for the token TTL."""
    global _version_token, _version_checked_at
    _dossier_cache.clear()
    _assertion_cache.clear()
    _synonym_cache.clear()
    with _version_lock:
        _version_token = None
        _version_checked_at = 0.0
    logger.info("chemical cache invalidated")


def stats() -> dict[str, Any]:
    return {
        "version_token": _version_token,
        "dossiers": _dossier_cache.stats(),
        "assertions": _assertion_cache.stats(),
        "synonyms": _synonym_cache.stats(),
    }
