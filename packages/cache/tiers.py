"""L1 in-process LRU, L2 MySQL exact, L3 MySQL semantic.

arch.md 7.1 puts L2 in Redis; there is no Redis on this host, so L2 is a MySQL
table. The tier boundary that matters is preserved: L1 is per-replica and
microseconds, L2/L3 are shared and milliseconds.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.orm import Session

from packages.cache.keys import CacheKey
from packages.config import get_settings
from packages.domain.enums import CacheTier, SafetyClass
from packages.domain.models import AnswerPayload
from packages.storage.repositories.cache import CacheRepository

logger = logging.getLogger(__name__)


@dataclass
class CacheLookup:
    tier: CacheTier = CacheTier.MISS
    payload: Optional[AnswerPayload] = None
    cache_key: Optional[str] = None
    score: float = 0.0
    reason: str = ""

    @property
    def hit(self) -> bool:
        return self.tier is not CacheTier.MISS and self.payload is not None


class LruCache:
    """Bounded, TTL'd, thread-safe. One instance per process."""

    def __init__(self, max_entries: int = 1024, ttl_seconds: int = 60) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at < time.time():
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        with self._lock:
            self._data[key] = (time.time() + ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        with self._lock:
            doomed = [k for k in self._data if k.startswith(prefix)]
            for k in doomed:
                del self._data[k]
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._data),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


_l1: Optional[LruCache] = None
_l1_lock = threading.Lock()


def get_l1() -> LruCache:
    global _l1
    if _l1 is None:
        with _l1_lock:
            if _l1 is None:
                s = get_settings().cache
                _l1 = LruCache(max_entries=s.l1_max_entries, ttl_seconds=s.l1_ttl_seconds)
    return _l1


@dataclass
class CacheStats:
    l1: dict[str, Any] = field(default_factory=dict)
    l2: dict[str, Any] = field(default_factory=dict)


class CacheService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.repo = CacheRepository(session)
        self.l1 = get_l1()

    # -- read -------------------------------------------------------------

    def probe(
        self,
        key: CacheKey,
        *,
        embedding: Optional[list[float]] = None,
        route_label: Optional[str] = None,
        locale: str = "en",
        safety_class: SafetyClass = SafetyClass.INFORMATIONAL,
    ) -> CacheLookup:
        if not self.settings.cache.enabled:
            return CacheLookup(reason="cache disabled")

        # L1
        cached = self.l1.get(key.digest)
        if cached is not None:
            return CacheLookup(
                tier=CacheTier.L1, payload=cached, cache_key=key.digest, score=1.0, reason="exact"
            )

        # L2
        row = self.repo.get(key.digest)
        if row and not row["is_negative"]:
            payload = self._deserialise(row["payload"])
            if payload is not None:
                self.repo.touch(key.digest)
                self.l1.put(key.digest, payload)
                return CacheLookup(
                    tier=CacheTier.L2, payload=payload, cache_key=key.digest, score=1.0, reason="exact"
                )
        if row and row["is_negative"]:
            # A cached upstream failure: still a miss for answering purposes,
            # but it stops us stampeding the failing source again (arch.md 7.3).
            return CacheLookup(reason="negative cache")

        # L3 — a candidate, never an answer on its own.
        if embedding:
            return self._probe_semantic(
                key, embedding, route_label=route_label, locale=locale, safety_class=safety_class
            )

        return CacheLookup(reason="miss")

    def _probe_semantic(
        self,
        key: CacheKey,
        embedding: list[float],
        *,
        route_label: Optional[str],
        locale: str,
        safety_class: SafetyClass,
    ) -> CacheLookup:
        threshold = self.settings.cache.semantic_threshold
        candidates = self.repo.semantic_candidates(
            embedding=embedding,
            scope=key.scope,
            locale=locale,
            route_label=route_label,
            top_k=3,
        )
        if not candidates:
            return CacheLookup(reason="miss")

        best = candidates[0]
        if best.score < threshold:
            return CacheLookup(reason=f"below tau_cache ({best.score:.3f} < {threshold})")

        # arch.md 7.3 (b): a compatible fingerprint, or the answer was
        # personalised against a context this user no longer has.
        if best.payload.get("context_fingerprint") != key.fingerprint:
            return CacheLookup(reason="fingerprint mismatch")

        # arch.md 7.3 (c): medical_sensitive needs an entailment check before a
        # paraphrase hit is trusted. Until that chain exists, refuse the hit
        # rather than serve an unverified medical answer.
        if safety_class is SafetyClass.MEDICAL_SENSITIVE:
            return CacheLookup(reason="medical_sensitive: L3 requires entailment check")

        payload = self._deserialise(best.payload.get("payload"))
        if payload is None:
            return CacheLookup(reason="undeserialisable payload")

        self.repo.touch(best.ref)
        return CacheLookup(
            tier=CacheTier.L3,
            payload=payload,
            cache_key=best.ref,
            score=best.score,
            reason="semantic",
        )

    # -- write ------------------------------------------------------------

    def store(
        self,
        key: CacheKey,
        payload: AnswerPayload,
        *,
        category: Optional[str] = None,
        route_label: Optional[str] = None,
        locale: str = "en",
        embedding: Optional[list[float]] = None,
        model_id: str = "",
        has_safety_flags: bool = False,
    ) -> bool:
        """Returns True if the entry was written."""
        if not self.settings.cache.enabled:
            return False

        # arch.md 7.3 "never cached".
        if has_safety_flags:
            return False
        if not payload.blocks:
            return False

        ttl = self.settings.cache.ttl_for(category)
        serialised = payload.model_copy(update={"route_debug": None}).model_dump_json()

        self.repo.put(
            cache_key=key.digest,
            payload=serialised,
            ttl_seconds=ttl,
            scope=key.scope,
            route_label=route_label,
            category=category,
            norm_question=key.normalised_question,
            context_fingerprint=key.fingerprint,
            prompt_version=self.settings.prompt_version,
            model_id=model_id,
            kb_version=self.settings.kb_version,
            locale=locale,
            embedding=embedding,
        )
        self.l1.put(key.digest, payload)
        return True

    def store_negative(self, key: CacheKey, reason: str, locale: str = "en") -> None:
        """Upstream failures are cached briefly and never persisted as answers."""
        self.repo.put(
            cache_key=key.digest,
            payload=reason[:2000],
            ttl_seconds=self.settings.cache.negative_ttl_seconds,
            scope=key.scope,
            norm_question=key.normalised_question,
            locale=locale,
            is_negative=True,
        )

    # -- invalidation (arch.md 7.3 event-driven bust) ---------------------

    def invalidate_user(self, user_id: str) -> int:
        """Called on profile edit, health sync, new lab report, memory delete."""
        self.l1.clear()  # L1 has no scope index; a full clear is cheap and correct
        return self.repo.invalidate_scope(user_id)

    def invalidate_versions(self) -> int:
        """Called after a prompt/model/KB version bump."""
        self.l1.clear()
        return self.repo.invalidate_by_version(
            prompt_version=self.settings.prompt_version,
            kb_version=self.settings.kb_version,
        )

    def invalidate_category(self, category: str) -> int:
        self.l1.clear()
        return self.repo.invalidate_by_category(category)

    def maintenance(self) -> dict[str, int]:
        return {
            "expired": self.repo.purge_expired(),
            "pruned": self.repo.prune_cold(),
        }

    def stats(self) -> CacheStats:
        return CacheStats(l1=self.l1.stats(), l2=self.repo.stats())

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _deserialise(raw: Any) -> Optional[AnswerPayload]:
        if raw is None:
            return None
        if isinstance(raw, AnswerPayload):
            return raw
        try:
            return AnswerPayload.model_validate_json(raw)
        except Exception as exc:
            # A schema change can strand old rows; treat as a miss, not a 500.
            logger.warning("cache payload failed validation, treating as miss: %s", exc)
            return None
