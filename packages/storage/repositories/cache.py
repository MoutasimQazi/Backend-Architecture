"""L2 exact and L3 semantic cache rows. Redis's job, done by MySQL."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.storage.vectors import ScoredId, cosine, pack, unpack


class CacheRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, cache_key: str) -> Optional[dict[str, Any]]:
        row = self.session.execute(
            text(
                "SELECT cache_key, scope, route_label, category, payload, context_fingerprint, "
                "       prompt_version, model_id, kb_version, locale, is_negative, hit_count, expires_at "
                "FROM cache_entry WHERE cache_key = :k AND expires_at > UTC_TIMESTAMP(3)"
            ),
            {"k": cache_key},
        ).mappings().first()
        return dict(row) if row else None

    def touch(self, cache_key: str) -> None:
        self.session.execute(
            text(
                "UPDATE cache_entry SET hit_count = hit_count + 1, "
                "last_hit_at = UTC_TIMESTAMP(3) WHERE cache_key = :k"
            ),
            {"k": cache_key},
        )

    def put(
        self,
        cache_key: str,
        payload: str,
        ttl_seconds: int,
        scope: str = "global",
        route_label: Optional[str] = None,
        category: Optional[str] = None,
        norm_question: Optional[str] = None,
        context_fingerprint: Optional[str] = None,
        prompt_version: Optional[str] = None,
        model_id: Optional[str] = None,
        kb_version: Optional[str] = None,
        locale: str = "en",
        embedding: Optional[list[float]] = None,
        is_negative: bool = False,
    ) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
        self.session.execute(
            text(
                """
                INSERT INTO cache_entry
                    (cache_key, scope, route_label, category, norm_question, payload,
                     context_fingerprint, prompt_version, model_id, kb_version, locale,
                     embedding, is_negative, expires_at)
                VALUES
                    (:k, :scope, :route, :cat, :nq, :payload,
                     :fp, :pv, :model, :kbv, :locale,
                     :emb, :neg, :exp)
                ON DUPLICATE KEY UPDATE
                    payload = VALUES(payload),
                    context_fingerprint = VALUES(context_fingerprint),
                    embedding = VALUES(embedding),
                    is_negative = VALUES(is_negative),
                    expires_at = VALUES(expires_at)
                """
            ),
            {
                "k": cache_key,
                "scope": scope,
                "route": route_label,
                "cat": category,
                "nq": norm_question,
                "payload": payload,
                "fp": context_fingerprint,
                "pv": prompt_version,
                "model": model_id,
                "kbv": kb_version,
                "locale": locale,
                "emb": pack(embedding) if embedding else None,
                "neg": 1 if is_negative else 0,
                "exp": expires_at.replace(tzinfo=None),
            },
        )

    def semantic_candidates(
        self,
        embedding: list[float],
        scope: str,
        locale: str = "en",
        route_label: Optional[str] = None,
        top_k: int = 5,
        limit_rows: int = 500,
    ) -> list[ScoredId]:
        """L3 probe.

        Scanning the live rows for one scope and scoring in Python is fast at
        this corpus size; the row cap keeps a pathological scope bounded. The
        caller still has to verify fingerprint and threshold (arch.md 7.3) — a
        hit here is a candidate, not an answer.
        """
        clause = "AND route_label = :route" if route_label else ""
        params: dict[str, Any] = {
            "scope": scope,
            "locale": locale,
            "lim": max(1, min(limit_rows, 5000)),
        }
        if route_label:
            params["route"] = route_label

        rows = self.session.execute(
            text(
                f"""
                SELECT cache_key, embedding, context_fingerprint, category, payload
                FROM cache_entry
                WHERE scope = :scope AND locale = :locale AND is_negative = 0
                  AND embedding IS NOT NULL AND expires_at > UTC_TIMESTAMP(3) {clause}
                ORDER BY last_hit_at DESC, created_at DESC
                LIMIT :lim
                """
            ),
            params,
        ).mappings().all()

        scored: list[ScoredId] = []
        for row in rows:
            vector = unpack(row["embedding"])
            if not vector:
                continue
            score = cosine(embedding, vector)
            scored.append(
                ScoredId(
                    ref=row["cache_key"],
                    score=score,
                    payload={
                        "context_fingerprint": row["context_fingerprint"],
                        "category": row["category"],
                        "payload": row["payload"],
                    },
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    # -- invalidation (arch.md 7.3) ---------------------------------------

    def invalidate_scope(self, scope: str) -> int:
        result = self.session.execute(
            text("DELETE FROM cache_entry WHERE scope = :scope"), {"scope": scope}
        )
        return result.rowcount or 0

    def invalidate_by_category(self, category: str) -> int:
        result = self.session.execute(
            text("DELETE FROM cache_entry WHERE category = :cat"), {"cat": category}
        )
        return result.rowcount or 0

    def invalidate_by_version(
        self,
        prompt_version: Optional[str] = None,
        kb_version: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> int:
        """Drop everything that did *not* use the current versions."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if prompt_version:
            clauses.append("prompt_version <> :pv")
            params["pv"] = prompt_version
        if kb_version:
            clauses.append("kb_version <> :kbv")
            params["kbv"] = kb_version
        if model_id:
            clauses.append("model_id <> :model")
            params["model"] = model_id
        if not clauses:
            return 0
        result = self.session.execute(
            text(f"DELETE FROM cache_entry WHERE {' OR '.join(clauses)}"), params
        )
        return result.rowcount or 0

    def purge_expired(self, limit: int = 5000) -> int:
        result = self.session.execute(
            text("DELETE FROM cache_entry WHERE expires_at <= UTC_TIMESTAMP(3) LIMIT :lim"),
            {"lim": limit},
        )
        return result.rowcount or 0

    def prune_cold(self, older_than_days: int = 30, max_hits: int = 1, limit: int = 2000) -> int:
        """arch.md 7.3 eviction: nightly prune on last_hit_at and hit count."""
        result = self.session.execute(
            text(
                "DELETE FROM cache_entry "
                "WHERE hit_count <= :maxhits "
                "  AND COALESCE(last_hit_at, created_at) < (UTC_TIMESTAMP(3) - INTERVAL :days DAY) "
                "LIMIT :lim"
            ),
            {"maxhits": max_hits, "days": older_than_days, "lim": limit},
        )
        return result.rowcount or 0

    def stats(self) -> dict[str, Any]:
        row = self.session.execute(
            text(
                "SELECT COUNT(*) AS entries, "
                "       SUM(hit_count) AS hits, "
                "       SUM(CASE WHEN expires_at <= UTC_TIMESTAMP(3) THEN 1 ELSE 0 END) AS expired "
                "FROM cache_entry"
            )
        ).mappings().first()
        return {
            "entries": int(row["entries"] or 0),
            "hits": int(row["hits"] or 0),
            "expired": int(row["expired"] or 0),
        }
