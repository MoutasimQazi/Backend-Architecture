"""Conversation log, rolling summary, structured memory, and traces."""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.domain.models import NodeTiming


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def append_turn(
        self,
        turn_id: str,
        session_id: str,
        user_id: str,
        role: str,
        content: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        route_label: Optional[str] = None,
        route_stage: Optional[str] = None,
        route_confidence: Optional[float] = None,
        cache_tier: Optional[str] = None,
        latency_ms: Optional[float] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO conversation_turn
                    (turn_id, session_id, user_id, role, content, payload,
                     route_label, route_stage, route_confidence, cache_tier,
                     latency_ms, tokens_in, tokens_out, cost_usd)
                VALUES
                    (:tid, :sid, :uid, :role, :content, :payload,
                     :route, :stage, :conf, :tier,
                     :lat, :tin, :tout, :cost)
                ON DUPLICATE KEY UPDATE
                    content = VALUES(content),
                    payload = VALUES(payload)
                """
            ),
            {
                "tid": turn_id,
                "sid": session_id,
                "uid": user_id,
                "role": role,
                "content": content,
                "payload": json.dumps(payload, default=str) if payload is not None else None,
                "route": route_label,
                "stage": route_stage,
                "conf": route_confidence,
                "tier": cache_tier,
                "lat": int(latency_ms) if latency_ms is not None else None,
                "tin": tokens_in,
                "tout": tokens_out,
                "cost": cost_usd,
            },
        )

    def recent_turns(self, session_id: str, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Short-term memory. Scoped by user as well as session so a guessed
        session id leaks nothing."""
        rows = self.session.execute(
            text(
                "SELECT role, content, created_at FROM conversation_turn "
                "WHERE session_id = :sid AND user_id = :uid "
                "ORDER BY created_at DESC LIMIT :lim"
            ),
            {"sid": session_id, "uid": user_id, "lim": max(1, min(limit, 50))},
        ).mappings().all()
        return [dict(r) for r in reversed(rows)]

    def get_summary(self, session_id: str, user_id: str) -> Optional[str]:
        row = self.session.execute(
            text(
                "SELECT summary FROM conversation_summary WHERE session_id = :sid AND user_id = :uid"
            ),
            {"sid": session_id, "uid": user_id},
        ).first()
        return row[0] if row else None

    def put_summary(self, session_id: str, user_id: str, summary: str, turn_count: int = 0) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO conversation_summary (session_id, user_id, summary, turn_count)
                VALUES (:sid, :uid, :summary, :n)
                ON DUPLICATE KEY UPDATE summary = VALUES(summary), turn_count = VALUES(turn_count)
                """
            ),
            {"sid": session_id, "uid": user_id, "summary": summary, "n": turn_count},
        )

    # -- structured memory (arch.md 12) -----------------------------------

    def remember(
        self,
        user_id: str,
        kind: str,
        value: str,
        confidence: float = 1.0,
        source_turn_id: Optional[str] = None,
    ) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO user_memory (user_id, kind, value, confidence, source_turn_id, active)
                VALUES (:uid, :kind, :val, :conf, :tid, 1)
                ON DUPLICATE KEY UPDATE
                    confidence = GREATEST(confidence, VALUES(confidence)),
                    active = 1,
                    source_turn_id = COALESCE(VALUES(source_turn_id), source_turn_id)
                """
            ),
            {
                "uid": user_id,
                "kind": kind,
                "val": value[:512],
                "conf": confidence,
                "tid": source_turn_id,
            },
        )

    def recall(self, user_id: str, kind: Optional[str] = None) -> list[dict[str, Any]]:
        clause = "AND kind = :kind" if kind else ""
        params: dict[str, Any] = {"uid": user_id}
        if kind:
            params["kind"] = kind
        rows = self.session.execute(
            text(
                f"SELECT id, kind, value, confidence, updated_at FROM user_memory "
                f"WHERE user_id = :uid AND active = 1 {clause} ORDER BY kind, updated_at DESC"
            ),
            params,
        ).mappings().all()
        return [dict(r) for r in rows]

    def forget(self, user_id: str, memory_id: int) -> bool:
        """User-initiated deletion. The caller busts that user's caches."""
        result = self.session.execute(
            text("UPDATE user_memory SET active = 0 WHERE id = :mid AND user_id = :uid"),
            {"mid": memory_id, "uid": user_id},
        )
        return (result.rowcount or 0) > 0


class TraceRepository:
    """One trace per turn (arch.md 14)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record(self, turn_id: str, timings: list[NodeTiming], attributes: Optional[dict[str, Any]] = None) -> None:
        for timing in timings:
            self.session.execute(
                text(
                    "INSERT INTO trace_span (turn_id, node, started_at, duration_ms, ok, error, attributes) "
                    "VALUES (:tid, :node, :started, :dur, :ok, :err, :attrs)"
                ),
                {
                    "tid": turn_id,
                    "node": timing.node,
                    "started": timing.started_at.replace(tzinfo=None),
                    "dur": timing.duration_ms,
                    "ok": 1 if timing.ok else 0,
                    "err": timing.error,
                    "attrs": json.dumps(attributes, default=str) if attributes else None,
                },
            )

    def node_latency(self, node: str, hours: int = 24) -> dict[str, Any]:
        row = self.session.execute(
            text(
                "SELECT COUNT(*) n, AVG(duration_ms) avg_ms, MAX(duration_ms) max_ms, "
                "       SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) errors "
                "FROM trace_span WHERE node = :node AND started_at >= (NOW() - INTERVAL :h HOUR)"
            ),
            {"node": node, "h": hours},
        ).mappings().first()
        return dict(row) if row else {}
