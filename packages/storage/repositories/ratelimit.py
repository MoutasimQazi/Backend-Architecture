"""Fixed-window rate limiting on MySQL.

Not as sharp as a Redis token bucket — a burst can straddle a window boundary —
but it is shared across replicas, needs no extra service, and the failure mode
is "slightly generous", which is the right way to be wrong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class LimitResult:
    allowed: bool
    hits: int
    limit: int
    window: str
    retry_after_seconds: int = 0


class RateLimitRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _hit(self, subject: str, window_key: str, expires_at: datetime, limit: int) -> LimitResult:
        # One statement, so two concurrent requests cannot both read "n-1".
        self.session.execute(
            text(
                """
                INSERT INTO rate_limit_bucket (subject, window_key, hits, expires_at)
                VALUES (:subj, :win, 1, :exp)
                ON DUPLICATE KEY UPDATE hits = hits + 1
                """
            ),
            {"subj": subject[:128], "win": window_key, "exp": expires_at.replace(tzinfo=None)},
        )
        row = self.session.execute(
            text("SELECT hits FROM rate_limit_bucket WHERE subject = :subj AND window_key = :win"),
            {"subj": subject[:128], "win": window_key},
        ).first()
        hits = int(row[0]) if row else 1
        retry_after = 0
        if hits > limit:
            retry_after = max(1, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
        return LimitResult(
            allowed=hits <= limit,
            hits=hits,
            limit=limit,
            window=window_key,
            retry_after_seconds=retry_after,
        )

    def check_minute(self, subject: str, limit: int) -> LimitResult:
        now = datetime.now(timezone.utc)
        window_key = now.strftime("%Y-%m-%dT%H:%M")
        expires_at = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return self._hit(subject, window_key, expires_at, limit)

    def check_day(self, subject: str, limit: int) -> LimitResult:
        now = datetime.now(timezone.utc)
        window_key = now.strftime("%Y-%m-%d")
        expires_at = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return self._hit(subject, window_key, expires_at, limit)

    def purge_expired(self, limit: int = 5000) -> int:
        result = self.session.execute(
            text("DELETE FROM rate_limit_bucket WHERE expires_at <= UTC_TIMESTAMP(3) LIMIT :lim"),
            {"lim": limit},
        )
        return result.rowcount or 0
