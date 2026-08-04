"""DB-backed job queue.

No broker on this host, so the queue is a table. Claiming uses the portable
two-step: a bounded UPDATE stamps a random token onto the rows it wins, then a
SELECT reads back exactly those rows. That is atomic on any MySQL/MariaDB and
does not need SELECT ... FOR UPDATE SKIP LOCKED (MariaDB < 10.6 lacks it).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.domain.enums import JobStatus, JobType
from packages.domain.models import JobRecord


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _to_record(row: dict[str, Any]) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        job_type=JobType(row["job_type"]),
        status=JobStatus(row["status"]),
        user_id=row.get("user_id"),
        payload=_loads(row.get("payload")) or {},
        result=_loads(row.get("result")),
        error=row.get("error"),
        attempts=int(row.get("attempts") or 0),
        max_attempts=int(row.get("max_attempts") or 3),
        priority=int(row.get("priority") or 100),
        available_at=row.get("available_at") or datetime.now(timezone.utc),
        lease_expires_at=row.get("lease_expires_at"),
        created_at=row.get("created_at") or datetime.now(timezone.utc),
        updated_at=row.get("updated_at") or datetime.now(timezone.utc),
    )


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(
        self,
        job_type: JobType,
        payload: dict[str, Any],
        user_id: Optional[str] = None,
        priority: int = 100,
        max_attempts: int = 3,
        delay_seconds: int = 0,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """Returns the job id. With an idempotency key, a repeat enqueue
        returns the existing job instead of creating a duplicate."""
        if idempotency_key:
            existing = self.session.execute(
                text("SELECT job_id FROM job WHERE idempotency_key = :k"),
                {"k": idempotency_key},
            ).first()
            if existing:
                return existing[0]

        job_id = uuid.uuid4().hex
        available_at = datetime.now(timezone.utc) + timedelta(seconds=max(0, delay_seconds))

        self.session.execute(
            text(
                """
                INSERT INTO job
                    (job_id, job_type, status, user_id, idempotency_key, payload,
                     max_attempts, priority, available_at)
                VALUES
                    (:jid, :jtype, 'queued', :uid, :ikey, :payload,
                     :maxatt, :prio, :avail)
                """
            ),
            {
                "jid": job_id,
                "jtype": str(job_type.value),
                "uid": user_id,
                "ikey": idempotency_key,
                "payload": json.dumps(payload, default=str),
                "maxatt": max_attempts,
                "prio": priority,
                "avail": available_at.replace(tzinfo=None),
            },
        )
        return job_id

    def get(self, job_id: str, user_id: Optional[str] = None) -> Optional[JobRecord]:
        """`user_id` scopes the read so one user cannot poll another's job."""
        clause = "AND (user_id = :uid OR user_id IS NULL)" if user_id else ""
        params: dict[str, Any] = {"jid": job_id}
        if user_id:
            params["uid"] = user_id

        row = self.session.execute(
            text(f"SELECT * FROM job WHERE job_id = :jid {clause}"), params
        ).mappings().first()
        return _to_record(dict(row)) if row else None

    def claim(self, worker_id: str, batch_size: int = 5, lease_seconds: int = 600) -> list[JobRecord]:
        token = uuid.uuid4().hex
        lease_until = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)

        claimed = self.session.execute(
            text(
                """
                UPDATE job
                SET status = 'running',
                    claim_token = :token,
                    worker_id = :wid,
                    attempts = attempts + 1,
                    lease_expires_at = :lease
                WHERE status = 'queued'
                  AND available_at <= UTC_TIMESTAMP(3)
                  AND attempts < max_attempts
                ORDER BY priority ASC, created_at ASC
                LIMIT :lim
                """
            ),
            {
                "token": token,
                "wid": worker_id[:64],
                "lease": lease_until.replace(tzinfo=None),
                "lim": max(1, batch_size),
            },
        ).rowcount

        if not claimed:
            return []

        rows = self.session.execute(
            text("SELECT * FROM job WHERE claim_token = :token"), {"token": token}
        ).mappings().all()
        return [_to_record(dict(r)) for r in rows]

    def succeed(self, job_id: str, result: dict[str, Any]) -> None:
        self.session.execute(
            text(
                "UPDATE job SET status = 'succeeded', result = :res, error = NULL, "
                "claim_token = NULL, lease_expires_at = NULL, "
                "finished_at = UTC_TIMESTAMP(3) WHERE job_id = :jid"
            ),
            {"jid": job_id, "res": json.dumps(result, default=str)},
        )

    def fail(self, job_id: str, error: str, retry_in_seconds: Optional[int] = None) -> None:
        """Requeue if attempts remain and a backoff was given; else mark failed."""
        if retry_in_seconds is not None:
            available_at = datetime.now(timezone.utc) + timedelta(seconds=retry_in_seconds)
            self.session.execute(
                text(
                    """
                    UPDATE job
                    SET status = CASE WHEN attempts < max_attempts THEN 'queued' ELSE 'failed' END,
                        error = :err,
                        claim_token = NULL,
                        lease_expires_at = NULL,
                        available_at = :avail,
                        finished_at = CASE WHEN attempts < max_attempts THEN NULL ELSE UTC_TIMESTAMP(3) END
                    WHERE job_id = :jid
                    """
                ),
                {"jid": job_id, "err": error[:4000], "avail": available_at.replace(tzinfo=None)},
            )
            return

        self.session.execute(
            text(
                "UPDATE job SET status = 'failed', error = :err, claim_token = NULL, "
                "lease_expires_at = NULL, finished_at = UTC_TIMESTAMP(3) WHERE job_id = :jid"
            ),
            {"jid": job_id, "err": error[:4000]},
        )

    def reap_expired_leases(self) -> int:
        """A worker killed mid-job (cPanel does that) leaves rows 'running'
        forever. Anything past its lease goes back on the queue."""
        result = self.session.execute(
            text(
                """
                UPDATE job
                SET status = CASE WHEN attempts < max_attempts THEN 'queued' ELSE 'failed' END,
                    error = COALESCE(error, 'lease expired'),
                    claim_token = NULL,
                    lease_expires_at = NULL
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < UTC_TIMESTAMP(3)
                """
            )
        )
        return result.rowcount or 0

    def list_for_user(self, user_id: str, limit: int = 20) -> list[JobRecord]:
        rows = self.session.execute(
            text(
                "SELECT * FROM job WHERE user_id = :uid ORDER BY created_at DESC LIMIT :lim"
            ),
            {"uid": user_id, "lim": max(1, min(limit, 100))},
        ).mappings().all()
        return [_to_record(dict(r)) for r in rows]

    def purge_old(self, ttl_seconds: int) -> int:
        result = self.session.execute(
            text(
                "DELETE FROM job WHERE status IN ('succeeded','cancelled') "
                "AND finished_at < (UTC_TIMESTAMP(3) - INTERVAL :s SECOND) LIMIT 1000"
            ),
            {"s": ttl_seconds},
        )
        return result.rowcount or 0

    def queue_depth(self) -> dict[str, int]:
        rows = self.session.execute(
            text("SELECT status, COUNT(*) AS n FROM job GROUP BY status")
        ).all()
        return {status: int(n) for status, n in rows}
