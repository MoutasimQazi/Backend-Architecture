"""Job worker.

Run it two ways, both of which cPanel supports:

    python -m apps.worker.worker            # long-running, if you have a
                                            # persistent process slot
    python -m apps.worker.worker --once     # one pass, for a cron entry:
                                            # * * * * * cd /path && python -m apps.worker.worker --once

The advisory lock means overlapping cron ticks are harmless — a second process
that finds the lock held exits instead of double-processing.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import time
import traceback
from typing import Optional

# Allow `python apps/worker/worker.py` as well as `-m`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from packages.config import get_settings  # noqa: E402
from packages.jobs import handlers as _handlers  # noqa: E402,F401  (registers handlers)
from packages.jobs.registry import JobContext, get_handler, registered_types  # noqa: E402
from packages.storage.db import advisory_lock, session_scope  # noqa: E402
from packages.storage.repositories.jobs import JobRepository  # noqa: E402

logger = logging.getLogger("worker")

WORKER_LOCK = "fitness_worker"
MAINTENANCE_LOCK = "fitness_maint"

_shutdown = False


def _handle_signal(signum, _frame) -> None:
    global _shutdown
    logger.info("signal %s received; finishing current batch then exiting", signum)
    _shutdown = True


def _worker_id() -> str:
    return f"{socket.gethostname()[:40]}:{os.getpid()}"


def _backoff_seconds(attempt: int) -> int:
    """Exponential with a ceiling: 30s, 120s, 480s, capped at 15 min."""
    return min(30 * (4 ** max(0, attempt - 1)), 900)


def process_batch(worker_id: str) -> int:
    """Claim and run one batch. Returns how many jobs were processed."""
    settings = get_settings()
    processed = 0

    with session_scope() as session:
        repo = JobRepository(session)
        jobs = repo.claim(
            worker_id=worker_id,
            batch_size=settings.jobs.batch_size,
            lease_seconds=settings.jobs.lease_seconds,
        )

    if not jobs:
        return 0

    for job in jobs:
        handler_fn = get_handler(job.job_type)
        if handler_fn is None:
            logger.error("no handler for job type %s (job %s)", job.job_type, job.job_id)
            with session_scope() as session:
                JobRepository(session).fail(job.job_id, f"no handler for {job.job_type}")
            continue

        started = time.time()
        try:
            # Each job gets its own transaction: one failure cannot roll back
            # the work of the job before it in the batch.
            with session_scope() as session:
                ctx = JobContext(
                    job_id=job.job_id,
                    session=session,
                    payload=job.payload,
                    user_id=job.user_id,
                    attempt=job.attempts,
                )
                result = handler_fn(ctx)
                JobRepository(session).succeed(job.job_id, result or {})

            processed += 1
            logger.info(
                "job %s (%s) ok in %.0fms", job.job_id, job.job_type, (time.time() - started) * 1000
            )

        except Exception as exc:
            logger.exception("job %s (%s) failed", job.job_id, job.job_type)
            detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
            retry_in: Optional[int] = (
                _backoff_seconds(job.attempts) if job.attempts < job.max_attempts else None
            )
            try:
                with session_scope() as session:
                    JobRepository(session).fail(job.job_id, detail, retry_in_seconds=retry_in)
            except Exception:
                logger.exception("could not record failure for job %s", job.job_id)

    return processed


def run_maintenance() -> dict[str, int]:
    """Sweeps whose cost is wasted if two workers do them at once."""
    with advisory_lock(MAINTENANCE_LOCK, timeout_seconds=0) as got_lock:
        if not got_lock:
            return {}
        with session_scope() as session:
            stats = _handlers.run_maintenance(session)
            JobRepository(session).purge_old(get_settings().jobs.result_ttl_seconds)
    if any(stats.values()):
        logger.info("maintenance: %s", stats)
    return stats


def run_once() -> int:
    """One pass: reap orphaned leases, process a batch, sweep. Cron-friendly."""
    with advisory_lock(WORKER_LOCK, timeout_seconds=0) as got_lock:
        if not got_lock:
            logger.info("another worker holds the lock; exiting")
            return 0
        with session_scope() as session:
            reaped = JobRepository(session).reap_expired_leases()
        if reaped:
            logger.warning("requeued %d job(s) with expired leases", reaped)
        processed = process_batch(_worker_id())
    run_maintenance()
    return processed


def run_forever() -> None:
    settings = get_settings()
    worker_id = _worker_id()
    interval = settings.jobs.poll_interval_seconds
    deadline = (
        time.time() + settings.jobs.max_runtime_seconds
        if settings.jobs.max_runtime_seconds
        else None
    )
    last_maintenance = 0.0

    logger.info(
        "worker %s started; handlers: %s; poll %.1fs",
        worker_id,
        ", ".join(registered_types()),
        interval,
    )

    idle_rounds = 0
    while not _shutdown:
        if deadline and time.time() > deadline:
            logger.info("max runtime reached; exiting for a clean restart")
            break
        try:
            with advisory_lock(WORKER_LOCK, timeout_seconds=0) as got_lock:
                if not got_lock:
                    time.sleep(interval)
                    continue
                processed = process_batch(worker_id)

            idle_rounds = 0 if processed else idle_rounds + 1

            # Sweep every ~5 minutes, not every tick.
            if time.time() - last_maintenance > 300:
                run_maintenance()
                last_maintenance = time.time()

        except Exception:
            logger.exception("worker loop error; continuing")
            idle_rounds += 1

        # Back off when the queue is empty so an idle worker is not a
        # once-a-second query against a shared MySQL server.
        sleep_for = interval if idle_rounds < 5 else min(interval * 4, 30)
        time.sleep(sleep_for)

    logger.info("worker stopped")


def main() -> int:
    parser = argparse.ArgumentParser(description="Health & Product Assistant job worker")
    parser.add_argument("--once", action="store_true", help="process one batch and exit (for cron)")
    parser.add_argument("--maintenance-only", action="store_true", help="run sweeps and exit")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    problems = settings.validate()
    if any("DB_" in p for p in problems):
        logger.error("configuration problem: %s", "; ".join(problems))
        return 2

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.maintenance_only:
        print(run_maintenance())
        return 0

    if args.once:
        count = run_once()
        logger.info("processed %d job(s)", count)
        return 0

    run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
