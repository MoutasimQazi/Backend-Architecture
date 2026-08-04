"""Worker package.

Consumes the MySQL-backed job queue. There is no Redis or Celery on this host,
so `job` is the queue and a MySQL advisory lock provides mutual exclusion
between overlapping cron ticks. See `apps/worker/worker.py`.
"""

__all__ = ["worker"]
