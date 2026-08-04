"""Enqueue a job from the shell.

    python scripts/enqueue_job.py context_aggregate '{"user_id": "u_123"}'
    python scripts/enqueue_job.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.domain.enums import JobType  # noqa: E402
from packages.jobs import handlers as _handlers  # noqa: E402,F401  (registers handlers)
from packages.jobs.registry import registered_types  # noqa: E402
from packages.storage.db import session_scope  # noqa: E402
from packages.storage.repositories.jobs import JobRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Enqueue a job")
    parser.add_argument("job_type", nargs="?", help="job type, e.g. context_aggregate")
    parser.add_argument("payload", nargs="?", default="{}", help="JSON payload")
    parser.add_argument("--user", dest="user_id", default=None)
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--list", action="store_true", help="list registered job types")
    args = parser.parse_args()

    if args.list or not args.job_type:
        print("registered job types:")
        for name in registered_types():
            print(f"  {name}")
        return 0

    try:
        job_type = JobType(args.job_type)
    except ValueError:
        print(f"unknown job type '{args.job_type}'. Known: {', '.join(registered_types())}")
        return 2

    try:
        payload = json.loads(args.payload)
    except ValueError as exc:
        print(f"invalid JSON payload: {exc}")
        return 2

    with session_scope() as session:
        job_id = JobRepository(session).enqueue(
            job_type=job_type,
            payload=payload,
            user_id=args.user_id or payload.get("user_id"),
            priority=args.priority,
        )

    print(f"enqueued {job_type.value}: {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
