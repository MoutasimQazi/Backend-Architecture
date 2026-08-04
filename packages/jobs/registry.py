"""Handler registry.

A job type maps to one callable. Registration is by decorator so adding a job
means adding a function, not editing a dispatch table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from packages.domain.enums import JobType

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    """What a handler is given. The session is inside the worker's transaction."""

    job_id: str
    session: Session
    payload: dict[str, Any]
    user_id: Optional[str] = None
    attempt: int = 1

    def require(self, key: str) -> Any:
        if key not in self.payload:
            raise ValueError(f"job {self.job_id}: payload missing required key '{key}'")
        return self.payload[key]


JobHandler = Callable[[JobContext], dict[str, Any]]

_handlers: dict[str, JobHandler] = {}


def handler(job_type: JobType) -> Callable[[JobHandler], JobHandler]:
    def decorator(func: JobHandler) -> JobHandler:
        key = str(job_type.value)
        if key in _handlers:
            logger.warning("job handler for %s re-registered by %s", key, func.__name__)
        _handlers[key] = func
        return func

    return decorator


def get_handler(job_type: JobType | str) -> Optional[JobHandler]:
    key = str(job_type.value) if isinstance(job_type, JobType) else str(job_type)
    return _handlers.get(key)


def registered_types() -> list[str]:
    return sorted(_handlers)
