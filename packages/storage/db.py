"""Engine, sessions, and the MySQL-native primitives that stand in for Redis."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from packages.config import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_engine(
            s.db.url,
            pool_size=s.db.pool_size,
            max_overflow=s.db.max_overflow,
            # Shared hosts drop idle connections without warning; pre_ping turns
            # a hard 2006 "server has gone away" into a transparent reconnect.
            pool_pre_ping=True,
            pool_recycle=s.db.pool_recycle,
            pool_timeout=s.db.pool_timeout,
            echo=s.db.echo,
            future=True,
            connect_args={"connect_timeout": 10},
        )
        logger.info("db engine created: %s", s.db.url_safe)
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), expire_on_commit=False, future=True, class_=Session
        )
    return _session_factory


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on clean exit, rolls back on exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError as exc:
        logger.error("db ping failed: %s", exc)
        return False


def server_info() -> dict[str, Any]:
    with get_engine().connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).scalar_one()
        db_name = conn.execute(text("SELECT DATABASE()")).scalar_one()
    return {"version": version, "database": db_name}


# --------------------------------------------------------------------------
# Advisory locks. MySQL has these natively, so the worker gets mutual
# exclusion without Redis. Locks are held per *connection*, so this must run
# on a dedicated connection, not a pooled session that may be handed back.
# --------------------------------------------------------------------------


@contextlib.contextmanager
def advisory_lock(name: str, timeout_seconds: int = 0) -> Iterator[bool]:
    """Yield True if the named lock was acquired, False otherwise.

    MySQL truncates lock names above 64 chars, so callers should keep them short.
    """
    conn = get_engine().connect()
    acquired = False
    try:
        result = conn.execute(
            text("SELECT GET_LOCK(:name, :timeout)"),
            {"name": name[:64], "timeout": timeout_seconds},
        ).scalar()
        acquired = result == 1
        yield acquired
    finally:
        if acquired:
            with contextlib.suppress(Exception):
                conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": name[:64]})
        conn.close()
