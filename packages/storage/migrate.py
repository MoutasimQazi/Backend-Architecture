"""Migration runner.

Alembic assumes a shell and a working directory that cPanel does not reliably
give you. This applies plain .sql files in filename order and records each in
`schema_migration`, which is all the project needs and runs anywhere Python does.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from packages.storage.db import advisory_lock, get_engine

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
LOCK_NAME = "fitness_migrate"


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def discover() -> list[Migration]:
    if not MIGRATIONS_DIR.is_dir():
        return []
    found = [
        Migration(version=p.stem, path=p)
        for p in sorted(MIGRATIONS_DIR.glob("*.sql"))
    ]
    return found


def _split_statements(sql: str) -> list[str]:
    """Split on semicolons at statement level.

    The schema files contain no stored routines or string literals with
    semicolons, so a comment-stripped split is sufficient and avoids pulling in
    a SQL parser.
    """
    without_comments = re.sub(r"^\s*--.*$", "", sql, flags=re.MULTILINE)
    statements = [s.strip() for s in without_comments.split(";")]
    return [s for s in statements if s]


def applied_versions() -> set[str]:
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migration (
                  version    VARCHAR(64) NOT NULL,
                  applied_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
                  checksum   CHAR(64) NULL,
                  PRIMARY KEY (version)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
        )
        conn.commit()
        rows = conn.execute(text("SELECT version FROM schema_migration")).all()
    return {r[0] for r in rows}


def pending() -> list[Migration]:
    done = applied_versions()
    return [m for m in discover() if m.version not in done]


def run(dry_run: bool = False) -> list[str]:
    """Apply every pending migration. Returns the versions applied."""
    engine = get_engine()
    applied: list[str] = []

    # Two cPanel processes (a web request and a cron tick) can start together.
    with advisory_lock(LOCK_NAME, timeout_seconds=30) as got_lock:
        if not got_lock:
            raise RuntimeError("could not acquire migration lock; another run is in progress")

        todo = pending()
        if not todo:
            logger.info("no pending migrations")
            return []

        for migration in todo:
            statements = _split_statements(migration.sql)
            logger.info(
                "applying %s (%d statements)%s",
                migration.version,
                len(statements),
                " [dry-run]" if dry_run else "",
            )
            if dry_run:
                applied.append(migration.version)
                continue

            # DDL in MySQL is not transactional, so a failure mid-file leaves
            # partial state. Every statement is IF NOT EXISTS, so re-running
            # after a fix is safe.
            with engine.connect() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
                conn.execute(
                    text(
                        "INSERT INTO schema_migration (version, checksum) VALUES (:v, :c) "
                        "ON DUPLICATE KEY UPDATE applied_at = CURRENT_TIMESTAMP(3)"
                    ),
                    {"v": migration.version, "c": migration.checksum},
                )
                conn.commit()
            applied.append(migration.version)

    return applied


def mark_applied(versions: list[str] | None = None) -> list[str]:
    """Record migrations as applied without running them.

    For a schema created out-of-band — e.g. imported through phpMyAdmin. The
    tables exist but `schema_migration` is empty, so without this the runner
    would try them again (harmless, since every statement is IF NOT EXISTS, but
    it leaves the bookkeeping wrong and hides which version is really live).
    """
    engine = get_engine()
    applied_versions()  # ensures the bookkeeping table exists

    known = {m.version: m for m in discover()}
    targets = versions if versions is not None else list(known)

    recorded: list[str] = []
    with engine.connect() as conn:
        for version in targets:
            migration = known.get(version)
            if migration is None:
                logger.warning("unknown migration '%s'; skipping", version)
                continue
            conn.execute(
                text(
                    "INSERT INTO schema_migration (version, checksum) VALUES (:v, :c) "
                    "ON DUPLICATE KEY UPDATE checksum = VALUES(checksum)"
                ),
                {"v": version, "c": migration.checksum},
            )
            recorded.append(version)
        conn.commit()
    return recorded


def status() -> dict[str, list[str]]:
    done = applied_versions()
    all_versions = [m.version for m in discover()]
    return {
        "applied": sorted(v for v in all_versions if v in done),
        "pending": [v for v in all_versions if v not in done],
    }
