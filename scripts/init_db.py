"""Check the connection, apply migrations, report status.

    python scripts/init_db.py --check      # connectivity + config only
    python scripts/init_db.py              # apply pending migrations
    python scripts/init_db.py --status     # what is applied vs pending
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.config import get_settings  # noqa: E402
from packages.storage import migrate  # noqa: E402
from packages.storage.db import ping, server_info  # noqa: E402


def check() -> int:
    settings = get_settings()
    print(f"target: {settings.db.url_safe}")

    problems = settings.validate()
    if problems:
        print("\nconfiguration problems:")
        for problem in problems:
            print(f"  - {problem}")

    if not settings.db.password:
        print("\nDB_PASSWORD is empty — set it in .env before continuing.")
        return 2

    if not ping():
        print("\ncould not connect. Common causes on cPanel:")
        print("  - the DB user is not granted on the database")
        print("  - your IP is not in cPanel > Remote MySQL (needed for off-host access)")
        print("  - the password contains characters that were mangled by the shell")
        return 1

    info = server_info()
    print(f"\nconnected: MySQL {info['version']}, database '{info['database']}'")

    # FULLTEXT and the JSON columns both need a reasonably modern server.
    version = str(info["version"])
    if version.startswith("5."):
        print(f"warning: server is {version}; this schema expects MySQL 8.0+ or MariaDB 10.5+")

    return 0


def verify() -> int:
    """Confirm every table the code reads actually exists.

    Worth its own command when the schema was imported by hand: a single
    statement that failed silently mid-import surfaces here rather than as a
    1146 at runtime.
    """
    from sqlalchemy import text as sql_text

    from packages.storage.db import get_engine

    expected = set()
    for migration in migrate.discover():
        for statement in migrate._split_statements(migration.sql):
            match = re.search(
                r"CREATE TABLE IF NOT EXISTS\s+`?(\w+)`?", statement, re.IGNORECASE
            )
            if match:
                expected.add(match.group(1))

    with get_engine().connect() as conn:
        rows = conn.execute(
            sql_text("SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()")
        ).all()
    present = {str(r[0]) for r in rows}

    missing = sorted(expected - present)
    print(f"\ntables expected: {len(expected)}, present: {len(expected & present)}")
    if missing:
        print(f"MISSING: {', '.join(missing)}")
        print("\nRe-import the migration file(s) that create these, then run --verify again.")
        return 1

    print("all expected tables exist")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise the database")
    parser.add_argument("--check", action="store_true", help="connectivity check only")
    parser.add_argument("--status", action="store_true", help="show migration status")
    parser.add_argument("--dry-run", action="store_true", help="list what would run")
    parser.add_argument(
        "--mark-applied",
        action="store_true",
        help="record migrations as applied without running them "
        "(for a schema imported through phpMyAdmin)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check that every expected table exists",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rc = check()
    if rc != 0 or args.check:
        return rc

    if args.verify:
        return verify()

    if args.status:
        state = migrate.status()
        print(f"\napplied: {', '.join(state['applied']) or '(none)'}")
        print(f"pending: {', '.join(state['pending']) or '(none)'}")
        return 0

    if args.mark_applied:
        recorded = migrate.mark_applied()
        print(f"\nrecorded as applied: {', '.join(recorded) or '(none)'}")
        return verify()

    applied = migrate.run(dry_run=args.dry_run)
    if applied:
        print(f"\napplied: {', '.join(applied)}")
    else:
        print("\nschema is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
