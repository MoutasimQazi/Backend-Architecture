"""Deployment preflight.

One command that answers "is this box ready to take traffic?". Run it on the
host after install and after any config change.

    python scripts/preflight.py
    python scripts/preflight.py --fix    # create missing dirs, apply migrations

Exit codes: 0 ready · 1 blocking problem · 2 ready with warnings.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLOCKERS: list[str] = []
WARNINGS: list[str] = []
OK: list[str] = []


def blocker(message: str) -> None:
    BLOCKERS.append(message)


def warn(message: str) -> None:
    WARNINGS.append(message)


def ok(message: str) -> None:
    OK.append(message)


def check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        blocker(f"Python {major}.{minor} — this codebase needs 3.10+")
    else:
        ok(f"Python {major}.{minor}")


def check_dependencies() -> None:
    required = {
        "fastapi": "the API",
        "pydantic": "the domain contract",
        "sqlalchemy": "storage",
        "pymysql": "the MySQL driver",
        "jwt": "authentication (PyJWT)",
        "numpy": "vector scoring",
        "httpx": "the fetch broker",
        "openai": "model providers",
    }
    optional = {
        "cv2": "barcode scanning (opencv-python-headless)",
        "zxingcpp": "barcode scanning (zxing-cpp)",
        "a2wsgi": "the Passenger ASGI bridge",
        "dotenv": "reading .env (python-dotenv)",
    }

    for module, why in required.items():
        try:
            __import__(module)
        except ImportError:
            blocker(f"missing required package '{module}' — needed for {why}")
    else:
        if not any(b.startswith("missing required") for b in BLOCKERS):
            ok(f"all {len(required)} required packages import")

    missing_optional = []
    for module, why in optional.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(f"{module} ({why})")

    if missing_optional:
        warn("optional packages missing: " + "; ".join(missing_optional))
    else:
        ok("all optional packages import")


def check_config(fix: bool) -> None:
    from packages.config import get_settings

    settings = get_settings()

    if not Path(settings.root / ".env").is_file():
        warn(".env not found — configuration is coming from the environment only")
    else:
        ok(".env present")

    for problem in settings.validate():
        # A missing model key degrades rather than breaks: templates and the
        # scan pipeline still work without one.
        if "model provider" in problem:
            warn(problem + " — chat will return placeholders")
        elif "DB_" in problem:
            blocker(problem)
        else:
            warn(problem)

    if settings.env == "production" and settings.security.allow_header_auth:
        blocker("ALLOW_HEADER_AUTH is on in production — any caller can claim any user id")

    blob_dir = Path(settings.storage.blob_dir)
    if not blob_dir.is_dir():
        if fix:
            blob_dir.mkdir(parents=True, exist_ok=True)
            ok(f"created BLOB_DIR at {blob_dir}")
        else:
            blocker(f"BLOB_DIR does not exist: {blob_dir} (run with --fix to create)")
    elif not os.access(blob_dir, os.W_OK):
        blocker(f"BLOB_DIR is not writable: {blob_dir}")
    else:
        ok(f"BLOB_DIR writable: {blob_dir}")

    if "public_html" in str(blob_dir):
        blocker(f"BLOB_DIR is inside public_html — uploaded health data would be web-readable")


def check_database(fix: bool) -> None:
    from packages.storage import migrate
    from packages.storage.db import ping, server_info

    if not ping():
        blocker("cannot connect to the database")
        return

    info = server_info()
    ok(f"database reachable: {info['version']} / {info['database']}")

    state = migrate.status()
    if state["pending"]:
        if fix:
            applied = migrate.run()
            ok(f"applied migrations: {', '.join(applied)}")
        else:
            blocker(f"pending migrations: {', '.join(state['pending'])} (run with --fix)")
    else:
        ok(f"schema up to date ({len(state['applied'])} migration(s) applied)")

    from sqlalchemy import text

    from packages.storage.db import get_engine

    with get_engine().connect() as conn:
        counts = {
            "faq_item": conn.execute(text("SELECT COUNT(*) FROM faq_item WHERE status='live'")).scalar(),
            "chemical": conn.execute(text("SELECT COUNT(*) FROM chemical")).scalar(),
            "hazard_rule": conn.execute(text("SELECT COUNT(*) FROM hazard_rule WHERE active=1")).scalar(),
        }

    if not counts["faq_item"]:
        warn("no live FAQ items — run scripts/seed_faq.py")
    else:
        ok(f"{counts['faq_item']} live FAQ item(s)")

    if not counts["chemical"]:
        warn("Chemical KB is empty — run scripts/seed_chemicals.py")
    elif counts["chemical"] < 100:
        warn(
            f"Chemical KB holds only {counts['chemical']} chemicals — most real panels "
            "will report ingredients as unrecognised until the ETL has run"
        )
    else:
        ok(f"{counts['chemical']} chemicals in the KB")

    if not counts["hazard_rule"]:
        blocker("no active hazard rules — scans cannot produce a verdict")
    else:
        ok(f"{counts['hazard_rule']} active hazard rule(s)")


def check_schema_hygiene() -> None:
    from scripts.check_ddl import check as ddl_check

    problems = ddl_check()
    if problems:
        blocker(f"{len(problems)} reserved-word collision(s) in the schema — run scripts/check_ddl.py")
    else:
        ok("no reserved-word collisions in the schema")


def check_providers() -> None:
    from packages.chains.providers import available_providers

    providers = available_providers()
    if not providers:
        warn("no model provider configured — chat degrades to templates and placeholders")
        return
    ok(f"model provider(s) configured: {', '.join(providers)}")

    from packages.config import get_settings

    if not get_settings().models.openai_api_key:
        warn(
            "no OPENAI_API_KEY — embeddings are unavailable, so the L3 semantic cache, "
            "the vector half of FAQ retrieval and the resolver's embedding stage stay off "
            "(all degrade to lexical rather than failing)"
        )


def check_app_boots() -> None:
    try:
        from apps.api.main import app  # noqa: F401

        ok("API app imports")
    except Exception as exc:
        blocker(f"API app failed to import: {type(exc).__name__}: {exc}")

    try:
        from packages.jobs import handlers as _  # noqa: F401
        from packages.jobs.registry import registered_types

        types = registered_types()
        if len(types) < 8:
            warn(f"only {len(types)} job handler(s) registered")
        else:
            ok(f"{len(types)} job handlers registered")
    except Exception as exc:
        blocker(f"worker failed to import: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deployment preflight")
    parser.add_argument("--fix", action="store_true", help="create dirs and apply migrations")
    parser.add_argument("--skip-db", action="store_true", help="skip database checks")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.ERROR)

    print("Preflight\n" + "=" * 60)

    check_python()
    check_dependencies()
    check_schema_hygiene()
    check_config(args.fix)
    check_app_boots()
    check_providers()
    if not args.skip_db:
        try:
            check_database(args.fix)
        except Exception as exc:
            blocker(f"database check failed: {type(exc).__name__}: {exc}")

    for line in OK:
        print(f"  ok       {line}")
    for line in WARNINGS:
        print(f"  WARN     {line}")
    for line in BLOCKERS:
        print(f"  BLOCKER  {line}")

    print("=" * 60)
    if BLOCKERS:
        print(f"NOT READY — {len(BLOCKERS)} blocker(s), {len(WARNINGS)} warning(s)")
        return 1
    if WARNINGS:
        print(f"READY, with {len(WARNINGS)} warning(s) — review them before going live")
        return 2
    print("READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
