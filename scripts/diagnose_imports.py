"""Diagnose a ModuleNotFoundError for a local package.

Prints what Python can actually see, so an import failure on a host is a fact
rather than a guess.

    python scripts/diagnose_imports.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print(f"cwd            : {os.getcwd()}")
print(f"repo root      : {ROOT}")
print(f"python         : {sys.executable}")
print(f"version        : {sys.version.split()[0]}")

print("\nsys.path:")
for entry in sys.path[:12]:
    print(f"  {entry!r}")

print("\n-- expected package directories --")
expected = [
    "packages",
    "packages/jobs",
    "packages/chains",
    "packages/orchestrator",
    "packages/storage",
    "packages/product",
    "packages/evidence",
    "packages/etl",
    "packages/restaurant",
    "packages/evals",
    "packages/tools",
    "packages/retrievers",
    "packages/guards",
    "packages/connectors",
    "packages/cache",
    "packages/common",
    "packages/config",
    "packages/domain",
    "apps",
    "apps/api",
    "apps/worker",
]

missing_dirs = []
missing_init = []
for rel in expected:
    path = ROOT / rel
    if not path.is_dir():
        missing_dirs.append(rel)
        print(f"  MISSING DIR   {rel}")
        continue
    init = path / "__init__.py"
    files = sorted(p.name for p in path.glob("*.py"))
    marker = "" if init.is_file() else "   <-- no __init__.py"
    if not init.is_file():
        missing_init.append(rel)
    print(f"  ok  {rel:<28} {len(files)} .py file(s){marker}")
    if rel == "packages/jobs":
        print(f"       contents: {files or '(EMPTY)'}")

print("\n-- import resolution --")
for name in ("packages", "packages.jobs", "packages.jobs.registry", "packages.jobs.handlers"):
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        print(f"  {name:<28} find_spec raised {type(exc).__name__}: {exc}")
        continue
    if spec is None:
        print(f"  {name:<28} NOT FOUND")
    else:
        origin = spec.origin or f"namespace {list(spec.submodule_search_locations or [])}"
        print(f"  {name:<28} -> {origin}")

print("\n-- actual import attempt --")
try:
    from packages.jobs import handlers  # noqa: F401
    from packages.jobs.registry import registered_types

    print(f"  ok: {len(registered_types())} job handlers registered")
except Exception as exc:
    print(f"  FAILED {type(exc).__name__}: {exc}")
    import traceback

    traceback.print_exc()

print("\n-- verdict --")
if missing_dirs:
    print(f"  {len(missing_dirs)} directory/ies did not upload: {', '.join(missing_dirs)}")
    print("  Re-upload them. FTP clients routinely skip directories they think are empty.")
elif missing_init:
    print(f"  present but missing __init__.py: {', '.join(missing_init)}")
    print("  Create the empty files, or re-upload with hidden/small files included.")
else:
    print("  all directories and __init__.py files are present")
