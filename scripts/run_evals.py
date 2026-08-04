"""Run the golden-set evals (arch.md 14).

    python scripts/run_evals.py            # all suites
    python scripts/run_evals.py --offline  # only those needing no DB

Exits non-zero if any suite misses its gate, so it can gate a deploy or CI.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.evals.harness import run_all  # noqa: E402
from packages.evals.suites import ALL_SUITES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run golden-set evals")
    parser.add_argument("--offline", action="store_true", help="skip suites needing a database")
    parser.add_argument("--verbose", action="store_true", help="show every failing case")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    from packages.chains.providers import is_configured

    session_factory = None
    if not args.offline:
        from packages.storage.db import session_scope

        session_factory = session_scope

    suites = [s for s in ALL_SUITES if not (args.offline and s.requires_db)]
    reports = run_all(suites, session_factory, have_model=is_configured())

    print()
    for report in reports:
        print("  " + report.summary())
        if report.failures and (args.verbose or not report.meets_gate):
            for case in report.failures[:8]:
                print(f"      {case.case_id}: expected {case.expected!r}, got {case.actual!r}"
                      + (f"  [{case.note}]" if case.note else ""))

    failed = [r for r in reports if not r.meets_gate]
    ran = [r for r in reports if not r.skipped]

    print()
    if failed:
        print(f"{len(failed)} of {len(ran)} suite(s) below gate")
        return 1

    print(f"all {len(ran)} suite(s) met their gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
