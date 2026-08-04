"""Eval harness.

arch.md 14 defines the gates:

| Set                    | Measures                        | Gate           |
|------------------------|---------------------------------|----------------|
| router                 | label accuracy, confusion       | no regression  |
| FAQ retrieval          | recall@1, precision at tau       | >= baseline    |
| ingredient parsing     | token -> chemical F1             | >= baseline    |
| hazard rules           | exact match vs curated verdicts | 100%           |
| citation faithfulness  | claim-level entailment          | >= threshold   |
| personalisation safety | no allergen missed              | 100%           |
| cache correctness      | sampled false-hit rate          | below threshold|

The 100% gates are the ones that matter most: they cover deterministic code, so
anything below 100% is a bug rather than model variance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


@dataclass
class EvalCase:
    case_id: str
    passed: bool
    expected: Any = None
    actual: Any = None
    note: str = ""


@dataclass
class EvalReport:
    suite: str
    cases: list[EvalCase] = field(default_factory=list)
    gate: float = 1.0
    skipped: bool = False
    skip_reason: str = ""

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def accuracy(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0

    @property
    def meets_gate(self) -> bool:
        if self.skipped:
            return True
        return self.accuracy >= self.gate

    @property
    def failures(self) -> list[EvalCase]:
        return [c for c in self.cases if not c.passed]

    def summary(self) -> str:
        if self.skipped:
            return f"{self.suite:<24} SKIPPED ({self.skip_reason})"
        status = "PASS" if self.meets_gate else "FAIL"
        return (
            f"{self.suite:<24} {status}  {self.passed}/{self.total} "
            f"({self.accuracy:.0%}, gate {self.gate:.0%})"
        )


class EvalSuite:
    """A named set of cases with a pass gate."""

    name: str = "suite"
    gate: float = 1.0
    requires_db: bool = False
    requires_model: bool = False

    def run(self, session: Optional[Any] = None) -> EvalReport:
        raise NotImplementedError


def load_golden(filename: str) -> list[dict[str, Any]]:
    path = GOLDEN_DIR / filename
    if not path.is_file():
        logger.warning("golden set %s not found", path)
        return []
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_all(
    suites: list[EvalSuite],
    session_factory: Optional[Callable[[], Any]] = None,
    *,
    have_model: bool = False,
) -> list[EvalReport]:
    reports: list[EvalReport] = []

    for suite in suites:
        if suite.requires_model and not have_model:
            reports.append(
                EvalReport(
                    suite=suite.name,
                    gate=suite.gate,
                    skipped=True,
                    skip_reason="no model provider configured",
                )
            )
            continue

        if suite.requires_db and session_factory is None:
            reports.append(
                EvalReport(
                    suite=suite.name,
                    gate=suite.gate,
                    skipped=True,
                    skip_reason="no database session",
                )
            )
            continue

        try:
            if suite.requires_db:
                with session_factory() as session:  # type: ignore[misc]
                    reports.append(suite.run(session))
            else:
                reports.append(suite.run())
        except Exception as exc:
            logger.exception("suite %s crashed", suite.name)
            reports.append(
                EvalReport(
                    suite=suite.name,
                    gate=suite.gate,
                    cases=[EvalCase(case_id="__crash__", passed=False, note=str(exc))],
                )
            )

    return reports
