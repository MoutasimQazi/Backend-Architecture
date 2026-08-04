"""Golden sets and the eval harness (arch.md 14).

Run in CI on every prompt or model change. Without these a regression in the
router or the hazard rules is invisible until a user reports it.
"""

from packages.evals.harness import EvalCase, EvalReport, EvalSuite, run_all

__all__ = ["EvalCase", "EvalReport", "EvalSuite", "run_all"]
