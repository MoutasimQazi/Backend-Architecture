"""The eval suites (arch.md 14)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from packages.evals.harness import EvalCase, EvalReport, EvalSuite, load_golden

logger = logging.getLogger(__name__)


class RouterSuite(EvalSuite):
    """Label accuracy against a labelled golden set. Gate: no regression.

    Set at 0.85 rather than 1.0: the router is partly heuristic and partly a
    model, so demanding perfection would make the gate noise. What it catches
    is a real drop.
    """

    name = "router"
    gate = 0.85
    requires_db = True

    def run(self, session: Optional[Any] = None) -> EvalReport:
        from packages.orchestrator.router import Router

        report = EvalReport(suite=self.name, gate=self.gate)
        router = Router(session)

        for case in load_golden("router.json"):
            outcome = router.route(case["message"])
            actual = str(outcome.decision.label.value)
            report.cases.append(
                EvalCase(
                    case_id=case["id"],
                    passed=actual == case["expected"],
                    expected=case["expected"],
                    actual=actual,
                    note=outcome.decision.rationale[:80],
                )
            )
        return report


class HazardRulesSuite(EvalSuite):
    """Exact match against curated verdicts. Gate: 100% — this is
    deterministic code, so anything less is a bug, not variance."""

    name = "hazard_rules"
    gate = 1.0
    requires_db = False

    def run(self, session: Optional[Any] = None) -> EvalReport:
        from packages.domain.enums import HazardLevel, ResolutionMethod
        from packages.domain.models import ResolvedIngredient
        from packages.product.rules import DEFAULT_RULES, HazardRulesEngine

        report = EvalReport(suite=self.name, gate=self.gate)

        rules = [
            {
                "rule_id": r["rule_id"],
                "version": 1,
                "priority": r["priority"],
                "condition": r["condition"],
                "effect": r["effect"],
            }
            for r in DEFAULT_RULES
        ]
        engine = HazardRulesEngine(rules, jurisdiction="IN", product_class="cosmetic")

        for case in load_golden("hazard_rules.json"):
            ingredient = ResolvedIngredient(
                raw_token="Test",
                position=0,
                chemical_id="test",
                display_name="Test",
                confidence=1.0,
                resolution_method=ResolutionMethod.EXACT_INCI,
            )
            finding = engine.assess(ingredient, {}, case["assertions"])

            fired = {r.split("@")[0] for r in finding.rule_ids}
            expected_rules = set(case.get("expect_rules", []))

            problems = []
            if str(finding.hazard_level.value) != case["expect_level"]:
                problems.append(
                    f"level {finding.hazard_level.value} != {case['expect_level']}"
                )
            if fired != expected_rules:
                problems.append(f"rules {sorted(fired)} != {sorted(expected_rules)}")
            if case.get("expect_endocrine") and not finding.endocrine_flag:
                problems.append("endocrine flag not set")
            if case.get("expect_allergen") and not finding.allergen_flag:
                problems.append("allergen flag not set")

            report.cases.append(
                EvalCase(
                    case_id=case["id"],
                    passed=not problems,
                    expected=case["expect_level"],
                    actual=str(finding.hazard_level.value),
                    note="; ".join(problems),
                )
            )
        return report


class PersonalisationSafetySuite(EvalSuite):
    """No allergen missed on synthetic profiles. Gate: 100%.

    The strictest gate in the system, and correctly so: a missed allergen is
    the failure that physically hurts someone.
    """

    name = "personalisation_safety"
    gate = 1.0
    requires_db = False

    def run(self, session: Optional[Any] = None) -> EvalReport:
        from packages.domain.enums import ResolutionMethod
        from packages.domain.models import (
            MedicalSnapshot,
            Profile,
            ResolvedIngredient,
            UserContext,
        )
        from packages.product.personal import PersonalRiskMatcher

        report = EvalReport(suite=self.name, gate=self.gate)

        for case in load_golden("personalisation_safety.json"):
            unresolved = set(case.get("unresolved", []))
            ingredients = [
                ResolvedIngredient(
                    raw_token=name,
                    position=i,
                    chemical_id=None if name in unresolved else f"c{i}",
                    display_name=None if name in unresolved else name,
                    confidence=0.0 if name in unresolved else 1.0,
                    resolution_method=(
                        ResolutionMethod.UNRESOLVED
                        if name in unresolved
                        else ResolutionMethod.EXACT_INCI
                    ),
                )
                for i, name in enumerate(case["ingredients"])
            ]

            context = UserContext(
                user_id="eval",
                profile=Profile(
                    preferences=case.get("preferences", []),
                    pregnancy_status=case.get("pregnancy_status"),
                ),
                medical=MedicalSnapshot(
                    allergies=case.get("allergies", []),
                    conditions=case.get("conditions", []),
                ),
            )

            assessment = PersonalRiskMatcher(context).assess(ingredients, [])
            flagged = {f.display_name for f in assessment.flags}

            problems = []
            for name in case.get("must_flag", []):
                if name not in flagged:
                    problems.append(f"MISSED {name!r}")
            for name in case.get("must_not_flag", []):
                if name in flagged:
                    problems.append(f"false positive on {name!r}")
            if not case.get("must_flag") and not case.get("must_not_flag") and flagged:
                problems.append(f"unexpected flags: {sorted(flagged)}")

            report.cases.append(
                EvalCase(
                    case_id=case["id"],
                    passed=not problems,
                    expected=case.get("must_flag", []),
                    actual=sorted(flagged),
                    note="; ".join(problems),
                )
            )
        return report


class IngredientParsingSuite(EvalSuite):
    """Panel → tokens. Gate: 100% — the parser is deterministic."""

    name = "ingredient_parsing"
    gate = 1.0
    requires_db = False

    def run(self, session: Optional[Any] = None) -> EvalReport:
        from packages.product.ocr import extract_ingredient_panel
        from packages.product.parser import parse_panel

        report = EvalReport(suite=self.name, gate=self.gate)

        for case in load_golden("ingredient_parsing.json"):
            panel, had_header = extract_ingredient_panel(case["panel"])
            parsed = parse_panel(panel, had_header=had_header)
            tokens = [t.text for t in parsed.tokens]

            problems = []
            if tokens != case["expect_tokens"]:
                problems.append(f"{tokens} != {case['expect_tokens']}")

            expected_traces = case.get("expect_traces")
            if expected_traces is not None:
                traces = [t.text for t in parsed.tokens if t.is_trace]
                if traces != expected_traces:
                    problems.append(f"traces {traces} != {expected_traces}")

            for token_text, expected_quals in (case.get("expect_qualifiers") or {}).items():
                token = next((t for t in parsed.tokens if t.text == token_text), None)
                if token is None:
                    problems.append(f"missing token {token_text!r}")
                elif not all(q in token.qualifiers for q in expected_quals):
                    problems.append(f"{token_text} qualifiers {token.qualifiers} != {expected_quals}")

            report.cases.append(
                EvalCase(
                    case_id=case["id"],
                    passed=not problems,
                    expected=case["expect_tokens"],
                    actual=tokens,
                    note="; ".join(problems),
                )
            )
        return report


class FaqRetrievalSuite(EvalSuite):
    """recall@1 for questions that should reach their FAQ item."""

    name = "faq_retrieval"
    gate = 0.80
    requires_db = True

    def run(self, session: Optional[Any] = None) -> EvalReport:
        from packages.retrievers.faq import FaqRetriever

        report = EvalReport(suite=self.name, gate=self.gate)
        retriever = FaqRetriever(session)

        expectations = [
            ("tdee_exact", "What is TDEE?", "faq_tdee"),
            ("tdee_para", "what does TDEE mean", "faq_tdee"),
            ("tdee_loose", "explain total daily energy expenditure to me", "faq_tdee"),
            ("protein_exact", "How much protein should I eat per day?", "faq_protein_target"),
            ("protein_para", "daily protein intake", "faq_protein_target"),
            ("protein_loose", "how many grams of protein do I need each day", "faq_protein_target"),
            ("rest_para", "do I need rest days", "faq_rest_days"),
            ("sleep_para", "why is sleep important for recovery", "faq_sleep_recovery"),
            ("privacy_para", "is my health data private", "faq_data_privacy"),
            ("scan_para", "how to scan a barcode", "faq_scan_product"),
            ("bmi_para", "what is a good BMI", "faq_bmi_meaning"),
        ]

        for case_id, question, expected in expectations:
            results, _ = retriever.retrieve(question, top_k=1)
            actual = results[0].faq_id if results else None
            report.cases.append(
                EvalCase(
                    case_id=case_id,
                    passed=actual == expected,
                    expected=expected,
                    actual=actual,
                    note=f"score {results[0].score:.2f}" if results else "no hit",
                )
            )
        return report


class CacheCorrectnessSuite(EvalSuite):
    """The cache key must isolate users, contexts and versions. Gate: 100% —
    a false hit means one user seeing another's answer."""

    name = "cache_correctness"
    gate = 1.0
    requires_db = False

    def run(self, session: Optional[Any] = None) -> EvalReport:
        from packages.cache import build_cache_key, context_fingerprint
        from packages.domain.models import SectionMeta, UserContext

        report = EvalReport(suite=self.name, gate=self.gate)

        def case(case_id: str, passed: bool, note: str = "") -> None:
            report.cases.append(EvalCase(case_id=case_id, passed=passed, note=note))

        base = build_cache_key("q", "FAQ", scope="global")
        case("same_inputs_same_key", base.digest == build_cache_key("q", "FAQ", scope="global").digest)
        case("scope_isolates", base.digest != build_cache_key("q", "FAQ", scope="u1").digest)
        case("users_isolated",
             build_cache_key("q", "FAQ", scope="u1").digest
             != build_cache_key("q", "FAQ", scope="u2").digest)
        case("route_changes_key", base.digest != build_cache_key("q", "PERSONAL", scope="global").digest)
        case("locale_changes_key",
             base.digest != build_cache_key("q", "FAQ", scope="global", locale="hi").digest)
        case("prompt_version_changes_key",
             base.digest != build_cache_key("q", "FAQ", scope="global", prompt_version="v2").digest)
        case("kb_version_changes_key",
             base.digest != build_cache_key("q", "FAQ", scope="global", kb_version="v2").digest)
        case("model_changes_key",
             base.digest != build_cache_key("q", "FAQ", scope="global", model_id="x").digest)
        case("normalisation_collapses",
             build_cache_key("What is TDEE?", "FAQ").digest
             == build_cache_key("what is tdee!!", "FAQ").digest)

        context = UserContext(user_id="u1")
        context.meta["nutrition"] = SectionMeta(version="7")
        context.meta["activity"] = SectionMeta(version="3")
        before = context_fingerprint(context, ["nutrition"])
        context.meta["activity"] = SectionMeta(version="4")
        case(
            "unrelated_section_does_not_bust",
            context_fingerprint(context, ["nutrition"]) == before,
            "a workout sync must not invalidate a nutrition answer",
        )
        case(
            "read_section_does_bust",
            context_fingerprint(context, ["nutrition", "activity"]) != before,
        )
        return report


class IndependenceSuite(EvalSuite):
    """arch.md 9.3 — the inverted 'non-funded' filter must stay fixed.

    Public funding must score *above* unfunded, which is the exact opposite of
    what the old rule did.
    """

    name = "independence"
    gate = 1.0
    requires_db = False

    def run(self, session: Optional[Any] = None) -> EvalReport:
        from packages.evidence.independence import FunderClass, classify_funder, score_independence

        report = EvalReport(suite=self.name, gate=self.gate)

        def case(case_id: str, passed: bool, note: str = "") -> None:
            report.cases.append(EvalCase(case_id=case_id, passed=passed, note=note))

        case("nih_is_public", classify_funder(["National Institutes of Health"]) is FunderClass.PUBLIC)
        case("icmr_is_public", classify_funder(["ICMR"]) is FunderClass.PUBLIC)
        case("horizon_is_public", classify_funder(["Horizon 2020"]) is FunderClass.PUBLIC)
        case("wellcome_is_charitable",
             classify_funder(["Wellcome Trust"]) is FunderClass.CHARITABLE)
        case("unilever_is_industry", classify_funder(["Unilever Ltd"]) is FunderClass.INDUSTRY)
        case("ilsi_is_trade", classify_funder(["ILSI Europe"]) is FunderClass.TRADE_ASSOCIATION)
        case("empty_is_none", classify_funder([]) is FunderClass.NONE)

        public = score_independence(["NIH"]).value
        unfunded = score_independence([]).value
        industry = score_independence(["Nestlé S.A."]).value

        case(
            "public_beats_unfunded",
            public > unfunded,
            f"public {public} must exceed unfunded {unfunded} - the old rule had this backwards",
        )
        case("unfunded_beats_industry", unfunded > industry, f"{unfunded} vs {industry}")
        case(
            "co_funding_flags_industry",
            classify_funder(["NIH", "Coca-Cola Company"]) is FunderClass.INDUSTRY,
            "industry co-funding must not be laundered by a public co-funder",
        )
        case(
            "coi_lowers_score",
            score_independence(["NIH"], declared_coi=True).value < public,
        )
        case(
            "preregistration_raises_score",
            score_independence(["NIH"], registry_status="prospective").value >= public,
        )
        return report


ALL_SUITES: list[EvalSuite] = [
    RouterSuite(),
    FaqRetrievalSuite(),
    IngredientParsingSuite(),
    HazardRulesSuite(),
    PersonalisationSafetySuite(),
    CacheCorrectnessSuite(),
    IndependenceSuite(),
]
