"""Product analysis orchestration (arch.md 8.1).

    capture -> decode barcode -> identify product -> OCR label ->
    parse + normalise -> Chemical KB lookup -> hazard rules ->
    personal risk -> structured verdict

The change that matters: **the runtime path touches zero external toxicology
APIs.** Where the old `app.py` fanned out to 6 upstream APIs plus an LLM call
per ingredient inside the HTTP request (~90 calls per scan), this does a small
number of batched local queries. Unknown ingredients enqueue a research job and
the response says so, rather than blocking the turn.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.orm import Session

from packages.domain.enums import (
    JobType,
    ProductUnidentifiedReason,
    ResolutionMethod,
    Verdict,
)
from packages.domain.models import (
    HazardFinding,
    ProductAnalysis,
    ProductIdentity,
    ResolvedIngredient,
    UserContext,
)
from packages.product import barcode as barcode_mod
from packages.product.ocr import OcrService, extract_ingredient_panel, stitch
from packages.product.parser import ParsedPanel, parse_panel
from packages.product.personal import PersonalRiskMatcher
from packages.product.resolver import IngredientResolver
from packages.product.rules import HazardRulesEngine
from packages.storage.repositories.chemicals import ChemicalRepository
from packages.storage.repositories.jobs import JobRepository
from packages.storage.repositories.products import ProductRepository

logger = logging.getLogger(__name__)

# Class of product decides how the same chemical is graded (arch.md 8.5).
DEFAULT_PRODUCT_CLASS = "cosmetic"


@dataclass
class AnalysisTrace:
    """Where the time went. Surfaced under debug, and the thing to watch when
    the p95 scan-latency target is the gate on cutting over."""

    barcode_ms: float = 0.0
    identify_ms: float = 0.0
    ocr_ms: float = 0.0
    parse_ms: float = 0.0
    resolve_ms: float = 0.0
    rules_ms: float = 0.0
    external_calls: int = 0
    ocr_cache_hits: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return round(
            self.barcode_ms + self.identify_ms + self.ocr_ms
            + self.parse_ms + self.resolve_ms + self.rules_ms,
            2,
        )


class ProductAnalyzer:
    def __init__(
        self,
        session: Session,
        *,
        user_id: Optional[str] = None,
        jurisdiction: str = "IN",
    ) -> None:
        self.session = session
        self.user_id = user_id
        self.jurisdiction = jurisdiction
        self.products = ProductRepository(session)
        self.chemicals = ChemicalRepository(session)
        self.ocr = OcrService(session)
        self.resolver = IngredientResolver(session)
        self.jobs = JobRepository(session)

    def analyze(
        self,
        images: list[bytes],
        *,
        context: Optional[UserContext] = None,
        product_class: Optional[str] = None,
    ) -> tuple[ProductAnalysis, AnalysisTrace]:
        trace = AnalysisTrace()
        analysis = ProductAnalysis()

        if not images:
            analysis.unidentified_reason = ProductUnidentifiedReason.UNREADABLE
            trace.notes.append("no images supplied")
            return analysis, trace

        identity, ingredients_text, resolved_class = self._identify(images, trace, product_class)
        analysis.identity = identity

        if not ingredients_text:
            analysis.unidentified_reason = self._failure_reason(identity, trace)
            return analysis, trace

        # -- parse ---------------------------------------------------------
        started = time.perf_counter()
        panel_text, had_header = extract_ingredient_panel(ingredients_text)
        parsed: ParsedPanel = parse_panel(panel_text, had_header=had_header)
        trace.parse_ms = round((time.perf_counter() - started) * 1000, 2)

        if not parsed.tokens:
            analysis.unidentified_reason = self._failure_reason(identity, trace)
            trace.notes.append("panel produced no usable tokens")
            return analysis, trace

        # -- resolve -------------------------------------------------------
        started = time.perf_counter()
        analysis.ingredients = self.resolver.resolve_panel(parsed.tokens)
        trace.resolve_ms = round((time.perf_counter() - started) * 1000, 2)

        # -- assess --------------------------------------------------------
        started = time.perf_counter()
        analysis.hazards = self._assess(analysis.ingredients, resolved_class)

        matcher = PersonalRiskMatcher(
            context, cross_reactants=self._cross_reactants(context)
        )
        assessment = matcher.assess(analysis.ingredients, analysis.hazards)
        analysis.personal_flags = assessment.flags
        analysis.verdict = assessment.verdict
        trace.rules_ms = round((time.perf_counter() - started) * 1000, 2)

        # -- unknowns become jobs, not a blocked turn (arch.md 8.4) --------
        unresolved = [i for i in analysis.ingredients if not i.resolved]
        if unresolved:
            analysis.pending_chemical_ids = [i.raw_token for i in unresolved]
            analysis.pending_job_ids = self._enqueue_research(unresolved)

        from packages.config import get_settings

        analysis.kb_version = get_settings().kb_version
        return analysis, trace

    # -- identification ---------------------------------------------------

    def _identify(
        self,
        images: list[bytes],
        trace: AnalysisTrace,
        product_class: Optional[str],
    ) -> tuple[Optional[ProductIdentity], str, str]:
        """Returns (identity, ingredients_text, product_class).

        Ingredient text in order of trust: the product database, then the label
        OCR. Falling through to OCR is what makes an unlisted barcode
        recoverable — the panel is nearly always in the same photo, and
        the old `app.py` used to discard it the moment a barcode was found.
        """
        resolved_class = product_class or DEFAULT_PRODUCT_CLASS

        # 1. barcode
        started = time.perf_counter()
        scan = barcode_mod.scan(images)
        trace.barcode_ms = round((time.perf_counter() - started) * 1000, 2)

        if not barcode_mod.backend_available():
            trace.notes.append(f"barcode backend unavailable: {barcode_mod.backend_error()}")

        identity: Optional[ProductIdentity] = None
        ingredients_text = ""

        if scan.found:
            started = time.perf_counter()
            candidates = [c for c in (scan.barcode, scan.gtin14) if c]

            # 2a. local table first — a repeat scan never leaves the box
            cached = self.products.by_barcode(candidates)
            if cached:
                identity = ProductRepository.to_identity(cached)
                ingredients_text = cached.get("ingredients_text") or ""
                resolved_class = cached.get("product_class") or resolved_class
                trace.notes.append("product served from local cache")
            else:
                # 2b. upstream cascade
                from packages.connectors.openfacts import OpenFactsConnector

                record = OpenFactsConnector().lookup(scan.barcode)
                trace.external_calls += 1

                if record.found:
                    resolved_class = record.product_class or resolved_class
                    product_id = self.products.upsert(
                        barcode=scan.barcode,
                        barcode_format=scan.symbology,
                        name=record.name,
                        brand=record.brand,
                        category=record.category,
                        product_class=resolved_class,
                        ingredients_text=record.ingredients_text,
                        source=record.source,
                        confidence=record.confidence,
                    )
                    identity = ProductIdentity(
                        product_id=product_id,
                        barcode=scan.barcode,
                        barcode_format=scan.symbology,
                        name=record.name,
                        brand=record.brand,
                        category=record.category,
                        source=record.source,
                        confidence=record.confidence,
                        fetched_at=record.fetched_at,
                    )
                    ingredients_text = record.ingredients_text or ""
                else:
                    identity = ProductIdentity(
                        barcode=scan.barcode,
                        barcode_format=scan.symbology,
                        confidence=scan.confidence,
                    )
                    trace.notes.append("barcode read but product not in any database")

            trace.identify_ms = round((time.perf_counter() - started) * 1000, 2)

        elif scan.saw_non_retail_symbol:
            trace.notes.append(f"non-retail symbol(s): {', '.join(scan.rejected_symbols)}")

        # 3. OCR fallback — always, when the database gave us no panel
        if not ingredients_text.strip():
            started = time.perf_counter()
            results = self.ocr.read_many(images)
            trace.ocr_ms = round((time.perf_counter() - started) * 1000, 2)
            trace.ocr_cache_hits = sum(1 for r in results if r.cached)
            trace.external_calls += sum(1 for r in results if not r.cached)

            ingredients_text = stitch(results)
            if not ingredients_text:
                errors = [r.error for r in results if r.error]
                if errors:
                    trace.notes.append(f"ocr failed: {errors[0]}")

        return identity, ingredients_text, resolved_class

    def _failure_reason(
        self, identity: Optional[ProductIdentity], trace: AnalysisTrace
    ) -> ProductUnidentifiedReason:
        """arch.md 8.7 keeps three subtypes because the right user action differs."""
        if any("non-retail symbol" in note for note in trace.notes):
            return ProductUnidentifiedReason.NON_RETAIL_SYMBOL
        if identity is not None and identity.barcode:
            return ProductUnidentifiedReason.UNLISTED_PRODUCT
        return ProductUnidentifiedReason.UNREADABLE

    # -- assessment -------------------------------------------------------

    def _assess(
        self, ingredients: list[ResolvedIngredient], product_class: str
    ) -> list[HazardFinding]:
        resolved_ids = [i.chemical_id for i in ingredients if i.chemical_id]
        if not resolved_ids:
            return []

        # Three batched queries for the whole panel, regardless of its length.
        dossiers = self.chemicals.get_many(resolved_ids)
        assertions = self.chemicals.assertions_for(
            resolved_ids, jurisdiction=self.jurisdiction, product_class=product_class
        )
        evidence = self.chemicals.evidence_for(resolved_ids)

        engine = HazardRulesEngine(
            self.chemicals.active_rules(
                product_class=product_class, jurisdiction=self.jurisdiction
            ),
            jurisdiction=self.jurisdiction,
            product_class=product_class,
        )

        findings: list[HazardFinding] = []
        for ingredient in ingredients:
            if not ingredient.chemical_id:
                continue
            findings.append(
                engine.assess(
                    ingredient,
                    dossiers.get(ingredient.chemical_id, {}),
                    assertions.get(ingredient.chemical_id, []),
                    evidence.get(ingredient.chemical_id, []),
                    total_ingredients=len(ingredients),
                )
            )
        return findings

    def _cross_reactants(self, context: Optional[UserContext]) -> dict[str, list[dict[str, Any]]]:
        if context is None or not context.medical.allergies:
            return {}
        keys = [a.lower().strip() for a in context.medical.allergies if a]
        return self.chemicals.cross_reactants(keys)

    @staticmethod
    def _research_priority_for_test(
        unresolved: list[ResolvedIngredient],
    ) -> list[ResolvedIngredient]:
        return _research_priority(unresolved)

    def _enqueue_research(self, unresolved: list[ResolvedIngredient]) -> list[str]:
        job_ids: list[str] = []
        for ingredient in _research_priority(unresolved)[:10]:
            token = ingredient.raw_token.strip()
            if not token:
                continue
            # Idempotent on the token: ten users scanning the same unlisted
            # product enqueue one research job, not ten.
            job_ids.append(
                self.jobs.enqueue(
                    job_type=JobType.CHEMICAL_RESEARCH,
                    payload={"token": token},
                    user_id=self.user_id,
                    priority=200,
                    idempotency_key=f"chem:{token.lower()[:150]}",
                )
            )
        return job_ids


# Ingredient panels are ordered by weight, so the first entries are bulk
# (sugar, water, flour) and the last are the additives — colours, preservatives,
# fragrance allergens. That is the exact inverse of how interesting they are to
# a hazard scanner.
#
# Taking the first N unresolved tokens therefore spent the whole research
# budget on "sugar" and "corn syrup" while Red #40, Tartrazine, Sunset Yellow
# and Brilliant Blue — the four ingredients on that pack that actually carry
# regulatory warnings — were never looked up at all.

# Colour indices, E-numbers, CI numbers, FD&C names: near-certain additives.
_ADDITIVE_PATTERN = re.compile(
    r"(#\s*\d|\bE\s?\d{3}\b|\bci\s*\d{4,5}\b|\bfd&?c\b|\bred\b|\byellow\b|\bblue\b|\bgreen\b"
    r"|\blake\b|\bdye\b|\bcolou?r\b|\btartrazine\b|\bcarmine\b"
    r"|\bparaben\b|\bbenzoate\b|\bsorbate\b|\bsulph?ite\b|\bnitrite\b|\bnitrate\b|\bbht\b|\bbha\b"
    r"|\bglutamate\b|\bmsg\b|\baspartame\b|\bsucralose\b|\bsaccharin\b|\bacesulfame\b"
    r"|\bedta\b|\bphthalate\b|\btriclosan\b|\bformaldehyde\b)",
    re.IGNORECASE,
)

# Bulk foodstuffs and category words. Worth researching eventually, but they
# are rarely the reason a product is unsafe, and several are not single
# compounds at all so PubChem cannot resolve them.
_BULK_PATTERN = re.compile(
    r"^(sugar|salt|water|aqua|corn syrup|glucose|fructose|sucrose|flour|wheat flour|starch|"
    r"modified food starch|milk|cream|butter|egg|eggs|oil|palm oil|vegetable oil|"
    r"natural and artificial flavou?rs?|natural flavou?rs?|artificial flavou?rs?|"
    r"artificial colou?r|spices?|yeast|honey|cocoa|dextrin|maltodextrin)$",
    re.IGNORECASE,
)


def _research_priority(unresolved: list[ResolvedIngredient]) -> list[ResolvedIngredient]:
    """Order unresolved tokens by how likely they are to matter.

    Stable within a band, so panel order still breaks ties and the result is
    deterministic — which matters because the same product must enqueue the
    same jobs every time for the idempotency keys to collapse.
    """

    def rank(item: ResolvedIngredient) -> int:
        token = (item.raw_token or "").strip()
        if _ADDITIVE_PATTERN.search(token):
            return 0
        if _BULK_PATTERN.match(token):
            return 2
        return 1

    return sorted(unresolved, key=rank)
