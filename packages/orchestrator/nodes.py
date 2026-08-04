"""Graph nodes (arch.md 4.3).

Each function takes and returns `ConversationState` and writes only the fields
it owns. They are plain functions so the sequencer in `pipeline.py` can be
replaced by a LangGraph `StateGraph` in phase 2 without touching this file.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from packages.cache import CacheService, build_cache_key, context_fingerprint
from packages.common.text import normalise_question
from packages.config import get_settings
from packages.domain.enums import (
    BlockType,
    CacheTier,
    ConsentScope,
    FaqCategory,
    JobType,
    ProductUnidentifiedReason,
    RouteLabel,
    SafetyClass,
)
from packages.domain.enums import SafetyFlagKind, Severity
from packages.domain.models import (
    AnswerBlock,
    AnswerPayload,
    ConversationState,
    DataGap,
    SafetyFlag,
    SectionMeta,
    UserContext,
)
from packages.orchestrator import templates
from packages.orchestrator.router import Router
from packages.storage.repositories.conversation import ConversationRepository
from packages.storage.repositories.faq import FaqRepository
from packages.storage.repositories.health import HealthRepository
from packages.storage.repositories.jobs import JobRepository
from packages.storage.repositories.users import UserRepository

logger = logging.getLogger(__name__)

# Which consent scope gates which context section.
SECTION_SCOPES = {
    "profile": ConsentScope.PROFILE,
    "vitals": ConsentScope.VITALS,
    "nutrition": ConsentScope.NUTRITION,
    "activity": ConsentScope.ACTIVITY,
    "medical": ConsentScope.LABS,
}


def ingest(state: ConversationState, session: Session) -> ConversationState:
    """Normalise input and make sure the user row exists."""
    users = UserRepository(session)
    if not users.exists(state.request.user_id):
        users.upsert(
            state.request.user_id,
            locale=state.request.locale,
            jurisdiction=state.request.jurisdiction,
        )
    state.input.text = (state.input.text or "").strip()
    return state


def guard_in(state: ConversationState, session: Session) -> ConversationState:
    """Safety triage runs here; the router reuses the same detectors so a
    blocking flag short-circuits before any spend."""
    return state


def context_build(state: ConversationState, session: Session) -> ConversationState:
    """Assemble `UserContext` from consent-filtered, precomputed aggregates.

    arch.md 6.1: bounded indexed reads only. When an aggregate is missing, the
    section is built live once and a rebuild job is queued, so the slow path
    happens at most once per user rather than once per request.
    """
    user_id = state.request.user_id
    users = UserRepository(session)
    health = HealthRepository(session, user_id)

    state.consent = users.get_consent(user_id)
    context = UserContext(user_id=user_id)
    needs_rebuild = False

    profile, profile_version = users.get_profile(user_id)
    if state.consent.allows(ConsentScope.PROFILE):
        context.profile = profile
        context.meta["profile"] = SectionMeta(
            completeness=1.0 if profile.display_name or profile.weight_kg else 0.0,
            version=str(profile_version or 0),
        )
    else:
        context.meta["profile"] = SectionMeta(
            withheld=True, withheld_reason="consent scope 'profile' not granted"
        )
        state.flags.data_gaps.append(
            DataGap(
                field="profile",
                why_it_matters="answers cannot be personalised without it",
                how_to_supply="grant the profile permission in settings",
            )
        )

    for section in ("vitals", "nutrition", "activity", "medical"):
        scope = SECTION_SCOPES[section]
        if not state.consent.allows(scope):
            context.meta[section] = SectionMeta(
                withheld=True, withheld_reason=f"consent scope '{scope.value}' not granted"
            )
            continue

        aggregate = health.get_aggregate(section)
        if aggregate is not None:
            payload, meta = aggregate
            _apply_section(context, section, payload)
            context.meta[section] = meta
            continue

        needs_rebuild = True
        _apply_section(context, section, _build_live(health, section))
        context.meta[section] = SectionMeta(completeness=0.5, version="0")

    derived = health.get_aggregate("derived")
    if derived is not None:
        payload, meta = derived
        _apply_section(context, "derived", payload)
        context.meta["derived"] = meta

    if needs_rebuild:
        JobRepository(session).enqueue(
            job_type=JobType.CONTEXT_AGGREGATE,
            payload={"user_id": user_id},
            user_id=user_id,
            priority=50,
            idempotency_key=f"agg:{user_id}",
        )

    state.context = context
    return state


def _apply_section(context: UserContext, section: str, payload: dict) -> None:
    """Validate an aggregate blob into its typed section, tolerating drift."""
    if not payload:
        return
    from packages.domain.models import Activity, Derived, MedicalSnapshot, Nutrition, Vitals

    model = {
        "vitals": Vitals,
        "nutrition": Nutrition,
        "activity": Activity,
        "medical": MedicalSnapshot,
        "derived": Derived,
    }.get(section)
    if model is None:
        return
    try:
        setattr(context, section, model.model_validate(payload))
    except Exception as exc:
        # A stale aggregate written by older code should degrade the answer,
        # not fail the turn.
        logger.warning("aggregate for section %s failed validation: %s", section, exc)


def _build_live(health: HealthRepository, section: str) -> dict:
    builder = {
        "vitals": health.build_vitals,
        "nutrition": health.build_nutrition,
        "activity": health.build_activity,
        "medical": health.latest_medical,
    }.get(section)
    if builder is None:
        return {}
    try:
        return builder().model_dump(mode="json")
    except Exception:
        logger.exception("live build failed for section %s", section)
        return {}


def route(state: ConversationState, session: Session) -> ConversationState:
    outcome = Router(session).route(
        state.input.text,
        locale=state.request.locale,
        has_attachments=bool(state.input.attachments),
    )
    state.route = outcome.decision
    state.candidates.faq_hits = outcome.faq_hits
    state.flags.safety_flags.extend(outcome.safety_flags)
    return state


def cache_probe(state: ConversationState, session: Session) -> tuple[ConversationState, Optional[AnswerPayload]]:
    """arch.md 4.2: runs right after the router for every branch except
    PRODUCT/RESTAURANT with new attachments. A hit short-circuits to compose."""
    if state.route is None:
        return state, None
    if state.route.label in (RouteLabel.PRODUCT, RouteLabel.RESTAURANT) and state.input.attachments:
        return state, None
    if state.route.label is RouteLabel.UNSAFE:
        return state, None

    key = _cache_key_for(state)
    state.telemetry.cache_key = key.digest

    # L3 needs a query vector. Embedding is skipped for SMALLTALK — those hit
    # L1/L2 on the exact key anyway, and paying for an embedding to
    # semantically match "hi" against "hello" is pure waste.
    embedding = None
    if state.route.label is not RouteLabel.SMALLTALK:
        from packages.chains.embeddings import embed_query

        embedding = embed_query(state.input.text)

    lookup = CacheService(session).probe(
        key,
        embedding=embedding,
        route_label=str(state.route.label.value),
        locale=state.request.locale,
        safety_class=_safety_class_for(state),
    )
    state.telemetry.cache_status = lookup.tier
    if lookup.hit and lookup.payload is not None:
        return state, lookup.payload
    return state, None


def _cache_key_for(state: ConversationState):
    settings = get_settings()
    label = str(state.route.label.value) if state.route else "UNKNOWN"

    # arch.md 7.2: personalised answers are scoped to the user, so they can
    # never be served to anyone else. Only impersonal routes share globally.
    personal = state.route is not None and state.route.label in (
        RouteLabel.PERSONAL,
        RouteLabel.PRODUCT,
        RouteLabel.RESTAURANT,
    )
    faq_personalised = bool(
        state.candidates.faq_hits
        and state.candidates.faq_hits[0].item
        and state.candidates.faq_hits[0].item.required_slots
    )
    scope = state.request.user_id if (personal or faq_personalised) else "global"

    sections_used = state.context.sections_read() if state.context else []
    return build_cache_key(
        state.input.text,
        label,
        scope=scope,
        fingerprint=context_fingerprint(state.context, sections_used),
        locale=state.request.locale,
        prompt_version=settings.prompt_version,
        model_id=settings.models.large_model,
        kb_version=settings.kb_version,
    )


def _safety_class_for(state: ConversationState) -> SafetyClass:
    hit = state.candidates.faq_hits[0] if state.candidates.faq_hits else None
    if hit and hit.item:
        return hit.item.safety_class
    if state.route and state.route.category is FaqCategory.MEDICAL:
        return SafetyClass.MEDICAL_SENSITIVE
    return SafetyClass.INFORMATIONAL


# -- branch nodes: produce structured blocks, never user-facing prose -------


def safety_response(state: ConversationState, session: Session) -> ConversationState:
    blocking = next((f for f in state.flags.safety_flags if f.blocking), None)
    if blocking is None:
        return state
    state.draft.answer_blocks = [templates.safety_block(blocking.kind)]
    return state


def template_reply(state: ConversationState, session: Session) -> ConversationState:
    name = state.context.profile.display_name if state.context else None
    greeting = f"Hi {name.split()[0]}" if name else "Hi"
    state.draft.answer_blocks = [
        templates.text_block(
            "smalltalk_1",
            f"{greeting} — ask me about your training, nutrition, health data, "
            f"or scan a product label and I'll break down the ingredients.",
        )
    ]
    return state


def faq_answer(state: ConversationState, session: Session) -> ConversationState:
    """arch.md 5.3 fallback rule: FAQ is provisional. If required slots cannot
    be filled, fall through rather than emit a half-personalised template."""
    hit = state.candidates.faq_hits[0] if state.candidates.faq_hits else None
    if hit is None or hit.item is None:
        return personal_agent(state, session)

    item = hit.item
    gaps = templates.missing_slots(item, state.context)
    if gaps and not item.variants.without_data:
        logger.info("faq %s missing slots %s; falling through to personal", item.id, gaps)
        if state.route:
            state.route.rationale += f" | fell through: unfilled slots {gaps}"
        return personal_agent(state, session)

    text, data_gaps = templates.render_faq(item, state.context)
    state.draft.answer_blocks = [templates.faq_block("faq_1", text, item.id, item.version)]
    state.flags.data_gaps.extend(data_gaps)
    state.draft.disclaimers = templates.disclaimers_for(
        item.safety_class, has_data_gaps=bool(data_gaps)
    )
    return state


def personal_agent(state: ConversationState, session: Session) -> ConversationState:
    """The tool-using agent branch (arch.md 6.2).

    The model gets a context *summary* and reaches for detail through
    user-scoped tools. Every tool is bound to this caller's `user_id`, so it
    cannot address another user's rows whatever the model asks for.
    """
    from packages.chains.personal import PersonalAgent
    from packages.chains.providers import is_configured
    from packages.tools.health_tools import HealthToolset

    FaqRepository(session).record_unmatched(
        state.input.text,
        state.request.locale,
        route_label=str(state.route.label.value) if state.route else None,
    )

    context = state.context
    blocks: list[AnswerBlock] = []

    # The metric card is rendered from structured data regardless of whether
    # the model answers — the client can display it natively, and it survives a
    # provider outage.
    if context is not None and context.vitals.latest:
        blocks.append(
            AnswerBlock(
                block_id="metrics_1",
                type=BlockType.METRIC_CARD,
                text=None,
                data={
                    "metrics": [
                        {
                            "metric": m.metric,
                            "value": m.value,
                            "unit": m.unit,
                            "measured_on": str(m.measured_on) if m.measured_on else None,
                        }
                        for m in context.vitals.latest
                    ],
                    "trends": context.derived.trends,
                },
            )
        )

    if not is_configured():
        blocks.append(
            templates.text_block(
                "personal_unavailable_1",
                "I can see your data, but no language model is configured on this "
                "deployment, so I can't talk you through it yet.",
            )
        )
        state.draft.answer_blocks = blocks
        return state

    conversation = ConversationRepository(session)
    history = conversation.recent_turns(
        state.request.session_id, state.request.user_id, limit=6
    )
    memory_summary = conversation.get_summary(state.request.session_id, state.request.user_id)

    toolset = HealthToolset(session, state.request.user_id)
    result = PersonalAgent(toolset).run(
        state.input.text,
        context=context,
        history=[{"role": t["role"], "content": t["content"] or ""} for t in history],
        memory_summary=memory_summary,
    )

    for completion in result.completions:
        state.telemetry.token_costs.append(completion.to_cost("personal_agent"))

    if not result.ok:
        state.flags.degraded_sources.append("model_provider")
        blocks.append(
            templates.text_block(
                "personal_error_1",
                "I couldn't reach the reasoning service just now. Your data is above — "
                "try again in a moment.",
            )
        )
        state.draft.answer_blocks = blocks
        return state

    blocks.append(templates.text_block("personal_1", result.text))
    state.draft.answer_blocks = blocks

    # Tool outputs are the evidence guard_out verifies the prose against.
    state.candidates.cache_hits.append(
        {"kind": "tool_evidence", "evidence": result.evidence_strings()}
    )

    if result.tools_used:
        logger.info("personal agent used tools: %s", ", ".join(result.tools_used))

    # arch.md 12: the rolling summary updates off the critical path.
    JobRepository(session).enqueue(
        job_type=JobType.MEMORY_SUMMARISE,
        payload={"session_id": state.request.session_id, "user_id": state.request.user_id},
        user_id=state.request.user_id,
        priority=250,
        delay_seconds=5,
        idempotency_key=f"mem:{state.request.session_id}:{len(history)}",
    )
    return state


def evidence_pipeline(state: ConversationState, session: Session) -> ConversationState:
    """Answer from the stored evidence corpus when it covers the question;
    otherwise enqueue a deep-research job rather than fan out mid-turn
    (arch.md 9.4)."""
    from packages.chains.providers import is_configured
    from packages.evidence.synthesis import EvidenceSynthesiser
    from packages.storage.repositories.evidence import EvidenceRepository

    question = state.input.text
    have_evidence = bool(EvidenceRepository(session).search(question, limit=3))

    if have_evidence and is_configured():
        synthesiser = EvidenceSynthesiser(session)
        result = synthesiser.answer(question)
        for completion in synthesiser.costs:
            state.telemetry.token_costs.append(completion.to_cost("evidence_synthesise"))

        if result.ok:
            blocks = [templates.text_block("research_1", result.answer)]

            if result.disagreement:
                blocks.append(
                    AnswerBlock(
                        block_id="disagreement_1",
                        type=BlockType.TEXT,
                        text=f"Where the evidence disagrees: {result.disagreement}",
                        data={"consensus": result.consensus},
                    )
                )

            blocks.append(
                AnswerBlock(
                    block_id="evidence_1",
                    type=BlockType.EVIDENCE_LIST,
                    text=None,
                    data={
                        "consensus": result.consensus,
                        "sources": [
                            {
                                "id": c.citation_id,
                                "title": c.title,
                                "url": c.url,
                                "tier": str(c.tier.value),
                            }
                            for c in result.citations
                        ],
                    },
                )
            )

            state.draft.answer_blocks = blocks
            state.draft.citations = result.citations
            state.draft.disclaimers = templates.disclaimers_for(SafetyClass.GUIDANCE)

            for caveat in result.caveats:
                state.flags.data_gaps.append(
                    DataGap(field="evidence", why_it_matters=caveat)
                )
            return state

    # Not covered by the corpus: gather it in the background and say so.
    job_id = JobRepository(session).enqueue(
        job_type=JobType.DEEP_RESEARCH,
        payload={"question": question, "locale": state.request.locale},
        user_id=state.request.user_id,
        priority=120,
        idempotency_key=f"research:{normalise_question(question)[:150]}",
    )
    state.draft.answer_blocks = [
        templates.job_pending_block("research_1", [job_id], "that research question")
    ]
    return state


def product_analyzer(state: ConversationState, session: Session) -> ConversationState:
    """Run the scan pipeline and emit structured blocks.

    Everything here is deterministic. The LLM explanation layer (arch.md 8.6)
    attaches later and may only narrate these findings — it cannot add an
    ingredient, and `guard_out` re-checks that it did not.
    """
    from packages.product import ProductAnalyzer
    from packages.storage.blobs import BlobStore

    images = _load_attachment_bytes(state, session)
    if not images:
        state.draft.answer_blocks = [
            templates.product_unidentified_block(ProductUnidentifiedReason.UNREADABLE)
        ]
        state.draft.disclaimers = templates.disclaimers_for(
            SafetyClass.INFORMATIONAL, is_product=True
        )
        return state

    analyzer = ProductAnalyzer(
        session,
        user_id=state.request.user_id,
        jurisdiction=state.request.jurisdiction,
    )
    analysis, trace = analyzer.analyze(images, context=state.context)
    state.candidates.analyzer_result = analysis

    logger.info(
        "scan: %d ingredient(s), %d resolved, %d external call(s), %.0fms",
        len(analysis.ingredients),
        len(analysis.ingredients) - analysis.unresolved_count,
        trace.external_calls,
        trace.total_ms,
    )

    if analysis.unidentified_reason is not None:
        state.draft.answer_blocks = [
            templates.product_unidentified_block(analysis.unidentified_reason)
        ]
        state.draft.disclaimers = templates.disclaimers_for(
            SafetyClass.INFORMATIONAL, is_product=True
        )
        return state

    state.draft.answer_blocks = templates.product_blocks(analysis)
    state.draft.citations = templates.citations_for(analysis)
    state.draft.disclaimers = templates.disclaimers_for(
        SafetyClass.INFORMATIONAL,
        is_product=True,
        has_data_gaps=analysis.unresolved_count > 0,
    )

    # arch.md 8.6: the LLM's only job is the explanation layer. It narrates the
    # findings the rules produced; it cannot change the verdict, and _strip_
    # hallucinated + guard_out both re-check what it named.
    from packages.chains.providers import is_configured

    if is_configured() and analysis.hazards:
        from packages.chains.explain import ProductExplainer

        explainer = ProductExplainer()
        explanation = explainer.explain(analysis)
        for completion in explainer.costs:
            state.telemetry.token_costs.append(completion.to_cost("product_explain"))

        if explanation is not None:
            state.draft.answer_blocks.insert(
                0, templates.text_block("explain_1", explanation.overall)
            )
            if explanation.ingredient_notes:
                table = next(
                    (b for b in state.draft.answer_blocks
                     if b.type is BlockType.INGREDIENT_TABLE),
                    None,
                )
                if table is not None:
                    notes = {n.name.lower(): n.note for n in explanation.ingredient_notes}
                    for row in table.data.get("rows", []):
                        note = notes.get(str(row.get("name", "")).lower())
                        if note:
                            row["explanation"] = note

    if analysis.unresolved_count:
        state.flags.data_gaps.append(
            DataGap(
                field="ingredients",
                why_it_matters=(
                    f"{analysis.unresolved_count} of {len(analysis.ingredients)} ingredients "
                    "are not in the knowledge base yet, so they are not part of this assessment"
                ),
                how_to_supply="a clearer photo of the panel usually resolves misread names",
            )
        )
    return state


def _load_attachment_bytes(state: ConversationState, session: Session) -> list[bytes]:
    """Resolve attachment handles to bytes. Failures degrade, never raise."""
    from packages.storage.blobs import BlobStore

    store = BlobStore(session)
    images: list[bytes] = []

    for attachment in state.input.attachments:
        blob = store.get(attachment.attachment_id, user_id=state.request.user_id)
        if blob is None:
            logger.warning(
                "attachment %s not found for user %s",
                attachment.attachment_id,
                state.request.user_id,
            )
            state.flags.degraded_sources.append(f"attachment:{attachment.attachment_id}")
            continue
        try:
            images.append(blob.read())
        except OSError as exc:
            logger.warning("could not read blob %s: %s", blob.blob_id, exc)
            state.flags.degraded_sources.append(f"attachment:{attachment.attachment_id}")

    return images


def restaurant_analyzer(state: ConversationState, session: Session) -> ConversationState:
    job_id = JobRepository(session).enqueue(
        job_type=JobType.RESTAURANT_INVESTIGATION,
        payload={"query": state.input.text},
        user_id=state.request.user_id,
        priority=150,
    )
    state.draft.answer_blocks = [
        templates.job_pending_block("restaurant_1", [job_id], "that place")
    ]
    return state


# -- compose / guard_out / persist -----------------------------------------


def compose(state: ConversationState, session: Session) -> ConversationState:
    """Render blocks into a payload. Discovers no new facts."""
    payload = AnswerPayload(
        blocks=list(state.draft.answer_blocks),
        citations=list(state.draft.citations),
        disclaimers=list(state.draft.disclaimers),
        data_gaps=list(state.flags.data_gaps),
        confidence=state.route.confidence if state.route else 0.0,
    )

    if state.flags.data_gaps and payload.confidence > 0.5:
        payload.confidence = 0.5
        payload.confidence_reason = "some context sections were unavailable"

    payload.route_debug = {
        "label": str(state.route.label.value) if state.route else None,
        "stage": str(state.route.stage.value) if state.route else None,
        "rationale": state.route.rationale if state.route else None,
        "cache": str(state.telemetry.cache_status.value),
        "sections_read": state.context.sections_read() if state.context else [],
    }
    state.payload = payload
    return state


def guard_out(state: ConversationState, session: Session) -> ConversationState:
    """arch.md 4.3: verifies, never silently rewrites facts.

    The claim-vs-citation and allergen checks need the analyzer's structured
    findings; what is enforceable now is that a cited block exists for every
    citation and that nothing empty ships.
    """
    if state.payload is None:
        return state

    block_ids = {b.block_id for b in state.payload.blocks}
    for citation in state.payload.citations:
        dangling = [b for b in citation.supports_block_ids if b not in block_ids]
        if dangling:
            logger.warning("citation %s references missing blocks %s", citation.citation_id, dangling)
            citation.supports_block_ids = [
                b for b in citation.supports_block_ids if b in block_ids
            ]

    if not state.payload.blocks:
        state.payload.blocks = [
            templates.text_block(
                "empty_1", "I don't have an answer for that yet. Could you rephrase it?"
            )
        ]
        state.payload.confidence = 0.0
        state.payload.confidence_reason = "no blocks were produced"

    _verify_product_claims(state)
    _verify_entailment(state)
    return state


def _verify_entailment(state: ConversationState) -> None:
    """arch.md 4.4: check each claim against the spans cited for it.

    Unsupported sentences are not silently deleted — that would leave prose
    that no longer says what it set out to. They downgrade confidence and are
    recorded, so a persistently unfaithful branch shows up in the metrics
    rather than disappearing.
    """
    if state.payload is None or state.route is None:
        return

    # Only branches that fetched evidence can be checked against it.
    if state.route.label not in (RouteLabel.PERSONAL, RouteLabel.RESEARCH):
        return

    evidence: list[str] = []
    for candidate in state.candidates.cache_hits:
        if candidate.get("kind") == "tool_evidence":
            evidence.extend(candidate.get("evidence") or [])

    if not evidence:
        return

    prose = "\n".join(
        b.text for b in state.payload.blocks if b.text and b.type is BlockType.TEXT
    )
    if not prose.strip():
        return

    from packages.chains.verify import ClaimVerifier

    verifier = ClaimVerifier()
    report = verifier.verify(prose, evidence)
    for completion in verifier.costs:
        state.telemetry.token_costs.append(completion.to_cost("claim_verify"))

    if not report.checked:
        return

    penalty = report.confidence_penalty
    if penalty > 0:
        state.payload.confidence = round(max(0.0, state.payload.confidence * (1 - penalty)), 3)
        state.payload.confidence_reason = (
            f"{len(report.unsupported)} of "
            f"{len(report.supported) + len(report.partial) + len(report.unsupported)} "
            "statements were not supported by the data fetched"
        )

    if report.unsupported:
        logger.warning(
            "unsupported claims in turn %s: %s",
            state.request.turn_id,
            " | ".join(s[:80] for s in report.unsupported[:3]),
        )
        state.flags.safety_flags.append(
            SafetyFlag(
                kind=SafetyFlagKind.UNSUPPORTED_CLAIM,
                detail=f"{len(report.unsupported)} sentence(s) not entailed by fetched data",
                severity=Severity.MODERATE,
            )
        )


def _verify_product_claims(state: ConversationState) -> None:
    """arch.md 8.6: verify every ingredient the prose names appears in the
    structured findings, and that the verdict matches the enum the rules chose.

    This is the check that makes the LLM explanation layer safe to add. It runs
    now, before that layer exists, so the guarantee is in place first rather
    than retrofitted around a model that has already shipped.
    """
    analysis = state.candidates.analyzer_result
    if analysis is None or state.payload is None:
        return

    verdict_block = next(
        (b for b in state.payload.blocks if b.type is BlockType.HAZARD_BADGE), None
    )
    if verdict_block is not None:
        claimed = verdict_block.data.get("verdict")
        expected = str(analysis.verdict.value)
        if claimed != expected:
            logger.error(
                "guard_out: verdict %r does not match the rules' %r; correcting",
                claimed,
                expected,
            )
            verdict_block.data["verdict"] = expected
            state.flags.safety_flags.append(
                SafetyFlag(
                    kind=SafetyFlagKind.UNSUPPORTED_CLAIM,
                    detail=f"verdict was {claimed!r}, rules produced {expected!r}",
                    severity=Severity.HIGH,
                )
            )

    # Every name the prose uses must come from the panel we actually read.
    known = {
        (i.display_name or i.raw_token).lower()
        for i in analysis.ingredients
        if (i.display_name or i.raw_token)
    }
    table_block = next(
        (b for b in state.payload.blocks if b.type is BlockType.INGREDIENT_TABLE), None
    )
    if table_block is not None:
        for row in table_block.data.get("rows", []):
            name = str(row.get("name", "")).lower()
            if name and name not in known:
                logger.error("guard_out: ingredient table names unknown %r", name)
                state.flags.safety_flags.append(
                    SafetyFlag(
                        kind=SafetyFlagKind.UNSUPPORTED_CLAIM,
                        detail=f"ingredient {name!r} is not in the structured findings",
                        severity=Severity.HIGH,
                    )
                )


def persist(state: ConversationState, session: Session) -> ConversationState:
    """Cache, log the turn, write traces. Runs after the response is assembled."""
    if state.payload is None:
        return state

    settings = get_settings()

    # arch.md 7.3: a cache hit is not re-stored, and flagged turns never cache.
    if state.telemetry.cache_status is CacheTier.MISS and state.route is not None:
        if state.route.label is not RouteLabel.UNSAFE:
            key = _cache_key_for(state)
            category = state.route.category

            # Store the query vector alongside, so a later paraphrase can find
            # this entry through L3 (arch.md 7.1).
            embedding = None
            if state.route.label is not RouteLabel.SMALLTALK:
                from packages.chains.embeddings import embed_query

                embedding = embed_query(state.input.text)

            CacheService(session).store(
                key,
                state.payload,
                category=str(category.value) if category else None,
                route_label=str(state.route.label.value),
                locale=state.request.locale,
                embedding=embedding,
                model_id=settings.models.large_model,
                has_safety_flags=bool(state.flags.safety_flags),
            )

    conversation = ConversationRepository(session)
    conversation.append_turn(
        turn_id=state.request.turn_id,
        session_id=state.request.session_id,
        user_id=state.request.user_id,
        role="user",
        content=state.input.text,
    )
    conversation.append_turn(
        turn_id=f"{state.request.turn_id}:a",
        session_id=state.request.session_id,
        user_id=state.request.user_id,
        role="assistant",
        content=state.payload.rendered_text(),
        payload=state.payload.model_dump(mode="json", exclude={"route_debug"}),
        route_label=str(state.route.label.value) if state.route else None,
        route_stage=str(state.route.stage.value) if state.route else None,
        route_confidence=state.route.confidence if state.route else None,
        cache_tier=str(state.telemetry.cache_status.value),
        tokens_in=sum(c.prompt_tokens for c in state.telemetry.token_costs),
        tokens_out=sum(c.completion_tokens for c in state.telemetry.token_costs),
        cost_usd=state.telemetry.total_usd,
    )
    return state
