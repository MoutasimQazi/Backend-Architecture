"""Job handlers.

Everything unbounded runs here rather than inside an HTTP request (arch.md
principle 3): chemical research, deep research, restaurant investigation,
memory summarisation, aggregate rebuilds, embedding backfills and the sweeps.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from packages.domain.enums import JobType
from packages.jobs.registry import JobContext, handler
from packages.storage.repositories.cache import CacheRepository
from packages.storage.repositories.health import HealthRepository
from packages.storage.repositories.jobs import JobRepository
from packages.storage.repositories.ratelimit import RateLimitRepository

logger = logging.getLogger(__name__)


@handler(JobType.CONTEXT_AGGREGATE)
def rebuild_user_aggregates(ctx: JobContext) -> dict[str, Any]:
    """arch.md 6.1: rolling windows are maintained by a job, not per request.

    Writing a section bumps its `version`, which is exactly what busts the
    cache entries that read that section — and nothing else.
    """
    user_id = ctx.payload.get("user_id") or ctx.user_id
    if not user_id:
        raise ValueError("context_aggregate requires user_id")

    repo = HealthRepository(ctx.session, user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    written: list[str] = []

    vitals = repo.build_vitals()
    repo.put_aggregate(
        "vitals",
        vitals.model_dump(mode="json"),
        completeness=1.0 if vitals.latest else 0.0,
        fresh_as_of=now,
    )
    written.append("vitals")

    nutrition = repo.build_nutrition()
    repo.put_aggregate(
        "nutrition",
        nutrition.model_dump(mode="json"),
        completeness=1.0 if nutrition.latest_day else 0.0,
        fresh_as_of=now,
    )
    written.append("nutrition")

    activity = repo.build_activity()
    repo.put_aggregate(
        "activity",
        activity.model_dump(mode="json"),
        completeness=1.0 if activity.recent else 0.0,
        fresh_as_of=now,
    )
    written.append("activity")

    medical = repo.latest_medical()
    repo.put_aggregate(
        "medical",
        medical.model_dump(mode="json"),
        completeness=1.0 if medical.report_date else 0.0,
        fresh_as_of=now,
    )
    written.append("medical")

    derived = repo.build_derived()
    repo.put_aggregate("derived", derived.model_dump(mode="json"), completeness=1.0, fresh_as_of=now)
    written.append("derived")

    # The user's cached answers were built on the old aggregates.
    CacheRepository(ctx.session).invalidate_scope(user_id)

    return {"user_id": user_id, "sections": written}


@handler(JobType.MEMORY_SUMMARISE)
def summarise_memory(ctx: JobContext) -> dict[str, Any]:
    """arch.md 12: the rolling summary and structured facts update off the
    critical path, after the turn has already been answered."""
    from packages.chains.memory import FactExtractor, MemorySummariser
    from packages.chains.providers import is_configured
    from packages.storage.repositories.conversation import ConversationRepository

    session_id = ctx.require("session_id")
    user_id = ctx.payload.get("user_id") or ctx.user_id
    if not user_id:
        raise ValueError("memory_summarise requires user_id")

    if not is_configured():
        return {"status": "skipped", "reason": "no model provider configured"}

    repo = ConversationRepository(ctx.session)
    turns = repo.recent_turns(session_id, user_id, limit=20)
    if not turns:
        return {"status": "skipped", "reason": "no turns"}

    previous = repo.get_summary(session_id, user_id)
    summary = MemorySummariser().summarise(
        [{"role": t["role"], "content": t["content"] or ""} for t in turns], previous
    )
    if summary and summary != previous:
        repo.put_summary(session_id, user_id, summary, turn_count=len(turns))

    # Structured facts are queryable and user-deletable; the prose summary is not.
    facts = FactExtractor().extract(
        [{"role": t["role"], "content": t["content"] or ""} for t in turns]
    )
    for fact in facts:
        repo.remember(user_id, fact.kind, fact.value, confidence=fact.confidence)

    return {
        "session_id": session_id,
        "summary_updated": bool(summary and summary != previous),
        "facts_extracted": len(facts),
    }


@handler(JobType.PROFILE_CAPTURE)
def capture_profile(ctx: JobContext) -> dict[str, Any]:
    """Store health, fitness and preference facts the user stated in chat.

    Runs off the critical path: the turn is already answered by the time this
    executes, so the user never waits on bookkeeping. Idempotent per turn.

    Consent is checked per destination, not once at the top — someone may have
    granted PROFILE but not VITALS, and that must mean "remember my allergy,
    do not record my weight" rather than all-or-nothing.
    """
    from packages.chains.capture import HealthCaptureChain, normalise_metric
    from packages.chains.providers import is_configured
    from packages.domain.enums import ConsentScope
    from packages.storage.repositories.conversation import ConversationRepository
    from packages.storage.repositories.health import HealthRepository
    from packages.storage.repositories.users import UserRepository

    user_id = ctx.payload.get("user_id") or ctx.user_id
    if not user_id:
        raise ValueError("profile_capture requires user_id")

    if not is_configured():
        return {"status": "skipped", "reason": "no model provider configured"}

    consent = UserRepository(ctx.session).get_consent(user_id)
    granted = set(consent.granted_scopes)
    may_write_vitals = ConsentScope.VITALS in granted
    may_write_profile = ConsentScope.PROFILE in granted

    if not (may_write_vitals or may_write_profile):
        return {"status": "skipped", "reason": "no consent for vitals or profile"}

    session_id = ctx.payload.get("session_id")
    messages = ctx.payload.get("messages")
    if not messages and session_id:
        turns = ConversationRepository(ctx.session).recent_turns(session_id, user_id, limit=8)
        messages = [{"role": t["role"], "content": t["content"] or ""} for t in turns]
    if not messages:
        return {"status": "skipped", "reason": "no messages"}

    captured = HealthCaptureChain().capture(messages)

    health = HealthRepository(ctx.session, user_id)
    conversation = ConversationRepository(ctx.session)
    written = {"metrics": 0, "preferences": 0, "dropped": 0}
    added_medical: dict[str, list[str]] = {}

    if may_write_vitals:
        for metric in captured.metrics:
            if metric.confidence < 0.6:
                written["dropped"] += 1
                continue
            normalised = normalise_metric(metric.metric, metric.value, metric.unit)
            if normalised is None:
                written["dropped"] += 1
                continue
            value, unit = normalised
            health.record_metric(metric.metric, value, unit, source="chat")
            written["metrics"] += 1

    if may_write_profile:
        medical = captured.medical
        added_medical = health.merge_medical(
            allergies=medical.allergies,
            conditions=medical.conditions,
            medications=medical.medications,
            source="chat",
        )
        if medical.pregnancy_status and medical.pregnancy_status != "none":
            UserRepository(ctx.session).upsert_profile(
                user_id, pregnancy_status=medical.pregnancy_status
            )

        for preference in captured.preferences:
            if preference.confidence < 0.6:
                written["dropped"] += 1
                continue
            conversation.remember(
                user_id,
                preference.kind,
                preference.value,
                confidence=preference.confidence,
                source_turn_id=ctx.payload.get("turn_id"),
            )
            written["preferences"] += 1

    # What the assistant knows about this user changed, so anything it said
    # earlier from the old context may no longer be the right answer.
    if written["metrics"] or written["preferences"] or any(added_medical.values()):
        CacheRepository(ctx.session).invalidate_scope(user_id)

    return {
        "user_id": user_id,
        "metrics": written["metrics"],
        "preferences": written["preferences"],
        "dropped": written["dropped"],
        "medical_added": {k: v for k, v in added_medical.items() if v},
    }


@handler(JobType.CHEMICAL_RESEARCH)
def research_chemical(ctx: JobContext) -> dict[str, Any]:
    """arch.md 8.4: an unknown ingredient enqueues here rather than blocking
    the scan. The dossier lands as `draft` for review before it is published."""
    from packages.etl.chemical import ChemicalEtl

    token = ctx.require("token")
    outcome = ChemicalEtl(ctx.session).ingest(str(token), with_evidence=True)

    if not outcome.ok:
        # Not raised: "PubChem does not know this ingredient" is a legitimate
        # result, not a failure to retry three times.
        return {"token": token, "status": "not_found", "reason": outcome.error}

    return {
        "token": token,
        "status": "drafted",
        "chemical_id": outcome.chemical_id,
        "created": outcome.created,
        "synonyms": outcome.synonyms_added,
        "assertions": outcome.assertions_added,
        "evidence": outcome.evidence_added,
        "external_calls": outcome.external_calls,
        "notes": outcome.notes,
        "review_required": True,
    }


@handler(JobType.ETL_CHEMICAL_KB)
def etl_chemical_kb(ctx: JobContext) -> dict[str, Any]:
    """Bulk ingest. Payload: {"names": [...]}."""
    from packages.etl.chemical import ChemicalEtl

    names = ctx.payload.get("names") or []
    if not isinstance(names, list) or not names:
        raise ValueError("etl_chemical_kb requires a non-empty 'names' list")

    etl = ChemicalEtl(ctx.session)
    ingested, failed, calls = [], [], 0

    # Bounded per job: a 5,000-name list is many jobs, not one that runs for
    # hours and dies holding a lease.
    for name in names[:50]:
        outcome = etl.ingest(str(name), with_evidence=ctx.payload.get("with_evidence", True))
        calls += outcome.external_calls
        (ingested if outcome.ok else failed).append(
            outcome.chemical_id if outcome.ok else {"name": name, "error": outcome.error}
        )

    return {
        "ingested": ingested,
        "failed": failed,
        "external_calls": calls,
        "remaining": max(0, len(names) - 50),
    }


@handler(JobType.DEEP_RESEARCH)
def deep_research(ctx: JobContext) -> dict[str, Any]:
    """arch.md 9.4: broad questions exceeding the source budget become a job.

    Gathers literature, scores independence, stores it, then synthesises — so
    the next person asking the same thing is answered from the store.
    """
    from packages.connectors.literature import EuropePMCConnector
    from packages.domain.enums import SourceTier
    from packages.evidence.independence import score_independence
    from packages.evidence.synthesis import EvidenceSynthesiser
    from packages.storage.repositories.evidence import EvidenceRepository

    question = ctx.require("question")
    repo = EvidenceRepository(ctx.session)

    papers = EuropePMCConnector().search(str(question), limit=12)
    stored = 0
    for paper in papers:
        score = score_independence(paper.grants)
        repo.upsert_document(
            paper.source_id,
            title=paper.title,
            container=paper.container,
            url=paper.url,
            tier=SourceTier.T3_PRIMARY,
            year=paper.year,
            study_design=paper.study_design,
            funder_class=str(score.funder_class.value),
            independence=score.value,
            abstract=paper.abstract,
        )
        if paper.abstract:
            repo.add_chunk(paper.source_id, 0, paper.abstract)
        stored += 1

    synthesis = EvidenceSynthesiser(ctx.session).answer(str(question))

    return {
        "question": question,
        "sources_stored": stored,
        "sources_used": synthesis.sources_used,
        "answer": synthesis.answer,
        "consensus": synthesis.consensus,
        "disagreement": synthesis.disagreement,
        "caveats": synthesis.caveats,
        "citations": [c.model_dump(mode="json") for c in synthesis.citations],
        "error": synthesis.error,
    }


@handler(JobType.RESTAURANT_INVESTIGATION)
def investigate_restaurant(ctx: JobContext) -> dict[str, Any]:
    """arch.md 11.3: async from day one, with progressive results."""
    from packages.restaurant.analyzer import RestaurantAnalyzer

    query = ctx.payload.get("query") or ctx.payload.get("place_id")
    if not query:
        raise ValueError("restaurant_investigation requires a query or place_id")

    report = RestaurantAnalyzer().analyze(str(query), place_id=ctx.payload.get("place_id"))

    return {
        "query": query,
        "place_resolved": report.place.resolved,
        "summary": report.summary_line(),
        "findings": report.to_blocks(),
        "no_adverse_findings": report.no_adverse_findings,
        "stages_completed": report.stages_completed,
        "stages_unavailable": report.stages_unavailable,
    }


@handler(JobType.OCR)
def run_ocr(ctx: JobContext) -> dict[str, Any]:
    """OCR one stored blob, populating the hash-keyed cache."""
    from packages.product.ocr import OcrService
    from packages.storage.blobs import BlobStore

    blob_id = ctx.require("blob_id")
    blob = BlobStore(ctx.session).get(str(blob_id))
    if blob is None:
        return {"blob_id": blob_id, "status": "not_found"}

    result = OcrService(ctx.session).read(blob.read())
    return {
        "blob_id": blob_id,
        "status": "ok" if result.ok else "failed",
        "cached": result.cached,
        "chars": len(result.text),
        "error": result.error,
    }


@handler(JobType.EMBEDDING_BACKFILL)
def backfill_embeddings(ctx: JobContext) -> dict[str, Any]:
    """Embed whatever still lacks a vector.

    Re-queues itself while work remains, so one cron tick eventually embeds a
    whole corpus without any single job holding a lease for hours.
    """
    from packages.chains.embeddings import (
        backfill_chemical_synonyms,
        backfill_evidence_chunks,
        backfill_faq_surfaces,
    )
    from packages.chains.providers import is_configured

    if not is_configured():
        return {"status": "skipped", "reason": "no model provider configured"}

    results = {
        "faq": backfill_faq_surfaces(ctx.session),
        "chemicals": backfill_chemical_synonyms(ctx.session),
        "evidence": backfill_evidence_chunks(ctx.session),
    }

    remaining = sum(r["pending"] for r in results.values())
    if remaining:
        JobRepository(ctx.session).enqueue(
            job_type=JobType.EMBEDDING_BACKFILL,
            payload={},
            priority=300,
            delay_seconds=30,
        )

    return {**results, "remaining": remaining}


def run_maintenance(session) -> dict[str, int]:
    """Housekeeping the worker runs on a slow tick.

    With no Redis there is nothing that expires on its own, so expiry has to be
    swept: cache TTLs, rate-limit windows, finished jobs, orphaned blobs, and
    leases left behind by a killed process.
    """
    from packages.storage.blobs import BlobStore

    cache_repo = CacheRepository(session)
    job_repo = JobRepository(session)
    rate_repo = RateLimitRepository(session)

    stats = {
        "cache_expired": cache_repo.purge_expired(),
        "cache_pruned": cache_repo.prune_cold(),
        "rate_limits_purged": rate_repo.purge_expired(),
        "leases_reaped": job_repo.reap_expired_leases(),
    }

    try:
        stats["blobs_purged"] = BlobStore(session).purge_expired()
    except Exception:
        logger.exception("blob purge failed; continuing")
        stats["blobs_purged"] = 0

    return stats
