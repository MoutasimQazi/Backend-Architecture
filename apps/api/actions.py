"""Action handlers behind the single endpoint.

One HTTP route, several actions. Each handler takes the parsed body, the
principal and a session, and returns a JSON-serialisable dict. Nothing here
holds business logic — it adapts the envelope to the packages that do.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Optional

from fastapi import HTTPException
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from apps.api.security import Principal, require_admin
from packages.cache import CacheService
from packages.config import get_settings
from packages.domain.enums import ConsentScope, JobType
from packages.domain.models import Attachment
from packages.orchestrator import new_state, run_turn, total_latency_ms
from packages.storage.repositories.cache import CacheRepository
from packages.storage.repositories.conversation import ConversationRepository
from packages.storage.repositories.faq import FaqRepository
from packages.storage.repositories.health import HealthRepository
from packages.storage.repositories.jobs import JobRepository
from packages.storage.repositories.users import UserRepository

logger = logging.getLogger(__name__)

Handler = Callable[[dict, Principal, Session], dict[str, Any]]
_actions: dict[str, Handler] = {}


def action(name: str) -> Callable[[Handler], Handler]:
    def decorator(func: Handler) -> Handler:
        _actions[name] = func
        return func

    return decorator


def get_action(name: str) -> Optional[Handler]:
    return _actions.get(name)


def action_names() -> list[str]:
    return sorted(_actions)


def _parse_date(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field}: expected ISO date") from exc


# --------------------------------------------------------------------------
# chat — the default action
# --------------------------------------------------------------------------


@action("chat")
def handle_chat(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    # Accept the legacy shape too: the old client sent {messages:[...], context:{...}}.
    # arch.md 15 phase 1 keeps that path accepted-but-ignored until traffic moves.
    message = body.get("message")
    if message is None:
        legacy = body.get("messages") or []
        if legacy and isinstance(legacy, list):
            last = legacy[-1]
            if isinstance(last, dict):
                message = last.get("content")
    if message is None:
        message = ""

    if not isinstance(message, str):
        raise HTTPException(status_code=400, detail="message must be a string")
    if len(message) > 8000:
        raise HTTPException(status_code=400, detail="message too long (max 8000 chars)")

    settings = get_settings()
    raw_attachments = body.get("attachments") or []
    if len(raw_attachments) > settings.storage.max_attachments:
        raise HTTPException(
            status_code=400,
            detail=f"too many attachments (max {settings.storage.max_attachments})",
        )

    raw_attachments = _materialise_attachments(raw_attachments, principal, session)

    attachments: list[Attachment] = []
    for item in raw_attachments:
        if not isinstance(item, dict):
            continue
        try:
            attachments.append(Attachment.model_validate(item))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid attachment: {exc}") from exc

    state = new_state(
        user_id=principal.user_id,
        message=message,
        session_id=body.get("session_id"),
        attachments=attachments,
        locale=body.get("locale") or "en",
        jurisdiction=body.get("jurisdiction") or "IN",
        client_version=body.get("client_version"),
        client_hints=body.get("client_hints") or {},
    )

    state = run_turn(state, session)
    payload = state.payload
    if payload is None:
        raise HTTPException(status_code=500, detail="no payload produced")

    pending_jobs = [
        job_id
        for block in payload.blocks
        for job_id in (block.data.get("job_ids") or [])
        if block.data
    ]

    response: dict[str, Any] = {
        "action": "chat",
        "turn_id": state.request.turn_id,
        "session_id": state.request.session_id,
        "message": payload.rendered_text(),
        "payload": payload.model_dump(mode="json", exclude={"route_debug"}),
        "route": {
            "label": str(state.route.label.value) if state.route else None,
            "confidence": state.route.confidence if state.route else 0.0,
            "stage": str(state.route.stage.value) if state.route else None,
        },
        "cache": str(state.telemetry.cache_status.value),
        "pending_jobs": pending_jobs,
        "latency_ms": total_latency_ms(state),
    }

    # route_debug is internal (arch.md 10); it ships only when the caller is an
    # admin and asked for it.
    if body.get("debug") and principal.is_admin:
        response["route_debug"] = payload.route_debug
        response["timings"] = [t.model_dump(mode="json") for t in state.telemetry.node_timings]

    return response


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


@action("job_status")
def handle_job_status(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    repo = JobRepository(session)
    job_id = body.get("job_id")

    if job_id:
        job = repo.get(str(job_id), user_id=principal.user_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"action": "job_status", "job": job.model_dump(mode="json")}

    jobs = repo.list_for_user(principal.user_id, limit=int(body.get("limit") or 20))
    return {"action": "job_status", "jobs": [j.model_dump(mode="json") for j in jobs]}


@action("job_enqueue")
def handle_job_enqueue(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    raw_type = body.get("job_type")
    try:
        job_type = JobType(str(raw_type))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown job_type '{raw_type}'") from exc

    # Only jobs a user may legitimately trigger for themselves. ETL and
    # maintenance types are operator-only.
    if job_type in (JobType.ETL_CHEMICAL_KB,) and not principal.is_admin:
        raise HTTPException(status_code=403, detail="admin scope required for this job type")

    job_id = JobRepository(session).enqueue(
        job_type=job_type,
        payload=body.get("payload") or {},
        user_id=principal.user_id,
        priority=int(body.get("priority") or 100),
        idempotency_key=body.get("idempotency_key"),
    )
    return {"action": "job_enqueue", "job_id": job_id, "status": "queued"}


# --------------------------------------------------------------------------
# sync — the data plane that replaces csv_health_data in the payload
# --------------------------------------------------------------------------


@action("sync")
def handle_sync(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    """arch.md 3: the app writes health data here instead of shipping the whole
    dataset on every chat turn."""
    user_id = principal.user_id
    users = UserRepository(session)
    users.upsert(user_id, locale=body.get("locale") or "en")

    written: dict[str, int] = {}

    profile = body.get("profile")
    if isinstance(profile, dict):
        users.upsert_profile(user_id, **profile)
        written["profile"] = 1

    for metric in body.get("metrics") or []:
        if not isinstance(metric, dict):
            continue
        session.execute(
            sql_text(
                """
                INSERT INTO health_metric (user_id, metric, measured_on, value, unit, source)
                VALUES (:uid, :m, :d, :v, :u, :src)
                ON DUPLICATE KEY UPDATE value = VALUES(value), unit = VALUES(unit)
                """
            ),
            {
                "uid": user_id,
                "m": str(metric.get("metric"))[:64],
                "d": _parse_date(metric.get("measured_on"), "metrics[].measured_on"),
                "v": metric.get("value"),
                "u": metric.get("unit"),
                "src": metric.get("source"),
            },
        )
        written["metrics"] = written.get("metrics", 0) + 1

    for day in body.get("nutrition") or []:
        if not isinstance(day, dict):
            continue
        session.execute(
            sql_text(
                """
                INSERT INTO nutrition_day
                    (user_id, consumed_on, calories, protein_g, carbs_g, fat_g,
                     fiber_g, sugar_g, sodium_mg, water_ml, source)
                VALUES (:uid, :d, :cal, :p, :c, :f, :fb, :s, :so, :w, :src)
                ON DUPLICATE KEY UPDATE
                    calories = VALUES(calories), protein_g = VALUES(protein_g),
                    carbs_g = VALUES(carbs_g), fat_g = VALUES(fat_g),
                    fiber_g = VALUES(fiber_g), sugar_g = VALUES(sugar_g),
                    sodium_mg = VALUES(sodium_mg), water_ml = VALUES(water_ml)
                """
            ),
            {
                "uid": user_id,
                "d": _parse_date(day.get("consumed_on"), "nutrition[].consumed_on"),
                "cal": day.get("calories"),
                "p": day.get("protein_g"),
                "c": day.get("carbs_g"),
                "f": day.get("fat_g"),
                "fb": day.get("fiber_g"),
                "s": day.get("sugar_g"),
                "so": day.get("sodium_mg"),
                "w": day.get("water_ml"),
                "src": day.get("source"),
            },
        )
        written["nutrition"] = written.get("nutrition", 0) + 1

    for act in body.get("activities") or []:
        if not isinstance(act, dict):
            continue
        session.execute(
            sql_text(
                """
                INSERT INTO activity_session
                    (user_id, activity_type, started_at, duration_min, distance_m,
                     calories, load_score, source)
                VALUES (:uid, :t, :start, :dur, :dist, :cal, :load, :src)
                """
            ),
            {
                "uid": user_id,
                "t": str(act.get("activity_type") or "unknown")[:64],
                "start": act.get("started_at"),
                "dur": act.get("duration_min"),
                "dist": act.get("distance_m"),
                "cal": act.get("calories"),
                "load": act.get("load"),
                "src": act.get("source"),
            },
        )
        written["activities"] = written.get("activities", 0) + 1

    medical = body.get("medical")
    if isinstance(medical, dict):
        import json as _json

        session.execute(
            sql_text(
                """
                INSERT INTO medical_report
                    (user_id, report_date, bmi, systolic, diastolic, hba1c,
                     lipids, labs, flags, conditions, allergies, medications)
                VALUES (:uid, :d, :bmi, :sys, :dia, :hba1c,
                        :lipids, :labs, :flags, :cond, :allerg, :meds)
                """
            ),
            {
                "uid": user_id,
                "d": _parse_date(medical.get("report_date"), "medical.report_date"),
                "bmi": medical.get("bmi"),
                "sys": medical.get("systolic"),
                "dia": medical.get("diastolic"),
                "hba1c": medical.get("hba1c"),
                "lipids": _json.dumps(medical.get("lipids") or {}),
                "labs": _json.dumps(medical.get("labs") or {}),
                "flags": _json.dumps(medical.get("flags") or []),
                "cond": _json.dumps(medical.get("conditions") or []),
                "allerg": _json.dumps(medical.get("allergies") or []),
                "meds": _json.dumps(medical.get("medications") or []),
            },
        )
        written["medical"] = 1

    # arch.md 7.3 event-driven bust: new data invalidates this user's answers,
    # and the aggregates that feed the fingerprint get rebuilt off the request.
    CacheService(session).invalidate_user(user_id)
    job_id = JobRepository(session).enqueue(
        job_type=JobType.CONTEXT_AGGREGATE,
        payload={"user_id": user_id},
        user_id=user_id,
        priority=20,
        idempotency_key=f"agg:{user_id}",
    )

    return {"action": "sync", "written": written, "aggregate_job_id": job_id}


# --------------------------------------------------------------------------
# consent, memory
# --------------------------------------------------------------------------


@action("upload")
def handle_upload(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    """Store attachments and return handles.

    Bytes enter here and nowhere else. The chat turn then carries handles, so a
    retry costs no re-upload.

    Two shapes arrive here. A multipart request has already been read into
    `{"bytes": ...}` items by the endpoint — that is the normal path for
    images. A JSON request carries `{"data": "https://..."}`, which is
    resolved through the fetch broker. Base64 is not accepted in either.
    """
    from packages.storage.blobs import BlobStore, decode_attachment_payload

    settings = get_settings()
    raw_items = body.get("attachments") or body.get("images") or []
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    if not isinstance(raw_items, list) or not raw_items:
        raise HTTPException(status_code=400, detail="attachments must be a non-empty list")
    if len(raw_items) > settings.storage.max_attachments:
        raise HTTPException(
            status_code=400,
            detail=f"too many attachments (max {settings.storage.max_attachments})",
        )

    store = BlobStore(session)
    stored: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, item in enumerate(raw_items):
        declared = item.get("mime_type") if isinstance(item, dict) else None

        try:
            if isinstance(item, dict) and isinstance(item.get("bytes"), (bytes, bytearray)):
                # Multipart part: already bytes, nothing to decode.
                raw, sniffed_mime = bytes(item["bytes"]), None
            else:
                payload = item.get("data") if isinstance(item, dict) else item
                raw, sniffed_mime = decode_attachment_payload(payload)
            blob = store.put(
                raw,
                user_id=principal.user_id,
                declared_mime=declared or sniffed_mime,
                kind="upload",
            )
        except ValueError as exc:
            # One bad attachment must not sink the others.
            errors.append({"index": index, "error": str(exc)})
            continue

        stored.append(
            {
                "attachment_id": blob.blob_id,
                "mime_type": blob.mime_type,
                "size_bytes": blob.size_bytes,
                "sha256": blob.sha256,
                "deduplicated": blob.deduplicated,
            }
        )

    if not stored:
        raise HTTPException(
            status_code=400,
            detail=f"no attachment could be stored: {errors[0]['error'] if errors else 'unknown'}",
        )

    return {"action": "upload", "attachments": stored, "errors": errors}


def _materialise_attachments(
    raw_attachments: list[Any], principal: Principal, session: Session
) -> list[Any]:
    """Turn raw bytes or URLs into stored handles.

    A multipart request delivers `{"bytes": ...}` items and a JSON request may
    deliver a URL string; the graph only ever works with `attachment_id`
    handles. Doing the conversion here means `chat` and `scan` accept exactly
    the same shapes — previously only `scan` did, so posting an image to
    `chat` failed validation with a raw pydantic error.
    """
    if not raw_attachments:
        return raw_attachments

    already_handles = all(
        isinstance(a, dict) and a.get("attachment_id") for a in raw_attachments
    )
    if already_handles:
        return raw_attachments

    uploaded = handle_upload({"attachments": raw_attachments}, principal, session)
    return [
        {"attachment_id": a["attachment_id"], "mime_type": a["mime_type"]}
        for a in uploaded["attachments"]
    ]


@action("scan")
def handle_scan(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    """Upload and analyse in one call — the mobile scan path.

    `chat` with attachments does the same thing through the graph; this exists
    because the scanner screen wants the structured analysis without a
    conversational turn wrapped around it.
    """
    result = handle_chat({**body, "action": "chat", "message": body.get("message") or ""}, principal, session)
    result["action"] = "scan"
    return result


@action("consent")
def handle_consent(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    users = UserRepository(session)

    # Every user-scoped table has a foreign key to app_user, and until now that
    # row was only created by the chat path and by `sync`. A freshly issued
    # token calling `consent` first — which is the natural onboarding order,
    # since consent should precede sending any data — failed on the foreign
    # key. Ensuring the row here makes consent-before-data work.
    users.upsert(
        principal.user_id,
        locale=body.get("locale") or "en",
        jurisdiction=body.get("jurisdiction") or "IN",
    )

    op = str(body.get("op") or "list")

    if op == "list":
        state = users.get_consent(principal.user_id)
        return {
            "action": "consent",
            "granted": [str(s.value) for s in state.granted_scopes],
            "available": [str(s.value) for s in ConsentScope],
        }

    raw_scopes = body.get("scopes") or []
    if isinstance(raw_scopes, str):
        raw_scopes = [raw_scopes]

    scopes: list[ConsentScope] = []
    for raw in raw_scopes:
        try:
            scopes.append(ConsentScope(str(raw)))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown consent scope '{raw}'") from exc

    if op == "grant":
        for scope in scopes:
            users.grant(principal.user_id, scope)
    elif op == "revoke":
        for scope in scopes:
            users.revoke(principal.user_id, scope)
    else:
        raise HTTPException(status_code=400, detail="op must be list, grant or revoke")

    # What the assistant may see changed, so what it previously said may no
    # longer be permissible to repeat.
    CacheService(session).invalidate_user(principal.user_id)
    state = users.get_consent(principal.user_id)
    return {"action": "consent", "granted": [str(s.value) for s in state.granted_scopes]}


@action("memory")
def handle_memory(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    """arch.md 12: memory is user-visible and user-editable."""
    repo = ConversationRepository(session)
    op = str(body.get("op") or "list")

    if op == "list":
        return {"action": "memory", "memories": repo.recall(principal.user_id, body.get("kind"))}

    if op == "forget":
        memory_id = body.get("memory_id")
        if memory_id is None:
            raise HTTPException(status_code=400, detail="memory_id is required")
        removed = repo.forget(principal.user_id, int(memory_id))
        if removed:
            CacheService(session).invalidate_user(principal.user_id)
        return {"action": "memory", "forgotten": removed}

    if op == "remember":
        kind, value = body.get("kind"), body.get("value")
        if not kind or not value:
            raise HTTPException(status_code=400, detail="kind and value are required")
        repo.remember(principal.user_id, str(kind), str(value))
        CacheService(session).invalidate_user(principal.user_id)
        return {"action": "memory", "stored": True}

    raise HTTPException(status_code=400, detail="op must be list, remember or forget")


@action("history")
def handle_history(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    turns = ConversationRepository(session).recent_turns(
        str(session_id), principal.user_id, limit=int(body.get("limit") or 20)
    )
    return {
        "action": "history",
        "turns": [
            {"role": t["role"], "content": t["content"], "created_at": str(t["created_at"])}
            for t in turns
        ],
    }


@action("context")
def handle_context(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    """What the assistant can currently see about you. Transparency, and the
    fastest way to debug a wrong personalised answer."""
    repo = HealthRepository(session, principal.user_id)
    users = UserRepository(session)
    consent = users.get_consent(principal.user_id)
    profile, version = users.get_profile(principal.user_id)

    return {
        "action": "context",
        "profile": profile.model_dump(mode="json"),
        "profile_version": version,
        "consent": [str(s.value) for s in consent.granted_scopes],
        "aggregate_versions": repo.aggregate_versions(),
        "latest_metrics": [m.model_dump(mode="json") for m in repo.latest_metrics()],
    }


# --------------------------------------------------------------------------
# admin
# --------------------------------------------------------------------------


@action("admin.cache")
def handle_admin_cache(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    require_admin(principal)
    service = CacheService(session)
    op = str(body.get("op") or "stats")

    if op == "stats":
        stats = service.stats()
        return {"action": "admin.cache", "l1": stats.l1, "l2": stats.l2}
    if op == "purge_expired":
        return {"action": "admin.cache", "result": service.maintenance()}
    if op == "invalidate_user":
        target = body.get("user_id")
        if not target:
            raise HTTPException(status_code=400, detail="user_id is required")
        return {"action": "admin.cache", "removed": service.invalidate_user(str(target))}
    if op == "invalidate_versions":
        return {"action": "admin.cache", "removed": service.invalidate_versions()}
    if op == "invalidate_all":
        removed = CacheRepository(session).invalidate_by_version(prompt_version="__none__")
        service.l1.clear()
        return {"action": "admin.cache", "removed": removed}

    raise HTTPException(status_code=400, detail=f"unknown cache op '{op}'")


@action("admin.faq")
def handle_admin_faq(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    """Authoring lives in the admin app (arch.md 5.1); this is its backend."""
    require_admin(principal)
    repo = FaqRepository(session)
    op = str(body.get("op") or "list")

    if op == "upsert":
        from packages.domain.models import FaqItem

        raw = body.get("item")
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="item object is required")
        paraphrases = raw.pop("paraphrases", [])
        try:
            item = FaqItem.model_validate(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid FaqItem: {exc}") from exc
        item.paraphrases = list(paraphrases)

        repo.upsert_item(item)
        surfaces = repo.replace_surfaces(item.id, item)
        CacheService(session).invalidate_category(str(item.category.value))
        return {"action": "admin.faq", "id": item.id, "surfaces": surfaces}

    if op == "get":
        faq_id = body.get("id")
        if not faq_id:
            raise HTTPException(status_code=400, detail="id is required")
        item = repo.get(str(faq_id))
        if item is None:
            raise HTTPException(status_code=404, detail="faq not found")
        return {"action": "admin.faq", "item": item.model_dump(mode="json")}

    if op == "unmatched":
        return {
            "action": "admin.faq",
            "unmatched": [
                {**row, "last_seen": str(row["last_seen"])}
                for row in repo.top_unmatched(
                    min_hits=int(body.get("min_hits") or 3),
                    limit=int(body.get("limit") or 50),
                )
            ],
        }

    raise HTTPException(status_code=400, detail=f"unknown faq op '{op}'")


@action("admin.jobs")
def handle_admin_jobs(body: dict, principal: Principal, session: Session) -> dict[str, Any]:
    require_admin(principal)
    repo = JobRepository(session)
    op = str(body.get("op") or "depth")

    if op == "depth":
        return {"action": "admin.jobs", "depth": repo.queue_depth()}
    if op == "reap":
        return {"action": "admin.jobs", "reaped": repo.reap_expired_leases()}

    raise HTTPException(status_code=400, detail=f"unknown jobs op '{op}'")
