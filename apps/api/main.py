"""The API — one endpoint.

Everything goes through `POST /`, discriminated by `action` (default `chat`).
`GET /` returns status. This matches the shape the existing Flask app already
exposes, so clients keep the same URL, and it keeps cPanel's routing trivial:
there is exactly one path to map.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.actions import action_names, get_action
from apps.api.security import authenticate, enforce_rate_limit
from packages.config import get_settings
from packages.storage.db import ping, server_info, session_scope

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Health & Product Assistant",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-User-Id", "X-Admin-Token", "X-Request-Id"],
    max_age=600,
)

# JSON bodies carry text only — attachments arrive as multipart file parts, so
# 2 MB is generous for a conversation turn. It used to also have to hold
# base64-encoded images, which inflate by 4/3 and so capped a "max 8 MB" upload
# at roughly 1.5 MB of actual image.
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024

# Form fields that may carry a file part. `file`/`files` are the documented
# names; `attachments`/`images` are accepted because the old client used them.
_FILE_FIELDS = {"file", "files", "attachment", "attachments", "image", "images"}


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error [%s] %s %s", request_id, request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "internal error", "request_id": request_id},
        )

    response.headers["X-Request-Id"] = request_id
    logger.info(
        "%s %s -> %s in %.0fms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
        request_id,
    )
    return response


@app.get("/")
async def status_endpoint() -> dict[str, Any]:
    """Unauthenticated status. Reports configuration health without leaking
    values — a misconfigured deploy should be visible, not guessable."""
    db_ok = ping()
    info = server_info() if db_ok else {}
    problems = settings.validate()

    return {
        "success": True,
        "service": "Health & Product Assistant",
        "status": "ok" if db_ok and not problems else "degraded",
        "env": settings.env,
        "database": {
            "connected": db_ok,
            "server": info.get("version"),
            "name": info.get("database"),
        },
        "versions": {"prompt": settings.prompt_version, "kb": settings.kb_version},
        "providers": {
            "openai": bool(settings.models.openai_api_key),
            "deepseek": bool(settings.models.deepseek_api_key),
            "huggingface": bool(settings.models.hf_token),
        },
        "actions": action_names(),
        "config_problems": problems if settings.debug else len(problems),
    }


@app.options("/")
async def preflight() -> dict[str, Any]:
    return {"success": True}


async def _body_from_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if len(raw) > MAX_JSON_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"body exceeds {MAX_JSON_BODY_BYTES} bytes",
        )

    try:
        import json

        body = json.loads(raw) if raw else {}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


async def _body_from_multipart(request: Request) -> dict[str, Any]:
    """Build the action body from a multipart form.

    Images arrive as raw binary file parts. Structured parameters ride along
    either as ordinary form fields or as one `payload` field holding a JSON
    object, so a scan can still carry a message, jurisdiction and client hints
    without a second round trip.

    Every part is size-checked as it is read; the total is bounded by
    max_attachments x max_upload_bytes rather than one flat body limit.
    """
    import json

    storage = settings.storage

    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"malformed multipart body: {exc}") from exc

    body: dict[str, Any] = {}
    files: list[dict[str, Any]] = []

    try:
        for key, value in form.multi_items():
            if hasattr(value, "read"):  # an UploadFile
                if key not in _FILE_FIELDS:
                    continue
                if len(files) >= storage.max_attachments:
                    raise HTTPException(
                        status_code=400,
                        detail=f"too many attachments (max {storage.max_attachments})",
                    )
                raw = await value.read()
                if not raw:
                    continue
                if len(raw) > storage.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"'{value.filename or key}' is {len(raw)} bytes; "
                            f"the limit is {storage.max_upload_bytes}"
                        ),
                    )
                files.append(
                    {
                        "bytes": raw,
                        "mime_type": value.content_type or None,
                        "filename": value.filename or None,
                    }
                )
            elif key == "payload":
                try:
                    parsed = json.loads(value)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400, detail="'payload' must be a JSON object"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise HTTPException(status_code=400, detail="'payload' must be a JSON object")
                body.update(parsed)
            else:
                # A plain form field. Explicit fields win over `payload` so a
                # client can override one value without rebuilding the JSON.
                body[key] = value
    finally:
        await form.close()

    if files:
        body["attachments"] = files
        # Sending files at all means the caller wants them handled; `chat`
        # would silently ignore them if no action was named.
        body.setdefault("action", "scan")

    return body


@app.post("/")
async def unified_endpoint(request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")

    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "multipart/form-data":
        body = await _body_from_multipart(request)
    else:
        body = await _body_from_json(request)

    action_name = str(body.get("action") or "chat")
    handler = get_action(action_name)
    if handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown action '{action_name}'. Known: {', '.join(action_names())}",
        )

    principal = authenticate(request, body)

    with session_scope() as session:
        enforce_rate_limit(request, principal, session)
        result = handler(body, principal, session)

    return JSONResponse(
        content={"success": True, "request_id": request_id, **result},
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
        },
        headers=exc.headers or {},
    )
