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

MAX_BODY_BYTES = 2 * 1024 * 1024


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


@app.post("/")
async def unified_endpoint(request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")

    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"body exceeds {MAX_BODY_BYTES} bytes",
        )

    try:
        import json

        body = json.loads(raw) if raw else {}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

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
