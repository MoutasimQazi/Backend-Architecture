"""Authentication and rate limiting for the single endpoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from packages.config import get_settings
from packages.storage.repositories.ratelimit import RateLimitRepository

logger = logging.getLogger(__name__)

try:
    import jwt

    _HAS_JWT = True
except ImportError:  # pragma: no cover - PyJWT is in requirements
    jwt = None  # type: ignore[assignment]
    _HAS_JWT = False


@dataclass(frozen=True)
class Principal:
    user_id: str
    scopes: tuple[str, ...] = ()
    is_admin: bool = False
    auth_method: str = "jwt"


def _client_ip(request: Request) -> str:
    # cPanel fronts the app with Apache, so the socket peer is always
    # localhost; the real client is in the forwarding header.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def authenticate(request: Request, body: dict) -> Principal:
    """Bearer JWT, or the header escape hatch when explicitly enabled.

    ALLOW_HEADER_AUTH exists so the endpoint can be exercised before the mobile
    app issues tokens. `Settings.validate()` refuses to call a production config
    healthy while it is on.
    """
    settings = get_settings()

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        return _decode_jwt(token)

    if settings.security.allow_header_auth:
        user_id = request.headers.get("x-user-id") or body.get("user_id")
        if user_id:
            admin_token = request.headers.get("x-admin-token", "")
            is_admin = bool(
                settings.security.jwt_secret and admin_token == settings.security.jwt_secret
            )
            return Principal(user_id=str(user_id), auth_method="header", is_admin=is_admin)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode_jwt(token: str) -> Principal:
    settings = get_settings()

    if not _HAS_JWT:
        raise HTTPException(status_code=500, detail="PyJWT is not installed")
    if not settings.security.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured")

    try:
        claims = jwt.decode(
            token,
            settings.security.jwt_secret,
            algorithms=[settings.security.jwt_algorithm],
            audience=settings.security.jwt_audience or None,
            options={"verify_aud": bool(settings.security.jwt_audience)},
        )
    except Exception as exc:
        # The reason is logged, never returned: it tells an attacker which part
        # of the token to fix.
        logger.info("jwt rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="token has no subject")

    raw_scopes = claims.get("scopes") or claims.get("scope") or []
    if isinstance(raw_scopes, str):
        raw_scopes = raw_scopes.split()

    return Principal(
        user_id=str(user_id),
        scopes=tuple(str(s) for s in raw_scopes),
        is_admin=bool(claims.get("admin")) or "admin" in raw_scopes,
        auth_method="jwt",
    )


def enforce_rate_limit(request: Request, principal: Principal, session: Session) -> None:
    """Per-user and per-IP, both windows. Fail-open on a limiter error: the
    limiter protecting the service must not become the thing that breaks it."""
    settings = get_settings().security
    repo = RateLimitRepository(session)

    try:
        per_minute = repo.check_minute(f"user:{principal.user_id}", settings.rate_limit_per_minute)
        if not per_minute.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(per_minute.retry_after_seconds)},
            )

        per_day = repo.check_day(f"user:{principal.user_id}", settings.rate_limit_per_day)
        if not per_day.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="daily quota exceeded",
                headers={"Retry-After": str(per_day.retry_after_seconds)},
            )

        ip = _client_ip(request)
        if ip != "unknown":
            per_ip = repo.check_minute(f"ip:{ip}", settings.rate_limit_per_minute * 3)
            if not per_ip.allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="rate limit exceeded",
                    headers={"Retry-After": str(per_ip.retry_after_seconds)},
                )
    except HTTPException:
        raise
    except Exception:
        logger.exception("rate limiter error; allowing the request")


def require_admin(principal: Principal) -> None:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="admin scope required")
