"""The cache key.

arch.md 7.2: the key is the whole design. Everything that could change the
answer is in it, and `context_fingerprint` covers only the sections the answer
actually read — which is what makes caching a personalised answer safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from packages.common.text import normalise_question, sha256_hex
from packages.domain.models import UserContext


def context_fingerprint(
    context: Optional[UserContext], sections_used: Optional[list[str]] = None
) -> str:
    """Version-stamp of the context sections that contributed.

    A nutrition answer reads the nutrition section, so a workout sync bumps a
    version this fingerprint never looked at, and the entry survives.
    """
    if context is None:
        return "nocontext"

    parts = context.fingerprint_parts()
    if sections_used is not None:
        wanted = set(sections_used)
        parts = {k: v for k, v in parts.items() if k in wanted}

    if not parts:
        return "empty"
    return sha256_hex(*(f"{k}={v}" for k, v in sorted(parts.items())))[:32]


@dataclass(frozen=True)
class CacheKey:
    digest: str
    scope: str
    normalised_question: str
    fingerprint: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.digest


def build_cache_key(
    question: str,
    route_label: str,
    *,
    scope: str = "global",
    fingerprint: str = "nocontext",
    locale: str = "en",
    prompt_version: str = "v1",
    model_id: str = "",
    kb_version: str = "v1",
) -> CacheKey:
    """`scope` is 'global' or the user id — a personalised answer can never be
    served to another user because its scope is part of the key."""
    normalised = normalise_question(question)
    digest = sha256_hex(
        normalised,
        route_label,
        scope,
        fingerprint,
        locale,
        prompt_version,
        model_id,
        kb_version,
    )
    return CacheKey(
        digest=digest,
        scope=scope,
        normalised_question=normalised,
        fingerprint=fingerprint,
    )
