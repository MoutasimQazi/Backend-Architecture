"""The conversation graph, run as a sequence.

arch.md 4.2 draws this as a LangGraph `StateGraph`. The edges here are the same
edges; the sequencer is a for-loop because LangGraph's value on this host is
checkpointing and streaming, neither of which the Phase 0/1 nodes use yet.
Swapping it in is a change to this file only — `nodes.py` already has the
signature a graph node needs.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from packages.domain.enums import CacheTier, RouteLabel
from packages.domain.models import (
    Attachment,
    ConversationState,
    InputData,
    NodeTiming,
    RequestInfo,
)
from packages.orchestrator import nodes
from packages.storage.repositories.conversation import TraceRepository

logger = logging.getLogger(__name__)

Node = Callable[[ConversationState, Session], ConversationState]

# arch.md 4.2: the branch each route label runs.
BRANCHES: dict[RouteLabel, Node] = {
    RouteLabel.SMALLTALK: nodes.template_reply,
    RouteLabel.FAQ: nodes.faq_answer,
    RouteLabel.CACHED: nodes.faq_answer,
    RouteLabel.PERSONAL: nodes.personal_agent,
    RouteLabel.RESEARCH: nodes.evidence_pipeline,
    RouteLabel.PRODUCT: nodes.product_analyzer,
    RouteLabel.RESTAURANT: nodes.restaurant_analyzer,
    RouteLabel.UNSAFE: nodes.safety_response,
}


def _timed(state: ConversationState, name: str, fn: Callable[[], None]) -> None:
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    ok, error = True, None
    try:
        fn()
    except Exception as exc:
        ok, error = False, f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.telemetry.node_timings.append(
            NodeTiming(
                node=name,
                started_at=started,
                duration_ms=round((time.perf_counter() - clock) * 1000, 2),
                ok=ok,
                error=error,
            )
        )


def new_state(
    user_id: str,
    message: str,
    *,
    session_id: Optional[str] = None,
    attachments: Optional[list[Attachment]] = None,
    locale: str = "en",
    jurisdiction: str = "IN",
    client_version: Optional[str] = None,
    client_hints: Optional[dict] = None,
) -> ConversationState:
    return ConversationState(
        request=RequestInfo(
            user_id=user_id,
            session_id=session_id or uuid.uuid4().hex,
            turn_id=uuid.uuid4().hex,
            locale=locale,
            jurisdiction=jurisdiction,
            client_version=client_version,
        ),
        input=InputData(
            text=message,
            attachments=attachments or [],
            client_hints=client_hints or {},
        ),
    )


def run_turn(state: ConversationState, session: Session) -> ConversationState:
    """ingest -> guard_in -> context_build -> router -> [cache probe] ->
    branch -> compose -> guard_out -> persist."""

    _timed(state, "ingest", lambda: nodes.ingest(state, session))
    _timed(state, "guard_in", lambda: nodes.guard_in(state, session))

    # guard_in's triage happens inside the router's detectors; a blocking flag
    # from a previous node still short-circuits here.
    if state.blocked:
        _timed(state, "safety_response", lambda: nodes.safety_response(state, session))
        _timed(state, "compose", lambda: nodes.compose(state, session))
        _timed(state, "guard_out", lambda: nodes.guard_out(state, session))
        _flush_trace(state, session)
        return state

    _timed(state, "context_build", lambda: nodes.context_build(state, session))
    _timed(state, "router", lambda: nodes.route(state, session))

    if state.blocked:
        _timed(state, "safety_response", lambda: nodes.safety_response(state, session))
        _timed(state, "compose", lambda: nodes.compose(state, session))
        _timed(state, "guard_out", lambda: nodes.guard_out(state, session))
        _timed(state, "persist", lambda: nodes.persist(state, session))
        _flush_trace(state, session)
        return state

    cached_payload = None

    def probe() -> None:
        nonlocal cached_payload
        _, cached_payload = nodes.cache_probe(state, session)

    _timed(state, "cache_probe", probe)

    if cached_payload is not None:
        # A hit short-circuits straight to compose (arch.md 4.2).
        state.draft.answer_blocks = list(cached_payload.blocks)
        state.draft.citations = list(cached_payload.citations)
        state.draft.disclaimers = list(cached_payload.disclaimers)
        _timed(state, "compose", lambda: nodes.compose(state, session))
        _timed(state, "guard_out", lambda: nodes.guard_out(state, session))
        _timed(state, "persist", lambda: nodes.persist(state, session))
        _flush_trace(state, session)
        return state

    label = state.route.label if state.route else RouteLabel.PERSONAL
    branch = BRANCHES.get(label, nodes.personal_agent)
    _timed(state, f"branch:{label.value}", lambda: branch(state, session))

    _timed(state, "compose", lambda: nodes.compose(state, session))
    _timed(state, "guard_out", lambda: nodes.guard_out(state, session))
    _timed(state, "persist", lambda: nodes.persist(state, session))
    _flush_trace(state, session)
    return state


def _flush_trace(state: ConversationState, session: Session) -> None:
    """One trace per turn (arch.md 14). Never allowed to sink the response."""
    try:
        TraceRepository(session).record(
            state.request.turn_id,
            state.telemetry.node_timings,
            attributes={
                "route": str(state.route.label.value) if state.route else None,
                "cache": str(state.telemetry.cache_status.value),
                "cost_usd": state.telemetry.total_usd,
                "tokens": state.telemetry.total_tokens,
            },
        )
    except Exception:
        logger.exception("failed to write trace for turn %s", state.request.turn_id)


def total_latency_ms(state: ConversationState) -> float:
    return round(sum(t.duration_ms for t in state.telemetry.node_timings), 2)
