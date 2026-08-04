"""Model provider abstraction (arch.md 13).

Small models do small jobs. Routing, classification, extraction and reranking
run on the cheap model; only final synthesis touches the large one. Each chain
declares what it needs — tool calling, structured output, and crucially whether
it may see lab values at all.

DeepSeek and the HuggingFace router are OpenAI-compatible, so one client class
covers all three; only the base URL and key differ.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from packages.config import get_settings
from packages.domain.models import TokenCost

logger = logging.getLogger(__name__)


class ModelClass(str, Enum):
    SMALL = "small"      # routing, classification, extraction — latency-critical
    LARGE = "large"      # final synthesis and explanation only
    EMBEDDING = "embedding"


class DataPolicy(str, Enum):
    """What a chain is allowed to send to a provider (arch.md 6.3).

    Declared per chain rather than checked ad hoc, so "can this prompt contain
    lab values?" has one answer that lives next to the chain, not in the caller.
    """

    PUBLIC = "public"          # no user data at all
    PSEUDONYMOUS = "pseudo"    # user data, identifiers masked
    SENSITIVE = "sensitive"    # may include labs; requires a zero-retention endpoint


# USD per 1M tokens. Only used for cost accounting, so approximate is fine —
# the point is spotting drift, not billing.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # DeepSeek standard-hours, cache-miss rates. Cache hits bill lower, so a
    # cost reported from this table is a ceiling, never an understatement.
    # deepseek-chat and deepseek-reasoner were retired; the account now serves
    # only the v4 line. Their rates stay here so historical token_cost rows
    # still price correctly when replayed.
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}

# Models that spend completion tokens on hidden reasoning before emitting any
# content. Measured on deepseek-v4-flash: a router classification burns 46-221
# reasoning tokens, and when max_tokens runs out during that phase the reply
# comes back with finish_reason="length" and an EMPTY string — not an error.
#
# That failure is silent and total: the router's configured budget of 200 was
# below the measured worst case of 288, so the highest-volume chain in the
# system would have returned nothing on exactly the ambiguous messages it
# exists to classify. Budgets are therefore given reasoning headroom here
# rather than being re-tuned by hand at every call site.
_REASONING_MODELS = {"deepseek-v4-flash"}
_REASONING_HEADROOM = 512
_MIN_REASONING_BUDGET = 768

# deepseek-v4-pro is deliberately not used by this deployment. It is blocked
# rather than merely left unconfigured so that a stray env var or an explicit
# `model=` argument cannot route spend to it.
_BLOCKED_MODELS = {"deepseek-v4-pro"}

_unpriced_warned: set[str] = set()


class ProviderError(Exception):
    """Provider call failed after retries. Callers degrade; they never crash."""


@dataclass
class Completion:
    text: str = ""
    parsed: Optional[dict[str, Any]] = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    # Normalised to plain dicts so callers never touch the SDK's own types —
    # that is what keeps the provider swappable.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def usd(self) -> float:
        if self.model not in PRICING and self.model not in _unpriced_warned:
            # Reporting $0.00 for a model that is answering real traffic makes
            # every cost figure in the system wrong in the reassuring
            # direction. Say so once, loudly, rather than quietly under-count.
            _unpriced_warned.add(self.model)
            logger.warning(
                "no PRICING entry for '%s'; its cost is reported as $0.00. "
                "Add the per-1M input/output rates to PRICING.",
                self.model,
            )
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        return round(
            (self.prompt_tokens * rate_in + self.completion_tokens * rate_out) / 1_000_000, 8
        )

    def to_cost(self, node: str) -> TokenCost:
        return TokenCost(
            node=node,
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            usd=self.usd,
        )


@dataclass
class _CircuitBreaker:
    """Stops retry storms against a provider that is already down (arch.md 13)."""

    failures: int = 0
    opened_at: float = 0.0
    threshold: int = 5
    cooldown_seconds: int = 60
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self.failures < self.threshold:
                return False
            if time.time() - self.opened_at > self.cooldown_seconds:
                # Half-open: let one request through to test recovery.
                self.failures = self.threshold - 1
                return False
            return True

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.time()
                logger.error("circuit breaker opened after %d failures", self.failures)

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0


_breakers: dict[str, _CircuitBreaker] = {}
_clients: dict[str, Any] = {}
_lock = threading.Lock()


def _breaker(provider: str) -> _CircuitBreaker:
    with _lock:
        if provider not in _breakers:
            _breakers[provider] = _CircuitBreaker()
        return _breakers[provider]


def _client(provider: str) -> Any:
    """Lazily build an OpenAI-compatible client for the provider."""
    with _lock:
        if provider in _clients:
            return _clients[provider]

    settings = get_settings().models

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - openai is in requirements
        raise ProviderError("the openai package is not installed") from exc

    if provider == "openai":
        key, base_url = settings.openai_api_key, None
    elif provider == "deepseek":
        key, base_url = settings.deepseek_api_key, settings.deepseek_base_url
    elif provider == "huggingface":
        key, base_url = settings.hf_token, settings.hf_base_url
    else:
        raise ProviderError(f"unknown provider '{provider}'")

    if not key:
        raise ProviderError(f"no API key configured for '{provider}'")

    client = OpenAI(
        api_key=key,
        base_url=base_url,
        timeout=settings.request_timeout,
        max_retries=0,  # retries are handled here, with the breaker
    )
    with _lock:
        _clients[provider] = client
    return client


def available_providers() -> list[str]:
    """Configured providers in preference order (PROVIDER_ORDER).

    The order is the caller's stated preference, filtered by which keys are
    actually present. It used to be hardcoded openai-first, which meant a host
    holding both an OpenAI and a DeepSeek key sent everything to OpenAI and
    only ever reached DeepSeek when OpenAI was failing.
    """
    settings = get_settings().models
    keys = {
        "openai": settings.openai_api_key,
        "deepseek": settings.deepseek_api_key,
        "huggingface": settings.hf_token,
    }
    ordered = [p for p in settings.provider_order if keys.get(p)]

    # A key that is set but not named in PROVIDER_ORDER still belongs in the
    # fallback chain — dropping it would turn a typo into an outage.
    ordered += [p for p, key in keys.items() if key and p not in ordered]
    return ordered


def model_for(model_class: ModelClass, provider: Optional[str] = None) -> str:
    """The model name to send to `provider` for this class of work.

    SMALL_MODEL/LARGE_MODEL name OpenAI models. DeepSeek and HuggingFace serve
    different names, so each has its own pair; without that mapping the small/
    large split collapses and routing pays synthesis prices (or the call 404s
    on an unknown model name).
    """
    settings = get_settings().models

    if model_class is ModelClass.EMBEDDING:
        return settings.embedding_model

    table = {
        "deepseek": (settings.deepseek_small_model, settings.deepseek_large_model),
        "huggingface": (settings.hf_small_model, settings.hf_large_model),
    }
    small, large = table.get(provider or "openai", (settings.small_model, settings.large_model))
    return small if model_class is ModelClass.SMALL else large


def _resolve_model(provider: str, chosen: str) -> str:
    """Final say on which model name goes on the wire.

    Enforces the blocklist here, at the one place every call funnels through,
    so no env var, config typo or explicit `model=` can route spend to a model
    this deployment has ruled out.
    """
    if chosen not in _BLOCKED_MODELS:
        return chosen

    substitute = model_for(ModelClass.SMALL, provider)
    if substitute in _BLOCKED_MODELS:
        raise ProviderError(
            f"'{chosen}' is blocked and provider '{provider}' has no permitted substitute"
        )
    logger.warning("model '%s' is blocked for this deployment; using %s", chosen, substitute)
    return substitute


def _budget_for(model_name: str, requested: int) -> int:
    """Add reasoning headroom so the answer is not eaten by the thinking.

    A reasoning model charges its hidden reasoning against the same completion
    budget as the visible reply. Callers size max_tokens for the reply they
    want, which is the right thing for them to reason about, so the headroom
    is added here instead of inflating every call site.
    """
    if model_name not in _REASONING_MODELS:
        return requested
    return max(requested + _REASONING_HEADROOM, _MIN_REASONING_BUDGET)


def is_configured() -> bool:
    return bool(available_providers())


def complete(
    messages: list[dict[str, Any]],
    *,
    model_class: ModelClass = ModelClass.SMALL,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    provider: Optional[str] = None,
) -> Completion:
    """One completion, with retries, fallback and a circuit breaker.

    Providers are tried in order; a provider whose breaker is open is skipped
    entirely rather than waiting for another timeout.
    """
    settings = get_settings().models
    chain = [provider] if provider else available_providers()
    if not chain:
        raise ProviderError("no model provider is configured")

    last_error: Optional[Exception] = None

    for candidate in chain:
        breaker = _breaker(candidate)
        if breaker.is_open:
            logger.warning("skipping %s: circuit breaker open", candidate)
            continue

        # An explicit `model=` is the caller pinning one exact model, so it
        # wins. Otherwise each provider is asked for its own name for this
        # class of work — the small/large split has to survive a failover.
        provider_model = _resolve_model(candidate, model or model_for(model_class, candidate))
        provider_budget = _budget_for(provider_model, max_tokens)

        for attempt in range(settings.max_retries + 1):
            started = time.perf_counter()
            try:
                client = _client(candidate)
                kwargs: dict[str, Any] = {
                    "model": provider_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": provider_budget,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                usage = getattr(response, "usage", None)

                # A reasoning model that exhausts its budget while thinking
                # returns finish_reason="length" with empty content and no
                # error. Treating that as a valid empty answer is how a router
                # silently stops routing, so it is raised and retried with a
                # bigger budget instead.
                if (
                    not (choice.message.content or "").strip()
                    and not getattr(choice.message, "tool_calls", None)
                    and choice.finish_reason == "length"
                ):
                    provider_budget = max(provider_budget * 2, _MIN_REASONING_BUDGET * 2)
                    raise ProviderError(
                        f"{provider_model} exhausted its token budget before producing "
                        f"content; retrying with max_tokens={provider_budget}"
                    )

                calls = []
                for call in getattr(choice.message, "tool_calls", None) or []:
                    calls.append(
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments or "{}",
                            },
                        }
                    )

                breaker.record_success()
                return Completion(
                    text=choice.message.content or "",
                    model=provider_model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    finish_reason=choice.finish_reason,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    tool_calls=calls,
                )

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "%s attempt %d/%d failed: %s",
                    candidate,
                    attempt + 1,
                    settings.max_retries + 1,
                    exc,
                )
                if attempt < settings.max_retries:
                    time.sleep(min(2**attempt, 8))

        breaker.record_failure()

    raise ProviderError(f"all providers failed; last error: {last_error}")


def embed(texts: list[str], *, model: Optional[str] = None) -> list[list[float]]:
    """Batch embeddings. Order matches the input."""
    if not texts:
        return []

    settings = get_settings().models
    resolved = model or settings.embedding_model

    # Only OpenAI serves the embedding models this system is configured for.
    if not settings.openai_api_key:
        raise ProviderError("embeddings require OPENAI_API_KEY")

    breaker = _breaker("openai-embed")
    if breaker.is_open:
        raise ProviderError("embedding circuit breaker is open")

    for attempt in range(settings.max_retries + 1):
        try:
            client = _client("openai")
            response = client.embeddings.create(model=resolved, input=texts)
            breaker.record_success()
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.warning("embedding attempt %d failed: %s", attempt + 1, exc)
            if attempt < settings.max_retries:
                time.sleep(min(2**attempt, 8))
            else:
                breaker.record_failure()
                raise ProviderError(f"embedding failed: {exc}") from exc

    raise ProviderError("embedding failed")
