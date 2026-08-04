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
    "deepseek-chat": (0.27, 1.10),
}


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
    settings = get_settings().models
    out = []
    if settings.openai_api_key:
        out.append("openai")
    if settings.deepseek_api_key:
        out.append("deepseek")
    if settings.hf_token:
        out.append("huggingface")
    return out


def model_for(model_class: ModelClass) -> str:
    settings = get_settings().models
    return {
        ModelClass.SMALL: settings.small_model,
        ModelClass.LARGE: settings.large_model,
        ModelClass.EMBEDDING: settings.embedding_model,
    }[model_class]


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

    resolved_model = model or model_for(model_class)
    last_error: Optional[Exception] = None

    for candidate in chain:
        breaker = _breaker(candidate)
        if breaker.is_open:
            logger.warning("skipping %s: circuit breaker open", candidate)
            continue

        # DeepSeek and HF do not serve OpenAI's model names.
        provider_model = resolved_model
        if candidate == "deepseek":
            provider_model = "deepseek-chat"
        elif candidate == "huggingface":
            provider_model = get_settings().models.small_model

        for attempt in range(settings.max_retries + 1):
            started = time.perf_counter()
            try:
                client = _client(candidate)
                kwargs: dict[str, Any] = {
                    "model": provider_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                usage = getattr(response, "usage", None)

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
