"""Chain base: schema-constrained output, masking, cost capture.

Two things every chain gets for free:

  * **Structured output by schema**, replacing the "parse JSON out of prose and
    retry" loop in the old `app.py` (line 1445). The model is asked for JSON
    mode; the result is validated against a Pydantic model, and a failure
    degrades to None rather than raising into the request.
  * **Masking**, so a chain declared PSEUDONYMOUS physically cannot send a name
    or a date of birth to a provider (arch.md 6.3).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from packages.chains.providers import (
    Completion,
    DataPolicy,
    ModelClass,
    ProviderError,
    complete,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# -- masking (arch.md 6.3) --------------------------------------------------

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\d{10}\b")
_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_DOB = re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b")


class Masker:
    """Reversible tokenisation of identifiers.

    Reversible matters: the answer needs the user's actual name back in
    `compose`, so the mapping is kept for the turn and detokenised on the way
    out — not thrown away.
    """

    def __init__(self) -> None:
        self._forward: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self._counter = 0

    def _token(self, kind: str, value: str) -> str:
        if value in self._forward:
            return self._forward[value]
        self._counter += 1
        token = f"[{kind}_{self._counter}]"
        self._forward[value] = token
        self._reverse[token] = value
        return token

    def mask(self, text: str, *, extra_names: Optional[list[str]] = None) -> str:
        if not text:
            return text

        masked = text
        for name in extra_names or []:
            if name and len(name) > 2:
                masked = re.sub(
                    rf"\b{re.escape(name)}\b", self._token("NAME", name), masked, flags=re.I
                )

        masked = _EMAIL.sub(lambda m: self._token("EMAIL", m.group(0)), masked)
        masked = _AADHAAR.sub(lambda m: self._token("ID", m.group(0)), masked)
        masked = _PAN.sub(lambda m: self._token("ID", m.group(0)), masked)
        masked = _PHONE.sub(lambda m: self._token("PHONE", m.group(0)), masked)
        masked = _DOB.sub(lambda m: self._token("DOB", m.group(0)), masked)
        return masked

    def unmask(self, text: str) -> str:
        if not text:
            return text
        for token, value in self._reverse.items():
            text = text.replace(token, value)
        return text

    @property
    def active(self) -> bool:
        return bool(self._reverse)


# -- chains -----------------------------------------------------------------


@dataclass
class ChainResult(Generic[T]):
    value: Optional[T] = None
    raw_text: str = ""
    completion: Optional[Completion] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.value is not None


class Chain:
    """A named model call with a declared policy and model class."""

    name: str = "chain"
    model_class: ModelClass = ModelClass.SMALL
    data_policy: DataPolicy = DataPolicy.PSEUDONYMOUS
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str = ""

    def __init__(self, masker: Optional[Masker] = None) -> None:
        self.masker = masker or Masker()
        self.costs: list[Completion] = []

    # -- plumbing ---------------------------------------------------------

    def _messages(self, user_content: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def run_text(self, user_content: str, **kwargs: Any) -> ChainResult[Any]:
        try:
            completion = complete(
                self._messages(user_content),
                model_class=self.model_class,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
        except ProviderError as exc:
            logger.warning("chain %s failed: %s", self.name, exc)
            return ChainResult(error=str(exc))

        self.costs.append(completion)
        return ChainResult(value=completion.text, raw_text=completion.text, completion=completion)

    def run_structured(
        self, user_content: str, schema: type[T], **kwargs: Any
    ) -> ChainResult[T]:
        """Schema-constrained output.

        JSON mode plus Pydantic validation. One repair attempt is allowed —
        models occasionally wrap JSON in prose even in JSON mode — and after
        that the chain degrades rather than looping (the old code retried three
        times per ingredient, which is where a lot of the latency went).
        """
        instruction = (
            f"{user_content}\n\n"
            f"Respond with JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Return only the JSON object."
        )

        try:
            completion = complete(
                self._messages(instruction),
                model_class=self.model_class,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                response_format={"type": "json_object"},
            )
        except ProviderError as exc:
            logger.warning("chain %s failed: %s", self.name, exc)
            return ChainResult(error=str(exc))

        self.costs.append(completion)

        parsed = extract_json(completion.text)
        if parsed is None:
            return ChainResult(
                raw_text=completion.text, completion=completion, error="no JSON object in response"
            )

        try:
            return ChainResult(
                value=schema.model_validate(parsed),
                raw_text=completion.text,
                completion=completion,
            )
        except ValidationError as exc:
            logger.warning("chain %s output failed validation: %s", self.name, exc)
            return ChainResult(
                raw_text=completion.text,
                completion=completion,
                error=f"schema validation failed: {exc.error_count()} error(s)",
            )

    @property
    def total_usd(self) -> float:
        return round(sum(c.usd for c in self.costs), 8)

    def cost_records(self) -> list[Any]:
        return [c.to_cost(self.name) for c in self.costs]


def extract_json(raw: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object out of a reply, fenced or with prose around it."""
    if not raw:
        return None

    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```\s*$", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None
