"""OCR as its own step, cached on image hash.

arch.md 8.2: "one pass per image, results stored and reused everywhere (the
current double-OCR disappears because OCR is its own step with its own cache
keyed on image hash)".

In the old `app.py` the same photo was OCR'd by the classifier and again by the
chat path. Keying the cache on sha256 makes the second call free, and makes a
re-scan of a product the user already photographed free too.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.guards.fetch_broker import FetchError, get_broker

logger = logging.getLogger(__name__)

OCR_CACHE_TTL_DAYS = 90


@dataclass
class OcrResult:
    text: str = ""
    confidence: Optional[float] = None
    engine: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None
    image_sha256: str = ""
    regions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# The ingredient panel is what we want handed to the parser, not the whole
# pack (arch.md 8.2). Without layout boxes from the OCR service, the panel is
# located in the text: from the header to the next section that clearly isn't
# ingredients.
_PANEL_START = re.compile(
    r"(?:^|\n)\s*(ingredients?|composition|contains|inci|full\s+ingredients?|"
    r"ingredients?\s+list)\s*[:\-–]?",
    re.IGNORECASE,
)
_PANEL_END = re.compile(
    r"(?:^|\n)\s*(directions?\s+for\s+use|how\s+to\s+use|usage|warnings?|caution|"
    r"storage|store\s+in|keep\s+out\s+of|manufactured\s+by|marketed\s+by|"
    r"packed\s+by|imported\s+by|distributed\s+by|customer\s+care|consumer\s+care|"
    r"net\s+w|mrp|maximum\s+retail|best\s+before|expiry|use\s+before|batch|lot\s+no|"
    r"nutrition\s+facts?|nutritional\s+information)",
    re.IGNORECASE,
)


def extract_ingredient_panel(raw_text: str) -> tuple[str, bool]:
    """Return (panel_text, found_header).

    When no header is found the caller gets the whole text back with
    found_header=False — the parser then knows the tokens are unverified and
    should be treated with more suspicion.
    """
    if not raw_text:
        return "", False

    start_match = _PANEL_START.search(raw_text)
    if not start_match:
        return raw_text, False

    remainder = raw_text[start_match.end():]
    end_match = _PANEL_END.search(remainder)
    panel = remainder[: end_match.start()] if end_match else remainder
    return panel.strip(), True


class OcrService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.api_url = os.getenv("OCR_API_URL", "https://ocr.moveneticsdigital.com/")
        self.api_key = os.getenv("OCR_API_KEY", "")
        self.lang = os.getenv("OCR_LANG", "en")
        self.timeout = int(os.getenv("OCR_TIMEOUT", "120"))
        self.max_chars = int(os.getenv("OCR_MAX_CHARS", "8000"))

    # -- cache ------------------------------------------------------------

    def _cached(self, sha: str) -> Optional[OcrResult]:
        row = self.session.execute(
            text(
                "SELECT ocr_text, confidence, engine FROM ocr_result "
                "WHERE image_sha256 = :sha AND lang = :lang "
                "  AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP(3))"
            ),
            {"sha": sha, "lang": self.lang},
        ).mappings().first()

        if not row:
            return None

        return OcrResult(
            text=row["ocr_text"] or "",
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            engine=row["engine"],
            cached=True,
            image_sha256=sha,
        )

    def _store(self, sha: str, result: OcrResult) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=OCR_CACHE_TTL_DAYS)
        self.session.execute(
            text(
                """
                INSERT INTO ocr_result (image_sha256, lang, ocr_text, confidence, engine, expires_at)
                VALUES (:sha, :lang, :ocr_text, :conf, :engine, :exp)
                ON DUPLICATE KEY UPDATE
                    ocr_text = VALUES(ocr_text), confidence = VALUES(confidence),
                    engine = VALUES(engine), expires_at = VALUES(expires_at)
                """
            ),
            {
                "sha": sha,
                "lang": self.lang,
                "ocr_text": result.text[: self.max_chars],
                "conf": result.confidence,
                "engine": result.engine,
                "exp": expires_at.replace(tzinfo=None),
            },
        )

    # -- run --------------------------------------------------------------

    def read(self, raw: bytes, *, force: bool = False) -> OcrResult:
        """OCR one image. Cached on content hash, so a repeat is free."""
        if not raw:
            return OcrResult(error="empty image")

        sha = hashlib.sha256(raw).hexdigest()

        if not force:
            cached = self._cached(sha)
            if cached is not None:
                logger.debug("ocr cache hit for %s", sha[:12])
                return cached

        if not self.api_url:
            return OcrResult(image_sha256=sha, error="OCR_API_URL is not configured")

        try:
            result = self._call_service(raw, sha)
        except FetchError as exc:
            # A failed OCR is not cached: the next attempt should retry rather
            # than inherit a transient upstream outage (arch.md 7.3).
            logger.warning("ocr request failed for %s: %s", sha[:12], exc)
            return OcrResult(image_sha256=sha, error=str(exc))

        if result.ok:
            self._store(sha, result)
        return result

    def _call_service(self, raw: bytes, sha: str) -> OcrResult:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        response = get_broker().post(
            self.api_url,
            headers=headers,
            params={"lang": self.lang},
            files={"file": ("image.jpg", raw, "image/jpeg")},
            timeout=self.timeout,
        )

        if response.status_code != 200:
            return OcrResult(image_sha256=sha, error=f"OCR service returned {response.status_code}")

        try:
            payload = response.json()
        except ValueError:
            # Some deployments return bare text rather than JSON.
            return OcrResult(
                text=response.text[: self.max_chars], engine="ocr-api", image_sha256=sha
            )

        if isinstance(payload, dict):
            extracted = (
                payload.get("text")
                or payload.get("result")
                or payload.get("ocr_text")
                or ""
            )
            confidence = payload.get("confidence")
            return OcrResult(
                text=str(extracted)[: self.max_chars],
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                engine=str(payload.get("engine") or "ocr-api"),
                image_sha256=sha,
            )

        return OcrResult(text=str(payload)[: self.max_chars], engine="ocr-api", image_sha256=sha)

    def read_many(self, images: list[bytes]) -> list[OcrResult]:
        return [self.read(raw) for raw in images]


def stitch(results: list[OcrResult]) -> str:
    """Join several images of one wrapped label into a single text.

    arch.md 8.2 asks for multi-image stitching. Ordering is by input order —
    the client controls the sequence, and it knows how the user swept the pack.
    """
    parts = [r.text.strip() for r in results if r.ok]
    return "\n".join(parts)
