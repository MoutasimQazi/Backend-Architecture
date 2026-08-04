"""Open Food / Beauty / Products Facts.

One connector for all three: same API shape, different host and different
product class. Which is useful beyond saving code — the host that answered
tells us whether we are looking at a food, a cosmetic or a general product,
and arch.md 8.5 grades the same chemical differently in each.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from packages.guards.fetch_broker import FetchError, get_broker

logger = logging.getLogger(__name__)

# Ordered: food is the most common scan, then cosmetics, then everything else.
SOURCES: list[tuple[str, str, str]] = [
    ("off", "world.openfoodfacts.org", "food"),
    ("obf", "world.openbeautyfacts.org", "cosmetic"),
    ("opf", "world.openproductsfacts.org", "general"),
]

_FIELDS = ",".join(
    [
        "code",
        "product_name",
        "brands",
        "categories",
        "ingredients_text",
        "ingredients_text_en",
        "quantity",
        "image_url",
        "allergens_tags",
        "traces_tags",
        "nutriscore_grade",
    ]
)


@dataclass
class ProductRecord:
    barcode: str
    found: bool = False
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    product_class: Optional[str] = None
    ingredients_text: Optional[str] = None
    allergens: list[str] = field(default_factory=list)
    source: Optional[str] = None
    confidence: float = 0.0
    fetched_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def has_ingredients(self) -> bool:
        return bool((self.ingredients_text or "").strip())


class OpenFactsConnector:
    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def lookup(self, barcode: str, *, sources: Optional[list[str]] = None) -> ProductRecord:
        """Try each source in turn. First hit with ingredients wins.

        A hit *without* ingredients is remembered but does not stop the walk:
        the product name alone cannot drive an ingredient analysis, and a later
        source may carry the panel.
        """
        wanted = set(sources) if sources else None
        best_without_ingredients: Optional[ProductRecord] = None

        for key, host, product_class in SOURCES:
            if wanted and key not in wanted:
                continue

            record = self._fetch_one(barcode, key, host, product_class)
            if not record.found:
                continue
            if record.has_ingredients:
                return record
            if best_without_ingredients is None:
                best_without_ingredients = record

        if best_without_ingredients is not None:
            return best_without_ingredients

        return ProductRecord(barcode=barcode, found=False)

    def _fetch_one(
        self, barcode: str, key: str, host: str, product_class: str
    ) -> ProductRecord:
        url = f"https://{host}/api/v2/product/{barcode}"
        try:
            response = get_broker().get(
                url, params={"fields": _FIELDS}, timeout=self.timeout
            )
        except FetchError as exc:
            logger.info("%s lookup failed for %s: %s", key, barcode, exc)
            return ProductRecord(barcode=barcode, found=False, source=key, error=str(exc))

        if response.status_code == 404:
            return ProductRecord(barcode=barcode, found=False, source=key)
        if response.status_code != 200:
            return ProductRecord(
                barcode=barcode,
                found=False,
                source=key,
                error=f"HTTP {response.status_code}",
            )

        try:
            payload = response.json()
        except ValueError:
            return ProductRecord(barcode=barcode, found=False, source=key, error="bad JSON")

        if not isinstance(payload, dict) or payload.get("status") != 1:
            return ProductRecord(barcode=barcode, found=False, source=key)

        product = payload.get("product") or {}
        if not isinstance(product, dict):
            return ProductRecord(barcode=barcode, found=False, source=key)

        ingredients = (
            product.get("ingredients_text_en") or product.get("ingredients_text") or ""
        ).strip()

        name = (product.get("product_name") or "").strip() or None

        # A record with neither a name nor ingredients is a stub someone created
        # by scanning; treating it as "found" would suppress the OCR fallback.
        if not name and not ingredients:
            return ProductRecord(barcode=barcode, found=False, source=key)

        allergens = [
            tag.split(":", 1)[-1]
            for tag in (product.get("allergens_tags") or [])
            if isinstance(tag, str)
        ]

        confidence = 0.6
        if ingredients:
            confidence += 0.3
        if name:
            confidence += 0.1

        return ProductRecord(
            barcode=barcode,
            found=True,
            name=name,
            brand=(product.get("brands") or "").split(",")[0].strip() or None,
            category=(product.get("categories") or "").split(",")[0].strip() or None,
            product_class=product_class,
            ingredients_text=ingredients or None,
            allergens=allergens,
            source=key,
            confidence=round(min(confidence, 1.0), 2),
            fetched_at=datetime.now(timezone.utc),
        )
