"""Local product cache — the first step of the identification cascade."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.domain.models import ProductIdentity

PRODUCT_TTL_DAYS = 30


def _loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class ProductRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def by_barcode(self, barcodes: list[str]) -> Optional[dict[str, Any]]:
        """Look up any of the equivalent forms of one code at once.

        A scan yields EAN-13 while the stored row may be the GTIN-14 form, so
        the caller passes both and either hits.
        """
        candidates = [b for b in barcodes if b]
        if not candidates:
            return None

        placeholders = ", ".join(f":b{i}" for i in range(len(candidates)))
        params = {f"b{i}": b for i, b in enumerate(candidates)}
        row = self.session.execute(
            text(
                f"SELECT product_id, barcode, name, brand, category, product_class, "
                f"       ingredients_text, parsed_ingredients, source, confidence, fetched_at "
                f"FROM product "
                f"WHERE barcode IN ({placeholders}) "
                f"  AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP(3)) "
                f"ORDER BY confidence DESC LIMIT 1"
            ),
            params,
        ).mappings().first()

        if not row:
            return None
        return {**dict(row), "parsed_ingredients": _loads(row["parsed_ingredients"], [])}

    def upsert(
        self,
        *,
        barcode: str,
        name: Optional[str] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        product_class: Optional[str] = None,
        ingredients_text: Optional[str] = None,
        parsed_ingredients: Optional[list] = None,
        source: Optional[str] = None,
        confidence: float = 0.0,
        barcode_format: Optional[str] = None,
    ) -> str:
        existing = self.session.execute(
            text("SELECT product_id FROM product WHERE barcode = :b LIMIT 1"), {"b": barcode}
        ).first()
        product_id = existing[0] if existing else uuid.uuid4().hex

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        expires_at = now + timedelta(days=PRODUCT_TTL_DAYS)

        self.session.execute(
            text(
                """
                INSERT INTO product (product_id, barcode, barcode_format, name, brand, category,
                                     product_class, ingredients_text, parsed_ingredients,
                                     source, confidence, fetched_at, expires_at)
                VALUES (:pid, :barcode, :fmt, :name, :brand, :category,
                        :pclass, :ingredients, :parsed,
                        :source, :confidence, :fetched, :expires)
                ON DUPLICATE KEY UPDATE
                    name = COALESCE(VALUES(name), name),
                    brand = COALESCE(VALUES(brand), brand),
                    category = COALESCE(VALUES(category), category),
                    product_class = COALESCE(VALUES(product_class), product_class),
                    ingredients_text = COALESCE(VALUES(ingredients_text), ingredients_text),
                    parsed_ingredients = COALESCE(VALUES(parsed_ingredients), parsed_ingredients),
                    source = VALUES(source),
                    confidence = GREATEST(confidence, VALUES(confidence)),
                    fetched_at = VALUES(fetched_at),
                    expires_at = VALUES(expires_at)
                """
            ),
            {
                "pid": product_id,
                "barcode": barcode,
                "fmt": barcode_format,
                "name": name,
                "brand": brand,
                "category": category,
                "pclass": product_class,
                "ingredients": ingredients_text,
                "parsed": json.dumps(parsed_ingredients) if parsed_ingredients else None,
                "source": source,
                "confidence": confidence,
                "fetched": now,
                "expires": expires_at,
            },
        )
        return product_id

    @staticmethod
    def to_identity(row: dict[str, Any]) -> ProductIdentity:
        return ProductIdentity(
            product_id=row.get("product_id"),
            barcode=row.get("barcode"),
            name=row.get("name"),
            brand=row.get("brand"),
            category=row.get("category"),
            source=row.get("source"),
            confidence=float(row.get("confidence") or 0),
            fetched_at=row.get("fetched_at"),
        )
