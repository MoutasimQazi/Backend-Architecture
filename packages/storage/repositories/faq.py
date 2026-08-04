"""FAQ knowledge-base access: exact hash, FULLTEXT lexical, and vector rows."""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.common.text import norm_hash, normalise_question
from packages.domain.enums import FaqCategory, FaqStatus, SafetyClass
from packages.domain.models import FaqItem, FaqVariants, PersonalisationRule


def _loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _to_item(row: dict[str, Any]) -> FaqItem:
    variants = _loads(row.get("variants"), {}) or {}
    rules = _loads(row.get("personalisation_rules"), []) or []
    return FaqItem(
        id=row["id"],
        version=int(row.get("version") or 1),
        status=FaqStatus(row.get("status") or "draft"),
        category=FaqCategory(row.get("category") or "General"),
        canonical_question=row.get("canonical_question") or "",
        answer_template=row.get("answer_template") or "",
        variants=FaqVariants(**{k: v for k, v in variants.items() if k in FaqVariants.model_fields}),
        required_slots=_loads(row.get("required_slots"), []) or [],
        personalisation_rules=[
            PersonalisationRule(**{k: v for k, v in r.items() if k in PersonalisationRule.model_fields})
            for r in rules
            if isinstance(r, dict)
        ],
        safety_class=SafetyClass(row.get("safety_class") or "informational"),
        locale=row.get("locale") or "en",
        effective_from=row.get("effective_from"),
        effective_to=row.get("effective_to"),
        owner=row.get("owner"),
        reviewed_by=row.get("reviewed_by"),
        reviewed_at=row.get("reviewed_at"),
    )


class FaqRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # -- reads ------------------------------------------------------------

    def get(self, faq_id: str) -> Optional[FaqItem]:
        row = self.session.execute(
            text("SELECT * FROM faq_item WHERE id = :id"), {"id": faq_id}
        ).mappings().first()
        return _to_item(dict(row)) if row else None

    def get_many(self, faq_ids: list[str]) -> dict[str, FaqItem]:
        if not faq_ids:
            return {}
        # Expanding IN lists by hand keeps this on the plain text() API.
        placeholders = ", ".join(f":id{i}" for i in range(len(faq_ids)))
        params = {f"id{i}": fid for i, fid in enumerate(faq_ids)}
        rows = self.session.execute(
            text(f"SELECT * FROM faq_item WHERE id IN ({placeholders})"), params
        ).mappings().all()
        return {r["id"]: _to_item(dict(r)) for r in rows}

    def find_exact(self, question: str, locale: str = "en") -> Optional[tuple[str, str]]:
        """S1 of the cascade: O(1) hash lookup, no model. Returns (faq_id, surface)."""
        row = self.session.execute(
            text(
                """
                SELECT s.faq_id, s.surface_text
                FROM faq_surface s
                JOIN faq_item i ON i.id = s.faq_id
                WHERE s.norm_hash = :h AND s.locale = :loc
                  AND i.status = 'live'
                  AND (i.effective_from IS NULL OR i.effective_from <= UTC_TIMESTAMP(3))
                  AND (i.effective_to   IS NULL OR i.effective_to   >  UTC_TIMESTAMP(3))
                LIMIT 1
                """
            ),
            {"h": norm_hash(question, locale), "loc": locale},
        ).first()
        return (row[0], row[1]) if row else None

    def search_lexical(self, question: str, locale: str = "en", limit: int = 50) -> list[tuple[str, float]]:
        """BM25-ish half of the hybrid stage, via InnoDB FULLTEXT.

        Lexical recall is the point: 'TDEE', 'BMR' and 'INCI' are tokens an
        embedding blurs (arch.md 5.2).
        """
        query = normalise_question(question)
        if not query:
            return []
        rows = self.session.execute(
            text(
                """
                SELECT s.faq_id, MATCH(s.surface_text) AGAINST (:q IN NATURAL LANGUAGE MODE) AS score
                FROM faq_surface s
                JOIN faq_item i ON i.id = s.faq_id
                WHERE s.locale = :loc AND i.status = 'live'
                  AND MATCH(s.surface_text) AGAINST (:q IN NATURAL LANGUAGE MODE)
                ORDER BY score DESC
                LIMIT :lim
                """
            ),
            {"q": query, "loc": locale, "lim": max(1, min(limit, 200))},
        ).all()
        return [(faq_id, float(score)) for faq_id, score in rows]

    def embedding_rows(self, locale: str = "en") -> list[tuple[str, bytes, dict[str, Any]]]:
        """Every live surface with an embedding, for the in-process index."""
        rows = self.session.execute(
            text(
                """
                SELECT s.id, s.faq_id, s.surface_text, s.embedding, i.category
                FROM faq_surface s
                JOIN faq_item i ON i.id = s.faq_id
                WHERE s.locale = :loc AND i.status = 'live' AND s.embedding IS NOT NULL
                """
            ),
            {"loc": locale},
        ).mappings().all()
        return [
            (
                str(r["id"]),
                r["embedding"],
                {"faq_id": r["faq_id"], "surface": r["surface_text"], "category": r["category"]},
            )
            for r in rows
        ]

    def live_version_token(self, locale: str = "en") -> str:
        """Cheap fingerprint of the live KB; the vector index reloads when it
        changes, so a publish is visible without a restart or a poll timer."""
        row = self.session.execute(
            text(
                "SELECT COUNT(*) AS n, COALESCE(MAX(i.updated_at), '1970-01-01') AS latest "
                "FROM faq_surface s JOIN faq_item i ON i.id = s.faq_id "
                "WHERE s.locale = :loc AND i.status = 'live'"
            ),
            {"loc": locale},
        ).mappings().first()
        return f"{row['n']}:{row['latest']}"

    def surfaces_without_embedding(self, limit: int = 200) -> list[tuple[int, str]]:
        rows = self.session.execute(
            text(
                "SELECT id, surface_text FROM faq_surface WHERE embedding IS NULL LIMIT :lim"
            ),
            {"lim": limit},
        ).all()
        return [(int(i), t) for i, t in rows]

    # -- writes -----------------------------------------------------------

    def upsert_item(self, item: FaqItem) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO faq_item
                    (id, version, status, category, canonical_question, answer_template,
                     variants, required_slots, personalisation_rules, safety_class, locale,
                     effective_from, effective_to, owner, reviewed_by, reviewed_at)
                VALUES
                    (:id, :version, :status, :category, :cq, :tpl,
                     :variants, :slots, :rules, :safety, :locale,
                     :eff_from, :eff_to, :owner, :rev_by, :rev_at)
                ON DUPLICATE KEY UPDATE
                    version = VALUES(version),
                    status = VALUES(status),
                    category = VALUES(category),
                    canonical_question = VALUES(canonical_question),
                    answer_template = VALUES(answer_template),
                    variants = VALUES(variants),
                    required_slots = VALUES(required_slots),
                    personalisation_rules = VALUES(personalisation_rules),
                    safety_class = VALUES(safety_class),
                    locale = VALUES(locale),
                    effective_from = VALUES(effective_from),
                    effective_to = VALUES(effective_to),
                    owner = VALUES(owner),
                    reviewed_by = VALUES(reviewed_by),
                    reviewed_at = VALUES(reviewed_at)
                """
            ),
            {
                "id": item.id,
                "version": item.version,
                "status": str(item.status.value),
                "category": str(item.category.value),
                "cq": item.canonical_question,
                "tpl": item.answer_template,
                "variants": item.variants.model_dump_json(exclude_none=True),
                "slots": json.dumps(item.required_slots),
                "rules": json.dumps([r.model_dump(exclude_none=True) for r in item.personalisation_rules]),
                "safety": str(item.safety_class.value),
                "locale": item.locale,
                "eff_from": item.effective_from,
                "eff_to": item.effective_to,
                "owner": item.owner,
                "rev_by": item.reviewed_by,
                "rev_at": item.reviewed_at,
            },
        )

    def replace_surfaces(self, faq_id: str, item: FaqItem) -> int:
        """Rewrite this item's surface rows from its canonical + paraphrases."""
        self.session.execute(
            text("DELETE FROM faq_surface WHERE faq_id = :fid"), {"fid": faq_id}
        )
        surfaces = [(item.canonical_question, "canonical")] + [
            (p, "paraphrase") for p in item.paraphrases
        ]
        written = 0
        seen: set[str] = set()
        for surface_text, kind in surfaces:
            if not surface_text or not surface_text.strip():
                continue
            h = norm_hash(surface_text, item.locale)
            if h in seen:
                continue  # a duplicate paraphrase would only skew the index
            seen.add(h)
            self.session.execute(
                text(
                    "INSERT INTO faq_surface (faq_id, surface_text, norm_hash, kind, locale) "
                    "VALUES (:fid, :txt, :h, :kind, :loc)"
                ),
                {"fid": faq_id, "txt": surface_text.strip(), "h": h, "kind": kind, "loc": item.locale},
            )
            written += 1
        return written

    def set_surface_embedding(self, surface_id: int, embedding_blob: bytes, model: str, dim: int) -> None:
        self.session.execute(
            text(
                "UPDATE faq_surface SET embedding = :e, embedding_model = :m, embedding_dim = :d "
                "WHERE id = :sid"
            ),
            {"e": embedding_blob, "m": model, "d": dim, "sid": surface_id},
        )

    def record_unmatched(self, question: str, locale: str = "en", route_label: Optional[str] = None) -> None:
        """arch.md 5.1 paraphrase-mining loop: what we could not answer becomes
        next month's FAQ candidates."""
        normalised = normalise_question(question)
        if not normalised:
            return
        self.session.execute(
            text(
                """
                INSERT INTO unmatched_question (norm_text, norm_hash, locale, hits, route_label)
                VALUES (:txt, :h, :loc, 1, :route)
                ON DUPLICATE KEY UPDATE hits = hits + 1, route_label = COALESCE(VALUES(route_label), route_label)
                """
            ),
            {"txt": normalised, "h": norm_hash(question, locale), "loc": locale, "route": route_label},
        )

    def top_unmatched(self, min_hits: int = 3, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.session.execute(
            text(
                "SELECT norm_text, hits, locale, route_label, last_seen FROM unmatched_question "
                "WHERE promoted = 0 AND hits >= :minhits ORDER BY hits DESC LIMIT :lim"
            ),
            {"minhits": min_hits, "lim": limit},
        ).mappings().all()
        return [dict(r) for r in rows]
