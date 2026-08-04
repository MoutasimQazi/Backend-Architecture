"""Evidence document and chunk storage."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.domain.enums import SourceTier
from packages.domain.models import EvidenceRef

logger = logging.getLogger(__name__)


class EvidenceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_document(
        self,
        source_id: str,
        *,
        title: Optional[str] = None,
        container: Optional[str] = None,
        url: Optional[str] = None,
        tier: SourceTier = SourceTier.T3_PRIMARY,
        year: Optional[int] = None,
        study_design: Optional[str] = None,
        funder_class: Optional[str] = None,
        declared_coi: Optional[bool] = None,
        sponsor_role: Optional[str] = None,
        registry_status: Optional[str] = None,
        independence: Optional[float] = None,
        abstract: Optional[str] = None,
    ) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO evidence_document
                    (source_id, title, container, url, tier, published_year, study_design,
                     funder_class, declared_coi, sponsor_role, registry_status,
                     independence, abstract, retrieved_at)
                VALUES
                    (:sid, :title, :container, :url, :tier, :year, :design,
                     :funder, :coi, :role, :registry, :independence, :abstract, :now)
                ON DUPLICATE KEY UPDATE
                    title = COALESCE(VALUES(title), title),
                    container = COALESCE(VALUES(container), container),
                    url = COALESCE(VALUES(url), url),
                    tier = VALUES(tier),
                    published_year = COALESCE(VALUES(published_year), published_year),
                    study_design = COALESCE(VALUES(study_design), study_design),
                    funder_class = COALESCE(VALUES(funder_class), funder_class),
                    declared_coi = COALESCE(VALUES(declared_coi), declared_coi),
                    sponsor_role = COALESCE(VALUES(sponsor_role), sponsor_role),
                    registry_status = COALESCE(VALUES(registry_status), registry_status),
                    independence = COALESCE(VALUES(independence), independence),
                    abstract = COALESCE(VALUES(abstract), abstract),
                    retrieved_at = VALUES(retrieved_at)
                """
            ),
            {
                "sid": source_id[:96],
                "title": title,
                "container": container,
                "url": url[:768] if url else None,
                "tier": str(tier.value),
                "year": year,
                "design": study_design,
                "funder": funder_class,
                "coi": None if declared_coi is None else (1 if declared_coi else 0),
                "role": sponsor_role,
                "registry": registry_status,
                "independence": independence,
                "abstract": abstract[:60000] if abstract else None,
                "now": datetime.now(timezone.utc).replace(tzinfo=None),
            },
        )

    def add_chunk(self, source_id: str, chunk_index: int, chunk_text: str) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO evidence_chunk (source_id, chunk_index, chunk_text)
                VALUES (:sid, :idx, :txt)
                ON DUPLICATE KEY UPDATE chunk_text = VALUES(chunk_text)
                """
            ),
            {"sid": source_id[:96], "idx": chunk_index, "txt": chunk_text[:60000]},
        )

    def link_chemical(
        self, chemical_id: str, source_id: str, relation: str = "general", note: Optional[str] = None
    ) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO chemical_evidence (chemical_id, source_id, relation, note)
                VALUES (:cid, :sid, :rel, :note)
                ON DUPLICATE KEY UPDATE note = VALUES(note)
                """
            ),
            {"cid": chemical_id, "sid": source_id[:96], "rel": relation, "note": note},
        )

    def search(self, query: str, limit: int = 10) -> list[EvidenceRef]:
        """FULLTEXT over stored documents, ranked by tier and independence.

        This is the runtime path: a research question is answered from what the
        ETL already gathered, not by fanning out to six APIs mid-request.
        """
        if not query or len(query.strip()) < 3:
            return []

        rows = self.session.execute(
            text(
                """
                SELECT source_id, title, url, tier, published_year, study_design,
                       independence, retrieved_at,
                       MATCH(title, abstract) AGAINST (:q IN NATURAL LANGUAGE MODE) AS relevance
                FROM evidence_document
                WHERE MATCH(title, abstract) AGAINST (:q IN NATURAL LANGUAGE MODE)
                  AND tier <> 'blocked'
                ORDER BY relevance DESC, tier ASC, independence DESC
                LIMIT :lim
                """
            ),
            {"q": query.strip(), "lim": max(1, min(limit, 50))},
        ).mappings().all()

        refs: list[EvidenceRef] = []
        for row in rows:
            try:
                tier = SourceTier(row["tier"])
            except ValueError:
                tier = SourceTier.T4_SECONDARY
            refs.append(
                EvidenceRef(
                    source_id=row["source_id"],
                    tier=tier,
                    title=row["title"],
                    url=row["url"],
                    year=int(row["published_year"]) if row["published_year"] else None,
                    study_design=row["study_design"],
                    independence=float(row["independence"]) if row["independence"] is not None else None,
                    retrieved_at=row["retrieved_at"],
                )
            )
        return refs

    def chunks_for(self, source_ids: list[str], limit_per: int = 3) -> dict[str, list[str]]:
        if not source_ids:
            return {}
        placeholders = ", ".join(f":s{i}" for i in range(len(source_ids)))
        params = {f"s{i}": sid for i, sid in enumerate(source_ids)}
        rows = self.session.execute(
            text(
                f"SELECT source_id, chunk_text FROM evidence_chunk "
                f"WHERE source_id IN ({placeholders}) ORDER BY source_id, chunk_index"
            ),
            params,
        ).all()

        grouped: dict[str, list[str]] = {}
        for source_id, chunk_text in rows:
            bucket = grouped.setdefault(source_id, [])
            if len(bucket) < limit_per:
                bucket.append(chunk_text)
        return grouped

    def stats(self) -> dict[str, int]:
        row = self.session.execute(
            text(
                "SELECT (SELECT COUNT(*) FROM evidence_document) AS documents, "
                "       (SELECT COUNT(*) FROM evidence_chunk) AS chunks, "
                "       (SELECT COUNT(*) FROM chemical_evidence) AS links"
            )
        ).mappings().first()
        return {k: int(v or 0) for k, v in dict(row).items()} if row else {}
