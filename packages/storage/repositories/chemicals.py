"""Chemical KB access: dossiers, synonyms, assertions, rules, cross-reactants."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from packages.common.text import normalise_ingredient, sha256_hex
from packages.domain.enums import SourceTier
from packages.domain.models import EvidenceRef

logger = logging.getLogger(__name__)


def synonym_hash(value: str) -> str:
    return sha256_hex(normalise_ingredient(value))


def _loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class ChemicalRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # -- resolution -------------------------------------------------------

    def by_synonym_hash(self, hashes: list[str]) -> dict[str, tuple[str, str]]:
        """hash -> (chemical_id, kind). Batched: a 30-ingredient panel should
        cost one query, not thirty."""
        if not hashes:
            return {}

        placeholders = ", ".join(f":h{i}" for i in range(len(hashes)))
        params = {f"h{i}": h for i, h in enumerate(hashes)}
        rows = self.session.execute(
            text(
                f"SELECT norm_hash, chemical_id, kind FROM chemical_synonym "
                f"WHERE norm_hash IN ({placeholders})"
            ),
            params,
        ).mappings().all()

        out: dict[str, tuple[str, str]] = {}
        for row in rows:
            # An INCI match outranks a loose trade-name synonym for the same
            # surface form, so a stronger kind wins the slot.
            existing = out.get(row["norm_hash"])
            if existing is None or _kind_rank(row["kind"]) > _kind_rank(existing[1]):
                out[row["norm_hash"]] = (row["chemical_id"], row["kind"])
        return out

    def by_identifier(self, *, cas: Optional[str] = None, ec: Optional[str] = None,
                      e_number: Optional[str] = None) -> Optional[str]:
        clauses, params = [], {}
        if cas:
            clauses.append("cas = :cas")
            params["cas"] = cas
        if ec:
            clauses.append("ec = :ec")
            params["ec"] = ec
        if e_number:
            clauses.append("e_number = :en")
            params["en"] = e_number
        if not clauses:
            return None

        row = self.session.execute(
            text(f"SELECT chemical_id FROM chemical WHERE {' OR '.join(clauses)} LIMIT 1"),
            params,
        ).first()
        return row[0] if row else None

    def search_lexical(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        """FULLTEXT over synonyms — the cheap half of fuzzy matching."""
        normalised = normalise_ingredient(query)
        if not normalised or len(normalised) < 3:
            return []
        rows = self.session.execute(
            text(
                "SELECT chemical_id, MATCH(synonym) AGAINST (:q IN NATURAL LANGUAGE MODE) AS score "
                "FROM chemical_synonym "
                "WHERE MATCH(synonym) AGAINST (:q IN NATURAL LANGUAGE MODE) "
                "ORDER BY score DESC LIMIT :lim"
            ),
            {"q": normalised, "lim": limit},
        ).all()
        return [(cid, float(score)) for cid, score in rows]

    def all_synonyms(self, limit: int = 50000) -> list[tuple[str, str, str]]:
        """(chemical_id, synonym, norm_text) for the in-process fuzzy index."""
        rows = self.session.execute(
            text("SELECT chemical_id, synonym, norm_text FROM chemical_synonym LIMIT :lim"),
            {"lim": limit},
        ).all()
        return [(cid, syn, norm) for cid, syn, norm in rows]

    def synonym_count(self) -> int:
        return int(
            self.session.execute(text("SELECT COUNT(*) FROM chemical_synonym")).scalar() or 0
        )

    # -- dossiers ---------------------------------------------------------

    def get_many(self, chemical_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chemical_ids:
            return {}
        placeholders = ", ".join(f":c{i}" for i in range(len(chemical_ids)))
        params = {f"c{i}": cid for i, cid in enumerate(chemical_ids)}
        rows = self.session.execute(
            text(
                f"SELECT chemical_id, inci_name, display_name, cas, ec, e_number, "
                f"       formula, chem_class, functions, kb_version, review_status "
                f"FROM chemical WHERE chemical_id IN ({placeholders})"
            ),
            params,
        ).mappings().all()
        return {
            r["chemical_id"]: {**dict(r), "functions": _loads(r["functions"], []) or []}
            for r in rows
        }

    def assertions_for(
        self,
        chemical_ids: list[str],
        *,
        jurisdiction: Optional[str] = None,
        product_class: Optional[str] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """All assertions for these chemicals, in one query.

        **Jurisdiction does not filter.** arch.md 8.5 asks for "restricted/banned
        status per jurisdiction, with the user's jurisdiction applied" — report
        everything, then apply the user's. Excluding on jurisdiction looked
        right and was badly wrong: most substantive cosmetics regulation is EU
        (Annex II/III, the EU-26 allergens, the EDC list), so an Indian user got
        an empty hazard profile for a product the EU restricts. The finding
        records which jurisdiction each assertion came from, so the answer can
        say "restricted in the EU" rather than silently knowing nothing.

        `product_class` *does* filter: a food-additive limit genuinely does not
        apply to a face cream.
        """
        if not chemical_ids:
            return {}

        placeholders = ", ".join(f":c{i}" for i in range(len(chemical_ids)))
        params: dict[str, Any] = {f"c{i}": cid for i, cid in enumerate(chemical_ids)}

        clauses = [f"chemical_id IN ({placeholders})"]
        if product_class:
            clauses.append("(product_class IS NULL OR product_class = :pclass)")
            params["pclass"] = product_class

        rows = self.session.execute(
            text(
                f"SELECT chemical_id, domain, key_name, value, jurisdiction, product_class, "
                f"       limit_value, limit_unit, evidence_grade, source, source_url, kb_version "
                f"FROM chemical_assertion WHERE {' AND '.join(clauses)}"
            ),
            params,
        ).mappings().all()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["chemical_id"], []).append(dict(row))
        return grouped

    def evidence_for(self, chemical_ids: list[str], limit_per: int = 3) -> dict[str, list[EvidenceRef]]:
        if not chemical_ids:
            return {}
        placeholders = ", ".join(f":c{i}" for i in range(len(chemical_ids)))
        params = {f"c{i}": cid for i, cid in enumerate(chemical_ids)}
        rows = self.session.execute(
            text(
                f"""
                SELECT ce.chemical_id, d.source_id, d.title, d.url, d.tier, d.published_year,
                       d.study_design, d.independence, d.retrieved_at
                FROM chemical_evidence ce
                JOIN evidence_document d ON d.source_id = ce.source_id
                WHERE ce.chemical_id IN ({placeholders})
                ORDER BY d.tier ASC, d.independence DESC, d.published_year DESC
                """
            ),
            params,
        ).mappings().all()

        grouped: dict[str, list[EvidenceRef]] = {}
        for row in rows:
            bucket = grouped.setdefault(row["chemical_id"], [])
            if len(bucket) >= limit_per:
                continue
            try:
                tier = SourceTier(row["tier"])
            except ValueError:
                tier = SourceTier.T4_SECONDARY
            bucket.append(
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
        return grouped

    # -- rules ------------------------------------------------------------

    def active_rules(
        self, *, product_class: Optional[str] = None, jurisdiction: Optional[str] = None
    ) -> list[dict[str, Any]]:
        clauses = ["active = 1"]
        params: dict[str, Any] = {}
        if product_class:
            clauses.append("(product_class IS NULL OR product_class = :pclass)")
            params["pclass"] = product_class
        if jurisdiction:
            clauses.append("(jurisdiction IS NULL OR jurisdiction IN ('INTL', :jur))")
            params["jur"] = jurisdiction

        rows = self.session.execute(
            text(
                f"SELECT rule_id, version, product_class, jurisdiction, priority, "
                f"       condition_json, effect_json, description "
                f"FROM hazard_rule WHERE {' AND '.join(clauses)} "
                f"ORDER BY priority ASC, rule_id ASC"
            ),
            params,
        ).mappings().all()

        return [
            {
                **dict(r),
                "condition": _loads(r["condition_json"], {}) or {},
                "effect": _loads(r["effect_json"], {}) or {},
            }
            for r in rows
        ]

    def cross_reactants(self, allergen_keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        """allergen key -> chemicals that cross-react (arch.md 8.6)."""
        if not allergen_keys:
            return {}
        placeholders = ", ".join(f":a{i}" for i in range(len(allergen_keys)))
        params = {f"a{i}": key for i, key in enumerate(allergen_keys)}
        rows = self.session.execute(
            text(
                f"SELECT allergen_key, chemical_id, severity, source "
                f"FROM allergen_cross_reactant WHERE allergen_key IN ({placeholders})"
            ),
            params,
        ).mappings().all()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["allergen_key"], []).append(dict(row))
        return grouped

    # -- writes (used by the ETL and the seed script) ---------------------

    def upsert_chemical(self, **fields: Any) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO chemical (chemical_id, inci_name, display_name, cas, ec, e_number,
                                      formula, chem_class, functions, kb_version, review_status)
                VALUES (:chemical_id, :inci_name, :display_name, :cas, :ec, :e_number,
                        :formula, :chem_class, :functions, :kb_version, :review_status)
                ON DUPLICATE KEY UPDATE
                    inci_name = COALESCE(VALUES(inci_name), inci_name),
                    -- A curated display name outranks whatever the ETL derived
                    -- from a search term: 'Water' should not become 'Aqua'
                    -- because someone looked the dossier up by its INCI name.
                    display_name = IF(review_status = 'published',
                                      display_name, VALUES(display_name)),
                    cas = COALESCE(VALUES(cas), cas),
                    ec = COALESCE(VALUES(ec), ec),
                    e_number = COALESCE(VALUES(e_number), e_number),
                    formula = COALESCE(VALUES(formula), formula),
                    chem_class = COALESCE(VALUES(chem_class), chem_class),
                    functions = VALUES(functions),
                    kb_version = VALUES(kb_version),
                    -- Never downgrade a reviewed dossier. An ETL refresh must
                    -- not silently discard a human's sign-off; only an explicit
                    -- unpublish should do that.
                    review_status = IF(review_status = 'published',
                                       'published', VALUES(review_status))
                """
            ),
            {
                "chemical_id": fields["chemical_id"],
                "inci_name": fields.get("inci_name"),
                "display_name": fields.get("display_name") or fields["chemical_id"],
                "cas": fields.get("cas"),
                "ec": fields.get("ec"),
                "e_number": fields.get("e_number"),
                "formula": fields.get("formula"),
                "chem_class": fields.get("chem_class"),
                "functions": json.dumps(fields.get("functions") or []),
                "kb_version": fields.get("kb_version", "v1"),
                "review_status": fields.get("review_status", "published"),
            },
        )

    def add_synonym(self, chemical_id: str, synonym: str, kind: str = "synonym") -> None:
        normalised = normalise_ingredient(synonym)
        if not normalised:
            return
        self.session.execute(
            text(
                """
                INSERT INTO chemical_synonym (chemical_id, synonym, norm_text, norm_hash, kind)
                VALUES (:cid, :syn, :norm, :hash, :kind)
                ON DUPLICATE KEY UPDATE synonym = VALUES(synonym), kind = VALUES(kind)
                """
            ),
            {
                "cid": chemical_id,
                "syn": synonym[:255],
                "norm": normalised[:255],
                "hash": sha256_hex(normalised),
                "kind": kind,
            },
        )

    def add_assertion(self, chemical_id: str, domain: str, key_name: str, **fields: Any) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO chemical_assertion
                    (chemical_id, domain, key_name, value, jurisdiction, product_class,
                     limit_value, limit_unit, evidence_grade, source, source_url, kb_version)
                VALUES (:cid, :domain, :key, :value, :jur, :pclass,
                        :limit_value, :limit_unit, :grade, :source, :url, :kbv)
                """
            ),
            {
                "cid": chemical_id,
                "domain": domain,
                "key": key_name,
                "value": fields.get("value"),
                "jur": fields.get("jurisdiction"),
                "pclass": fields.get("product_class"),
                "limit_value": fields.get("limit_value"),
                "limit_unit": fields.get("limit_unit"),
                "grade": fields.get("evidence_grade"),
                "source": fields.get("source"),
                "url": fields.get("source_url"),
                "kbv": fields.get("kb_version", "v1"),
            },
        )

    def upsert_rule(self, rule_id: str, condition: dict, effect: dict, **fields: Any) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO hazard_rule (rule_id, version, active, product_class, jurisdiction,
                                         priority, condition_json, effect_json, description, owner)
                VALUES (:rid, :version, :active, :pclass, :jur, :prio, :cond, :effect, :desc, :owner)
                ON DUPLICATE KEY UPDATE
                    active = VALUES(active), product_class = VALUES(product_class),
                    jurisdiction = VALUES(jurisdiction), priority = VALUES(priority),
                    condition_json = VALUES(condition_json), effect_json = VALUES(effect_json),
                    description = VALUES(description)
                """
            ),
            {
                "rid": rule_id,
                "version": fields.get("version", 1),
                "active": 1 if fields.get("active", True) else 0,
                "pclass": fields.get("product_class"),
                "jur": fields.get("jurisdiction"),
                "prio": fields.get("priority", 100),
                "cond": json.dumps(condition),
                "effect": json.dumps(effect),
                "desc": fields.get("description"),
                "owner": fields.get("owner"),
            },
        )

    def add_cross_reactant(
        self, allergen_key: str, chemical_id: str, severity: str = "moderate", source: Optional[str] = None
    ) -> None:
        self.session.execute(
            text(
                """
                INSERT INTO allergen_cross_reactant (allergen_key, chemical_id, severity, source)
                VALUES (:key, :cid, :sev, :src)
                ON DUPLICATE KEY UPDATE severity = VALUES(severity), source = VALUES(source)
                """
            ),
            {"key": allergen_key.lower(), "cid": chemical_id, "sev": severity, "src": source},
        )

    def kb_stats(self) -> dict[str, int]:
        row = self.session.execute(
            text(
                "SELECT (SELECT COUNT(*) FROM chemical) AS chemicals, "
                "       (SELECT COUNT(*) FROM chemical_synonym) AS synonyms, "
                "       (SELECT COUNT(*) FROM chemical_assertion) AS assertions, "
                "       (SELECT COUNT(*) FROM hazard_rule WHERE active = 1) AS rules"
            )
        ).mappings().first()
        return {k: int(v or 0) for k, v in dict(row).items()} if row else {}


_KIND_RANK = {"inci": 5, "cas": 4, "ec": 4, "e_number": 4, "trade": 2, "synonym": 3}


def _kind_rank(kind: str) -> int:
    return _KIND_RANK.get(kind, 1)
