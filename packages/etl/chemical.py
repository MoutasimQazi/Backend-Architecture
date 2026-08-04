"""Chemical dossier ETL (arch.md 8.4).

Builds a dossier from PubChem (identity, synonyms, GHS, IARC) and EuropePMC
(absorption and safety evidence), writes it with per-field provenance, and
leaves it as `draft` for review.

Two properties this must have and does:

  * **Nothing published without review.** A dossier lands as `draft`; only a
    reviewer promotes it to `published`. The runtime resolver reads any status,
    but hazard rules only fire on assertions, and an unreviewed assertion is
    visibly attributed to its source.
  * **Idempotent.** Re-running refreshes rather than duplicating, so a failed
    run is safe to repeat.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from packages.common.text import normalise_ingredient
from packages.config import get_settings
from packages.connectors.literature import (
    EuropePMCConnector,
    PubChemConnector,
    extract_ghs_codes,
    extract_iarc_group,
)
from packages.domain.enums import SourceTier
from packages.evidence.independence import score_independence
from packages.storage.repositories.chemicals import ChemicalRepository
from packages.storage.repositories.evidence import EvidenceRepository

logger = logging.getLogger(__name__)

_CAS_IN_TEXT = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

# Synonyms worth keeping. PubChem returns hundreds per compound, most of them
# registry codes and vendor SKUs that would bloat the fuzzy corpus and cause
# false matches.
_JUNK_SYNONYM = re.compile(
    r"^(?:[A-Z]{1,4}[-\s]?\d{3,}|\d+|CHEMBL\d+|CHEBI:\d+|DTXSID\d+|UNII[-\s]|"
    r"EINECS|NSC\s?\d+|AKOS\d+|MFCD\d+|SCHEMBL\d+|BDBM\d+|Q\d{5,})",
    re.IGNORECASE,
)


@dataclass
class EtlOutcome:
    chemical_id: Optional[str] = None
    created: bool = False
    synonyms_added: int = 0
    assertions_added: int = 0
    evidence_added: int = 0
    external_calls: int = 0
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.chemical_id is not None


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalise_ingredient(name)).strip("_")[:64]


def _useful_synonym(value: str) -> bool:
    if not value or len(value) < 3 or len(value) > 120:
        return False
    if _JUNK_SYNONYM.match(value.strip()):
        return False
    # A synonym that is mostly digits is a registry number, not a name.
    letters = sum(c.isalpha() for c in value)
    return letters >= max(3, len(value) * 0.4)


class ChemicalEtl:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.chemicals = ChemicalRepository(session)
        self.evidence = EvidenceRepository(session)
        self.pubchem = PubChemConnector()
        self.literature = EuropePMCConnector()
        self.kb_version = get_settings().kb_version

    def ingest(self, name: str, *, with_evidence: bool = True) -> EtlOutcome:
        """Build or refresh one dossier."""
        outcome = EtlOutcome()
        clean_name = name.strip()
        if not clean_name:
            outcome.error = "empty name"
            return outcome

        # Labels use trade names; PubChem indexes chemical names. "Red #40"
        # 404s, "Allura Red AC" resolves to CID 33258 — the same substance,
        # and one of the colours carrying a mandatory EU warning. Without this
        # the most hazard-relevant ingredients on a food pack were exactly the
        # ones research could never find.
        lookup_name = expand_alias(clean_name)
        if lookup_name != clean_name:
            outcome.notes.append(f"alias {clean_name!r} -> {lookup_name!r}")

        # The id comes from the RESOLVED name so "Red #40", "FD&C Red No. 40"
        # and "Allura Red AC" converge on one dossier instead of three.
        chemical_id = _slug(lookup_name)
        if not chemical_id:
            outcome.error = f"could not derive an id from {name!r}"
            return outcome
        outcome.chemical_id = chemical_id

        cid = self.pubchem.resolve_cid(lookup_name)
        outcome.external_calls += 1
        if cid is None:
            outcome.error = f"PubChem has no compound named {lookup_name!r}"
            return outcome
        outcome.notes.append(f"pubchem cid {cid}")

        # The dossier is filed under the chemical name; the label wording is
        # kept as a synonym so the next scan of this pack resolves by hash on
        # the first rung instead of researching it again.
        display_source = lookup_name

        properties = self.pubchem.properties(cid)
        outcome.external_calls += 1
        synonyms = self.pubchem.synonyms(cid)
        outcome.external_calls += 1

        cas = next((s for s in synonyms if _CAS_IN_TEXT.fullmatch(s.strip())), None)

        existing = self.chemicals.get_many([chemical_id])
        outcome.created = chemical_id not in existing

        self.chemicals.upsert_chemical(
            chemical_id=chemical_id,
            inci_name=display_source.upper(),
            display_name=display_source.title(),
            cas=cas,
            formula=properties.get("MolecularFormula"),
            chem_class=None,
            functions=[],
            kb_version=self.kb_version,
            # Never auto-publish. A dossier assembled by regex from a public
            # API is a starting point for a reviewer, not a finished record.
            review_status="draft",
        )

        self.chemicals.add_synonym(chemical_id, display_source, kind="inci")
        if clean_name.lower() != display_source.lower():
            # The wording actually printed on the pack. This is the one that
            # makes the next scan resolve without another PubChem round trip.
            self.chemicals.add_synonym(chemical_id, clean_name, kind="synonym")
        outcome.synonyms_added += 1
        if cas:
            self.chemicals.add_synonym(chemical_id, cas, kind="cas")
            outcome.synonyms_added += 1

        for synonym in synonyms:
            if not _useful_synonym(synonym) or synonym.strip().lower() == clean_name.lower():
                continue
            self.chemicals.add_synonym(chemical_id, synonym, kind="synonym")
            outcome.synonyms_added += 1
            if outcome.synonyms_added >= 30:
                break

        outcome.assertions_added = self._ingest_hazards(chemical_id, cid, outcome)

        if with_evidence:
            outcome.evidence_added = self._ingest_evidence(chemical_id, clean_name, outcome)

        return outcome

    def _ingest_hazards(self, chemical_id: str, cid: int, outcome: EtlOutcome) -> int:
        sections = self.pubchem.hazard_sections(cid)
        outcome.external_calls += 1
        if not sections:
            outcome.notes.append("no hazard sections in PubChem")
            return 0

        # Replace this chemical's PubChem-sourced assertions rather than
        # appending — a refresh must not stack duplicates every run.
        self.session.execute(
            sql(
                "DELETE FROM chemical_assertion "
                "WHERE chemical_id = :cid AND source = 'PubChem'"
            ),
            {"cid": chemical_id},
        )

        added = 0
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"

        for code in extract_ghs_codes(sections.get("ghs", [])):
            self.chemicals.add_assertion(
                chemical_id,
                "hazard",
                "ghs_code",
                value=code,
                source="PubChem",
                source_url=url,
                evidence_grade="B",
                fetched_at=now,
                kb_version=self.kb_version,
            )
            added += 1

        iarc = extract_iarc_group(
            sections.get("carcinogenicity", []) + sections.get("toxicity_summary", [])
        )
        if iarc:
            self.chemicals.add_assertion(
                chemical_id,
                "hazard",
                "iarc_group",
                value=iarc,
                source="PubChem",
                source_url=url,
                evidence_grade="B",
                fetched_at=now,
                kb_version=self.kb_version,
            )
            added += 1
            outcome.notes.append(f"IARC group {iarc}")

        return added

    def _ingest_evidence(self, chemical_id: str, name: str, outcome: EtlOutcome) -> int:
        query = f'("{name}") AND (toxicity OR absorption OR safety OR carcinogenicity)'
        papers = self.literature.search(query, limit=8)
        outcome.external_calls += 1

        if not papers:
            outcome.notes.append("no literature found")
            return 0

        added = 0
        for paper in papers:
            score = score_independence(paper.grants)

            self.evidence.upsert_document(
                paper.source_id,
                title=paper.title,
                container=paper.container,
                url=paper.url,
                tier=SourceTier.T3_PRIMARY,
                year=paper.year,
                study_design=paper.study_design,
                funder_class=str(score.funder_class.value),
                independence=score.value,
                abstract=paper.abstract,
            )
            if paper.abstract:
                self.evidence.add_chunk(paper.source_id, 0, paper.abstract)
            self.evidence.link_chemical(chemical_id, paper.source_id, relation="general")
            added += 1

        return added

    def review(self, chemical_id: str, reviewer: str) -> bool:
        """Promote a reviewed dossier to published."""
        result = self.session.execute(
            sql(
                "UPDATE chemical SET review_status = 'published', reviewed_by = :who, "
                "reviewed_at = UTC_TIMESTAMP(3) WHERE chemical_id = :cid"
            ),
            {"who": reviewer[:191], "cid": chemical_id},
        )
        return (result.rowcount or 0) > 0

    def pending_review(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.session.execute(
            sql(
                "SELECT chemical_id, display_name, cas, kb_version, created_at "
                "FROM chemical WHERE review_status = 'draft' "
                "ORDER BY created_at DESC LIMIT :lim"
            ),
            {"lim": limit},
        ).mappings().all()
        return [dict(r) for r in rows]


# -- label-name aliases -----------------------------------------------------
#
# Ingredient panels use regulatory trade names; PubChem indexes chemical
# names. Verified against the live API: "red #40" 404s, "Allura Red AC" is
# CID 33258. Three of the four colours below carry the mandatory EU warning
# "may have an adverse effect on activity and attention in children", so a
# scanner that cannot look them up misses the most flaggable ingredients on a
# typical confectionery pack.
#
# Only unambiguous one-to-one mappings belong here. Mixtures with no single
# CID — caramel colour, shellac, carnauba wax, "natural flavours" — are left
# out deliberately: inventing a compound for them would attach real toxicology
# to the wrong substance.

_ALIASES: dict[str, str] = {
    # FD&C colours, by their chemical names.
    "red 40": "Allura Red AC",
    "red 40 lake": "Allura Red AC",
    "allura red": "Allura Red AC",
    "yellow 5": "Tartrazine",
    "yellow 5 lake": "Tartrazine",
    "yellow 6": "Sunset Yellow FCF",
    "yellow 6 lake": "Sunset Yellow FCF",
    "sunset yellow": "Sunset Yellow FCF",
    "blue 1": "Brilliant Blue FCF",
    "blue 1 lake": "Brilliant Blue FCF",
    "brilliant blue": "Brilliant Blue FCF",
    "blue 2": "Indigo carmine",
    "indigotine": "Indigo carmine",
    "red 3": "Erythrosine",
    "green 3": "Fast Green FCF",
    "carmine": "Carminic acid",
    "cochineal": "Carminic acid",
    # Common abbreviations and vernacular names.
    "msg": "Monosodium glutamate",
    "bha": "Butylated hydroxyanisole",
    "bht": "Butylated hydroxytoluene",
    "tbhq": "tert-Butylhydroquinone",
    "edta": "Edetic acid",
    "vitamin c": "Ascorbic acid",
    "vitamin e": "Tocopherol",
    "vitamin b3": "Niacin",
    "baking soda": "Sodium bicarbonate",
    "cream of tartar": "Potassium bitartrate",
    "table salt": "Sodium chloride",
    "epsom salt": "Magnesium sulfate",
    "saltpeter": "Potassium nitrate",
}

# "FD&C Red No. 40", "Red #40", "Red 40" and "F D & C red no 40" all reduce to
# "red 40" before the table is consulted.
_ALIAS_NORMALISE = re.compile(r"(?:\bfd\s*&?\s*c\b|\bno\.?\b|[#().,])", re.IGNORECASE)


def expand_alias(name: str) -> str:
    """Map a label's wording to the name PubChem will recognise.

    Returns the input unchanged when there is no confident mapping — an
    unknown ingredient is a correct outcome, a wrong dossier is not.
    """
    reduced = _ALIAS_NORMALISE.sub(" ", (name or "").lower())
    reduced = re.sub(r"\s+", " ", reduced).strip()
    return _ALIASES.get(reduced, name)
