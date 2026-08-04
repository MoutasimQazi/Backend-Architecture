"""Literature connectors: EuropePMC, PubMed, Crossref, OpenAlex, PubChem.

All fetch through the broker. Unlike the old `app.py`, these are *not* called
per ingredient inside a request — they run in ETL and deep-research jobs, and
their output lands in `evidence_document` for the runtime to read.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from packages.guards.fetch_broker import FetchError, get_broker

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    source_id: str
    title: str = ""
    abstract: str = ""
    url: Optional[str] = None
    container: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    grants: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def study_design(self) -> Optional[str]:
        """Best guess at design, from publication types and the title.

        Design drives ranking (arch.md 9.3), so getting it roughly right on the
        cheap beats getting it exactly right expensively.
        """
        haystack = " ".join(self.publication_types + [self.title]).lower()
        if "meta-analysis" in haystack or "meta analysis" in haystack:
            return "meta-analysis"
        if "systematic review" in haystack:
            return "systematic-review"
        if "randomized controlled trial" in haystack or "randomised controlled" in haystack:
            return "rct"
        if "clinical trial" in haystack:
            return "clinical-trial"
        if "cohort" in haystack:
            return "cohort"
        if "case-control" in haystack or "case control" in haystack:
            return "case-control"
        if "review" in haystack:
            return "review"
        if "in vitro" in haystack:
            return "in-vitro"
        if "animal" in haystack or "mice" in haystack or "rats" in haystack:
            return "animal"
        return None


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()


class EuropePMCConnector:
    """EuropePMC covers PubMed plus preprints and has a generous open API."""

    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        try:
            response = get_broker().get(
                self.BASE,
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": max(1, min(limit, 25)),
                    "resultType": "core",
                },
            )
        except FetchError as exc:
            logger.info("europepmc search failed: %s", exc)
            return []

        if response.status_code != 200:
            return []

        try:
            payload = response.json()
        except ValueError:
            return []

        results = (payload.get("resultList") or {}).get("result") or []
        papers: list[Paper] = []

        for item in results:
            doi = item.get("doi")
            pmid = item.get("pmid")
            source_id = f"doi:{doi}" if doi else (f"pmid:{pmid}" if pmid else None)
            if not source_id:
                continue

            grants = [
                g.get("agency")
                for g in (item.get("grantsList") or {}).get("grant", [])
                if isinstance(g, dict) and g.get("agency")
            ]

            papers.append(
                Paper(
                    source_id=source_id,
                    title=_clean(item.get("title")),
                    abstract=_clean(item.get("abstractText"))[:8000],
                    url=(f"https://doi.org/{doi}" if doi
                         else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None),
                    container=_clean(item.get("journalTitle")),
                    year=int(item["pubYear"]) if str(item.get("pubYear", "")).isdigit() else None,
                    doi=doi,
                    pmid=pmid,
                    authors=[_clean(item.get("authorString"))][:1],
                    publication_types=[
                        _clean(t) for t in (item.get("pubTypeList") or {}).get("pubType", [])
                    ],
                    grants=[_clean(g) for g in grants],
                    source="europepmc",
                )
            )

        return papers


class PubChemConnector:
    """PubChem for chemical identity and hazard sections."""

    BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"

    def resolve_cid(self, name: str) -> Optional[int]:
        try:
            response = get_broker().get(f"{self.BASE}/compound/name/{name}/cids/JSON")
        except FetchError as exc:
            logger.info("pubchem cid lookup failed for %r: %s", name, exc)
            return None

        if response.status_code != 200:
            return None
        try:
            cids = (response.json().get("IdentifierList") or {}).get("CID") or []
        except ValueError:
            return None
        return int(cids[0]) if cids else None

    def properties(self, cid: int) -> dict[str, Any]:
        props = "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES"
        try:
            response = get_broker().get(f"{self.BASE}/compound/cid/{cid}/property/{props}/JSON")
        except FetchError:
            return {}
        if response.status_code != 200:
            return {}
        try:
            rows = (response.json().get("PropertyTable") or {}).get("Properties") or []
        except ValueError:
            return {}
        return rows[0] if rows else {}

    def synonyms(self, cid: int, limit: int = 40) -> list[str]:
        try:
            response = get_broker().get(f"{self.BASE}/compound/cid/{cid}/synonyms/JSON")
        except FetchError:
            return []
        if response.status_code != 200:
            return []
        try:
            info = (response.json().get("InformationList") or {}).get("Information") or []
        except ValueError:
            return []
        if not info:
            return []
        return [_clean(s) for s in (info[0].get("Synonym") or [])[:limit]]

    def hazard_sections(self, cid: int) -> dict[str, list[str]]:
        """Pull GHS and toxicity strings out of the PUG-View record.

        The record is a deep, irregular tree; walking it for headings is more
        robust than indexing by position, which breaks whenever PubChem
        reorganises a page.
        """
        try:
            response = get_broker().get(f"{self.VIEW}/{cid}/JSON")
        except FetchError as exc:
            logger.info("pubchem view failed for cid %s: %s", cid, exc)
            return {}

        if response.status_code != 200:
            return {}
        try:
            payload = response.json()
        except ValueError:
            return {}

        wanted = {
            "GHS Classification": "ghs",
            "Carcinogen Classification": "carcinogenicity",
            "Toxicity Summary": "toxicity_summary",
            "Acute Effects": "acute_toxicity",
            "Health Hazards": "health_hazards",
        }
        found: dict[str, list[str]] = {}

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                heading = node.get("TOCHeading")
                if heading in wanted:
                    key = wanted[heading]
                    found.setdefault(key, []).extend(_strings(node))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        def _strings(node: Any) -> list[str]:
            out: list[str] = []
            if isinstance(node, dict):
                if "String" in node and isinstance(node["String"], str):
                    out.append(_clean(node["String"]))
                for value in node.values():
                    out.extend(_strings(value))
            elif isinstance(node, list):
                for item in node:
                    out.extend(_strings(item))
            return out

        walk(payload.get("Record", {}))
        return {k: sorted(set(v))[:20] for k, v in found.items() if v}


# GHS hazard statement codes, extracted from PubChem's free text.
_GHS_CODE = re.compile(r"\bH(\d{3})\b")


def extract_ghs_codes(strings: list[str]) -> list[str]:
    codes: set[str] = set()
    for text in strings:
        for match in _GHS_CODE.finditer(text):
            codes.add("H" + match.group(1))
    return sorted(codes)


_IARC_GROUP = re.compile(r"\bgroup\s*(1|2A|2B|3|4)\b", re.IGNORECASE)


def extract_iarc_group(strings: list[str]) -> Optional[str]:
    for text in strings:
        if "iarc" not in text.lower():
            continue
        match = _IARC_GROUP.search(text)
        if match:
            return match.group(1).upper()
    return None
