"""Token → chemical id (arch.md 8.3 step 3).

The ladder, cheapest and most certain first:

  1. exact INCI / normalised synonym hash   — one batched query for the panel
  2. CAS / EC / E-number pattern            — structured identifiers in the text
  3. fuzzy: OCR confusion model, then edit distance

An unresolved token is surfaced as "not recognised" (arch.md 8.3 step 4). It is
never silently researched and never silently dropped — both of those turn a
gap in the KB into an invisible gap in the analysis.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from packages.common.text import normalise_ingredient
from packages.domain.enums import ResolutionMethod
from packages.domain.models import ResolvedIngredient
from packages.product.parser import ParsedToken
from packages.storage.repositories.chemicals import ChemicalRepository, synonym_hash
from packages.storage.vectors import get_index

logger = logging.getLogger(__name__)

_CAS = re.compile(r"\b(\d{2,7})-(\d{2})-(\d)\b")
_EC = re.compile(r"\b(\d{3})-(\d{3})-(\d)\b")
_E_NUMBER = re.compile(r"\b[eE]\s?(\d{3}[a-z]?)\b")
_CI_NUMBER = re.compile(r"\bC\.?I\.?\s?(\d{5})\b", re.IGNORECASE)

# What OCR actually confuses on a printed ingredient panel. Applied as
# substitutions when an exact match fails, before falling back to edit
# distance — a targeted swap is far more precise than generic fuzziness.
_OCR_CONFUSIONS: list[tuple[str, str]] = [
    ("0", "o"), ("o", "0"),
    ("1", "l"), ("l", "1"), ("1", "i"), ("i", "1"),
    ("5", "s"), ("s", "5"),
    ("8", "b"), ("b", "8"),
    ("rn", "m"), ("m", "rn"),
    ("cl", "d"), ("d", "cl"),
    ("vv", "w"), ("w", "vv"),
    ("nn", "m"),
    ("ii", "u"),
]

FUZZY_THRESHOLD = 0.88
FUZZY_MIN_LENGTH = 5
# Deliberately high. A wrong chemical id produces a confident hazard verdict
# about a substance that is not in the product, which is worse than "not
# recognised" — so this rung only fires on a near-certain match.
EMBEDDING_THRESHOLD = 0.90


def validate_cas(cas: str) -> bool:
    """CAS check digit: sum of digits weighted by position from the right."""
    match = _CAS.fullmatch(cas.strip())
    if not match:
        return False
    body = (match.group(1) + match.group(2))[::-1]
    total = sum(int(char) * (index + 1) for index, char in enumerate(body))
    return total % 10 == int(match.group(3))


def extract_identifiers(text: str) -> dict[str, str]:
    """Pull structured identifiers out of a token's text and qualifiers."""
    found: dict[str, str] = {}

    cas_match = _CAS.search(text)
    if cas_match and validate_cas(cas_match.group(0)):
        found["cas"] = cas_match.group(0)

    ec_match = _EC.search(text)
    if ec_match:
        found["ec"] = ec_match.group(0)

    e_match = _E_NUMBER.search(text)
    if e_match:
        found["e_number"] = "E" + e_match.group(1).upper()

    ci_match = _CI_NUMBER.search(text)
    if ci_match:
        found["ci"] = "CI " + ci_match.group(1)

    return found


def _ocr_variants(normalised: str, max_variants: int = 40) -> list[str]:
    """Single-substitution variants under the confusion model."""
    variants: set[str] = set()
    for wrong, right in _OCR_CONFUSIONS:
        start = 0
        while len(variants) < max_variants:
            index = normalised.find(wrong, start)
            if index == -1:
                break
            variants.add(normalised[:index] + right + normalised[index + len(wrong):])
            start = index + 1
        if len(variants) >= max_variants:
            break
    variants.discard(normalised)
    return list(variants)


@dataclass
class _Candidate:
    chemical_id: str
    norm_text: str
    display: str


class _CorpusCache:
    """Process-wide synonym corpus, reloaded when the KB changes.

    Per-instance loading meant every scan pulled the whole synonym table over
    the wire. At the target of ~5,000 dossiers that is tens of thousands of
    rows per request, which would dominate scan latency.
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._candidates: list[_Candidate] = []
        self._token: Optional[str] = None
        self._loaded_at: float = 0.0
        self._lock = threading.Lock()

    def get(self, repo: ChemicalRepository, ttl_seconds: int = 600) -> list[_Candidate]:
        token = str(repo.synonym_count())
        with self._lock:
            fresh = (
                self._candidates
                and self._token == token
                and (time.time() - self._loaded_at) < ttl_seconds
            )
            if fresh:
                return self._candidates

            self._candidates = [
                _Candidate(chemical_id=cid, norm_text=norm, display=syn)
                for cid, syn, norm in repo.all_synonyms()
            ]
            self._token = token
            self._loaded_at = time.time()
            logger.info("synonym corpus loaded: %d entries", len(self._candidates))
            return self._candidates


_corpus_cache = _CorpusCache()


class IngredientResolver:
    """Resolves a whole panel at once.

    Batching matters: a 30-token panel resolved one token at a time is 30 round
    trips, and the fuzzy stage would reload the synonym corpus each time.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = ChemicalRepository(session)

    def _load_corpus(self) -> list[_Candidate]:
        return _corpus_cache.get(self.repo)

    def resolve_panel(self, tokens: list[ParsedToken]) -> list[ResolvedIngredient]:
        if not tokens:
            return []

        resolved: list[ResolvedIngredient] = [
            ResolvedIngredient(
                raw_token=token.raw,
                position=token.position,
                qualifiers=token.qualifiers,
            )
            for token in tokens
        ]

        # -- stage 1: exact hash, one query for the whole panel ------------
        hashes = [synonym_hash(token.text) for token in tokens]
        exact = self.repo.by_synonym_hash([h for h in hashes if h])

        for index, token in enumerate(tokens):
            hit = exact.get(hashes[index])
            if hit is None:
                continue
            chemical_id, kind = hit
            resolved[index].chemical_id = chemical_id
            resolved[index].confidence = 1.0
            resolved[index].resolution_method = (
                ResolutionMethod.EXACT_INCI if kind == "inci" else ResolutionMethod.SYNONYM
            )

        # -- stage 2: structured identifiers ------------------------------
        for index, token in enumerate(tokens):
            if resolved[index].resolved:
                continue
            haystack = " ".join([token.text, *token.qualifiers])
            identifiers = extract_identifiers(haystack)
            if not identifiers:
                continue

            chemical_id = self.repo.by_identifier(
                cas=identifiers.get("cas"),
                ec=identifiers.get("ec"),
                e_number=identifiers.get("e_number"),
            )
            if chemical_id is None:
                continue

            resolved[index].chemical_id = chemical_id
            resolved[index].confidence = 0.98
            resolved[index].resolution_method = (
                ResolutionMethod.CAS
                if "cas" in identifiers
                else ResolutionMethod.EC
                if "ec" in identifiers
                else ResolutionMethod.E_NUMBER
            )

        # -- stage 3: qualifiers as fallback surface ----------------------
        # "Tocopherol (Vitamin E)" — when the INCI term misread, the common
        # name in brackets often did not.
        pending = [i for i, r in enumerate(resolved) if not r.resolved and tokens[i].qualifiers]
        if pending:
            qualifier_hashes: dict[int, list[str]] = {
                i: [synonym_hash(q) for q in tokens[i].qualifiers] for i in pending
            }
            flat = [h for hs in qualifier_hashes.values() for h in hs if h]
            qualifier_hits = self.repo.by_synonym_hash(flat)
            for index, hs in qualifier_hashes.items():
                for h in hs:
                    hit = qualifier_hits.get(h)
                    if hit:
                        resolved[index].chemical_id = hit[0]
                        resolved[index].confidence = 0.85
                        resolved[index].resolution_method = ResolutionMethod.SYNONYM
                        break

        # -- stage 4: fuzzy ------------------------------------------------
        unresolved = [i for i, r in enumerate(resolved) if not r.resolved]
        if unresolved:
            self._resolve_fuzzy(tokens, resolved, unresolved)

        # -- names for whatever resolved -----------------------------------
        chemical_ids = [r.chemical_id for r in resolved if r.chemical_id]
        if chemical_ids:
            dossiers = self.repo.get_many(chemical_ids)
            for item in resolved:
                if item.chemical_id and item.chemical_id in dossiers:
                    dossier = dossiers[item.chemical_id]
                    item.display_name = dossier.get("display_name") or dossier.get("inci_name")

        return resolved

    def _resolve_fuzzy(
        self,
        tokens: list[ParsedToken],
        resolved: list[ResolvedIngredient],
        indexes: list[int],
    ) -> None:
        corpus = self._load_corpus()
        if not corpus:
            return

        by_norm: dict[str, _Candidate] = {c.norm_text: c for c in corpus}

        for index in indexes:
            normalised = normalise_ingredient(tokens[index].text)
            if len(normalised) < FUZZY_MIN_LENGTH:
                continue

            # 4a: OCR confusion substitutions, checked as exact matches.
            matched = False
            for variant in _ocr_variants(normalised):
                candidate = by_norm.get(variant)
                if candidate is not None:
                    resolved[index].chemical_id = candidate.chemical_id
                    resolved[index].confidence = 0.82
                    resolved[index].resolution_method = ResolutionMethod.FUZZY
                    matched = True
                    break
            if matched:
                continue

            # 4b: edit distance, length-bucketed so a 30-token panel does not
            # do 30 full scans of the corpus.
            best_score, best_candidate = 0.0, None
            length = len(normalised)
            for candidate in corpus:
                if abs(len(candidate.norm_text) - length) > 3:
                    continue
                score = SequenceMatcher(None, normalised, candidate.norm_text).ratio()
                if score > best_score:
                    best_score, best_candidate = score, candidate

            if best_candidate is not None and best_score >= FUZZY_THRESHOLD:
                resolved[index].chemical_id = best_candidate.chemical_id
                resolved[index].confidence = round(best_score * 0.9, 3)
                resolved[index].resolution_method = ResolutionMethod.FUZZY
                logger.debug(
                    "fuzzy: %r -> %s (%.3f)", tokens[index].text, best_candidate.chemical_id, best_score
                )
                continue

            # 4c: embedding nearest neighbour — the last rung of arch.md 8.3.
            # Catches names no edit distance reaches: a common name against an
            # INCI term, or a translated ingredient.
            self._resolve_by_embedding(tokens[index], resolved[index])

    def _resolve_by_embedding(self, token: ParsedToken, target: ResolvedIngredient) -> None:
        from packages.chains.embeddings import embed_query

        index = get_index("chemical_synonyms")
        token_count = str(self.repo.synonym_count())
        if index.is_stale(token_count):
            rows = self.session.execute(
                sql_text(
                    "SELECT id, chemical_id, synonym, embedding FROM chemical_synonym "
                    "WHERE embedding IS NOT NULL"
                )
            ).mappings().all()
            index.load(
                [
                    (str(r["id"]), r["embedding"], {"chemical_id": r["chemical_id"], "synonym": r["synonym"]})
                    for r in rows
                ],
                version_token=token_count,
            )

        if not index.size:
            return  # synonyms not embedded yet; the backfill has not run

        query = embed_query(token.text)
        if query is None:
            return

        hits = index.search(query, top_k=1, min_score=EMBEDDING_THRESHOLD)
        if not hits:
            return

        target.chemical_id = hits[0].payload.get("chemical_id")
        target.confidence = round(hits[0].score * 0.85, 3)
        target.resolution_method = ResolutionMethod.EMBEDDING
        logger.debug(
            "embedding: %r -> %s (%.3f)", token.text, target.chemical_id, hits[0].score
        )
