"""Vector storage and search without pgvector.

The host has MySQL, so embeddings live in VARBINARY columns as packed float32
and similarity is computed in-process with numpy. For the corpus sizes this
system actually has — a few thousand FAQ surfaces, a few tens of thousands of
chemical synonyms — a brute-force matmul is well under a millisecond and beats
the round-trip an index would save.

The interface (`search`) is what the retrievers depend on, so swapping in a real
vector index later is a change to this file alone.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is in requirements
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def pack(vector: Sequence[float]) -> bytes:
    """float32 little-endian. 1536 dims = 6144 bytes, inside VARBINARY(8192)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(raw: bytes) -> list[float]:
    if not raw:
        return []
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw[: count * 4]))


def normalise(vector: Sequence[float]) -> list[float]:
    """L2-normalise so cosine similarity reduces to a dot product."""
    if _HAS_NUMPY:
        arr = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return list(arr)
        return list(arr / norm)
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    if _HAS_NUMPY:
        va = np.asarray(a, dtype="float32")
        vb = np.asarray(b, dtype="float32")
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class ScoredId:
    ref: str
    score: float
    payload: dict = field(default_factory=dict)


class VectorIndex:
    """An in-process, L2-normalised matrix over rows loaded from MySQL.

    Reloaded when `version_token` changes — the KB version or a FAQ publish —
    rather than on a timer, so a publish is visible immediately and a quiet
    hour costs nothing.
    """

    def __init__(self, name: str, ttl_seconds: int = 300) -> None:
        self.name = name
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._matrix = None  # numpy array (n, dim) or None
        self._refs: list[str] = []
        self._payloads: list[dict] = []
        self._version_token: Optional[str] = None
        self._loaded_at: float = 0.0

    @property
    def size(self) -> int:
        return len(self._refs)

    def is_stale(self, version_token: Optional[str]) -> bool:
        if self._matrix is None:
            return True
        if version_token is not None and version_token != self._version_token:
            return True
        return (time.time() - self._loaded_at) > self.ttl_seconds

    def load(
        self,
        rows: Sequence[tuple[str, bytes, dict]],
        version_token: Optional[str] = None,
    ) -> None:
        """rows: (ref, packed_embedding, payload)."""
        with self._lock:
            refs: list[str] = []
            payloads: list[dict] = []
            vectors: list[list[float]] = []

            for ref, raw, payload in rows:
                vec = unpack(raw)
                if not vec:
                    continue
                refs.append(ref)
                payloads.append(payload)
                vectors.append(normalise(vec))

            self._refs = refs
            self._payloads = payloads
            self._version_token = version_token
            self._loaded_at = time.time()

            if _HAS_NUMPY and vectors:
                self._matrix = np.asarray(vectors, dtype="float32")
            else:
                self._matrix = vectors or None

            logger.info("vector index '%s' loaded: %d rows", self.name, len(refs))

    def search(self, query: Sequence[float], top_k: int = 20, min_score: float = 0.0) -> list[ScoredId]:
        with self._lock:
            if self._matrix is None or not self._refs:
                return []

            q = normalise(query)
            if not q:
                return []

            if _HAS_NUMPY and hasattr(self._matrix, "shape"):
                if self._matrix.shape[1] != len(q):
                    logger.warning(
                        "vector index '%s' dim %d != query dim %d; embedding model changed?",
                        self.name,
                        self._matrix.shape[1],
                        len(q),
                    )
                    return []
                scores = self._matrix @ np.asarray(q, dtype="float32")
                k = min(top_k, len(self._refs))
                # argpartition is O(n); a full sort of 50k rows would not be.
                idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
                idx = idx[np.argsort(-scores[idx])]
                return [
                    ScoredId(ref=self._refs[i], score=float(scores[i]), payload=self._payloads[i])
                    for i in idx
                    if float(scores[i]) >= min_score
                ]

            scored = [
                ScoredId(ref=ref, score=cosine(q, vec), payload=payload)
                for ref, vec, payload in zip(self._refs, self._matrix, self._payloads)
            ]
            scored.sort(key=lambda s: s.score, reverse=True)
            return [s for s in scored[:top_k] if s.score >= min_score]


_indexes: dict[str, VectorIndex] = {}
_indexes_lock = threading.Lock()


def get_index(name: str, ttl_seconds: int = 300) -> VectorIndex:
    with _indexes_lock:
        if name not in _indexes:
            _indexes[name] = VectorIndex(name, ttl_seconds=ttl_seconds)
        return _indexes[name]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """RRF over several ranked id lists (arch.md 5.2 stage 2).

    Rank-based rather than score-based, so BM25 scores and cosine scores can be
    fused without pretending they share a scale.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, ref in enumerate(ranking, start=1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
