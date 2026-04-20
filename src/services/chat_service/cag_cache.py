import json
import time
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger


# ── Defaults ─────────────────────────────────────────────────

_DEFAULT_COLLECTION = "cag_cache"


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


def _normalize_evidence(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize evidence into a consistent structure for caching.

    Supported input shapes:
      - evidence (preferred): List[Dict] with Qdrant payload-ish keys
          { "url", "title", "chunk_text", "strategy", "chunk_index", ...metadata }
      - evidence_urls (legacy): List[str]
    """
    evidence = response.get("evidence")
    if isinstance(evidence, list):
        normalized: list[dict] = []
        for item in evidence:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"url": item})
        return {"evidence": normalized, "evidence_urls": [e.get("url", "") for e in normalized if e.get("url")]}

    urls = response.get("evidence_urls")
    if isinstance(urls, list):
        return {"evidence": [{"url": u} for u in urls if isinstance(u, str)], "evidence_urls": [u for u in urls if isinstance(u, str)]}

    return {"evidence": [], "evidence_urls": []}


class CAGCache:
    """
    Qdrant-backed semantic cache for pre-computed RAG responses.

    Each point stores:
        vector   → query embedding (float32)
        payload  → query, answer, evidence_urls, ts

    Lookup is a KNN-1 search; a hit is declared when
    ``cosine_similarity ≥ similarity_threshold``.

    Usage::

        cache = CAGCache(embedder=embedder)

        # Semantic lookup
        cached = cache.get("How many leave days?")
        if cached:
            return cached["answer"]

        # Store new entry (semantically indexed)
        cache.set("What is the leave policy?", {"answer": "...", "evidence_urls": [...]})
    """

    def __init__(
        self,
        embedder: Any,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Initialise the semantic CAG cache.

        Args:
            embedder: Object with ``embed_query(text) -> List[float]``.
            collection_name: Qdrant collection for CAG (default: ``cag_cache``).
            dim: Embedding dimension (auto-detected from config if None).
            similarity_threshold: Min cosine similarity for a hit (0.90–0.95).
            ttl_seconds: Entries older than this are ignored; 0 → no expiry.
        """
        from infrastructure.config import (
            CAG_SIMILARITY_THRESHOLD,
            CAG_CACHE_TTL,
            EMBEDDING_DIM,
            CAG_COLLECTION_NAME,
        )

        self.embedder = embedder
        self.collection_name = collection_name or CAG_COLLECTION_NAME
        self.dim = dim or EMBEDDING_DIM
        self.similarity_threshold = similarity_threshold or CAG_SIMILARITY_THRESHOLD
        self.ttl_seconds = ttl_seconds or CAG_CACHE_TTL

        # Ensure the Qdrant collection exists
        self._available = False
        try:
            from infrastructure.db.qdrant_client import (
                get_qdrant_client,
                collection_exists,
            )

            self._client = get_qdrant_client()
            if not collection_exists(self.collection_name):
                self._create_collection()
            self._available = True
            logger.info(
                "✓ CAG cache ready (Qdrant collection='{}', dim={}, threshold={:.2f})",
                self.collection_name,
                self.dim,
                self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning(
                "CAG cache DISABLED — Qdrant unavailable: {}. "
                "All lookups will miss; every query runs full RAG.",
                exc,
            )

    # ── collection management ─────────────────────────────────

    def _create_collection(self) -> None:
        """Create the Qdrant collection for CAG cache."""
        from qdrant_client.http.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
                on_disk=False,  # keep in-memory for speed
            ),
        )
        logger.info(
            "Created CAG cache collection '{}' (dim={}, COSINE)",
            self.collection_name,
            self.dim,
        )

    # ── public API ────────────────────────────────────────────

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Semantic cache lookup via KNN-1 search.

        Embeds *query*, searches the Qdrant HNSW index, and returns
        the cached response if ``cosine_similarity ≥ threshold``.

        Args:
            query: Natural-language question.

        Returns:
            Dict with ``query``, ``answer``, ``evidence_urls``, ``ts``,
            ``score`` — or ``None`` on miss.
        """
        if not self._available:
            return None

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed: {}", exc)
            return None

        # KNN-1 search
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=1,
                score_threshold=self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning("CAG cache GET error: {}", exc)
            return None

        if not response.points:
            return None

        hit = response.points[0]
        similarity = hit.score
        payload = hit.payload or {}

        # TTL filtering: skip entries older than ttl_seconds
        if self.ttl_seconds and self.ttl_seconds > 0:
            entry_ts = payload.get("ts", 0)
            if entry_ts and (time.time() - float(entry_ts)) > self.ttl_seconds:
                return None

        cached_query = payload.get("query", "")
        logger.info(
            "CAG cache HIT (sim={:.3f}): '{}' → matched '{}'",
            similarity,
            query[:50],
            cached_query[:50],
        )

        evidence_raw = payload.get("evidence_urls", "[]")
        evidence_urls = _safe_json_loads(evidence_raw, default=[])

        # New: richer evidence objects (preferred)
        evidence = _safe_json_loads(payload.get("evidence", "[]"), default=[])
        if not evidence and evidence_urls:
            evidence = [{"url": u} for u in evidence_urls if isinstance(u, str)]

        return {
            "query": cached_query,
            "answer": payload.get("answer", ""),
            "evidence": evidence,
            "evidence_urls": evidence_urls,
            "ts": float(payload.get("ts", 0)),
            "score": similarity,
        }

    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response, indexed by the query's embedding.

        Args:
            query: Original user query.
            response: Dict with ``answer`` and optionally ``evidence_urls``.
        """
        if not self._available:
            return

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed on SET: {}", exc)
            return

        from qdrant_client.http.models import PointStruct

        point_id = str(uuid.uuid4())
        evidence_norm = _normalize_evidence(response)
        payload = {
            "query": query,
            "answer": response.get("answer", ""),
            # Keep legacy field for backward compatibility
            "evidence_urls": json.dumps(evidence_norm.get("evidence_urls", [])),
            # New: store full evidence objects (Qdrant payload-style)
            "evidence": json.dumps(evidence_norm.get("evidence", [])),
            "ts": time.time(),
        }

        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=query_vec, payload=payload)],
            )
            logger.debug("CAG cache SET: '{}' → point={}", query[:60], point_id)
        except Exception as exc:
            logger.warning("CAG cache SET error: {}", exc)

    def clear(self) -> None:
        """
        Drop and recreate the CAG cache collection.

        All cached entries are removed. The collection is recreated
        immediately so the cache is ready for new entries.
        """
        if not self._available:
            return

        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Dropped CAG cache collection '{}'", self.collection_name)
        except Exception:
            pass  # Collection may not exist

        self._create_collection()
        logger.info("CAG cache cleared and collection recreated")

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with ``total_cached``, ``backend``, ``collection``,
            ``similarity_threshold``, ``ttl_seconds``.
        """
        return {
            "total_cached": self._count(),
            "backend": "qdrant",
            "collection": self.collection_name,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds,
            "available": self._available,
        }

    # ── internal helpers ──────────────────────────────────────

    def _count(self) -> int:
        """Return number of cached entries."""
        if not self._available:
            return 0
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    # ── dunder helpers ────────────────────────────────────────

    def __len__(self) -> int:
        return self._count()

    def __contains__(self, query: str) -> bool:
        return self.get(query) is not None

    def __repr__(self) -> str:
        return (
            f"CAGCache(collection='{self.collection_name}', "
            f"threshold={self.similarity_threshold}, "
            f"ttl={self.ttl_seconds}s, "
            f"entries={self._count()}, "
            f"backend='qdrant')"
        )
