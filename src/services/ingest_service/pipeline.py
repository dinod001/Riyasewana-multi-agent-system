import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

# Allow running this file directly:
#   python src/services/ingest_service/pipeline.py
# by ensuring the `src/` directory is on sys.path so that
# `import infrastructure...` and `import services...` work.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from infrastructure.config import (
    QDRANT_COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    PROVIDER,
)
from infrastructure.llm.embeddings import get_default_embeddings
from infrastructure.db.qdrant_client import (
    ensure_collection,
    upsert_chunks,
    collection_info,
)
from services.ingest_service.chunkers import custom_chunker


def _setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
        colorize=True,
        enqueue=True,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    # Fast-fail with a helpful message when API keys are missing
    if PROVIDER == "openrouter":
        if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
            raise RuntimeError(
                "Missing API key for embeddings. Set OPENROUTER_API_KEY (recommended) "
                "or OPENAI_API_KEY in your environment/.env, then retry."
            )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "Missing OPENAI_API_KEY for embeddings. Set it in your environment/.env, then retry."
            )

    embeddings = get_default_embeddings(batch_size=EMBEDDING_BATCH_SIZE)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _normalize_for_qdrant(
    chunk_docs: list[dict],
    *,
    strategy: str = "custom",
) -> Tuple[list[Dict[str, Any]], list[str]]:
    """
    Convert chunker output into the structure expected by qdrant_client.upsert_chunks.

    Returns:
        (qdrant_chunks, texts_for_embedding)
    """
    qdrant_chunks: list[Dict[str, Any]] = []
    texts: list[str] = []

    for idx, doc in enumerate(chunk_docs):
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        metadata = doc.get("metadata") or {}
        url = metadata.get("source_link") or metadata.get("url") or ""
        title = metadata.get("title") or ""

        chunk: Dict[str, Any] = {
            "text": text,
            "url": url,
            "title": title,
            "strategy": strategy,
            "chunk_index": idx,
        }

        # Store the rest as payload metadata (nested metadata also fine)
        for k, v in metadata.items():
            if k not in ("source_link", "url", "title"):
                chunk[k] = v

        qdrant_chunks.append(chunk)
        texts.append(text)

    return qdrant_chunks, texts

def run_ingest_pipeline(data_path: Path) -> None:
    _setup_logging()

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    t0 = time.perf_counter()
    logger.info("Starting ingest pipeline")
    logger.info("Data path: {}", str(data_path))
    logger.info("Qdrant collection: {}", QDRANT_COLLECTION_NAME)
    logger.info("Embedding batch size: {}", EMBEDDING_BATCH_SIZE)

    try:
        t_chunk = time.perf_counter()
        raw_docs = custom_chunker(data_path)
        qdrant_chunks, texts = _normalize_for_qdrant(raw_docs, strategy="custom")
        logger.info(
            "Chunking done: {} raw docs -> {} chunks (took {:.2f}s)",
            len(raw_docs),
            len(qdrant_chunks),
            time.perf_counter() - t_chunk,
        )

        if not qdrant_chunks:
            logger.warning("No chunks produced; nothing to ingest.")
            return

        t_embed = time.perf_counter()
        embeddings = embed_texts(texts)
        logger.info(
            "Embedding done: {} vectors (took {:.2f}s)",
            len(embeddings),
            time.perf_counter() - t_embed,
        )

        if len(embeddings) != len(qdrant_chunks):
            raise RuntimeError(
                f"Embedding mismatch: {len(qdrant_chunks)} chunks vs {len(embeddings)} vectors"
            )

        t_collection = time.perf_counter()
        ensure_collection(QDRANT_COLLECTION_NAME)
        logger.info(
            "Collection ensured (took {:.2f}s)",
            time.perf_counter() - t_collection,
        )

        t_upsert = time.perf_counter()
        upserted = upsert_chunks(qdrant_chunks, embeddings, collection_name=QDRANT_COLLECTION_NAME)
        logger.info(
            "Upsert done: {} points (took {:.2f}s)",
            upserted,
            time.perf_counter() - t_upsert,
        )

        # Verify
        t_verify = time.perf_counter()
        info = collection_info(QDRANT_COLLECTION_NAME)
        logger.info(
            "Verify collection: name={} status={} points_count={} indexed_vectors_count={} dim={} distance={} (took {:.2f}s)",
            info.get("name"),
            info.get("status"),
            info.get("points_count"),
            info.get("indexed_vectors_count"),
            info.get("vector_size"),
            info.get("distance"),
            time.perf_counter() - t_verify,
        )

        logger.info("Ingest pipeline completed in {:.2f}s", time.perf_counter() - t0)
    except Exception:
        logger.exception("Ingest pipeline failed")
        raise


if __name__ == "__main__":
    # Usage:
    #   python src/services/ingest_service/pipeline.py
    #   python src/services/ingest_service/pipeline.py data/riyasewana_search_cars.json
    data_arg = sys.argv[1] if len(sys.argv) > 1 else "data/riyasewana_search_cars.json"
    run_ingest_pipeline(Path(data_arg))