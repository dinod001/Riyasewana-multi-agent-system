import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()


# Ensure `src/` is on sys.path so imports work when running pytest from repo root.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _has_any_embedding_key() -> bool:
    # Project supports OpenRouter routing; embeddings fallback may use OPENAI_API_KEY.
    return bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _has_qdrant_env() -> bool:
    return bool(os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY"))


@pytest.mark.integration
def test_cag_service_generate_smoke() -> None:
    if not _has_any_embedding_key():
        pytest.skip("Missing OPENROUTER_API_KEY or OPENAI_API_KEY")
    if not _has_qdrant_env():
        pytest.skip("Missing QDRANT_URL/QDRANT_API_KEY")

    from infrastructure.db.qdrant_client import count_points
    from infrastructure.llm.embeddings import get_default_embeddings
    from infrastructure.llm.llm_provider import get_chat_llm
    from services.chat_service.cag_cache import CAGCache
    from services.chat_service.cag_service import CAGService
    from services.chat_service.crag_service import CRAGService
    from services.chat_service.rag_service import QdrantRetriever
    from infrastructure.config import QDRANT_COLLECTION_NAME

    if count_points(QDRANT_COLLECTION_NAME) == 0:
        pytest.skip(
            f"Qdrant collection '{QDRANT_COLLECTION_NAME}' has 0 points. Run ingest pipeline first."
        )

    embedder = get_default_embeddings()
    llm = get_chat_llm(temperature=0)

    retriever = QdrantRetriever(embedder=embedder)
    crag = CRAGService(retriever=retriever, llm=llm)
    cache = CAGCache(embedder=embedder)
    cag = CAGService(crag_service=crag, cache=cache)

    query = "Mercedes-Benz price and location?"
    result = cag.generate(query, use_cache=False)

    # Display output when running with: pytest -s
    print("\n--- QUERY ---")
    print(query)
    print("\n--- ANSWER ---")
    print(result.get("answer", ""))

    evidence = result.get("evidence", []) or []
    print("\n--- EVIDENCE (preview) ---")
    for i, item in enumerate(evidence[:5], 1):
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        title = item.get("title", "")
        price = item.get("price", "")
        year = item.get("year", "")
        location = item.get("location", "")
        make = item.get("make", "")
        model = item.get("model", "")
        print(f"{i}. {title} | {price} | {year} | {make} {model} | {location} | {url}")

    assert isinstance(result, dict)
    assert isinstance(result.get("answer"), str)
    assert result["answer"].strip() != ""

    # Evidence should be cache-friendly JSON-like items
    assert isinstance(result.get("evidence"), list)
    assert isinstance(result.get("evidence_urls"), list)

