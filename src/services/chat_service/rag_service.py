"""
RAG (Retrieval-Augmented Generation) service using modern LangChain LCEL.

Provides:
- build_rag_chain: Create modern LCEL RAG chain
- RAGService: High-level RAG orchestration class
- Uses LangChain Expression Language (Runnables + | operator)
- Backed by Qdrant Cloud for vector retrieval

Architecture:
    Query → Retriever (Qdrant) → Format Docs → Prompt → LLM → Parse → Answer
    
Modern LCEL approach (NOT legacy chains):
    rag_chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
"""

from loguru import logger
from typing import Any, Dict, List, Optional
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from infrastructure.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD
from services.chat_service.rag_templates import RAG_TEMPLATE
from infrastructure.utils import format_docs
from infrastructure.db.qdrant_client import search_chunks


# ============================================================================
# Qdrant-backed LangChain Retriever
# ============================================================================


class QdrantRetriever(BaseRetriever):
    """
    LangChain-compatible retriever backed by Qdrant Cloud.

    Wraps the low-level ``search_chunks`` function so it can be used
    seamlessly inside LCEL chains (``retriever | format_docs``).
    """

    embedder: Any = None
    top_k: int = TOP_K_RESULTS
    score_threshold: float = SIMILARITY_THRESHOLD

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve relevant documents from Qdrant."""
        # Embed
        query_vec = self.embedder.embed_query(query)

        # Search Qdrant
        hits = search_chunks(
            query_vector=query_vec,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        # Convert to LangChain Documents
        docs = []
        for hit in hits:
            page_content = hit["chunk_text"]

            # Preserve any additional payload fields returned by search_chunks
            # (e.g., year/make/model/location/price/condition/fuel_type/etc.)
            extra_metadata = {
                k: v
                for k, v in hit.items()
                if k
                not in (
                    "chunk_text",
                    "url",
                    "title",
                    "strategy",
                    "chunk_index",
                    "score",
                )
            }
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "url": hit.get("url", ""),
                        "title": hit.get("title", ""),
                        "strategy": hit.get("strategy", ""),
                        "chunk_index": hit.get("chunk_index", 0),
                        "score": hit.get("score", 0.0),
                        "child_text": hit["chunk_text"],
                        **extra_metadata,
                    },
                )
            )
        return docs


def documents_to_evidence(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Convert retrieved LangChain documents into JSON-serializable evidence
    items compatible with CAG cache and API responses.
    """
    evidence: list[dict] = []
    for doc in docs:
        md = dict(doc.metadata or {})
        evidence.append(
            {
                "url": md.get("url", ""),
                "title": md.get("title", ""),
                "chunk_text": md.get("child_text", doc.page_content),
                "strategy": md.get("strategy", ""),
                "chunk_index": md.get("chunk_index", 0),
                "score": md.get("score", 0.0),
                **{k: v for k, v in md.items() if k not in ("url", "title", "child_text", "strategy", "chunk_index", "score")},
            }
        )
    return evidence


# ============================================================================
# LCEL RAG Chain Builder
# ============================================================================


def build_rag_chain(
    retriever: BaseRetriever,
    llm: Any,
    k: int = TOP_K_RESULTS,
    template: str = RAG_TEMPLATE,
) -> Runnable:
    """
    Build modern RAG chain using LangChain Expression Language (LCEL).
    
    This uses Runnables and the | operator - the MODERN LangChain way.
    NO legacy chains (RetrievalQA, create_stuff_documents_chain, etc.)
    
    Chain structure:
        1. RunnableParallel: Retrieves docs + passes question through
        2. format_docs: Converts docs to context string
        3. Prompt: Fills template with context + question
        4. LLM: Generates answer
        5. StrOutputParser: Extracts string from LLM response
    
    Args:
        retriever: LangChain retriever (QdrantRetriever, etc.)
        llm: LangChain LLM instance (ChatOpenAI, etc.)
        k: Number of docs to retrieve (default from config)
        template: Prompt template string
    
    Returns:
        Runnable chain that can be invoked with query string
    
    Usage:
        chain = build_rag_chain(retriever, llm)
        answer = chain.invoke("What are the cardiology services?")
    """
    # Update retriever k if specified
    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = k

    # Create prompt template (Runnable)
    rag_prompt = ChatPromptTemplate.from_template(template)

    # BUILD THE CHAIN (Modern LCEL approach!)
    rag_chain = (
        RunnableParallel(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================================
# High-level RAG Service
# ============================================================================


class RAGService:
    """
    High-level RAG service for question answering.
    
    Encapsulates:
    - Qdrant retriever management
    - RAG chain management
    - Evidence tracking
    - Timing metrics
    
    Usage:
        service = RAGService(embedder, llm)
        result = service.generate(query)
        logger.info(result['answer'])
        logger.info(result['evidence_urls'])
    """

    def __init__(
        self,
        embedder: Any,
        llm: Any,
        k: int = TOP_K_RESULTS,
        score_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize RAG service with Qdrant-backed retriever.

        Args:
            embedder: Embedding model (must have embed_query method).
            llm: LangChain LLM instance.
            k: Number of documents to retrieve.
            score_threshold: Minimum similarity score.
        """
        self.embedder = embedder
        self.llm = llm
        self.k = k

        # Build Qdrant retriever
        self.retriever = QdrantRetriever(
            embedder=embedder,
            top_k=k,
            score_threshold=score_threshold,
        )

        # Build LCEL chain
        self.chain = build_rag_chain(self.retriever, llm, k)

    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for query using RAG.
        
        Args:
            query: User question
        
        Returns:
            Dict with:
            - answer: Generated answer string
            - evidence: List of retrieved documents
            - evidence_urls: List of unique source URLs
            - generation_time: Seconds taken
        """
        start = time.time()

        # Retrieve evidence
        docs = self.retriever.invoke(query)

        # Generate answer
        answer = self.chain.invoke(query)

        elapsed = time.time() - start

        # Extract unique URLs
        evidence_items = documents_to_evidence(docs)
        evidence_urls = list(
            {item.get("url", "") for item in evidence_items if item.get("url")}
        )

        return {
            "answer": answer,
            "evidence": evidence_items,
            "evidence_urls": evidence_urls,
            "generation_time": elapsed,
            "num_docs": len(docs),
        }

    def stream(self, query: str):
        """
        Stream answer generation (for real-time UI).
        
        Args:
            query: User question
        
        Yields:
            String chunks as they're generated
        """
        for chunk in self.chain.stream(query):
            yield chunk

    def batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries in batch.
        
        Args:
            queries: List of user questions
        
        Returns:
            List of result dicts (same format as generate())
        """
        return [self.generate(query) for query in queries]


__all__ = ["build_rag_chain", "RAGService", "QdrantRetriever", "documents_to_evidence"]
