from loguru import logger
from typing import Any, Dict, List
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from infrastructure.config import (
    CRAG_CONFIDENCE_THRESHOLD,
    CRAG_EXPANDED_K,
    TOP_K_RESULTS
)
from services.chat_service.rag_templates import RAG_TEMPLATE
from services.chat_service.rag_service import QdrantRetriever, documents_to_evidence
from infrastructure.utils import format_docs, calculate_confidence


class CRAGService:
    """
    Corrective RAG service with automatic self-correction.
    
    Features:
    - Initial retrieval with confidence scoring
    - Automatic corrective retrieval if confidence low
    - Detailed metrics for debugging
    
    Usage:
        service = CRAGService(retriever, llm)
        result = service.generate(query, confidence_threshold=0.6)
        
        logger.info(result['answer'])
        logger.info(f"Confidence: {result['confidence_final']}")
        logger.info(f"Correction applied: {result['correction_applied']}")
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        initial_k: int = TOP_K_RESULTS,
        expanded_k: int = CRAG_EXPANDED_K
    ):
        """
        Initialize CRAG service.
        
        Args:
            retriever: LangChain retriever (QdrantRetriever, etc.)
            llm: LangChain LLM instance
            initial_k: Number of docs for initial retrieval
            expanded_k: Number of docs for corrective retrieval
        """
        self.retriever = retriever
        self.llm = llm
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def _set_k(self, k: int) -> None:
        """Set retrieval count on the retriever (supports both patterns)."""
        if isinstance(self.retriever, QdrantRetriever):
            self.retriever.top_k = k
        elif hasattr(self.retriever, "search_kwargs"):
            self.retriever.search_kwargs["k"] = k
    
    def generate(
        self,
        query: str,
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer with CRAG (Corrective RAG).
        
        Args:
            query: User question
            confidence_threshold: Minimum confidence score (0-1)
            verbose: Print progress logs
        
        Returns:
            Dict with:
            - answer: Generated answer
            - confidence_initial: Initial confidence score
            - confidence_final: Final confidence score
            - correction_applied: Whether corrective retrieval was used
            - docs_used: Number of documents in final generation
            - generation_time: Total time taken
            - evidence_urls: List of source URLs
        """
        if verbose:
            logger.info(f"🔍 Query: {query}")
            logger.success(f"🎯 Confidence threshold: {confidence_threshold}\n")
        
        # Step 1: Initial retrieval
        if verbose:
            logger.info(f"1️⃣  Initial retrieval (k={self.initial_k})...")
        
        self._set_k(self.initial_k)
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)

        if verbose:
            logger.info(f"   📊 Confidence: {confidence_initial:.2f}")

        # Step 2: Corrective retrieval if needed
        if confidence_initial >= confidence_threshold:
            if verbose:
                logger.success("   ✅ Confidence sufficient - proceeding with initial retrieval")
            final_docs = docs_initial
            confidence_final = confidence_initial
            correction_applied = False
        else:
            if verbose:
                logger.warning("   ⚠️  Low confidence - applying corrective retrieval...\n")
                logger.info(f"2️⃣  Corrective retrieval (k={self.expanded_k})...")

            self._set_k(self.expanded_k)
            docs_corrected = self.retriever.invoke(query)
            confidence_final = calculate_confidence(docs_corrected, query)

            if verbose:
                logger.info(f"   📊 Corrected confidence: {confidence_final:.2f}")
                improvement = (confidence_final - confidence_initial) * 100
                logger.info(f"   📈 Confidence improved by {improvement:.1f}%")

            final_docs = docs_corrected
            correction_applied = True
        
        # Step 3: Generate answer
        if verbose:
            logger.info(f"\n3️⃣  Generating answer...")
        
        start = time.time()
        
        # Format docs and generate
        context = format_docs(final_docs)
        prompt_input = {"context": context, "question": query}
        answer = (self.prompt | self.llm | StrOutputParser()).invoke(prompt_input)
        
        elapsed = time.time() - start
        
        # Extract evidence URLs
        evidence_items = documents_to_evidence(final_docs)
        evidence_urls = list(
            {item.get("url", "") for item in evidence_items if item.get("url")}
        )
        
        return {
            'answer': answer,
            'confidence_initial': confidence_initial,
            'confidence_final': confidence_final,
            'correction_applied': correction_applied,
            'docs_used': len(final_docs),
            'generation_time': elapsed,
            'evidence_urls': evidence_urls,
            'evidence': evidence_items,
            'documents': final_docs,
        }
    
    def batch_generate(
        self,
        queries: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries with CRAG.
        
        Args:
            queries: List of user questions
        
        Returns:
            List of result dicts (same format as generate())
        """
        results = []
        for query in queries:
            result = self.generate(query, verbose=False)
            results.append(result)
        return results


__all__ = ['CRAGService']

