"""
Embedding model provider.

Routes through OpenRouter when PROVIDER=openrouter, otherwise direct OpenAI.
"""

import os
from typing import Any
from langchain_openai import OpenAIEmbeddings

from infrastructure.config import EMBEDDING_MODEL, PROVIDER, OPENROUTER_BASE_URL, get_api_key


def get_default_embeddings(
    batch_size: int = 100,
    show_progress: bool = False,
    **kwargs: Any
) -> OpenAIEmbeddings:
    """
    Get an OpenAIEmbeddings instance configured for the active provider.

    When PROVIDER=openrouter, requests are routed through the OpenRouter
    unified API so that model IDs resolve correctly.

    The API key is always resolved to a string so that sync methods
    (embed_documents, embed_query) work. Passing None or a callable can
    cause "Sync client is not available" when using sync embedding.

    Args:
        batch_size: Number of texts to embed per API call.
        show_progress: Show progress bar during embedding.
        **kwargs: Additional arguments forwarded to OpenAIEmbeddings.

    Returns:
        A ready-to-use OpenAIEmbeddings instance.
    """
    llm_kwargs: dict[str, Any] = dict(
        model=EMBEDDING_MODEL,
        show_progress_bar=show_progress,
        **kwargs,
    )

    if PROVIDER == "openrouter":
        llm_kwargs["openai_api_base"] = OPENROUTER_BASE_URL
        api_key = get_api_key("openrouter")
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            llm_kwargs["openai_api_key"] = str(api_key)
    else:
        api_key = get_api_key("openai")
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            llm_kwargs["openai_api_key"] = str(api_key)

    return OpenAIEmbeddings(**llm_kwargs)
