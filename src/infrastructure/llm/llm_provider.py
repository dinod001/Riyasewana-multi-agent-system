"""
Chat LLM provider factory.

This repo's `infrastructure.config` resolves the active provider + model from YAML
(`config/param.yaml`, `config/models.yaml`) and secrets from environment/.env.

We keep the API surface small and stable:
  - get_chat_llm(): main user-facing LLM for RAG generation.

If you later reintroduce multi-model routing (router/extractor), add it here
*only* once the corresponding config constants exist.
"""

from typing import Optional, Any
from langchain_openai import ChatOpenAI

from infrastructure.config import (
    CHAT_MODEL,
    PROVIDER,
    OPENROUTER_BASE_URL,
    get_api_key,
)


def _build_llm(
    model: str,
    provider: str,
    temperature: float = 0,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Internal factory — builds a ChatOpenAI for any provider."""
    llm_kwargs: dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
        **kwargs,
    )

    if provider == "openrouter":
        llm_kwargs["openai_api_base"] = OPENROUTER_BASE_URL
        llm_kwargs["openai_api_key"] = get_api_key("openrouter")
    elif provider == "openai":
        llm_kwargs["openai_api_key"] = get_api_key("openai")
    else:
        # Fallback: try provider-specific key if configured
        llm_kwargs["openai_api_key"] = get_api_key(provider)

    return ChatOpenAI(**llm_kwargs)


def get_chat_llm(temperature: float = 0, **kwargs: Any) -> ChatOpenAI:
    """LLM for user-facing responses (synthesis, RAG generation).

    Uses the active provider/model from `infrastructure.config`.
    """
    return _build_llm(CHAT_MODEL, PROVIDER, temperature=temperature, **kwargs)
