"""
llm_client.py
=============

Single, shared LLM client.  We use LangChain's `ChatOpenAI` wrapper but point
it at OpenRouter's OpenAI-compatible endpoint.  This gives every component
identical behaviour (retries, streaming, tool calling, etc.) regardless of the
underlying provider.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from config import get_logger, settings

log = get_logger(__name__)


@lru_cache(maxsize=4)
def get_chat_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Return a cached `ChatOpenAI` instance configured for OpenRouter.

    Caching by (temperature, max_tokens) keeps the SDK client warm between
    LangGraph node invocations without re-reading environment variables.
    """
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy `.env.example` to `.env` and "
            "fill in your OpenRouter key."
        )

    default_headers = {}
    if settings.openrouter_referrer:
        default_headers["HTTP-Referer"] = settings.openrouter_referrer
    if settings.openrouter_title:
        default_headers["X-Title"] = settings.openrouter_title

    log.debug("Building ChatOpenAI client for model=%s", settings.model)
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=settings.max_tokens if max_tokens is None else max_tokens,
        timeout=settings.request_timeout,
        default_headers=default_headers or None,
    )
