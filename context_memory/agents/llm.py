"""
LLM agent wrappers using LiteLLM.
Provider-agnostic — model names come from config, never hardcoded.

IMPORTANT: litellm is imported LAZILY to avoid a Windows hang caused by
OpenSSL 3.0+ slow ssl.create_default_context().load_verify_locations().
The first LLM call will be ~2-3s slower as litellm loads; after that, instant.
"""

from __future__ import annotations

import os
import asyncio

from context_memory.config import settings

# Ensure the API key is available before litellm initializes
os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)

# ── Lazy litellm loader ──
_litellm = None

def _get_litellm():
    """Import litellm lazily — avoids the ~60s+ SSL hang on Windows at startup."""
    global _litellm
    if _litellm is None:
        import ssl
        _orig = ssl.create_default_context
        def _fast(**kw):
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        ssl.create_default_context = _fast

        import litellm
        litellm.ssl_verify = False
        _litellm = litellm

        ssl.create_default_context = _orig
    return _litellm


async def call_llm(model: str, prompt: str, system: str = "") -> str:
    """
    Generic async LLM call via LiteLLM.
    Retries up to 3 times with exponential backoff for transient errors.
    """
    litellm = _get_litellm()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    delays = [5, 15, 30]  # Exponential backoff

    for attempt in range(len(delays) + 1):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.Timeout) as e:
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])
            else:
                raise
    raise RuntimeError("LLM call failed after all retries")


async def summarize(text: str) -> str:
    """Generate an objective summary using the cheap Summarizer model."""
    return await call_llm(
        model=settings.summarizer_model,
        system="You are a precise summarizer. Write a concise, objective summary of the given text. Focus on key concepts, relationships, and functional purpose. Do not add opinions.",
        prompt=f"Summarize the following:\n\n{text}",
    )


async def verify(query: str, data_chunk: str) -> bool:
    """
    Boolean verification: does this chunk answer the query?
    Returns True/False. Used for anti-hallucination checks.
    """
    response = await call_llm(
        model=settings.verifier_model,
        system="You are a strict boolean evaluator. Reply only with TRUE or FALSE.",
        prompt=f"User asked: {query}\n\nData found:\n<data_begin>\n{data_chunk}\n<data_end>\n\nDoes this data contain what is needed to answer the user's question?",
    )
    return response.upper().startswith("TRUE")


async def embed(text: str) -> list[float]:
    """
    Generate an embedding vector. Retries on transient errors.
    """
    litellm = _get_litellm()
    delays = [5, 15, 30]
    for attempt in range(len(delays) + 1):
        try:
            response = await litellm.aembedding(
                model=settings.embedding_model,
                input=[text],
            )
            return response.data[0]["embedding"]
        except (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.Timeout) as e:
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])
            else:
                raise
    raise RuntimeError("Embedding failed after all retries")


async def embed_batch(texts: list[str], delay: float = 1.0) -> list[list[float]]:
    """
    Embed multiple texts with a delay between calls to respect rate limits.
    For free tier: processes one at a time with configurable delay.
    """
    embeddings = []
    for i, text in enumerate(texts):
        vector = await embed(text)
        embeddings.append(vector)
        if delay > 0 and i < len(texts) - 1:
            await asyncio.sleep(delay)
    return embeddings


def count_tokens(text: str) -> int:
    """Count tokens using LiteLLM's provider-aware token counter."""
    litellm = _get_litellm()
    return litellm.token_counter(model=settings.summarizer_model, text=text)
