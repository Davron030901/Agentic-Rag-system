"""Qdrant vector store wiring.

Embedded/local by default (no server): a persistent on-disk Qdrant is created
under ``QDRANT_PATH``. If ``QDRANT_URL`` is set, a hosted Qdrant is used instead.

The collection is provider-specific (``docs_openai`` / ``docs_gemini``) so the
vector dimension always matches the embedding model. We NEVER mix providers in
one collection.
"""

from __future__ import annotations

from functools import lru_cache

# NOTE: qdrant / langchain_qdrant are imported lazily inside functions so that
# importing the graph for unit tests does not require the heavy vector-store
# dependencies (all LLM/store boundaries are mocked in tests).
from .config import get_settings
from .llms import get_embeddings


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Return a cached Qdrant client (embedded or hosted)."""
    from qdrant_client import QdrantClient

    settings = get_settings()
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    # Embedded on-disk mode: one process owns the lock, which is fine for a
    # single-worker API. Documented as a known limitation for horizontal scaling.
    return QdrantClient(path=settings.qdrant_path)


def ensure_collection() -> str:
    """Create the provider collection if missing; return its name."""
    from qdrant_client.http import models as qmodels

    settings = get_settings()
    client = get_qdrant_client()
    name = settings.collection
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=settings.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
    return name


def get_vector_store():
    """Return a LangChain vector store bound to the provider collection."""
    from langchain_qdrant import QdrantVectorStore

    settings = get_settings()
    ensure_collection()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.collection,
        embedding=get_embeddings(),
    )


def reset_caches() -> None:
    """Clear cached client/store — used by tests to swap providers."""
    get_qdrant_client.cache_clear()
