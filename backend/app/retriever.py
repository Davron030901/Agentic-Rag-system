"""Top-K retriever built over the provider-specific Qdrant collection."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from .config import get_settings
from .vectorstore import get_vector_store


def get_retriever(k: int | None = None):
    """Return a LangChain retriever with a configurable top-K (default from env)."""
    settings = get_settings()
    top_k = k if k is not None else settings.top_k
    return get_vector_store().as_retriever(search_kwargs={"k": top_k})


def retrieve_documents(question: str, k: int | None = None) -> List[Document]:
    """Convenience helper returning the top-K documents for a question."""
    return get_retriever(k).invoke(question)
