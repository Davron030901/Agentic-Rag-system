"""External tools — currently the Tavily web-search fallback.

Kept behind a small function so the graph node can be unit-tested by mocking
``web_search`` without importing Tavily.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from .config import get_settings


def web_search(query: str, max_results: int = 4) -> List[Document]:
    """Run a Tavily web search and return results as ``Document`` objects.

    Each result becomes a Document whose metadata carries the URL/title so the
    generator can cite it and the frontend can render it as a web source.
    Returns an empty list if Tavily is not configured or the call fails.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query, max_results=max_results, search_depth="basic"
        )
    except Exception:  # pragma: no cover - network/provider dependent
        return []

    docs: List[Document] = []
    for item in response.get("results", []):
        content = item.get("content", "") or ""
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": item.get("url", ""),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "modality": "web",
                },
            )
        )
    return docs
