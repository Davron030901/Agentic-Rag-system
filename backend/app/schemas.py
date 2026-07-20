"""Pydantic schemas: structured LLM outputs, graph state, and API contracts.

Every place where an LLM's output feeds control flow uses a structured schema
here (``with_structured_output``) rather than string parsing.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured LLM outputs used by the graders (control-flow decisions).
# ---------------------------------------------------------------------------
class YesNo(BaseModel):
    """Binary structured judgment used by all graders."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer 'yes' or 'no' only."
    )
    reason: str = Field(
        default="",
        description="One short sentence justifying the score (for tracing).",
    )


# ---------------------------------------------------------------------------
# Source / citation model shared by graph + API.
# ---------------------------------------------------------------------------
class Source(BaseModel):
    """A single citation returned to the frontend."""

    type: Literal["doc", "web"]
    id: str = Field(description="Chunk point-id (doc) or URL (web).")
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: str = ""


# ---------------------------------------------------------------------------
# Graph state.
#
# Uses a TypedDict because LangGraph merges partial dict updates returned by
# each node. Documents are kept as a list of dicts ({page_content, metadata})
# so the state stays JSON-serialisable and easy to test without importing
# langchain Document everywhere.
# ---------------------------------------------------------------------------
class DocDict(TypedDict):
    page_content: str
    metadata: dict


class GraphState(TypedDict, total=False):
    """State threaded through the LangGraph state machine."""

    question: str
    documents: List[DocDict]
    generation: str

    # Whether a web search was used at any point in this run.
    web_search_used: bool

    # Ordered list of node names actually executed (frontend step pills).
    steps: List[str]

    # Citations backing the final answer.
    sources: List[dict]

    # Retry counters — enforce hard caps so the graph always terminates.
    regenerate_count: int
    web_escalation_count: int

    # Set true when we bail out via a retry cap and return best-effort answer.
    low_confidence: bool

    # Internal: next route computed by grade_generation, read by its edge.
    _route: str


# ---------------------------------------------------------------------------
# API request / response contracts.
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    steps: List[str]
    sources: List[Source]
    web_search_used: bool
    low_confidence: bool


class IngestPathRequest(BaseModel):
    """Ingest by pointing the server at a local directory or file path."""

    path: str = Field(min_length=1, description="File or directory path on the server.")


class FileIngestResult(BaseModel):
    """Structured per-file ingestion result (returned by POST /ingest)."""

    filename: str
    file_type: Literal["pdf", "docx", "text", "image"]
    chunks: int
    images_captioned: int
    pages: Optional[int] = None
    status: Literal["indexed", "error"]
    detail: str = ""
    collection: str
    provider: str


class IngestResponse(BaseModel):
    """Aggregate response for directory ingestion (POST /ingest/path)."""

    ingested_files: List[str]
    text_chunks: int
    image_captions: int
    points_upserted: int
    collection: str
    provider: str
    files: List[FileIngestResult] = []


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    collection: str
    embedding_dim: int
    web_search_enabled: bool
    version: str
