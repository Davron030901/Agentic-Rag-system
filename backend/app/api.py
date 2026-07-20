"""FastAPI application exposing /chat, /ingest and /health.

* Requests validated by Pydantic models.
* Errors are surfaced as clean JSON (no raw 500 stack traces to the client).
* CORS is configured from CORS_ALLOW_ORIGINS for the Vercel frontend.
"""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .graph import run_agent
from .ingestion import (
    SUPPORTED_EXTS,
    UnsupportedFormatError,
    ingest_bytes,
    ingest_paths,
    is_supported,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    FileIngestResult,
    HealthResponse,
    IngestPathRequest,
    IngestResponse,
)

logger = logging.getLogger("agentic_rag")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Adaptive Multimodal Agentic RAG Assistant",
    version=__version__,
    description="Self-grading RAG agent with web-search fallback and citations.",
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled(_request, exc: Exception):  # pragma: no cover - safety net
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + provider info. Must be publicly reachable post-deploy."""
    s = get_settings()
    return HealthResponse(
        status="ok",
        provider=s.provider.value,
        collection=s.collection,
        embedding_dim=s.embedding_dim,
        web_search_enabled=s.web_search_enabled,
        version=__version__,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question by running the adaptive-RAG agent."""
    try:
        result = run_agent(req.question)
    except Exception as exc:  # convert to a clean 502
        logger.exception("Agent run failed")
        raise HTTPException(status_code=502, detail="Agent failed to produce an answer.") from exc

    return ChatResponse(
        answer=result["answer"],
        steps=result["steps"],
        sources=result["sources"],
        web_search_used=result["web_search_used"],
        low_confidence=result["low_confidence"],
    )


@app.post("/ingest", response_model=FileIngestResult)
async def ingest(file: UploadFile = File(...)) -> FileIngestResult:
    """Ingest a single uploaded document and return a structured result.

    Accepts multipart form-data with a ``file`` field. Supported formats: PDF,
    .txt, .md, .docx, and standalone images (.png/.jpg/.jpeg/.webp). Unsupported
    formats are rejected (400) before any processing.
    """
    filename = file.filename or "upload.bin"
    if not is_supported(filename):
        allowed = ", ".join(sorted(SUPPORTED_EXTS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format for '{filename}'. Allowed: {allowed}.",
        )

    try:
        data = await file.read()
        result = ingest_bytes(filename, data)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # Corrupt file / uncaptionable image -> readable 422, not a stack trace.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(
            status_code=500, detail=f"Failed to ingest '{filename}'. The file may be corrupt."
        ) from exc

    s = get_settings()
    return FileIngestResult(
        filename=result.filename,
        file_type=result.file_type,
        chunks=result.chunks,
        images_captioned=result.images_captioned,
        pages=result.pages,
        status=result.status,
        detail=result.detail,
        collection=s.collection,
        provider=s.provider.value,
    )


@app.post("/ingest/path", response_model=IngestResponse)
def ingest_path_endpoint(req: IngestPathRequest) -> IngestResponse:
    """Ingest a server-side file or directory path (JSON body)."""
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    try:
        result = ingest_paths(req.path)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    s = get_settings()
    files = [
        FileIngestResult(
            filename=fr.filename,
            file_type=fr.file_type,
            chunks=fr.chunks,
            images_captioned=fr.images_captioned,
            pages=fr.pages,
            status=fr.status,
            detail=fr.detail,
            collection=s.collection,
            provider=s.provider.value,
        )
        for fr in result.files
    ]
    return IngestResponse(
        ingested_files=result.ingested_files,
        text_chunks=result.text_chunks,
        image_captions=result.image_captions,
        points_upserted=result.points_upserted,
        collection=s.collection,
        provider=s.provider.value,
        files=files,
    )
