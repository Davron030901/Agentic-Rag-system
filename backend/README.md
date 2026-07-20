---
title: Adaptive Agentic RAG Assistant
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Adaptive Agentic RAG — Backend (Hugging Face Space)

This directory is deployed **as-is** to a Hugging Face **Docker** Space. The
`Dockerfile` builds a FastAPI app that listens on **port 7860** (required by
Spaces).

## Endpoints
- `GET  /health` — liveness + provider info (returns 200).
- `POST /chat` — `{ "question": "..." }` → `{ answer, steps, sources, web_search_used, low_confidence }`.
- `POST /ingest` — multipart `file` upload. Accepts **PDF, .txt, .md, .docx, .png/.jpg/.jpeg/.webp**.
  Returns `{ filename, file_type, chunks, images_captioned, pages, status, collection, provider }`.
  Unsupported formats are rejected with `400`; corrupt files / uncaptionable images return `422`.
- `POST /ingest/path` — `{ "path": "..." }` ingest a server-side file/dir (aggregate response).

## Required Space secrets
Set these under **Settings → Variables and secrets** (never commit them):
- `OPENAI_API_KEY` **or** `GOOGLE_API_KEY` (provider auto-detected)
- `TAVILY_API_KEY`
- `CORS_ALLOW_ORIGINS` = your Vercel URL, e.g. `https://your-app.vercel.app`

Optional: `PROVIDER`, `QDRANT_URL`, `CHUNK_SIZE`, `TOP_K`, `MAX_REGENERATIONS`,
`MAX_WEB_ESCALATIONS`.

## Local run
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in keys
uvicorn app.api:app --host 0.0.0.0 --port 7860
```

## Tests
```bash
pip install -r requirements.txt
pytest            # fully mocked; no API keys or network required
```

See the repository root `README.md` for architecture diagrams, the full deploy
runbook, and the evaluation section.
