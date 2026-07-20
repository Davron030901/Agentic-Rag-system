# Hugging Face Spaces (Docker SDK) — MUST listen on port 7860.
FROM python:3.11-slim

# System deps for PyMuPDF wheels are bundled; keep image lean.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Embedded Qdrant needs a writable dir; Spaces runs as a non-root user.
RUN mkdir -p /app/data && chmod -R 777 /app/data

EXPOSE 7860

# Single worker: embedded Qdrant holds an on-disk lock (see README limitations).
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1