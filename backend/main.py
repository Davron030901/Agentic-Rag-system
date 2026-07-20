"""Process entrypoint. Hugging Face Spaces requires listening on port 7860."""

from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, workers=1)
