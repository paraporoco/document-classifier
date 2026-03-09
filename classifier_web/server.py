"""
classifier_web/server.py — FastAPI server for the document sensitivity classifier.

Responsibilities:
  - Serve the SPA (web/) as static files
  - Serve browser-compatible model artefacts (web_artefacts/) with cache headers
  - GET  /api/v1/health   — liveness check
  - GET  /api/v1/version  — metadata from manifest.json
  - POST /api/v1/classify — optional server-side classify (disabled by default)

Environment variables:
  ENABLE_SERVER_CLASSIFY=true   Enable POST /api/v1/classify (default: false)
  ARTEFACTS_DIR                 Path to web_artefacts/ (default: ./web_artefacts)
  HOST                          Bind address (default: 0.0.0.0)
  PORT                          Bind port (default: 8000)

Usage:
  uvicorn server:app --reload                         # development
  python server.py                                    # production
  ENABLE_SERVER_CLASSIFY=true uvicorn server:app      # with server-side classify
"""

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

HERE          = Path(__file__).parent
WEB_DIR       = HERE / "web"
ARTEFACTS_DIR = Path(os.getenv("ARTEFACTS_DIR", HERE / "web_artefacts"))
ENABLE_CLASSIFY = os.getenv("ENABLE_SERVER_CLASSIFY", "false").lower() == "true"

app = FastAPI(
    title="Document Sensitivity Classifier",
    version="1.1",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Artefact static files — long-lived cache (content-hashed filenames) ────────

class _ImmutableStaticFiles(StaticFiles):
    """StaticFiles with cache-immutable headers for content-hashed artefact files."""
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            # manifest.json is not hashed — revalidate it
            if path == "manifest.json":
                response.headers["Cache-Control"] = "no-cache"
            else:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


if ARTEFACTS_DIR.exists():
    app.mount(
        "/web_artefacts",
        _ImmutableStaticFiles(directory=str(ARTEFACTS_DIR)),
        name="web_artefacts",
    )


# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/version")
def version():
    manifest = ARTEFACTS_DIR / "manifest.json"
    if not manifest.exists():
        raise HTTPException(
            503,
            "Artefacts not found. Run: python classifier_production/convert_artefacts.py",
        )
    return JSONResponse(json.loads(manifest.read_text()))


# ── Optional server-side classify endpoint ─────────────────────────────────────
#
# Disabled by default. Enable with ENABLE_SERVER_CLASSIFY=true.
# Note: callers must extract text themselves — no file upload endpoint exists.
# Text sent to this endpoint leaves the client; this breaks the zero-egress
# guarantee for browser users and is intended only for automated pipelines.

if ENABLE_CLASSIFY:
    from pydantic import BaseModel

    _prod = HERE.parent / "classifier_production"
    sys.path.insert(0, str(_prod))
    import predict as _predict
    _predict.load()

    class _ClassifyRequest(BaseModel):
        text: str

    @app.post("/api/v1/classify")
    def classify(req: _ClassifyRequest):
        if not req.text.strip():
            raise HTTPException(400, "text must not be empty")
        return _predict.predict(req.text)


# ── SPA fallback — serve index.html for all unmatched routes ──────────────────
#
# Registered last so the API and artefact routes take priority.

@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    candidate = WEB_DIR / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, f"SPA not found. Expected: {index}")
    return FileResponse(index)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
