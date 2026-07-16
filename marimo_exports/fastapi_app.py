"""FastAPI gateway: static WASM book + JSON feature API.

Local live routes (no .json suffix):
  GET /api/health, /api/meta, /api/features/spectral?clean=, …

GitHub Pages cannot run Python — the same payloads are frozen as static files:
  docs/api/*.json  (export via marimo_exports/static_api.py)
  Public: https://bladerunner-hue.github.io/Neuroscience/api/

Usage (repo root):
  uvicorn marimo_exports.fastapi_app:app --reload --port 8765
  # or:  python marimo_exports/fastapi_app.py
  # or:  python marimo_exports/serve.py --fastapi
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from marimo_exports.static_api import (
    build_bakeoff,
    build_features_condition,
    build_features_spectral,
    build_features_subject,
    build_health,
    build_meta,
    build_qc,
    build_tf_results,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROCESSED = ROOT / "data" / "processed"
API_STATIC = DOCS / "api"

app = FastAPI(
    title="Neuroscience book API",
    description=(
        "Live FastAPI for local dev + identical static JSON mirror on GitHub Pages "
        "at /api/*.json (see docs/api/)."
    ),
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return build_health(DOCS)


@app.get("/api/meta")
def meta():
    m = build_meta()
    if not m.get("source") and not (PROCESSED / "book_bundle.json").exists():
        raise HTTPException(404, "book_bundle.json missing — run prepare_real_features.py")
    return m


@app.get("/api/features/spectral")
def features_spectral(
    clean: bool = Query(False, description="Use cleaned_spectral_features"),
):
    return build_features_spectral(clean=clean)


@app.get("/api/features/subject")
def features_subject():
    return build_features_subject()


@app.get("/api/features/condition")
def features_condition():
    return build_features_condition()


@app.get("/api/qc")
def run_qc():
    return build_qc()


@app.get("/api/bakeoff")
def bakeoff():
    data = build_bakeoff()
    if not data:
        raise HTTPException(404, "ml_bakeoff.json missing — run run_ml_bakeoff.py")
    return data


@app.get("/api/tf_results")
def tf_results():
    data = build_tf_results()
    if not data:
        raise HTTPException(404, "tf_results.json missing — run run_tf_offline.py")
    return data


@app.get("/api")
@app.get("/api/")
def api_root():
    """Prefer the static explorer (same UI as GitHub Pages)."""
    index = API_STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "message": "Static API explorer not built yet",
        "hint": "python marimo_exports/static_api.py",
        "live": [
            "/api/health",
            "/api/meta",
            "/api/features/spectral",
            "/api/features/spectral?clean=true",
            "/api/features/subject",
            "/api/features/condition",
            "/api/qc",
            "/api/bakeoff",
            "/api/tf_results",
            "/docs",
        ],
    }


# Static WASM book + frozen /api/*.json last so explicit routes win first
if DOCS.exists():
    index = DOCS / "index.html"

    @app.get("/")
    def gallery():
        if index.exists():
            return FileResponse(index)
        return JSONResponse(
            {
                "message": "docs/index.html missing — run: python marimo_exports/export_wasm.py --sync-docs",
                "api": "/api/health",
                "api_explorer": "/api/",
            }
        )

    # Mount static docs (wasm chapters + api/*.json) under /
    app.mount("/", StaticFiles(directory=str(DOCS), html=True), name="docs")
else:

    @app.get("/")
    def no_docs():
        return {
            "message": "docs/ missing",
            "hint": "python marimo_exports/export_wasm.py --sync-docs",
            "api": "/api/health",
        }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "marimo_exports.fastapi_app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
