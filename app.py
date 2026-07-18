#!/usr/bin/env python3
"""Production FastAPI + Marimo ASGI gateway for the ds000171 methods book.

Surfaces
--------
Local interactive book (server-side Python, full scipy/sklearn):
  GET  /book/                → chapter index (JSON + links)
  GET  /book/00_qc_dashboard → live marimo app (reactive)
  … same for 01–05 (06 TF train is local-only optional)

REST feature API (same payloads as GitHub Pages static mirror):
  GET  /api/health
  GET  /api/meta
  GET  /api/features/spectral[?clean=true]
  GET  /api/features/subject
  GET  /api/features/condition
  GET  /api/qc
  GET  /api/bakeoff
  GET  /api/tf_results
  GET  /api/architecture
  POST /api/predict/group    → lightweight Control vs MDD from subject features

Static GitHub Pages tree (WASM + frozen JSON):
  GET  /                     → docs/ gallery
  GET  /wasm/…               → browser WASM chapters
  GET  /api/*.json           → static API mirror (identical contracts)

Run (repo root, pyenv 3.12 recommended)::

    pip install -r requirements.txt
    export PYTHONPATH=marimo_notebooks
    uvicorn app:app --reload --port 8000
    # or:
    python app.py
    python marimo_exports/serve.py --book

GitHub Pages cannot run this process — it only serves ``docs/`` (WASM + static API).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "marimo_notebooks"
DOCS = ROOT / "docs"
PROCESSED = ROOT / "data" / "processed"

# Notebooks import helpers / spectral_methods / book_data by bare name
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marimo_exports.static_api import (  # noqa: E402
    build_bakeoff,
    build_datasets,
    build_features_condition,
    build_features_events,
    build_features_participants,
    build_features_spectral,
    build_features_subject,
    build_health,
    build_local_files,
    build_meta,
    build_qc,
    build_surfaces,
    build_table,
    build_tf_results,
)

# Public chapters (TF train stays local-only)
PUBLIC_CHAPTERS: list[tuple[str, str, str]] = [
    ("00_data_browser", "0 · Data browser", "Tables shared with /explore HTML"),
    ("00_qc_dashboard", "0 · QC Dashboard", "tSNR, multitaper, IsolationForest"),
    ("01_pre_flight", "I · Cohort & Design", "Who was scanned, paradigm"),
    ("02_eda_univariate", "II · Spectral Power", "Band power & PSDs"),
    ("03_eda_multivariate", "III · Algorithm Lab", "LOOCV bake-off"),
    ("04_feature_engineering", "IV · Features & Music Effects", "Contrasts & PCA"),
    ("05_tf_results", "V · Neural Net Results", "Precomputed TF metrics"),
    ("09_multi_dataset_analysis", "IX · Multi-dataset scale", "God summary · registry · TF view"),
]
# Local-only marimo (full Python; also listed in /api/surfaces)
LOCAL_CHAPTERS: list[tuple[str, str, str]] = [
    ("00_data_landscape", "0-local · Data Landscape", "Multi-set inventory + tables"),
    ("00_data_browser", "0-local · Data browser", "Same tables as /explore HTML"),
    ("06_tf_spectrogram_model", "V-local · TF Train", "Train on host; metrics via API"),
    ("07_spark_god_mode", "VII · Spark God Mode", "Catalyst multi-set (host)"),
]

app = FastAPI(
    title="Neuroscience ds000171 — API + Interactive Book",
    description=(
        "OpenNeuro ds000171 spectral biomarkers (adaptive multitaper + tSNR QC). "
        "Local: FastAPI REST + marimo.create_asgi_app() chapters. "
        "Public: GitHub Pages WASM + static /api/*.json mirror."
    ),
    version="1.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST — feature store (shared with static_api / GitHub Pages)
# ---------------------------------------------------------------------------
@app.get("/api/health")
def api_health():
    h = build_health(DOCS)
    h["surface"] = "fastapi-live"
    h["book_mount"] = "/book/"
    h["pages_gallery"] = "/" if DOCS.exists() else None
    return h


@app.get("/api/meta")
def api_meta():
    m = build_meta()
    if not m.get("source") and not (PROCESSED / "book_bundle.json").exists():
        raise HTTPException(404, "book_bundle.json missing — run prepare_real_features.py")
    return m


@app.get("/api/features/spectral")
def api_spectral(clean: bool = Query(False)):
    return build_features_spectral(clean=clean)


@app.get("/api/features/subject")
def api_subject():
    return build_features_subject()


@app.get("/api/features/condition")
def api_condition():
    return build_features_condition()


@app.get("/api/qc")
def api_qc():
    return build_qc()


@app.get("/api/bakeoff")
def api_bakeoff():
    data = build_bakeoff()
    if not data:
        raise HTTPException(404, "ml_bakeoff.json missing — run run_ml_bakeoff.py")
    return data


@app.get("/api/tf_results")
def api_tf():
    data = build_tf_results()
    if not data:
        raise HTTPException(404, "tf_results.json missing — run run_tf_offline.py")
    return data


@app.get("/api/architecture")
def api_architecture():
    """Machine-readable map of the repo stack (for dashboards and tooling)."""
    from marimo_exports.static_api import build_architecture

    arch = build_architecture()
    arch["surface"] = "fastapi-live"
    arch["explore"] = "/explore/"
    arch["book"] = "/book/"
    return arch


@app.get("/api/datasets")
def api_datasets():
    return build_datasets()


@app.get("/api/surfaces")
def api_surfaces():
    """Marimo + non-marimo visualization surfaces."""
    return build_surfaces()


@app.get("/api/local/files")
def api_local_files(
    include_raw_niftis: bool = Query(False, description="Include every NIfTI path (huge)"),
):
    """Browse host ``data/`` for the HTML explorer and marimo api_client."""
    return build_local_files(include_raw_niftis=include_raw_niftis)


@app.get("/api/table/{name}")
def api_table(
    name: str,
    limit: int | None = Query(None, ge=1, le=50_000),
):
    """Unified tables for marimo WASM, live marimo, and /explore HTML."""
    data = build_table(name, limit=limit)
    if data.get("error"):
        raise HTTPException(404, data["error"])
    return data


@app.get("/api/features/participants")
def api_participants():
    return build_features_participants()


@app.get("/api/features/events")
def api_events():
    return build_features_events()


@app.get("/api/god_run_summary")
def api_god_run_summary():
    """Multi-set Catalyst summary for explore + WASM (no full parquet)."""
    path = PROCESSED / "god_run_summary.json"
    if not path.exists():
        return {"n_runs": 0, "records": [], "by_dataset_group": [], "datasets": []}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/predict/group")
def predict_group(
    payload: dict[str, Any] = Body(
        ...,
        examples=[
            {"subject": "sub-control01"},
            {
                "features": {
                    "responder_score": 0.25,
                    "pos_music_vs_tones_bold": 0.4,
                    "music_vs_nonmusic_bold": 0.1,
                    "run_power_high_mean": 0.15,
                    "coh_ant_post_mean": 0.02,
                }
            },
        ],
    ),
):
    """Transparent Control vs MDD score from subject-level features.

    Not a production clinical model — fits a small LogisticRegression on the
    committed subject_features table (same columns as the LOOCV bake-off).
    Prefer the offline bake-off for published metrics.
    """
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    subj = build_features_subject()
    rows = subj.get("records") or []
    if len(rows) < 4:
        raise HTTPException(400, "Need subject_features with ≥4 rows")

    df = pd.DataFrame(rows)
    if "group" not in df.columns:
        raise HTTPException(400, "subject_features missing group column")

    feature_cols = [
        c
        for c in [
            "responder_score",
            "pos_music_vs_tones_bold",
            "neg_music_vs_tones_bold",
            "music_vs_nonmusic_bold",
            "pos_music_vs_tones_power_high",
            "pos_music_vs_tones_anterior",
            "run_power_high_mean",
            "run_centroid_mean",
            "coh_ant_post_mean",
            "coh_left_right_mean",
            "ant_minus_post_mean",
            "music_task_vs_nonmusic_power_high",
            "age",
        ]
        if c in df.columns
    ]
    if len(feature_cols) < 2:
        raise HTTPException(400, "Too few numeric feature columns")

    y = (df["group"].astype(str) == "MDD").astype(int).values
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", solver="lbfgs"
                ),
            ),
        ]
    )
    pipe.fit(X.values, y)

    subject = payload.get("subject") or payload.get("subject_id")
    lookup = None
    if subject:
        hit = df[df["subject"].astype(str) == str(subject)]
        if hit.empty:
            raise HTTPException(404, f"Unknown subject {subject!r}")
        lookup = hit.iloc[0].to_dict()
        x_row = hit[feature_cols].apply(pd.to_numeric, errors="coerce").values
    elif "features" in payload and isinstance(payload["features"], dict):
        feat = payload["features"]
        x_row = np.array(
            [[float(feat.get(c, np.nan)) for c in feature_cols]], dtype=float
        )
    else:
        raise HTTPException(
            400,
            "Provide {\"subject\": \"sub-…\"} or {\"features\": {col: value, …}}",
        )

    proba = float(pipe.predict_proba(x_row)[0, 1])
    pred = "MDD" if proba >= 0.5 else "Control"
    bake = build_bakeoff()
    winners = (bake or {}).get("winners") or {}

    # Simple coefficient ranking for explainability
    clf = pipe.named_steps["clf"]
    coefs = getattr(clf, "coef_", None)
    top = []
    if coefs is not None:
        pairs = sorted(
            zip(feature_cols, coefs[0].tolist()),
            key=lambda t: abs(t[1]),
            reverse=True,
        )
        top = [{"feature": a, "coef": round(float(b), 4)} for a, b in pairs[:6]]

    return {
        "prediction": pred,
        "mdd_probability": round(proba, 4),
        "model": "LogReg-L2 (in-process demo on subject_features — not LOOCV)",
        "feature_columns": feature_cols,
        "top_coefficients": top,
        "subject_lookup": (
            {
                "subject": lookup.get("subject"),
                "group": lookup.get("group"),
                "responder_score": lookup.get("responder_score"),
            }
            if lookup
            else None
        ),
        "bakeoff_group_winner": winners.get("group"),
        "caveat": (
            "For publication-quality numbers use scripts/run_ml_bakeoff.py "
            "(nested LOOCV) and Ch III WASM — this endpoint is for interactive demos."
        ),
    }


# ---------------------------------------------------------------------------
# Marimo ASGI — live reactive chapters (server-side Python)
# ---------------------------------------------------------------------------
def _build_marimo_asgi():
    import marimo

    builder = marimo.create_asgi_app(
        quiet=True,
        include_code=False,
        show_tracebacks=True,
    )
    for stem, _title, _desc in PUBLIC_CHAPTERS + LOCAL_CHAPTERS:
        nb = NOTEBOOKS / f"{stem}.py"
        if not nb.exists():
            continue
        # Mounted under /book → full URL /book/{stem}/
        builder = builder.with_app(path=f"/{stem}", root=str(nb.resolve()))
    return builder.build()


def _book_index_html(*, marimo_ok: bool, err: str | None) -> str:
    if not marimo_ok:
        return (
            f"<h1>Marimo book unavailable</h1><pre>{err or 'MARIMO_SKIP=1'}</pre>"
            "<p>Install marimo and restart, or use WASM: "
            "<a href='/wasm/00_qc_dashboard/'>/wasm/00_qc_dashboard/</a></p>"
        )
    items = "\n".join(
        f'<li><a href="/book/{stem}/"><strong>{title}</strong></a> — {desc}</li>'
        for stem, title, desc in PUBLIC_CHAPTERS
        if (NOTEBOOKS / f"{stem}.py").exists()
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Live marimo book</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:42rem;margin:2rem auto;padding:0 1rem;
  background:#0f1419;color:#e7e5e4}}
 a{{color:#6ee7b7}} li{{margin:.6rem 0}}
</style></head><body>
<h1>Live marimo book</h1>
<p>Server-side reactive Python (full scipy / sklearn). For the public browser-only
book see <a href="/">gallery</a> / WASM.</p>
<ul>{items}</ul>
<p><a href="/explore/">HTML data explorer</a> (non-marimo) ·
<a href="/docs">OpenAPI</a> · <a href="/api/surfaces">/api/surfaces</a> ·
<a href="/api/health">/api/health</a></p>
</body></html>"""


_MARIMO_OK = False
_MARIMO_ERR: str | None = None
_MARIMO_ASGI = None
try:
    if os.environ.get("MARIMO_SKIP", "0") not in ("1", "true", "TRUE"):
        _MARIMO_ASGI = _build_marimo_asgi()
        _MARIMO_OK = True
except Exception as exc:  # pragma: no cover
    _MARIMO_ERR = str(exc)


@app.get("/book", response_class=HTMLResponse)
@app.get("/book/", response_class=HTMLResponse)
def book_index():
    """Human index of live marimo chapters (registered before the ASGI mount)."""
    status = 200 if _MARIMO_OK else 503
    return HTMLResponse(
        _book_index_html(marimo_ok=_MARIMO_OK, err=_MARIMO_ERR),
        status_code=status,
    )


@app.get("/api/book")
def api_book_catalog():
    chapters = []
    for stem, title, desc in PUBLIC_CHAPTERS:
        chapters.append(
            {
                "stem": stem,
                "title": title,
                "description": desc,
                "scope": "public",
                "live_url": f"/book/{stem}/",
                "wasm_url": f"/wasm/{stem}/",
                "exists": (NOTEBOOKS / f"{stem}.py").exists(),
            }
        )
    for stem, title, desc in LOCAL_CHAPTERS:
        chapters.append(
            {
                "stem": stem,
                "title": title,
                "description": desc,
                "scope": "local",
                "live_url": f"/book/{stem}/",
                "wasm_url": None,
                "exists": (NOTEBOOKS / f"{stem}.py").exists(),
            }
        )
    return {
        "marimo_mounted": _MARIMO_OK,
        "error": _MARIMO_ERR,
        "explore_url": "/explore/",
        "chapters": chapters,
    }


@app.get("/explore", response_class=HTMLResponse)
@app.get("/explore/", response_class=HTMLResponse)
def explore_ui():
    """Non-marimo HTML table browser (same /api/table payloads as WASM/marimo)."""
    explore = DOCS / "explore" / "index.html"
    if explore.exists():
        return HTMLResponse(explore.read_text(encoding="utf-8"))
    # fallback minimal
    return HTMLResponse(
        "<h1>Explorer missing</h1><p>Expected docs/explore/index.html</p>"
        "<p><a href='/api/surfaces'>/api/surfaces</a> · "
        "<a href='/api/table/spectral'>/api/table/spectral</a></p>"
    )


# Mount after exact /book routes so the catalog is not swallowed
if _MARIMO_OK and _MARIMO_ASGI is not None:
    app.mount("/book", _MARIMO_ASGI)


# ---------------------------------------------------------------------------
# Static docs (WASM gallery + frozen API) — last so /api and /book win
# ---------------------------------------------------------------------------
if DOCS.exists():
    index = DOCS / "index.html"

    @app.get("/")
    def gallery():
        if index.exists():
            from fastapi.responses import FileResponse

            return FileResponse(index)
        return RedirectResponse("/book/")

    app.mount("/", StaticFiles(directory=str(DOCS), html=True), name="docs")
else:

    @app.get("/")
    def no_docs():
        return {
            "message": "docs/ missing — run: python marimo_exports/export_wasm.py --sync-docs",
            "book": "/book/",
            "api": "/api/health",
        }


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"Neuroscience book API + marimo on http://127.0.0.1:{port}/")
    print(f"  Live book:  http://127.0.0.1:{port}/book/")
    print(f"  REST:       http://127.0.0.1:{port}/api/health")
    print(f"  OpenAPI:    http://127.0.0.1:{port}/docs")
    print(f"  WASM:       http://127.0.0.1:{port}/wasm/00_qc_dashboard/")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
