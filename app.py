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
    build_features_condition,
    build_features_spectral,
    build_features_subject,
    build_health,
    build_meta,
    build_qc,
    build_tf_results,
)

# Public chapters (TF train stays local-only)
PUBLIC_CHAPTERS: list[tuple[str, str, str]] = [
    ("00_qc_dashboard", "0 · QC Dashboard", "tSNR, multitaper, IsolationForest"),
    ("01_pre_flight", "I · Cohort & Design", "Who was scanned, paradigm"),
    ("02_eda_univariate", "II · Spectral Power", "Band power & PSDs"),
    ("03_eda_multivariate", "III · Algorithm Lab", "LOOCV bake-off"),
    ("04_feature_engineering", "IV · Features & Music Effects", "Contrasts & PCA"),
    ("05_tf_results", "V · Neural Net Results", "Precomputed TF metrics"),
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
    return {
        "dataset": {
            "id": "ds000171",
            "name": "Neural Processing of Emotional Musical and Nonmusical Stimuli in Depression",
            "url": "https://openneuro.org/datasets/ds000171",
            "tr_sec": 3.0,
            "design": "30 s blocks · positive/negative music vs nonmusic · tones control",
        },
        "layers": {
            "raw": "data/raw/ds000171/ (gitignored BIDS NIfTI)",
            "processed": "data/processed/*.csv + book_bundle.json (committed)",
            "scripts": [
                "scripts/prepare_real_features.py  # NIfTI → adaptive multitaper PSD + tSNR + QC",
                "scripts/run_ml_bakeoff.py         # 13-model LOOCV → ml_bakeoff.json",
                "scripts/run_tf_offline.py         # STFT Conv2D / MLP → tf_results.json",
                "scripts/gen_book_data.py          # embed book_bundle into book_data.py",
            ],
            "notebooks": "marimo_notebooks/00–05 public · 06 TF local",
            "wasm": "docs/wasm/ + shared-assets/ (GitHub Pages)",
            "static_api": "docs/api/*.json (frozen FastAPI mirror)",
            "live_api": "this FastAPI process (uvicorn app:app)",
        },
        "psd_methods": ["welch", "uniform multitaper", "adaptive multitaper", "mne adaptive (optional)"],
        "qc": ["tsnr", "spectral_flatness", "spectral_entropy", "band_snr_high", "IsolationForest"],
        "cross_ref_openneuro": [
            {
                "id": "ds002725",
                "why": "Joint EEG-fMRI affective music — multimodal spectral validation",
                "url": "https://openneuro.org/datasets/ds002725",
            },
            {
                "id": "ds003085",
                "why": "Temporal dynamics of emotional music (BOLD)",
                "url": "https://openneuro.org/datasets/ds003085",
            },
            {
                "id": "ds003720",
                "why": "Music genre fMRI — auditory / genre spectral structure",
                "url": "https://openneuro.org/datasets/ds003720",
            },
            {
                "id": "ds004142",
                "why": "rt-fMRI neurofeedback reward saliency / valence",
                "url": "https://openneuro.org/datasets/ds004142",
            },
            {
                "id": "ds005700",
                "why": "NeuroEmo emotion recognition (includes depressed label)",
                "url": "https://openneuro.org/datasets/ds005700",
            },
            {
                "id": "ds006564",
                "why": "Naturalistic film + musical soundtracks; depression/anxiety traits",
                "url": "https://openneuro.org/datasets/ds006564",
            },
        ],
        "github_pages": "https://bladerunner-hue.github.io/Neuroscience/",
    }


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
    for stem, _title, _desc in PUBLIC_CHAPTERS:
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
<p><a href="/docs">OpenAPI</a> · <a href="/api/health">/api/health</a> ·
<a href="/api/architecture">/api/architecture</a></p>
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
    return {
        "marimo_mounted": _MARIMO_OK,
        "error": _MARIMO_ERR,
        "chapters": [
            {
                "stem": stem,
                "title": title,
                "description": desc,
                "live_url": f"/book/{stem}/",
                "wasm_url": f"/wasm/{stem}/",
                "exists": (NOTEBOOKS / f"{stem}.py").exists(),
            }
            for stem, title, desc in PUBLIC_CHAPTERS
        ],
    }


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
