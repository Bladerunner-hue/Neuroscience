# Music, Reward & Depression — Spectral Biomarkers (OpenNeuro ds000171)

Interactive methods book on emotional music processing in MDD: **real BOLD** features, classical ML bake-off, and optional **TensorFlow** neural nets.

## Canonical source: `marimo_notebooks/`

| Notebook | Chapter | GitHub Pages |
|----------|---------|--------------|
| `00_qc_dashboard.py` | 0 · QC Dashboard | ✅ WASM |
| `01_pre_flight.py` | I · Cohort & Design | ✅ WASM |
| `02_eda_univariate.py` | II · Spectral Power | ✅ WASM |
| `03_eda_multivariate.py` | III · Algorithm Lab | ✅ WASM |
| `04_feature_engineering.py` | IV · Features & Music Effects | ✅ WASM |
| `05_tf_results.py` | V · Neural Net Results | ✅ WASM (precomputed; **no TF in browser**) |
| `06_tf_spectrogram_model.py` | V-local · Train TensorFlow | ❌ local retrain only |
| `helpers.py` / `book_data.py` | Shared loaders + embedded tables | — |

**Legacy code** (old Jupyter steps, `src/neuro`, one-off generators) lives under `archives/` and is **not** used by the active book.

## Live book (GitHub Pages)

- **Home:** https://bladerunner-hue.github.io/Neuroscience/
- **Ch 0–V:** `/wasm/00_qc_dashboard/` … `/wasm/05_tf_results/`
- **Public API (static FastAPI mirror):** https://bladerunner-hue.github.io/Neuroscience/api/
  - `…/api/health.json` · `…/api/meta.json` · `…/api/features/spectral_clean.json`
  - `…/api/qc.json` · `…/api/bakeoff.json` · `…/api/openapi.json`
- **Deploy:** GitHub Actions → [deploy-pages.yml](https://github.com/Bladerunner-hue/Neuroscience/actions/workflows/deploy-pages.yml)

Pages source must be **GitHub Actions** (repo Settings → Pages).

> **Why “static FastAPI”?** GitHub Pages only serves files — it cannot run `uvicorn`.
> `marimo_exports/static_api.py` freezes the same JSON payloads as the local FastAPI app
> into `docs/api/`, so the public site exposes the API contract with a Swagger explorer.

## Workflow

```bash
# Features from local NIfTI (data/raw/ds000171/)
# Default PSD = adaptive DPSS multitaper (+ tSNR / IsolationForest QC)
python scripts/prepare_real_features.py --psd adaptive
# Fast recompute from bold_timeseries.csv (no NIfTI I/O):
# python scripts/prepare_real_features.py --from-timeseries --psd adaptive
# Optional production MNE adaptive:
# pip install mne && python scripts/prepare_real_features.py --psd mne

python scripts/run_ml_bakeoff.py          # sklearn LOOCV bake-off → ml_bakeoff.json
python scripts/gen_book_data.py           # usually already called by prepare

# Develop public chapters (reactive marimo)
export PYTHONPATH=marimo_notebooks
marimo edit marimo_notebooks/00_qc_dashboard.py   # multitaper + tSNR + IF + clean export
marimo edit marimo_notebooks/03_eda_multivariate.py

# TensorFlow offline (trains locally → JSON for public Ch V)
python scripts/run_tf_offline.py
# optional interactive retrain:
# marimo edit marimo_notebooks/06_tf_spectrogram_model.py

# Publish WASM book (injects helpers + spectral_methods + book_bundle JSON)
python marimo_exports/export_wasm.py --sync-docs --verify
# Re-check without rebuild:
# python marimo_exports/export_wasm.py --verify-only

# Serve book — FastAPI mounts WASM + live /api/* (preferred local)
python marimo_exports/serve.py --fastapi   # http://127.0.0.1:8765/
# curl http://127.0.0.1:8765/api/health
# curl http://127.0.0.1:8765/api/health.json   # same as GitHub Pages static mirror
# Rebuild static API only: python marimo_exports/static_api.py
# Plain static only: python marimo_exports/serve.py
```

### Access model (marimo · FastAPI · WASM · GitHub Pages)

| Surface | Role |
|---------|------|
| **`marimo edit`** | Authoring: full reactive graph, scipy multitaper demos, local data |
| **FastAPI** (`serve.py --fastapi`) | Local: live `/api/*` + static `docs/` (WASM + `docs/api/*.json`) |
| **GitHub Pages `/api/`** | **Static FastAPI mirror** — frozen JSON + Swagger UI (no Python host) |
| **WASM chapters** | Public book: Pyodide in-browser (no TF/MNE) |

**What each WASM chapter embeds** (via `export_wasm.py` base64 injection into cell 0):

- `helpers.py` — loaders, style, provenance  
- `spectral_methods.py` — DPSS uniform + adaptive multitaper, tSNR  
- `book_data.BOOK_BUNDLE` — adaptive PSD features (with `psd_f`/`psd_pxx`), cleaned runs, QC, bake-off, TF metrics, timeseries examples  

No NIfTI or TensorFlow in the browser. Ch 0 demos multitaper live; Ch II–IV use cleaned tabular + PSD arrays; Ch V only plots precomputed `tf_results`.

Push to `main` (paths under `marimo_notebooks/**`, `docs/**`, or the export script) redeploys Pages.

## Architecture (what is kept)

Full write-up: **[ARCHITECTURE.md](ARCHITECTURE.md)** · machine-readable: `GET /api/architecture`

```
marimo_notebooks/     ← SOURCE OF TRUTH (public book + local TF chapter)
app.py                ← FastAPI + marimo.create_asgi_app() (/book + /api)
scripts/
  prepare_real_features.py   # adaptive multitaper + tSNR + QC
  run_ml_bakeoff.py
  run_tf_offline.py
  gen_book_data.py
data/processed/       ← committed feature store + ml_bakeoff.json + tf_results.json
docs/                 ← GitHub Pages (gallery + wasm/shared-assets + api/*.json)
marimo_exports/       ← export_wasm.py, static_api.py, serve.py, fastapi_app.py
archives/             ← old pipelines, legacy src/notebooks (not active)
```

### Local production gateway (recommended)

```bash
export PYTHONPATH=marimo_notebooks:.
uvicorn app:app --reload --port 8000
# Live marimo:  http://127.0.0.1:8000/book/00_qc_dashboard/
# REST OpenAPI: http://127.0.0.1:8000/docs
# Static WASM:  http://127.0.0.1:8000/wasm/02_eda_univariate/
# or: python marimo_exports/serve.py --book --port 8000
```

### God-mode PySpark CLI (`neuro-tal-cli`)

Honour mode: Tungsten/Catalyst only (no RAPIDS, no Python UDFs).

```bash
python -m cli.neuro_tal_cli status
python -m cli.neuro_tal_cli prepare --datasets ds000171,ds002725
python -m cli.neuro_tal_cli train --epochs 30
python -m cli.neuro_tal_cli viz          # marimo 07 batch
python -m cli.neuro_tal_cli stream-seed && python -m cli.neuro_tal_cli stream-file --once
python -m cli.neuro_tal_cli stream-monitor   # marimo 08 streaming
python -m cli.neuro_tal_cli serve            # app.py
```

| Route | Role |
|-------|------|
| `/book/*` | Live reactive marimo (server Python) |
| `/api/*` | Features, QC, bake-off, TF metrics, `predict/group` |
| `/` · `/wasm/*` | Same static site as GitHub Pages |
| Pages `/api/*.json` | Frozen mirror when there is no Python host |

## Scientific pipeline (short)

1. Whole-brain + **spatial pseudo-ROI** BOLD → **adaptive multitaper** bands (Welch/uniform/MNE optional), trial-type epochs  
2. **QC gate**: tSNR + flatness/entropy/band-SNR + IsolationForest → `cleaned_spectral_features.csv`  
3. Music contrasts (`pos music − tones`, domain, anterior) + **responder score R**  
4. **13-model LOOCV bake-off** (RF / LogReg / GBM / …) → confusion + explainability  
5. **TensorFlow** (local): STFT Conv2D + dense MLPs on cleaned runs; head-to-head vs bake-off winners  
6. RecSys priors: personalise by valence + spectral engagement, not genre alone  

### Cross-reference OpenNeuro sets

[ds002725](https://openneuro.org/datasets/ds002725) (EEG–fMRI music) ·
[ds003085](https://openneuro.org/datasets/ds003085) (emotional music dynamics) ·
[ds003720](https://openneuro.org/datasets/ds003720) (genre) ·
[ds004142](https://openneuro.org/datasets/ds004142) (reward NF) ·
see [ARCHITECTURE.md](ARCHITECTURE.md) for the full table.

## Data

- OpenNeuro [ds000171](https://openneuro.org/datasets/ds000171) (Lepping et al.)  
- Raw `*.nii.gz` gitignored; processed CSV/JSON + `book_data.py` committed for Pages  

```bash
# optional: download more BOLD then re-prepare
# see download_and_prepare.sh
python scripts/prepare_real_features.py && python scripts/run_ml_bakeoff.py
```
