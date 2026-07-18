# Internal: Dual visualization system (marimo + non-marimo)

This document describes how the Neuroscience methods book serves **both**:

1. **Marimo notebooks** (interactive `.py` → live ASGI and/or WASM HTML)
2. **Non-marimo HTML/JSON** (static data explorer + frozen REST mirror)

…from a **single feature store**, deployable to **GitHub Pages** and runnable **locally**.

Audience: maintainers extending the book (new tables, new chapters, new datasets).

---

## 1. Goals

| Goal | Mechanism |
|------|-----------|
| See tables in-browser without Jupyter | Marimo WASM **or** HTML `/explore/` |
| Same numbers on Pages and laptop | `data/processed/*` → `/api/*` JSON → UIs |
| Heavy compute (TF, Spark, NIfTI) stays off Pages | Host scripts → write JSON/CSV → re-export |
| Project Pages under `/Neuroscience/` | **Relative** URLs (`../api/…`) never absolute `/api/` |
| One deploy artifact | Entire site = `docs/` tree |

---

## 2. Architecture (three surfaces, one store)

```
data/processed/*.csv|json     data/raw/<ds_id>/
         │                           │
         ▼                           │
  marimo_exports/static_api.py       │  (registry scan)
         │                           │
         ├───────────────────────────┘
         ▼
   docs/api/**/*.json          ← frozen FastAPI mirror (Pages-safe)
         │
         ├── docs/explore/index.html     NON-MARIMO table UI (fetch JSON)
         ├── docs/wasm/<chapter>/        MARIMO WASM (Pyodide + inject)
         └── docs/index.html             Gallery linking both

Local host only:
  uvicorn app:app
    /api/*        live builders (same payloads as static_api)
    /book/*       marimo.create_asgi_app() live notebooks
    /explore/     same HTML as docs/explore
    /wasm/*       static files from docs/
```

**Rule:** Anything shown on GitHub Pages must be a **file under `docs/`**.  
Live Python is optional for authors (`uvicorn`), never required for readers on Pages.

---

## 3. Directory contracts

```
docs/
  index.html                 # gallery (marimo chapters + explore + api)
  explore/index.html         # non-marimo data browser
  api/
    health.json
    datasets.json
    surfaces.json
    god_run_summary.json
    table/{spectral,subject,…}.json
    features/*.json
    local/files.json
    index.html               # static API explorer
  wasm/
    00_data_browser/
    00_qc_dashboard/
    …
    09_multi_dataset_analysis/
  shared-assets/             # shared Vite pool for all WASM chapters
  .nojekyll
```

Canonical **source notebooks**: `marimo_notebooks/*.py`  
Export tool: `python marimo_exports/export_wasm.py --sync-docs`

---

## 4. How to add a **new table** (available in marimo + explore + Pages)

1. **Produce the artifact** under `data/processed/` (CSV or JSON) from a script.
2. **Register a builder** in `marimo_exports/static_api.py`:
   - `build_features_*` / `build_table("name")`
   - Add to `export_static_api()` payload map → `docs/api/table/<name>.json`
3. **Expose live route** in `app.py` (and optionally `marimo_exports/fastapi_app.py`):
   - `GET /api/table/{name}`
4. **Consume in UIs**:
   - HTML: add `<option>` in `docs/explore/index.html`
   - Marimo: `from api_client import load_table` → `load_table("name")`
5. **CI**: add the JSON path to `.github/workflows/deploy-pages.yml` non-marimo checks.
6. **Deploy**: push to `main` or `workflow_dispatch`.

Do **not** put NIfTI or full god parquet into Pages. Pre-aggregate (e.g. `god_run_summary.json`).

---

## 5. How to add a **new marimo chapter** on Pages

1. Create `marimo_notebooks/NN_name.py` with PEP 723 deps (browser-safe for WASM).
2. Add stem to:
   - `marimo_exports/export_wasm.py` → `CANDIDATES`
   - `helpers.BOOK_CHAPTERS` (nav)
   - `app.PUBLIC_CHAPTERS` if it should appear under `/book/`
   - `docs/index.html` gallery card
   - CI chapter loop in `deploy-pages.yml`
3. Export injects shared modules (base64):
   - always: `helpers`, `book_data` (JSON)
   - as needed: `spectral_methods`, `api_client`, `multi_dataset_catalog`
   - multi-set: `GOD_RUN_SUMMARY`, `DATASETS_REGISTRY` on `book_data`
4. Keep WASM deps free of `tensorflow`, `pyspark`, `nibabel` when possible.
5. Run:

```bash
python marimo_exports/export_wasm.py --sync-docs --verify
```

---

## 6. How to add a **non-marimo** page

1. Put static HTML under `docs/<path>/index.html` (relative asset links only).
2. Fetch data only from `../api/...json` (or same-origin live `/api/` when not on `github.io`).
3. Link from `docs/index.html`.
4. Add CI file existence + optional `python -m http.server` curl smoke (see workflow).

**Pattern used by explore:** resolve API base as `".."` so project Pages
`https://<user>.github.io/<repo>/explore/` correctly hits
`https://<user>.github.io/<repo>/api/table/spectral.json`.

Never hard-code `fetch("/api/...")` as the only strategy — that breaks project Pages.

---

## 7. Host compute vs browser viz

| Work | Where | How results reach UIs |
|------|--------|------------------------|
| Multitaper / NIfTI | Host script | CSV → static_api |
| LOOCV bake-off | Host | `ml_bakeoff.json` → API + Ch III |
| TensorFlow train | Host (`run_tf_offline.py` / Ch 06) | `tf_results.json` → Ch V + explore |
| Spark Catalyst multi-set | Host (Ch 07 / Connect) | `god_features/` → `god_run_summary.json` → Ch IX + explore |
| Table browsing | Browser | explore HTML **or** marimo WASM |

**Honour (Spark):** spectral outside JVM; pure DataFrame/`groupBy`; no Python UDFs on hot paths.  
**TF:** train offline; WASM only plots metrics.

---

## 8. Local development matrix

```bash
cd interviews/Neuroscience
pip install -r requirements.txt
export PYTHONPATH=marimo_notebooks

# A) Full gateway (marimo live + API + explore + wasm static)
uvicorn app:app --reload --port 8000
#  http://127.0.0.1:8000/explore/
#  http://127.0.0.1:8000/book/00_data_browser/
#  http://127.0.0.1:8000/wasm/09_multi_dataset_analysis/
#  http://127.0.0.1:8000/api/table/spectral

# B) Static-only (what GitHub Pages is)
python marimo_exports/static_api.py
python -m http.server 8766 --directory docs
#  http://127.0.0.1:8766/explore/
#  http://127.0.0.1:8766/api/table/spectral.json
#  http://127.0.0.1:8766/wasm/00_data_browser/

# C) Author marimo against disk
marimo edit marimo_notebooks/00_data_browser.py
marimo edit marimo_notebooks/09_multi_dataset_analysis.py
```

Optional: point WASM at a live host API:

```bash
# when serving WASM from Pages or static server, set in explore UI
# or for Python api_client:
export NEURO_API_BASE=http://127.0.0.1:8000
```

---

## 9. CI / Pages checklist (must stay green)

Workflow: `.github/workflows/deploy-pages.yml`

1. Export WASM (`export_wasm.py --sync-docs --verify`)
2. Assert public WASM chapters + inject markers + shared-assets
3. Assert **non-marimo** `docs/explore/index.html` + full `docs/api/**` table set
4. Assert relative API strategy in explore (`../api` / `return ".."`)
5. `python -m http.server` curl smoke on docs/
6. FastAPI `TestClient` smoke for `/explore/`, `/api/table/*`
7. Upload **`docs/`** as Pages artifact

---

## 10. Decision guide: marimo `.py` vs static HTML

| Need | Prefer |
|------|--------|
| Reactive science narrative, widgets, linked plots | **Marimo** (export WASM) |
| Fast table browse, no 30s package load | **HTML explore** |
| Public reproducible book chapter | **Marimo WASM** + static API |
| TF/Spark/NIfTI job | **Host script** + write JSON; **do not** put JVM/TF in WASM |
| Machine integration | **`/api/*` JSON** (live or frozen) |

You can ship **both** for the same data: marimo chapter for story, explore for “just show the CSV.”

---

## 11. Key source files

| File | Role |
|------|------|
| `marimo_notebooks/api_client.py` | disk → API → bundle loader for marimo |
| `marimo_exports/static_api.py` | builders + freeze to `docs/api` |
| `marimo_exports/export_wasm.py` | inject modules + wasm export + sync docs |
| `app.py` | live FastAPI + marimo ASGI + `/explore/` |
| `docs/explore/index.html` | non-marimo UI |
| `docs/index.html` | gallery |
| `.github/workflows/deploy-pages.yml` | CI gates |

---

## 12. Anti-patterns

- Absolute `/api/...` fetches as the **only** path (breaks `github.io/<repo>/`).
- Embedding full NIfTI or god parquet in WASM.
- Training TensorFlow inside Pyodide for this project.
- Private per-chapter `assets/` after share-assets (chunk hash drift).
- Executing JSON with Python `true`/`false` into `BOOK_BUNDLE` (use `json.loads`).

---

*Last updated for the dual-surface Pages deploy (marimo WASM + HTML explore + static API).*
