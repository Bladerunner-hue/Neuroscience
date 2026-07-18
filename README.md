# Music, Reward & Depression — Spectral Biomarkers (OpenNeuro)

Interactive methods book on emotional music processing in MDD: **real BOLD** features, classical ML bake-off, optional **TensorFlow**, and a **multi-dataset** extension path.

Repo: https://github.com/Bladerunner-hue/Neuroscience.git

---

## Data location (important)

**Everything lives under `data/`:**

```
data/
├── README.md                 # layout + started-with / added
├── raw/
│   ├── ds000171/             # PRIMARY — full BOLD (~21G local)
│   ├── ds002725/             # cross-ref EEG-fMRI music (sample BOLD)
│   ├── ds003085/ … ds006564/ # cross-refs (meta and/or stubs)
│   └── README.md
└── processed/                # spectral CSVs, registry, god multi-set parquet
```

Open **`data/raw/`** in the file tree (not the repo root). Large NIfTI is gitignored but present locally after download. Each cross-ref folder has a `STATUS.txt` when BOLD is not yet pulled.

Refresh disk inventory:

```bash
python scripts/refresh_dataset_registry.py   # → data/processed/dataset_registry.json
```

---

## What we started with vs what we added

| Phase | Content | Path / link |
|-------|---------|-------------|
| **Started with** | OpenNeuro **[ds000171](https://openneuro.org/datasets/ds000171)** (Lepping et al.): 39 MDD/Control, music vs nonmusic BOLD | `data/raw/ds000171/` |
| | Feature store + bake-off + TF JSON | `data/processed/*.csv`, `*.json` |
| | Marimo public chapters 0–V | `marimo_notebooks/00_qc` … `05_*` |
| **Added** | Ranked cross-refs: [ds002725](https://openneuro.org/datasets/ds002725) (EEG-fMRI music), [ds003085](https://openneuro.org/datasets/ds003085), [ds004894](https://openneuro.org/datasets/ds004894), [ds006564](https://openneuro.org/datasets/ds006564), [ds003720](https://openneuro.org/datasets/ds003720), … | `data/raw/<id>/` |
| | Scientific catalog + live registry | `multi_dataset_catalog.py`, `dataset_registry.json` |
| | Multi-set god parquet (e.g. ds000171 + ds002725 sample) | `data/processed/god_*` |
| | Landscape chapter (breakdown + analysis) | `marimo_notebooks/00_data_landscape.py` |

**Why add cross-refs?** Cross-validate multitaper/tSNR spectral biomarkers beyond one scanner; multimodal (EEG, HR–insula); valence dynamics; ecological music; non-emotional genre baseline. Primary clinical contrast remains ds000171.

Full narrative: open the landscape notebook (below).

---

## Canonical source: `marimo_notebooks/`

| Notebook | Chapter | GitHub Pages |
|----------|---------|--------------|
| `00_data_landscape.py` | 0-local · **Data landscape** (started with / added / analysis) | ❌ local |
| `00_qc_dashboard.py` | 0 · QC Dashboard | ✅ WASM |
| `01_pre_flight.py` | I · Cohort & Design | ✅ WASM |
| `02_eda_univariate.py` | II · Spectral Power | ✅ WASM |
| `03_eda_multivariate.py` | III · Algorithm Lab | ✅ WASM |
| `04_feature_engineering.py` | IV · Features & Music Effects | ✅ WASM |
| `05_tf_results.py` | V · Neural Net Results | ✅ WASM (precomputed) |
| `06_tf_spectrogram_model.py` | V-local · Train TensorFlow | ❌ local |
| `07_spark_god_mode.py` | VII · Spark God Mode | ❌ local |
| `helpers.py` · `multi_dataset_catalog.py` · `spark_session.py` | Shared | — |

Legacy Jupyter / old `src/neuro` → `archives/` (inactive).

---

## Install (pip only)

```bash
cd interviews/Neuroscience   # or clone of Bladerunner-hue/Neuroscience
python3.12 -m venv .venv && source .venv/bin/activate   # recommended
pip install -r requirements.txt
# includes: marimo, polars, pandas, openneuro-py, nibabel, pyspark, scikit-learn, tensorflow, …
```

No Docker required for data download, features, marimo, or Spark Connect (native server optional).

---

## Workflow

```bash
# --- Primary features (ds000171 NIfTI under data/raw/ds000171/) ---
python scripts/prepare_real_features.py --psd adaptive
python scripts/run_ml_bakeoff.py

# --- Multi-dataset ---
python scripts/download_openneuro_cohorts.py
python scripts/download_openneuro_cohorts.py --with-bold --max-subjects 1 --only ds002725
python scripts/refresh_dataset_registry.py
python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725
python scripts/god_mode_bold_to_tfdata.py --smoke-tfdata

# --- Explore (start here for inventory) ---
export PYTHONPATH=marimo_notebooks
marimo edit marimo_notebooks/00_data_landscape.py
marimo edit marimo_notebooks/00_qc_dashboard.py

# Optional native Spark Connect (pip pyspark; full Spark binary for server)
# ./scripts/start_local_spark_connect.sh

# TensorFlow offline → public Ch V JSON
python scripts/run_tf_offline.py

# Publish WASM book
python marimo_exports/export_wasm.py --sync-docs --verify
python marimo_exports/serve.py --fastapi   # http://127.0.0.1:8765/
```

---

## Live book (GitHub Pages)

- **Home:** https://bladerunner-hue.github.io/Neuroscience/
- **Ch 0–V WASM:** `/wasm/00_qc_dashboard/` … `/wasm/05_tf_results/`
- **Static API:** https://bladerunner-hue.github.io/Neuroscience/api/

WASM embeds helpers + spectral_methods + book_bundle (no NIfTI / no TF in browser).

---

## Architecture

Full write-up: **[ARCHITECTURE.md](ARCHITECTURE.md)** · data map: **[data/README.md](data/README.md)**
