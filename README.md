# Music, Reward & Depression — Spectral Biomarkers (OpenNeuro ds000171)

Interactive methods book on emotional music processing in MDD: **real BOLD** features, classical ML bake-off, and optional **TensorFlow** neural nets.

## Canonical source: `marimo_notebooks/`

| Notebook | Chapter | GitHub Pages |
|----------|---------|--------------|
| `01_pre_flight.py` | I · Cohort & Design | ✅ WASM |
| `02_eda_univariate.py` | II · Spectral Power | ✅ WASM |
| `03_eda_multivariate.py` | III · Algorithm Lab | ✅ WASM |
| `04_feature_engineering.py` | IV · Features & Music Effects | ✅ WASM |
| `06_tf_spectrogram_model.py` | V · TensorFlow Neural Nets | ❌ **local only** (TF + GPU) |
| `helpers.py` / `book_data.py` | Shared loaders + embedded tables | — |

**Legacy code** (old Jupyter steps, `src/neuro`, one-off generators) lives under `archives/` and is **not** used by the active book.

## Live book (GitHub Pages)

- **Home:** https://bladerunner-hue.github.io/Neuroscience/
- **Ch I–IV:** `/wasm/01_pre_flight/` … `/wasm/04_feature_engineering/`
- **Deploy:** GitHub Actions → [deploy-pages.yml](https://github.com/Bladerunner-hue/Neuroscience/actions/workflows/deploy-pages.yml)

Pages source must be **GitHub Actions** (repo Settings → Pages).

## Workflow

```bash
# Features from local NIfTI (data/raw/ds000171/)
python scripts/prepare_real_features.py
python scripts/run_ml_bakeoff.py          # sklearn LOOCV bake-off → ml_bakeoff.json
python scripts/gen_book_data.py           # usually already called by prepare

# Develop public chapters
export PYTHONPATH=marimo_notebooks
marimo edit marimo_notebooks/03_eda_multivariate.py

# Local TensorFlow / neural nets (real spectrograms + MLPs)
marimo edit marimo_notebooks/06_tf_spectrogram_model.py
# or: python marimo_notebooks/06_tf_spectrogram_model.py

# Publish WASM book
python marimo_exports/export_wasm.py --sync-docs
python marimo_exports/serve.py   # http://127.0.0.1:8765/
```

Push to `main` (paths under `marimo_notebooks/**`, `docs/**`, or the export script) redeploys Pages.

## Architecture (what is kept)

```
marimo_notebooks/     ← SOURCE OF TRUTH (public book + local TF chapter)
scripts/
  prepare_real_features.py
  run_ml_bakeoff.py
  gen_book_data.py
  run_notebook.sh
data/processed/       ← committed feature store + ml_bakeoff.json
docs/                 ← GitHub Pages (gallery + wasm/)
marimo_exports/       ← export_wasm.py, serve.py
archives/             ← old pipelines, legacy src/notebooks (not active)
```

## Scientific pipeline (short)

1. Whole-brain + **spatial pseudo-ROI** BOLD → Welch bands, trial-type epochs  
2. Music contrasts (`pos music − tones`, domain, anterior) + **responder score R**  
3. **13-model LOOCV bake-off** (RF / LogReg / GBM / …) → confusion + explainability  
4. **TensorFlow** (local): STFT Conv2D + dense MLPs; head-to-head vs bake-off winners  
5. RecSys priors: personalise by valence + spectral engagement, not genre alone  

## Data

- OpenNeuro [ds000171](https://openneuro.org/datasets/ds000171) (Lepping et al.)  
- Raw `*.nii.gz` gitignored; processed CSV/JSON + `book_data.py` committed for Pages  

```bash
# optional: download more BOLD then re-prepare
# see download_and_prepare.sh
python scripts/prepare_real_features.py && python scripts/run_ml_bakeoff.py
```
