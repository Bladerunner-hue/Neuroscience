# Bladerunner-Hue Neuroscience — Music, Reward & Depression (Spectral Biomarkers)

**Focus:** Frequency-domain analysis of emotional music vs non-musical auditory processing in Major Depressive Disorder (OpenNeuro **ds000171** design).

## Canonical source: `marimo_notebooks/`

Everything interactive is driven by these marimo apps + shared `helpers.py`:

| Notebook | Chapter | WASM book |
|----------|---------|-----------|
| `01_pre_flight.py` | Event alignment | ✅ |
| `02_eda_univariate.py` | Welch PSD / spectral power | ✅ |
| `03_eda_multivariate.py` | Auditory–limbic coherence | ✅ |
| `04_feature_engineering.py` | Fingerprints + clusters | ✅ |
| `06_tf_spectrogram_model.py` | TensorFlow spectrogram CNN | ❌ local only |
| `helpers.py` | Shared data/viz/nav | — |

Legacy Jupyter / old `src/neuro` live under `archives/` and are **not** used by the active notebooks.

## Live interactive book (GitHub Pages)

Built **from** `marimo_notebooks/` → WASM → `docs/` → Pages.

- **Book home:** https://bladerunner-hue.github.io/Neuroscience/
- **Ch 1:** https://bladerunner-hue.github.io/Neuroscience/wasm/01_pre_flight/
- **Ch 2:** https://bladerunner-hue.github.io/Neuroscience/wasm/02_eda_univariate/
- **Ch 3:** https://bladerunner-hue.github.io/Neuroscience/wasm/03_eda_multivariate/
- **Ch 4:** https://bladerunner-hue.github.io/Neuroscience/wasm/04_feature_engineering/
- **Deploy workflow:** https://github.com/Bladerunner-hue/Neuroscience/actions/workflows/deploy-pages.yml

Pages source must be **GitHub Actions** (repo Settings → Pages).

## Workflow (develop → run → publish)

```bash
# 1) Develop (canonical)
marimo edit marimo_notebooks/01_pre_flight.py

# 2) Run as scripts (CI-friendly)
export PYTHONPATH=marimo_notebooks
python marimo_notebooks/01_pre_flight.py
python marimo_notebooks/02_eda_univariate.py
python marimo_notebooks/03_eda_multivariate.py
python marimo_notebooks/04_feature_engineering.py
# optional local TF:
# python marimo_notebooks/06_tf_spectrogram_model.py

# 3) Session snapshots (optional previews under __marimo__/session/)
marimo export session marimo_notebooks --force-overwrite

# 4) Build WASM book + copy into docs/ for GitHub Pages
python marimo_exports/export_wasm.py --sync-docs

# 5) Preview Pages locally
python marimo_exports/serve.py
# → http://127.0.0.1:8765/
```

Push to `main` (paths under `marimo_notebooks/**`, export script, or `docs/**`) triggers the same export + deploy.

## Architecture (no stale paths)

```
marimo_notebooks/          ← SOURCE OF TRUTH
  helpers.py
  01_…04_…py, 06_….py
  __marimo__/session/      ← regenerated session JSON per notebook

marimo_exports/
  export_wasm.py           ← injects helpers for Pyodide, builds wasm/
  serve.py                 ← local Pages preview

docs/                      ← GitHub Pages artifact
  index.html               ← book gallery
  wasm/<chapter>/          ← generated; do not hand-edit
```

`export_wasm.py` registers `helpers.py` into `sys.modules` before export so browser runs can `from helpers import …` without a filesystem package.

## Scientific story (short)

- Controls: elevated high-frequency spectral power + auditory–limbic coherence on **positive music**.
- MDD: blunted, **stimulus-specific** (tones ≈ controls).
- Spectral “responder” clusters → playlist RecSys features.

## Data & deps

- Optional real data: `data/raw/ds000171/` (gitignored). Notebooks use high-quality synthetic BOLD when data is absent (WASM / CI).
- See `requirements.txt`. Browser chapters use Pyodide-friendly stack (numpy, scipy, pandas, matplotlib, plotly, scikit-learn).

## One-time Pages setup

1. https://github.com/Bladerunner-hue/Neuroscience/settings/pages  
2. Build and deployment → **Source: GitHub Actions**
