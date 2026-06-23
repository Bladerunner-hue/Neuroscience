# Bladerunner-Hue Neuroscience — Music, Reward & Depression (Spectral Biomarkers)

**Focus:** Frequency-domain analysis of emotional music vs non-musical auditory processing in Major Depressive Disorder using **OpenNeuro ds000171** (39 participants: 19 MDD + 20 Controls).

## The Scientific Story
Music carries rich harmonic and temporal structure. We test whether:
- **Controls** show stronger high-frequency spectral power and auditory–limbic coherence during **positive music**.
- **MDD** participants exhibit **blunted** responses (anhedonia signature) that are **stimulus-specific** (little or no group difference on non-musical tones).

This yields visually rich, clinically actionable biomarkers with a direct line to **personalized music recommendation systems** (RecSys).

## Why This Repo Is Different
- **Reactive marimo notebooks** (primary) + classic Jupyter
- Heavy, production-grade **TensorFlow** usage (not just `model.fit`):
  - `tf.signal.stft` for spectrograms
  - Mixed precision + `MirroredStrategy`
  - Proper `tf.data` + AUTOTUNE
  - Probabilistic modeling hooks (TFP-ready)
- **Spark** vision for large-cohort parallel feature extraction
- Beautiful interactive Plotly dashboards + nilearn-ready brain visuals
- Hypothesis-driven sections + "Clinical Relevance + RecSys link" in every notebook

## Live Interactive Dashboards (Free Tier)

The notebooks are exported to **WebAssembly-powered HTML** that run 100% in the browser using Pyodide. No backend or Python installation is required.

**Live site:**
- **Main gallery**: [https://bladerunner-hue.github.io/Neuroscience/](https://bladerunner-hue.github.io/Neuroscience/)
- **01 — Pre-flight** (fully interactive): [https://bladerunner-hue.github.io/Neuroscience/wasm/01_pre_flight/](https://bladerunner-hue.github.io/Neuroscience/wasm/01_pre_flight/)
- **02 — Spectral Power** (fully interactive): [https://bladerunner-hue.github.io/Neuroscience/wasm/02_eda_univariate/](https://bladerunner-hue.github.io/Neuroscience/wasm/02_eda_univariate/)

These are automatically deployed on every relevant push using GitHub Actions + GitHub Pages (completely free for public repositories).

### Deployment Links
- **GitHub Pages site**: https://bladerunner-hue.github.io/Neuroscience/
- **Workflow runs**: https://github.com/Bladerunner-hue/Neuroscience/actions/workflows/deploy-pages.yml
- **Source code**: https://github.com/Bladerunner-hue/Neuroscience

## Working with Marimo Notebooks

The `Neuroscience` repository contains both the raw marimo notebooks (`.py`) and their exports. It is important to understand what each format provides.

### Static vs. Fully Interactive Exports

When you run `marimo export html notebook.py -o notebook.html` you get a **static** HTML snapshot. This file renders outputs but strips out the reactive engine, so sliders and other widgets do not work interactively. That is why simply opening the HTML files from `marimo_exports/` in a browser (or Zed) gives you a read-only view.

To get the **full notebook experience** you need either:
- The interactive `marimo edit` editor, **or**
- A WebAssembly (WASM) export.

### Working Interactively (Recommended)

Marimo’s source of truth is always the `.py` file.

```bash
# Work on the notebooks (live reactive UI in the browser)
marimo edit marimo_notebooks/01_pre_flight.py
marimo edit marimo_notebooks/02_eda_univariate.py
marimo edit marimo_notebooks/03_eda_multivariate.py
marimo edit marimo_notebooks/04_feature_engineering.py

# For WebAssembly export later, develop with --sandbox
# (this records dependencies so they can be installed in the browser)
marimo edit --sandbox marimo_notebooks/02_eda_univariate.py
```

The editor runs a live reactive session. Changes to the `.py` (even from Zed) are reflected immediately in the browser.

Many scientific packages (NumPy, SciPy, pandas, matplotlib via Pyodide, scikit-learn, etc.) are supported in the browser runtime.

### Quick Exploration of Existing Exports

The repository ships helper tools inside `marimo_exports/`:

```bash
# Interactive selector that embeds the HTML exports in an iframe
marimo edit marimo_exports/dashboard_browser.py

# Or a plain static server (no marimo needed)
python marimo_exports/serve.py
# then open http://localhost:8765/index.html
```

`dashboard_browser.py` gives you a nice selector + live preview of the exported dashboards.

### Making Exports Fully Interactive for Others (WebAssembly)

To share interactive notebooks that run entirely in the browser (no Python backend required):

```bash
# Read-only interactive WASM export
marimo export html-wasm marimo_notebooks/02_eda_univariate.py \
  -o marimo_exports/wasm/02_eda_univariate --mode run

# Editable interactive WASM export
marimo export html-wasm marimo_notebooks/02_eda_univariate.py \
  -o marimo_exports/wasm/02_eda_univariate --mode edit
```

This produces a self-contained folder (or single HTML) that embeds Python via WebAssembly. Open the `index.html` in any modern browser — the notebook runs client-side with full widget reactivity.

**Convenience script:**

```bash
python marimo_exports/export_wasm.py          # produces --mode run exports
python marimo_exports/export_wasm.py --edit   # produces editable WASM versions
```

The outputs land in `marimo_exports/wasm/`.

**Important:** Only notebooks using Pyodide-compatible packages will work. Notebooks that rely on TensorFlow (e.g. `06_tf_spectrogram_model.py`) cannot run fully in the browser.

### Hosting on GitHub Pages

1. Export using the WASM command above.
2. Copy the resulting directory (or the whole `wasm/` folder) into `docs/`.
3. Add a `.nojekyll` file in the published directory.
4. Enable GitHub Pages on the repo (source = `docs/` or `main` branch + `/docs`).

A helper `.nojekyll` is already present in `marimo_exports/`.

You can also automate this with a GitHub Action that runs the export script and deploys the `docs/` folder.

### Alternative: Sharing via molab

Paste a GitHub raw URL of a notebook (`.py`) into [molab](https://molab.marimo.io) to get:
- A static preview
- A temporary server run
- Or a fully in-browser WebAssembly run

Molab gives you shareable links and badges.

### Putting It All Together

* **Develop** in your editor + `marimo edit marimo_notebooks/XX.py` (or with `--sandbox`).
* **Explore** existing exports quickly with `marimo_exports/dashboard_browser.py` or `serve.py`.
* **Distribute** using `marimo export html-wasm ... --mode run` for self-contained interactive files.
* Commit session JSON under `__marimo__/session/` if you want static previews with outputs to be captured.
* Use only packages that Pyodide supports when targeting the browser.

This gives you (and others) the full reactive marimo experience for the neuroscience spectral biomarker dashboards.

## All Links

**Live Dashboards (GitHub Pages)**
- Gallery: https://bladerunner-hue.github.io/Neuroscience/
- 01 Pre-flight (WASM): https://bladerunner-hue.github.io/Neuroscience/wasm/01_pre_flight/
- 02 Spectral (WASM): https://bladerunner-hue.github.io/Neuroscience/wasm/02_eda_univariate/

**Repository & Automation**
- GitHub repo: https://github.com/Bladerunner-hue/Neuroscience
- Deployment Action: https://github.com/Bladerunner-hue/Neuroscience/actions/workflows/deploy-pages.yml
- Source notebooks: https://github.com/Bladerunner-hue/Neuroscience/tree/main/marimo_notebooks

**Helpful Tools & Docs**
- marimo: https://marimo.io
- marimo WebAssembly docs: https://docs.marimo.io/guides/webassembly
- molab (GitHub → browser mirror): https://molab.marimo.io

## Current Marimo Notebooks

```bash
marimo edit marimo_notebooks/01_pre_flight.py
marimo edit marimo_notebooks/02_eda_univariate.py
marimo edit marimo_notebooks/03_eda_multivariate.py
marimo edit marimo_notebooks/04_feature_engineering.py
marimo edit marimo_notebooks/06_tf_spectrogram_model.py   # TF-heavy (not suitable for WASM)
```

**Notebook roadmap:**
1. Pre-flight & event alignment
2. Spectral power (Welch PSD) ← priority
3. Multivariate coherence
4. Feature engineering + clustering
5. Preprocessing pipeline
6. Deep learning on spectrograms (TF demo)
7–8. Visual results + reproducibility

## HTML / WASM Exports

- Static snapshots: `marimo_exports/*.html`
- Interactive WASM versions: run the export script above (or manually with `html-wasm`)

Open `marimo_exports/index.html` or launch the dashboard browser for a nice gallery.

## Data & Dependencies

Data lives in `data/raw/ds000171/`. The notebooks use high-quality synthetic generators so they are runnable without the full dataset.

For browser (WASM) use, stick to packages known to work in Pyodide.

## Classic Notebooks

Legacy Jupyter notebooks remain in `notebooks/`.

---

This repository is built to deliver **reactive, high-quality neuroscience visualisations** that are easy to run, share, and explore.
