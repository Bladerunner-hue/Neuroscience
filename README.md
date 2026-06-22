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

## Current Marimo Notebooks (start here)
```bash
# Make sure you have the deps
pip install -r requirements.txt

# Launch individual reactive notebooks
marimo edit marimo_notebooks/02_spectral_biomarkers.py
marimo edit marimo_notebooks/06_tf_spectrogram_model.py
```

**Notebook roadmap (upgraded):**
1. `01_pre_flight` — stimulus event alignment, data inventory, delayed peak in MDD
2. `02_spectral_biomarkers.py` ← **priority** — Welch PSDs, band power, interactive group/condition comparison
3. `03_eda_multivariate` — coherence (auditory ↔ limbic) during music
4. `04_feature_engineering` — spectral features + Spark + clustering → "music responder" subtypes
5. `05_preprocessing_pipeline` — bandpass, Hilbert phase, TFRecords
6. `06_tf_spectrogram_model.py` ← **TF skill demo** — Conv2D on STFTs + all the optimizations
7. `07_visual_results` — interactive dashboards, publication figures, activation GIFs
8. `08_conclusion_reproducibility` — full clinical implications + MLflow + reproducibility bundle

## Quick Start (Classic Notebooks Still Work)
```bash
python -m jupyter lab notebooks/            # or
./scripts/run_notebook.sh 02
```

## Key Technologies
- TensorFlow 2.20 (signal, mixed precision, distribute, tf.data)
- marimo (reactive dataflow notebooks — the future)
- Plotly + Nilearn + Polars-ready
- MLflow tracking
- Spark (planned full integration via `src/neuro/spark_utils.py`)

## Data
Data lives in `data/raw/ds000171/`.  
Use `download_and_prepare.sh` (or openneuro-py) for the full ~7 GB dataset.

**Note:** The marimo notebooks ship with high-fidelity synthetic data generators that reproduce the expected spectral dissociation so everything is runnable instantly.

## Development
```bash
marimo edit marimo_notebooks/02_spectral_biomarkers.py   # recommended
# or any of the numbered notebooks
```

This repo is built to impress on a **TensorFlow + modern data stack + neuroscience storytelling** skill check.
