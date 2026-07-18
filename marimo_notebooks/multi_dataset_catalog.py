"""Canonical multi-dataset catalog for the Neuroscience book.

Scientific + integration metadata for OpenNeuro cohorts that extend
ds000171 (Lepping et al.). Used by ``00_data_landscape`` and download scripts.
"""
from __future__ import annotations

from typing import Any

# Ranked integration catalog (feedback-aligned). Keys = OpenNeuro dataset ids.
MULTI_DATASET_CATALOG: dict[str, dict[str, Any]] = {
    "ds000171": {
        "title": "Neural Processing of Emotional Musical and Nonmusical Stimuli in Depression",
        "short_title": "Emotional music vs nonmusic in MDD",
        "authors": "Lepping et al.",
        "role": "primary",
        "priority": 0,
        "url": "https://openneuro.org/datasets/ds000171",
        "n_participants_nominal": 39,
        "cohort": "19 MDD (currently depressed) + 20 never-depressed controls; SCID; no meds at scan",
        "modality": ["fMRI-BOLD", "T1w"],
        "scanner": "Siemens Skyra 3T",
        "tasks": [
            "music (positive/negative emotional excerpts)",
            "nonmusic (tones / non-musical auditory)",
        ],
        "why_neuro": (
            "Core clinical contrast: limbic / auditory responses to emotional music "
            "in MDD vs controls; defines spectral biomarkers and responder score R."
        ),
        "why_preprocess": (
            "Full BIDS events + BOLD already drive prepare_real_features.py "
            "(adaptive multitaper, tSNR, IsolationForest, condition epochs)."
        ),
        "integration": (
            "Feature store gold standard: spectral_features / condition_features / "
            "subject_features + R; ML bake-off + TF spectrogram models."
        ),
        "match_level": "primary",
        "modal_extra": [],
    },
    "ds002725": {
        "title": "Joint EEG-fMRI during affective music listening",
        "short_title": "EEG-fMRI affective music",
        "authors": "Daly et al., 2020",
        "role": "cross_ref",
        "priority": 1,
        "url": "https://openneuro.org/datasets/ds002725",
        "n_participants_nominal": 21,
        "cohort": "21 healthy adults",
        "modality": ["fMRI-BOLD", "EEG", "T1w"],
        "scanner": "EEG-fMRI (see dataset README)",
        "tasks": [
            "classicalMusic",
            "generated/synthetic affective music",
            "washout",
        ],
        "why_neuro": (
            "Highest match for affective music listening. Healthy multimodal baseline "
            "to cross-validate ds000171 spectral biomarkers (BOLD PSD vs EEG) and "
            "limbic/auditory affect decoding."
        ),
        "why_preprocess": (
            "Parallel per-subject BOLD (nibabel) + optional MNE EEG; Spark SQL for "
            "batch QC/fusion features; Delta provenance dataset_id=ds002725."
        ),
        "integration": (
            "Align affect dimensions with pos/neg music; same multitaper bands + tSNR; "
            "flag rows with dataset_id; marimo cross-set EDA (healthy vs MDD maps)."
        ),
        "match_level": "highest",
        "modal_extra": ["EEG"],
        "bold_include_globs": [
            "sub-*/func/*bold.nii.gz",
            "sub-*/func/*events.tsv",
            "sub-*/anat/*T1w.nii.gz",
        ],
    },
    "ds003085": {
        "title": "Temporal Dynamics of Emotional Music",
        "short_title": "Happy/sad music dynamics",
        "authors": "Sachs et al.",
        "role": "cross_ref",
        "priority": 2,
        "url": "https://openneuro.org/datasets/ds003085",
        "n_participants_nominal": 39,
        "cohort": "~39 participants (happy/sad music blocks)",
        "modality": ["fMRI-BOLD"],
        "tasks": ["happy music", "sad music", "rest (if present)"],
        "why_neuro": (
            "Direct emotional-music fMRI dynamics — temporal/spectral evolution of "
            "happy/sad BOLD vs ds000171 music valence contrasts."
        ),
        "why_preprocess": (
            "Distributed epoching + multitaper/dynamic spectral features; "
            "Delta partitioned by valence; marimo filters by emotion."
        ),
        "integration": (
            "Standardize valence labels; matching power/centroid bands; "
            "optional multi-dataset LOOCV / domain adaptation."
        ),
        "match_level": "high",
        "modal_extra": [],
        "bold_include_globs": [
            "sub-*/func/*bold.nii.gz",
            "sub-*/func/*events.tsv",
        ],
    },
    "ds006564": {
        "title": "Naturalistic film + controlled musical soundtracks",
        "short_title": "Naturalistic music-film ToM",
        "authors": "Bravo et al.",
        "role": "cross_ref",
        "priority": 3,
        "url": "https://openneuro.org/datasets/ds006564",
        "n_participants_nominal": None,
        "cohort": "Film viewing with manipulated musical soundtracks; ToM / predictive priming",
        "modality": ["fMRI-BOLD"],
        "tasks": ["naturalistic film", "controlled musical soundtracks"],
        "why_neuro": (
            "Ecological music-in-context: predictive/memory effects in emotion/ToM "
            "networks overlapping limbic/auditory systems relevant to MDD."
        ),
        "why_preprocess": (
            "Spark-scale naturalistic runs; music-property modulation features; "
            "Delta time-travel across soundtrack conditions."
        ),
        "integration": (
            "Spectral features during music epochs; align predictive/emotion axes "
            "with ds000171 music contrasts."
        ),
        "match_level": "medium-high",
        "modal_extra": [],
    },
    "ds004894": {
        "title": "HR and insula activity to music × interoceptive sensitivity",
        "short_title": "Music + HR + insula",
        "authors": "Maekawa et al.",
        "role": "cross_ref",
        "priority": 4,
        "url": "https://openneuro.org/datasets/ds004894",
        "n_participants_nominal": 50,
        "cohort": "~49–52 participants; music + interoception (heartbeat tasks)",
        "modality": ["fMRI-BOLD", "PPG/HR"],
        "tasks": ["music (tonal/atonal)", "heartbeat counting/discrimination", "valence ratings"],
        "why_neuro": (
            "Music emotion + peripheral HR + insula — ties to ds000171 anterior "
            "pseudo-ROIs and MDD interoception/emotion literature."
        ),
        "why_preprocess": (
            "Parallel fMRI + physio sync; HRV + insula spectral features; "
            "Delta multimodal store."
        ),
        "integration": (
            "Correlate HR with anterior/insula spectral proxies; extend R with "
            "interoception/HR features."
        ),
        "match_level": "high-multimodal",
        "modal_extra": ["PPG", "HR"],
        "bold_include_globs": [
            "sub-*/func/*bold.nii.gz",
            "sub-*/func/*events.tsv",
        ],
    },
    "ds003720": {
        "title": "Music Genre fMRI Dataset",
        "short_title": "Genre listening baseline",
        "authors": "Nakai et al.",
        "role": "cross_ref",
        "priority": 5,
        "url": "https://openneuro.org/datasets/ds003720",
        "n_participants_nominal": 5,
        "cohort": "5 subjects × 10 genres × many runs (multi-band EPI)",
        "modality": ["fMRI-BOLD"],
        "tasks": ["music genre listening (10 genres)"],
        "why_neuro": (
            "Non-emotional music-processing baseline vs emotion-specific design "
            "in ds000171; high within-subject reliability."
        ),
        "why_preprocess": (
            "Spark parallel across runs/genres for spectral features; "
            "genre-level rollups."
        ),
        "integration": (
            "Control comparison for non-emotional music structure; "
            "genre spectral embeddings."
        ),
        "match_level": "medium",
        "modal_extra": [],
        "bold_include_globs": [
            "sub-*/func/*bold.nii.gz",
            "sub-*/func/*events.tsv",
        ],
    },
    "ds004142": {
        "title": "rt-fMRI neurofeedback reward valence",
        "short_title": "Reward valence neurofeedback",
        "authors": "—",
        "role": "cross_ref",
        "priority": 6,
        "url": "https://openneuro.org/datasets/ds004142",
        "n_participants_nominal": 10,
        "cohort": "rt-fMRI reward / saliency valence",
        "modality": ["fMRI-BOLD"],
        "tasks": ["neurofeedback", "reward valence"],
        "why_neuro": "Reward/saliency circuits related to anhedonia constructs.",
        "why_preprocess": "Secondary clinical context; lighter music match.",
        "integration": "Optional reward-network covariates; not primary music PSD.",
        "match_level": "lower",
        "modal_extra": [],
    },
    "ds005700": {
        "title": "NeuroEmo emotion recognition",
        "short_title": "Visual emotion (NeuroEmo)",
        "authors": "—",
        "role": "cross_ref",
        "priority": 7,
        "url": "https://openneuro.org/datasets/ds005700",
        "n_participants_nominal": None,
        "cohort": "Visual emotion with culturally relevant clips",
        "modality": ["fMRI-BOLD"],
        "tasks": ["visual emotion categories"],
        "why_neuro": "Emotion-category contrast; weaker auditory/music match.",
        "why_preprocess": "Lower priority for spectral music pipeline.",
        "integration": "Optional domain for emotion labels only.",
        "match_level": "lower",
        "modal_extra": [],
    },
}


def catalog_rows() -> list[dict[str, Any]]:
    """Flat table rows for marimo / Polars (static scientific catalog)."""
    rows = []
    for ds_id, m in sorted(
        MULTI_DATASET_CATALOG.items(), key=lambda kv: kv[1].get("priority", 99)
    ):
        rows.append(
            {
                "dataset": ds_id,
                "priority": m.get("priority"),
                "role": m.get("role"),
                "match_level": m.get("match_level"),
                "short_title": m.get("short_title"),
                "n_nominal": m.get("n_participants_nominal"),
                "modality": ", ".join(m.get("modality") or []),
                "modal_extra": ", ".join(m.get("modal_extra") or []) or "—",
                "why_neuro": m.get("why_neuro"),
                "integration": m.get("integration"),
                "url": m.get("url"),
            }
        )
    return rows


def integration_roadmap_md() -> str:
    return """
### Integration roadmap (neuroscience + Spark/Delta + marimo)

**Analysis goals**

| Goal | Primary | Cross-refs |
|------|---------|------------|
| Cross-validate multitaper PSD / tSNR / flatness | ds000171 | ds002725, ds003085, ds003720 |
| Multimodal (EEG-BOLD) | — | ds002725 |
| Multimodal (HR–insula) | — | ds004894 |
| Temporal valence dynamics | music contrasts | ds003085 |
| Naturalistic music-in-context | — | ds006564 |
| Clinical MDD vs healthy baseline | MDD×Control | healthy music cohorts |

**Preprocessing policy (pyspark-tal style)**

1. **Spectral / FFT outside JVM** — `pre_ingest_bold_to_parquet.py` (scipy multitaper).  
2. **Spark** — pure DataFrame/SQL rollups, QC aggregates, multi-dataset joins (`dataset_id`, subject, task, run).  
3. **Delta Lake (optional scale path)** — partitioned feature tables with provenance + time-travel.  
4. **Marimo** — Polars for book-scale reactive tables; Spark Connect when multi-set / larger *n*.  
5. **Native Connect** — `./scripts/start_local_spark_connect.sh` (pip `pyspark==4.1.1` + matching Spark binary).

**Practical sequence**

```bash
# 1) Metadata (+ limited BOLD) for cross-refs
python scripts/download_openneuro_cohorts.py
python scripts/download_openneuro_cohorts.py --with-bold --max-subjects 1 --only ds002725,ds004894

# 2) Refresh on-disk registry scan
python scripts/refresh_dataset_registry.py

# 3) Pre-ingest NIfTI → god parquet (multi-set)
python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725

# 4) Catalyst rollups (+ optional Connect / Delta)
python scripts/god_mode_bold_to_tfdata.py --smoke-tfdata
# ./scripts/start_local_spark_connect.sh
# python scripts/god_mode_bold_to_tfdata.py --remote sc://localhost:15002

# 5) Explore
export PYTHONPATH=marimo_notebooks
marimo edit marimo_notebooks/00_data_landscape.py
```

**Feature-store schema (unified)**  
Keys: `dataset_id`, `subject`, `task`, `run`, `group`  
Core measures: `power_{low,mid,high}`, `spectral_centroid`, `spectral_flatness`, `spectral_entropy`, `tsnr`, QC flags  
Provenance: `psd_method`, `tr_sec`, `n_volumes`, `pipeline_version`
"""
