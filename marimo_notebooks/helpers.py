"""Shared helpers for the interactive neuroscience book.

Loads **real** OpenNeuro ds000171 features when present on disk; otherwise uses
the embedded processed subset in ``book_data.BOOK_BUNDLE`` (for WASM / CI).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None  # type: ignore

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
CONTROL_COLOR = "#1B4F72"
MDD_COLOR = "#922B21"
MUSIC_COLOR = "#196F3D"
NONMUSIC_COLOR = "#B9770E"
HIGHLIGHT = "#6C3483"
GROUP_PALETTE = {"Control": CONTROL_COLOR, "MDD": MDD_COLOR}
CONDITION_PALETTE = {"music": MUSIC_COLOR, "nonmusic": NONMUSIC_COLOR}

REPO_CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd(),
]
DATA_DIR = Path("data/raw/ds000171")
PROCESSED_DIR = Path("data/processed")

BOOK_CHAPTERS: list[tuple[str, str, str]] = [
    (
        "00_data_browser",
        "0 · Data browser",
        "Feature tables shared with HTML /explore and the static API.",
    ),
    (
        "00_qc_dashboard",
        "0 · QC Dashboard",
        "tSNR, multitaper PSD, IsolationForest, clean-run export.",
    ),
    ("01_pre_flight", "I · Cohort & Design", "Who was studied, what was heard, and how BOLD meets music."),
    (
        "02_eda_univariate",
        "II · Spectral Power",
        "Multitaper/Welch PSD as a window onto rhythmic BOLD energy.",
    ),
    ("03_eda_multivariate", "III · Algorithm Lab", "Bake-off, best model, confusion, explainability, RecSys."),
    ("04_feature_engineering", "IV · Features & Music Effects", "Inventory, harmonization, spatial proxies, PCA."),
    ("05_tf_results", "V · Neural Net Results", "Precomputed TensorFlow CNN/MLP metrics (trained offline)."),
    (
        "09_multi_dataset_analysis",
        "IX · Multi-dataset scale",
        "Cross-cohort god summary, registry, TF metrics (Spark/TF host jobs).",
    ),
]
BOOK_LOCAL_ONLY = [
    (
        "00_data_landscape",
        "0-local · Data Landscape",
        "Multi-source inventory, scientific value, consistency (pandas + optional Spark Connect).",
    ),
    (
        "00_data_browser",
        "0-local · Data browser",
        "Same /api/table store as HTML /explore (disk · API · WASM).",
    ),
    (
        "06_tf_spectrogram_model",
        "V-local · Train TensorFlow",
        "Retrain STFT CNN + MLPs locally (optional); results ship via 05.",
    ),
    (
        "07_spark_god_mode",
        "VII · Spark God Mode",
        "Catalyst multi-dataset rollups + TFRecords hand-off (native Connect).",
    ),
    (
        "08_spark_streaming",
        "VIII · Spark Streaming",
        "Structured Streaming file/Kafka path with checkpoints.",
    ),
]


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def set_global_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-v0_8-talk", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    rc = {
        "figure.figsize": (10, 5.5),
        "figure.dpi": 120,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFC",
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook", palette="colorblind", rc=rc)
    else:
        plt.rcParams.update(rc)


def callout(kind: str, title: str, body: str) -> str:
    icons = {
        "hypothesis": "Hypothesis",
        "insight": "Key insight",
        "clinical": "Clinical note",
        "method": "Method",
        "data": "Data",
        "caveat": "Caveat",
    }
    label = icons.get(kind, kind.title())
    return f"""
### {label}: {title}

{body}
"""


def hypothesis_card(hypothesis: str, prediction: str) -> str:
    return callout(
        "hypothesis",
        hypothesis,
        f"**Expected pattern.** {prediction}\n\n"
        "Music carries rich harmonic and temporal structure. We expect **controls** to show "
        "stronger high-frequency BOLD energy and tighter auditory–limbic coupling during "
        "*positive music*, while **MDD** shows a blunted, stimulus-specific signature of anhedonia.",
    )


def key_insight_card(insight: str, evidence: str, effect_size: str = "") -> str:
    eff = f"\n\n**Effect size.** {effect_size}" if effect_size else ""
    return callout("insight", insight, f"{evidence}{eff}")


def clinical_relevance_card(text: str, recsys_link: bool = True) -> str:
    link = (
        "\n\n**RecSys angle.** Spectral “responder” fingerprints can inform personalized "
        "playlist systems that target reward engagement rather than genre labels alone."
        if recsys_link
        else ""
    )
    return callout("clinical", "Why this matters", f"{text}{link}")


def book_nav(current_stem: str) -> str:
    stems = [s for s, _, _ in BOOK_CHAPTERS]
    titles = {s: t for s, t, _ in BOOK_CHAPTERS}
    if current_stem not in stems:
        links = " · ".join(f"[{t}](../{s}/)" for s, t, _ in BOOK_CHAPTERS)
        return f"---\n**Book chapters:** {links} · **Home** [Gallery](../../)\n"
    i = stems.index(current_stem)
    left = (
        f"**← Prev** [{titles[stems[i - 1]]}](../{stems[i - 1]}/)"
        if i > 0
        else "**Start of book**"
    )
    right = (
        f"**Next →** [{titles[stems[i + 1]]}](../{stems[i + 1]}/)"
        if i < len(stems) - 1
        else "**Home** [Gallery](../../)"
    )
    return f"---\n{left} · {right}\n"


def _find_processed() -> Path | None:
    for root in REPO_CANDIDATES:
        p = root / "data" / "processed"
        if (p / "spectral_features.csv").exists() or (p / "book_bundle.json").exists():
            return p
    return None


def load_book_bundle() -> dict:
    """Real processed bundle: disk first, then embedded ``book_data``."""
    proc = _find_processed()
    if proc is not None:
        j = proc / "book_bundle.json"
        if j.exists():
            return json.loads(j.read_text())
    try:
        from book_data import BOOK_BUNDLE  # type: ignore

        return BOOK_BUNDLE
    except Exception:
        return {
            "source": "synthetic-fallback",
            "tr_sec": 3.0,
            "n_participants_full": 0,
            "n_bold_runs": 0,
            "participants": [],
            "events_summary": [],
            "spectral_features": [],
            "psd_examples": {},
            "peri_examples": {},
            "timeseries_examples": {},
        }


def load_participants_df() -> pd.DataFrame:
    proc = _find_processed()
    if proc and (proc / "participants_clean.csv").exists():
        return pd.read_csv(proc / "participants_clean.csv")
    bundle = load_book_bundle()
    if bundle.get("participants"):
        return pd.DataFrame(bundle["participants"])
    # last resort synthetic
    rows = []
    for i in range(20):
        rows.append(
            {
                "participant_id": f"sub-control{i+1:02d}" if i < 10 else f"sub-mdd{i-9:02d}",
                "sex": "F" if i % 2 == 0 else "M",
                "age": 20 + (i % 15),
                "group": "Never-Depressed Control" if i < 10 else "Major Depressive Disorder",
                "group_short": "Control" if i < 10 else "MDD",
            }
        )
    return pd.DataFrame(rows)


def _normalize_psd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure psd_f / psd_pxx are JSON strings when present (CSV vs WASM list)."""
    if df is None or getattr(df, "empty", True):
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    for col in ("psd_f", "psd_pxx"):
        if col not in out.columns:
            continue

        def _as_jsonable(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return v
            if isinstance(v, str):
                return v
            if isinstance(v, (list, tuple)):
                return json.dumps(list(v))
            return v

        out[col] = out[col].map(_as_jsonable)
    return out


def load_spectral_features() -> pd.DataFrame:
    """Run-level spectral table. Disk first; WASM uses injected ``book_data`` bundle."""
    proc = _find_processed()
    if proc and (proc / "spectral_features.csv").exists():
        return _normalize_psd_columns(pd.read_csv(proc / "spectral_features.csv"))
    bundle = load_book_bundle()
    if bundle.get("spectral_features"):
        return _normalize_psd_columns(pd.DataFrame(bundle["spectral_features"]))
    return pd.DataFrame()


def load_cleaned_spectral_features() -> pd.DataFrame:
    """QC-gated runs (no IsolationForest outliers). WASM-safe via embedded bundle."""
    proc = _find_processed()
    if proc and (proc / "cleaned_spectral_features.csv").exists():
        return _normalize_psd_columns(pd.read_csv(proc / "cleaned_spectral_features.csv"))
    bundle = load_book_bundle()
    if bundle.get("cleaned_spectral_features"):
        return _normalize_psd_columns(pd.DataFrame(bundle["cleaned_spectral_features"]))
    sf = load_spectral_features()
    if sf.empty:
        return sf
    if "qc_outlier" in sf.columns:
        return sf.loc[sf["qc_outlier"].astype(int) == 0].copy()
    if "qc_flag_any" in sf.columns:
        return sf.loc[sf["qc_flag_any"].astype(int) == 0].copy()
    return sf


def filter_clean_runs(
    df: pd.DataFrame,
    *,
    drop_outliers: bool = True,
    drop_any_flag: bool = False,
) -> pd.DataFrame:
    """Filter a run-level table using QC columns when present."""
    if df is None or getattr(df, "empty", True):
        return df if df is not None else pd.DataFrame()
    out = df
    if drop_outliers and "qc_outlier" in out.columns:
        out = out.loc[out["qc_outlier"] == 0]
    if drop_any_flag and "qc_flag_any" in out.columns:
        out = out.loc[out["qc_flag_any"] == 0]
    return out.copy()


def as_table(df) -> pd.DataFrame:
    """Normalize any small frame to a plain pandas DataFrame for ``mo.ui.table``.

    GitHub Pages / marimo WASM must **not** pull optional columnar stacks:
    the Pyodide worker auto-imports an unavailable Arrow package when the
    notebook source mentions certain dataframe libraries. Use this helper
    (and pandas) for all public-book visualization.
    """
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        out = df.copy()
    elif isinstance(df, list):
        out = pd.DataFrame(df)
    elif isinstance(df, dict):
        out = pd.DataFrame(df)
    elif hasattr(df, "to_pandas"):
        try:
            out = df.to_pandas()
        except Exception:
            try:
                out = pd.DataFrame(df.to_dicts())  # type: ignore[attr-defined]
            except Exception:
                return pd.DataFrame()
    else:
        try:
            out = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()
    # Plain numpy dtypes only (no pandas nullable extension types)
    for c in out.columns:
        s = out[c]
        dtype = str(s.dtype)
        if dtype in ("Int64", "Int32", "UInt64", "UInt32", "Float64", "Float32"):
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif dtype in ("boolean", "bool") or "boolean" in dtype.lower():
            out[c] = s.map(lambda v: None if pd.isna(v) else bool(v))
    return out


def load_spectral_features_frame() -> pd.DataFrame:
    """Run-level spectral table as pandas (public WASM + host)."""
    return as_table(load_spectral_features())


def load_condition_features_frame() -> pd.DataFrame:
    return as_table(load_condition_features())


def load_subject_features_frame() -> pd.DataFrame:
    return as_table(load_subject_features())


def load_bold_timeseries() -> pd.DataFrame:
    proc = _find_processed()
    if proc and (proc / "bold_timeseries.csv").exists():
        return pd.read_csv(proc / "bold_timeseries.csv")
    # rebuild sparse frame from bundle examples
    bundle = load_book_bundle()
    rows = []
    for _k, ex in bundle.get("timeseries_examples", {}).items():
        for t, b in zip(ex["time"], ex["bold_z"]):
            rows.append(
                {
                    "subject": ex["subject"],
                    "group": ex["group"],
                    "task": ex["task"],
                    "run": 1,
                    "time": t,
                    "bold_z": b,
                }
            )
    return pd.DataFrame(rows)


def load_events_summary() -> pd.DataFrame:
    proc = _find_processed()
    if proc and (proc / "events_summary.csv").exists():
        return pd.read_csv(proc / "events_summary.csv")
    bundle = load_book_bundle()
    if bundle.get("events_summary"):
        return pd.DataFrame(bundle["events_summary"])
    return pd.DataFrame()


def load_condition_features() -> pd.DataFrame:
    """Trial-type features: positive/negative music, tones, nonmusic."""
    proc = _find_processed()
    if proc and (proc / "condition_features.csv").exists():
        return pd.read_csv(proc / "condition_features.csv")
    bundle = load_book_bundle()
    if bundle.get("condition_features"):
        return pd.DataFrame(bundle["condition_features"])
    return pd.DataFrame()


def load_subject_features() -> pd.DataFrame:
    """Subject-level wide table with music-effect contrasts for ML."""
    proc = _find_processed()
    if proc and (proc / "subject_features.csv").exists():
        return pd.read_csv(proc / "subject_features.csv")
    bundle = load_book_bundle()
    if bundle.get("subject_features"):
        return pd.DataFrame(bundle["subject_features"])
    return pd.DataFrame()


def load_spatial_connectivity() -> pd.DataFrame:
    """Anterior–posterior / L–R / S–I band coherence (pseudo-ROI seeds)."""
    proc = _find_processed()
    if proc and (proc / "spatial_connectivity.csv").exists():
        return pd.read_csv(proc / "spatial_connectivity.csv")
    bundle = load_book_bundle()
    if bundle.get("spatial_connectivity"):
        return pd.DataFrame(bundle["spatial_connectivity"])
    return pd.DataFrame()


def load_ml_bakeoff() -> dict:
    """Precomputed LOOCV algorithm bake-off (winners, CMs, importances)."""
    proc = _find_processed()
    if proc and (proc / "ml_bakeoff.json").exists():
        return json.loads((proc / "ml_bakeoff.json").read_text())
    bundle = load_book_bundle()
    return bundle.get("ml_bakeoff") or {}


def load_tf_results() -> dict:
    """Precomputed TensorFlow offline train results (no TF needed to load)."""
    proc = _find_processed()
    if proc and (proc / "tf_results.json").exists():
        return json.loads((proc / "tf_results.json").read_text())
    bundle = load_book_bundle()
    return bundle.get("tf_results") or {}


def load_psd_examples() -> dict:
    """Group×task example PSDs for WASM fallbacks (when per-run PSD columns missing)."""
    return load_book_bundle().get("psd_examples") or {}


def load_run_qc() -> pd.DataFrame:
    """Run-level QC table (tSNR, flatness, IsolationForest flags)."""
    proc = _find_processed()
    if proc and (proc / "run_qc.csv").exists():
        return pd.read_csv(proc / "run_qc.csv")
    bundle = load_book_bundle()
    if bundle.get("run_qc"):
        return pd.DataFrame(bundle["run_qc"])
    # fallback: derive from spectral features if QC columns present
    sf = load_spectral_features()
    if not sf.empty and "tsnr" in sf.columns:
        cols = [
            c
            for c in [
                "subject",
                "group",
                "task",
                "run",
                "tsnr",
                "spectral_flatness",
                "spectral_entropy",
                "band_snr_high",
                "ts_spike_frac",
                "qc_outlier",
                "qc_flag_any",
            ]
            if c in sf.columns
        ]
        return sf[cols]
    return pd.DataFrame()


def load_dataset_registry() -> dict:
    """OpenNeuro multi-cohort registry (primary + cross-refs)."""
    for root in REPO_CANDIDATES:
        p = root / "data" / "processed" / "dataset_registry.json"
        if p.exists():
            return json.loads(p.read_text())
    return {}


def inventory_data_sources() -> list[dict]:
    """Catalog raw BIDS, processed feature store, and god-mode parquet tables.

    Each row documents *what exists*, *scale*, and *scientific role* so chapters
    can open with a multi-source view before QC / spectral analysis.
    """
    proc = _find_processed()
    rows: list[dict] = []

    # --- registry / raw BIDS ---
    reg = load_dataset_registry()
    for ds_id, meta in reg.items():
        local = None
        for root in REPO_CANDIDATES:
            cand = root / str(meta.get("local_path") or f"data/raw/{ds_id}")
            if cand.exists():
                local = cand
                break
        rows.append(
            {
                "layer": "raw_bids",
                "name": ds_id,
                "artifact": str(meta.get("local_path") or f"data/raw/{ds_id}"),
                "exists": bool(local and local.exists()),
                "n_rows": int(meta.get("n_subjects_on_disk") or 0),
                "n_extra": int(meta.get("n_bold_files") or 0),
                "role": meta.get("role") or "cross_ref",
                "scientific_value": meta.get("why") or meta.get("title") or "",
                "status": meta.get("status") or ("ok" if local else "missing"),
            }
        )
    if not reg:
        # primary only if registry absent
        for root in REPO_CANDIDATES:
            p = root / "data" / "raw" / "ds000171"
            if p.exists():
                rows.append(
                    {
                        "layer": "raw_bids",
                        "name": "ds000171",
                        "artifact": "data/raw/ds000171",
                        "exists": True,
                        "n_rows": 0,
                        "n_extra": 0,
                        "role": "primary",
                        "scientific_value": "Core MDD vs Control music/tones design",
                        "status": "ok",
                    }
                )
                break

    # --- classic feature store (book / WASM path) ---
    feature_specs = [
        (
            "spectral_features.csv",
            "run_level PSD bands + tSNR/QC",
            "Per-run spectral energy; unit of QC and univariate EDA",
        ),
        (
            "cleaned_spectral_features.csv",
            "IsolationForest-gated runs",
            "Analysis-ready runs after QC filters",
        ),
        (
            "condition_features.csv",
            "trial_type epochs",
            "Music vs nonmusic vs tones contrasts",
        ),
        (
            "subject_features.csv",
            "subject-wide + responder score R",
            "ML bake-off / personalized music effect",
        ),
        (
            "run_qc.csv",
            "tSNR / flatness / spike / IF flags",
            "Run-level quality gate before claims",
        ),
        (
            "participants_clean.csv",
            "demographics + group",
            "Cohort composition and stratification",
        ),
        (
            "events_summary.csv",
            "block design inventory",
            "Stimulus timing / trial_type coverage",
        ),
        (
            "spatial_connectivity.csv",
            "pseudo-ROI band coherence",
            "Anterior–posterior / L–R spatial proxies",
        ),
        (
            "bold_timeseries.csv",
            "mean BOLD z traces",
            "Time-domain examples / peri-stimulus plots",
        ),
        (
            "ml_bakeoff.json",
            "LOOCV leaderboards",
            "Classical ML gold standard on small n",
        ),
        (
            "tf_results.json",
            "offline CNN/MLP metrics",
            "Neural baselines without retrain",
        ),
        (
            "book_bundle.json",
            "WASM embed payload",
            "Browser book without local NIfTI",
        ),
    ]
    if proc is not None:
        for fname, content, value in feature_specs:
            path = proc / fname
            n_rows = 0
            if path.exists() and path.suffix == ".csv":
                try:
                    n_rows = sum(1 for _ in path.open()) - 1
                except OSError:
                    n_rows = 0
            elif path.exists() and path.suffix == ".json":
                try:
                    blob = json.loads(path.read_text())
                    n_rows = len(blob) if isinstance(blob, (list, dict)) else 1
                except Exception:
                    n_rows = 1
            rows.append(
                {
                    "layer": "feature_store",
                    "name": fname,
                    "artifact": f"data/processed/{fname}",
                    "exists": path.exists(),
                    "n_rows": int(n_rows),
                    "n_extra": int(path.stat().st_size) if path.exists() else 0,
                    "role": "primary_book",
                    "scientific_value": f"{content} — {value}",
                    "status": "ok" if path.exists() else "missing",
                }
            )

    # --- god-mode / multi-dataset scale path ---
    god_specs = [
        (
            "god_parquet_bold/run_spectral",
            "multi-dataset run spectral parquet",
            "Horizontal scale: same schema across OpenNeuro cohorts",
        ),
        (
            "god_parquet_bold/epoch_ts",
            "epoch time-series arrays",
            "Block-locked BOLD for streaming / TF hand-off",
        ),
        (
            "god_features/run_level",
            "Catalyst run features",
            "Spark-rolled run metrics (no Python UDFs)",
        ),
        (
            "god_features/subject_level",
            "Catalyst subject rollups",
            "Music vs nonmusic contrasts at subject grain",
        ),
        (
            "god_features/run_qc",
            "Catalyst QC table",
            "Distributed QC flags aligned to multi-set schema",
        ),
        (
            "dataset_registry.json",
            "cohort inventory",
            "What is on disk vs planned cross-refs",
        ),
    ]
    for rel, content, value in god_specs:
        path = None
        for root in REPO_CANDIDATES:
            cand = root / "data" / "processed" / rel
            if cand.exists():
                path = cand
                break
        n_rows = 0
        n_extra = 0
        if path is not None:
            if path.is_dir():
                parts = list(path.glob("**/*.parquet"))
                n_extra = sum(p.stat().st_size for p in parts)
                # Avoid optional parquet engines here (not available in WASM).
                n_rows = len(parts)
            elif path.suffix == ".json":
                try:
                    blob = json.loads(path.read_text())
                    n_rows = len(blob) if isinstance(blob, dict) else 1
                    n_extra = path.stat().st_size
                except Exception:
                    n_extra = path.stat().st_size
        rows.append(
            {
                "layer": "god_mode",
                "name": rel.split("/")[-1],
                "artifact": f"data/processed/{rel}",
                "exists": path is not None and path.exists(),
                "n_rows": int(n_rows),
                "n_extra": int(n_extra),
                "role": "scale_path",
                "scientific_value": f"{content} — {value}",
                "status": "ok" if path is not None and path.exists() else "missing",
            }
        )

    return rows


def inventory_dataframe() -> pd.DataFrame:
    """Small inventory table for marimo ``mo.ui.table``."""
    return pd.DataFrame(inventory_data_sources())


def data_sources_scientific_md() -> str:
    """Markdown primer: what each layer brings to the music–reward question."""
    return """
### Why multiple sources (scientific logic)

| Layer | Grain | What it contributes to the claim |
|-------|-------|----------------------------------|
| **raw BIDS** (`ds000171` + cross-refs) | subject / run NIfTI + events | Ground truth timing, demographics, full BOLD field |
| **feature store** (CSV / book_bundle) | run · condition · subject | Reproducible spectral biomarkers for the *public* book |
| **god-mode parquet** | multi-dataset run / epoch | Same schema at larger *n*; Catalyst rollups without re-writing WASM |
| **ML / TF JSON** | subject metrics | Classical LOOCV + neural baselines (frozen evidence) |

**Primary design (ds000171):** 39 participants (19 MDD, 20 never-depressed Control) × emotional **music** vs **nonmusic** / tones (Siemens Skyra 3T).
Cross-refs (ds002725 EEG-fMRI music, ds003085 happy/sad dynamics, ds004894 HR–insula, ds006564 naturalistic, ds003720 genre, …) test whether spectral signatures *generalize* — they do **not** replace the primary clinical contrast.

**Consistency rule:** multi-dataset tables share keys `(dataset, subject, task, run)`; spectral columns use the same band definitions as `prepare_real_features.py`. Spark only *rolls up* pre-ingested parquet — multitaper stays outside the JVM. Delta Lake is optional for ACID multi-set stores.

See `multi_dataset_catalog.py` for ranked integration metadata.
"""


def load_god_run_level_df() -> pd.DataFrame:
    """Multi-dataset Catalyst run_level as pandas.

    Prefer ``god_run_summary.json`` records (WASM-safe). On host, try reading
    parquet via pandas if an engine is installed; never hard-require it.
    """
    # Prefer pre-aggregated JSON (always works on Pages / Pyodide)
    god = _load_god_run_summary_dict()
    recs = god.get("records") or []
    if recs:
        return pd.DataFrame(recs)

    for root in REPO_CANDIDATES:
        p = root / "data" / "processed" / "god_features" / "run_level"
        if not p.exists():
            continue
        # Optional host path — import by string pieces so WASM source scanners
        # do not auto-install unavailable packages.
        try:
            import importlib

            eng = importlib.import_module("pand" + "as")
            return eng.read_parquet(str(p))
        except Exception:
            try:
                parts = sorted(p.glob("**/*.parquet"))
                if not parts:
                    continue
                frames = []
                for part in parts:
                    try:
                        frames.append(pd.read_parquet(part))
                    except Exception:
                        continue
                if frames:
                    return pd.concat(frames, ignore_index=True)
            except Exception:
                pass
    return pd.DataFrame()


def _repo_root() -> Path | None:
    for root in REPO_CANDIDATES:
        if (root / "data").exists() or (root / "marimo_notebooks").exists():
            return root
    return None


def _load_god_run_summary_dict() -> dict:
    """Disk / book_data god_run_summary (WASM-safe summary path)."""
    for root in REPO_CANDIDATES:
        p = root / "data" / "processed" / "god_run_summary.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    try:
        import book_data as bd  # type: ignore

        if hasattr(bd, "GOD_RUN_SUMMARY") and bd.GOD_RUN_SUMMARY:
            return dict(bd.GOD_RUN_SUMMARY)
    except Exception:
        pass
    return {}


def studies_table_rows() -> list[dict]:
    """One row per OpenNeuro study under data/raw + registry/catalog metadata.

    Used by marimo chapters so every page can surface cross-ref cohorts, not only
    the primary ds000171 feature store.
    """
    reg = load_dataset_registry()
    cat: dict = {}
    try:
        from multi_dataset_catalog import MULTI_DATASET_CATALOG

        cat = MULTI_DATASET_CATALOG
    except Exception:
        pass

    root = _repo_root()
    raw_ids: set[str] = set()
    if root is not None:
        raw = root / "data" / "raw"
        if raw.exists():
            raw_ids = {
                p.name
                for p in raw.iterdir()
                if p.is_dir() and p.name.startswith("ds")
            }

    ids = sorted(
        set(reg.keys()) | set(cat.keys()) | raw_ids,
        key=lambda d: (cat.get(d, {}).get("priority", reg.get(d, {}).get("priority", 99)), d),
    )
    rows: list[dict] = []
    for ds_id in ids:
        meta = dict(reg.get(ds_id) or {})
        c = cat.get(ds_id) or {}
        local_path = meta.get("local_path") or f"data/raw/{ds_id}"
        on_disk = False
        if root is not None:
            on_disk = (root / local_path).exists() or (root / "data" / "raw" / ds_id).exists()
        n_sub = int(meta.get("n_subjects_on_disk") or 0)
        n_bold = int(meta.get("n_bold_files") or 0)
        status = meta.get("status")
        if not status:
            if not on_disk:
                status = "missing"
            elif n_bold > 0 or n_sub > 0 or meta.get("downloaded_meta"):
                status = "ok"
            else:
                status = "stub"
        rows.append(
            {
                "dataset": ds_id,
                "priority": c.get("priority", meta.get("priority")),
                "role": meta.get("role") or c.get("role") or "cross_ref",
                "match_level": meta.get("match_level") or c.get("match_level"),
                "short_title": meta.get("short_title")
                or c.get("short_title")
                or meta.get("title")
                or c.get("title")
                or ds_id,
                "status": status,
                "n_subjects_on_disk": n_sub,
                "n_bold_files": n_bold,
                "n_nominal": meta.get("n_participants_nominal")
                or c.get("n_participants_nominal"),
                "modality": ", ".join(
                    meta.get("modality") or c.get("modality") or []
                )
                or "—",
                "local_path": local_path,
                "url": meta.get("url") or c.get("url"),
                "why": (meta.get("why") or c.get("why_neuro") or "")[:180],
                "integration": (meta.get("integration") or c.get("integration") or "")[
                    :160
                ],
                "on_disk": on_disk,
            }
        )
    return rows


def studies_dataframe() -> pd.DataFrame:
    """Pandas table of all catalogued / on-disk OpenNeuro studies."""
    return pd.DataFrame(studies_table_rows())


def multi_dataset_run_ids() -> list[str]:
    """Dataset IDs that appear in multi-set spectral runs (god path or summary)."""
    df = load_multi_dataset_runs()
    if df is None or getattr(df, "empty", True) or "dataset" not in df.columns:
        return ["ds000171"]
    return sorted(df["dataset"].astype(str).unique().tolist())


def load_multi_dataset_runs() -> pd.DataFrame:
    """Unified run-level spectral table with a ``dataset`` column.

    Priority:
      1. ``god_run_summary.json`` / god run_level (multi-set)
      2. Primary book ``spectral_features.csv`` tagged as ds000171
    """
    try:
        pdf = load_god_run_level_df()
        if pdf is not None and not getattr(pdf, "empty", True):
            if "dataset" not in pdf.columns:
                pdf = pdf.copy()
                pdf.insert(0, "dataset", "ds000171")
            return pdf
    except Exception:
        pass

    sf = load_spectral_features()
    if sf is None or getattr(sf, "empty", True):
        return pd.DataFrame()
    out = sf.copy()
    if "dataset" not in out.columns:
        out.insert(0, "dataset", "ds000171")
    return out


def load_multi_dataset_runs_frame() -> pd.DataFrame:
    """Multi-set runs as pandas (alias of :func:`load_multi_dataset_runs`)."""
    return as_table(load_multi_dataset_runs())


def load_raw_participants(ds_id: str = "ds000171") -> pd.DataFrame:
    """Read participants.tsv (or participants_clean for primary) for any study."""
    if ds_id == "ds000171":
        try:
            clean = load_participants_df()
            if clean is not None and not clean.empty:
                return clean
        except Exception:
            pass
    root = _repo_root()
    if root is None:
        return pd.DataFrame()
    candidates = [
        root / "data" / "raw" / ds_id / "participants.tsv",
        root / "data" / "raw" / ds_id / ds_id / "participants.tsv",
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p, sep="\t")
            except Exception:
                continue
    return pd.DataFrame()


def multi_studies_overview_md() -> str:
    """Short markdown block listing every study under data/raw for chapter intros."""
    rows = studies_table_rows()
    if not rows:
        return (
            "### OpenNeuro studies\n\n"
            "*No multi-dataset registry yet — run "
            "`python scripts/refresh_dataset_registry.py`.*"
        )
    lines = [
        "### OpenNeuro studies in this book",
        "",
        "| Dataset | Role | Status | Subjects | BOLD | Focus |",
        "|---------|------|--------|---------:|-----:|-------|",
    ]
    for r in rows:
        focus = (r.get("short_title") or "")[:42]
        lines.append(
            f"| [`{r['dataset']}`]({r.get('url') or '#'}) | {r.get('role')} | "
            f"**{r.get('status')}** | {r.get('n_subjects_on_disk')} | "
            f"{r.get('n_bold_files')} | {focus} |"
        )
    multi_ids = multi_dataset_run_ids()
    lines.extend(
        [
            "",
            f"**Spectral multi-set runs available for:** "
            f"{', '.join(f'`{d}`' for d in multi_ids)}  ",
            "Primary clinical claims use **ds000171**; cross-refs test generalization.  ",
            "Explore: `00_data_landscape` · `09_multi_dataset_analysis` · `/explore/`.",
        ]
    )
    return "\n".join(lines)


def primary_cohort_summary() -> dict:
    """Quick stats from processed primary feature store."""
    sf = load_spectral_features()
    parts = load_participants_df()
    subj = load_subject_features()
    out: dict = {
        "n_spectral_runs": int(len(sf)) if sf is not None else 0,
        "n_participants_meta": int(len(parts)) if parts is not None else 0,
        "n_subject_features": int(len(subj)) if subj is not None else 0,
    }
    if sf is not None and not sf.empty:
        if "group" in sf.columns:
            out["runs_by_group"] = sf["group"].value_counts().to_dict()
        if "task" in sf.columns:
            out["runs_by_task"] = sf["task"].value_counts().to_dict()
        for c in ("tsnr", "power_high", "spectral_flatness"):
            if c in sf.columns:
                out[f"mean_{c}"] = float(pd.to_numeric(sf[c], errors="coerce").mean())
    return out


def data_dictionary_md() -> str:
    return """
### What each stimulus is (ds000171 events)

| `trial_type` | Domain | Valence | Meaning in the paradigm |
|---|---|---|---|
| `positive_music` | music | positive | Validated emotionally positive musical excerpts |
| `negative_music` | music | negative | Validated emotionally negative musical excerpts |
| `positive_nonmusic` | non-music | positive | Non-musical positive auditory material |
| `negative_nonmusic` | non-music | negative | Non-musical negative auditory material |
| `tones` | control | neutral | Tone blocks / baseline auditory control |
| `response` | task | — | Brief response window (excluded from spectra) |

**How we use BOLD:** whole-brain spatial mean → z-score within run → (a) run-level Welch PSD, (b) epoch means locked to each `trial_type`, (c) contrasts such as *positive music − tones* (music effect) and *positive music − negative music* (valence within music).
"""


def data_provenance_md() -> str:
    b = load_book_bundle()
    n_full = b.get("n_participants_full", 0)
    n_runs = b.get("n_bold_runs", 0)
    n_subj = b.get("n_subjects_with_bold", 0)
    src = b.get("source", "unknown")
    studies = studies_table_rows()
    n_studies = len(studies)
    multi_ids = multi_dataset_run_ids()
    multi_note = (
        ", ".join(f"`{d}`" for d in multi_ids)
        if multi_ids
        else "`ds000171` only"
    )
    return f"""
### Data provenance

| | |
|---|---|
| **Primary dataset** | OpenNeuro [ds000171](https://openneuro.org/datasets/ds000171) — Lepping et al. |
| **OpenNeuro studies on disk / registry** | **{n_studies}** (see multi-set table below) |
| **Multi-set spectral runs** | {multi_note} |
| **Full cohort (metadata)** | **{n_full}** participants |
| **Subjects with BOLD in this book** | **{n_subj}** |
| **BOLD runs processed (primary store)** | **{n_runs}** |
| **Spatial proxies** | Anterior/posterior, L/R, S/I slabs + A–P coherence |
| **PSD** | Multitaper (adaptive default) + Welch baseline · tSNR QC · method=`{b.get("psd_method", "?")}` |
| **Cleaned runs** | **{b.get("n_cleaned_runs", "—")}** (IsolationForest-gated) |
| **Feature store** | `data/processed/` · WASM embed (helpers + spectral_methods + book_bundle JSON) |
| **Source** | `{src}` |

{multi_studies_overview_md()}

{data_dictionary_md()}
"""


# Keep synthetic generator only for unit demos / optional augmentation
def make_synthetic_bold_dataset(
    n_subjects: int = 20,
    n_timepoints: int = 105,
    tr: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Legacy synthetic generator (not used for the public book when real data exist)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_timepoints)) * float(tr)
    n_control = n_subjects // 2
    groups = ["Control"] * n_control + ["MDD"] * (n_subjects - n_control)
    rows = []
    for i, grp in enumerate(groups):
        base = rng.normal(0, 1, len(t)).cumsum()
        base = (base - base.mean()) / (base.std() + 1e-8)
        for task in ("music", "nonmusic"):
            sig = base + 0.3 * np.sin(2 * np.pi * 0.05 * t)
            if task == "music" and grp == "Control":
                sig = sig + 0.5 * np.sin(2 * np.pi * 0.1 * t)
            for tp, v in enumerate(sig):
                rows.append(
                    {
                        "subject": f"sub-{grp.lower()}{i:02d}",
                        "group": grp,
                        "task": task,
                        "condition": task,
                        "trial_type": "positive_music" if task == "music" else "tones",
                        "time": float(t[tp]),
                        "bold": float(v),
                        "bold_z": float(v),
                    }
                )
    return pd.DataFrame(rows)
