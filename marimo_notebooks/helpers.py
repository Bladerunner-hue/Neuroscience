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
]
BOOK_LOCAL_ONLY = [
    (
        "06_tf_spectrogram_model",
        "V-local · Train TensorFlow",
        "Retrain STFT CNN + MLPs locally (optional); results ship via 05.",
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


def pandas_to_polars(pdf: pd.DataFrame):
    """Convert pandas → Polars without requiring pyarrow (WASM / Pyodide safe).

    ``pl.from_pandas`` needs pyarrow when columns use nullable extension dtypes
    (e.g. ``Int64``, ``boolean``). GitHub Pages WASM does not ship pyarrow.
    """
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return None
    if pdf is None or getattr(pdf, "empty", True):
        return pl.DataFrame()

    # Fast path: list-of-dicts from book_bundle (no pandas intermediate)
    # Prefer building from cleaned numpy/object columns.
    data: dict = {}
    for c in pdf.columns:
        s = pdf[c]
        dtype = str(s.dtype)
        try:
            if dtype in ("Int64", "Int32", "UInt64", "UInt32", "boolean", "Float64", "Float32"):
                # nullable extension dtypes → plain float/bool (NaN for missing ints)
                if "boolean" in dtype.lower() or dtype == "bool":
                    data[c] = [
                        None if pd.isna(v) else bool(v) for v in s.tolist()
                    ]
                else:
                    data[c] = pd.to_numeric(s, errors="coerce").astype("float64").tolist()
            elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
                data[c] = pd.to_numeric(s, errors="coerce").astype("float64").tolist()
            elif pd.api.types.is_bool_dtype(s):
                data[c] = s.astype("bool").tolist()
            elif pd.api.types.is_datetime64_any_dtype(s):
                data[c] = s.astype("datetime64[ns]").astype(str).tolist()
            else:
                # strings / objects / nested lists (psd_f as list or JSON str)
                vals = []
                for v in s.tolist():
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        vals.append(None)
                    elif isinstance(v, (list, dict, tuple)):
                        vals.append(v)
                    else:
                        try:
                            if pd.isna(v):
                                vals.append(None)
                            else:
                                vals.append(v if not isinstance(v, (np.generic,)) else v.item())
                        except (TypeError, ValueError):
                            vals.append(str(v))
                data[c] = vals
        except Exception:
            data[c] = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in s.tolist()]

    try:
        return pl.DataFrame(data)
    except Exception:
        # Last resort: stringify everything
        simple = {
            c: [None if v is None else str(v) for v in col]
            for c, col in data.items()
        }
        return pl.DataFrame(simple)


def load_spectral_features_polars():
    """Polars loader for reactive QC / EDA (WASM-safe, no pyarrow required)."""
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return None
    proc = _find_processed()
    if proc and (proc / "spectral_features.csv").exists():
        try:
            return pl.read_csv(proc / "spectral_features.csv")
        except Exception:
            pass
    # Prefer list-of-dicts from embedded book_bundle (no pandas → no pyarrow)
    bundle = load_book_bundle()
    rows = bundle.get("spectral_features") or []
    if rows:
        try:
            return pl.DataFrame(rows)
        except Exception:
            return pandas_to_polars(pd.DataFrame(rows))
    pdf = load_spectral_features()
    return pandas_to_polars(pdf)


def load_condition_features_polars():
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return None
    proc = _find_processed()
    if proc and (proc / "condition_features.csv").exists():
        try:
            return pl.read_csv(proc / "condition_features.csv")
        except Exception:
            pass
    bundle = load_book_bundle()
    rows = bundle.get("condition_features") or []
    if rows:
        try:
            return pl.DataFrame(rows)
        except Exception:
            return pandas_to_polars(pd.DataFrame(rows))
    return pandas_to_polars(load_condition_features())


def load_subject_features_polars():
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return None
    proc = _find_processed()
    if proc and (proc / "subject_features.csv").exists():
        try:
            return pl.read_csv(proc / "subject_features.csv")
        except Exception:
            pass
    bundle = load_book_bundle()
    rows = bundle.get("subject_features") or []
    if rows:
        try:
            return pl.DataFrame(rows)
        except Exception:
            return pandas_to_polars(pd.DataFrame(rows))
    return pandas_to_polars(load_subject_features())


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
    return f"""
### Data provenance

| | |
|---|---|
| **Dataset** | OpenNeuro [ds000171](https://openneuro.org/datasets/ds000171) — Lepping et al. |
| **Full cohort (metadata)** | **{n_full}** participants |
| **Subjects with BOLD in this book** | **{n_subj}** |
| **BOLD runs processed** | **{n_runs}** |
| **Spatial proxies** | Anterior/posterior, L/R, S/I slabs + A–P coherence |
| **PSD** | Multitaper (adaptive default) + Welch baseline · tSNR QC · method=`{b.get("psd_method", "?")}` |
| **Cleaned runs** | **{b.get("n_cleaned_runs", "—")}** (IsolationForest-gated) |
| **Feature store** | `data/processed/` · WASM embed (helpers + spectral_methods + book_bundle JSON) |
| **Source** | `{src}` |

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
