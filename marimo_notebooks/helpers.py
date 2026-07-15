"""Shared helpers for marimo_notebooks (canonical library).

WASM-safe: no hard imports of TensorFlow, PySpark, nilearn, or nibabel.
Optional loaders degrade gracefully when data/packages are missing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
CONTROL_COLOR = "#2E86AB"
MDD_COLOR = "#C73E1D"
MUSIC_COLOR = "#4CAF50"
NONMUSIC_COLOR = "#FF9800"
HIGHLIGHT = "#7B2CBF"
GROUP_PALETTE = {"Control": CONTROL_COLOR, "MDD": MDD_COLOR}
CONDITION_PALETTE = {"music": MUSIC_COLOR, "nonmusic": NONMUSIC_COLOR}

DATA_DIR = Path("data/raw/ds000171")

# Book chapter order (stem → title) — used by nav + export
BOOK_CHAPTERS: list[tuple[str, str]] = [
    ("01_pre_flight", "01 — Pre-flight"),
    ("02_eda_univariate", "02 — Spectral Power"),
    ("03_eda_multivariate", "03 — Coherence"),
    ("04_feature_engineering", "04 — Features & Clusters"),
]
BOOK_LOCAL_ONLY = [
    ("06_tf_spectrogram_model", "06 — TF Spectrograms (local)"),
]


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    """numpy 1.x / 2.x compatible integral."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def set_global_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-talk")
    except OSError:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="colorblind",
        rc={
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
        },
    )


def hypothesis_card(hypothesis: str, prediction: str) -> str:
    return f"""
### 🔬 Hypothesis
**{hypothesis}**

**Expected pattern:** {prediction}

> Music has rich spectral structure. We predict **controls** show elevated high-frequency power and stronger auditory–limbic coherence during *positive music*. **MDD** (anhedonia) exhibits blunted responses.
"""


def key_insight_card(insight: str, evidence: str, effect_size: str = "") -> str:
    eff = f" | Effect: {effect_size}" if effect_size else ""
    return f"""
### 💡 Key Insight
**{insight}**{eff}

*Evidence:* {evidence}
"""


def clinical_relevance_card(text: str, recsys_link: bool = True) -> str:
    link = (
        "\n\n**→ RecSys angle:** Spectral responder fingerprints can drive personalized "
        "playlist recommendation for anhedonic patients."
        if recsys_link
        else ""
    )
    return f"""
### 🏥 Clinical Relevance
{text}{link}
"""


def book_nav(current_stem: str, *, wasm: bool = True) -> str:
    """Markdown prev/next links for the interactive book."""
    stems = [s for s, _ in BOOK_CHAPTERS]
    titles = dict(BOOK_CHAPTERS)
    if current_stem not in stems:
        # local-only chapter
        return (
            "---\n"
            f"**Book chapters (WASM):** "
            + " · ".join(f"[{t}](../{s}/)" for s, t in BOOK_CHAPTERS)
            + " · **Home** [Gallery](../../)\n"
        )
    i = stems.index(current_stem)
    parts = ["---"]
    prev_s = stems[i - 1] if i > 0 else None
    next_s = stems[i + 1] if i < len(stems) - 1 else None
    left = f"**← Previous** [{titles[prev_s]}](../{prev_s}/)" if prev_s else "**Book start**"
    right = (
        f"**Next →** [{titles[next_s]}](../{next_s}/)"
        if next_s
        else "**Home** [Gallery](../../)"
    )
    parts.append(f"{left} · {right}")
    return "\n".join(parts)


def make_synthetic_bold_dataset(
    n_subjects: int = 20,
    n_timepoints: int = 105,
    tr: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic BOLD-like traces with group × condition structure (ds000171-faithful)."""
    rng = np.random.default_rng(seed)
    n_timepoints = int(n_timepoints)
    tr = float(tr)
    t = np.arange(n_timepoints) * tr

    n_control = n_subjects // 2
    groups = ["Control"] * n_control + ["MDD"] * (n_subjects - n_control)
    block_dur = max(1, int(31.5 / tr))
    labels = (
        ["tones"] * block_dur
        + ["negative_music"] * block_dur
        + ["positive_music"] * block_dur
    )
    trial_labels = (labels * ((n_timepoints // max(1, len(labels))) + 1))[:n_timepoints]

    try:
        from scipy.ndimage import gaussian_filter1d

        def smooth(x: np.ndarray) -> np.ndarray:
            return gaussian_filter1d(x, sigma=1.2)
    except Exception:

        def smooth(x: np.ndarray) -> np.ndarray:
            k = 3
            pad = np.pad(x, (k, k), mode="edge")
            ker = np.ones(2 * k + 1) / (2 * k + 1)
            return np.convolve(pad, ker, mode="valid")

    rows: list[dict] = []
    for i, grp in enumerate(groups):
        base = rng.normal(0, 1, n_timepoints).cumsum()
        base = (base - base.mean()) / (base.std() + 1e-8)
        for cond in ("nonmusic", "music_neg", "music_pos"):
            signal = base.copy()
            if cond == "music_pos":
                boost = 0.9 if grp == "Control" else 0.25
                signal += boost * np.sin(2 * np.pi * 0.065 * t + rng.uniform(0, np.pi))
                signal += 0.6 * boost * np.sin(2 * np.pi * 0.12 * t)
            elif cond == "music_neg":
                boost = 0.55 if grp == "Control" else 0.35
                signal += boost * np.sin(2 * np.pi * 0.04 * t)
            else:
                signal += 0.3 * np.sin(2 * np.pi * 0.025 * t)
            signal += rng.normal(0, 0.6, n_timepoints)
            signal = smooth(signal)
            for tp in range(n_timepoints):
                rows.append(
                    {
                        "subject": f"sub-{grp.lower()}{i:02d}",
                        "group": grp,
                        "condition": "music" if "music" in cond else "nonmusic",
                        "trial_type": trial_labels[tp],
                        "time": float(t[tp]),
                        "bold": float(signal[tp]),
                    }
                )
    return pd.DataFrame(rows)


def band_power(f: np.ndarray, Pxx: np.ndarray, low: float, high: float) -> float:
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return trapz_integral(Pxx[mask], f[mask])


def load_participants_direct() -> pd.DataFrame:
    path = DATA_DIR / "participants.tsv"
    if not path.exists():
        rows = []
        for i in range(10):
            is_ctrl = i < 5
            rows.append(
                {
                    "participant_id": f"sub-{'control' if is_ctrl else 'mdd'}{i % 5 + 1:02d}",
                    "group": (
                        "Never-Depressed Control"
                        if is_ctrl
                        else "Major Depressive Disorder"
                    ),
                    "group_short": "Control" if is_ctrl else "MDD",
                }
            )
        return pd.DataFrame(rows)
    df = pd.read_csv(path, sep="\t")
    df["group_short"] = df["group"].map(
        {"Major Depressive Disorder": "MDD", "Never-Depressed Control": "Control"}
    )
    return df


def load_events_direct(subject: str, task: str, run: int) -> pd.DataFrame:
    path = DATA_DIR / subject / "func" / f"{subject}_task-{task}_run-{run}_events.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame()


def load_bold_mean_direct(subject: str, task: str, run: int) -> np.ndarray:
    path = DATA_DIR / subject / "func" / f"{subject}_task-{task}_run-{run}_bold.nii.gz"
    if not path.exists():
        return np.array([])
    try:
        import nibabel as nib

        return nib.load(str(path)).get_fdata().mean(axis=(0, 1, 2))
    except Exception:
        return np.array([])


def get_connect_spark(remote_uri: str = "sc://localhost:15002"):
    """Lazy Spark Connect client (local/cluster only — not available in WASM)."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise ImportError(
            "pyspark is not installed. pip install 'pyspark[connect]'"
        ) from e
    return (
        SparkSession.builder.remote(remote_uri)
        .appName("NeuroscienceConnect")
        .getOrCreate()
    )
