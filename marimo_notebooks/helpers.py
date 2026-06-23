"""Clean helpers for marimo notebooks - direct, no old src wrappers."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
from plotly.subplots import make_subplots

# Colors
CONTROL_COLOR = "#2E86AB"
MDD_COLOR = "#C73E1D"
MUSIC_COLOR = "#4CAF50"
NONMUSIC_COLOR = "#FF9800"
HIGHLIGHT = "#7B2CBF"
GROUP_PALETTE = {"Control": CONTROL_COLOR, "MDD": MDD_COLOR}
CONDITION_PALETTE = {"music": MUSIC_COLOR, "nonmusic": NONMUSIC_COLOR}

def set_global_style():
    plt.style.use("seaborn-v0_8-talk")
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

def get_plotly_template():
    return "plotly_white"

def hypothesis_card(hypothesis: str, prediction: str) -> str:
    return f"""
### 🔬 Hypothesis
**{hypothesis}**

**Expected pattern:** {prediction}

> Music has rich spectral structure. We predict **controls** will show elevated high-frequency / beta-gamma power and stronger auditory-limbic coherence specifically during *positive music*. **MDD** (anhedonia) will exhibit blunted responses.
"""

def key_insight_card(insight: str, evidence: str, effect_size: str = "") -> str:
    eff = f" | Effect: {effect_size}" if effect_size else ""
    return f"""
### 💡 Key Insight
**{insight}**{eff}

*Evidence:* {evidence}
"""

def clinical_relevance_card(text: str, recsys_link: bool = True) -> str:
    link = "\n\n**→ RecSys angle:** Spectral 'responder' fingerprint could drive personalized playlist recommendation (positive music for anhedonic patients)." if recsys_link else ""
    return f"""
### 🏥 Clinical Relevance
{text}{link}
"""

def make_synthetic_bold_dataset(
    n_subjects: int = 20,
    n_timepoints: int = 105,
    tr: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_timepoints = int(n_timepoints)
    tr = float(tr)
    t = np.arange(n_timepoints) * tr

    rows = []
    groups = ["Control"] * (n_subjects // 2) + ["MDD"] * (n_subjects - n_subjects // 2)

    conditions = ["nonmusic", "music_neg", "music_pos"]
    block_dur = max(1, int(31.5 / tr))
    labels = ["tones"] * block_dur + ["negative_music"] * block_dur + ["positive_music"] * block_dur
    trial_labels = (labels * ((n_timepoints // max(1, len(labels))) + 1))[:n_timepoints]

    for i, grp in enumerate(groups):
        base = rng.normal(0, 1, n_timepoints).cumsum()
        base = (base - base.mean()) / base.std()

        for cond in conditions:
            signal = base.copy()

            if cond == "music_pos":
                boost = 0.9 if grp == "Control" else 0.25
                freq = 0.065
                signal += boost * np.sin(2 * np.pi * freq * t + rng.uniform(0, np.pi))
                signal += 0.6 * boost * np.sin(2 * np.pi * 0.12 * t)
            elif cond == "music_neg":
                boost = 0.55 if grp == "Control" else 0.35
                signal += boost * np.sin(2 * np.pi * 0.04 * t)
            else:
                signal += 0.3 * np.sin(2 * np.pi * 0.025 * t)

            signal += rng.normal(0, 0.6, n_timepoints)
            from scipy.ndimage import gaussian_filter1d
            signal = gaussian_filter1d(signal, sigma=1.2)

            for tp in range(n_timepoints):
                rows.append({
                    "subject": f"sub-{grp.lower()}{i:02d}",
                    "group": grp,
                    "condition": "music" if "music" in cond else "nonmusic",
                    "trial_type": trial_labels[tp],
                    "time": t[tp],
                    "bold": signal[tp],
                })

    return pd.DataFrame(rows)

def band_power(f: np.ndarray, Pxx: np.ndarray, low: float, high: float) -> float:
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))

# Direct data loaders - no wrappers
DATA_DIR = Path("data/raw/ds000171")

def load_participants_direct() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")
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
    if path.exists():
        import nibabel as nib
        img = nib.load(str(path))
        return img.get_fdata().mean(axis=(0, 1, 2))
    return np.array([])

def get_direct_runs_summary() -> pd.DataFrame:
    """Direct simple summary without full wrapper."""
    participants = load_participants_direct()
    runs = []
    for sub in participants["participant_id"]:
        for f in (DATA_DIR / sub / "func").glob(f"{sub}_*_bold.nii.gz"):
            # parse
            parts = f.stem.split("_")
            task = parts[2].replace("task-", "")
            run = int(parts[3].replace("run-", ""))
            runs.append({"subject": sub, "task": task, "run": run})
    return pd.DataFrame(runs)

# Spark Connect (prod ready)
from pyspark.sql import SparkSession
def get_connect_spark(remote_uri: str = "sc://localhost:15002") -> "SparkSession":
    """Lightweight client. Heavy lifting on remote Spark cluster."""
    return (
        SparkSession.builder
        .remote(remote_uri)
        .appName("NeuroscienceConnect")
        .getOrCreate()
    )
