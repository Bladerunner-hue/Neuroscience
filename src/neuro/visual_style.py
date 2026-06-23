"""Visual style, interactive helpers, and insight renderers for the Music & Depression spectral project.

Designed for both classic Jupyter and marimo reactive notebooks.
- Consistent "talk" styling
- Plotly interactive first-class
- Hypothesis / Insight / Clinical Relevance cards
- Easy integration with mo.ui in marimo
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
from plotly.subplots import make_subplots

from neuro.bids import inventory_runs
from neuro.config import DATA_ROOT
from neuro.features import parse_events

try:
    from nilearn import plotting as nplt
    from nilearn.image import mean_img
    NILEARN_VIZ_AVAILABLE = True
except Exception:
    NILEARN_VIZ_AVAILABLE = False

# Palettes tuned for story: Controls (cool), MDD (warm muted)
CONTROL_COLOR = "#2E86AB"  # deep teal-blue
MDD_COLOR = "#C73E1D"      # warm red
MUSIC_COLOR = "#4CAF50"    # positive green
NONMUSIC_COLOR = "#FF9800" # tones orange
HIGHLIGHT = "#7B2CBF"      # spectral accent

GROUP_PALETTE = {"Control": CONTROL_COLOR, "MDD": MDD_COLOR}
CONDITION_PALETTE = {"music": MUSIC_COLOR, "nonmusic": NONMUSIC_COLOR}

def set_global_style():
    """Apply global style for matplotlib/seaborn (classic notebooks + marimo)."""
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
    """Return a clean modern plotly template."""
    return "plotly_white"

def add_narrative_header(title: str, subtitle: str = "") -> str:
    """Markdown header block for marimo or jupyter display."""
    md = f"# {title}\n\n"
    if subtitle:
        md += f"_{subtitle}_\n\n"
    return md

def hypothesis_card(hypothesis: str, prediction: str) -> str:
    """Beautiful hypothesis section."""
    return f"""
### 🔬 Hypothesis
**{hypothesis}**

**Expected pattern:** {prediction}

> Music has rich spectral structure. We predict **controls** will show elevated high-frequency / beta-gamma power and stronger auditory-limbic coherence specifically during *positive music*. **MDD** (anhedonia) will exhibit blunted responses.
"""

def key_insight_card(insight: str, evidence: str, effect_size: str = "") -> str:
    """Actionable key insight."""
    eff = f" | Effect: {effect_size}" if effect_size else ""
    return f"""
### 💡 Key Insight
**{insight}**{eff}

*Evidence:* {evidence}
"""

def clinical_relevance_card(text: str, recsys_link: bool = True) -> str:
    """Link findings to real-world (music therapy + RecSys)."""
    link = "\n\n**→ RecSys angle:** Spectral 'responder' fingerprint could drive personalized playlist recommendation (positive music for anhedonic patients)." if recsys_link else ""
    return f"""
### 🏥 Clinical Relevance
{text}{link}
"""

def plot_psd_comparison_plotly(
    freqs: np.ndarray,
    spectra: dict[str, np.ndarray],
    title: str = "Power Spectral Density — Music vs Non-Music",
    bands: tuple[float, float] | None = (0.01, 0.1),
) -> go.Figure:
    """
    Interactive Plotly PSD comparison.
    spectra keys: e.g. "Control / positive_music", "MDD / positive_music"
    """
    fig = go.Figure()
    for label, power in spectra.items():
        is_mdd = "MDD" in label
        color = MDD_COLOR if is_mdd else CONTROL_COLOR
        dash = "dash" if "nonmusic" in label.lower() or "tones" in label.lower() else "solid"
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=power,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5, dash=dash),
            )
        )

    if bands:
        fig.add_vrect(
            x0=bands[0], x1=bands[1],
            fillcolor="rgba(123, 44, 191, 0.12)",
            layer="below", line_width=0,
            annotation_text="BOLD relevant band", annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (a.u.) — log scale recommended",
        template=get_plotly_template(),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        hovermode="x unified",
        height=520,
    )
    fig.update_yaxes(type="log")
    return fig

def plot_group_difference_bars_plotly(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
    condition_col: str = "condition",
    title: str = "Spectral Feature by Group × Condition",
) -> go.Figure:
    """Grouped bar or box showing effect of interest."""
    fig = px.box(
        df,
        x=condition_col,
        y=value_col,
        color=group_col,
        color_discrete_map=GROUP_PALETTE,
        points="all",
        title=title,
        template=get_plotly_template(),
    )
    fig.update_layout(height=480, boxmode="group")
    return fig

def plot_stimulus_alignment_plotly(
    time: np.ndarray,
    bold_mean: np.ndarray,
    events: pd.DataFrame,
    title: str = "BOLD Response Aligned to Music Onsets",
) -> go.Figure:
    """Show event onsets vs average BOLD trace (great for 01 and 02)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=bold_mean, mode="lines", name="Mean BOLD", line=dict(color=HIGHLIGHT)))

    for _, row in events.iterrows():
        if "music" in str(row.get("trial_type", "")).lower():
            color = MUSIC_COLOR
            label = "Music"
        else:
            color = NONMUSIC_COLOR
            label = "Non-music"
        fig.add_vline(
            x=row["onset"],
            line=dict(color=color, width=1.5, dash="dot"),
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="BOLD (z)",
        template=get_plotly_template(),
        height=420,
    )
    return fig

def make_synthetic_bold_dataset(
    n_subjects: int = 20,
    n_timepoints: int = 105,
    tr: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic multi-subject ROI time series with the desired spectral
    signature for the 'Music vs Non-Musical Emotional Auditory Processing in Depression' story.

    Returns long-form DataFrame with columns:
    subject, group, condition, time, bold, trial_type
    """
    rng = np.random.default_rng(seed)
    n_timepoints = int(n_timepoints)
    tr = float(tr)
    fs = 1.0 / tr
    t = np.arange(n_timepoints) * tr

    rows = []
    groups = ["Control"] * (n_subjects // 2) + ["MDD"] * (n_subjects - n_subjects // 2)

    # Block structure similar to real events (tones / negative_music / positive_music)
    conditions = ["nonmusic", "music_neg", "music_pos"]
    block_dur = max(1, int(31.5 / tr))  # ~10-11 samples
    labels = ["tones"] * block_dur + ["negative_music"] * block_dur + ["positive_music"] * block_dur
    # Repeat to fill
    trial_labels = (labels * ((n_timepoints // max(1, len(labels))) + 1))[:n_timepoints]

    for i, grp in enumerate(groups):
        # Base 1/f-like + oscillations
        base = rng.normal(0, 1, n_timepoints).cumsum()
        base = (base - base.mean()) / base.std()

        for cond in conditions:
            signal = base.copy()

            # Condition & group specific spectral modulation (the science story)
            if cond == "music_pos":
                # Positive music: richer harmonics → higher freq power
                boost = 0.9 if grp == "Control" else 0.25   # blunted in MDD
                # Add stronger ~0.04-0.09 Hz content (beta-ish relative to fs~0.33)
                freq = 0.065
                signal += boost * np.sin(2 * np.pi * freq * t + rng.uniform(0, np.pi))
                signal += 0.6 * boost * np.sin(2 * np.pi * 0.12 * t)   # higher "gamma" component

            elif cond == "music_neg":
                boost = 0.55 if grp == "Control" else 0.35
                signal += boost * np.sin(2 * np.pi * 0.04 * t)

            else:  # nonmusic / tones — little group difference
                signal += 0.3 * np.sin(2 * np.pi * 0.025 * t)

            # Physiological noise
            signal += rng.normal(0, 0.6, n_timepoints)

            # Smooth a little
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


def load_real_sample_ts_and_events(max_subjects: int = 2, tasks=("music", "nonmusic")):
    """
    Load a small number of real runs for demos (fast subset of BIDS).
    Returns list of dicts: {subject, group_short, task, run, t, bold_mean, events_df, tr}
    Falls back gracefully.
    """
    import nibabel as nib

    runs_df = inventory_runs()
    avail = runs_df[runs_df["bold_exists"]].copy()

    # Prefer first control + first mdd with music/nonmusic
    samples = []
    for gshort in ["ND", "MDD"]:
        group_runs = avail[avail["group_short"] == ("Control" if gshort=="ND" else "MDD")]
        for task in tasks:
            task_runs = group_runs[group_runs["task"].str.contains(task, case=False, na=False)]
            if len(task_runs) == 0:
                continue
            row = task_runs.iloc[0]
            try:
                img = nib.load(row["bold_path"])
                data = img.get_fdata()
                # Mean time series across brain (fast univariate proxy)
                bold_mean = data.mean(axis=(0,1,2))
                t = np.arange(len(bold_mean)) * (row["tr"] or 3.0)

                ev_path = row["events_path"]
                if ev_path and Path(ev_path).exists():
                    events = parse_events(ev_path)
                else:
                    events = pd.DataFrame()
                samples.append({
                    "subject": row["subject"],
                    "group_short": row["group_short"],
                    "task": row["task"],
                    "run": row["run"],
                    "t": t,
                    "bold_mean": bold_mean,
                    "events": events,
                    "tr": row["tr"] or 3.0,
                    "bold_path": row["bold_path"],
                })
                if len(samples) >= max_subjects * len(tasks):
                    break
            except Exception as e:
                print("Warning loading", row["bold_path"], ":", e)
        if len(samples) >= max_subjects * 2:
            break
    return samples

def compute_welch_psd(
    bold: np.ndarray,
    tr: float = 3.0,
    nperseg: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return freqs, Pxx using Welch. Works on 1D or (time, roi) -> mean over roi."""
    from scipy.signal import welch
    fs = 1.0 / tr
    if bold.ndim > 1:
        bold = bold.mean(axis=1)  # average over parcels / voxels for demo
    f, Pxx = welch(bold, fs=fs, nperseg=min(nperseg, len(bold)//2), detrend="linear")
    return f, Pxx

# Simple helper to extract band power
def band_power(f: np.ndarray, Pxx: np.ndarray, low: float, high: float) -> float:
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))


# ------------------------------------------------------------------
# Matplotlib + Nilearn visualization helpers (for classic + marimo mo.pyplot)
# ------------------------------------------------------------------
def plot_psd_matplotlib(ax, f, Pxx, label, color, linestyle="-"):
    ax.semilogy(f, Pxx, label=label, color=color, linestyle=linestyle, linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)

def plot_stimulus_events_matplotlib(ax, t, bold, events, tr, title="BOLD + Event Alignment"):
    ax.plot(t, bold, color=HIGHLIGHT, linewidth=1.5, label="Mean BOLD")
    for _, ev in events.iterrows():
        onset = ev.get("onset", 0)
        tt = str(ev.get("trial_type", ""))
        if "positive" in tt:
            ax.axvline(onset, color=MUSIC_COLOR, linestyle="--", alpha=0.8, label="pos music" if "pos" not in ax.get_legend_handles_labels()[1] else "")
        elif "negative" in tt:
            ax.axvline(onset, color="#E74C3C", linestyle=":", alpha=0.7)
        elif "tones" in tt or "nonmusic" in tt.lower():
            ax.axvline(onset, color=NONMUSIC_COLOR, linestyle="-.", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_coherence_matplotlib(ax, f, coh, label, color):
    ax.plot(f, coh, label=label, color=color, linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

def plot_nilearn_glass_brain_demo(fig=None, title="Seed-based demo (placeholder)"):
    """Wrapper that produces a nilearn glass brain if possible, else a text fig."""
    if fig is None:
        fig = plt.figure(figsize=(8, 5))
    if NILEARN_VIZ_AVAILABLE:
        # We don't have full stat maps here, so produce a simple mean image plot if possible
        # For demo we just annotate
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Nilearn glass brain / seed map\n(plug in real stat_img from 03/07)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.axis("off")
    else:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "nilearn not fully available for viz", ha="center")
        ax.axis("off")
    fig.suptitle(title)
    return fig

def style_matplotlib_fig(fig, title=None):
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
