"""01 — Pre-flight + Event Alignment. Canonical marimo notebook."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "plotly",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="01 — Pre-flight")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd  # required for helpers + tables (must be declared for WASM)
    import matplotlib.pyplot as plt
    import plotly.express as px
    import scipy  # noqa: F401  # ensure scipy is installed under Pyodide

    from helpers import (
        CONTROL_COLOR,
        HIGHLIGHT,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        set_global_style,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        HIGHLIGHT,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        mo,
        np,
        pd,
        plt,
        px,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
# 01 — Pre-flight + Event Alignment

**Chapter 1** · Music vs non-musical emotional auditory processing in depression.

Runs on synthetic BOLD that mirrors the OpenNeuro ds000171 design (Control vs MDD × tones / music).
"""
    )
    return


@app.cell
def _(hypothesis_card, mo):
    mo.md(
        hypothesis_card(
            "Positive music shows delayed peak in MDD — possible reward anticipation deficit.",
            "Controls have sharper earlier BOLD response after music onsets; MDD is blunted/delayed.",
        )
    )
    return


@app.cell
def _(mo):
    n_sub = mo.ui.slider(6, 16, value=8, step=1, label="Synthetic subjects")
    mo.md("## Reactive controls")
    n_sub
    return (n_sub,)


@app.cell
def _(make_synthetic_bold_dataset, mo, n_sub):
    synth = make_synthetic_bold_dataset(int(n_sub.value), n_timepoints=105, tr=3.0)
    summary = (
        synth.groupby(["group", "condition"], as_index=False)
        .agg(n_rows=("bold", "size"), mean_bold=("bold", "mean"))
        .round(3)
    )
    mo.md("## Cohort summary")
    mo.ui.table(summary)
    mo.ui.table(synth.head(8))
    return (synth,)


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    MUSIC_COLOR,
    mo,
    np,
    pd,
    plt,
    px,
    synth,
):
    def mean_trace(group: str, trial: str):
        sub = (
            synth[(synth["group"] == group) & (synth["trial_type"] == trial)]
            .groupby("time", as_index=False)["bold"]
            .mean()
        )
        return sub["time"].values, sub["bold"].values

    t_c, b_c = mean_trace("Control", "positive_music")
    t_m, b_m = mean_trace("MDD", "positive_music")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_c, b_c, color=CONTROL_COLOR, lw=2.5, label="Control · positive music")
    ax.plot(t_m, b_m, color=MDD_COLOR, lw=2.5, ls="--", label="MDD · positive music")
    for _onset, _label in [(0, "tones"), (31.5, "neg music"), (63, "pos music")]:
        color = MUSIC_COLOR if "music" in _label else "#FF9800"
        ax.axvline(_onset, color=color, ls=":", alpha=0.7)
    ax.set_title("Event alignment: mean BOLD during positive music")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BOLD (a.u.)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mo.output.append(fig)

    peak_c = float(t_c[int(np.argmax(b_c))]) if len(b_c) else float("nan")
    peak_m = float(t_m[int(np.argmax(b_m))]) if len(b_m) else float("nan")
    mo.md(f"**Peak latency proxy** — Control: **{peak_c:.1f}s** · MDD: **{peak_m:.1f}s**")

    plot_df = pd.DataFrame(
        {
            "time": list(t_c) + list(t_m),
            "bold": list(b_c) + list(b_m),
            "group": ["Control"] * len(t_c) + ["MDD"] * len(t_m),
        }
    )
    mo.ui.plotly(
        px.line(
            plot_df,
            x="time",
            y="bold",
            color="group",
            color_discrete_map={"Control": CONTROL_COLOR, "MDD": MDD_COLOR},
            title="Interactive: positive-music BOLD traces",
        )
    )
    return


@app.cell
def _(HIGHLIGHT, MUSIC_COLOR, mo, plt, synth):
    subj = sorted(synth["subject"].unique())[0]
    s = synth[(synth["subject"] == subj) & (synth["condition"] == "music")].sort_values(
        "time"
    )
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(s["time"], s["bold"], color=HIGHLIGHT, lw=1.8)
    for _t0 in (0, 31.5, 63, 94.5):
        ax2.axvline(_t0, color=MUSIC_COLOR, ls="--", alpha=0.55)
    ax2.set_title(f"Single-subject music run: {subj}")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    mo.output.append(fig2)
    return


@app.cell
def _(book_nav, clinical_relevance_card, key_insight_card, mo):
    mo.md(
        key_insight_card(
            "Positive music shows delayed / blunted peak in MDD.",
            "Temporal misalignment seeds spectral analysis in chapter 02.",
            "Controls: earlier, higher-amplitude response.",
        )
    )
    mo.md(
        clinical_relevance_card(
            "Event alignment reveals reward-processing dynamics used downstream as spectral biomarker features."
        )
    )
    mo.md(book_nav("01_pre_flight"))
    return


if __name__ == "__main__":
    app.run()
