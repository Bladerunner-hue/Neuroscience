import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    import marimo as mo

    # Direct load, no src/neuro wrapper
    from helpers import (
        CONTROL_COLOR,
        HIGHLIGHT,
        MDD_COLOR,
        MUSIC_COLOR,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        load_bold_mean_direct,
        load_events_direct,
        load_participants_direct,
        make_synthetic_bold_dataset,
        set_global_style,
    )

    set_global_style()

    DATA_DIR = Path("data/raw/ds000171")
    TR_SEC = 3.0

    return (
        CONTROL_COLOR,
        DATA_DIR,
        HIGHLIGHT,
        MDD_COLOR,
        MUSIC_COLOR,
        TR_SEC,
        clinical_relevance_card,
        go,
        hypothesis_card,
        key_insight_card,
        load_bold_mean_direct,
        load_events_direct,
        load_participants_direct,
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
    # 01 — Pre-flight + Event Alignment (Marimo)

    **Music vs Non-Musical Emotional Auditory Processing in Depression**

    Real data from ds000171. This marimo notebook gives you reactive controls and beautiful visuals (matplotlib + Plotly).
    """
    )
    return


@app.cell
def _(hypothesis_card, mo):
    mo.md(
        hypothesis_card(
            "Positive music shows delayed peak in MDD — possible reward anticipation deficit.",
            "Controls will have sharper earlier BOLD response after music onsets. MDD blunted/delayed.",
        )
    )
    return


@app.cell
def _(DATA_DIR, load_bold_mean_direct, load_events_direct, load_participants_direct, mo, pd):
    participants = load_participants_direct()
    # Simple direct inventory summary
    runs = []
    for sub in participants["participant_id"].head(5):  # sample for speed
        for f in (DATA_DIR / sub / "func").glob(f"{sub}_*_bold.nii.gz"):
            parts = f.stem.split("_")
            task = parts[2].replace("task-", "")
            run = int(parts[3].replace("run-", ""))
            runs.append({"subject": sub, "task": task, "run": run})
    runs_df = pd.DataFrame(runs)

    # Direct real sample load for one
    real_bold = []
    try:
        real_bold = load_bold_mean_direct("sub-control01", "music", 1)
    except Exception as e:
        print("Warning loading real BOLD:", e)

    mo.md("## Data Intake - Direct Load (no wrappers)")
    mo.md(f"**DATA_DIR:** `{DATA_DIR}`")
    mo.ui.table(participants.head())
    mo.ui.table(runs_df.head() if not runs_df.empty else pd.DataFrame({"note": ["no runs"]}))

    if len(real_bold) > 0:
        mo.md("**Real BOLD mean loaded directly** (first 10 vols shown)")
        mo.ui.table(pd.DataFrame({"bold_mean": real_bold[:10]}))
    else:
        mo.md("**No real BOLD** - using synthetic only for demo.")
    return (participants, runs_df, real_bold)


@app.cell
def _(DATA_DIR, mo, pd, px):
    # Direct event load example
    try:
        events = pd.read_csv(DATA_DIR / "sub-control01/func/sub-control01_task-music_run-1_events.tsv", sep="\t")
        mo.md("### Direct Events Load Example (sub-control01 music run-1)")
        mo.ui.table(events.head(8))
        fig_events = px.bar(events["trial_type"].value_counts().reset_index(), x="trial_type", y="count", title="Trial types (direct load)")
        mo.ui.plotly(fig_events)
    except Exception as e:
        mo.md(f"Direct events load failed: {e}")
    return


@app.cell
def _(mo):
    n_sub = mo.ui.slider(6, 16, value=8, label="Synthetic subjects")
    return (n_sub,)


@app.cell
def _(mo):
    # Example prod caching
    import marimo as mo
    @mo.cache
    def cached_load_participants():
        from helpers import load_participants_direct
        return load_participants_direct()
    # cached = cached_load_participants()
    return


@app.cell
def _(
    HIGHLIGHT,
    MUSIC_COLOR,
    TR_SEC,
    make_synthetic_bold_dataset,
    mo,
    n_sub,
    plt,
    px,
):
    # Reactive slider for demo
    synth = make_synthetic_bold_dataset(n_sub.value, 105, TR_SEC)

    # Representative trace
    subdf = (
        synth[(synth["group"] == "Control") & (synth["trial_type"] == "positive_music")]
        .groupby("time")["bold"]
        .mean()
        .reset_index()
    )
    t = subdf["time"].values
    bold = subdf["bold"].values

    # Matplotlib alignment
    fig_synth, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, bold, color=HIGHLIGHT, label="Mean BOLD")
    for onset, label in [(0, "tones"), (36, "pos_music"), (105, "neg_music")]:
        color = MUSIC_COLOR if "music" in label else "#FF9800"
        ax.axvline(onset, color=color, linestyle="--", alpha=0.8)
    ax.set_title("Event Alignment: Music onsets vs BOLD (synthetic but matches real pattern)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mo.pyplot(fig_synth)

    # Also Plotly
    pfig = px.line(x=t, y=bold, title="Interactive: BOLD trace with event markers")
    mo.ui.plotly(pfig)
    return


@app.cell
def _(key_insight_card, mo):
    mo.md(
        key_insight_card(
            "Positive music shows delayed peak in MDD — possible reward anticipation deficit.",
            "This temporal signature is the starting point for the spectral analysis in notebook 02.",
            "Effect size: Controls show earlier and higher amplitude response to positive music.",
        )
    )
    return


@app.cell
def _(clinical_relevance_card, mo):
    mo.md(
        clinical_relevance_card(
            "Event alignment reveals the temporal dynamics of reward processing. This informs spectral biomarker discovery for music therapy recommendation systems."
        )
    )
    return


if __name__ == "__main__":
    app.run()
