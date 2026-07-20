"""Chapter I — Cohort, paradigm, real BOLD (public book)."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="I · Cohort & Design")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy  # noqa: F401
    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        hypothesis_card,
        key_insight_card,
        load_bold_timeseries,
        load_events_summary,
        load_multi_dataset_runs,
        load_participants_df,
        load_raw_participants,
        load_spectral_features,
        multi_dataset_run_ids,
        multi_studies_overview_md,
        as_table,
        set_global_style,
        studies_dataframe,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        hypothesis_card,
        key_insight_card,
        load_bold_timeseries,
        load_events_summary,
        load_multi_dataset_runs,
        load_participants_df,
        load_raw_participants,
        load_spectral_features,
        mo,
        multi_dataset_run_ids,
        multi_studies_overview_md,
        np,
        as_table,
        pd,
        plt,
        studies_dataframe,
    )


@app.cell
def _(data_provenance_md, hypothesis_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
# I · Cohort & Design

### Music, reward, and depression — reading real BOLD against a clinical question

OpenNeuro **ds000171** (Lepping et al.): never-depressed controls and participants with major depressive disorder listening to emotional **music** and **non-musical** auditory material.

Local multi-source inventory (cross-refs, god-mode parquet, Spark Connect consistency):  
`marimo edit marimo_notebooks/00_data_landscape.py` — run that chapter **before** deep QC when you care about the full data range.
"""
            ),
            mo.md(data_provenance_md()),
            mo.md(
                r"""
## Scientific framing

**Anhedonia** is reduced capacity for pleasure. Music is a strong reward cue engaging auditory and limbic circuits. If depression dulls reward *specifically* for structured music (more than for simple tones), differences should be **stimulus-specific**.
"""
            ),
            mo.md(
                hypothesis_card(
                    "Positive music elicits a delayed or blunted BOLD peak in MDD.",
                    "Controls should show earlier, sharper mean BOLD rises after music-related onsets; MDD responses are slower or lower — especially for music, not tones.",
                )
            ),
        ]
    )
    return


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, load_participants_df, mo, plt):
    parts = load_participants_df()
    n = len(parts)
    counts = parts["group_short"].value_counts()
    age_g = parts.groupby("group_short")["age"].agg(["mean", "std", "count"])
    fig_demo, axes_demo = plt.subplots(1, 2, figsize=(10, 4))
    colors = [
        CONTROL_COLOR if _g == "Control" else MDD_COLOR for _g in counts.index
    ]
    axes_demo[0].bar(
        counts.index.astype(str),
        counts.values,
        color=colors,
        edgecolor="white",
    )
    axes_demo[0].set_title(f"Full cohort (n = {n})")
    axes_demo[0].set_ylabel("Participants")
    for _i, _v in enumerate(counts.values):
        axes_demo[0].text(
            _i, _v + 0.3, str(int(_v)), ha="center", fontweight="semibold"
        )
    for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
        _ages = parts.loc[parts.group_short == _g, "age"]
        axes_demo[1].hist(
            _ages, bins=8, alpha=0.55, label=_g, color=_c, edgecolor="white"
        )
    axes_demo[1].set_title("Age distribution")
    axes_demo[1].set_xlabel("Age (years)")
    axes_demo[1].legend(frameon=False)
    fig_demo.tight_layout()
    mo.vstack(
        [
            mo.md(
                "## Who is in the study?\n\nDemographics from the **full published cohort** in `participants.tsv`."
            ),
            fig_demo,
            mo.md("**Age by group**"),
            mo.ui.table(age_g.reset_index().round(2)),
            mo.ui.table(parts.head(12)),
        ]
    )
    return (parts,)


@app.cell
def _(load_events_summary, mo):
    ev = load_events_summary()
    _blocks = [
        mo.md(
            r"""
## The listening paradigm

Each subject completed **music** and **non-music** runs. Event files mark trial onsets — essential so later chapters can test *stimulus specificity*.
"""
        )
    ]
    if ev.empty:
        _blocks.append(mo.md("*No events summary available.*"))
    else:
        by_task = (
            ev.groupby("task")
            .agg(n_files=("subject", "count"), subjects=("subject", "nunique"))
            .reset_index()
        )
        _blocks.extend(
            [
                mo.md(f"**Event files inventoried:** {len(ev)}"),
                mo.ui.table(by_task),
                mo.ui.table(ev.head(8)),
            ]
        )
    mo.vstack(_blocks)
    return (ev,)


@app.cell
def _(load_bold_timeseries, mo):
    ts = load_bold_timeseries()
    subjects = sorted(ts["subject"].unique()) if not ts.empty else []
    sub_ui = mo.ui.dropdown(
        options=subjects or ["(no data)"],
        value=subjects[0] if subjects else "(no data)",
        label="Subject",
    )
    mo.vstack(
        [
            mo.md(
                r"""
## Real BOLD: whole-brain mean traces

For each downloaded NIfTI we compute the spatial mean BOLD per TR (3 s), then z-score within run. Grounded in **actual scans**, not mock oscillators.
"""
            ),
            sub_ui,
        ]
    )
    return sub_ui, ts


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    MUSIC_COLOR,
    key_insight_card,
    mo,
    plt,
    sub_ui,
    ts,
):
    if ts.empty or sub_ui.value == "(no data)":
        _out = mo.md(
            "**No real timeseries.** Run `python scripts/prepare_real_features.py`."
        )
    else:
        _sub = sub_ui.value
        sdf = ts[ts.subject == _sub]
        _group = sdf["group"].iloc[0]
        fig_ts, axes_ts = plt.subplots(2, 1, figsize=(10, 5.5))
        for _ax, _task, _col in zip(
            axes_ts, ["music", "nonmusic"], [MUSIC_COLOR, "#B9770E"]
        ):
            gdf = sdf[sdf.task == _task].sort_values("time")
            if gdf.empty:
                _ax.set_title(f"{_task}: not available")
                continue
            _ax.plot(gdf["time"], gdf["bold_z"], color=_col, lw=1.4)
            _ax.axhline(0, color="#999", lw=0.8)
            _ax.set_ylabel("BOLD (z)")
            _ax.set_title(f"{_sub} · {_group} · {_task}")
            _ax.grid(True, alpha=0.3)
        axes_ts[-1].set_xlabel("Time (s)")
        fig_ts.tight_layout()
        _out = mo.vstack(
            [
                fig_ts,
                mo.md(
                    key_insight_card(
                        "Music and non-music runs are not interchangeable.",
                        "Even whole-brain means show structured fluctuations; Chapter II turns these series into frequency-domain biomarkers.",
                    )
                ),
            ]
        )
    _out
    return


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, load_spectral_features, mo, np, plt):
    feats = load_spectral_features()
    _blocks = [
        mo.md(
            r"""
## Peak latency on real runs

Peri-stimulus windows after music-related onsets; time of maximum mean BOLD.
"""
        )
    ]
    if feats.empty:
        _blocks.append(mo.md("*Spectral features not available.*"))
    else:
        agg = (
            feats.dropna(subset=["peak_latency_s"])
            .groupby(["group", "task"], as_index=False)["peak_latency_s"]
            .mean()
        )
        fig_pk, ax_pk = plt.subplots(figsize=(7.5, 4))
        _tasks = ["music", "nonmusic"]
        _x = np.arange(len(_tasks))
        _w = 0.35
        for _i, (_g, _c) in enumerate(
            [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]
        ):
            _vals = []
            for _t in _tasks:
                _row = agg[(agg.group == _g) & (agg.task == _t)]
                _vals.append(
                    float(_row.peak_latency_s.iloc[0])
                    if len(_row)
                    else float("nan")
                )
            ax_pk.bar(
                _x + (_i - 0.5) * _w,
                _vals,
                _w,
                label=_g,
                color=_c,
                edgecolor="white",
            )
        ax_pk.set_xticks(_x)
        ax_pk.set_xticklabels(_tasks)
        ax_pk.set_ylabel("Mean peak latency (s)")
        ax_pk.set_title("Event-aligned peak latency (processed real runs)")
        ax_pk.legend(frameon=False)
        fig_pk.tight_layout()
        _cols = [
            c
            for c in [
                "subject",
                "group",
                "task",
                "run",
                "peak_latency_s",
                "peak_amp",
                "power_high",
            ]
            if c in feats.columns
        ]
        _blocks.extend(
            [
                fig_pk,
                mo.ui.table(agg.round(2)),
                mo.ui.table(feats[_cols].round(3)),
            ]
        )
    mo.vstack(_blocks)
    return (feats,)


@app.cell
def _(
    load_multi_dataset_runs,
    load_raw_participants,
    mo,
    multi_dataset_run_ids,
    multi_studies_overview_md,
    as_table,
    pd,
    studies_dataframe,
):
    studies = studies_dataframe()
    multi = load_multi_dataset_runs()
    ds_ids = multi_dataset_run_ids()
    blocks = [
        mo.md(
            r"""
## Related OpenNeuro studies (multi-dataset)

The public book claims are powered by **ds000171**. Additional cohorts under
`data/raw/` extend music / affect / multimodal coverage for generalization checks.
"""
        ),
        mo.md(multi_studies_overview_md()),
    ]
    if studies is not None and not getattr(studies, "empty", True):
        cols = [
            c
            for c in (
                "dataset",
                "role",
                "match_level",
                "status",
                "n_subjects_on_disk",
                "n_bold_files",
                "n_nominal",
                "short_title",
                "modality",
                "url",
            )
            if c in studies.columns
        ]
        blocks.append(mo.ui.table(as_table(studies[cols]), selection=None, page_size=12))

    # Show participants meta for every study that has participants.tsv
    part_blocks = []
    for ds in (studies["dataset"].tolist() if studies is not None and not studies.empty else []):
        if ds == "ds000171":
            continue
        p = load_raw_participants(str(ds))
        if p is not None and not p.empty:
            part_blocks.append(
                mo.md(f"**`{ds}` participants.tsv** · n={len(p)}")
            )
            part_blocks.append(mo.ui.table(as_table(p.head(8)), selection=None, page_size=8))
    if part_blocks:
        blocks.append(mo.md("### Cross-ref demographics on disk"))
        blocks.extend(part_blocks)

    if multi is not None and not getattr(multi, "empty", True) and "dataset" in multi.columns:
        n_by = multi.groupby("dataset").size().reset_index(name="n_runs")
        blocks.append(
            mo.md(
                f"### Spectral multi-set coverage  \n"
                f"Runs with extracted PSD features: **{', '.join(f'`{d}`' for d in ds_ids)}**"
            )
        )
        blocks.append(mo.ui.table(as_table(n_by), selection=None))
    mo.vstack(blocks)
    return multi, studies


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## Concepts to carry forward

1. **Stimulus specificity** — always contrast music vs non-music.
2. **Temporal locking** — event files turn continuous BOLD into trial-aware features.
3. **Provenance** — browser builds use the same processed tables extracted from real NIfTI.
4. **Multi-dataset** — cross-refs under `data/raw/` test whether spectral signatures generalize beyond one scanner/cohort.
"""
            ),
            mo.md(
                clinical_relevance_card(
                    "A delayed music-evoked BOLD peak is a candidate correlate of blunted anticipatory pleasure — relevant to music therapy and engagement-aware playlists."
                )
            ),
            mo.md(book_nav("01_pre_flight")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
