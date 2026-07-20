"""Chapter II — Univariate spectral analysis + music valence visuals."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="II · Spectral Power")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    from helpers import (
        CONTROL_COLOR,
        HIGHLIGHT,
        MDD_COLOR,
        MUSIC_COLOR,
        NONMUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        filter_clean_runs,
        hypothesis_card,
        key_insight_card,
        load_cleaned_spectral_features,
        load_condition_features,
        load_multi_dataset_runs,
        load_spectral_features,
        multi_dataset_run_ids,
        pandas_to_polars,
        set_global_style,
        studies_dataframe,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        HIGHLIGHT,
        MDD_COLOR,
        MUSIC_COLOR,
        NONMUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        filter_clean_runs,
        hypothesis_card,
        json,
        key_insight_card,
        load_cleaned_spectral_features,
        load_condition_features,
        load_multi_dataset_runs,
        load_spectral_features,
        mo,
        multi_dataset_run_ids,
        np,
        pandas_to_polars,
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
# II · Univariate Spectral Analysis

### Multitaper/Welch power, band structure, and music valence

Univariate analysis treats **one series / one feature at a time**. We ask:

1. How is spectral energy distributed across frequency bands (QC-cleaned runs)?  
2. How do **positive music**, **negative music**, and **tones** differ in amplitude and high-band power?  
3. Is any group difference **music-specific** (vs non-music / tones)?

Run-level PSDs use **adaptive multitaper** when the feature store is regenerated
(`prepare_real_features.py --psd adaptive`); Ch 0 compares Welch / uniform / adaptive live.
"""
            ),
            mo.md(data_provenance_md()),
            mo.md(
                hypothesis_card(
                    "MDD shows reduced high-frequency BOLD power and weaker positive-music lift vs tones.",
                    "Controls should show clearer spectral separation between positive music and tones.",
                )
            ),
        ]
    )
    return


@app.cell
def _(
    CONTROL_COLOR,
    HIGHLIGHT,
    MDD_COLOR,
    json,
    filter_clean_runs,
    load_cleaned_spectral_features,
    load_spectral_features,
    mo,
    np,
    pd,
    plt,
):
    # Prefer QC-cleaned runs for published spectral claims; fall back to full store
    _all = load_spectral_features()
    _clean = load_cleaned_spectral_features()
    runs = _clean if _clean is not None and not getattr(_clean, "empty", True) else _all
    if runs is _all and _all is not None and not getattr(_all, "empty", True):
        runs = filter_clean_runs(_all, drop_outliers=True)
    _blocks = [
        mo.md(
            f"## 1. Run-level PSDs (QC-cleaned · n={len(runs)}/{len(_all) if _all is not None else 0})  \n"
            "Default feature store uses **adaptive multitaper** when regenerated via "
            "`prepare_real_features.py --psd adaptive`."
        )
    ]
    if runs.empty:
        _blocks.append(mo.md("*No run-level spectral features.*"))
    else:
        fig_psd, axes_psd = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for _ax, _task in zip(axes_psd, ["music", "nonmusic"]):
            for _g, _c, _ls in [
                ("Control", CONTROL_COLOR, "-"),
                ("MDD", MDD_COLOR, "--"),
            ]:
                sub = runs[(runs.group == _g) & (runs.task == _task)]
                _curves, _f_ref = [], None
                for _, _row in sub.iterrows():
                    if "psd_f" not in _row or pd.isna(_row.get("psd_f")):
                        continue
                    try:
                        _raw_f, _raw_p = _row["psd_f"], _row["psd_pxx"]
                        _f = np.array(
                            json.loads(_raw_f) if isinstance(_raw_f, str) else _raw_f
                        )
                        _p = np.array(
                            json.loads(_raw_p) if isinstance(_raw_p, str) else _raw_p
                        )
                    except Exception:
                        continue
                    if _f_ref is None:
                        _f_ref = _f
                    _curves.append(
                        _p if len(_f) == len(_f_ref) else np.interp(_f_ref, _f, _p)
                    )
                if _curves and _f_ref is not None:
                    _ax.semilogy(
                        _f_ref,
                        np.mean(np.stack(_curves), axis=0),
                        color=_c,
                        ls=_ls,
                        lw=2.2,
                        label=_g,
                    )
            _ax.axvspan(0.08, 0.15, color=HIGHLIGHT, alpha=0.12, label="high band")
            _ax.set_title(f"Task: {_task}")
            _ax.set_xlabel("Frequency (Hz)")
            _ax.grid(True, which="both", alpha=0.3)
            _ax.legend(frameon=False, fontsize=8)
        axes_psd[0].set_ylabel("Power (log)")
        fig_psd.suptitle("Mean Welch PSD by group × task (processed real runs)", y=1.02)
        fig_psd.tight_layout()
        _blocks.append(fig_psd)
    mo.vstack(_blocks)
    return (runs,)


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, key_insight_card, mo, np, pd, plt, runs):
    _blocks = [mo.md("## 2. Univariate band features")]
    if runs is None or getattr(runs, "empty", True):
        _blocks.append(mo.md("*No runs.*"))
    else:
        agg = runs.groupby(["group", "task"], as_index=False)[
            ["power_low", "power_mid", "power_high", "spectral_centroid", "peak_amp"]
        ].mean()
        fig_b, axes_b = plt.subplots(1, 3, figsize=(12, 3.8))
        _tasks = ["music", "nonmusic"]
        _x = np.arange(2)
        for _ax, _metric, _title in zip(
            axes_b,
            ["power_high", "spectral_centroid", "peak_amp"],
            ["High-band fraction", "Centroid (Hz)", "Event peak amp"],
        ):
            for _i, (_g, _c) in enumerate(
                [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]
            ):
                _vals = []
                for _t in _tasks:
                    _row = agg[(agg.group == _g) & (agg.task == _t)]
                    _vals.append(
                        float(_row[_metric].iloc[0]) if len(_row) else np.nan
                    )
                _ax.bar(
                    _x + (_i - 0.5) * 0.35,
                    _vals,
                    0.35,
                    label=_g,
                    color=_c,
                    edgecolor="white",
                )
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_tasks)
            _ax.set_title(_title)
            _ax.legend(frameon=False, fontsize=8)
            _ax.grid(True, axis="y", alpha=0.3)
        fig_b.tight_layout()

        def _m(_g, _t, _col="power_high"):
            _row = agg[(agg.group == _g) & (agg.task == _t)]
            return float(_row[_col].iloc[0]) if len(_row) else np.nan

        ratio = _m("Control", "music") / max(_m("MDD", "music"), 1e-9)
        _blocks.extend(
            [
                fig_b,
                mo.ui.table(agg.round(4)),
                mo.md(
                    key_insight_card(
                        "Compare music and non-music before claiming a depression effect.",
                        f"High-band (music): Control {_m('Control','music'):.3f} vs MDD {_m('MDD','music'):.3f}; "
                        f"non-music: {_m('Control','nonmusic'):.3f} vs {_m('MDD','nonmusic'):.3f}.",
                        effect_size=f"Control/MDD high-band (music) ≈ {ratio:.2f}×",
                    )
                ),
            ]
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    load_condition_features,
    mo,
    np,
    pd,
    plt,
    key_insight_card,
):
    cond = load_condition_features()
    _blocks = [
        mo.md(
            r"""
## 3. Music valence & domain (trial-type univariate)

This is the clearest answer to *“what music has what effect?”*  
Each bar is mean **epoch BOLD (z)** or **high-band power** for a stimulus class averaged within group.
"""
        )
    ]
    if cond.empty:
        _blocks.append(mo.md("*No trial-type features.*"))
    else:
        order = [
            "positive_music",
            "negative_music",
            "tones",
            "positive_nonmusic",
            "negative_nonmusic",
        ]
        stims = [s for s in order if s in set(cond.trial_type)]
        fig_v, axes_v = plt.subplots(1, 2, figsize=(11, 4.2))
        _x = np.arange(len(stims))
        for _ax, _col, _title in zip(
            axes_v,
            ["mean_bold", "power_high"],
            ["Mean epoch BOLD (z)", "High-band power in epoch"],
        ):
            for _i, (_g, _c) in enumerate(
                [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]
            ):
                _vals = []
                for _s in stims:
                    _sub = cond[(cond.group == _g) & (cond.trial_type == _s)]
                    _vals.append(
                        float(_sub[_col].mean()) if len(_sub) else np.nan
                    )
                _ax.bar(
                    _x + (_i - 0.5) * 0.35,
                    _vals,
                    0.35,
                    label=_g,
                    color=_c,
                    edgecolor="white",
                )
            _ax.axhline(0, color="#999", lw=0.7)
            _ax.set_xticks(_x)
            _ax.set_xticklabels([s.replace("_", "\n") for s in stims], fontsize=8)
            _ax.set_title(_title)
            _ax.legend(frameon=False, fontsize=8)
            _ax.grid(True, axis="y", alpha=0.3)
        fig_v.tight_layout()

        pivot = (
            cond.groupby(["group", "trial_type"])[["mean_bold", "peak_amp", "power_high"]]
            .mean()
            .round(3)
            .reset_index()
        )

        def gm(g, t, c="mean_bold"):
            s = cond[(cond.group == g) & (cond.trial_type == t)]
            return float(s[c].mean()) if len(s) else np.nan

        bullets = mo.md(
            f"""
### Reading the music map

| Stimulus | Control mean BOLD | MDD mean BOLD | Note |
|---|---:|---:|---|
| Positive music | {gm('Control','positive_music'):+.3f} | {gm('MDD','positive_music'):+.3f} | Target “rewarding music” condition |
| Negative music | {gm('Control','negative_music'):+.3f} | {gm('MDD','negative_music'):+.3f} | Negative musical valence |
| Tones | {gm('Control','tones'):+.3f} | {gm('MDD','tones'):+.3f} | Neutral auditory baseline |
| Positive non-music | {gm('Control','positive_nonmusic'):+.3f} | {gm('MDD','positive_nonmusic'):+.3f} | Valence-matched non-music control |
| Negative non-music | {gm('Control','negative_nonmusic'):+.3f} | {gm('MDD','negative_nonmusic'):+.3f} | Valence-matched non-music control |

**Effect of positive music vs tones (mean BOLD):**  
Control Δ = {gm('Control','positive_music')-gm('Control','tones'):+.3f},  
MDD Δ = {gm('MDD','positive_music')-gm('MDD','tones'):+.3f}.
"""
        )
        _blocks.extend(
            [
                fig_v,
                mo.ui.table(pivot),
                bullets,
                mo.md(
                    key_insight_card(
                        "Positive music is not the same intervention as “any music”.",
                        "Separating positive vs negative music and music vs non-music shows distinct BOLD footprints. "
                        "Use contrasts (pos music − tones) in ML so the model learns *music effects*, not session baselines.",
                    )
                ),
            ]
        )
    mo.vstack(_blocks)
    return (cond,)


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, cond, json, mo, np, plt):
    _blocks = [mo.md("## 4. Peri-stimulus waveforms by music valence")]
    if cond is None or getattr(cond, "empty", True) or "peri_stim" not in cond.columns:
        _blocks.append(mo.md("*No peri-stimulus series stored.*"))
    else:
        fig_w, axes_w = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for _ax, _trial, _title in zip(
            axes_w,
            ["positive_music", "negative_music"],
            ["Positive music", "Negative music"],
        ):
            for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
                _peri_curves = []
                for _, _row in cond[
                    (cond.group == _g) & (cond.trial_type == _trial)
                ].iterrows():
                    try:
                        _peri = (
                            json.loads(_row["peri_stim"])
                            if isinstance(_row["peri_stim"], str)
                            else _row["peri_stim"]
                        )
                    except Exception:
                        _peri = []
                    if _peri:
                        _peri_curves.append(np.asarray(_peri, dtype=float))
                if not _peri_curves:
                    continue
                _L = int(np.median([len(c) for c in _peri_curves]))
                _stacked = np.stack(
                    [
                        np.interp(
                            np.linspace(0, 1, _L), np.linspace(0, 1, len(c)), c
                        )
                        for c in _peri_curves
                    ]
                )
                _t = np.arange(_L) * 3.0
                _mu = _stacked.mean(0)
                _se = _stacked.std(0) / np.sqrt(len(_stacked))
                _ax.plot(_t, _mu, color=_c, lw=2, label=_g)
                _ax.fill_between(_t, _mu - _se, _mu + _se, color=_c, alpha=0.2)
            _ax.set_title(_title)
            _ax.set_xlabel("Time from onset (s)")
            _ax.axhline(0, color="#999", lw=0.7)
            _ax.grid(True, alpha=0.3)
            _ax.legend(frameon=False, fontsize=8)
        axes_w[0].set_ylabel("BOLD (z)")
        fig_w.suptitle("Mean peri-stimulus ± SEM (real epochs)", y=1.02)
        fig_w.tight_layout()
        _blocks.append(fig_w)
    mo.vstack(_blocks)
    return


@app.cell
def _(
    CONTROL_COLOR,
    HIGHLIGHT,
    MDD_COLOR,
    MUSIC_COLOR,
    load_multi_dataset_runs,
    mo,
    multi_dataset_run_ids,
    pd,
    pandas_to_polars,
    plt,
    studies_dataframe,
):
    multi = load_multi_dataset_runs()
    studies = studies_dataframe()
    ds_ids = multi_dataset_run_ids()
    blocks = [
        mo.md(
            f"""
## Cross-dataset spectral compare

Beyond primary ds000171, multi-set runs (god path / summary) currently include:
**{', '.join(f'`{d}`' for d in ds_ids)}**.

Cross-refs do **not** replace the clinical contrast — they check whether high-band /
tSNR structure is in a comparable range after schema alignment.
"""
        )
    ]
    if studies is not None and not getattr(studies, "empty", True):
        sc = [
            c
            for c in ("dataset", "role", "status", "n_bold_files", "short_title", "match_level")
            if c in studies.columns
        ]
        blocks.append(mo.ui.table(pandas_to_polars(studies[sc]), selection=None, page_size=10))

    if multi is not None and not getattr(multi, "empty", True) and "dataset" in multi.columns:
        metric_cols = [c for c in ("power_high", "power_mid", "power_low", "tsnr", "spectral_centroid") if c in multi.columns]
        if metric_cols:
            by_ds = multi.groupby("dataset", as_index=False)[metric_cols].mean(numeric_only=True)
            by_ds["n_runs"] = multi.groupby("dataset").size().values
            blocks.append(mo.md("### Band / QC means by dataset"))
            blocks.append(mo.ui.table(pandas_to_polars(by_ds.round(4)), selection=None))

        if "power_high" in multi.columns:
            fig_x, axes_x = plt.subplots(1, 2, figsize=(10, 3.8))
            ds_list = sorted(multi["dataset"].astype(str).unique())
            colors = [CONTROL_COLOR, MUSIC_COLOR, MDD_COLOR, HIGHLIGHT]
            for ax, col, title in zip(
                axes_x,
                ["power_high", "tsnr"] if "tsnr" in multi.columns else ["power_high", "power_high"],
                ["power_high by dataset", "tSNR by dataset"],
            ):
                if col not in multi.columns:
                    continue
                data, labels = [], []
                for ds in ds_list:
                    v = pd.to_numeric(
                        multi.loc[multi["dataset"].astype(str) == ds, col], errors="coerce"
                    ).dropna()
                    if len(v):
                        data.append(v.values)
                        labels.append(ds)
                if data:
                    bp = ax.boxplot(data, labels=labels, patch_artist=True)
                    for i, patch in enumerate(bp["boxes"]):
                        patch.set_facecolor(colors[i % len(colors)])
                        patch.set_alpha(0.65)
                    ax.set_title(title)
                    ax.tick_params(axis="x", labelrotation=20)
                    ax.grid(True, axis="y", alpha=0.3)
            fig_x.tight_layout()
            blocks.append(fig_x)

        show = [
            c
            for c in (
                "dataset",
                "subject",
                "group",
                "task",
                "run",
                "power_high",
                "tsnr",
                "spectral_centroid",
            )
            if c in multi.columns
        ]
        blocks.append(mo.md("### Multi-set run sample"))
        blocks.append(
            mo.ui.table(
                pandas_to_polars(multi[show].head(30).round(4) if show else multi.head(30)),
                selection=None,
                page_size=12,
            )
        )
    else:
        blocks.append(
            mo.md(
                "*No multi-set spectral table — pre-ingest cross-ref BOLD "
                "(`pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725`).*"
            )
        )
    mo.vstack(blocks)
    return multi, studies


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## Univariate takeaways

1. **Always stratify by task and trial_type** — “music” is not one thing.  
2. **Positive vs negative music** can diverge in mean BOLD and peak shape.  
3. **Tones** are the within-subject baseline for music effects (`pos music − tones`).  
4. Peri-stimulus waveforms show **when** engagement peaks after onset.  
5. Expanded BOLD subset + Chapter III **algorithm bake-off** turn these univariate maps into ranked, explainable models for RecSys priors.  
6. **Cross-dataset** rows (god multi-set) check that spectral scales transfer to healthy / multimodal music cohorts.
"""
            ),
            mo.md(
                clinical_relevance_card(
                    "If positive music fails to elevate BOLD relative to tones in a patient, a recommender should not treat “music” as a uniform therapy — valence and spectral content must be personalised."
                )
            ),
            mo.md(book_nav("02_eda_univariate")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
