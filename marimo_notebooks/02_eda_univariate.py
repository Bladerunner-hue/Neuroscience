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
        hypothesis_card,
        key_insight_card,
        load_condition_features,
        load_spectral_features,
        set_global_style,
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
        hypothesis_card,
        json,
        key_insight_card,
        load_condition_features,
        load_spectral_features,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(data_provenance_md, hypothesis_card, mo):
    mo.md(
        r"""
# II · Univariate Spectral Analysis

### Welch power, band structure, and music valence

Univariate analysis treats **one series / one feature at a time**. We ask:

1. How is spectral energy distributed across frequency bands (real runs)?  
2. How do **positive music**, **negative music**, and **tones** differ in amplitude and high-band power?  
3. Is any group difference **music-specific** (vs non-music / tones)?
"""
    )
    mo.md(data_provenance_md())
    mo.md(
        hypothesis_card(
            "MDD shows reduced high-frequency BOLD power and weaker positive-music lift vs tones.",
            "Controls should show clearer spectral separation between positive music and tones.",
        )
    )
    return


@app.cell
def _(
    CONTROL_COLOR,
    HIGHLIGHT,
    MDD_COLOR,
    json,
    load_spectral_features,
    mo,
    np,
    pd,
    plt,
):
    runs = load_spectral_features()
    mo.md("## 1. Run-level Welch PSDs (real BOLD means)")
    if runs.empty:
        mo.md("*No run-level spectral features.*")
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
                        _f = np.array(json.loads(_row["psd_f"]))
                        _p = np.array(json.loads(_row["psd_pxx"]))
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
        fig_psd
    return (runs,)


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, key_insight_card, mo, np, pd, plt, runs):
    mo.md("## 2. Univariate band features")
    if runs is None or getattr(runs, "empty", True):
        mo.md("*No runs.*")
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
        mo.vstack(
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
    mo.md(
        r"""
## 3. Music valence & domain (trial-type univariate)

This is the clearest answer to *“what music has what effect?”*  
Each bar is mean **epoch BOLD (z)** or **high-band power** for a stimulus class averaged within group.
"""
    )
    if cond.empty:
        mo.md("*No trial-type features.*")
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
        # narrative bullets from data
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
        mo.vstack(
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
    return (cond,)


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, cond, json, mo, np, plt):
    mo.md("## 4. Peri-stimulus waveforms by music valence")
    if cond is None or getattr(cond, "empty", True) or "peri_stim" not in cond.columns:
        mo.md("*No peri-stimulus series stored.*")
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
        fig_w
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## Univariate takeaways

1. **Always stratify by task and trial_type** — “music” is not one thing.  
2. **Positive vs negative music** can diverge in mean BOLD and peak shape.  
3. **Tones** are the within-subject baseline for music effects.  
4. Small BOLD subset ⇒ effect sizes are **directional evidence** for the book narrative; Chapter III adds multivariate ML with confusion matrices.
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
