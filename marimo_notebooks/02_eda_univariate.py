"""Chapter II — Spectral power on real BOLD (public book)."""

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
        CONTROL_COLOR, HIGHLIGHT, MDD_COLOR, book_nav,
        clinical_relevance_card, data_provenance_md, hypothesis_card,
        key_insight_card, load_spectral_features, set_global_style,
    )
    set_global_style()
    return (
        CONTROL_COLOR, HIGHLIGHT, MDD_COLOR, book_nav,
        clinical_relevance_card, data_provenance_md, hypothesis_card,
        json, key_insight_card, load_spectral_features, mo, np, pd, plt,
    )


@app.cell
def _(data_provenance_md, hypothesis_card, mo):
    mo.md(r"""
# II · Spectral Power

### Welch PSD as a language for rhythmic BOLD energy

The power spectral density asks: *how much energy lives in each frequency of the BOLD fluctuation?* Welch’s method segments, windows, and averages periodograms for stable short-run estimates.
""")
    mo.md(data_provenance_md())
    mo.md(hypothesis_card(
        "MDD shows reduced high-frequency BOLD power during music.",
        "Controls should carry more mid/high-band power during music; non-music should shrink the group gap.",
    ))
    mo.md(r"""
## Method

| Step | Choice |
|------|--------|
| TR | 3 s → fs ≈ 0.33 Hz |
| Estimator | `scipy.signal.welch` |
| Bands | low 0.01–0.04 · mid 0.04–0.08 · high 0.08–0.15 Hz |
| Features | band fractions + spectral centroid |

All PSDs below come from **real whole-brain mean BOLD**.
""")
    return


@app.cell
def _(CONTROL_COLOR, HIGHLIGHT, MDD_COLOR, json, load_spectral_features, mo, np, plt):
    feats = load_spectral_features()
    mo.md("## Real PSDs by group and task")
    if feats.empty:
        mo.md("*No spectral features.*")
    else:
        fig_psd, axes_psd = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for _ax, _task in zip(axes_psd, ["music", "nonmusic"]):
            for _g, _c, _ls in [("Control", CONTROL_COLOR, "-"), ("MDD", MDD_COLOR, "--")]:
                sub = feats[(feats.group == _g) & (feats.task == _task)]
                curves = []
                f_ref = None
                for _, _row in sub.iterrows():
                    try:
                        _f = np.array(json.loads(_row["psd_f"]))
                        _p = np.array(json.loads(_row["psd_pxx"]))
                    except Exception:
                        continue
                    if f_ref is None:
                        f_ref = _f
                    curves.append(_p if len(_f) == len(f_ref) else np.interp(f_ref, _f, _p))
                if curves and f_ref is not None:
                    _ax.semilogy(f_ref, np.mean(curves, axis=0), color=_c, ls=_ls, lw=2.2, label=_g)
            _ax.axvspan(0.08, 0.15, color=HIGHLIGHT, alpha=0.12)
            _ax.set_title(f"Task: {_task}")
            _ax.set_xlabel("Frequency (Hz)")
            _ax.grid(True, which="both", alpha=0.3)
            _ax.legend(frameon=False, fontsize=9)
        axes_psd[0].set_ylabel("Power (Welch, log)")
        fig_psd.suptitle("Mean PSD across processed real runs", y=1.02)
        fig_psd.tight_layout()
        fig_psd
    return (feats,)


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, feats, key_insight_card, mo, np, plt):
    mo.md("## Band summaries on real runs")
    if feats is None or getattr(feats, "empty", True):
        mo.md("*No features.*")
    else:
        agg = feats.groupby(["group", "task"], as_index=False)[
            ["power_low", "power_mid", "power_high", "spectral_centroid"]
        ].mean()
        fig_b, axes_b = plt.subplots(1, 2, figsize=(10, 4))
        _tasks = ["music", "nonmusic"]
        _x = np.arange(2)
        _w = 0.35
        for _ax, _metric, _title in zip(
            axes_b, ["power_high", "spectral_centroid"],
            ["High-band power fraction", "Spectral centroid (Hz)"],
        ):
            for _i, (_g, _c) in enumerate([("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]):
                _vals = []
                for _t in _tasks:
                    _row = agg[(agg.group == _g) & (agg.task == _t)]
                    _vals.append(float(_row[_metric].iloc[0]) if len(_row) else np.nan)
                _ax.bar(_x + (_i - 0.5) * _w, _vals, _w, label=_g, color=_c, edgecolor="white")
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_tasks)
            _ax.set_title(_title)
            _ax.legend(frameon=False, fontsize=9)
            _ax.grid(True, axis="y", alpha=0.3)
        fig_b.tight_layout()

        def _mm(_g, _t, _m="power_high"):
            _row = agg[(agg.group == _g) & (agg.task == _t)]
            return float(_row[_m].iloc[0]) if len(_row) else np.nan

        c_m, d_m = _mm("Control", "music"), _mm("MDD", "music")
        c_n, d_n = _mm("Control", "nonmusic"), _mm("MDD", "nonmusic")
        ratio = c_m / d_m if d_m and d_m > 0 else np.nan
        mo.vstack([
            fig_b,
            mo.ui.table(agg.round(4)),
            mo.md(key_insight_card(
                "Music separates groups more than non-music in this subset.",
                f"High-band — music: Control {c_m:.3f} vs MDD {d_m:.3f}; non-music: {c_n:.3f} vs {d_n:.3f}.",
                effect_size=f"Control/MDD high-band (music) ≈ {ratio:.2f}×" if ratio == ratio else "n/a",
            )),
        ])
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack([
        mo.md(r"""
## How to read a PSD

- Not EEG “brain waves” — BOLD is sluggish.
- Relative bands matter more than absolute power.
- Stimulus specificity is the scientific hinge.
- This is a processed subset; treat effect sizes as directional.
"""),
        mo.md(clinical_relevance_card(
            "A music-selective drop in high-frequency BOLD energy is a candidate digital biomarker of anhedonia."
        )),
        mo.md(book_nav("02_eda_univariate")),
    ])
    return


if __name__ == "__main__":
    app.run()
