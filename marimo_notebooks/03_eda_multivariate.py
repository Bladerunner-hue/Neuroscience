"""Chapter III — Coherence (public book)."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="III · Network Timing")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import coherence
    from helpers import (
        CONTROL_COLOR, MDD_COLOR, book_nav, clinical_relevance_card,
        data_provenance_md, hypothesis_card, key_insight_card,
        load_bold_timeseries, set_global_style, trapz_integral,
    )
    set_global_style()
    return (
        CONTROL_COLOR, MDD_COLOR, book_nav, clinical_relevance_card,
        coherence, data_provenance_md, hypothesis_card, key_insight_card,
        load_bold_timeseries, mo, np, pd, plt, trapz_integral,
    )


@app.cell
def _(data_provenance_md, hypothesis_card, mo):
    mo.md(r"""
# III · Network Timing

### Coherence as coupling language

\[
C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f)\,P_{yy}(f)}
\]

With whole-brain means we use an early vs late half proxy — a transparent demonstration on **real** mean BOLD, not a mock network.
""")
    mo.md(data_provenance_md())
    mo.md(hypothesis_card(
        "Music boosts coupling in controls; MDD lacks that boost.",
        "Band coherence (0.03–0.10 Hz) should rise for music in controls relative to non-music and MDD.",
    ))
    return


@app.cell
def _(
    CONTROL_COLOR, MDD_COLOR, coherence, key_insight_card,
    load_bold_timeseries, mo, np, pd, plt, trapz_integral,
):
    ts = load_bold_timeseries()
    mo.md("## Coherence from real runs")

    def _split(series):
        n = len(series)
        a, b = series[: n // 2], series[n // 2 : 2 * (n // 2)]
        m = min(len(a), len(b))
        return a[:m], b[:m]

    if ts.empty:
        mo.md("*No timeseries.*")
    else:
        records = []
        fig_c, axes_c = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for _ax, _task in zip(axes_c, ["music", "nonmusic"]):
            for _g, _c, _ls in [("Control", CONTROL_COLOR, "-"), ("MDD", MDD_COLOR, "--")]:
                sub = ts[(ts.group == _g) & (ts.task == _task)]
                curves, f_ref = [], None
                for (_sid, _run), sg in sub.groupby(["subject", "run"]):
                    sig = sg.sort_values("time")["bold_z"].values
                    if len(sig) < 20:
                        continue
                    x, y = _split(sig)
                    f, cxy = coherence(x, y, fs=1/3.0, nperseg=min(28, max(8, len(x)//2)))
                    if f_ref is None:
                        f_ref = f
                    curves.append(np.interp(f_ref, f, cxy))
                    mask = (f_ref > 0.03) & (f_ref < 0.10)
                    band = trapz_integral(np.interp(f_ref, f, cxy)[mask], f_ref[mask]) if np.any(mask) else 0.0
                    records.append({"subject": _sid, "run": _run, "group": _g, "task": _task, "band_coh": band})
                if curves and f_ref is not None:
                    _ax.plot(f_ref, np.mean(curves, axis=0), color=_c, ls=_ls, lw=2.2, label=_g)
            _ax.axvspan(0.03, 0.10, color="#6C3483", alpha=0.1)
            _ax.set_title(f"Task: {_task}")
            _ax.set_xlabel("Frequency (Hz)")
            _ax.grid(True, alpha=0.3)
            _ax.legend(frameon=False)
        axes_c[0].set_ylabel("Coherence")
        fig_c.suptitle("Mean coherence (early vs late proxy)", y=1.02)
        fig_c.tight_layout()
        cdf = pd.DataFrame(records)
        agg = cdf.groupby(["group", "task"], as_index=False)["band_coh"].mean() if len(cdf) else pd.DataFrame()
        fig_b2, ax_b2 = plt.subplots(figsize=(7, 3.8))
        if len(agg):
            _x = np.arange(2)
            for _i, (_g, _c) in enumerate([("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]):
                _vals = [float(agg[(agg.group==_g)&(agg.task==_t)].band_coh.iloc[0]) if len(agg[(agg.group==_g)&(agg.task==_t)]) else np.nan for _t in ["music","nonmusic"]]
                ax_b2.bar(_x + (_i-0.5)*0.35, _vals, 0.35, label=_g, color=_c, edgecolor="white")
            ax_b2.set_xticks(_x)
            ax_b2.set_xticklabels(["music", "nonmusic"])
            ax_b2.set_ylabel("Band coherence")
            ax_b2.set_title("Integrated coherence by group × task")
            ax_b2.legend(frameon=False)
            ax_b2.grid(True, axis="y", alpha=0.3)
        mo.vstack([
            fig_c, fig_b2,
            mo.ui.table(agg.round(4)) if len(agg) else mo.md("*No rows*"),
            mo.md(key_insight_card(
                "Music vs non-music changes coupling structure.",
                "Early/late proxy on real mean BOLD — methodological demonstration, not ROI connectivity.",
            )),
        ])
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack([
        mo.md(r"""
## Limits

- Whole-brain means collapse space.
- Coherence assumes approximate linearity.
- Subset size is small; treat as directional evidence.
"""),
        mo.md(clinical_relevance_card(
            "If music fails to engage limbic valuation circuits, adaptive music medicine needs different spectral priors."
        )),
        mo.md(book_nav("03_eda_multivariate")),
    ])
    return


if __name__ == "__main__":
    app.run()
