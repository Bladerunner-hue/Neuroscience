"""Chapter 0 — QC dashboard: tSNR, spectral quality, IsolationForest, cleaned features.

Reactive marimo + Polars + @mo.cache. Public WASM chapter (no TensorFlow).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "polars", "matplotlib", "scikit-learn"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="0 · QC Dashboard")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        key_insight_card,
        load_book_bundle,
        load_spectral_features,
        set_global_style,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        IsolationForest,
        MDD_COLOR,
        MUSIC_COLOR,
        StandardScaler,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        key_insight_card,
        load_book_bundle,
        load_spectral_features,
        mo,
        np,
        pd,
        pl,
        plt,
    )


@app.cell
def _(data_provenance_md, mo):
    mo.vstack(
        [
            mo.md(
                r"""
# 0 · Quality Control Dashboard

### Clean runs before claiming biomarkers

This chapter closes the QC gap: **tSNR**, **spectral flatness / entropy / band-SNR**,
**spike fraction**, and **IsolationForest** outliers — with reactive thresholds.

| Metric | Meaning |
|---|---|
| `tsnr` | mean / std of raw slab mean BOLD (higher = cleaner) |
| `spectral_flatness` | Wiener entropy of PSD (high = noise-like) |
| `spectral_entropy` | normalized Shannon entropy of PSD |
| `band_snr_high` | high-band power vs out-of-band floor |
| `ts_spike_frac` | fraction of large frame-to-frame jumps |
| `qc_outlier` | IsolationForest flag (−1 → 1) |

**Architecture note.** Heavy TensorFlow stays offline (`run_tf_offline.py` → Ch V).
This QC layer is pure sklearn + Polars and ships on GitHub Pages WASM.
"""
            ),
            mo.md(data_provenance_md()),
        ]
    )
    return


@app.cell
def _(mo):
    cont = mo.ui.slider(
        0.05, 0.25, value=0.12, step=0.01, label="IsolationForest contamination"
    )
    flat_q = mo.ui.slider(
        0.80, 0.99, value=0.90, step=0.01, label="High-flatness quantile flag"
    )
    spike_thr = mo.ui.slider(
        0.05, 0.30, value=0.15, step=0.01, label="Spike fraction threshold"
    )
    corr_thr = mo.ui.slider(
        0.70, 0.99, value=0.90, step=0.01, label="Collinearity |r| prune"
    )
    drop_outliers = mo.ui.checkbox(True, label="Exclude IF outliers from clean table")
    mo.vstack(
        [
            mo.md("## Controls (reactive)"),
            mo.hstack([cont, flat_q], justify="start"),
            mo.hstack([spike_thr, corr_thr, drop_outliers], justify="start"),
        ]
    )
    return cont, corr_thr, drop_outliers, flat_q, spike_thr


@app.cell
def _(load_book_bundle, load_spectral_features, mo, pl):
    @mo.cache
    def load_runs_pl():
        df = load_spectral_features()
        if df is None or getattr(df, "empty", True):
            return pl.DataFrame()
        # drop huge PSD columns for interactive work
        drop = [c for c in ("psd_f", "psd_pxx") if c in df.columns]
        return pl.from_pandas(df.drop(columns=drop, errors="ignore"))

    runs_pl = load_runs_pl()
    bundle = load_book_bundle()
    mo.md(
        f"**Loaded runs:** {runs_pl.height} · columns: `{runs_pl.columns[:12]}…`"
        if runs_pl.height
        else "*No spectral features — run `python scripts/prepare_real_features.py`.*"
    )
    return bundle, runs_pl


@app.cell
def _(
    IsolationForest,
    StandardScaler,
    cont,
    flat_q,
    key_insight_card,
    mo,
    np,
    pl,
    runs_pl,
    spike_thr,
):
    @mo.cache
    def compute_qc(df_pl: pl.DataFrame, contamination: float, flat_q_: float, spike_thr_: float):
        if df_pl.height == 0:
            return df_pl, {}
        pdf = df_pl.to_pandas()
        # ensure QC columns exist (recompute IF even if prepare already flagged)
        qc_feats = [
            c
            for c in [
                "tsnr",
                "spectral_flatness",
                "spectral_entropy",
                "band_snr_high",
                "ts_spike_frac",
                "power_high",
                "spectral_centroid",
                "total_power",
            ]
            if c in pdf.columns
        ]
        stats = {"n": len(pdf), "qc_feats": qc_feats}
        if len(qc_feats) >= 2 and len(pdf) >= 5:
            X = pdf[qc_feats].apply(lambda s: s.astype(float))
            X = X.fillna(X.median(numeric_only=True))
            Xs = StandardScaler().fit_transform(X.values)
            iso = IsolationForest(
                n_estimators=200,
                contamination=float(contamination),
                random_state=42,
            )
            pred = iso.fit_predict(Xs)
            pdf["qc_outlier"] = (pred == -1).astype(int)
            pdf["qc_score"] = -iso.score_samples(Xs)  # higher = more anomalous
        else:
            pdf["qc_outlier"] = 0
            pdf["qc_score"] = 0.0

        if "spectral_flatness" in pdf.columns:
            thr_f = float(pdf["spectral_flatness"].quantile(float(flat_q_)))
            pdf["qc_high_flatness"] = (pdf["spectral_flatness"] >= thr_f).astype(int)
            stats["flatness_thr"] = thr_f
        else:
            pdf["qc_high_flatness"] = 0
        if "ts_spike_frac" in pdf.columns:
            pdf["qc_high_spike"] = (pdf["ts_spike_frac"] > float(spike_thr_)).astype(int)
        else:
            pdf["qc_high_spike"] = 0
        if "tsnr" in pdf.columns:
            med, sd = pdf["tsnr"].median(), pdf["tsnr"].std()
            pdf["qc_low_tsnr"] = (pdf["tsnr"] < med - sd).astype(int)
            stats["tsnr_thr"] = float(med - sd) if sd == sd else None
        else:
            pdf["qc_low_tsnr"] = 0

        pdf["qc_flag_any"] = (
            (pdf["qc_outlier"] == 1)
            | (pdf["qc_high_flatness"] == 1)
            | (pdf["qc_high_spike"] == 1)
            | (pdf["qc_low_tsnr"] == 1)
        ).astype(int)
        stats["n_outlier"] = int(pdf["qc_outlier"].sum())
        stats["n_flag_any"] = int(pdf["qc_flag_any"].sum())
        return pl.from_pandas(pdf), stats

    qc_pl, qc_stats = compute_qc(
        runs_pl, float(cont.value), float(flat_q.value), float(spike_thr.value)
    )
    mo.vstack(
        [
            mo.md("## 1. QC summary"),
            mo.md(
                f"Runs **{qc_stats.get('n', 0)}** · IF outliers **{qc_stats.get('n_outlier', 0)}** · "
                f"any flag **{qc_stats.get('n_flag_any', 0)}** · "
                f"features `{qc_stats.get('qc_feats', [])}`"
            ),
            mo.md(
                key_insight_card(
                    "QC is a gate, not a conclusion.",
                    "Flagged runs should be reviewed before LOOCV or TF metrics; "
                    "small-*n* studies are sensitive to a single bad PSD.",
                )
            ),
        ]
    )
    return qc_pl, qc_stats


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    MUSIC_COLOR,
    mo,
    np,
    pl,
    plt,
    qc_pl,
):
    _blocks = [mo.md("## 2. QC metric distributions")]
    if qc_pl.height == 0:
        _blocks.append(mo.md("*No data.*"))
    else:
        metrics = [
            c
            for c in [
                "tsnr",
                "spectral_flatness",
                "spectral_entropy",
                "band_snr_high",
                "ts_spike_frac",
                "power_high",
            ]
            if c in qc_pl.columns
        ]
        n = len(metrics)
        if n:
            _fig_hist, _axes_h = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
            if n == 1:
                _axes_h = [_axes_h]
            pdf = qc_pl.to_pandas()
            for _axh, col in zip(_axes_h, metrics):
                for g, c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
                    vals = pdf.loc[pdf["group"] == g, col].dropna()
                    if len(vals):
                        _axh.hist(
                            vals,
                            bins=8,
                            alpha=0.55,
                            label=g,
                            color=c,
                            edgecolor="white",
                        )
                _axh.set_title(col, fontsize=10)
                _axh.legend(frameon=False, fontsize=7)
                _axh.grid(True, axis="y", alpha=0.3)
            _fig_hist.tight_layout()
            _blocks.append(_fig_hist)

        # scatter tsnr vs flatness
        if "tsnr" in qc_pl.columns and "spectral_flatness" in qc_pl.columns:
            _fig_sc, _ax_sc = plt.subplots(figsize=(6.5, 4.2))
            pdf = qc_pl.to_pandas()
            for g, c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
                s = pdf[pdf.group == g]
                _ax_sc.scatter(
                    s["tsnr"],
                    s["spectral_flatness"],
                    c=c,
                    s=70,
                    label=g,
                    edgecolors="white",
                    alpha=0.85,
                )
            bad = (
                pdf[pdf["qc_flag_any"] == 1]
                if "qc_flag_any" in pdf.columns
                else pdf.iloc[0:0]
            )
            if len(bad):
                _ax_sc.scatter(
                    bad["tsnr"],
                    bad["spectral_flatness"],
                    facecolors="none",
                    edgecolors=MUSIC_COLOR,
                    s=140,
                    linewidths=1.6,
                    label="flagged",
                )
            _ax_sc.set_xlabel("tSNR")
            _ax_sc.set_ylabel("Spectral flatness")
            _ax_sc.set_title("Quality map (ring = any QC flag)")
            _ax_sc.legend(frameon=False, fontsize=8)
            _ax_sc.grid(True, alpha=0.3)
            _fig_sc.tight_layout()
            _blocks.append(_fig_sc)
    mo.vstack(_blocks)
    return


@app.cell
def _(corr_thr, drop_outliers, key_insight_card, mo, np, pl, plt, qc_pl):
    _blocks = [mo.md("## 3. Flagged runs + collinearity on clean set")]
    if qc_pl.height == 0:
        _blocks.append(mo.md("*No data.*"))
        clean_pl = qc_pl
    else:
        flagged = qc_pl.filter(pl.col("qc_flag_any") == 1).select(
            [
                c
                for c in [
                    "subject",
                    "group",
                    "task",
                    "run",
                    "tsnr",
                    "spectral_flatness",
                    "band_snr_high",
                    "ts_spike_frac",
                    "qc_outlier",
                    "qc_low_tsnr",
                    "qc_high_flatness",
                    "qc_high_spike",
                ]
                if c in qc_pl.columns
            ]
        )
        _blocks.extend(
            [
                mo.md(f"**Flagged runs:** {flagged.height}"),
                mo.ui.table(flagged.to_pandas().round(4))
                if flagged.height
                else mo.md("*No flags at current thresholds.*"),
            ]
        )
        if drop_outliers.value and "qc_outlier" in qc_pl.columns:
            clean_pl = qc_pl.filter(pl.col("qc_outlier") == 0)
        else:
            clean_pl = qc_pl

        # collinearity among continuous music-relevant columns
        num_cols = [
            c
            for c in [
                "power_low",
                "power_mid",
                "power_high",
                "spectral_centroid",
                "spectral_flatness",
                "band_snr_high",
                "tsnr",
                "peak_amp",
                "coh_ant_post",
            ]
            if c in clean_pl.columns
        ]
        if len(num_cols) >= 2 and clean_pl.height >= 3:
            corr = clean_pl.select(num_cols).to_pandas().corr()
            _fig_corr, _ax_corr = plt.subplots(figsize=(7.2, 6))
            _im = _ax_corr.imshow(
                corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
            )
            _ax_corr.set_xticks(range(len(num_cols)))
            _ax_corr.set_yticks(range(len(num_cols)))
            _ax_corr.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
            _ax_corr.set_yticklabels(num_cols, fontsize=8)
            thr = float(corr_thr.value)
            pairs = []
            for i, a in enumerate(num_cols):
                for j, b in enumerate(num_cols):
                    if j <= i:
                        continue
                    r = corr.values[i, j]
                    if abs(r) >= thr:
                        pairs.append((a, b, float(r)))
            _ax_corr.set_title(
                f"Clean-set correlation (|r|≥{thr:.2f} pairs: {len(pairs)})"
            )
            _fig_corr.colorbar(_im, ax=_ax_corr, fraction=0.046)
            _fig_corr.tight_layout()
            pair_df = (
                pl.DataFrame(
                    {
                        "a": [p[0] for p in pairs],
                        "b": [p[1] for p in pairs],
                        "r": [p[2] for p in pairs],
                    }
                )
                if pairs
                else pl.DataFrame({"a": [], "b": [], "r": []})
            )
            _blocks.extend(
                [
                    _fig_corr,
                    mo.md("**High-|r| pairs to prune before linear models**"),
                    mo.ui.table(pair_df.to_pandas().round(3))
                    if pair_df.height
                    else mo.md("*None at this threshold.*"),
                ]
            )
        _blocks.append(
            mo.md(
                key_insight_card(
                    "Clean set size matters for LOOCV honesty.",
                    f"Clean runs after IF filter: **{clean_pl.height}** / {qc_pl.height}. "
                    "Export this table for bake-off / TF offline retrain.",
                )
            )
        )
    mo.vstack(_blocks)
    return (clean_pl,)


@app.cell
def _(clean_pl, mo, pl, qc_pl):
    _blocks = [mo.md("## 4. Export-ready clean feature table")]
    export_df = None
    if clean_pl.height == 0:
        _blocks.append(mo.md("*Nothing to export.*"))
    else:
        keep = [
            c
            for c in clean_pl.columns
            if c not in ("psd_f", "psd_pxx")
        ]
        export_df = clean_pl.select(keep)
        _blocks.extend(
            [
                mo.md(
                    f"**Rows:** {export_df.height} · **Cols:** {len(export_df.columns)}  \n"
                    "Save locally for ML: write from prepare, or copy from this table."
                ),
                mo.ui.table(export_df.head(20).to_pandas().round(4)),
                mo.md(
                    "```bash\n"
                    "# After adjusting prepare QC thresholds, refresh store:\n"
                    "python scripts/prepare_real_features.py\n"
                    "python scripts/run_ml_bakeoff.py\n"
                    "python scripts/run_tf_offline.py   # optional\n"
                    "python marimo_exports/export_wasm.py --sync-docs\n"
                    "```"
                ),
            ]
        )
    mo.vstack(_blocks)
    return (export_df,)


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## Takeaways

1. **tSNR + spectral flatness** catch many bad mean-BOLD runs without ROI masks.  
2. **IsolationForest** is a multivariate QC gate — re-tune contamination with the slider.  
3. **Collinearity prune** on the clean set protects LogReg coefficients (Ch III).  
4. **TF stays offline**; QC + classical ML stay WASM-friendly.  
5. Re-run prepare after code changes so `run_qc.csv` and book embeds stay in sync.
"""
            ),
            mo.md(
                clinical_relevance_card(
                    "Publishing music-effect biomarkers without run-level QC risks RecSys priors "
                    "driven by noise. Flag high-flatness / low-tSNR runs before claiming valence effects."
                )
            ),
            mo.md(book_nav("00_qc_dashboard")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
