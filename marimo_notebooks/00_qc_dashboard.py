"""Chapter 0 — QC dashboard: tSNR, multitaper PSD, IsolationForest, cleaned features.

Reactive marimo + Polars + @mo.cache. Public WASM chapter (no TensorFlow / MNE).
"""

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["marimo", "numpy", "pandas", "polars", "matplotlib", "scikit-learn", "scipy"]
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
        load_bold_timeseries,
        load_cleaned_spectral_features,
        load_multi_dataset_runs,
        load_spectral_features,
        load_spectral_features_polars,
        multi_dataset_run_ids,
        pandas_to_polars,
        set_global_style,
        studies_dataframe,
    )
    from spectral_methods import (
        adaptive_multitaper_psd,
        compare_psd_methods,
        compute_tsnr,
        multitaper_psd,
        spectral_feats,
        welch_psd,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        IsolationForest,
        MDD_COLOR,
        MUSIC_COLOR,
        StandardScaler,
        adaptive_multitaper_psd,
        book_nav,
        clinical_relevance_card,
        compare_psd_methods,
        compute_tsnr,
        data_provenance_md,
        key_insight_card,
        load_book_bundle,
        load_bold_timeseries,
        load_cleaned_spectral_features,
        load_multi_dataset_runs,
        load_spectral_features,
        load_spectral_features_polars,
        mo,
        multi_dataset_run_ids,
        multitaper_psd,
        np,
        pd,
        pandas_to_polars,
        pl,
        plt,
        spectral_feats,
        studies_dataframe,
        welch_psd,
    )


@app.cell
def _(data_provenance_md, mo):
    mo.vstack(
        [
            mo.md(
                r"""
# 0 · Quality Control Dashboard

### Clean runs before claiming biomarkers

This chapter closes the QC gap: **tSNR**, **DPSS multitaper** (uniform + adaptive),
**spectral flatness / entropy / band-SNR**, **spike fraction**, and
**IsolationForest** outliers — with reactive thresholds.

| Metric | Meaning |
|---|---|
| `tsnr` | mean / std of raw slab mean BOLD (higher = cleaner) |
| `psd_method` | `welch` · `uniform` multitaper · `adaptive` Thomson weights |
| `spectral_flatness` | Wiener entropy of PSD (high = noise-like) |
| `spectral_entropy` | normalized Shannon entropy of PSD |
| `band_snr_high` | high-band power vs out-of-band floor |
| `ts_spike_frac` | fraction of large frame-to-frame jumps |
| `qc_outlier` | IsolationForest flag |

**Why multitaper + tSNR?** BOLD epochs are short (TR=3 s). Welch alone is high-variance
in the high band (0.08–0.15 Hz). DPSS multitaper lowers variance; **adaptive** weights
down leaky high-order tapers on 1/f-like spectra. Low tSNR runs inject noise into both
the classical bake-off and the TF spectrogram model — gate them here.

**Architecture.** Pure scipy + sklearn + Polars → ships on GitHub Pages WASM.
MNE `adaptive=True` is optional offline (`prepare_real_features.py --psd mne`).
TF stays offline (`run_tf_offline.py` → Ch V).
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
def _(
    load_book_bundle,
    load_spectral_features_polars,
    load_spectral_features,
    mo,
    pandas_to_polars,
    pl,
):
    @mo.cache
    def load_runs_pl():
        pl_df = load_spectral_features_polars()
        if pl_df is not None and getattr(pl_df, "height", 0):
            drop = [c for c in ("psd_f", "psd_pxx") if c in pl_df.columns]
            return pl_df.drop(drop) if drop else pl_df
        df = load_spectral_features()
        if df is None or getattr(df, "empty", True):
            return pl.DataFrame()
        drop = [c for c in ("psd_f", "psd_pxx") if c in df.columns]
        # WASM-safe: never pl.from_pandas (needs pyarrow for nullable dtypes)
        return pandas_to_polars(df.drop(columns=drop, errors="ignore"))

    runs_pl = load_runs_pl()
    bundle = load_book_bundle()
    _psd = bundle.get("psd_method", "welch (legacy embed)")
    mo.md(
        f"**Loaded runs:** {runs_pl.height} · PSD method in store: **`{_psd}`**  \n"
        f"columns: `{runs_pl.columns[:12]}…`"
        if runs_pl.height
        else "*No spectral features — run `python scripts/prepare_real_features.py --psd adaptive`.*"
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
    pd,
    pandas_to_polars,
    pl,
    runs_pl,
    spike_thr,
):
    @mo.cache
    def compute_qc(df_pl: pl.DataFrame, contamination: float, flat_q_: float, spike_thr_: float):
        if df_pl.height == 0:
            return df_pl, {}
        # to_pandas is fine on small QC tables; back to polars without pyarrow
        pdf = df_pl.to_pandas()
        for c in pdf.columns:
            if str(pdf[c].dtype) in ("Int64", "Float64", "boolean"):
                pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
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
            X = pdf[qc_feats].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            X = X.fillna(X.median(numeric_only=True))
            Xs = StandardScaler().fit_transform(X.values.astype(float))
            iso = IsolationForest(
                n_estimators=200,
                contamination=float(contamination),
                random_state=42,
            )
            pred = iso.fit_predict(Xs)
            pdf["qc_outlier"] = (pred == -1).astype(np.int64)
            pdf["qc_score"] = -iso.score_samples(Xs)
        else:
            pdf["qc_outlier"] = np.int64(0)
            pdf["qc_score"] = 0.0

        if "spectral_flatness" in pdf.columns:
            thr_f = float(pdf["spectral_flatness"].quantile(float(flat_q_)))
            pdf["qc_high_flatness"] = (pdf["spectral_flatness"] >= thr_f).astype(np.int64)
            stats["flatness_thr"] = thr_f
        else:
            pdf["qc_high_flatness"] = np.int64(0)
        if "ts_spike_frac" in pdf.columns:
            pdf["qc_high_spike"] = (pdf["ts_spike_frac"] > float(spike_thr_)).astype(
                np.int64
            )
        else:
            pdf["qc_high_spike"] = np.int64(0)
        if "tsnr" in pdf.columns:
            med, sd = pdf["tsnr"].median(), pdf["tsnr"].std()
            pdf["qc_low_tsnr"] = (pdf["tsnr"] < med - sd).astype(np.int64)
            stats["tsnr_thr"] = float(med - sd) if sd == sd else None
        else:
            pdf["qc_low_tsnr"] = np.int64(0)

        pdf["qc_flag_any"] = (
            (pdf["qc_outlier"] == 1)
            | (pdf["qc_high_flatness"] == 1)
            | (pdf["qc_high_spike"] == 1)
            | (pdf["qc_low_tsnr"] == 1)
        ).astype(np.int64)
        stats["n_outlier"] = int(pdf["qc_outlier"].sum())
        stats["n_flag_any"] = int(pdf["qc_flag_any"].sum())
        return pandas_to_polars(pdf), stats

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
        keep = [c for c in clean_pl.columns if c not in ("psd_f", "psd_pxx")]
        export_df = clean_pl.select(keep)
        _csv = export_df.write_csv()
        _blocks.extend(
            [
                mo.md(
                    f"**Rows:** {export_df.height} · **Cols:** {len(export_df.columns)}  \n"
                    "Download for bake-off / TF offline retrain, or re-run prepare to write "
                    "`data/processed/cleaned_spectral_features.csv`."
                ),
                mo.download(
                    data=_csv.encode("utf-8") if isinstance(_csv, str) else _csv,
                    filename="cleaned_spectral_features_qc.csv",
                    mimetype="text/csv",
                    label="Download cleaned + QC'd features (CSV)",
                ),
                mo.ui.table(export_df.head(20).to_pandas().round(4)),
                mo.md(
                    "```bash\n"
                    "# Adaptive multitaper + tSNR + IsolationForest (default):\n"
                    "python scripts/prepare_real_features.py --psd adaptive\n"
                    "# Fast recompute from stored timeseries (no NIfTI I/O):\n"
                    "python scripts/prepare_real_features.py --from-timeseries --psd adaptive\n"
                    "# Production MNE adaptive (optional):\n"
                    "pip install mne && python scripts/prepare_real_features.py --psd mne\n"
                    "python scripts/run_ml_bakeoff.py\n"
                    "python scripts/run_tf_offline.py   # optional\n"
                    "python marimo_exports/export_wasm.py --sync-docs\n"
                    "# Local book + API:\n"
                    "uvicorn marimo_exports.fastapi_app:app --reload --port 8765\n"
                    "```"
                ),
            ]
        )
    mo.vstack(_blocks)
    return (export_df,)


@app.cell
def _(bundle, load_bold_timeseries, np):
    """Provide one real BOLD series for the multitaper demo cell."""

    def load_bold_ts():
        try:
            ts_df = load_bold_timeseries()
            if ts_df is not None and not getattr(ts_df, "empty", True):
                g = ts_df.sort_values("volume").groupby(
                    ["subject", "task", "run"], sort=False
                )
                (sub, task, run), first = next(iter(g))
                return first["bold_z"].to_numpy(dtype=float), {
                    "subject": sub,
                    "task": task,
                    "run": int(run),
                }
        except Exception:
            pass
        ex = (bundle or {}).get("timeseries_examples") or {}
        if not ex:
            return None, {}
        key = next(iter(ex))
        row = ex[key]
        return np.asarray(row["bold_z"], dtype=float), {
            "subject": row.get("subject", key),
            "task": row.get("task", "?"),
            "run": 1,
        }

    return (load_bold_ts,)


@app.cell
def _(
    adaptive_multitaper_psd,
    compare_psd_methods,
    load_bold_ts,
    mo,
    multitaper_psd,
    np,
    pd,
    plt,
    welch_psd,
):
    """Live comparison: Welch vs uniform vs adaptive multitaper on one BOLD run."""
    _blocks = [
        mo.md(
            r"""
## 5. Multitaper methods — uniform vs adaptive (live demo)

DPSS (Slepian) tapers solve the spectral concentration problem. The first
$K \approx 2NW - 1$ tapers concentrate energy inside $[-W, W]$.

| Method | Weights | Bias on 1/f BOLD | Cost |
|---|---|---|---|
| **Welch** | single Hann window | medium | cheap |
| **Uniform multitaper** | equal $1/K$ | higher leakage in valleys | cheap |
| **Adaptive (Thomson)** | $w_k(f) \propto \lambda_k^2 / (\lambda_k^2 \hat S(f) + (1-\lambda_k)\hat N)$ | lower | iterative |

Production offline can also use **MNE** `psd_array_multitaper(adaptive=True, low_bias=True)`.
"""
        )
    ]
    ts, meta = load_bold_ts()
    if ts is None or len(ts) < 16:
        _blocks.append(
            mo.md("*No example time series in bundle — run prepare to embed examples.*")
        )
    else:
        tr = 3.0
        fs = 1.0 / tr
        f_w, p_w = welch_psd(ts, fs)
        f_u, p_u = multitaper_psd(ts, fs, nw=3.5)
        f_a, p_a, w_a = adaptive_multitaper_psd(ts, fs, nw=3.5)
        feats = compare_psd_methods(ts, tr=tr, nw=3.5)
        _fig, _ax = plt.subplots(figsize=(8.5, 4.2))
        _ax.semilogy(f_w, p_w, label="Welch", alpha=0.75, lw=1.6)
        _ax.semilogy(f_u, p_u, label="Uniform MT", alpha=0.85, lw=1.8)
        _ax.semilogy(f_a, p_a, label="Adaptive MT", lw=2.2)
        _ax.axvspan(0.08, 0.15, color="#6C3483", alpha=0.12, label="high band")
        _ax.set_xlabel("Frequency (Hz)")
        _ax.set_ylabel("Power")
        _ax.set_title(
            f"PSD methods on {meta.get('subject', '?')} · {meta.get('task', '?')}"
        )
        _ax.legend(frameon=False, fontsize=8)
        _ax.grid(True, which="both", alpha=0.3)
        _fig.tight_layout()
        _cmp = {
            "method": list(feats.keys()),
            "power_high": [round(feats[m]["power_high"], 4) for m in feats],
            "flatness": [round(feats[m]["spectral_flatness"], 4) for m in feats],
            "band_snr_high": [round(feats[m]["band_snr_high"], 4) for m in feats],
            "centroid": [round(feats[m]["spectral_centroid"], 4) for m in feats],
        }
        _blocks.extend(
            [
                _fig,
                mo.md(
                    f"**Effective adaptive weight range** (taper × freq): "
                    f"min={float(np.min(w_a)):.3f} · max={float(np.max(w_a)):.3f} · "
                    f"mean={float(np.mean(w_a)):.3f}"
                ),
                mo.ui.table(pd.DataFrame(_cmp)),
            ]
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    MUSIC_COLOR,
    load_multi_dataset_runs,
    mo,
    multi_dataset_run_ids,
    np,
    pd,
    pandas_to_polars,
    plt,
    studies_dataframe,
):
    """Cross-ref studies from data/raw + multi-set god spectral runs."""
    studies = studies_dataframe()
    multi = load_multi_dataset_runs()
    ds_ids = multi_dataset_run_ids()
    blocks = [
        mo.md(
            f"""
## Multi-dataset QC (OpenNeuro cross-refs)

Primary book QC above uses the **ds000171** feature store.  
Multi-set spectral runs (god path) currently cover: **{', '.join(f'`{d}`' for d in ds_ids)}**.
"""
        )
    ]
    if studies is not None and not getattr(studies, "empty", True):
        show_cols = [
            c
            for c in (
                "dataset",
                "role",
                "status",
                "match_level",
                "n_subjects_on_disk",
                "n_bold_files",
                "short_title",
                "modality",
            )
            if c in studies.columns
        ]
        blocks.append(mo.md("### Studies under `data/raw/` + registry"))
        blocks.append(mo.ui.table(pandas_to_polars(studies[show_cols]), selection=None, page_size=12))

    if multi is not None and not getattr(multi, "empty", True) and "dataset" in multi.columns:
        agg_cols = [c for c in ("tsnr", "power_high", "spectral_centroid", "power_low") if c in multi.columns]
        if agg_cols:
            g = multi.groupby("dataset", as_index=False)[agg_cols].mean(numeric_only=True)
            g["n_runs"] = multi.groupby("dataset").size().values
            blocks.append(mo.md("### Multi-set run means (god / summary)"))
            blocks.append(mo.ui.table(pandas_to_polars(g.round(4)), selection=None))
            if "tsnr" in multi.columns:
                fig_m, ax_m = plt.subplots(figsize=(7.2, 3.6))
                for i, ds in enumerate(sorted(multi["dataset"].astype(str).unique())):
                    vals = pd.to_numeric(multi.loc[multi["dataset"].astype(str) == ds, "tsnr"], errors="coerce").dropna()
                    if len(vals):
                        ax_m.hist(
                            vals,
                            bins=min(12, max(4, len(vals) // 2)),
                            alpha=0.55,
                            label=ds,
                            color=[CONTROL_COLOR, MUSIC_COLOR, MDD_COLOR][i % 3],
                            edgecolor="white",
                        )
                ax_m.set_xlabel("tSNR")
                ax_m.set_title("tSNR by dataset (multi-set)")
                ax_m.legend(frameon=False, fontsize=8)
                fig_m.tight_layout()
                blocks.append(fig_m)
        sample_cols = [
            c
            for c in (
                "dataset",
                "subject",
                "group",
                "task",
                "run",
                "tsnr",
                "power_high",
                "spectral_centroid",
            )
            if c in multi.columns
        ]
        blocks.append(mo.md("### Sample multi-set runs"))
        blocks.append(
            mo.ui.table(
                pandas_to_polars(multi[sample_cols].head(40).round(4) if sample_cols else multi.head(40)),
                selection=None,
                page_size=12,
            )
        )
    else:
        blocks.append(
            mo.md(
                "*No multi-set runs yet. On host:*  \n"
                "`python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725`  \n"
                "`python scripts/god_mode_bold_to_tfdata.py --smoke-tfdata`"
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
## Takeaways

1. **tSNR + spectral flatness** catch many bad mean-BOLD runs without ROI masks.
2. **Adaptive multitaper** is the default production PSD; uniform is fine for prototypes; MNE is optional offline.
3. **IsolationForest** is a multivariate QC gate — re-tune contamination with the slider.
4. **Collinearity prune** on the clean set protects LogReg coefficients (Ch III).
5. **TF stays offline**; QC + classical ML stay WASM-friendly (FastAPI serves the same static WASM book).
6. Re-run prepare after code changes so `cleaned_spectral_features.csv` and book embeds stay in sync.
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
