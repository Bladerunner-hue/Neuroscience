"""04 — Feature engineering & responder clusters. Canonical marimo notebook."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "plotly",
#     "scipy",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="04 — Features & Clusters")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt  # noqa: F401  # pull for Pyodide package graph
    import plotly.express as px
    from scipy.signal import welch
    from sklearn.cluster import SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        set_global_style,
        trapz_integral,
    )

    set_global_style()

    def extract_spectral_fingerprint(bold, fs=1 / 3.0, nperseg=32):
        f, Pxx = welch(bold, fs=fs, nperseg=min(nperseg, max(8, len(bold) // 2)))
        total = trapz_integral(Pxx, f) + 1e-12
        low_m = (f >= 0.01) & (f < 0.04)
        mid_m = (f >= 0.04) & (f < 0.08)
        high_m = (f >= 0.08) & (f <= 0.15)
        low = trapz_integral(Pxx[low_m], f[low_m]) if np.any(low_m) else 0.0
        mid = trapz_integral(Pxx[mid_m], f[mid_m]) if np.any(mid_m) else 0.0
        high = trapz_integral(Pxx[high_m], f[high_m]) if np.any(high_m) else 0.0
        centroid = float(np.sum(f * Pxx) / total)
        return {
            "power_low": low / total,
            "power_mid": mid / total,
            "power_high": high / total,
            "spectral_centroid": centroid,
            "total_power": total,
        }

    return (
        CONTROL_COLOR,
        GaussianMixture,
        MDD_COLOR,
        SpectralClustering,
        StandardScaler,
        book_nav,
        clinical_relevance_card,
        extract_spectral_fingerprint,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        mo,
        np,
        pd,
        px,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
# 04 — Feature Engineering & Responder Clusters

**Chapter 4** · Spectral fingerprints + GMM / spectral clustering for music-responder subtypes.
"""
    )
    return


@app.cell
def _(hypothesis_card, mo):
    mo.md(
        hypothesis_card(
            "Spectral feature space reveals strong vs blunted music responders that enrich MDD/Control labels.",
            "Clusters become features for a RecSys recommending music a patient is likely to respond to.",
        )
    )
    return


@app.cell
def _(mo):
    n_sub = mo.ui.slider(8, 24, value=16, step=2, label="Subjects")
    n_clust = mo.ui.slider(2, 5, value=3, step=1, label="Number of clusters")
    mo.md("## Reactive controls")
    mo.hstack([n_sub, n_clust], justify="start")
    return n_clust, n_sub


@app.cell
def _(
    extract_spectral_fingerprint,
    make_synthetic_bold_dataset,
    mo,
    n_sub,
    np,
    pd,
):
    synth = make_synthetic_bold_dataset(int(n_sub.value), n_timepoints=105, tr=3.0)
    fingerprints = []
    for subj in synth["subject"].unique():
        sdata = synth[synth.subject == subj]
        grp = sdata["group"].iloc[0]
        pos = sdata[sdata.trial_type == "positive_music"]["bold"].values
        if len(pos) < 20:
            continue
        feats = extract_spectral_fingerprint(pos)
        feats["subject"] = subj
        feats["group"] = grp
        fingerprints.append(feats)
    feat_df = pd.DataFrame(fingerprints)
    rng = np.random.default_rng(0)
    feat_df["depr_proxy"] = (feat_df.group == "MDD").astype(float) * 0.6 + rng.normal(
        0, 0.15, len(feat_df)
    )
    mo.md("## Subject-level spectral fingerprints")
    mo.ui.table(feat_df.round(4))
    return (feat_df,)


@app.cell
def _(
    CONTROL_COLOR,
    GaussianMixture,
    MDD_COLOR,
    SpectralClustering,
    StandardScaler,
    feat_df,
    mo,
    n_clust,
    pd,
    px,
):
    features = ["power_low", "power_mid", "power_high", "spectral_centroid"]
    X = StandardScaler().fit_transform(feat_df[features])
    k = int(n_clust.value)

    sc = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", random_state=42)
    gmm = GaussianMixture(n_components=k, random_state=42)
    out = feat_df.copy()
    out["cluster_sc"] = sc.fit_predict(X)
    out["cluster_gmm"] = gmm.fit_predict(X)

    mo.md("## Clustering on power fingerprints")
    mo.ui.plotly(
        px.scatter(
            out,
            x="spectral_centroid",
            y="power_high",
            color="cluster_gmm",
            symbol="group",
            hover_data=["subject"],
            title="Spectral fingerprints colored by GMM cluster",
            color_continuous_scale="Viridis",
        )
    )
    cross = pd.crosstab(out["cluster_gmm"], out["group"], normalize="index").round(3)
    mo.md("**Cluster membership by clinical group (GMM, row-normalized)**")
    mo.ui.table(cross)

    cmin, cmax = out["spectral_centroid"].min(), out["spectral_centroid"].max()
    out["responder_score"] = out["power_high"] + 0.5 * (
        out["spectral_centroid"] - cmin
    ) / (cmax - cmin + 1e-12)
    mo.ui.plotly(
        px.histogram(
            out,
            x="responder_score",
            color="group",
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Control": CONTROL_COLOR, "MDD": MDD_COLOR},
            title="Music-responder score distribution",
        )
    )
    return (out,)


@app.cell
def _(book_nav, clinical_relevance_card, key_insight_card, mo, out):
    high = out.nlargest(max(1, len(out) // 3), "responder_score")
    frac_ctrl = float((high["group"] == "Control").mean())
    mo.md(
        key_insight_card(
            "Clusters separate music responders — basis for personalized music RecSys.",
            "High power_high + centroid cluster is Control-enriched; many MDD fall into low-response clusters.",
            effect_size=f"top-tertile responders ~{100 * frac_ctrl:.0f}% Control",
        )
    )
    mo.md(
        clinical_relevance_card(
            "Music-responder spectral clusters are direct inputs for playlist recommendation systems."
        )
    )
    mo.md(
        """
## Production path (local only)

```text
Spark Connect / Arrow UDFs for subject-parallel feature extraction.
TensorFlow training (chapter 06) stays outside Spark.
```
"""
    )
    mo.md(book_nav("04_feature_engineering"))
    return


if __name__ == "__main__":
    app.run()
