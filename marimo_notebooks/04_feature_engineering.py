"""Chapter IV — Responder maps (public book)."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="IV · Responder Maps")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from helpers import (
        CONTROL_COLOR, MDD_COLOR, book_nav, clinical_relevance_card,
        data_provenance_md, hypothesis_card, key_insight_card,
        load_spectral_features, set_global_style,
    )
    set_global_style()
    return (
        CONTROL_COLOR, GaussianMixture, MDD_COLOR, StandardScaler,
        book_nav, clinical_relevance_card, data_provenance_md,
        hypothesis_card, key_insight_card, load_spectral_features,
        mo, np, pd, plt,
    )


@app.cell
def _(data_provenance_md, hypothesis_card, mo):
    mo.md(r"""
# IV · Responder Maps

### Spectral fingerprints → subtypes → recommendation

We build a feature vector per **real** run, then cluster for subtypes that may cut across diagnostic labels.
""")
    mo.md(data_provenance_md())
    mo.md(hypothesis_card(
        "Spectral space reveals music-responder subtypes.",
        "A high-centroid, high-band cluster should over-represent controls on music runs.",
    ))
    return


@app.cell
def _(load_spectral_features, mo):
    feats = load_spectral_features()
    mo.md("## Feature table (real processed runs)")
    if feats.empty:
        mo.md("*No features — run prepare_real_features.py*")
    else:
        _cols = [c for c in ["subject","group","task","run","power_low","power_mid","power_high","spectral_centroid","peak_latency_s","peak_amp"] if c in feats.columns]
        mo.ui.table(feats[_cols].round(4))
    return (feats,)


@app.cell
def _(mo):
    mo.md("## Clustering music runs")
    n_clust = mo.ui.slider(2, 4, value=3, step=1, label="GMM clusters")
    n_clust
    return (n_clust,)


@app.cell
def _(
    CONTROL_COLOR, GaussianMixture, MDD_COLOR, StandardScaler,
    feats, key_insight_card, mo, n_clust, np, pd, plt,
):
    music = feats[feats.task == "music"].copy() if not feats.empty else pd.DataFrame()
    fcols = [c for c in ["power_low","power_mid","power_high","spectral_centroid"] if c in music.columns]
    if len(music) < 3 or len(fcols) < 2:
        mo.md("*Not enough music runs to cluster.*")
    else:
        X = StandardScaler().fit_transform(music[fcols])
        k = min(int(n_clust.value), len(music))
        music = music.copy()
        music["cluster"] = GaussianMixture(n_components=k, random_state=42).fit_predict(X)
        fig_sc, ax_sc = plt.subplots(figsize=(7.5, 4.5))
        for _cl in sorted(music["cluster"].unique()):
            sub = music[music.cluster == _cl]
            ax_sc.scatter(sub["spectral_centroid"], sub["power_high"], s=80, alpha=0.85, label=f"cluster {_cl}")
        for _, _row in music.iterrows():
            ax_sc.annotate(str(_row["subject"]).replace("sub-",""), (_row["spectral_centroid"], _row["power_high"]), fontsize=8, alpha=0.8)
        ax_sc.set_xlabel("Spectral centroid (Hz)")
        ax_sc.set_ylabel("High-band power fraction")
        ax_sc.set_title("Music-run fingerprints (GMM) — real BOLD")
        ax_sc.legend(frameon=False)
        ax_sc.grid(True, alpha=0.3)
        cross = pd.crosstab(music["cluster"], music["group"], normalize="index").round(3)
        cmin, cmax = music["spectral_centroid"].min(), music["spectral_centroid"].max()
        music["responder_score"] = music["power_high"] + 0.5 * (music["spectral_centroid"] - cmin) / (cmax - cmin + 1e-12)
        fig_h, ax_h = plt.subplots(figsize=(7, 3.5))
        for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
            vals = music.loc[music.group == _g, "responder_score"]
            if len(vals):
                ax_h.hist(vals, bins=6, alpha=0.55, label=_g, color=_c, edgecolor="white")
        ax_h.set_xlabel("Responder score")
        ax_h.set_title("Music responder score (real subset)")
        ax_h.legend(frameon=False)
        ax_h.grid(True, axis="y", alpha=0.3)
        top = music.nlargest(max(1, len(music)//3), "responder_score")
        frac = float((top.group == "Control").mean())
        mo.vstack([
            fig_sc,
            mo.md("**Cluster × clinical group (row-normalized)**"),
            mo.ui.table(cross),
            fig_h,
            mo.md(key_insight_card(
                "Subtypes are spectral, not only diagnostic.",
                "GMM partitions music runs in band-fraction × centroid space.",
                effect_size=f"top-tertile music runs ≈ {100*frac:.0f}% Control",
            )),
        ])
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack([
        mo.md(r"""
## From features to a RecSys sketch

```text
BOLD run → Welch bands + centroid + peak timing
        → standardized vector → cluster / responder score
        → playlist prior (audio brightness, tempo, mode)
```
"""),
        mo.md(clinical_relevance_card(
            "Responder maps turn research into structure: who engages with which musical energy."
        )),
        mo.md(book_nav("04_feature_engineering")),
    ])
    return


if __name__ == "__main__":
    app.run()
