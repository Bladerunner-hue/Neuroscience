"""Chapter IV — Feature engineering, harmonization, music-effect map (public book)."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="IV · Features & Music Effects")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        NONMUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        hypothesis_card,
        key_insight_card,
        load_condition_features,
        load_participants_df,
        load_spectral_features,
        load_subject_features,
        set_global_style,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        NONMUSIC_COLOR,
        PCA,
        StandardScaler,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        hypothesis_card,
        key_insight_card,
        load_condition_features,
        load_participants_df,
        load_spectral_features,
        load_subject_features,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(data_provenance_md, hypothesis_card, mo):
    mo.md(
        r"""
# IV · Feature Engineering & Music Effects

### Inventory → harmonization → what each music condition does

This chapter is the **analysis control room**. Before modeling (Chapter III extension / logistic regression), we:

1. State **exactly what data we have**  
2. Check **harmonization** needs (age/sex imbalance, scaling)  
3. Build **trial-type features** so we can say *positive music vs tones* vs *negative music*  
4. Visualize **music effects** subject by subject  
"""
    )
    mo.md(data_provenance_md())
    mo.md(
        hypothesis_card(
            "Positive music should lift BOLD relative to tones more in controls than MDD.",
            "Valence (positive vs negative music) and domain (music vs non-music) are separable effects — we quantify both.",
        )
    )
    return


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    load_condition_features,
    load_participants_df,
    load_spectral_features,
    load_subject_features,
    mo,
    pd,
    plt,
):
    parts = load_participants_df()
    runs = load_spectral_features()
    cond = load_condition_features()
    subj = load_subject_features()

    mo.md("## 1. Data inventory (what we actually use)")

    inv = pd.DataFrame(
        [
            {
                "layer": "Full cohort metadata",
                "n": len(parts),
                "grain": "participant",
                "used_for": "Demographics, harmonization (age/sex)",
            },
            {
                "layer": "BOLD runs processed",
                "n": len(runs),
                "grain": "run (task×subject)",
                "used_for": "Run-level Welch PSD",
            },
            {
                "layer": "Trial-type epochs",
                "n": len(cond),
                "grain": "trial_type within run",
                "used_for": "Music valence / domain effects",
            },
            {
                "layer": "Subject feature rows",
                "n": len(subj),
                "grain": "subject",
                "used_for": "ML, contrasts, responder scores",
            },
        ]
    )
    bold_subs = sorted(runs["subject"].unique()) if not runs.empty else []
    mo.vstack(
        [
            mo.ui.table(inv),
            mo.md(
                f"**Subjects with BOLD in the feature store:** `{', '.join(bold_subs)}`  \n"
                f"Full cohort remains n={len(parts)} (Control={int((parts.group_short=='Control').sum())}, "
                f"MDD={int((parts.group_short=='MDD').sum())})."
            ),
        ]
    )
    return cond, parts, runs, subj


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, mo, parts, plt, runs, subj):
    mo.md(
        r"""
## 2. Harmonization checklist

fMRI group comparisons are easily confounded by **age**, **sex**, and **uneven sampling**.
Here we inspect the full cohort vs the BOLD subset and standardize continuous features for ML.
"""
    )
    fig_h, axes_h = plt.subplots(1, 3, figsize=(11, 3.6))
    # age full
    for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
        axes_h[0].hist(
            parts.loc[parts.group_short == _g, "age"],
            bins=8,
            alpha=0.55,
            label=_g,
            color=_c,
            edgecolor="white",
        )
    axes_h[0].set_title("Age — full cohort")
    axes_h[0].legend(frameon=False, fontsize=8)
    # sex
    sex_ct = parts.groupby(["group_short", "sex"]).size().unstack(fill_value=0)
    sex_ct.plot(kind="bar", ax=axes_h[1], color=["#AF7AC5", "#5DADE2"], edgecolor="white")
    axes_h[1].set_title("Sex × group — full cohort")
    axes_h[1].tick_params(axis="x", rotation=0)
    # age subset with bold
    if not subj.empty:
        for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
            axes_h[2].hist(
                subj.loc[subj.group == _g, "age"].dropna(),
                bins=5,
                alpha=0.55,
                label=_g,
                color=_c,
                edgecolor="white",
            )
        axes_h[2].set_title("Age — BOLD subset")
        axes_h[2].legend(frameon=False, fontsize=8)
    fig_h.tight_layout()

    notes = []
    if not parts.empty:
        age_diff = parts.groupby("group_short")["age"].mean()
        notes.append(
            f"Mean age Control={age_diff.get('Control', float('nan')):.1f}, "
            f"MDD={age_diff.get('MDD', float('nan')):.1f}."
        )
    if not subj.empty:
        notes.append(
            f"BOLD subset n={len(subj)} "
            f"(Control={int((subj.group=='Control').sum())}, MDD={int((subj.group=='MDD').sum())}) "
            "— **small**; treat ML metrics as exploratory LOOCV, not definitive."
        )
    notes.append(
        "Harmonization policy in this book: (1) z-score BOLD within run, "
        "(2) StandardScaler on ML features, (3) report age/sex explicitly, "
        "(4) never pool music and non-music without a task factor."
    )
    mo.vstack([fig_h, mo.md("\n\n".join(f"- {n}" for n in notes))])
    return


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    MUSIC_COLOR,
    NONMUSIC_COLOR,
    cond,
    key_insight_card,
    mo,
    np,
    pd,
    plt,
):
    mo.md(
        r"""
## 3. What each music condition does (trial-type map)

Event files define **positive_music**, **negative_music**, **tones**, and non-music analogues.
For each epoch we take the mean whole-brain BOLD (z) and peak amplitude of the mean peri-stimulus waveform.
"""
    )
    if cond.empty:
        mo.md("*No condition features — re-run prepare_real_features.py*")
    else:
        order = [
            "positive_music",
            "negative_music",
            "tones",
            "positive_nonmusic",
            "negative_nonmusic",
        ]
        colors = {
            "positive_music": MUSIC_COLOR,
            "negative_music": "#7B241C",
            "tones": "#7F8C8D",
            "positive_nonmusic": NONMUSIC_COLOR,
            "negative_nonmusic": "#6E2C00",
        }
        heat = (
            cond.groupby(["group", "trial_type"])["mean_bold"]
            .mean()
            .unstack()
            .reindex(columns=[c for c in order if c in cond.trial_type.unique()])
        )
        fig_m, axes_m = plt.subplots(1, 2, figsize=(11, 4.2))
        # grouped bars mean bold
        stims = [c for c in order if c in cond.trial_type.unique()]
        _x = np.arange(len(stims))
        for _i, (_g, _c) in enumerate([("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]):
            _vals = []
            for _s in stims:
                _sub = cond[(cond.group == _g) & (cond.trial_type == _s)]
                _vals.append(float(_sub.mean_bold.mean()) if len(_sub) else np.nan)
            axes_m[0].bar(
                _x + (_i - 0.5) * 0.35,
                _vals,
                0.35,
                label=_g,
                color=_c,
                edgecolor="white",
            )
        axes_m[0].axhline(0, color="#999", lw=0.8)
        axes_m[0].set_xticks(_x)
        axes_m[0].set_xticklabels(
            [s.replace("_", "\n") for s in stims], fontsize=8
        )
        axes_m[0].set_ylabel("Mean BOLD (z)")
        axes_m[0].set_title("Mean BOLD by stimulus type")
        axes_m[0].legend(frameon=False)

        # peak amp
        for _i, (_g, _c) in enumerate([("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]):
            _vals = []
            for _s in stims:
                _sub = cond[(cond.group == _g) & (cond.trial_type == _s)]
                _vals.append(float(_sub.peak_amp.mean()) if len(_sub) else np.nan)
            axes_m[1].bar(
                _x + (_i - 0.5) * 0.35,
                _vals,
                0.35,
                label=_g,
                color=_c,
                edgecolor="white",
            )
        axes_m[1].set_xticks(_x)
        axes_m[1].set_xticklabels(
            [s.replace("_", "\n") for s in stims], fontsize=8
        )
        axes_m[1].set_ylabel("Peak amplitude")
        axes_m[1].set_title("Peri-stimulus peak by stimulus type")
        axes_m[1].legend(frameon=False)
        fig_m.tight_layout()

        # effect story text from numbers
        def _mean(g, trial, col="mean_bold"):
            s = cond[(cond.group == g) & (cond.trial_type == trial)]
            return float(s[col].mean()) if len(s) else np.nan

        story = (
            f"**Positive music mean BOLD** — Control {_mean('Control','positive_music'):+.3f}, "
            f"MDD {_mean('MDD','positive_music'):+.3f}.  \n"
            f"**Negative music** — Control {_mean('Control','negative_music'):+.3f}, "
            f"MDD {_mean('MDD','negative_music'):+.3f}.  \n"
            f"**Tones** — Control {_mean('Control','tones'):+.3f}, "
            f"MDD {_mean('MDD','tones'):+.3f}."
        )
        mo.vstack(
            [
                fig_m,
                mo.ui.table(heat.round(3).reset_index()),
                mo.md(story),
                mo.md(
                    key_insight_card(
                        "Stimulus identity matters as much as diagnosis.",
                        "Positive music, negative music, and tones are not interchangeable labels — "
                        "each has a distinct mean/peak profile. Downstream ML uses *contrasts* "
                        "(e.g. positive music − tones) so we measure **music effects**, not raw baselines.",
                    )
                ),
            ]
        )
    return


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, key_insight_card, mo, np, pd, plt, subj):
    mo.md(
        r"""
## 4. Subject-level music-effect contrasts

| Contrast | Interpretation |
|---|---|
| `pos_music_vs_tones_bold` | Does **positive music** elevate BOLD vs tones? |
| `pos_music_vs_neg_music_bold` | Valence within music (pos − neg) |
| `music_vs_nonmusic_bold` | Music domain vs non-music domain (avg valence) |
| `pos_music_vs_pos_nonmusic_bold` | Is the lift music-specific at matched positive valence? |
"""
    )
    if subj.empty:
        mo.md("*No subject features.*")
    else:
        contrast_cols = [
            c
            for c in [
                "pos_music_vs_tones_bold",
                "pos_music_vs_neg_music_bold",
                "music_vs_nonmusic_bold",
                "pos_music_vs_pos_nonmusic_bold",
                "pos_music_vs_tones_power_high",
            ]
            if c in subj.columns
        ]
        show = subj[
            ["subject", "group", "age", "sex"] + contrast_cols
        ].round(3)
        fig_c, ax_c = plt.subplots(figsize=(8, 4))
        # lollipop of pos_music_vs_tones
        if "pos_music_vs_tones_bold" in subj.columns:
            s2 = subj.dropna(subset=["pos_music_vs_tones_bold"]).sort_values(
                "pos_music_vs_tones_bold"
            )
            cols = [
                CONTROL_COLOR if g == "Control" else MDD_COLOR for g in s2["group"]
            ]
            ax_c.barh(
                s2["subject"].str.replace("sub-", ""),
                s2["pos_music_vs_tones_bold"],
                color=cols,
                edgecolor="white",
            )
            ax_c.axvline(0, color="#333", lw=0.8)
            ax_c.set_xlabel("Positive music − tones (mean BOLD z)")
            ax_c.set_title("Individual music effect (positive music vs tones)")
        fig_c.tight_layout()
        mo.vstack(
            [
                mo.ui.table(show),
                fig_c,
                mo.md(
                    key_insight_card(
                        "Music effects are heterogeneous across people.",
                        "Some controls show strong positive-music lift vs tones; others do not. "
                        "MDD is similarly mixed in this small subset — which is exactly why "
                        "responder clustering and ML matter more than a single group mean.",
                    )
                ),
            ]
        )
    return


@app.cell
def _(PCA, StandardScaler, mo, np, pd, plt, subj, CONTROL_COLOR, MDD_COLOR):
    mo.md(
        r"""
## 5. Feature space after scaling (PCA view)

Continuous music-effect features are **StandardScaler**-normalized (zero mean, unit variance) before PCA / ML.
That is the main numeric harmonization step for comparability across feature units.
"""
    )
    if subj.empty:
        mo.md("*No subject features for PCA.*")
    else:
        feat_cols = [
            c
            for c in subj.columns
            if c
            not in ("subject", "group", "age", "sex", "sex_m", "age_z")
            and pd.api.types.is_numeric_dtype(subj[c])
        ]
        Xdf = subj[feat_cols].apply(pd.to_numeric, errors="coerce")
        # drop all-nan columns
        Xdf = Xdf.dropna(axis=1, how="all")
        # impute column means for remaining nans (small-n pragmatic)
        Xdf = Xdf.fillna(Xdf.mean())
        if Xdf.shape[1] < 2 or len(Xdf) < 2:
            mo.md("*Not enough complete features for PCA.*")
        else:
            Xs = StandardScaler().fit_transform(Xdf.values)
            pca = PCA(n_components=min(2, Xs.shape[1], Xs.shape[0]))
            Z = pca.fit_transform(Xs)
            fig_p, ax_p = plt.subplots(figsize=(6.5, 5))
            for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
                m = (subj["group"].values == _g)[: len(Z)]
                ax_p.scatter(Z[m, 0], Z[m, 1] if Z.shape[1] > 1 else np.zeros(m.sum()), s=90, c=_c, label=_g, edgecolors="white")
            for _i, _row in subj.reset_index(drop=True).iterrows():
                if _i < len(Z):
                    ax_p.annotate(
                        str(_row["subject"]).replace("sub-", ""),
                        (Z[_i, 0], Z[_i, 1] if Z.shape[1] > 1 else 0),
                        fontsize=8,
                    )
            ax_p.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.0f}% var)")
            if Z.shape[1] > 1:
                ax_p.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.0f}% var)")
            ax_p.set_title("Subject feature space (scaled music effects)")
            ax_p.legend(frameon=False)
            ax_p.grid(True, alpha=0.3)
            # loadings
            load = pd.DataFrame(
                pca.components_.T,
                index=Xdf.columns,
                columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            )
            mo.vstack(
                [
                    fig_p,
                    mo.md("**PCA loadings** (which music-effect features drive the axes)"),
                    mo.ui.table(load.round(3).reset_index().rename(columns={"index": "feature"})),
                ]
            )
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## Pipeline summary

```text
OpenNeuro ds000171 NIfTI + events.tsv
    → whole-brain mean BOLD (z per run)
    → run-level Welch PSD  (power_low/mid/high, centroid)
    → trial-type epochs    (positive_music, negative_music, tones, …)
    → subject contrasts    (pos_music − tones, music − nonmusic, …)
    → StandardScaler + PCA / ML (next chapters)
```

**Harmonization:** within-run z-scoring; age/sex reported; feature scaling for models; task never collapsed blindly.
"""
            ),
            mo.md(
                clinical_relevance_card(
                    "Knowing *which* musical valence moves BOLD — and in whom — is the prerequisite for personalized music medicine and for recommenders that optimise engagement rather than genre tags."
                )
            ),
            mo.md(book_nav("04_feature_engineering")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
