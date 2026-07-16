"""Chapter III — Multivariate analysis, ML, confusion matrices, explainability."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy", "scikit-learn"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="III · Multivariate & ML")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import coherence
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score,
        f1_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
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
        load_bold_timeseries,
        load_condition_features,
        load_spectral_features,
        load_subject_features,
        set_global_style,
        trapz_integral,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        LeaveOneOut,
        LogisticRegression,
        MDD_COLOR,
        MUSIC_COLOR,
        NONMUSIC_COLOR,
        Pipeline,
        RandomForestClassifier,
        SimpleImputer,
        StandardScaler,
        accuracy_score,
        book_nav,
        classification_report,
        clinical_relevance_card,
        coherence,
        confusion_matrix,
        cross_val_predict,
        data_provenance_md,
        f1_score,
        hypothesis_card,
        key_insight_card,
        load_bold_timeseries,
        load_condition_features,
        load_spectral_features,
        load_subject_features,
        mo,
        np,
        pd,
        plt,
        trapz_integral,
    )


@app.cell
def _(data_provenance_md, hypothesis_card, mo):
    mo.md(
        r"""
# III · Multivariate Analysis & Machine Learning

### From joint structure → classifiers → explainability

This chapter answers three questions on **real** OpenNeuro features:

1. **How do features co-vary?** (correlation structure of music-effect variables)
2. **Can simple models separate groups / stimulus domains?** (logistic regression, random forest + LOOCV)
3. **Which features drive decisions?** (coefficients, impurity importance, confusion matrices)

We keep a transparent **coherence** demo (early vs late half of whole-brain mean BOLD) as a network-timing proxy, then pivot to supervised ML on trial-type and subject contrasts so the story stays about **what music does**.
"""
    )
    mo.md(data_provenance_md())
    mo.md(
        hypothesis_card(
            "Music-effect contrasts (not raw baselines) separate groups and stimulus domains.",
            "Positive music − tones and music-domain vs non-music should carry predictive signal; "
            "coefficients should highlight those contrasts if the anhedonia story is right.",
        )
    )
    return


@app.cell
def _(
    load_condition_features,
    load_spectral_features,
    load_subject_features,
    mo,
    pd,
):
    runs_ml = load_spectral_features()
    cond_ml = load_condition_features()
    subj_ml = load_subject_features()
    mo.md("## 0. Feature tables used for ML")
    _inv = pd.DataFrame(
        [
            {
                "table": "spectral_features (run)",
                "n": len(runs_ml),
                "target_examples": "task (music vs nonmusic), group",
            },
            {
                "table": "condition_features (trial_type × run)",
                "n": len(cond_ml),
                "target_examples": "domain, valence, trial_type",
            },
            {
                "table": "subject_features (wide contrasts)",
                "n": len(subj_ml),
                "target_examples": "group (Control vs MDD) — LOOCV, exploratory",
            },
        ]
    )
    mo.vstack(
        [
            mo.ui.table(_inv),
            mo.md(
                "Subject-level n is small → **leave-one-out CV only**, treat accuracy as directional. "
                "Run- and condition-level tables give more folds for domain / valence models."
            ),
        ]
    )
    return cond_ml, runs_ml, subj_ml


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    key_insight_card,
    mo,
    np,
    pd,
    plt,
    subj_ml,
):
    mo.md(
        r"""
## 1. Multivariate structure — correlation of music effects

Before fitting classifiers we inspect **joint** structure among subject-level contrasts.
High collinearity means logistic coefficients must be interpreted with care (and RF can still rank importance).
"""
    )
    if subj_ml.empty:
        mo.md("*No subject features.*")
    else:
        _corr_cols = [
            c
            for c in [
                "pos_music_vs_tones_bold",
                "neg_music_vs_tones_bold",
                "pos_music_vs_neg_music_bold",
                "pos_music_vs_pos_nonmusic_bold",
                "music_vs_nonmusic_bold",
                "pos_music_vs_tones_power_high",
                "run_power_high_mean",
                "run_centroid_mean",
                "positive_music_mean_bold",
                "negative_music_mean_bold",
                "tones_mean_bold",
            ]
            if c in subj_ml.columns
        ]
        _Xcorr = subj_ml[_corr_cols].apply(pd.to_numeric, errors="coerce")
        _corr_mat = _Xcorr.corr()
        _fig_corr, _ax_corr = plt.subplots(figsize=(8.5, 7))
        _im_corr = _ax_corr.imshow(
            _corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )
        _ax_corr.set_xticks(range(len(_corr_cols)))
        _ax_corr.set_yticks(range(len(_corr_cols)))
        _short = [c.replace("_bold", "").replace("_", "\n") for c in _corr_cols]
        _ax_corr.set_xticklabels(_short, fontsize=7, rotation=45, ha="right")
        _ax_corr.set_yticklabels(_short, fontsize=7)
        _ax_corr.set_title("Subject-level music-effect correlation matrix")
        for _i in range(len(_corr_cols)):
            for _j in range(len(_corr_cols)):
                _v = _corr_mat.values[_i, _j]
                if np.isfinite(_v):
                    _ax_corr.text(
                        _j,
                        _i,
                        f"{_v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if abs(_v) < 0.6 else "white",
                    )
        _fig_corr.colorbar(
            _im_corr, ax=_ax_corr, fraction=0.046, pad=0.04, label="Pearson r"
        )
        _fig_corr.tight_layout()

        _fig_sc, _ax_sc = plt.subplots(figsize=(6.5, 4.5))
        if (
            "pos_music_vs_tones_bold" in subj_ml.columns
            and "music_vs_nonmusic_bold" in subj_ml.columns
        ):
            for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
                _s = subj_ml[subj_ml.group == _g]
                _ax_sc.scatter(
                    _s["pos_music_vs_tones_bold"],
                    _s["music_vs_nonmusic_bold"],
                    s=100,
                    c=_c,
                    label=_g,
                    edgecolors="white",
                    zorder=3,
                )
                for _, _r in _s.iterrows():
                    _ax_sc.annotate(
                        str(_r["subject"]).replace("sub-", ""),
                        (
                            _r["pos_music_vs_tones_bold"],
                            _r["music_vs_nonmusic_bold"],
                        ),
                        fontsize=8,
                        xytext=(4, 4),
                        textcoords="offset points",
                    )
            _ax_sc.axhline(0, color="#999", lw=0.8)
            _ax_sc.axvline(0, color="#999", lw=0.8)
            _ax_sc.set_xlabel("Positive music − tones (BOLD z)")
            _ax_sc.set_ylabel("Music domain − non-music domain (BOLD z)")
            _ax_sc.set_title("Bivariate music-effect space")
            _ax_sc.legend(frameon=False)
            _ax_sc.grid(True, alpha=0.3)
        _fig_sc.tight_layout()
        mo.vstack(
            [
                _fig_corr,
                _fig_sc,
                mo.md(
                    key_insight_card(
                        "Music effects live in a joint space, not one scalar.",
                        "Correlated contrasts (valence × domain × tones) are the multivariate fingerprint "
                        "we feed to logistic regression and random forests. Points far in the upper-right "
                        "show both positive-music lift and music-domain preference.",
                    )
                ),
            ]
        )
    return


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    coherence,
    key_insight_card,
    load_bold_timeseries,
    mo,
    np,
    pd,
    plt,
    trapz_integral,
):
    _ts = load_bold_timeseries()
    mo.md(
        r"""
## 2. Coherence proxy (network timing)

With whole-brain means we cannot form true ROI–ROI networks. We split each run into **early vs late** halves and compute magnitude-squared coherence as a transparent **timing / self-coupling** demo on real mean BOLD.
"""
    )
    if _ts.empty:
        mo.md("*No timeseries.*")
    else:
        _records = []
        _fig_c, _axes_c = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for _ax, _task in zip(_axes_c, ["music", "nonmusic"]):
            for _g, _c, _ls in [
                ("Control", CONTROL_COLOR, "-"),
                ("MDD", MDD_COLOR, "--"),
            ]:
                _sub = _ts[(_ts.group == _g) & (_ts.task == _task)]
                _curves, _f_ref = [], None
                for (_sid, _run), _sg in _sub.groupby(["subject", "run"]):
                    _sig = _sg.sort_values("time")["bold_z"].values
                    if len(_sig) < 20:
                        continue
                    _n = len(_sig)
                    _a, _b = _sig[: _n // 2], _sig[_n // 2 : 2 * (_n // 2)]
                    _m = min(len(_a), len(_b))
                    _x, _y = _a[:_m], _b[:_m]
                    _f, _cxy = coherence(
                        _x, _y, fs=1 / 3.0, nperseg=min(28, max(8, len(_x) // 2))
                    )
                    if _f_ref is None:
                        _f_ref = _f
                    _curves.append(np.interp(_f_ref, _f, _cxy))
                    _mask = (_f_ref > 0.03) & (_f_ref < 0.10)
                    _band = (
                        trapz_integral(
                            np.interp(_f_ref, _f, _cxy)[_mask], _f_ref[_mask]
                        )
                        if np.any(_mask)
                        else 0.0
                    )
                    _records.append(
                        {
                            "subject": _sid,
                            "run": _run,
                            "group": _g,
                            "task": _task,
                            "band_coh": _band,
                        }
                    )
                if _curves and _f_ref is not None:
                    _ax.plot(
                        _f_ref,
                        np.mean(_curves, axis=0),
                        color=_c,
                        ls=_ls,
                        lw=2.2,
                        label=_g,
                    )
            _ax.axvspan(0.03, 0.10, color="#6C3483", alpha=0.1)
            _ax.set_title(f"Task: {_task}")
            _ax.set_xlabel("Frequency (Hz)")
            _ax.grid(True, alpha=0.3)
            _ax.legend(frameon=False)
        _axes_c[0].set_ylabel("Coherence")
        _fig_c.suptitle("Mean coherence (early vs late proxy)", y=1.02)
        _fig_c.tight_layout()
        _cdf = pd.DataFrame(_records)
        _agg_coh = (
            _cdf.groupby(["group", "task"], as_index=False)["band_coh"].mean()
            if len(_cdf)
            else pd.DataFrame()
        )
        _fig_b2, _ax_b2 = plt.subplots(figsize=(7, 3.8))
        if len(_agg_coh):
            _xpos = np.arange(2)
            for _i, (_g, _c) in enumerate(
                [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]
            ):
                _vals = [
                    float(
                        _agg_coh[
                            (_agg_coh.group == _g) & (_agg_coh.task == _t)
                        ].band_coh.iloc[0]
                    )
                    if len(_agg_coh[(_agg_coh.group == _g) & (_agg_coh.task == _t)])
                    else np.nan
                    for _t in ["music", "nonmusic"]
                ]
                _ax_b2.bar(
                    _xpos + (_i - 0.5) * 0.35,
                    _vals,
                    0.35,
                    label=_g,
                    color=_c,
                    edgecolor="white",
                )
            _ax_b2.set_xticks(_xpos)
            _ax_b2.set_xticklabels(["music", "nonmusic"])
            _ax_b2.set_ylabel("Band coherence")
            _ax_b2.set_title("Integrated coherence by group × task")
            _ax_b2.legend(frameon=False)
            _ax_b2.grid(True, axis="y", alpha=0.3)
        mo.vstack(
            [
                _fig_c,
                _fig_b2,
                mo.ui.table(_agg_coh.round(4))
                if len(_agg_coh)
                else mo.md("*No rows*"),
                mo.md(
                    key_insight_card(
                        "Music vs non-music changes coupling structure.",
                        "Early/late proxy on real mean BOLD — methodological demonstration, not ROI connectivity.",
                    )
                ),
            ]
        )
    return


@app.cell
def _(
    LeaveOneOut,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    SimpleImputer,
    StandardScaler,
    accuracy_score,
    classification_report,
    cond_ml,
    confusion_matrix,
    cross_val_predict,
    f1_score,
    key_insight_card,
    mo,
    np,
    pd,
    plt,
    runs_ml,
):
    mo.md(
        r"""
## 3. AI/ML — logistic regression & random forest

### Targets we can actually train

| Grain | Target | Why it matters for music |
|---|---|---|
| **Run** | `task` music vs nonmusic | Does the spectral fingerprint of a *run* encode musical context? |
| **Condition epoch** | `domain` music / nonmusic / tones | Stimulus-family discrimination from epoch features |
| **Condition epoch** | `valence` positive / negative / neutral | Emotional valence from BOLD summary stats |
| **Subject** | `group` Control vs MDD | Exploratory LOOCV on music-effect contrasts |

All models: **impute → scale → classifier**, predictions via **leave-one-out** (honest for small n).
"""
    )

    def _plot_cm(cm, labels, title, ax, cmap="Blues"):
        ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        _thresh = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
        for _ii in range(cm.shape[0]):
            for _jj in range(cm.shape[1]):
                ax.text(
                    _jj,
                    _ii,
                    format(int(cm[_ii, _jj]), "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[_ii, _jj] > _thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )

    def _loo_predict(X, y, model):
        y = np.asarray(y)
        _classes = np.unique(y)
        if len(_classes) < 2 or len(y) < 3:
            return y, y.copy(), _classes, None
        _pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler()),
                ("clf", model),
            ]
        )
        try:
            _yhat = cross_val_predict(_pipe, X, y, cv=LeaveOneOut())
        except Exception as e:
            return y, np.array(["err"] * len(y)), _classes, str(e)
        return y, _yhat, _classes, None

    def _cm_panel(y, yhat, classes, title, ax):
        _labels = list(classes)
        _cm = confusion_matrix(y, yhat, labels=_labels)
        _plot_cm(_cm, _labels, title, ax)
        _acc = accuracy_score(y, yhat) if len(y) else 0.0
        _f1 = f1_score(y, yhat, average="macro", zero_division=0)
        return _cm, _acc, _f1

    _results = []
    _figs = []

    # --- A: run-level task ---
    if not runs_ml.empty and "task" in runs_ml.columns:
        _run_feats = [
            c
            for c in [
                "power_low",
                "power_mid",
                "power_high",
                "spectral_centroid",
                "peak_amp",
                "peak_latency_s",
            ]
            if c in runs_ml.columns
        ]
        _Xr = runs_ml[_run_feats].apply(pd.to_numeric, errors="coerce").values
        _yr = runs_ml["task"].astype(str).values
        _fig_run, _axes_run = plt.subplots(1, 2, figsize=(10, 4))
        for _ax, _name, _model in zip(
            _axes_run,
            ["LogisticRegression", "RandomForest"],
            [
                LogisticRegression(max_iter=2000, class_weight="balanced"),
                RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced"
                ),
            ],
        ):
            _yt, _yp, _cls, _err = _loo_predict(_Xr, _yr, _model)
            if _err:
                _ax.set_title(f"{_name}: {_err}")
                _results.append(
                    {
                        "level": "run",
                        "target": "task",
                        "model": _name,
                        "n": len(_yr),
                        "acc": np.nan,
                        "f1_macro": np.nan,
                        "note": _err,
                    }
                )
            else:
                _cm, _acc, _f1 = _cm_panel(
                    _yt, _yp, _cls, f"Run · task · {_name}", _ax
                )
                _results.append(
                    {
                        "level": "run",
                        "target": "task",
                        "model": _name,
                        "n": len(_yr),
                        "acc": round(_acc, 3),
                        "f1_macro": round(_f1, 3),
                        "note": "LOOCV",
                    }
                )
        _fig_run.suptitle(
            "Confusion: music vs nonmusic from run spectral features", y=1.02
        )
        _fig_run.tight_layout()
        _figs.append(_fig_run)

    # --- B: condition domain + valence ---
    _report_dom = "No condition features."
    if not cond_ml.empty and "domain" in cond_ml.columns:
        _cfeats = [
            c
            for c in [
                "mean_bold",
                "std_bold",
                "peak_amp",
                "peak_latency_s",
                "power_high",
                "power_mid",
                "power_low",
                "spectral_centroid",
            ]
            if c in cond_ml.columns
        ]
        _Xc = cond_ml[_cfeats].apply(pd.to_numeric, errors="coerce").values
        _yd = cond_ml["domain"].astype(str).values
        _yv = cond_ml["valence"].astype(str).values

        _fig_dom, _axes_dom = plt.subplots(1, 2, figsize=(10, 4))
        for _ax, _name, _model in zip(
            _axes_dom,
            ["LogReg domain", "RF domain"],
            [
                LogisticRegression(max_iter=2000, class_weight="balanced"),
                RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced"
                ),
            ],
        ):
            _yt, _yp, _cls, _err = _loo_predict(_Xc, _yd, _model)
            if _err:
                _ax.set_title(_err)
            else:
                _cm, _acc, _f1 = _cm_panel(_yt, _yp, _cls, _name, _ax)
                _results.append(
                    {
                        "level": "condition",
                        "target": "domain",
                        "model": _name,
                        "n": len(_yd),
                        "acc": round(_acc, 3),
                        "f1_macro": round(_f1, 3),
                        "note": "LOOCV",
                    }
                )
        _fig_dom.suptitle(
            "Confusion: domain (music / nonmusic / tones) from epoch features",
            y=1.02,
        )
        _fig_dom.tight_layout()
        _figs.append(_fig_dom)

        _fig_val, _axes_val = plt.subplots(1, 2, figsize=(10, 4))
        for _ax, _name, _model in zip(
            _axes_val,
            ["LogReg valence", "RF valence"],
            [
                LogisticRegression(max_iter=2000, class_weight="balanced"),
                RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced"
                ),
            ],
        ):
            _yt, _yp, _cls, _err = _loo_predict(_Xc, _yv, _model)
            if _err:
                _ax.set_title(_err)
            else:
                _cm, _acc, _f1 = _cm_panel(_yt, _yp, _cls, _name, _ax)
                _results.append(
                    {
                        "level": "condition",
                        "target": "valence",
                        "model": _name,
                        "n": len(_yv),
                        "acc": round(_acc, 3),
                        "f1_macro": round(_f1, 3),
                        "note": "LOOCV",
                    }
                )
        _fig_val.suptitle(
            "Confusion: valence (positive / negative / neutral)", y=1.02
        )
        _fig_val.tight_layout()
        _figs.append(_fig_val)

        _yt, _yp, _cls, _err = _loo_predict(
            _Xc,
            _yd,
            RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            ),
        )
        _report_dom = (
            classification_report(_yt, _yp, zero_division=0)
            if _err is None
            else _err
        )

    res_df = pd.DataFrame(_results)
    mo.vstack(
        _figs
        + [
            mo.md("### LOOCV metrics summary"),
            mo.ui.table(res_df) if len(res_df) else mo.md("*No models run.*"),
            mo.md("### Domain classification report (RF, LOOCV)"),
            mo.md(f"```\n{_report_dom}\n```"),
            mo.md(
                key_insight_card(
                    "Confusion matrices show *where* models fail.",
                    "Off-diagonals between music and nonmusic (or positive vs negative) "
                    "reveal which stimulus families share BOLD fingerprints — critical for "
                    "recommender design that must not confuse valence or domain.",
                )
            ),
        ]
    )
    return (res_df,)


@app.cell
def _(
    LeaveOneOut,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    SimpleImputer,
    StandardScaler,
    accuracy_score,
    confusion_matrix,
    cross_val_predict,
    f1_score,
    key_insight_card,
    mo,
    np,
    pd,
    plt,
    subj_ml,
):
    mo.md(
        r"""
## 4. Subject-level group classification (exploratory)

Target: **Control vs MDD** from music-effect contrasts.
With n≈5 this is **not** a clinical claim — it is a reproducible LOOCV template that scales when more BOLD subjects are added.
"""
    )
    group_models = {}
    group_res = pd.DataFrame()
    if subj_ml.empty or subj_ml["group"].nunique() < 2:
        mo.md("*Need ≥2 groups in subject_features.*")
    else:
        _feat_cols = [
            c
            for c in subj_ml.columns
            if c not in ("subject", "group", "age", "sex", "sex_m", "age_z")
            and pd.api.types.is_numeric_dtype(subj_ml[c])
        ]
        _Xg = subj_ml[_feat_cols].apply(pd.to_numeric, errors="coerce")
        _Xg = _Xg.dropna(axis=1, how="all")
        _feat_cols = list(_Xg.columns)
        _yg = subj_ml["group"].astype(str).values
        _Xgv = _Xg.values

        def _plot_cm_g(cm, labels, title, ax, cmap="Blues"):
            ax.imshow(cm, interpolation="nearest", cmap=cmap)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")
            _thresh = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
            for _ii in range(cm.shape[0]):
                for _jj in range(cm.shape[1]):
                    ax.text(
                        _jj,
                        _ii,
                        format(int(cm[_ii, _jj]), "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[_ii, _jj] > _thresh else "black",
                        fontsize=12,
                        fontweight="bold",
                    )

        _fig_g, _axes_g = plt.subplots(1, 2, figsize=(10, 4))
        _rows_g = []
        for _ax, _name, _model in zip(
            _axes_g,
            ["LogisticRegression", "RandomForest"],
            [
                LogisticRegression(max_iter=3000, class_weight="balanced"),
                RandomForestClassifier(
                    n_estimators=300, random_state=42, class_weight="balanced"
                ),
            ],
        ):
            _pipe = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="mean")),
                    ("sc", StandardScaler()),
                    ("clf", _model),
                ]
            )
            if len(_yg) < 3:
                _ax.set_title(f"{_name}: n too small")
                continue
            _yhat = cross_val_predict(_pipe, _Xgv, _yg, cv=LeaveOneOut())
            _labels = sorted(np.unique(_yg))
            _cm = confusion_matrix(_yg, _yhat, labels=_labels)
            _plot_cm_g(_cm, _labels, f"Group · {_name}", _ax)
            _acc = accuracy_score(_yg, _yhat)
            _f1 = f1_score(_yg, _yhat, average="macro", zero_division=0)
            _rows_g.append(
                {
                    "model": _name,
                    "n": len(_yg),
                    "acc_loocv": round(_acc, 3),
                    "f1_macro": round(_f1, 3),
                }
            )
            _pipe.fit(_Xgv, _yg)
            group_models[_name] = {
                "pipe": _pipe,
                "features": _feat_cols,
                "yhat": _yhat,
                "y": _yg,
            }
        _fig_g.suptitle(
            "Confusion: Control vs MDD from subject music-effect features", y=1.02
        )
        _fig_g.tight_layout()
        group_res = pd.DataFrame(_rows_g)
        _pred_tbl = subj_ml[["subject", "group"]].copy()
        if "LogisticRegression" in group_models:
            _pred_tbl["pred_logreg"] = group_models["LogisticRegression"]["yhat"]
        if "RandomForest" in group_models:
            _pred_tbl["pred_rf"] = group_models["RandomForest"]["yhat"]
        mo.vstack(
            [
                _fig_g,
                mo.ui.table(group_res),
                mo.md("**Per-subject LOOCV predictions**"),
                mo.ui.table(_pred_tbl),
                mo.md(
                    key_insight_card(
                        "Small-n LOOCV is a template, not a clinical accuracy claim.",
                        f"Features used: {len(_feat_cols)} music-effect / spectral columns. "
                        "When more subjects land in the feature store, the same pipeline re-runs without redesign.",
                    )
                ),
            ]
        )
    return group_models, group_res


@app.cell
def _(
    CONTROL_COLOR,
    LogisticRegression,
    MDD_COLOR,
    MUSIC_COLOR,
    Pipeline,
    RandomForestClassifier,
    SimpleImputer,
    StandardScaler,
    cond_ml,
    group_models,
    key_insight_card,
    mo,
    np,
    pd,
    plt,
):
    mo.md(
        r"""
## 5. Explainability — what drives the models?

| Method | Reads as |
|---|---|
| **Logistic coefficients** (after scaling) | Direction & magnitude of linear association with class log-odds |
| **RF feature importance** (Gini / impurity) | How often a feature splits well across trees |
| **Music-effect map** | Human-readable “what musics do” aligned with model features |

Scaled features ⇒ coefficients are **comparable across units**. Positive coef for class “music” means higher feature → more likely music.
"""
    )
    _expl = []

    if not cond_ml.empty and "domain" in cond_ml.columns:
        _cfeats2 = [
            c
            for c in [
                "mean_bold",
                "std_bold",
                "peak_amp",
                "peak_latency_s",
                "power_high",
                "power_mid",
                "power_low",
                "spectral_centroid",
            ]
            if c in cond_ml.columns
        ]
        _Xc2 = cond_ml[_cfeats2].apply(pd.to_numeric, errors="coerce")
        _yd2 = cond_ml["domain"].astype(str).values
        _y_bin = (cond_ml["domain"].astype(str) == "music").astype(int).values
        _pipe_lr = Pipeline(
            [
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=3000, class_weight="balanced"),
                ),
            ]
        )
        _pipe_rf = Pipeline(
            [
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        _pipe_lr.fit(_Xc2.values, _y_bin)
        _pipe_rf.fit(_Xc2.values, _yd2)
        _coefs = _pipe_lr.named_steps["clf"].coef_.ravel()
        _imp = _pipe_rf.named_steps["clf"].feature_importances_
        _expl_df = pd.DataFrame(
            {
                "feature": _cfeats2,
                "logreg_coef_music_vs_rest": _coefs,
                "rf_importance_domain": _imp,
            }
        ).sort_values("rf_importance_domain", ascending=False)

        _fig_ex, _axes_ex = plt.subplots(1, 2, figsize=(11, 4.2))
        _order_c = np.argsort(_coefs)
        _axes_ex[0].barh(
            np.array(_cfeats2)[_order_c],
            _coefs[_order_c],
            color=[
                MUSIC_COLOR if v > 0 else "#7F8C8D" for v in _coefs[_order_c]
            ],
            edgecolor="white",
        )
        _axes_ex[0].axvline(0, color="#333", lw=0.8)
        _axes_ex[0].set_title("LogReg coef: music vs rest (scaled)")
        _axes_ex[0].set_xlabel("Coefficient")
        _order_i = np.argsort(_imp)
        _axes_ex[1].barh(
            np.array(_cfeats2)[_order_i],
            _imp[_order_i],
            color=MUSIC_COLOR,
            edgecolor="white",
        )
        _axes_ex[1].set_title("RF importance: domain (3-class)")
        _axes_ex[1].set_xlabel("Importance")
        _fig_ex.tight_layout()
        _expl.extend(
            [
                mo.md("### Epoch-level: music domain vs rest"),
                _fig_ex,
                mo.ui.table(_expl_df.round(4)),
            ]
        )

    if "LogisticRegression" in group_models:
        _gm = group_models["LogisticRegression"]
        _pipe_g = _gm["pipe"]
        _feats_g = _gm["features"]
        _clf = _pipe_g.named_steps["clf"]
        if hasattr(_clf, "coef_"):
            _classes_g = list(_clf.classes_)
            _coef_g = _clf.coef_.ravel()
            _pos_class = (
                _classes_g[1] if len(_classes_g) > 1 else _classes_g[0]
            )
            _fig_sg, _ax_sg = plt.subplots(
                figsize=(8, max(3.5, 0.28 * len(_feats_g)))
            )
            _order_g = np.argsort(_coef_g)
            _bar_colors = [
                CONTROL_COLOR
                if (_pos_class == "Control" and v > 0)
                or (_pos_class == "MDD" and v < 0)
                else MDD_COLOR
                for v in _coef_g[_order_g]
            ]
            _ax_sg.barh(
                np.array(_feats_g)[_order_g],
                _coef_g[_order_g],
                color=_bar_colors,
                edgecolor="white",
            )
            _ax_sg.axvline(0, color="#333", lw=0.8)
            _ax_sg.set_title(
                f"LogReg coefficients toward class **{_pos_class}**"
            )
            _ax_sg.set_xlabel("Scaled coefficient")
            _fig_sg.tight_layout()
            _coef_tbl = (
                pd.DataFrame({"feature": _feats_g, "coef": _coef_g})
                .sort_values("coef", key=lambda s: s.abs(), ascending=False)
                .round(4)
            )
            _expl.extend(
                [
                    mo.md(
                        f"### Subject-level group model (positive class = `{_pos_class}`)"
                    ),
                    _fig_sg,
                    mo.ui.table(_coef_tbl.head(15)),
                ]
            )

    if "RandomForest" in group_models:
        _gm_rf = group_models["RandomForest"]
        _pipe_rf_g = _gm_rf["pipe"]
        _feats_rf = _gm_rf["features"]
        _imp_rf = _pipe_rf_g.named_steps["clf"].feature_importances_
        _fig_rf, _ax_rf = plt.subplots(
            figsize=(8, max(3.5, 0.28 * len(_feats_rf)))
        )
        _order_rf = np.argsort(_imp_rf)
        _ax_rf.barh(
            np.array(_feats_rf)[_order_rf],
            _imp_rf[_order_rf],
            color=MDD_COLOR,
            edgecolor="white",
        )
        _ax_rf.set_title("RF feature importance — group (Control vs MDD)")
        _ax_rf.set_xlabel("Importance")
        _fig_rf.tight_layout()
        _expl.append(_fig_rf)

    if not cond_ml.empty:
        _lab_col = (
            "stim_label" if "stim_label" in cond_ml.columns else "trial_type"
        )
        _pivot = (
            cond_ml.groupby(["group", _lab_col])[
                ["mean_bold", "peak_amp", "power_high"]
            ]
            .mean()
            .round(3)
            .reset_index()
        )

        def _gm_mean(g, trial, col="mean_bold"):
            _s = cond_ml[(cond_ml.group == g) & (cond_ml.trial_type == trial)]
            return float(_s[col].mean()) if len(_s) else np.nan

        _story = mo.md(
            f"""
### What each music condition does (data → model features)

| Stimulus | Control mean BOLD | MDD mean BOLD | Role in ML |
|---|---:|---:|---|
| Positive music | {_gm_mean('Control','positive_music'):+.3f} | {_gm_mean('MDD','positive_music'):+.3f} | High reward candidate; enters `pos_music_*` and contrasts |
| Negative music | {_gm_mean('Control','negative_music'):+.3f} | {_gm_mean('MDD','negative_music'):+.3f} | Negative valence music; valence contrast |
| Tones | {_gm_mean('Control','tones'):+.3f} | {_gm_mean('MDD','tones'):+.3f} | Neutral baseline for `*_vs_tones_*` contrasts |
| Positive non-music | {_gm_mean('Control','positive_nonmusic'):+.3f} | {_gm_mean('MDD','positive_nonmusic'):+.3f} | Domain control at matched positive valence |
| Negative non-music | {_gm_mean('Control','negative_nonmusic'):+.3f} | {_gm_mean('MDD','negative_nonmusic'):+.3f} | Domain control at matched negative valence |

**Derived effects used by subject-level models:**

- `pos_music_vs_tones_bold` = positive music − tones
- `pos_music_vs_neg_music_bold` = positive − negative music (valence)
- `music_vs_nonmusic_bold` = avg(music) − avg(nonmusic) (domain)
- `pos_music_vs_pos_nonmusic_bold` = music-specific lift at positive valence

Control Δ(pos music − tones) = **{_gm_mean('Control','positive_music')-_gm_mean('Control','tones'):+.3f}**,
MDD Δ = **{_gm_mean('MDD','positive_music')-_gm_mean('MDD','tones'):+.3f}**.
"""
        )
        _expl.extend(
            [
                _story,
                mo.md("**Condition means (all groups)**"),
                mo.ui.table(_pivot),
                mo.md(
                    key_insight_card(
                        "Explainability closes the loop from spectrogram to playlist.",
                        "If RF/LogReg rank `pos_music_vs_tones` or high-band power highly, "
                        "the model is using the same music-effect language as the univariate chapters — "
                        "not an opaque embedding.",
                    )
                ),
            ]
        )

    if not _expl:
        mo.md("*Insufficient data for explainability panels.*")
    else:
        mo.vstack(_expl)
    return


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    MUSIC_COLOR,
    NONMUSIC_COLOR,
    cond_ml,
    key_insight_card,
    mo,
    pd,
    plt,
    subj_ml,
):
    mo.md(
        r"""
## 6. Music-effect decision board (synthesis)

One view that ties inventory → univariate effects → multivariate models for **what musics do**.
"""
    )
    if cond_ml.empty:
        mo.md("*No condition features.*")
    else:
        _stims_order = [
            "positive_music",
            "negative_music",
            "tones",
            "positive_nonmusic",
            "negative_nonmusic",
        ]
        _stims = [s for s in _stims_order if s in set(cond_ml.trial_type)]
        _fig_board, _axes_b = plt.subplots(1, 2, figsize=(11, 4.5))
        _heat = (
            cond_ml.groupby(["group", "trial_type"])["mean_bold"]
            .mean()
            .unstack()
            .reindex(columns=_stims)
        )
        _im_h = _axes_b[0].imshow(
            _heat.values.astype(float),
            cmap="RdYlGn",
            aspect="auto",
            vmin=-0.6,
            vmax=0.6,
        )
        _axes_b[0].set_xticks(range(len(_stims)))
        _axes_b[0].set_xticklabels(
            [s.replace("_", "\n") for s in _stims], fontsize=8
        )
        _axes_b[0].set_yticks(range(len(_heat.index)))
        _axes_b[0].set_yticklabels(list(_heat.index))
        _axes_b[0].set_title("Mean BOLD (z) by group × stimulus")
        for _i in range(_heat.shape[0]):
            for _j in range(_heat.shape[1]):
                _v = _heat.values[_i, _j]
                if pd.notna(_v):
                    _axes_b[0].text(
                        _j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=9
                    )
        _fig_board.colorbar(_im_h, ax=_axes_b[0], fraction=0.046)

        if not subj_ml.empty and "pos_music_vs_tones_bold" in subj_ml.columns:
            _s2 = subj_ml.dropna(subset=["pos_music_vs_tones_bold"]).sort_values(
                "pos_music_vs_tones_bold"
            )
            _cols = [
                CONTROL_COLOR if g == "Control" else MDD_COLOR
                for g in _s2["group"]
            ]
            _axes_b[1].barh(
                _s2["subject"].str.replace("sub-", ""),
                _s2["pos_music_vs_tones_bold"],
                color=_cols,
                edgecolor="white",
            )
            _axes_b[1].axvline(0, color="#333", lw=0.8)
            _axes_b[1].set_xlabel("Positive music − tones")
            _axes_b[1].set_title("Individual music-effect scores")
        _fig_board.tight_layout()

        _how = mo.md(
            r"""
### How we use the data (end-to-end)

```text
events.tsv trial_type  →  epoch mean BOLD + Welch bands
                         →  condition_features.csv
run NIfTI mean BOLD    →  run Welch PSD
                         →  spectral_features.csv
subject aggregation    →  music contrasts (pos−tones, music−nonmusic, …)
                         →  subject_features.csv
ML (this chapter)      →  LOOCV logistic + RF
                         →  confusion matrices + coefficients / importances
```

| Music condition | Typical analysis role |
|---|---|
| **Positive music** | Primary “rewarding music” condition; RecSys positive exemplar |
| **Negative music** | Controls for “any structured music” vs positive valence |
| **Tones** | Within-subject neutral baseline |
| **Positive / negative non-music** | Domain-specificity of affective auditory effects |
"""
        )
        mo.vstack(
            [
                _fig_board,
                _how,
                mo.md(
                    key_insight_card(
                        "A clear picture: stimulus identity → contrast → model weight.",
                        "We do not ask “does music work?” We ask which **valence/domain** moves BOLD, "
                        "whether that pattern is **music-specific**, and whether classifiers recover "
                        "those same dimensions — with confusion matrices showing residual confusions.",
                    )
                ),
            ]
        )
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## Limits (read before quoting accuracy)

- Whole-brain means collapse spatial specificity (no true limbic ROI here).
- BOLD subject n is small → LOOCV accuracy is **exploratory**.
- Coherence is an early/late proxy, not seed-based connectivity.
- Correlated music contrasts inflate coefficient variance — prefer RF ranks + domain knowledge.
- Response events are excluded from spectral epochs by design.

## Multivariate takeaways

1. **Feature inventory → correlation → classifiers → explainability** is the full loop.
2. **Confusion matrices** make misclassifications (e.g. music ↔ nonmusic) visible.
3. **Logistic coefficients** and **RF importances** should align with univariate music maps.
4. **Positive music vs tones** is the primary clinical/RecSys contrast in this book.
"""
            ),
            mo.md(
                clinical_relevance_card(
                    "If a patient’s confusion pattern collapses positive and negative music, or music and non-music, "
                    "playlist systems should not optimise a single ‘music score’ — they should personalise by valence "
                    "and spectral engagement features the models ranked as informative."
                )
            ),
            mo.md(book_nav("03_eda_multivariate")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
