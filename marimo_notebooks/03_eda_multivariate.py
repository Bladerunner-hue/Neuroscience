"""Chapter III — Algorithm lab: bake-off winners, confusion, explainability, RecSys."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="III · Algorithm Lab")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        NONMUSIC_COLOR,
        HIGHLIGHT,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        hypothesis_card,
        key_insight_card,
        load_condition_features,
        load_ml_bakeoff,
        load_spatial_connectivity,
        load_spectral_features,
        load_subject_features,
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
        key_insight_card,
        load_condition_features,
        load_ml_bakeoff,
        load_spatial_connectivity,
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
# III · Algorithm Lab

### Explore solvers → pick the best → explain for RecSys

| Original limit | Solution shipped here |
|---|---|
| Whole-brain means lose space | **Spatial pseudo-ROIs** (A/P, L/R, S/I) + anterior condition means |
| Small BOLD *n* | **15 subjects / 33 runs** downloaded from OpenNeuro; model at condition & run grain |
| Early/late self-coherence only | **Seed-style A–P / L–R / S–I** band coherence |
| Collinear music contrasts | **|r|>0.9 prune** + RF ranks + L1 / Elastic-Net |
| Which algorithm? | **13-model LOOCV bake-off** ranked by **macro-F1** |

Primary clinical / RecSys contrast: **positive music − tones**.
"""
    )
    mo.md(data_provenance_md())
    mo.md(
        hypothesis_card(
            "No single favourite classifier — the bake-off picks the best per target.",
            "Domain often favours tree ensembles under collinearity; valence and task may stay linear.",
        )
    )
    return


@app.cell
def _(
    load_condition_features,
    load_ml_bakeoff,
    load_spatial_connectivity,
    load_spectral_features,
    load_subject_features,
    mo,
    pd,
):
    runs = load_spectral_features()
    cond = load_condition_features()
    subj = load_subject_features()
    conn = load_spatial_connectivity()
    bake = load_ml_bakeoff()
    mo.md("## 0. Expanded feature store + precomputed bake-off")
    _inv = pd.DataFrame(
        [
            {
                "table": "spectral_features",
                "n": len(runs),
                "subjects": int(runs.subject.nunique()) if len(runs) else 0,
            },
            {
                "table": "condition_features",
                "n": len(cond),
                "subjects": int(cond.subject.nunique()) if len(cond) else 0,
            },
            {
                "table": "subject_features",
                "n": len(subj),
                "subjects": int(subj.subject.nunique()) if len(subj) else 0,
            },
            {
                "table": "spatial_connectivity",
                "n": len(conn),
                "subjects": int(conn.subject.nunique()) if len(conn) else 0,
            },
        ]
    )
    _win = bake.get("winners", {})
    mo.vstack(
        [
            mo.ui.table(_inv),
            mo.md(
                f"**Protocol:** `{bake.get('protocol', 'LOOCV bake-off')}`  \n"
                f"**Subjects / runs / condition rows:** "
                f"{bake.get('n_subjects', '?')} / {bake.get('n_runs', '?')} / {bake.get('n_condition_rows', '?')}  \n"
                f"**Winners:** "
                + (
                    ", ".join(f"`{k}`→**{v}**" for k, v in _win.items())
                    if _win
                    else "*re-run offline bake-off*"
                )
            ),
        ]
    )
    return bake, cond, conn, runs, subj


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    cond,
    conn,
    key_insight_card,
    mo,
    np,
    plt,
    runs,
):
    mo.md(
        r"""
## 1. Spatial solutions — pseudo-ROIs & seed-style coherence

Brain mask = high temporal variance. Seeds = anterior/posterior, left/right, superior/inferior slabs.
Coupling = magnitude-squared coherence (0.03–0.10 Hz) between seeds — a transparent substitute for atlas limbic ROIs.
"""
    )
    _stack = []
    if not conn.empty:
        _fig_co, _axes_co = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for _ax, _metric, _title in zip(
            _axes_co,
            ["coh_ant_post", "coh_left_right", "coh_sup_inf"],
            ["Anterior ↔ Posterior", "Left ↔ Right", "Superior ↔ Inferior"],
        ):
            _x = np.arange(2)
            for _i, (_g, _c) in enumerate(
                [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]
            ):
                _vals = [
                    float(
                        conn[(conn.group == _g) & (conn.task == _t)][_metric].mean()
                    )
                    if len(conn[(conn.group == _g) & (conn.task == _t)])
                    else np.nan
                    for _t in ["music", "nonmusic"]
                ]
                _ax.bar(
                    _x + (_i - 0.5) * 0.35,
                    _vals,
                    0.35,
                    label=_g,
                    color=_c,
                    edgecolor="white",
                )
            _ax.set_xticks(_x)
            _ax.set_xticklabels(["music", "nonmusic"])
            _ax.set_title(_title)
            _ax.legend(frameon=False, fontsize=8)
            _ax.grid(True, axis="y", alpha=0.3)
        _fig_co.suptitle("Seed-style band coherence by group × task", y=1.02)
        _fig_co.tight_layout()
        _stack.append(_fig_co)

    if not runs.empty and "power_high_anterior" in runs.columns:
        _fig_sp, _ax_sp = plt.subplots(figsize=(9, 4))
        _regs = ["anterior", "posterior", "left", "right"]
        _cols = [f"power_high_{r}" for r in _regs if f"power_high_{r}" in runs.columns]
        _labs = [c.replace("power_high_", "") for c in _cols]
        _x = np.arange(len(_labs))
        for _i, (_g, _c) in enumerate(
            [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]
        ):
            _vals = [float(runs.loc[runs.group == _g, c].mean()) for c in _cols]
            _ax_sp.bar(
                _x + (_i - 0.5) * 0.35,
                _vals,
                0.35,
                label=_g,
                color=_c,
                edgecolor="white",
            )
        _ax_sp.set_xticks(_x)
        _ax_sp.set_xticklabels(_labs)
        _ax_sp.set_ylabel("High-band power fraction")
        _ax_sp.set_title("Spatial high-band energy (pseudo-ROIs)")
        _ax_sp.legend(frameon=False)
        _ax_sp.grid(True, axis="y", alpha=0.3)
        _fig_sp.tight_layout()
        _stack.append(_fig_sp)

    if not cond.empty and "anterior_mean_bold" in cond.columns:
        _order = [
            "positive_music",
            "negative_music",
            "tones",
            "positive_nonmusic",
            "negative_nonmusic",
        ]
        _stims = [s for s in _order if s in set(cond.trial_type)]
        _fig_ant, _ax_ant = plt.subplots(figsize=(9, 4))
        _x = np.arange(len(_stims))
        for _i, (_g, _c) in enumerate(
            [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]
        ):
            _vals = []
            for _s in _stims:
                _sub = cond[(cond.group == _g) & (cond.trial_type == _s)]
                _vals.append(
                    float(_sub.anterior_mean_bold.mean()) if len(_sub) else np.nan
                )
            _ax_ant.bar(
                _x + (_i - 0.5) * 0.35,
                _vals,
                0.35,
                label=_g,
                color=_c,
                edgecolor="white",
            )
        _ax_ant.axhline(0, color="#999", lw=0.8)
        _ax_ant.set_xticks(_x)
        _ax_ant.set_xticklabels(
            [s.replace("_", "\n") for s in _stims], fontsize=8
        )
        _ax_ant.set_ylabel("Anterior slab mean BOLD (z)")
        _ax_ant.set_title("Anterior pseudo-ROI by stimulus")
        _ax_ant.legend(frameon=False)
        _fig_ant.tight_layout()
        _stack.append(_fig_ant)

    _stack.append(
        mo.md(
            key_insight_card(
                "Spatial proxies restore the dimension whole-brain means discarded.",
                "Anterior slab means under positive music feed the RecSys responder score.",
            )
        )
    )
    mo.vstack(_stack)
    return


@app.cell
def _(
    key_insight_card,
    mo,
    np,
    pd,
    plt,
    subj,
):
    mo.md(
        r"""
## 2. Multicollinearity among music effects

Correlated contrasts inflate logistic coefficient variance. We inspect the core matrix and prefer **RF ranks** + pruned linear models.
"""
    )
    _core = [
        c
        for c in [
            "pos_music_vs_tones_bold",
            "neg_music_vs_tones_bold",
            "pos_music_vs_neg_music_bold",
            "music_vs_nonmusic_bold",
            "pos_music_vs_tones_power_high",
            "pos_music_vs_tones_anterior",
            "responder_score",
            "run_power_high_mean",
            "coh_ant_post_mean",
            "music_task_vs_nonmusic_power_high",
        ]
        if c in subj.columns
    ]
    _blocks = []
    if subj.empty or len(_core) < 2:
        _blocks.append(mo.md("*Need subject features.*"))
    else:
        _X = subj[_core].apply(pd.to_numeric, errors="coerce")
        _corr = _X.corr()
        _fig_c, _ax_c = plt.subplots(figsize=(8, 6.5))
        _im = _ax_c.imshow(
            _corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )
        _ax_c.set_xticks(range(len(_core)))
        _ax_c.set_yticks(range(len(_core)))
        _labs = [c.replace("_", "\n") for c in _core]
        _ax_c.set_xticklabels(_labs, fontsize=6, rotation=45, ha="right")
        _ax_c.set_yticklabels(_labs, fontsize=6)
        _ax_c.set_title("Core music-effect correlations")
        for _i in range(len(_core)):
            for _j in range(len(_core)):
                _v = _corr.values[_i, _j]
                if np.isfinite(_v):
                    _ax_c.text(
                        _j,
                        _i,
                        f"{_v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if abs(_v) > 0.6 else "black",
                    )
        _fig_c.colorbar(_im, ax=_ax_c, fraction=0.046)
        _fig_c.tight_layout()
        _blocks.extend(
            [
                _fig_c,
                mo.md(
                    key_insight_card(
                        "Prune high collinearity before reading linear coefficients.",
                        "Trees still see the full set; linear winners use L1 / Elastic-Net and pruned features.",
                    )
                ),
            ]
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(MUSIC_COLOR, bake, key_insight_card, mo, pd, plt):
    _blocks = [
        mo.md(
            r"""
## 3. Algorithm bake-off — winners by LOOCV macro-F1

**Zoo:** LogReg L2 / L1 / Elastic-Net · Ridge · Linear SVM · RBF SVM · RandomForest · ExtraTrees · GBM · kNN · GaussianNB · LDA · DecisionTree  

**Ranking:** macro-F1 → balanced accuracy → accuracy (leave-one-out).
"""
        )
    ]
    best_summary = pd.DataFrame()
    if not bake or "tasks" not in bake:
        _blocks.append(
            mo.md(
                "*No precomputed bake-off — run the offline bake-off script / prepare pipeline.*"
            )
        )
    else:
        _figs = []
        _sum = []
        for _task, _tdata in bake["tasks"].items():
            _lb = pd.DataFrame(_tdata.get("leaderboard", []))
            if not len(_lb):
                continue
            _ok = _lb[_lb.status == "ok"].head(10) if "status" in _lb else _lb.head(10)
            if not len(_ok):
                continue
            _fig_b, _ax_b = plt.subplots(figsize=(9, 4.0))
            _colors = [
                MUSIC_COLOR if i == 0 else "#5D6D7E" for i in range(len(_ok))
            ]
            _ax_b.barh(
                _ok["model"][::-1],
                _ok["f1_macro"][::-1],
                color=_colors[::-1],
                edgecolor="white",
            )
            _ax_b.set_xlabel("LOOCV macro-F1")
            _best = _tdata.get("best", _ok.iloc[0]["model"])
            _ax_b.set_title(f"Bake-off · {_task} · winner = {_best}")
            _ax_b.set_xlim(0, 1.05)
            _ax_b.grid(True, axis="x", alpha=0.3)
            _fig_b.tight_layout()
            _figs.append(_fig_b)
            _row0 = _ok.iloc[0]
            _sum.append(
                {
                    "target": _task,
                    "best_model": _best,
                    "f1_macro": _row0.get("f1_macro"),
                    "bal_acc": _row0.get("bal_acc"),
                    "acc": _row0.get("acc"),
                    "n": _row0.get("n"),
                }
            )
        best_summary = pd.DataFrame(_sum)
        _domain_lb = pd.DataFrame(
            bake["tasks"].get("domain", {}).get("leaderboard", [])
        )
        _group_lb = pd.DataFrame(
            bake["tasks"].get("group", {}).get("leaderboard", [])
        )
        _blocks.extend(
            _figs
            + [
                mo.md("### Winners"),
                mo.ui.table(best_summary),
                mo.md("### Full domain leaderboard"),
                mo.ui.table(_domain_lb),
                mo.md("### Full group leaderboard"),
                mo.ui.table(_group_lb),
                mo.md(
                    key_insight_card(
                        "Best algorithm is empirical per target.",
                        "Re-run the offline bake-off when *n* or features change — do not hard-code a favourite model.",
                    )
                ),
            ]
        )
    mo.vstack(_blocks)
    return (best_summary,)


@app.cell
def _(bake, key_insight_card, mo, np, plt):
    _blocks = [
        mo.md(
            r"""
## 4. Confusion matrices for winning models

Off-diagonals answer: *does the brain fingerprint collapse valence or domain?*
"""
        )
    ]
    if not bake or "tasks" not in bake:
        _blocks.append(mo.md("*No bake-off confusion matrices.*"))
    else:
        _tasks = [
            t
            for t in ["domain", "valence", "task", "group"]
            if t in bake["tasks"] and bake["tasks"][t].get("confusion")
        ]
        _n = len(_tasks)
        if _n == 0:
            _blocks.append(mo.md("*No confusion matrices stored.*"))
        else:
            _fig_all, _axes = plt.subplots(1, _n, figsize=(4.3 * _n, 4.2))
            if _n == 1:
                _axes = [_axes]
            _reports = []
            for _ax, _task in zip(_axes, _tasks):
                _td = bake["tasks"][_task]
                _cm = np.array(_td["confusion"])
                _labels = _td.get("labels", [])
                _best = _td.get("best", "?")
                _ax.imshow(_cm, interpolation="nearest", cmap="Blues")
                _ax.set_title(f"{_task}\n{_best}", fontsize=10)
                _ax.set_xticks(range(len(_labels)))
                _ax.set_yticks(range(len(_labels)))
                _ax.set_xticklabels(_labels, fontsize=8)
                _ax.set_yticklabels(_labels, fontsize=8)
                _ax.set_xlabel("Predicted")
                _ax.set_ylabel("True")
                _th = _cm.max() / 2.0 if _cm.size and _cm.max() > 0 else 0.5
                for _i in range(_cm.shape[0]):
                    for _j in range(_cm.shape[1]):
                        _ax.text(
                            _j,
                            _i,
                            int(_cm[_i, _j]),
                            ha="center",
                            va="center",
                            color="white" if _cm[_i, _j] > _th else "black",
                            fontsize=12,
                            fontweight="bold",
                        )
                # simple metrics
                _diag = float(np.trace(_cm))
                _tot = float(_cm.sum()) or 1.0
                _reports.append(
                    f"- **{_task}** (`{_best}`): accuracy ≈ {_diag/_tot:.1%} on LOOCV"
                )
            _fig_all.suptitle("Confusion matrices — bake-off winners", y=1.02)
            _fig_all.tight_layout()
            _blocks.extend(
                [
                    _fig_all,
                    mo.md("\n".join(_reports)),
                    mo.md(
                        key_insight_card(
                            "Off-diagonals drive playlist logic.",
                            "If positive ↔ negative confuses valence, do **not** optimise a single music score — "
                            "split by valence and spectral engagement.",
                        )
                    ),
                ]
            )
    mo.vstack(_blocks)
    return


@app.cell
def _(MUSIC_COLOR, bake, key_insight_card, mo, np, pd, plt):
    _blocks = [
        mo.md(
            r"""
## 5. Explainability — RF ranks (collinearity-robust)

Reference **Random Forest importances** for each target (always shown, even when the winner is linear).
Align top features with univariate maps: high-band power, anterior means, peaks.
"""
        )
    ]
    if not bake or "tasks" not in bake:
        _blocks.append(mo.md("*No importances.*"))
    else:
        _expl = []
        for _task in ["domain", "valence", "task", "group"]:
            if _task not in bake["tasks"]:
                continue
            _td = bake["tasks"][_task]
            _imp = _td.get("rf_importance") or {}
            if not _imp:
                continue
            _items = sorted(_imp.items(), key=lambda kv: kv[1])
            _names = [k for k, _ in _items]
            _vals = [v for _, v in _items]
            _fig, _ax = plt.subplots(figsize=(8, max(3.0, 0.28 * len(_names))))
            _ax.barh(_names, _vals, color=MUSIC_COLOR, edgecolor="white")
            _ax.set_xlabel("RF importance")
            _ax.set_title(
                f"{_task} · winner={_td.get('best')} · RF reference ranks"
            )
            _fig.tight_layout()
            _tbl = (
                pd.DataFrame({"feature": list(_imp.keys()), "importance": list(_imp.values())})
                .sort_values("importance", ascending=False)
                .round(4)
            )
            _expl.extend(
                [
                    mo.md(f"### `{_task}`"),
                    _fig,
                    mo.ui.table(_tbl.head(10)),
                ]
            )
            if _td.get("logreg_coef"):
                _coef = _td["logreg_coef"]
                _citems = sorted(_coef.items(), key=lambda kv: abs(kv[1]))
                _fig2, _ax2 = plt.subplots(
                    figsize=(8, max(3.0, 0.28 * len(_citems)))
                )
                _ax2.barh(
                    [k for k, _ in _citems],
                    [v for _, v in _citems],
                    color=["#196F3D" if v > 0 else "#7F8C8D" for _, v in _citems],
                    edgecolor="white",
                )
                _ax2.axvline(0, color="#333", lw=0.8)
                _ax2.set_title(f"{_task} · linear winner coefficients")
                _fig2.tight_layout()
                _expl.append(_fig2)
        _expl.append(
            mo.md(
                key_insight_card(
                    "Explainability should echo the univariate music story.",
                    "If RF ranks `anterior_mean_bold`, `power_high`, or peaks highly, models agree with Chapter II.",
                )
            )
        )
        _blocks.extend(_expl)
    mo.vstack(_blocks)
    return


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    key_insight_card,
    mo,
    np,
    pd,
    plt,
    subj,
):
    _blocks = [
        mo.md(
            r"""
## 6. RecSys responder fingerprints

\[
R = \mathrm{mean}\big(\text{pos music} - \text{tones},\;
\text{music} - \text{nonmusic},\;
\text{pos music ant.} - \text{tones ant.}\big)
\]

High \(R\) → prioritise **positive, high-engagement** playlists.  
Low \(R\) → do **not** treat music as a uniform therapy — explore valence and non-music carefully.
"""
        )
    ]
    if subj.empty or "responder_score" not in subj.columns:
        _blocks.append(
            mo.md("*No responder_score — re-run prepare_real_features.py.*")
        )
    else:
        _s = subj.dropna(subset=["responder_score"]).sort_values(
            "responder_score"
        )
        _fig_r, _axes_r = plt.subplots(1, 2, figsize=(11, 4.5))
        _cols = [
            CONTROL_COLOR if g == "Control" else MDD_COLOR for g in _s["group"]
        ]
        _axes_r[0].barh(
            _s["subject"].str.replace("sub-", ""),
            _s["responder_score"],
            color=_cols,
            edgecolor="white",
        )
        _axes_r[0].axvline(0, color="#333", lw=0.8)
        _axes_r[0].set_xlabel("Responder score R")
        _axes_r[0].set_title("Individual music-reward fingerprints")

        if (
            "pos_music_vs_tones_bold" in subj.columns
            and "music_vs_nonmusic_bold" in subj.columns
        ):
            for _g, _c in [("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)]:
                _sg = subj[subj.group == _g]
                _sz = 80 + 180 * np.abs(
                    _sg["responder_score"].fillna(0).values
                )
                _axes_r[1].scatter(
                    _sg["pos_music_vs_tones_bold"],
                    _sg["music_vs_nonmusic_bold"],
                    s=np.clip(_sz, 50, 260),
                    c=_c,
                    label=_g,
                    edgecolors="white",
                    alpha=0.9,
                )
                for _, _row in _sg.iterrows():
                    _axes_r[1].annotate(
                        str(_row["subject"]).replace("sub-", ""),
                        (
                            _row["pos_music_vs_tones_bold"],
                            _row["music_vs_nonmusic_bold"],
                        ),
                        fontsize=7,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )
            _axes_r[1].axhline(0, color="#999", lw=0.7)
            _axes_r[1].axvline(0, color="#999", lw=0.7)
            _axes_r[1].set_xlabel("Positive music − tones")
            _axes_r[1].set_ylabel("Music − nonmusic")
            _axes_r[1].set_title("Playlist geometry (size ∝ |R|)")
            _axes_r[1].legend(frameon=False)
        _fig_r.tight_layout()

        _med = float(_s["responder_score"].median())
        _s2 = _s.copy()
        _s2["playlist_prior"] = np.where(
            _s2["responder_score"] >= _med,
            "A · positive / high-engagement music",
            "B · valence-cautious / explore non-music",
        )
        _show = _s2[
            [
                c
                for c in [
                    "subject",
                    "group",
                    "responder_score",
                    "pos_music_vs_tones_bold",
                    "music_vs_nonmusic_bold",
                    "pos_music_vs_tones_anterior",
                    "playlist_prior",
                ]
                if c in _s2.columns
            ]
        ].round(3)
        _blocks.extend(
            [
                _fig_r,
                mo.ui.table(_show),
                mo.md(
                    f"""
### Playlist policy sketch

| Prior | Rule | Cut |
|---|---|---|
| **A · Engage** | Positive music, high spectral energy | R ≥ median ({_med:+.3f}) |
| **B · Cautious** | Do not assume “music helps”; test valence & non-music | R < median |

Spectral **responder fingerprints** inform personalised playlists that target reward engagement rather than genre labels alone.
"""
                ),
                mo.md(
                    key_insight_card(
                        "Fingerprints beat genre tags for reward engagement.",
                        "Collapsed valence/domain confusion patterns force multi-axis personalisation.",
                    )
                ),
            ]
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(best_summary, clinical_relevance_card, key_insight_card, mo, pd):
    mo.md("## 7. Solution checklist vs original limits")
    _chk = pd.DataFrame(
        [
            {
                "limit": "Whole-brain means collapse space",
                "solution": "Pseudo-ROI slabs + anterior condition means + spatial high-band",
                "status": "addressed",
            },
            {
                "limit": "Small BOLD subject n",
                "solution": "15 subjects / 33 runs from OpenNeuro; condition & run LOOCV",
                "status": "improved (still exploratory)",
            },
            {
                "limit": "Early/late coherence only",
                "solution": "A–P / L–R / S–I seed-style MSC by task",
                "status": "addressed",
            },
            {
                "limit": "Collinear music contrasts",
                "solution": "|r| prune + L1/Elastic-Net + RF reference ranks",
                "status": "addressed",
            },
            {
                "limit": "Which algorithm?",
                "solution": "13-model LOOCV bake-off by macro-F1",
                "status": "addressed — winners §3",
            },
            {
                "limit": "Response events in spectra",
                "solution": "Excluded by design (button-press windows)",
                "status": "by design",
            },
        ]
    )
    _txt = ""
    if best_summary is not None and len(best_summary):
        _lines = [
            f"- **{r.target}** → `{r.best_model}` (F1={r.f1_macro}, bal_acc={r.bal_acc}, n={r.n})"
            for r in best_summary.itertuples()
        ]
        _txt = "### Best algorithms this run\n\n" + "\n".join(_lines)
    mo.vstack(
        [
            mo.ui.table(_chk),
            mo.md(_txt) if _txt else mo.md(""),
            mo.md(
                key_insight_card(
                    "Full loop: inventory → correlation → bake-off → confusion → explain → RecSys.",
                    "Positive music vs tones remains the primary clinical contrast; algorithms rank which fingerprint features carry that signal cleanly.",
                )
            ),
            mo.md(
                clinical_relevance_card(
                    "If a patient’s confusion pattern collapses positive and negative music, or music and non-music, "
                    "playlist systems should not optimise a single ‘music score’ — they should personalise by valence "
                    "and spectral engagement features the winning models ranked as informative."
                )
            ),
        ]
    )
    return


@app.cell
def _(book_nav, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## Multivariate takeaways

1. **Spatial proxies + more BOLD** shrink the whole-brain / small-*n* caveats.  
2. **Bake-off → best model** is re-runnable; winners differ by target.  
3. **Confusion matrices** show whether valence/domain collapse — that drives RecSys policy.  
4. **RF ranks + pruned / L1 linear models** handle collinear music contrasts.  
5. **Responder score \(R\)** turns primary contrasts into a playlist prior.  
6. **Neural nets (local TensorFlow):** STFT CNN + MLPs in `06_tf_spectrogram_model.py` — compare F1 to winners above before promoting a deep model into RecSys.
"""
            ),
            mo.md(book_nav("03_eda_multivariate")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
