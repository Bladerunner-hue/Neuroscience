#!/usr/bin/env python3
"""Offline LOOCV algorithm bake-off → data/processed/ml_bakeoff.json + book_bundle.

Run after prepare_real_features.py:
  python scripts/run_ml_bakeoff.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"


def zoo(seed=42):
    return {
        "LogReg-L2": LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="lbfgs"
        ),
        "LogReg-L1": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            l1_ratio=1.0,
            C=1.0,
        ),
        "LogReg-ElasticNet": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            l1_ratio=0.5,
        ),
        "Ridge": RidgeClassifier(class_weight="balanced"),
        "LinearSVM": LinearSVC(
            class_weight="balanced", max_iter=5000, dual="auto"
        ),
        "RBF-SVM": SVC(kernel="rbf", class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=seed, class_weight="balanced"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150, random_state=seed, class_weight="balanced"
        ),
        "GBM": GradientBoostingClassifier(random_state=seed),
        "kNN-5": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "DecisionTree": DecisionTreeClassifier(
            random_state=seed, class_weight="balanced", max_depth=4
        ),
    }


def loo_eval(X, y, models):
    y = np.asarray(y)
    rows, preds = [], {}
    if len(np.unique(y)) < 2 or len(y) < 4:
        return rows, preds
    for name, clf in models.items():
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler()),
                ("clf", clf),
            ]
        )
        try:
            yhat = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
            rows.append(
                {
                    "model": name,
                    "n": int(len(y)),
                    "acc": round(float(accuracy_score(y, yhat)), 3),
                    "bal_acc": round(float(balanced_accuracy_score(y, yhat)), 3),
                    "f1_macro": round(
                        float(f1_score(y, yhat, average="macro", zero_division=0)),
                        3,
                    ),
                    "status": "ok",
                }
            )
            preds[name] = yhat.tolist()
        except Exception as e:
            rows.append(
                {
                    "model": name,
                    "n": int(len(y)),
                    "acc": None,
                    "bal_acc": None,
                    "f1_macro": None,
                    "status": str(e)[:80],
                }
            )
    rows = sorted(
        rows,
        key=lambda r: (
            r["f1_macro"] is not None,
            r["f1_macro"] or -1,
            r["bal_acc"] or -1,
        ),
        reverse=True,
    )
    return rows, preds


def rf_importance(X, y, feat_names):
    pipe = Pipeline(
        [
            ("imp", SimpleImputer()),
            ("sc", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, random_state=0, class_weight="balanced"
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return {
        k: round(float(v), 4)
        for k, v in zip(feat_names, pipe.named_steps["clf"].feature_importances_)
    }


def main() -> None:
    cond = pd.read_csv(OUT / "condition_features.csv")
    runs = pd.read_csv(OUT / "spectral_features.csv")
    subj = pd.read_csv(OUT / "subject_features.csv")
    tasks = {}

    cfeats = [
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
            "anterior_mean_bold",
        ]
        if c in cond.columns
    ]
    Xc = cond[cfeats].apply(pd.to_numeric, errors="coerce").values
    for target, y in [
        ("domain", cond["domain"].astype(str).values),
        ("valence", cond["valence"].astype(str).values),
    ]:
        rows, preds = loo_eval(Xc, y, zoo())
        best = rows[0]["model"] if rows and rows[0]["status"] == "ok" else None
        labels = sorted(set(y))
        cm = (
            confusion_matrix(y, preds[best], labels=labels).tolist()
            if best
            else None
        )
        tasks[target] = {
            "leaderboard": rows,
            "best": best,
            "labels": labels,
            "confusion": cm,
            "features": cfeats,
            "rf_importance": rf_importance(Xc, y, cfeats),
            "y_true": y.tolist(),
            "y_pred": preds.get(best),
        }

    rfeats = [
        c
        for c in [
            "power_low",
            "power_mid",
            "power_high",
            "spectral_centroid",
            "peak_amp",
            "peak_latency_s",
            "coh_ant_post",
            "coh_left_right",
            "ant_minus_post_mean",
            "power_high_anterior",
            "power_high_posterior",
        ]
        if c in runs.columns
    ]
    Xr = runs[rfeats].apply(pd.to_numeric, errors="coerce").values
    yr = runs["task"].astype(str).values
    rows, preds = loo_eval(Xr, yr, zoo())
    best = rows[0]["model"] if rows and rows[0]["status"] == "ok" else None
    labels = sorted(set(yr))
    cm = (
        confusion_matrix(yr, preds[best], labels=labels).tolist() if best else None
    )
    tasks["task"] = {
        "leaderboard": rows,
        "best": best,
        "labels": labels,
        "confusion": cm,
        "features": rfeats,
        "rf_importance": rf_importance(Xr, yr, rfeats),
        "y_true": yr.tolist(),
        "y_pred": preds.get(best),
    }

    core = [
        c
        for c in [
            "pos_music_vs_tones_bold",
            "neg_music_vs_tones_bold",
            "pos_music_vs_neg_music_bold",
            "pos_music_vs_pos_nonmusic_bold",
            "music_vs_nonmusic_bold",
            "pos_music_vs_tones_power_high",
            "pos_music_vs_tones_anterior",
            "responder_score",
            "run_power_high_mean",
            "coh_ant_post_mean",
            "coh_left_right_mean",
            "ant_minus_post_mean",
            "music_task_vs_nonmusic_power_high",
            "music_task_vs_nonmusic_coh_ap",
            "age",
        ]
        if c in subj.columns
    ]
    Xg = subj[core].apply(pd.to_numeric, errors="coerce")
    corr = Xg.corr()
    pref = [
        "responder_score",
        "pos_music_vs_tones_bold",
        "music_vs_nonmusic_bold",
        "pos_music_vs_tones_anterior",
        "coh_ant_post_mean",
    ]
    drop = set()
    cols = list(Xg.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > 0.9:
                ia = pref.index(a) if a in pref else 99
                ib = pref.index(b) if b in pref else 99
                drop.add(b if ia <= ib else a)
    keep = [c for c in cols if c not in drop]
    Xgv = Xg[keep].values
    yg = subj["group"].astype(str).values
    rows, preds = loo_eval(Xgv, yg, zoo())
    best = rows[0]["model"] if rows and rows[0]["status"] == "ok" else None
    labels = sorted(set(yg))
    cm = (
        confusion_matrix(yg, preds[best], labels=labels).tolist() if best else None
    )
    coef = None
    if best and str(best).startswith("LogReg"):
        p2 = Pipeline(
            [
                ("imp", SimpleImputer()),
                ("sc", StandardScaler()),
                ("clf", zoo()[best]),
            ]
        )
        p2.fit(Xgv, yg)
        coef = {
            k: round(float(v), 4)
            for k, v in zip(keep, p2.named_steps["clf"].coef_.ravel())
        }
    tasks["group"] = {
        "leaderboard": rows,
        "best": best,
        "labels": labels,
        "confusion": cm,
        "features": keep,
        "rf_importance": rf_importance(Xgv, yg, keep),
        "logreg_coef": coef,
        "y_true": yg.tolist(),
        "y_pred": preds.get(best),
        "pruned_dropped": sorted(drop),
    }

    out = {
        "protocol": "LOOCV · impute · scale · rank by macro-F1 then bal_acc",
        "n_subjects": int(subj.subject.nunique()),
        "n_condition_rows": int(len(cond)),
        "n_runs": int(len(runs)),
        "winners": {k: v["best"] for k, v in tasks.items()},
        "tasks": tasks,
    }
    (OUT / "ml_bakeoff.json").write_text(json.dumps(out, indent=2))
    print("Winners:", out["winners"])
    for k, v in tasks.items():
        top = [(r["model"], r["f1_macro"]) for r in v["leaderboard"][:3]]
        print(k, "top3:", top)

    bundle_path = OUT / "book_bundle.json"
    if bundle_path.exists():
        b = json.loads(bundle_path.read_text())
        b["ml_bakeoff"] = out
        b["n_subjects_with_bold"] = int(subj.subject.nunique())
        b["n_bold_runs"] = int(len(runs))
        bundle_path.write_text(json.dumps(b))
        gen = ROOT / "scripts" / "gen_book_data.py"
        if gen.exists():
            subprocess.check_call([sys.executable, str(gen)], cwd=str(ROOT))
    print("Wrote", OUT / "ml_bakeoff.json")


if __name__ == "__main__":
    main()
