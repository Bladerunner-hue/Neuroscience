#!/usr/bin/env python3
"""Train TensorFlow models locally and write results for the public book page.

Produces:
  data/processed/tf_results.json   — metrics, confusion matrices, history, comparison
  data/processed/book_bundle.json  — embeds tf_results (via prepare or this script)
  marimo_notebooks/book_data.py    — regenerated if gen_book_data.py exists

The browser never loads TensorFlow; chapter 05 only visualizes this JSON.

  CUDA_VISIBLE_DEVICES= python scripts/run_tf_offline.py
  USE_GPU=1 python scripts/run_tf_offline.py   # optional GPU
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Default CPU for reliable CI / broken cuDNN hosts
if os.environ.get("USE_GPU", "0") not in ("1", "true", "TRUE", "yes"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    confusion_matrix,
)


def load_tables():
    ts = pd.read_csv(OUT / "bold_timeseries.csv")
    subj = pd.read_csv(OUT / "subject_features.csv")
    cond = pd.read_csv(OUT / "condition_features.csv")
    bake = {}
    if (OUT / "ml_bakeoff.json").exists():
        bake = json.loads((OUT / "ml_bakeoff.json").read_text())
    return ts, subj, cond, bake


def build_spectrograms(ts: pd.DataFrame):
    specs, y, tasks, subjects = [], [], [], []
    for (sid, run), g in ts.groupby(["subject", "run"]):
        sig = g.sort_values("time")["bold_z"].values.astype("float32")
        if len(sig) < 24:
            continue
        x = tf.convert_to_tensor(sig)
        x = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-6)
        stft = tf.signal.stft(
            x, frame_length=16, frame_step=4, fft_length=32, window_fn=tf.signal.hann_window
        )
        mag = tf.abs(stft).numpy()
        specs.append(mag)
        y.append(1 if str(g["group"].iloc[0]) == "MDD" else 0)
        tasks.append(str(g["task"].iloc[0]))
        subjects.append(sid)
    if not specs:
        return None
    T = int(np.median([s.shape[0] for s in specs]))
    F = int(np.median([s.shape[1] for s in specs]))
    stack = []
    for s in specs:
        out = np.zeros((T, F), dtype="float32")
        t, f = min(T, s.shape[0]), min(F, s.shape[1])
        out[:t, :f] = s[:t, :f]
        stack.append(out)
    X = np.expand_dims(np.stack(stack), -1)
    return {
        "X": X,
        "y": np.array(y, dtype="int32"),
        "task": np.array(tasks),
        "subject": np.array(subjects),
        "shape": list(X.shape[1:]),
    }


def train_cnn(spec: dict, epochs: int = 12):
    X, y, subjects = spec["X"], spec["y"], spec["subject"]
    uniq = np.unique(subjects)
    rng = np.random.RandomState(42)
    rng.shuffle(uniq)
    n_val = max(2, len(uniq) // 4)
    val_subs = set(uniq[:n_val])
    tr = np.array([s not in val_subs for s in subjects])
    va = ~tr
    if tr.sum() < 2 or va.sum() < 1:
        tr = np.ones(len(X), dtype=bool)
        tr[-2:] = False
        va = ~tr

    def build():
        inp = tf.keras.Input(shape=tuple(spec["shape"]))
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        m = tf.keras.Model(inp, out)
        m.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return m

    model = build()
    with tf.device("/CPU:0"):
        hist = model.fit(
            X[tr],
            y[tr],
            validation_data=(X[va], y[va]),
            epochs=epochs,
            batch_size=min(8, max(2, int(tr.sum()))),
            verbose=0,
        )
        prob = model.predict(X[va], verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    yt = y[va]
    cm = confusion_matrix(yt, pred, labels=[0, 1]).tolist()
    return {
        "name": "STFT Conv2D",
        "target": "group",
        "split": "subject_holdout",
        "val_subjects": sorted(val_subs),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "acc": round(float(accuracy_score(yt, pred)), 3),
        "f1_macro": round(float(f1_score(yt, pred, average="macro", zero_division=0)), 3),
        "labels": ["Control", "MDD"],
        "confusion": cm,
        "history": {
            "accuracy": [float(v) for v in hist.history.get("accuracy", [])],
            "val_accuracy": [float(v) for v in hist.history.get("val_accuracy", [])],
            "auc": [float(v) for v in hist.history.get("auc", [])],
            "val_auc": [float(v) for v in hist.history.get("val_auc", [])],
        },
        "spectrogram_shape": spec["shape"],
        "n_spectrograms": int(len(X)),
    }


def train_mlp_group(subj: pd.DataFrame, epochs: int = 40):
    feat_cols = [
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
            "coh_left_right_mean",
            "ant_minus_post_mean",
            "music_task_vs_nonmusic_power_high",
            "age",
        ]
        if c in subj.columns
    ]
    Xdf = subj[feat_cols].apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.mean())
    X = StandardScaler().fit_transform(Xdf.values.astype("float32"))
    y = (subj["group"].astype(str).values == "MDD").astype("int32")

    def build(n_in):
        m = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_in,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        m.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return m

    preds, trues = [], []
    with tf.device("/CPU:0"):
        for tr, te in LeaveOneOut().split(X):
            m = build(X.shape[1])
            m.fit(
                X[tr],
                y[tr],
                epochs=epochs,
                batch_size=min(8, max(2, len(tr))),
                verbose=0,
            )
            p = float(m.predict(X[te], verbose=0).ravel()[0])
            preds.append(1 if p >= 0.5 else 0)
            trues.append(int(y[te][0]))
    yt, yp = np.array(trues), np.array(preds)
    return {
        "name": "Dense MLP LOOCV",
        "target": "group",
        "split": "leave_one_out",
        "n": int(len(yt)),
        "features": feat_cols,
        "acc": round(float(accuracy_score(yt, yp)), 3),
        "f1_macro": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 3),
        "labels": ["Control", "MDD"],
        "confusion": confusion_matrix(yt, yp, labels=[0, 1]).tolist(),
        "y_true": yt.tolist(),
        "y_pred": yp.tolist(),
        "subjects": subj["subject"].astype(str).tolist(),
    }


def train_mlp_domain(cond: pd.DataFrame, epochs: int = 40):
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
    Xdf = cond[cfeats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = StandardScaler().fit_transform(Xdf.values.astype("float32"))
    domains = cond["domain"].astype(str).values
    classes = sorted(np.unique(domains))
    lab_map = {c: i for i, c in enumerate(classes)}
    y = np.array([lab_map[c] for c in domains], dtype="int32")
    subs = cond["subject"].astype(str).values if "subject" in cond.columns else None
    if subs is not None:
        uniq = np.unique(subs)
        rng = np.random.RandomState(0)
        rng.shuffle(uniq)
        n_val = max(2, len(uniq) // 4)
        val_set = set(uniq[:n_val])
        tr = np.array([s not in val_set for s in subs])
        va = ~tr
    else:
        idx = np.arange(len(X))
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        split = int(0.75 * len(idx))
        tr = np.zeros(len(X), dtype=bool)
        tr[idx[:split]] = True
        va = ~tr

    m = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(len(classes), activation="softmax"),
        ]
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    with tf.device("/CPU:0"):
        hist = m.fit(
            X[tr],
            y[tr],
            validation_data=(X[va], y[va]),
            epochs=epochs,
            batch_size=min(16, max(4, int(tr.sum()))),
            verbose=0,
        )
        yp = np.argmax(m.predict(X[va], verbose=0), axis=1)
    yt = y[va]
    return {
        "name": "Dense MLP domain",
        "target": "domain",
        "split": "subject_holdout",
        "n_val": int(va.sum()),
        "features": cfeats,
        "classes": classes,
        "acc": round(float(accuracy_score(yt, yp)), 3),
        "f1_macro": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 3),
        "confusion": confusion_matrix(yt, yp, labels=list(range(len(classes)))).tolist(),
        "history": {
            "accuracy": [float(v) for v in hist.history.get("accuracy", [])],
            "val_accuracy": [float(v) for v in hist.history.get("val_accuracy", [])],
        },
    }


def main() -> None:
    print("TensorFlow", tf.__version__, "devices", tf.config.list_physical_devices())
    ts, subj, cond, bake = load_tables()
    results = {
        "tensorflow_version": tf.__version__,
        "device": "CPU" if not tf.config.list_physical_devices("GPU") else "GPU",
        "protocol": "Local offline train → JSON → WASM chapter 05 (no TF in browser)",
        "n_subjects": int(subj["subject"].nunique()) if len(subj) else 0,
        "models": {},
        "comparison": [],
        "sklearn_winners": bake.get("winners", {}) if bake else {},
    }

    print("Building spectrograms…")
    spec = build_spectrograms(ts)
    if spec is not None:
        print(f"  {spec['X'].shape[0]} spectrograms shape={spec['shape']}")
        print("Training STFT CNN…")
        results["models"]["cnn_group"] = train_cnn(spec, epochs=12)
        print("  CNN", results["models"]["cnn_group"]["f1_macro"])
    else:
        print("  no spectrograms")

    if len(subj) and subj["group"].nunique() >= 2:
        print("Training subject MLP LOOCV…")
        results["models"]["mlp_group"] = train_mlp_group(subj, epochs=40)
        print("  MLP group", results["models"]["mlp_group"]["f1_macro"])

    if len(cond) and "domain" in cond.columns:
        print("Training domain MLP…")
        results["models"]["mlp_domain"] = train_mlp_domain(cond, epochs=40)
        print("  MLP domain", results["models"]["mlp_domain"]["f1_macro"])

    # Comparison table vs sklearn bake-off
    rows = []
    if bake and bake.get("tasks"):
        for t in ("group", "domain"):
            td = bake["tasks"].get(t, {})
            lb = td.get("leaderboard") or []
            if lb:
                rows.append(
                    {
                        "target": t,
                        "family": "sklearn",
                        "model": td.get("best", lb[0].get("model")),
                        "f1_macro": lb[0].get("f1_macro"),
                    }
                )
    for key, m in results["models"].items():
        rows.append(
            {
                "target": m.get("target"),
                "family": "tensorflow",
                "model": m.get("name"),
                "f1_macro": m.get("f1_macro"),
            }
        )
    results["comparison"] = rows
    # winners per target
    winners = {}
    for t in {r["target"] for r in rows if r.get("target")}:
        sub = [r for r in rows if r["target"] == t and r.get("f1_macro") is not None]
        if sub:
            best = max(sub, key=lambda r: r["f1_macro"])
            winners[t] = best
    results["recommended"] = winners

    path = OUT / "tf_results.json"
    path.write_text(json.dumps(results, indent=2))
    print("Wrote", path)

    # Embed into book_bundle for WASM
    bundle_path = OUT / "book_bundle.json"
    if bundle_path.exists():
        b = json.loads(bundle_path.read_text())
        b["tf_results"] = results
        bundle_path.write_text(json.dumps(b))
        gen = ROOT / "scripts" / "gen_book_data.py"
        if gen.exists():
            subprocess.check_call([sys.executable, str(gen)], cwd=str(ROOT))
            print("book_data.py regenerated with tf_results")
    print("Done.")


if __name__ == "__main__":
    main()
