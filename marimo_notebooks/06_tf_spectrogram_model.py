"""Chapter V — TensorFlow neural nets on real ds000171 features (local only)."""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scikit-learn", "tensorflow"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="V · TensorFlow Neural Nets")


@app.cell
def _():
    import os as _os

    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Default CPU: broken host cuDNN stacks are common. Opt-in GPU with USE_GPU=1.
    if _os.environ.get("USE_GPU", "0") not in ("1", "true", "TRUE", "yes"):
        _os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    gpus = tf.config.list_physical_devices("GPU")
    for _gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(_gpu, True)
        except Exception:
            pass
    TF_DEVICE = f"GPU×{len(gpus)}" if gpus else "CPU"

    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        hypothesis_card,
        key_insight_card,
        load_bold_timeseries,
        load_cleaned_spectral_features,
        load_condition_features,
        load_ml_bakeoff,
        load_spectral_features,
        load_subject_features,
        set_global_style,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        LeaveOneOut,
        MDD_COLOR,
        MUSIC_COLOR,
        TF_DEVICE,
        accuracy_score,
        book_nav,
        classification_report,
        clinical_relevance_card,
        confusion_matrix,
        data_provenance_md,
        f1_score,
        hypothesis_card,
        key_insight_card,
        load_bold_timeseries,
        load_cleaned_spectral_features,
        load_condition_features,
        load_ml_bakeoff,
        load_spectral_features,
        load_subject_features,
        mo,
        np,
        pd,
        plt,
        StandardScaler,
        tf,
    )


@app.cell
def _(TF_DEVICE, data_provenance_md, hypothesis_card, mo, tf):
    mo.md(
        f"""
# V · TensorFlow Neural Networks

### Local GPU/CPU chapter (not WASM)

TensorFlow **{tf.__version__}** · device: **{TF_DEVICE}**

This chapter extends the classical bake-off (Ch III) with **neural nets** on **real** OpenNeuro features:

| Model | Input | Target |
|---|---|---|
| **Conv2D + STFT** | Run-level real mean BOLD spectrograms | Control vs MDD |
| **Dense MLP** | Subject music-effect contrasts (+ age) | Control vs MDD |
| **Dense MLP** | Condition epoch features | Domain (music / nonmusic / tones) |

We compare net metrics to the **sklearn bake-off winners** so you see when deep models help on this *n*.
"""
    )
    mo.md(data_provenance_md())
    mo.md(
        hypothesis_card(
            "STFT ConvNets capture spectral structure that tabular models only see as summary bands.",
            "On small *n*, regularized MLPs may match RF; spectrogram CNNs need careful LOO-style splits by subject.",
        )
    )
    return


@app.cell
def _(
    load_bold_timeseries,
    load_cleaned_spectral_features,
    load_condition_features,
    load_ml_bakeoff,
    load_spectral_features,
    load_subject_features,
    mo,
    np,
    pd,
):
    ts = load_bold_timeseries()
    _all = load_spectral_features()
    _clean = load_cleaned_spectral_features()
    runs = _clean if _clean is not None and not getattr(_clean, "empty", True) else _all
    cond = load_condition_features()
    subj = load_subject_features()
    bake = load_ml_bakeoff()
    mo.md(
        "## 0. Real data inventory for TF  \n"
        f"Spectral runs after QC: **{len(runs)}** / {len(_all)} "
        "(prefer `cleaned_spectral_features.csv` from adaptive multitaper + IsolationForest)."
    )
    _inv = pd.DataFrame(
        [
            {"layer": "bold_timeseries runs", "n": ts.groupby(["subject", "run"]).ngroups if len(ts) else 0},
            {"layer": "spectral runs (clean)", "n": len(runs)},
            {"layer": "condition epochs", "n": len(cond)},
            {"layer": "subjects", "n": len(subj)},
        ]
    )
    _win = bake.get("winners", {}) if bake else {}
    mo.vstack(
        [
            mo.ui.table(_inv),
            mo.md(
                "**Sklearn bake-off winners (Ch III):** "
                + (
                    ", ".join(f"`{k}`→**{v}**" for k, v in _win.items())
                    if _win
                    else "*run scripts/run_ml_bakeoff.py*"
                )
            ),
        ]
    )
    return bake, cond, runs, subj, ts


@app.cell
def _(mo):
    epochs_cnn = mo.ui.slider(3, 40, value=10, step=1, label="CNN epochs")
    epochs_mlp = mo.ui.slider(10, 120, value=40, step=5, label="MLP epochs")
    lr = mo.ui.slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="Learning rate")
    dropout = mo.ui.slider(0.0, 0.5, value=0.3, step=0.05, label="Dropout")
    mo.md("## Training controls")
    mo.hstack([epochs_cnn, epochs_mlp, lr, dropout], justify="start")
    return dropout, epochs_cnn, epochs_mlp, lr


@app.cell
def _(
    key_insight_card,
    mo,
    np,
    pd,
    plt,
    runs,
    tf,
    ts,
):
    mo.md(
        r"""
## 1. STFT spectrograms from **real** run BOLD

Each run → z-scored whole-brain mean → `tf.signal.stft` magnitude image.
"""
    )
    _specs, _ylabels, _tasks, _subjects = [], [], [], []
    if not ts.empty:
        for (_sid, _run), _g in ts.groupby(["subject", "run"]):
            _sig = _g.sort_values("time")["bold_z"].values.astype("float32")
            if len(_sig) < 24:
                continue
            _x = tf.convert_to_tensor(_sig)
            _x = (_x - tf.reduce_mean(_x)) / (tf.math.reduce_std(_x) + 1e-6)
            _stft = tf.signal.stft(
                _x,
                frame_length=16,
                frame_step=4,
                fft_length=32,
                window_fn=tf.signal.hann_window,
            )
            _mag = tf.abs(_stft).numpy()
            _specs.append(_mag)
            _grp = str(_g["group"].iloc[0])
            _ylabels.append(1 if _grp == "MDD" else 0)
            _tasks.append(str(_g["task"].iloc[0]))
            _subjects.append(_sid)

    if not _specs:
        mo.md("*No timeseries for STFT — check data/processed/bold_timeseries.csv*")
        X_spec = np.zeros((0, 1, 1, 1), dtype="float32")
        y_spec = np.array([], dtype="int32")
        task_spec = np.array([])
        subj_spec = np.array([])
        spec_shape = (1, 1, 1)
    else:
        # pad/crop to common shape
        _T = int(np.median([s.shape[0] for s in _specs]))
        _F = int(np.median([s.shape[1] for s in _specs]))
        _stack = []
        for _s in _specs:
            _out = np.zeros((_T, _F), dtype="float32")
            _t = min(_T, _s.shape[0])
            _f = min(_F, _s.shape[1])
            _out[:_t, :_f] = _s[:_t, :_f]
            _stack.append(_out)
        X_spec = np.expand_dims(np.stack(_stack), -1)
        y_spec = np.array(_ylabels, dtype="int32")
        task_spec = np.array(_tasks)
        subj_spec = np.array(_subjects)
        spec_shape = tuple(X_spec.shape[1:])

        _fig_s, _axes_s = plt.subplots(1, 2, figsize=(10, 3.8))
        for _ax, _lab, _title in zip(
            _axes_s, [0, 1], ["Control example", "MDD example"]
        ):
            _idx = np.where(y_spec == _lab)[0]
            if len(_idx):
                _im = _axes_s[0 if _lab == 0 else 1].imshow(
                    X_spec[_idx[0], :, :, 0].T,
                    aspect="auto",
                    origin="lower",
                    cmap="magma",
                )
            _ax.set_title(_title)
            _ax.set_xlabel("Time frame")
            _ax.set_ylabel("Freq bin")
        _fig_s.suptitle("Real BOLD STFT spectrograms (first of each class)", y=1.02)
        _fig_s.tight_layout()
        mo.vstack(
            [
                mo.md(
                    f"**{len(X_spec)}** run spectrograms · shape `{spec_shape}` · "
                    f"MDD={int(y_spec.sum())} Control={int((1-y_spec).sum())}"
                ),
                _fig_s,
                mo.md(
                    key_insight_card(
                        "Spectrograms keep time–frequency structure that band scalars compress.",
                        "CNN can learn music-task spectral motifs without hand-built Welch bands.",
                    )
                ),
            ]
        )
    return X_spec, spec_shape, subj_spec, task_spec, y_spec


@app.cell
def _(
    X_spec,
    confusion_matrix,
    dropout,
    epochs_cnn,
    f1_score,
    key_insight_card,
    lr,
    mo,
    np,
    plt,
    accuracy_score,
    spec_shape,
    subj_spec,
    tf,
    y_spec,
):
    mo.md(
        r"""
## 2. ConvNet on spectrograms (subject-aware holdout)

We hold out **entire subjects** (not random runs) so validation is not inflated by within-subject leakage.
"""
    )
    cnn_metrics = {}
    hist_cnn = None
    if len(X_spec) < 6 or len(np.unique(y_spec)) < 2:
        mo.md("*Too few spectrograms for CNN training.*")
        model_cnn = None
        y_val_cnn = np.array([])
        y_pred_cnn = np.array([])
    else:
        _subjects = np.unique(subj_spec)
        _rng_cnn = np.random.RandomState(42)
        _rng_cnn.shuffle(_subjects)
        n_val_sub = max(2, len(_subjects) // 4)
        val_subs = set(_subjects[:n_val_sub])
        train_m = np.array([s not in val_subs for s in subj_spec])
        val_m = ~train_m
        if train_m.sum() < 2 or val_m.sum() < 1:
            train_m = np.ones(len(X_spec), dtype=bool)
            train_m[-2:] = False
            val_m = ~train_m

        X_tr, y_tr = X_spec[train_m], y_spec[train_m]
        X_va, y_va = X_spec[val_m], y_spec[val_m]

        def build_cnn(shape, drop, learning_rate):
            inp = tf.keras.Input(shape=shape, name="spectrogram")
            x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D()(x)
            x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPool2D()(x)
            x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(float(drop))(x)
            x = tf.keras.layers.Dense(32, activation="relu")(x)
            out = tf.keras.layers.Dense(1, activation="sigmoid", name="is_mdd")(x)
            m = tf.keras.Model(inp, out, name="bold_stft_cnn")
            m.compile(
                optimizer=tf.keras.optimizers.Adam(float(learning_rate)),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
            )
            return m

        model_cnn = build_cnn(spec_shape, dropout.value, lr.value)
        with tf.device("/CPU:0"):
            hist_cnn = model_cnn.fit(
                X_tr,
                y_tr,
                validation_data=(X_va, y_va),
                epochs=int(epochs_cnn.value),
                batch_size=min(8, max(2, len(X_tr))),
                verbose=0,
            )

        _prob = model_cnn.predict(X_va, verbose=0).ravel()
        y_pred_cnn = (_prob >= 0.5).astype(int)
        y_val_cnn = y_va
        cnn_metrics = {
            "val_acc": round(float(accuracy_score(y_va, y_pred_cnn)), 3),
            "val_f1": round(
                float(f1_score(y_va, y_pred_cnn, average="macro", zero_division=0)),
                3,
            ),
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "val_subjects": sorted(val_subs),
        }

        _fig_h, _ax_h = plt.subplots(figsize=(8, 3.5))
        if hist_cnn is not None:
            _ax_h.plot(hist_cnn.history.get("accuracy", []), label="train acc")
            _ax_h.plot(hist_cnn.history.get("val_accuracy", []), label="val acc")
            if "auc" in hist_cnn.history:
                _ax_h.plot(hist_cnn.history["auc"], label="train AUC", ls="--")
            if "val_auc" in hist_cnn.history:
                _ax_h.plot(hist_cnn.history["val_auc"], label="val AUC", ls="--")
        _ax_h.set_xlabel("Epoch")
        _ax_h.set_title("STFT CNN training curves")
        _ax_h.legend(frameon=False, fontsize=8)
        _ax_h.grid(True, alpha=0.3)
        _fig_h.tight_layout()

        _cm = confusion_matrix(y_va, y_pred_cnn, labels=[0, 1])
        _fig_cm, _ax_cm = plt.subplots(figsize=(4.2, 3.8))
        _ax_cm.imshow(_cm, cmap="Blues")
        _ax_cm.set_xticks([0, 1])
        _ax_cm.set_yticks([0, 1])
        _ax_cm.set_xticklabels(["Control", "MDD"])
        _ax_cm.set_yticklabels(["Control", "MDD"])
        _ax_cm.set_xlabel("Predicted")
        _ax_cm.set_ylabel("True")
        _ax_cm.set_title("CNN subject-holdout CM")
        for _i in range(2):
            for _j in range(2):
                _ax_cm.text(
                    _j,
                    _i,
                    int(_cm[_i, _j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if _cm[_i, _j] > _cm.max() / 2 else "black",
                )
        _fig_cm.tight_layout()

        mo.vstack(
            [
                mo.md(f"**CNN metrics:** `{cnn_metrics}`"),
                _fig_h,
                _fig_cm,
                mo.md(
                    key_insight_card(
                        "Subject-level holdout is the honest split for fMRI nets.",
                        "If val F1 is near chance, the spectrogram CNN needs more runs or ROI-rich inputs — "
                        "tabular MLPs on music contrasts may still win at this *n*.",
                    )
                ),
            ]
        )
    return cnn_metrics, hist_cnn, model_cnn, y_pred_cnn, y_val_cnn


@app.cell
def _(
    LeaveOneOut,
    StandardScaler,
    accuracy_score,
    confusion_matrix,
    dropout,
    epochs_mlp,
    f1_score,
    key_insight_card,
    lr,
    mo,
    np,
    pd,
    plt,
    subj,
    tf,
):
    mo.md(
        r"""
## 3. Dense MLP on subject music-effect features (LOOCV)

Tabular net competing with Ch III GBM/LogReg on the same **responder / contrast** space.
"""
    )
    mlp_group = {}
    if subj.empty or subj["group"].nunique() < 2:
        mo.md("*Need subject_features with both groups.*")
        loo_true_g = np.array([])
        loo_pred_g = np.array([])
    else:
        _feat_cols = [
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
        _Xdf = subj[_feat_cols].apply(pd.to_numeric, errors="coerce")
        _Xdf = _Xdf.fillna(_Xdf.mean())
        _X = StandardScaler().fit_transform(_Xdf.values.astype("float32"))
        _y = (subj["group"].astype(str).values == "MDD").astype("int32")

        def build_mlp(n_in, drop, learning_rate):
            m = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(n_in,)),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(float(drop)),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dropout(float(drop) * 0.5),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ],
                name="subject_mlp",
            )
            m.compile(
                optimizer=tf.keras.optimizers.Adam(float(learning_rate)),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            return m

        _preds = []
        _trues = []
        for _tr, _te in LeaveOneOut().split(_X):
            _m = build_mlp(_X.shape[1], dropout.value, lr.value)
            _m.fit(
                _X[_tr],
                _y[_tr],
                epochs=int(epochs_mlp.value),
                batch_size=min(8, max(2, len(_tr))),
                verbose=0,
            )
            _p = float(_m.predict(_X[_te], verbose=0).ravel()[0])
            _preds.append(1 if _p >= 0.5 else 0)
            _trues.append(int(_y[_te][0]))

        loo_true_g = np.array(_trues)
        loo_pred_g = np.array(_preds)
        mlp_group = {
            "acc": round(float(accuracy_score(loo_true_g, loo_pred_g)), 3),
            "f1_macro": round(
                float(
                    f1_score(loo_true_g, loo_pred_g, average="macro", zero_division=0)
                ),
                3,
            ),
            "n": int(len(loo_true_g)),
            "features": _feat_cols,
        }
        _cm = confusion_matrix(loo_true_g, loo_pred_g, labels=[0, 1])
        _fig, _ax = plt.subplots(figsize=(4.2, 3.8))
        _ax.imshow(_cm, cmap="Blues")
        _ax.set_xticks([0, 1])
        _ax.set_yticks([0, 1])
        _ax.set_xticklabels(["Control", "MDD"])
        _ax.set_yticklabels(["Control", "MDD"])
        _ax.set_title("MLP LOOCV · group")
        _ax.set_xlabel("Predicted")
        _ax.set_ylabel("True")
        for _i in range(2):
            for _j in range(2):
                _ax.text(
                    _j,
                    _i,
                    int(_cm[_i, _j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if _cm[_i, _j] > _cm.max() / 2 else "black",
                )
        _fig.tight_layout()
        mo.vstack(
            [
                mo.md(f"**Subject MLP LOOCV:** `{mlp_group}`"),
                mo.md(f"Features: `{_feat_cols}`"),
                _fig,
                mo.md(
                    key_insight_card(
                        "MLP LOOCV is the deep analogue of Ch III group models.",
                        "Compare F1 to GBM/LogReg winners — nets win only if they beat that bar without leakage.",
                    )
                ),
            ]
        )
    return loo_pred_g, loo_true_g, mlp_group


@app.cell
def _(
    StandardScaler,
    accuracy_score,
    cond,
    confusion_matrix,
    dropout,
    epochs_mlp,
    f1_score,
    key_insight_card,
    lr,
    mo,
    np,
    pd,
    plt,
    tf,
):
    mo.md(
        r"""
## 4. MLP for stimulus **domain** (music / nonmusic / tones)

Multi-class softmax on condition epoch features — same grain as the RF domain winner.
"""
    )
    mlp_domain = {}
    if cond.empty or "domain" not in cond.columns:
        mo.md("*No condition features.*")
    else:
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
                "anterior_mean_bold",
            ]
            if c in cond.columns
        ]
        _Xdf = cond[_cfeats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        _X = StandardScaler().fit_transform(_Xdf.values.astype("float32"))
        _domains = cond["domain"].astype(str).values
        _classes = sorted(np.unique(_domains))
        _lab_map = {c: i for i, c in enumerate(_classes)}
        _y = np.array([_lab_map[c] for c in _domains], dtype="int32")
        _n_cls = len(_classes)

        # subject-grouped split if subject present
        if "subject" in cond.columns:
            _subs = cond["subject"].astype(str).values
            _uniq = np.unique(_subs)
            _rng_dom = np.random.RandomState(0)
            _rng_dom.shuffle(_uniq)
            _n_val = max(2, len(_uniq) // 4)
            _val_set = set(_uniq[:_n_val])
            _tr = np.array([s not in _val_set for s in _subs])
            _va = ~_tr
        else:
            _idx = np.arange(len(_X))
            _rng_dom = np.random.RandomState(0)
            _rng_dom.shuffle(_idx)
            _split = int(0.75 * len(_idx))
            _tr = np.zeros(len(_X), dtype=bool)
            _tr[_idx[:_split]] = True
            _va = ~_tr

        def build_domain_mlp(n_in, n_cls, drop, learning_rate):
            m = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(n_in,)),
                    tf.keras.layers.Dense(48, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(float(drop)),
                    tf.keras.layers.Dense(24, activation="relu"),
                    tf.keras.layers.Dense(n_cls, activation="softmax"),
                ],
                name="domain_mlp",
            )
            m.compile(
                optimizer=tf.keras.optimizers.Adam(float(learning_rate)),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            return m

        _m = build_domain_mlp(_X.shape[1], _n_cls, dropout.value, lr.value)
        _hist = _m.fit(
            _X[_tr],
            _y[_tr],
            validation_data=(_X[_va], _y[_va]),
            epochs=int(epochs_mlp.value),
            batch_size=min(16, max(4, int(_tr.sum()))),
            verbose=0,
        )
        _yp = np.argmax(_m.predict(_X[_va], verbose=0), axis=1)
        _yt = _y[_va]
        mlp_domain = {
            "acc": round(float(accuracy_score(_yt, _yp)), 3),
            "f1_macro": round(
                float(f1_score(_yt, _yp, average="macro", zero_division=0)), 3
            ),
            "n_val": int(_va.sum()),
            "classes": _classes,
        }
        _cm = confusion_matrix(_yt, _yp, labels=list(range(_n_cls)))
        _fig_c, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
        _axes[0].plot(_hist.history.get("accuracy", []), label="train")
        _axes[0].plot(_hist.history.get("val_accuracy", []), label="val")
        _axes[0].set_title("Domain MLP accuracy")
        _axes[0].legend(frameon=False)
        _axes[0].grid(True, alpha=0.3)
        _axes[1].imshow(_cm, cmap="Blues")
        _axes[1].set_xticks(range(_n_cls))
        _axes[1].set_yticks(range(_n_cls))
        _axes[1].set_xticklabels(_classes, fontsize=8)
        _axes[1].set_yticklabels(_classes, fontsize=8)
        _axes[1].set_title("Domain MLP confusion")
        for _i in range(_n_cls):
            for _j in range(_n_cls):
                _axes[1].text(
                    _j,
                    _i,
                    int(_cm[_i, _j]),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white" if _cm[_i, _j] > _cm.max() / 2 else "black",
                )
        _fig_c.tight_layout()
        mo.vstack(
            [
                mo.md(f"**Domain MLP:** `{mlp_domain}`"),
                _fig_c,
                mo.md(
                    key_insight_card(
                        "Domain MLP should be judged against RandomForest (Ch III winner).",
                        "If RF F1 > MLP F1, keep RF for RecSys tabular pipelines and reserve nets for spectrogram inputs.",
                    )
                ),
            ]
        )
    return (mlp_domain,)


@app.cell
def _(
    MUSIC_COLOR,
    bake,
    cnn_metrics,
    key_insight_card,
    mlp_domain,
    mlp_group,
    mo,
    np,
    pd,
    plt,
):
    mo.md(
        r"""
## 5. Classical vs neural — which solver wins?

Head-to-head on comparable targets (higher macro-F1 / acc is better; small-*n* caveat applies).
"""
    )
    _rows = []
    if bake and bake.get("tasks"):
        for _t in ["group", "domain"]:
            _td = bake["tasks"].get(_t, {})
            _lb = _td.get("leaderboard") or []
            if _lb:
                _rows.append(
                    {
                        "target": _t,
                        "family": "sklearn bake-off",
                        "model": _td.get("best", _lb[0].get("model")),
                        "metric": "f1_macro",
                        "value": _lb[0].get("f1_macro"),
                    }
                )
    if mlp_group:
        _rows.append(
            {
                "target": "group",
                "family": "TensorFlow MLP",
                "model": "Dense MLP LOOCV",
                "metric": "f1_macro",
                "value": mlp_group.get("f1_macro"),
            }
        )
    if cnn_metrics:
        _rows.append(
            {
                "target": "group",
                "family": "TensorFlow CNN",
                "model": "STFT Conv2D (subj holdout)",
                "metric": "f1_macro",
                "value": cnn_metrics.get("val_f1"),
            }
        )
    if mlp_domain:
        _rows.append(
            {
                "target": "domain",
                "family": "TensorFlow MLP",
                "model": "Dense MLP (subj holdout)",
                "metric": "f1_macro",
                "value": mlp_domain.get("f1_macro"),
            }
        )
    cmp_df = pd.DataFrame(_rows)
    if len(cmp_df):
        _fig, _ax = plt.subplots(figsize=(9, 4))
        _cols = [
            MUSIC_COLOR if "TensorFlow" in f else "#5D6D7E"
            for f in cmp_df["family"]
        ]
        _labels = [f"{r.target} · {r.model}" for r in cmp_df.itertuples()]
        _ax.barh(
            _labels[::-1],
            cmp_df["value"].values[::-1],
            color=_cols[::-1],
            edgecolor="white",
        )
        _ax.set_xlabel("Score (F1 macro where available)")
        _ax.set_xlim(0, 1.05)
        _ax.set_title("Sklearn bake-off vs TensorFlow nets")
        _ax.grid(True, axis="x", alpha=0.3)
        _fig.tight_layout()
        _best_lines = []
        for _t in cmp_df["target"].unique():
            _sub = cmp_df[cmp_df.target == _t].dropna(subset=["value"])
            if len(_sub):
                _b = _sub.sort_values("value", ascending=False).iloc[0]
                _best_lines.append(
                    f"- **{_t}**: `{_b['model']}` ({_b['family']}) = **{_b['value']}**"
                )
        mo.vstack(
            [
                mo.ui.table(cmp_df),
                _fig,
                mo.md("### Recommended solver\n\n" + "\n".join(_best_lines)),
                mo.md(
                    key_insight_card(
                        "Use the best family per target — not “always deep learning”.",
                        "RecSys tabular fingerprints often stay with RF/GBM/LogReg; STFT CNNs are the right tool when "
                        "raw temporal–spectral structure is the signal.",
                    )
                ),
            ]
        )
    else:
        mo.md("*No comparison rows — train sections above first.*")
    return (cmp_df,)


@app.cell
def _(
    book_nav,
    clinical_relevance_card,
    mo,
    cmp_df,
):
    mo.vstack(
        [
            mo.md(
                r"""
## TF takeaways

1. **Real BOLD spectrograms** (not mock oscillators) feed the ConvNet.  
2. **Subject-aware splits** prevent run leakage.  
3. **MLP LOOCV** on music-effect contrasts is the deep peer of Ch III.  
4. **Pick the winner per target** from sklearn vs TF comparison.  
5. This chapter is **local-only** (TensorFlow is not in the WASM Pages stack).

```bash
# Local
export PYTHONPATH=marimo_notebooks
marimo edit marimo_notebooks/06_tf_spectrogram_model.py
# or
python marimo_notebooks/06_tf_spectrogram_model.py
```
"""
            ),
            mo.md(
                clinical_relevance_card(
                    "When tabular models already rank pos-music−tones and anterior power, a net must beat that baseline "
                    "before it enters a clinical RecSys pipeline. Prefer interpretable winners for playlist priors."
                )
            ),
            mo.md(
                "← Public book: [Algorithm Lab](../03_eda_multivariate/) · "
                "[Features](../04_feature_engineering/) · **Home** [Gallery](../../)"
            ),
            mo.md(book_nav("04_feature_engineering")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
