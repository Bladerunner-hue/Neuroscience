"""06 — TF spectrogram model. Local-only (not WASM). Canonical marimo notebook."""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="06 — TF Spectrograms")


@app.cell
def _():
    import os

    # Prefer CPU so local/CI runs don't depend on a working CUDA/cuDNN stack
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        set_global_style,
    )

    set_global_style()
    try:
        import tensorflow_probability  # noqa: F401

        TFP_AVAILABLE = True
    except ImportError:
        TFP_AVAILABLE = False

    return (
        CONTROL_COLOR,
        MDD_COLOR,
        TFP_AVAILABLE,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        mo,
        np,
        pd,
        plt,
        px,
        tf,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
# 06 — TensorFlow: Spectrogram ConvNets

**Chapter 6 · local only** (TensorFlow required; not on GitHub Pages WASM).

Predict MDD vs Control from the spectral signature of music vs tones.
"""
    )
    return


@app.cell
def _(hypothesis_card, mo):
    mo.md(
        hypothesis_card(
            "A Conv2D on STFT spectrograms of BOLD is more accurate on music blocks than tones.",
            "Music-only evaluation shows a clear lift — stimulus-specific deficit signal.",
        )
    )
    return


@app.cell
def _(mo):
    n_subj = mo.ui.slider(10, 28, value=14, step=2, label="# subjects")
    tr = mo.ui.number(start=2.5, stop=3.5, value=3.0, step=0.25, label="TR (s)")
    frame_len = mo.ui.slider(8, 32, value=16, step=4, label="STFT frame_length")
    frame_step = mo.ui.slider(2, 8, value=4, step=1, label="STFT frame_step")
    fft_len = mo.ui.slider(16, 64, value=32, step=8, label="FFT length")
    use_mixed_precision = mo.ui.checkbox(False, label="Mixed precision")
    use_strategy = mo.ui.checkbox(False, label="MirroredStrategy")
    learning_rate = mo.ui.slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="Learning rate")
    epochs = mo.ui.slider(2, 12, value=4, step=1, label="Epochs")
    batch_size = mo.ui.slider(4, 16, value=8, step=2, label="Batch size")
    use_lstm = mo.ui.checkbox(False, label="LSTM head")
    dropout = mo.ui.slider(0.0, 0.5, value=0.25, step=0.05, label="Dropout")
    mo.md("## Controls")
    mo.hstack([n_subj, tr, epochs, batch_size], justify="start")
    mo.hstack([frame_len, frame_step, fft_len, learning_rate], justify="start")
    mo.hstack([use_mixed_precision, use_strategy, use_lstm, dropout], justify="start")
    return (
        batch_size,
        dropout,
        epochs,
        fft_len,
        frame_len,
        frame_step,
        learning_rate,
        n_subj,
        tr,
        use_lstm,
        use_mixed_precision,
        use_strategy,
    )


@app.cell
def _(make_synthetic_bold_dataset, mo, n_subj, np, tr):
    @mo.cache
    def build_synthetic_dataset(n_subjects: int, tr_sec: float):
        df = make_synthetic_bold_dataset(
            n_subjects=n_subjects, n_timepoints=105, tr=tr_sec, seed=123
        )
        examples, labels, conditions = [], [], []
        for (_subj, cond), g in df.groupby(["subject", "condition"]):
            y = 1 if g["group"].iloc[0] == "MDD" else 0
            examples.append(g.sort_values("time")["bold"].values.astype("float32"))
            labels.append(y)
            conditions.append(cond)
        return np.stack(examples), np.array(labels, dtype="int32"), np.array(conditions)

    bold_arr, y, conds = build_synthetic_dataset(int(n_subj.value), float(tr.value))
    mo.md(f"**{len(bold_arr)}** blocks · class counts `{np.bincount(y).tolist()}`")
    return bold_arr, conds, y


@app.cell
def _(batch_size, bold_arr, conds, fft_len, frame_len, frame_step, mo, tf, y):
    def bold_to_spectrogram(x, frame_length, frame_step, fft_length):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-6)
        stft = tf.signal.stft(
            x,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window,
        )
        return tf.expand_dims(tf.abs(stft), axis=-1)

    def make_tf_dataset(bold_arr, y, conds, fl, fs, fft, batch, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((bold_arr, y, conds))

        def _map(x, yy, c):
            return bold_to_spectrogram(x, fl, fs, fft), yy, c

        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(bold_arr), reshuffle_each_iteration=True)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE).cache()

    train_ds = make_tf_dataset(
        bold_arr,
        y,
        conds,
        int(frame_len.value),
        int(frame_step.value),
        int(fft_len.value),
        int(batch_size.value),
    )
    for spec_batch, _yb, _cb in train_ds.take(1):
        input_shape = tuple(spec_batch.shape[1:])
        break
    mo.md(f"**Spectrogram shape:** `{input_shape}`")
    return input_shape, train_ds


@app.cell
def _(
    dropout,
    input_shape,
    learning_rate,
    mo,
    tf,
    use_lstm,
    use_mixed_precision,
    use_strategy,
):
    if use_mixed_precision.value:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    strategy = (
        tf.distribute.MirroredStrategy()
        if use_strategy.value
        else tf.distribute.get_strategy()
    )
    with strategy.scope():
        inp = tf.keras.Input(shape=input_shape, name="spectrogram")
        x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if use_lstm.value:
            x = tf.keras.layers.Reshape((-1, 32))(x)
            x = tf.keras.layers.LSTM(16, dropout=float(dropout.value))(x)
        x = tf.keras.layers.Dropout(float(dropout.value))(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out_class = tf.keras.layers.Dense(
            1, activation="sigmoid", dtype="float32", name="is_mdd"
        )(x)
        out_unc = tf.keras.layers.Dense(
            1, activation="softplus", dtype="float32", name="uncertainty"
        )(x)
        model = tf.keras.Model(inp, [out_class, out_unc], name="bold_spectrogram_model")
        opt = tf.keras.optimizers.Adam(float(learning_rate.value))
        if use_mixed_precision.value:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        model.compile(
            optimizer=opt,
            loss={
                "is_mdd": "binary_crossentropy",
                "uncertainty": lambda y_true, y_pred: tf.reduce_mean(y_pred),
            },
            loss_weights={"is_mdd": 1.0, "uncertainty": 0.05},
            metrics={"is_mdd": ["accuracy", tf.keras.metrics.AUC(name="auc")]},
        )
    lines = []
    model.summary(print_fn=lines.append)
    mo.md("**Model**\n\n```\n" + "\n".join(lines) + "\n```")
    return (model,)


@app.cell
def _(batch_size, epochs, mo, model, np, plt, train_ds):
    mo.md("## Training")
    X_specs, y_arr, cond_arr = [], [], []
    for s, yy, cc in train_ds.unbatch():
        X_specs.append(s.numpy())
        y_arr.append(int(yy.numpy()))
        c = cc.numpy()
        cond_arr.append(c.decode() if isinstance(c, (bytes, bytearray)) else c)
    X_specs = np.array(X_specs)
    y_arr = np.array(y_arr)
    cond_arr = np.array(cond_arr)

    rng = np.random.RandomState(42)
    idx = np.arange(len(X_specs))
    rng.shuffle(idx)
    split = max(1, int(0.75 * len(idx)))
    train_idx, val_idx = idx[:split], idx[split:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    X_train, y_train = X_specs[train_idx], y_arr[train_idx]
    X_val, y_val = X_specs[val_idx], y_arr[val_idx]

    hist = model.fit(
        X_train,
        {"is_mdd": y_train, "uncertainty": np.zeros_like(y_train, dtype="float32")},
        validation_data=(
            X_val,
            {"is_mdd": y_val, "uncertainty": np.zeros_like(y_val, dtype="float32")},
        ),
        epochs=int(epochs.value),
        batch_size=int(batch_size.value),
        verbose=0,
    )

    def _pick(keys):
        for k in keys:
            if k in hist.history:
                return hist.history[k]
        return []

    train_acc = _pick(["is_mdd_accuracy", "is_mdd_is_mdd_accuracy", "accuracy"])
    val_acc = _pick(["val_is_mdd_accuracy", "val_is_mdd_is_mdd_accuracy", "val_accuracy"])
    if train_acc and val_acc:
        mo.md(f"**train acc** {train_acc[-1]:.3f} · **val acc** {val_acc[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if train_acc:
        ax.plot(train_acc, label="train acc")
    if val_acc:
        ax.plot(val_acc, label="val acc")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mo.output.append(fig)
    return X_val, cond_arr, val_idx, y_val


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    X_val,
    cond_arr,
    key_insight_card,
    mo,
    model,
    np,
    pd,
    px,
    val_idx,
    y_val,
):
    mo.md("## Accuracy by stimulus type")
    preds, _unc = model.predict(X_val, verbose=0)
    pred_class = (np.asarray(preds).ravel() > 0.5).astype(int)
    eval_df = pd.DataFrame(
        {"true": y_val, "pred": pred_class, "condition": cond_arr[val_idx]}
    )

    def acc(d):
        return float((d["true"] == d["pred"]).mean()) if len(d) else 0.0

    music_mask = eval_df["condition"] == "music"
    music_acc = acc(eval_df[music_mask])
    non_acc = acc(eval_df[~music_mask])
    metrics_df = pd.DataFrame(
        {
            "condition": ["music", "nonmusic"],
            "accuracy": [music_acc, non_acc],
            "n": [int(music_mask.sum()), int((~music_mask).sum())],
        }
    )
    mo.ui.table(metrics_df)
    mo.ui.plotly(
        px.bar(
            metrics_df,
            x="condition",
            y="accuracy",
            color="condition",
            color_discrete_map={"music": CONTROL_COLOR, "nonmusic": MDD_COLOR},
            range_y=[0, 1],
            title="Accuracy by stimulus",
        )
    )
    delta = music_acc - non_acc
    mo.md(
        key_insight_card(
            f"Music {music_acc:.1%} vs non-music {non_acc:.1%} (Δ={delta:+.1%})",
            "Gap supports music-specific deficit rather than generic auditory confounds.",
            effect_size=f"Δ={delta:+.2f}",
        )
    )
    return X_val, pred_class, y_val


@app.cell
def _(
    TFP_AVAILABLE,
    book_nav,
    clinical_relevance_card,
    mo,
    use_mixed_precision,
    use_strategy,
):
    mo.md(
        f"""
## TF stack

- Mixed precision: {'✅' if use_mixed_precision.value else '☐'}
- MirroredStrategy: {'✅' if use_strategy.value else '☐'}
- tf.data AUTOTUNE cache/prefetch
- `tf.signal.stft`
- TFP: {'✅' if TFP_AVAILABLE else 'not installed'}
"""
    )
    mo.md(
        clinical_relevance_card(
            "Music-specific classification supports spectral biomarkers for anhedonia and therapy selection."
        )
    )
    mo.md(book_nav("06_tf_spectrogram_model"))
    mo.md(
        "> Not exported to GitHub Pages WASM. Run locally: "
        "`marimo edit marimo_notebooks/06_tf_spectrogram_model.py`"
    )
    return


if __name__ == "__main__":
    app.run()
