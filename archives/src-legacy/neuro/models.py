"""TensorFlow models: ROI transformer, LSTM, 3D CNN baseline."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_roi_transformer(
    n_rois: int,
    n_time: int,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    n_classes: int = 2,
) -> keras.Model:
    inputs = keras.Input(shape=(n_time, n_rois))
    x = layers.Dense(embed_dim)(inputs)
    positions = tf.range(start=0, limit=n_time, delta=1)
    pos_emb = layers.Embedding(input_dim=n_time, output_dim=embed_dim)(positions)
    x = x + pos_emb

    for _ in range(num_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x1, x1)
        x2 = layers.Add()([x, attn])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ffn = layers.Dense(embed_dim * 2, activation="gelu")(x3)
        ffn = layers.Dense(embed_dim)(ffn)
        x = layers.Add()([x2, ffn])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name="roi_transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_classifier(n_rois: int, n_time: int, n_classes: int = 2) -> keras.Model:
    inputs = keras.Input(shape=(n_time, n_rois))
    x = layers.LSTM(64, return_sequences=False)(inputs)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="roi_lstm")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_3d_cnn(input_shape: tuple, n_classes: int = 2) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv3D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling3D(2)(x)
    x = layers.Conv3D(32, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="cnn3d")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_tf_dataset(X, y, batch_size: int = 8, training: bool = True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(y), 64))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)