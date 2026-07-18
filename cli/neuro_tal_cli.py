#!/usr/bin/env python3
"""neuro-tal-cli — God-mode PySpark TAL CLI (honour: Tungsten/Catalyst only).

Commands
--------
  prepare        Pre-ingest + Catalyst → god_features + TFRecords
  train          TFRecords → tf.data → Checkpoint/SavedModel
  viz            Marimo 07 (batch god-mode)
  stream-seed    Seed stream_inbox from god parquet
  stream-file    File-source Structured Streaming (+ checkpoint)
  stream-kafka   Kafka source/sink streaming
  stream-status  Print stream_status.json
  stream-monitor Marimo 08 streaming dashboard
  serve          FastAPI + marimo ASGI (app.py)
  status         God-mode artifact / honour flags

Examples::

    python -m cli.neuro_tal_cli prepare --datasets ds000171,ds002725
    python -m cli.neuro_tal_cli train --epochs 20
    python -m cli.neuro_tal_cli stream-seed
    python -m cli.neuro_tal_cli stream-file --once
    python -m cli.neuro_tal_cli stream-monitor
    python -m cli.neuro_tal_cli serve --port 8000
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = typer.Typer(
    name="neuro-tal-cli",
    help="Neuroscience PySpark-TAL CLI (God Mode · pure Tungsten/Catalyst honour).",
    add_completion=False,
    no_args_is_help=True,
)


def _run(cmd: list[str], env: dict | None = None) -> int:
    typer.echo("$ " + " ".join(cmd))
    e = os.environ.copy()
    if env:
        e.update(env)
    e.setdefault("PYTHONPATH", str(ROOT / "marimo_notebooks") + os.pathsep + str(ROOT))
    return subprocess.call(cmd, cwd=str(ROOT), env=e)


def _meta_path() -> Path:
    return ROOT / "data" / "processed" / "god_config_metadata.json"


def _load_meta() -> dict:
    p = _meta_path()
    if not p.exists():
        return {}
    try:
        import msgspec

        return msgspec.json.decode(p.read_bytes())
    except Exception:
        return json.loads(p.read_text())


@app.command()
def prepare(
    datasets: str = typer.Option(
        "ds000171,ds002725",
        help="Comma-separated dataset ids under data/raw/",
    ),
    backend: str = typer.Option(
        "scipy",
        help="Pre-ingest FFT backend: scipy (default) or cupy",
    ),
    remote: Optional[str] = typer.Option(
        None,
        "--remote",
        help="Spark Connect URL (implies Connect mode), e.g. sc://localhost:15002",
    ),
    connect: bool = typer.Option(
        False,
        "--connect",
        help="Use Spark Connect (default remote sc://localhost:15002)",
    ),
    max_runs: Optional[int] = typer.Option(
        None, help="Cap BOLD files per dataset during pre-ingest"
    ),
    skip_pre_ingest: bool = typer.Option(
        False, help="Skip NIfTI pre-ingest; only re-run Catalyst on existing Parquet"
    ),
    no_tfrecords: bool = typer.Option(False, help="Catalyst Parquet only"),
    smoke_tfdata: bool = typer.Option(True, help="Build one tf.data batch after write"),
):
    """Run honour pipeline: pre-ingest → Catalyst AQE → TFRecords."""
    if not skip_pre_ingest:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "pre_ingest_bold_to_parquet.py"),
            "--datasets",
            datasets,
            "--backend",
            backend,
        ]
        if max_runs is not None:
            cmd += ["--max-runs", str(max_runs)]
        code = _run(cmd)
        if code != 0:
            raise typer.Exit(code)

    cmd = [sys.executable, str(ROOT / "scripts" / "god_mode_bold_to_tfdata.py")]
    if connect or remote:
        cmd += ["--remote", remote or "sc://localhost:15002"]
    if no_tfrecords:
        cmd.append("--no-tfrecords")
    if smoke_tfdata and not no_tfrecords:
        cmd.append("--smoke-tfdata")

    code = _run(cmd)
    if code != 0:
        raise typer.Exit(code)
    typer.secho("✅ prepare complete (god_features + god_tfrecords)", fg=typer.colors.GREEN)


@app.command()
def train(
    tfrecords_path: Path = typer.Option(
        ROOT / "data" / "processed" / "god_tfrecords",
        help="Directory with *.tfrecord",
    ),
    epochs: int = typer.Option(30, help="Training epochs"),
    batch_size: int = typer.Option(8, help="Batch size (small-n friendly)"),
    checkpoint_dir: Path = typer.Option(
        ROOT / "checkpoints" / "god_mlp", help="CheckpointManager dir"
    ),
    saved_model_dir: Path = typer.Option(
        ROOT / "saved_model" / "god_mlp", help="SavedModel export dir"
    ),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate"),
    use_gpu: bool = typer.Option(
        False, "--gpu", help="Allow GPU (default CPU to keep Spark honour separate)"
    ),
):
    """TFRecords → tf.data.Dataset → MLP train + weight encapsulation."""
    import numpy as np
    import tensorflow as tf

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    meta = _load_meta()
    feature_cols: list[str] = list(
        meta.get("feature_columns")
        or [
            "run_power_high_mean",
            "run_power_mid_mean",
            "run_power_low_mean",
            "run_centroid_mean",
            "run_tsnr_mean",
            "n_runs",
            "music_power_high_mean",
            "nonmusic_power_high_mean",
            "music_vs_nonmusic_power_high",
            "responder_score_proxy",
        ]
    )

    pattern = str(tfrecords_path / "*.tfrecord")
    files = list(tfrecords_path.glob("*.tfrecord"))
    if not files:
        typer.secho(f"No TFRecords under {tfrecords_path} — run prepare first", fg=typer.colors.RED)
        raise typer.Exit(1)

    # === TFRecord binary parsing (good practice) ===
    feature_spec: dict = {
        "dataset": tf.io.FixedLenFeature([], tf.string),
        "subject": tf.io.FixedLenFeature([], tf.string),
        "group": tf.io.FixedLenFeature([], tf.string),
        "label_mdd": tf.io.FixedLenFeature([], tf.int64),
    }
    for c in feature_cols:
        feature_spec[c] = tf.io.FixedLenFeature([], tf.float32)

    def parse_tfrecord(example_proto):
        p = tf.io.parse_single_example(example_proto, feature_spec)
        y = p.pop("label_mdd")
        # drop strings for dense model
        for k in ("dataset", "subject", "group"):
            p.pop(k, None)
        x = tf.stack([p[c] for c in feature_cols], axis=0)
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        return x, y

    n = len(files)  # examples counted later
    # Count examples
    raw = tf.data.TFRecordDataset([str(f) for f in files])
    n_ex = sum(1 for _ in raw)
    steps = max(1, n_ex // batch_size)

    ds = (
        tf.data.TFRecordDataset(
            [str(f) for f in files], num_parallel_reads=tf.data.AUTOTUNE
        )
        .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(max(32, n_ex))
        .repeat()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    n_features = len(feature_cols)
    # Small dense MLP — tabular god features (STFT CNN lives in 06 / run_tf_offline)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="god_tabular_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    # === Weight encapsulation ===
    ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(ckpt, str(checkpoint_dir), max_to_keep=5)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "weights.keras"),
            save_weights_only=False,
            save_best_only=True,
            monitor="loss",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(ROOT / "logs" / "god_mlp"), histogram_freq=0
        ),
    ]

    typer.echo(
        f"Training god MLP · n_examples≈{n_ex} · features={n_features} · "
        f"steps/epoch={steps} · epochs={epochs}"
    )
    history = model.fit(
        ds,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        verbose=1,
    )
    manager.save()
    # Final SavedModel (production serving)
    model.export(str(saved_model_dir))  # Keras 3 export
    # Also classic weights
    model.save_weights(str(checkpoint_dir / "final.weights.h5"))

    hist_path = checkpoint_dir / "history.json"
    hist_path.write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2)
    )
    typer.secho(
        f"✅ checkpoints → {checkpoint_dir}\n"
        f"✅ SavedModel  → {saved_model_dir}\n"
        f"✅ history     → {hist_path}",
        fg=typer.colors.GREEN,
    )


@app.command()
def viz(
    edit: bool = typer.Option(True, help="marimo edit (False → marimo run)"),
):
    """Launch Marimo chapter 07 (Spark god-mode + multi-dataset viz)."""
    nb = ROOT / "marimo_notebooks" / "07_spark_god_mode.py"
    if not nb.exists():
        typer.secho(f"Missing {nb}", fg=typer.colors.RED)
        raise typer.Exit(1)
    cmd = [
        "marimo",
        "edit" if edit else "run",
        str(nb),
    ]
    raise typer.Exit(
        _run(
            cmd,
            env={"PYTHONPATH": str(ROOT / "marimo_notebooks") + os.pathsep + str(ROOT)},
        )
    )


@app.command()
def serve(
    port: int = typer.Option(8000, help="HTTP port"),
    skip_marimo: bool = typer.Option(False, help="REST + static only"),
):
    """Launch FastAPI + marimo ASGI (app.py)."""
    env = {}
    if skip_marimo:
        env["MARIMO_SKIP"] = "1"
    env["PORT"] = str(port)
    env["PYTHONPATH"] = str(ROOT / "marimo_notebooks") + os.pathsep + str(ROOT)
    raise typer.Exit(
        _run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
        )
    )


@app.command()
def status():
    """Print god-mode artifact paths and honour flags."""
    meta = _load_meta()
    checks = {
        "registry": ROOT / "data" / "processed" / "dataset_registry.json",
        "god_parquet": ROOT / "data" / "processed" / "god_parquet_bold" / "run_spectral",
        "god_features": ROOT / "data" / "processed" / "god_features" / "subject_level",
        "god_tfrecords": ROOT / "data" / "processed" / "god_tfrecords",
        "meta": _meta_path(),
        "chapter_07": ROOT / "marimo_notebooks" / "07_spark_god_mode.py",
        "start_connect_sh": ROOT / "scripts" / "start_local_spark_connect.sh",
        "data_landscape": ROOT / "marimo_notebooks" / "00_data_landscape.py",
    }
    typer.echo("=== neuro-tal god-mode status ===")
    for k, p in checks.items():
        ok = p.exists()
        typer.echo(f"  {'✓' if ok else '·'} {k}: {p.relative_to(ROOT)}")
    typer.echo("hint: native Connect via ./scripts/start_local_spark_connect.sh (pip pyspark; no Docker)")
    if meta:
        typer.echo("meta: " + json.dumps(meta, indent=2)[:800])
    honour = (meta or {}).get("honour") or {
        "jvm_gpu": False,
        "python_udfs": False,
        "spectral_outside_spark": True,
        "tfdata_handoff": True,
    }
    typer.echo("honour: " + json.dumps(honour))


@app.command("download")
def download_cmd(
    with_bold: bool = typer.Option(False, help="Pull limited BOLD"),
    only: str = typer.Option("", help="Comma dataset ids"),
    max_subjects: int = typer.Option(1, help="Max subjects when --with-bold"),
):
    """Download OpenNeuro cross-ref cohorts into data/raw/."""
    cmd = [sys.executable, str(ROOT / "scripts" / "download_openneuro_cohorts.py")]
    if with_bold:
        cmd.append("--with-bold")
        cmd += ["--max-subjects", str(max_subjects)]
    if only:
        cmd += ["--only", only]
    raise typer.Exit(_run(cmd))


# ---------------------------------------------------------------------------
# Structured Streaming (checkpoint + dynamic SQL + optional Kafka)
# ---------------------------------------------------------------------------
@app.command("stream-seed")
def stream_seed(force: bool = typer.Option(False, help="Overwrite existing seed files")):
    """Copy god run_spectral parquet into stream_inbox/."""
    cmd = [sys.executable, str(ROOT / "scripts" / "spark_streaming_bold.py"), "seed"]
    if force:
        cmd.append("--force")
    raise typer.Exit(_run(cmd))


@app.command("stream-file")
def stream_file(
    once: bool = typer.Option(True, "--once/--continuous", help="Single micro-batch (default)"),
    trigger_seconds: int = typer.Option(10, help="processingTime trigger when continuous"),
    remote: Optional[str] = typer.Option(None, help="Spark Connect URL"),
    connect: bool = typer.Option(False, help="Use sc://localhost:15002"),
    max_files: int = typer.Option(20, help="maxFilesPerTrigger"),
):
    """File-source streaming → foreachBatch dynamic SQL → stream_features/."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "spark_streaming_bold.py"),
        "file",
        "--trigger-seconds",
        str(trigger_seconds),
        "--max-files",
        str(max_files),
    ]
    if once:
        cmd.append("--once")
    if connect or remote:
        cmd += ["--remote", remote or "sc://localhost:15002"]
    raise typer.Exit(_run(cmd))


@app.command("stream-kafka")
def stream_kafka(
    bootstrap: str = typer.Option("localhost:9092", help="Kafka bootstrap.servers"),
    topic_in: str = typer.Option("bold-incoming"),
    topic_out: str = typer.Option("bold-features"),
    remote: Optional[str] = typer.Option(None),
    connect: bool = typer.Option(False),
    once: bool = typer.Option(False, help="availableNow single shot"),
    trigger_seconds: int = typer.Option(10),
):
    """Kafka source → Catalyst enrich → Kafka sink + parquet mirror."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "spark_streaming_bold.py"),
        "kafka",
        "--bootstrap",
        bootstrap,
        "--topic-in",
        topic_in,
        "--topic-out",
        topic_out,
        "--trigger-seconds",
        str(trigger_seconds),
    ]
    if once:
        cmd.append("--once")
    if connect or remote:
        cmd += ["--remote", remote or "sc://localhost:15002"]
    raise typer.Exit(_run(cmd))


@app.command("stream-status")
def stream_status():
    """Print streaming status + config JSON."""
    raise typer.Exit(
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "spark_streaming_bold.py"),
                "status",
            ]
        )
    )


@app.command("stream-monitor")
def stream_monitor(
    edit: bool = typer.Option(True, help="marimo edit (False → run)"),
):
    """Launch Marimo VIII streaming dashboard."""
    nb = ROOT / "marimo_notebooks" / "08_spark_streaming.py"
    if not nb.exists():
        typer.secho(f"Missing {nb}", fg=typer.colors.RED)
        raise typer.Exit(1)
    cmd = ["marimo", "edit" if edit else "run", str(nb)]
    raise typer.Exit(
        _run(
            cmd,
            env={"PYTHONPATH": str(ROOT / "marimo_notebooks") + os.pathsep + str(ROOT)},
        )
    )


if __name__ == "__main__":
    app()
