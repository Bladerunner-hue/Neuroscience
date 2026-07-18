#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyspark", "tensorflow", "msgspec", "pyarrow", "numpy"]
# ///
"""Pure Tungsten/Catalyst BOLD feature scale-out → Parquet + TFRecords.

Honour mode
-----------
- No RAPIDS / GPU plugins in the JVM
- No Pandas, no Python UDFs
- Spectral bands already computed outside Spark (pre_ingest_bold_to_parquet.py)
- Spark only: native DataFrame/SQL expressions, windows, AQE
- tf.data is the sacred hand-off (GPU only in TensorFlow)

Usage::

    # Local master (default — works without Spark Connect server)
    python scripts/god_mode_bold_to_tfdata.py

    # Spark Connect (native — no Docker)
    #   ./scripts/start_local_spark_connect.sh
    python scripts/god_mode_bold_to_tfdata.py --remote sc://localhost:15002

    # Skip TFRecords (Spark-only feature tables)
    python scripts/god_mode_bold_to_tfdata.py --no-tfrecords
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import msgspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOD = ROOT / "data" / "processed" / "god_parquet_bold"
FEATURES_OUT = ROOT / "data" / "processed" / "god_features"
TFREC_OUT = ROOT / "data" / "processed" / "god_tfrecords"
META_OUT = ROOT / "data" / "processed" / "god_config_metadata.json"


def build_spark(remote: str | None, app_name: str = "Neuroscience-God-Tungsten-Honour"):
    from pyspark.sql import SparkSession

    b = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config(
            "spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={ROOT / '.spark-tmp'}",
        )
        .config("spark.local.dir", str(ROOT / ".spark-tmp"))
    )
    if remote:
        # Spark Connect client
        b = b.remote(remote)
    else:
        b = b.master("local[*]")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def catalyst_pipeline(spark, spectral_path: Path, epoch_path: Path | None = None):
    """Pure Catalyst feature engineering — native columns only."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Prefer precomputed spectral bands (honour: no FFT in Spark)
    df = spark.read.parquet(str(spectral_path))

    # Ensure types
    df = (
        df.withColumn("power_high", F.col("power_high").cast("double"))
        .withColumn("power_mid", F.col("power_mid").cast("double"))
        .withColumn("power_low", F.col("power_low").cast("double"))
        .withColumn("spectral_centroid", F.col("spectral_centroid").cast("double"))
        .withColumn("tsnr", F.col("tsnr").cast("double"))
        .withColumn("run", F.col("run").cast("int"))
    )

    # Optional: pure array stats from epoch_ts if present (no UDF)
    if epoch_path and epoch_path.exists() and any(epoch_path.glob("*.parquet")):
        ep = spark.read.parquet(str(epoch_path))
        # array mean / std via higher-order aggregate (Catalyst)
        ep = ep.withColumn(
            "ts_mean_catalyst",
            F.expr(
                "aggregate(time_series_array, 0D, (acc, x) -> acc + x) "
                "/ cast(size(time_series_array) as double)"
            ),
        ).withColumn(
            "ts_len",
            F.expr("size(time_series_array)"),
        )
        # Join keys
        keys = ["dataset", "subject", "task", "run"]
        ep_small = ep.select(*keys, "ts_mean_catalyst", "ts_len")
        df = df.join(ep_small, on=keys, how="left")

    # Music-ish task flag — order matters: "nonmusic" contains the substring "music"
    t = F.lower(F.col("task"))
    df = df.withColumn(
        "is_music_task",
        F.when(t.isin("nonmusic", "tones", "washout", "rest"), F.lit(0))
        .when(t.contains("nonmusic"), F.lit(0))
        .when(t.isin("music") | t.contains("classicalmusic") | t.startswith("genmusic"), F.lit(1))
        .when(t.contains("music"), F.lit(1))
        .otherwise(F.lit(0)),
    )

    # Subject-level rollups (Catalyst groupBy + agg only)
    subj = df.groupBy("dataset", "subject", "group").agg(
        F.avg("power_high").alias("run_power_high_mean"),
        F.avg("power_mid").alias("run_power_mid_mean"),
        F.avg("power_low").alias("run_power_low_mean"),
        F.avg("spectral_centroid").alias("run_centroid_mean"),
        F.avg("tsnr").alias("run_tsnr_mean"),
        F.count(F.lit(1)).alias("n_runs"),
        F.avg(
            F.when(F.col("is_music_task") == 1, F.col("power_high"))
        ).alias("music_power_high_mean"),
        F.avg(
            F.when(F.col("is_music_task") == 0, F.col("power_high"))
        ).alias("nonmusic_power_high_mean"),
    )

    # Responder-style contrast (native)
    subj = subj.withColumn(
        "music_vs_nonmusic_power_high",
        F.col("music_power_high_mean") - F.col("nonmusic_power_high_mean"),
    ).withColumn(
        "responder_score_proxy",
        F.col("music_vs_nonmusic_power_high"),
    )

    # Dataset-level QC (window z-score + join percentile_approx — no Python UDF)
    w = Window.partitionBy("dataset")
    stats = df.groupBy("dataset").agg(
        F.expr("percentile_approx(tsnr, 0.1)").alias("tsnr_p10"),
        F.mean("power_high").alias("ph_mean"),
        F.stddev("power_high").alias("ph_std"),
    )
    run_qc = (
        df.join(stats, on="dataset", how="left")
        .withColumn(
            "power_high_z",
            (F.col("power_high") - F.col("ph_mean"))
            / (F.col("ph_std") + F.lit(1e-8)),
        )
        .withColumn(
            "qc_low_tsnr",
            F.when(F.col("tsnr") < F.col("tsnr_p10"), F.lit(1)).otherwise(F.lit(0)),
        )
        .drop("ph_mean", "ph_std")
    )

    return df, subj, run_qc


def write_tfrecords_from_subject_pdf(pdf, out_dir: Path) -> int:
    """Pure TensorFlow TFRecord writer (outside Spark — no JVM GPU)."""
    import tensorflow as tf

    out_dir.mkdir(parents=True, exist_ok=True)
    # clear old
    for p in out_dir.glob("*.tfrecord"):
        p.unlink()

    feature_cols = [
        c
        for c in pdf.columns
        if c
        not in (
            "dataset",
            "subject",
            "group",
        )
        and np.issubdtype(pdf[c].dtype, np.number)
    ]
    path = out_dir / "subjects-00000-of-00001.tfrecord"
    n = 0
    with tf.io.TFRecordWriter(str(path)) as w:
        for _, row in pdf.iterrows():
            feats = {}
            feats["dataset"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[str(row.get("dataset", "")).encode("utf-8")]
                )
            )
            feats["subject"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[str(row.get("subject", "")).encode("utf-8")]
                )
            )
            feats["group"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[str(row.get("group", "")).encode("utf-8")]
                )
            )
            label = 1 if str(row.get("group", "")).upper().find("MDD") >= 0 or str(row.get("group", "")).upper().find("DEP") >= 0 else 0
            # also Control vs MDD for ds000171
            g = str(row.get("group", ""))
            if g == "MDD":
                label = 1
            elif g == "Control":
                label = 0
            feats["label_mdd"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(label)])
            )
            for c in feature_cols:
                v = row[c]
                try:
                    fv = float(v) if v == v else 0.0
                except Exception:
                    fv = 0.0
                feats[c] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[fv])
                )
            ex = tf.train.Example(features=tf.train.Features(feature=feats))
            w.write(ex.SerializeToString())
            n += 1
    return n


def build_tf_dataset(tfrec_glob: str, feature_cols: list[str], batch: int = 8):
    """Sacred hand-off: GPU only inside TF runtime."""
    import tensorflow as tf

    spec = {
        "dataset": tf.io.FixedLenFeature([], tf.string),
        "subject": tf.io.FixedLenFeature([], tf.string),
        "group": tf.io.FixedLenFeature([], tf.string),
        "label_mdd": tf.io.FixedLenFeature([], tf.int64),
    }
    for c in feature_cols:
        spec[c] = tf.io.FixedLenFeature([], tf.float32)

    def _parse(x):
        p = tf.io.parse_single_example(x, spec)
        y = p.pop("label_mdd")
        # numeric vector
        xvec = tf.stack([p[c] for c in feature_cols], axis=0)
        return xvec, y

    files = tf.data.Dataset.list_files(tfrec_glob, shuffle=False)
    ds = (
        tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(64)
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--remote",
        default=None,
        help="Spark Connect URL, e.g. sc://localhost:15002 (default: local[*])",
    )
    p.add_argument("--no-tfrecords", action="store_true")
    p.add_argument("--smoke-tfdata", action="store_true", help="Build one tf.data batch")
    args = p.parse_args(argv)

    spectral_path = GOD / "run_spectral"
    epoch_path = GOD / "epoch_ts"
    if not spectral_path.exists() or not any(spectral_path.glob("*.parquet")):
        print(
            f"Missing {spectral_path} — run:\n"
            "  python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171",
            file=sys.stderr,
        )
        return 1

    (ROOT / ".spark-tmp").mkdir(parents=True, exist_ok=True)
    spark = build_spark(args.remote)
    print("Spark:", spark.version, "remote=" + (args.remote or "local[*]"))

    try:
        run_df, subj_df, run_qc = catalyst_pipeline(spark, spectral_path, epoch_path)

        FEATURES_OUT.mkdir(parents=True, exist_ok=True)
        run_out = FEATURES_OUT / "run_level"
        subj_out = FEATURES_OUT / "subject_level"
        qc_out = FEATURES_OUT / "run_qc"

        for path in (run_out, subj_out, qc_out):
            if path.exists():
                import shutil

                shutil.rmtree(path)

        # Catalyst write Parquet (columnar Tungsten)
        run_df.write.mode("overwrite").parquet(str(run_out))
        subj_df.write.mode("overwrite").parquet(str(subj_out))
        run_qc.write.mode("overwrite").parquet(str(qc_out))
        print(f"Wrote Catalyst tables under {FEATURES_OUT}")

        # Collect subject table for TFRecords via Arrow (still no Python UDF in Spark plan)
        # toPandas uses Arrow when enabled — acceptable client-side materialization for small n
        subj_pdf = subj_df.toPandas()
        print("subject_level rows:", len(subj_pdf))
        print(subj_pdf.head())

        feature_cols = [
            c
            for c in subj_pdf.columns
            if c not in ("dataset", "subject", "group")
            and np.issubdtype(subj_pdf[c].dtype, np.number)
        ]

        meta = {
            "spark_version": spark.version,
            "remote": args.remote,
            "n_subjects": int(len(subj_pdf)),
            "n_runs": int(run_df.count()),
            "feature_columns": feature_cols,
            "god_features": str(FEATURES_OUT.relative_to(ROOT)),
            "god_tfrecords": str(TFREC_OUT.relative_to(ROOT)),
            "honour": {
                "jvm_gpu": False,
                "python_udfs": False,
                "pandas_in_spark": False,
                "spectral_outside_spark": True,
                "tfdata_handoff": True,
            },
        }
        META_OUT.write_bytes(msgspec.json.encode(meta))
        print("msgspec meta →", META_OUT)

        if not args.no_tfrecords:
            n = write_tfrecords_from_subject_pdf(subj_pdf, TFREC_OUT)
            print(f"TFRecords: {n} examples → {TFREC_OUT}")

        if args.smoke_tfdata and not args.no_tfrecords:
            ds = build_tf_dataset(
                str(TFREC_OUT / "*.tfrecord"), feature_cols, batch=min(8, max(1, len(subj_pdf)))
            )
            for xb, yb in ds.take(1):
                print("tf.data batch X", xb.shape, "y", yb.numpy())

        # Dataset mix counts (Catalyst SQL)
        print("=== mix by dataset ===")
        run_df.groupBy("dataset", "group").count().orderBy("dataset", "group").show(
            truncate=False
        )
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
