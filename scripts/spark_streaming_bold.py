#!/usr/bin/env python3
"""Structured Streaming for god-mode BOLD features (Spark Connect–ready).

Honour mode
-----------
- Micro-batch Structured Streaming (stable with Connect)
- Durable checkpointLocation (mandatory for production)
- Pure Catalyst transforms; dynamic SQL via foreachBatch + temp view
- Optional Kafka source/sink (packages on the JVM)
- File source for local demos without Kafka

Layout::

    data/processed/stream_inbox/     # drop new run_spectral parquet parts here
    data/processed/stream_features/  # micro-batch feature sink (parquet)
    data/processed/stream_checkpoints/  # query recovery state
    data/processed/stream_config.json   # msgspec-friendly runtime params

Usage::

    # Seed inbox from existing god parquet, run one micro-batch, stop
    python scripts/spark_streaming_bold.py file --once

    # Continuous file stream (Ctrl+C to stop)
    python scripts/spark_streaming_bold.py file --trigger-seconds 5

    # Kafka (requires kafka + spark-sql-kafka package)
    python scripts/spark_streaming_bold.py kafka \\
        --bootstrap localhost:9092 --topic-in bold-incoming --topic-out bold-features
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import msgspec

ROOT = Path(__file__).resolve().parents[1]
GOD_SPECTRAL = ROOT / "data" / "processed" / "god_parquet_bold" / "run_spectral"
INBOX = ROOT / "data" / "processed" / "stream_inbox"
FEATURES = ROOT / "data" / "processed" / "stream_features"
CHECKPOINTS = ROOT / "data" / "processed" / "stream_checkpoints"
STREAM_CFG = ROOT / "data" / "processed" / "stream_config.json"
STATUS_JSON = ROOT / "data" / "processed" / "stream_status.json"


def _spark(remote: str | None, packages: str | None = None):
    from pyspark.sql import SparkSession

    b = (
        SparkSession.builder.appName("Neuro-TAL-StructuredStreaming")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.local.dir", str(ROOT / ".spark-tmp"))
        .config(
            "spark.sql.streaming.schemaInference",
            "true",
        )
    )
    if packages:
        b = b.config("spark.jars.packages", packages)
    if remote:
        b = b.remote(remote)
    else:
        b = b.master("local[2]")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _load_cfg() -> dict:
    default = {
        "current_group": "MDD",
        "min_power_high": 0.0,
        "trigger_seconds": 10,
        "max_files_per_trigger": 20,
    }
    if STREAM_CFG.exists():
        try:
            user = msgspec.json.decode(STREAM_CFG.read_bytes())
            if isinstance(user, dict):
                default.update(user)
        except Exception:
            pass
    return default


def _write_status(payload: dict) -> None:
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_bytes(msgspec.json.encode(payload))


def seed_inbox(*, force: bool = False) -> int:
    """Copy god run_spectral parquet into stream_inbox for file-source demos."""
    INBOX.mkdir(parents=True, exist_ok=True)
    src_parts = list(GOD_SPECTRAL.glob("*.parquet"))
    if not src_parts:
        print(f"No source parquet under {GOD_SPECTRAL}", file=sys.stderr)
        return 0
    n = 0
    for i, src in enumerate(src_parts):
        dest = INBOX / f"seed-{i:04d}-{src.name}"
        if dest.exists() and not force:
            continue
        shutil.copy2(src, dest)
        n += 1
    print(f"seeded {n} parquet file(s) → {INBOX}")
    return n


def catalyst_enrich(df):
    """Pure Catalyst columns on a batch or streaming DF (no Python UDF)."""
    from pyspark.sql import functions as F

    t = F.lower(F.col("task"))
    df = df.withColumn(
        "is_music_task",
        F.when(t.isin("nonmusic", "tones", "washout", "rest"), F.lit(0))
        .when(t.contains("nonmusic"), F.lit(0))
        .when(
            t.isin("music")
            | t.contains("classicalmusic")
            | t.startswith("genmusic"),
            F.lit(1),
        )
        .when(t.contains("music"), F.lit(1))
        .otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "power_high", F.col("power_high").cast("double")
    ).withColumn("power_mid", F.col("power_mid").cast("double"))
    return df


def make_foreach_batch(cfg: dict, sink_path: Path):
    """Dynamic SQL per micro-batch via temp view (Catalyst).

    Must use ``batch_df.sparkSession`` (not an outer session) for views/SQL.
    """

    def dynamic_batch(batch_df, batch_id: int):
        # Connect-safe empty check (avoid RDD API)
        try:
            empty = len(batch_df.take(1)) == 0
        except Exception:
            empty = batch_df.count() == 0
        if empty:
            _write_status(
                {
                    "batch_id": int(batch_id),
                    "rows": 0,
                    "ts": time.time(),
                    "note": "empty batch",
                }
            )
            return

        ss = batch_df.sparkSession
        batch_df = catalyst_enrich(batch_df)
        view = f"stream_batch_{int(batch_id)}"
        batch_df.createOrReplaceTempView(view)

        # Reload cfg each batch → runtime dynamism without restart
        live = _load_cfg()
        group = str(live.get("current_group", cfg.get("current_group", "MDD"))).replace(
            "'", ""
        )
        min_ph = float(live.get("min_power_high", 0.0))

        # Fully Catalyst SQL — parameterized cohort tag
        # Quote reserved column name `group`
        sql = f"""
        SELECT
          dataset,
          subject,
          `group` AS group_label,
          task,
          run,
          power_low,
          power_mid,
          power_high,
          spectral_centroid,
          tsnr,
          is_music_task,
          CASE WHEN `group` = '{group}' THEN 'target' ELSE 'other' END AS cohort,
          {int(batch_id)} AS batch_id
        FROM {view}
        WHERE power_high >= {min_ph}
        """
        result = ss.sql(sql)
        n = result.count()
        result.write.mode("append").parquet(str(sink_path))
        _write_status(
            {
                "batch_id": int(batch_id),
                "rows": int(n),
                "ts": time.time(),
                "current_group": group,
                "min_power_high": min_ph,
                "sink": str(sink_path),
            }
        )
        print(f"  batch {batch_id}: wrote {n} rows → {sink_path}")

    return dynamic_batch


def run_file_stream(
    *,
    remote: str | None,
    trigger_seconds: int,
    once: bool,
    checkpoint: Path,
    max_files: int,
) -> int:
    spark = _spark(remote)
    cfg = _load_cfg()
    FEATURES.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    INBOX.mkdir(parents=True, exist_ok=True)

    if not any(INBOX.glob("*.parquet")):
        seed_inbox()

    try:
        stream_df = (
            spark.readStream.format("parquet")
            .option("path", str(INBOX))
            .option("maxFilesPerTrigger", str(max_files))
            .load()
        )
        # Keep schema lean for streaming
        cols = [
            c
            for c in stream_df.columns
            if c
            in (
                "dataset",
                "subject",
                "group",
                "task",
                "run",
                "power_low",
                "power_mid",
                "power_high",
                "spectral_centroid",
                "tsnr",
                "psd_method",
                "n_volumes",
                "tr_sec",
            )
        ]
        stream_df = stream_df.select(*cols) if cols else stream_df

        writer = (
            stream_df.writeStream.foreachBatch(make_foreach_batch(cfg, FEATURES))
            .option("checkpointLocation", str(checkpoint))
            .outputMode("update")
        )
        if once:
            # Spark 3.3+ prefers availableNow; fall back to once
            try:
                writer = writer.trigger(availableNow=True)
            except TypeError:
                writer = writer.trigger(once=True)
        else:
            writer = writer.trigger(processingTime=f"{trigger_seconds} seconds")

        query = writer.start()
        print(
            f"Streaming query id={query.id} name={query.name} "
            f"checkpoint={checkpoint} once={once}"
        )
        if once:
            query.awaitTermination()
            prog = query.lastProgress
            # lastProgress may contain UUID objects
            prog_safe = json.loads(json.dumps(prog, default=str)) if prog else None
            print("lastProgress:", json.dumps(prog_safe, indent=2) if prog_safe else None)
            _write_status(
                {
                    "isActive": False,
                    "lastProgress": prog_safe,
                    "batch_id": (prog_safe or {}).get("batchId"),
                    "numInputRows": (prog_safe or {}).get("numInputRows"),
                    "ts": time.time(),
                    "mode": "file",
                    "once": True,
                }
            )
        else:
            # Run until interrupted; poll status for client dashboards
            try:
                while query.isActive:
                    st = query.status
                    lp = query.lastProgress
                    _write_status(
                        {
                            "isActive": query.isActive,
                            "status": st,
                            "lastProgress": lp,
                            "ts": time.time(),
                        }
                    )
                    time.sleep(max(1, trigger_seconds // 2))
            except KeyboardInterrupt:
                print("stopping…")
                query.stop()
            query.awaitTermination(30)
    finally:
        spark.stop()
    return 0


def run_kafka_stream(
    *,
    remote: str | None,
    bootstrap: str,
    topic_in: str,
    topic_out: str,
    checkpoint: Path,
    trigger_seconds: int,
    once: bool,
) -> int:
    """Kafka source → Catalyst enrich → Kafka JSON sink + parquet mirror."""
    packages = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
    spark = _spark(remote, packages=packages if not remote else None)
    cfg = _load_cfg()
    checkpoint.mkdir(parents=True, exist_ok=True)
    FEATURES.mkdir(parents=True, exist_ok=True)

    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("dataset", StringType()),
            StructField("subject", StringType()),
            StructField("group", StringType()),
            StructField("task", StringType()),
            StructField("run", IntegerType()),
            StructField("power_low", DoubleType()),
            StructField("power_mid", DoubleType()),
            StructField("power_high", DoubleType()),
            StructField("spectral_centroid", DoubleType()),
            StructField("tsnr", DoubleType()),
        ]
    )

    try:
        raw = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", bootstrap)
            .option("subscribe", topic_in)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .load()
        )
        parsed = (
            raw.selectExpr("CAST(value AS STRING) AS json")
            .select(F.from_json(F.col("json"), schema).alias("data"))
            .select("data.*")
            .filter(F.col("subject").isNotNull())
        )
        enriched = catalyst_enrich(parsed)

        def batch_kafka(batch_df, batch_id: int):
            try:
                empty = len(batch_df.take(1)) == 0
            except Exception:
                empty = batch_df.count() == 0
            if empty:
                return
            n = batch_df.count()
            # Parquet mirror for tf.data / marimo
            batch_df.write.mode("append").parquet(str(FEATURES))
            # Kafka sink: JSON lines
            out = batch_df.select(
                F.lit(None).cast("string").alias("key"),
                F.to_json(F.struct(*batch_df.columns)).alias("value"),
            )
            (
                out.write.format("kafka")
                .option("kafka.bootstrap.servers", bootstrap)
                .option("topic", topic_out)
                .save()
            )
            _write_status(
                {
                    "batch_id": int(batch_id),
                    "rows": int(n),
                    "ts": time.time(),
                    "topic_out": topic_out,
                    "mode": "kafka",
                }
            )

        writer = (
            enriched.writeStream.foreachBatch(batch_kafka)
            .option("checkpointLocation", str(checkpoint))
            .outputMode("update")
        )
        if once:
            writer = writer.trigger(availableNow=True)
        else:
            writer = writer.trigger(processingTime=f"{trigger_seconds} seconds")
        query = writer.start()
        print(f"Kafka streaming {topic_in} → {topic_out} id={query.id}")
        if once:
            query.awaitTermination()
        else:
            query.awaitTermination()
    finally:
        spark.stop()
    return 0


def print_status() -> int:
    if STATUS_JSON.exists():
        print(STATUS_JSON.read_text())
    else:
        print("{}", file=sys.stderr)
        return 1
    if STREAM_CFG.exists():
        print("--- config ---")
        print(STREAM_CFG.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    seed = sub.add_parser("seed", help="Seed stream_inbox from god run_spectral")
    seed.add_argument("--force", action="store_true")

    st = sub.add_parser("status", help="Print stream_status.json")

    file_p = sub.add_parser("file", help="File-source Structured Streaming")
    file_p.add_argument("--remote", default=None)
    file_p.add_argument("--trigger-seconds", type=int, default=10)
    file_p.add_argument("--once", action="store_true", help="Single micro-batch then exit")
    file_p.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS / "file",
    )
    file_p.add_argument("--max-files", type=int, default=20)

    kafka_p = sub.add_parser("kafka", help="Kafka source → features → Kafka sink")
    kafka_p.add_argument("--remote", default=None)
    kafka_p.add_argument("--bootstrap", default="localhost:9092")
    kafka_p.add_argument("--topic-in", default="bold-incoming")
    kafka_p.add_argument("--topic-out", default="bold-features")
    kafka_p.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "kafka")
    kafka_p.add_argument("--trigger-seconds", type=int, default=10)
    kafka_p.add_argument("--once", action="store_true")

    args = p.parse_args(argv)
    STREAM_CFG.parent.mkdir(parents=True, exist_ok=True)
    if not STREAM_CFG.exists():
        STREAM_CFG.write_bytes(
            msgspec.json.encode(
                {
                    "current_group": "MDD",
                    "min_power_high": 0.0,
                    "trigger_seconds": 10,
                }
            )
        )

    if args.cmd == "seed":
        return 0 if seed_inbox(force=args.force) >= 0 else 1
    if args.cmd == "status":
        return print_status()
    if args.cmd == "file":
        return run_file_stream(
            remote=args.remote,
            trigger_seconds=args.trigger_seconds,
            once=args.once,
            checkpoint=args.checkpoint,
            max_files=args.max_files,
        )
    if args.cmd == "kafka":
        return run_kafka_stream(
            remote=args.remote,
            bootstrap=args.bootstrap,
            topic_in=args.topic_in,
            topic_out=args.topic_out,
            checkpoint=args.checkpoint,
            trigger_seconds=args.trigger_seconds,
            once=args.once,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
