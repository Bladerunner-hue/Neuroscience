"""PySpark session helpers."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

from neuro.config import REPO_ROOT, SPARK_MASTER


def get_spark(
    app_name: str = "BladerunnerNeuro",
    master: str | None = None,
    driver_memory: str = "8g",
) -> SparkSession:
    master = master or os.environ.get("SPARK_MASTER", SPARK_MASTER)
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.local.dir", str(REPO_ROOT / ".spark-tmp"))
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def participants_spark(spark: SparkSession, participants_pdf):
    return spark.createDataFrame(participants_pdf)