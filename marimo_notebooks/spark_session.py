"""Native Spark session helpers for marimo chapters (no Docker).

Prefer Spark Connect when a local server is running:

    bash scripts/start_local_spark_connect.sh

Client and server major.minor must match (repo: PySpark / Spark **4.1.1**).

Honour rules (pyspark-tal style):
  - Pure DataFrame / SQL API only
  - No Python UDFs, no pandas interop on hot paths
  - Spectral work stays outside Spark (pre_ingest)
"""
from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_CONNECT = "sc://localhost:15002"
DEFAULT_APP = "NeuroMarimo"
# Fail fast when Connect is down — do not block marimo cells for minutes.
CONNECT_PROBE_TIMEOUT_S = 0.35


@dataclass
class SparkInfo:
    mode: str  # "connect" | "local" | "unavailable"
    url: Optional[str]
    version: Optional[str]
    detail: str


def repo_root() -> Path:
    here = Path.cwd()
    for cand in (here, here.parent, Path(__file__).resolve().parent.parent):
        if (cand / "data" / "processed").exists() or (cand / "scripts").exists():
            return cand
    return here


def _parse_connect_host_port(url: str) -> Tuple[str, int]:
    """Parse ``sc://host:port`` (or host:port) → (host, port)."""
    u = url.strip()
    if "://" not in u:
        u = "sc://" + u
    parsed = urlparse(u)
    host = parsed.hostname or "localhost"
    port = parsed.port or 15002
    return host, int(port)


def connect_port_open(
    url: str = DEFAULT_CONNECT, *, timeout_s: float = CONNECT_PROBE_TIMEOUT_S
) -> bool:
    """TCP probe so we never hang on a missing Connect server."""
    host, port = _parse_connect_host_port(url)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def get_spark(
    *,
    prefer_connect: bool = True,
    connect_url: str = DEFAULT_CONNECT,
    app_name: str = DEFAULT_APP,
    allow_local: bool = True,
    master_local: str = "local[*]",
    force_connect: bool = False,
) -> Tuple[Any, SparkInfo]:
    """Return ``(spark_session | None, SparkInfo)``.

    Order:
      1. Spark Connect remote (native server — not Docker) if port is open
      2. Optional ``local[*]`` JVM in-process (still pure Catalyst)
      3. Unavailable → notebook falls back to Polars

    ``force_connect=True`` skips the TCP probe (still may hang if server is half-dead).
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None, SparkInfo(
            mode="unavailable",
            url=None,
            version=None,
            detail="pyspark not installed (pip install pyspark==4.1.1)",
        )

    connect_err = "skipped"
    if prefer_connect:
        if force_connect or connect_port_open(connect_url):
            try:
                spark = (
                    SparkSession.builder.remote(connect_url)
                    .appName(app_name)
                    .getOrCreate()
                )
                ver = getattr(spark, "version", None) or _client_version()
                spark.sql("SELECT 1 AS ok").collect()
                return spark, SparkInfo(
                    mode="connect",
                    url=connect_url,
                    version=str(ver),
                    detail="Spark Connect (native server)",
                )
            except Exception as exc:  # noqa: BLE001 — optional path
                connect_err = f"{type(exc).__name__}: {exc}"
        else:
            host, port = _parse_connect_host_port(connect_url)
            connect_err = (
                f"port {host}:{port} closed — run "
                "./scripts/start_local_spark_connect.sh"
            )

    if allow_local:
        try:
            spark = (
                SparkSession.builder.master(master_local)
                .appName(app_name)
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.shuffle.partitions", "8")
                .config("spark.driver.memory", "2g")
                .config("spark.ui.showConsoleProgress", "false")
                .getOrCreate()
            )
            spark.sparkContext.setLogLevel("WARN")
            ver = getattr(spark, "version", None) or _client_version()
            return spark, SparkInfo(
                mode="local",
                url=master_local,
                version=str(ver),
                detail=(
                    f"in-process {master_local} ({connect_err})"
                    if prefer_connect
                    else f"in-process {master_local}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return None, SparkInfo(
                mode="unavailable",
                url=None,
                version=_client_version(),
                detail=f"local Spark failed: {type(exc).__name__}: {exc}",
            )

    return None, SparkInfo(
        mode="unavailable",
        url=None,
        version=_client_version(),
        detail=f"Connect unavailable: {connect_err}",
    )


def _client_version() -> Optional[str]:
    try:
        import pyspark

        return pyspark.__version__
    except Exception:
        return None


def path_uri(p: Path) -> str:
    """Absolute file URI for Spark readers."""
    return Path(p).resolve().as_uri()


def read_parquet_spark(spark: Any, path: Path):
    """Read parquet directory or file with Spark (no pandas)."""
    return spark.read.parquet(path_uri(path))


def read_csv_spark(spark: Any, path: Path, *, header: bool = True, infer: bool = True):
    """Small feature CSVs only — explicit infer is fine for book-scale tables."""
    reader = spark.read.option("header", str(header).lower())
    if infer:
        reader = reader.option("inferSchema", "true")
    return reader.csv(path_uri(path))


def versions_aligned(client: Optional[str], server_release_line: Optional[str]) -> bool:
    """True if major.minor match (Connect is strict)."""
    if not client or not server_release_line:
        return True
    cm = re.match(r"(\d+\.\d+)", client)
    sm = re.search(r"(\d+\.\d+)", server_release_line)
    if not cm or not sm:
        return True
    return cm.group(1) == sm.group(1)
