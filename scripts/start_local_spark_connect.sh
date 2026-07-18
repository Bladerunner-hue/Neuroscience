#!/usr/bin/env bash
# Start a *native* Spark Connect server for the Neuroscience book.
#
# No Docker. Client and server must share the same Spark major.minor
# (this repo uses PySpark 4.1.1 ↔ Spark 4.1.1 under /opt/spark by default).
#
# Usage:
#   ./scripts/start_local_spark_connect.sh          # start (daemon)
#   ./scripts/start_local_spark_connect.sh --fg      # foreground (logs to terminal)
#   ./scripts/start_local_spark_connect.sh --stop    # stop
#   ./scripts/start_local_spark_connect.sh --status  # port + version check
#
# Then in Python / marimo:
#   from spark_session import get_spark
#   spark, mode = get_spark()   # prefers sc://localhost:15002
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SPARK_CONNECT_PORT:-15002}"
DRIVER_MEM="${SPARK_DRIVER_MEMORY:-4g}"

# Resolve SPARK_HOME: env → /opt/spark → pyspark package layout (if sbin present)
resolve_spark_home() {
  if [[ -n "${SPARK_HOME:-}" && -x "${SPARK_HOME}/sbin/start-connect-server.sh" ]]; then
    echo "${SPARK_HOME}"
    return
  fi
  if [[ -x /opt/spark/sbin/start-connect-server.sh ]]; then
    echo /opt/spark
    return
  fi
  # Some full Spark tarballs are installed elsewhere
  for cand in \
    "${HOME}/spark" \
    "${HOME}/spark-4.1.1-bin-hadoop3" \
    /usr/local/spark; do
    if [[ -x "${cand}/sbin/start-connect-server.sh" ]]; then
      echo "${cand}"
      return
    fi
  done
  return 1
}

client_version() {
  python3 - <<'PY' 2>/dev/null || true
try:
    import pyspark
    print(pyspark.__version__)
except Exception:
    pass
PY
}

server_release() {
  local home="$1"
  if [[ -f "${home}/RELEASE" ]]; then
    head -1 "${home}/RELEASE" | sed -n 's/.*Spark \([0-9.]*\).*/\1/p'
  fi
}

versions_compatible() {
  local c="$1" s="$2"
  [[ -z "$c" || -z "$s" ]] && return 0
  local cm sm
  cm="$(echo "$c" | cut -d. -f1-2)"
  sm="$(echo "$s" | cut -d. -f1-2)"
  [[ "$cm" == "$sm" ]]
}

cmd_status() {
  local home cv sv
  home="$(resolve_spark_home 2>/dev/null || true)"
  cv="$(client_version)"
  sv="$(server_release "${home:-}")"
  echo "repo:          ${ROOT}"
  echo "SPARK_HOME:    ${home:-NOT FOUND}"
  echo "client pyspark:${cv:-unknown}"
  echo "server RELEASE:${sv:-unknown}"
  if [[ -n "${home:-}" ]] && ! versions_compatible "$cv" "$sv"; then
    echo "WARNING: client ${cv} vs server ${sv} — Spark Connect is strict; align versions."
  fi
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q ":${PORT}"; then
      echo "port ${PORT}:    LISTENING"
    else
      echo "port ${PORT}:    free (server not running)"
    fi
  elif command -v lsof >/dev/null 2>&1; then
    if lsof -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "port ${PORT}:    LISTENING"
    else
      echo "port ${PORT}:    free (server not running)"
    fi
  fi
  echo
  echo "Connect URL:   sc://localhost:${PORT}"
  echo "UI (typical):  http://localhost:4040"
}

cmd_stop() {
  local home
  home="$(resolve_spark_home)" || {
    echo "SPARK_HOME not found; cannot stop via sbin." >&2
    exit 1
  }
  export SPARK_HOME="$home"
  if [[ -x "${home}/sbin/stop-connect-server.sh" ]]; then
    "${home}/sbin/stop-connect-server.sh" || true
  else
    # Fallback: kill connect server JVM for this port
    pkill -f "SparkConnectServer" 2>/dev/null || true
  fi
  echo "Stopped Spark Connect (if it was running)."
}

cmd_start() {
  local fg=0
  if [[ "${1:-}" == "--fg" ]]; then
    fg=1
  fi

  local home cv sv
  home="$(resolve_spark_home)" || {
    cat >&2 <<'EOF'
Could not find a native Spark install with sbin/start-connect-server.sh.

Install a full Apache Spark binary matching your PySpark version, e.g.:
  # https://spark.apache.org/downloads.html  → Spark 4.1.1 pre-built for Hadoop 3
  tar xf spark-4.1.1-bin-hadoop3.tgz
  export SPARK_HOME=$PWD/spark-4.1.1-bin-hadoop3

This repo expects PySpark 4.1.1 via pip (see requirements.txt).
Install a matching full Spark binary for the Connect server only.
EOF
    exit 1
  }
  export SPARK_HOME="$home"
  cv="$(client_version)"
  sv="$(server_release "$home")"
  if ! versions_compatible "$cv" "$sv"; then
    echo "ERROR: PySpark client ${cv:-?} != Spark server ${sv:-?} (major.minor)." >&2
    echo "Align versions before starting Connect." >&2
    exit 1
  fi

  # Bind processed + raw data for multi-source notebooks
  export SPARK_LOCAL_DIRS="${SPARK_LOCAL_DIRS:-${ROOT}/.spark-tmp}"
  mkdir -p "${SPARK_LOCAL_DIRS}"

  local extra=(
    --conf "spark.sql.adaptive.enabled=true"
    --conf "spark.sql.adaptive.coalescePartitions.enabled=true"
    --conf "spark.sql.adaptive.skewJoin.enabled=true"
    --conf "spark.sql.execution.arrow.pyspark.enabled=true"
    --conf "spark.driver.memory=${DRIVER_MEM}"
    --conf "spark.driver.host=127.0.0.1"
  )

  echo "Starting Spark Connect from SPARK_HOME=${SPARK_HOME}"
  echo "  version server=${sv:-?} client=${cv:-?}  port=${PORT}"
  echo "  data root: ${ROOT}/data"

  if [[ "$fg" -eq 1 ]]; then
    # Foreground: no daemonize
    export SPARK_NO_DAEMONIZE=1
    exec "${SPARK_HOME}/sbin/start-connect-server.sh" "${extra[@]}"
  else
    "${SPARK_HOME}/sbin/start-connect-server.sh" "${extra[@]}"
    sleep 1
    cmd_status
    echo
    echo "In marimo:"
    echo "  from spark_session import get_spark"
    echo "  spark, info = get_spark()  # remote sc://localhost:${PORT}"
  fi
}

case "${1:-start}" in
  start)  cmd_start ;;
  --fg|fg) cmd_start --fg ;;
  --stop|stop) cmd_stop ;;
  --status|status) cmd_status ;;
  -h|--help|help)
    sed -n '2,20p' "$0"
    ;;
  *)
    echo "Unknown arg: $1 (use start|stop|status|--fg)" >&2
    exit 2
    ;;
esac
