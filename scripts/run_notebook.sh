#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export SPARK_MASTER="${SPARK_MASTER:-local[*]}"
NB="$1"
echo "=== Executing $NB ==="
jupyter nbconvert --to notebook --execute "notebooks/${NB}.ipynb" \
  --output "notebooks/${NB}.ipynb" \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=python3
echo "=== Done $NB ==="