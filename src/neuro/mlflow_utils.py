"""MLflow helpers for notebook experiments."""

from __future__ import annotations

from pathlib import Path

import mlflow

from neuro.config import REPO_ROOT

EXPERIMENT = "neuroscience-ds000171"


def start_run(notebook_name: str):
    mlflow.set_tracking_uri(REPO_ROOT / "mlruns")
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.start_run(run_name=notebook_name)