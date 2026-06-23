"""MLflow helpers for notebook experiments."""

from __future__ import annotations

import mlflow

from neuro.config import REPO_ROOT

EXPERIMENT = "neuroscience-ds000171"
TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"


def start_run(notebook_name: str):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.start_run(run_name=notebook_name)