"""Publication-quality figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from neuro.config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="notebook")


def savefig(name: str, dpi: int = 150) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_group_demographics(participants: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    participants.groupby("group_short").size().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Group counts")
    axes[0].set_ylabel("N")
    sns.boxplot(data=participants, x="group_short", y="age", ax=axes[1])
    axes[1].set_title("Age by group")
    return savefig("01_group_demographics.png")


def plot_tr_histogram(runs: pd.DataFrame) -> Path:
    plt.figure(figsize=(6, 4))
    runs.loc[runs["bold_exists"], "tr"].dropna().hist(bins=20, color="#55A868")
    plt.xlabel("TR (s)")
    plt.ylabel("Runs")
    plt.title("Repetition time distribution")
    return savefig("01_tr_distribution.png")


def plot_trial_counts(events_df: pd.DataFrame) -> Path:
    plt.figure(figsize=(8, 4))
    events_df["trial_type"].value_counts().plot(kind="barh", color="#C44E52")
    plt.title("Stimulus trial types (example subject)")
    return savefig("02_trial_counts.png")


def plot_connectivity_heatmap(conn: np.ndarray, title: str, fname: str) -> Path:
    plt.figure(figsize=(7, 6))
    sns.heatmap(conn, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True)
    plt.title(title)
    return savefig(fname)


def plot_training_history(history, fname: str = "06_training_history.png") -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    return savefig(fname)