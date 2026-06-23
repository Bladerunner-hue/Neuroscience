"""Inline notebook visualizations (seaborn + matplotlib, no file export)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def plot_group_demographics(participants: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=participants, x="group_short", hue="group_short", ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("Group counts")
    sns.boxplot(data=participants, x="group_short", y="age", hue="group_short", ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title("Age by group")
    fig.tight_layout()
    return fig


def plot_tr_histogram(runs: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    tr = runs.loc[runs["bold_exists"], "tr"].dropna()
    sns.histplot(tr, bins=20, color="#55A868", ax=ax)
    ax.set_xlabel("TR (s)")
    ax.set_ylabel("Runs")
    ax.set_title("Repetition time distribution")
    fig.tight_layout()
    return fig


def plot_trial_counts(events_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = events_df["trial_type"].value_counts().reset_index()
    counts.columns = ["trial_type", "count"]
    sns.barplot(data=counts, y="trial_type", x="count", hue="trial_type", ax=ax, palette="rocket", legend=False)
    ax.set_title("Stimulus trial types")
    fig.tight_layout()
    return fig


def plot_connectivity_heatmap(conn: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(conn, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_training_history(history) -> plt.Figure:
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
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str = "Confusion matrix") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig