#!/usr/bin/env python3
"""Build all ADR notebooks — inline seaborn/sklearn/mlflow, no HTML/PNG export."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"

SETUP = """import sys
from pathlib import Path
REPO = Path.cwd()
for candidate in [REPO, REPO.parent, REPO.parent.parent]:
    if (candidate / "src" / "neuro").exists():
        REPO = candidate
        break
sys.path.insert(0, str(REPO / "src"))
import os
os.chdir(REPO)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
sns.set_theme(style="whitegrid", context="notebook")
"""


def nb(title: str, cells: list, name: str) -> None:
    doc = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.10 (neuro)",
                "language": "python",
                "name": "neuro-310",
            }
        },
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]}],
    }
    for c in cells:
        if c["type"] == "md":
            doc["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [c["text"]]})
        else:
            doc["cells"].append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": [ln + "\n" for ln in c["text"].splitlines()],
                }
            )
    (NB / f"{name}.ipynb").write_text(json.dumps(doc, indent=1))


nb(
    "01 — PreFlight: Data Intake & Quality Control",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import json, psutil
import pandas as pd
import nibabel as nib
import mlflow
from neuro.config import DATA_ROOT, TR_SEC
from neuro.bids import validate_bids, group_counts
from neuro.spark_utils import get_spark, participants_spark
from neuro.mlflow_utils import start_run
from neuro import viz

with start_run("01_pre_flight"):
    report = validate_bids()
    participants = report["participants"]
    runs = report["runs"]
    mlflow.log_param("n_subjects", report["n_subjects"])
    mlflow.log_param("tr_sec", report["tr_json_music"])
    print("=== BIDS Pre-flight ===")
    for k, v in report.items():
        if k not in ("missing", "runs", "participants"):
            print(f"{k}: {v}")
    print("\\nGroup counts:")
    display(group_counts())
    print(f"\\nMemory available: {psutil.virtual_memory().available / 1e9:.1f} GB")
"""},
        {"type": "code", "text": """
spark = get_spark("BladerunnerNeuro_PreFlight")
spark.sql("SELECT 'Spark ready for scaled preprocessing' AS status").show()
participants_spark(spark, participants).groupBy("group_short").count().show()
"""},
        {"type": "code", "text": """
if report["n_runs_available"]:
    sample = runs[runs["bold_exists"]].iloc[0]
    img = nib.load(sample["bold_path"])
    print("Sample:", sample["bold_path"])
    print("Shape:", img.shape, "Zooms:", img.header.get_zooms())
viz.plot_group_demographics(participants)
plt.show()
viz.plot_tr_histogram(runs)
plt.show()
report["missing"].head(10)
"""},
    ],
    "01_pre_flight",
)

nb(
    "02 — EDA Univariate",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from neuro.bids import validate_bids, load_participants
from neuro.features import parse_events
from neuro.mlflow_utils import start_run
from neuro import viz

with start_run("02_eda_univariate"):
    report = validate_bids()
    runs = report["runs"]
    participants = load_participants()
    available = runs[runs["bold_exists"]]
    mlflow.log_param("n_runs", len(available))
    print(f"Available runs: {len(available)}")
"""},
        {"type": "code", "text": """
t1_path = available[available["t1_path"].notna()].iloc[0]["t1_path"]
t1_data = nib.load(t1_path).get_fdata()
t1_vals = t1_data[t1_data > 0].ravel()
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(t1_vals, bins=80, kde=True, color="#8172B3", ax=ax)
ax.set_title("T1w voxel intensities (non-zero)")
ax.set_xlabel("Intensity")
plt.show()
"""},
        {"type": "code", "text": """
row = available.iloc[0]
bold = nib.load(row["bold_path"])
data = bold.get_fdata()
mean_ts = data.reshape(-1, data.shape[-1]).mean(axis=0)
ts_df = pd.DataFrame({"volume": np.arange(len(mean_ts)), "signal": mean_ts})
fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(data=ts_df, x="volume", y="signal", ax=ax, color="#4C72B0")
ax.set_title(f"Global mean BOLD — {row['subject']}")
plt.show()
"""},
        {"type": "code", "text": """
events = pd.read_csv(row["events_path"], sep="\\t")
viz.plot_trial_counts(events)
plt.show()
demo = participants.groupby(["group_short", "sex"]).size().reset_index(name="n")
sns.barplot(data=demo, x="group_short", y="n", hue="sex", palette="Set2")
plt.title("Participants by group and sex")
plt.show()
display(demo)
"""},
    ],
    "02_eda_univariate",
)

nb(
    "03 — EDA Multivariate: Group Differences",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from neuro.bids import validate_bids
from neuro.features import get_schaefer_masker, extract_roi_timeseries, connectivity_matrix
from neuro.mlflow_utils import start_run
from neuro import viz

with start_run("03_eda_multivariate"):
    report = validate_bids()
    runs = report["runs"][report["runs"]["bold_exists"]].head(4)
    masker = get_schaefer_masker(n_rois=100)
    mlflow.log_param("n_rois", 100)
    conn_mats = []
    for _, row in runs.iterrows():
        ts = extract_roi_timeseries(row["bold_path"], masker)
        conn_mats.append(connectivity_matrix(ts))
    mean_conn = np.mean(conn_mats, axis=0)
    viz.plot_connectivity_heatmap(mean_conn, "Mean ROI connectivity (subset)")
    plt.show()
"""},
        {"type": "code", "text": """
from collections import defaultdict
sub_conn = defaultdict(list)
for _, row in report["runs"][report["runs"]["bold_exists"]].iterrows():
    ts = extract_roi_timeseries(row["bold_path"], masker)
    sub_conn[(row["subject"], row["group_short"])].append(connectivity_matrix(ts))

records = []
for (sub, grp), mats in sub_conn.items():
    vec = np.mean(mats, axis=0)[np.triu_indices_from(np.mean(mats, axis=0), k=1)]
    records.append({"subject": sub, "group": grp, "features": vec})
feat_df = pd.DataFrame(records)
X = np.stack(feat_df["features"].values)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
plot_df = feat_df[["subject", "group"]].copy()
plot_df["PC1"], plot_df["PC2"] = coords[:, 0], coords[:, 1]
mlflow.log_metric("pca_explained_var_pc1", float(pca.explained_variance_ratio_[0]))
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="group", style="group", s=120, ax=ax)
ax.set_title("PCA of ROI connectivity (per subject)")
plt.show()

mdd = [np.mean(v, axis=0) for (_, g), v in sub_conn.items() if g == "MDD"][:5]
nd = [np.mean(v, axis=0) for (_, g), v in sub_conn.items() if g == "ND"][:5]
if mdd and nd:
    diff = np.mean(mdd, axis=0) - np.mean(nd, axis=0)
    viz.plot_connectivity_heatmap(diff, "MDD − ND connectivity diff")
    plt.show()
"""},
    ],
    "03_eda_multivariate",
)

nb(
    "04 — Feature Engineering & Spark Scaling",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from sklearn.preprocessing import StandardScaler
import mlflow
from neuro.bids import validate_bids
from neuro.features import build_feature_table, save_features_parquet, extract_roi_timeseries, get_schaefer_masker
from neuro.spark_utils import get_spark
from neuro.mlflow_utils import start_run

with start_run("04_feature_engineering"):
    report = validate_bids()
    runs = report["runs"]
    spark = get_spark("BladerunnerNeuro_Features")
    runs_pdf = runs[runs["bold_exists"]].copy()
    mlflow.log_param("n_runs", len(runs_pdf))
    print(f"Building features for {len(runs_pdf)} runs")
"""},
        {"type": "code", "text": """
masker = get_schaefer_masker(n_rois=100)

@F.udf(returnType=FloatType())
def roi_mean_scalar(path):
    ts = extract_roi_timeseries(path, masker)
    return float(ts.mean())

files_sdf = spark.createDataFrame(runs_pdf[["subject", "task", "run", "bold_path", "group_short"]])
files_sdf = files_sdf.withColumn("roi_global_mean", roi_mean_scalar(F.col("bold_path")))
spark_means = files_sdf.groupBy("group_short").avg("roi_global_mean").toPandas()
display(spark_means)
sns.barplot(data=spark_means, x="group_short", y="avg(roi_global_mean)", hue="group_short", palette="muted", legend=False)
plt.title("Spark-aggregated ROI global mean by group")
plt.show()
"""},
        {"type": "code", "text": """
feat_df = build_feature_table(runs_pdf, n_rois=100)
path = save_features_parquet(feat_df)
mlflow.log_artifact(str(path))
summary = feat_df[["ts_mean", "ts_std", "ts_entropy"]].describe()
display(summary)
sns.pairplot(feat_df, vars=["ts_mean", "ts_std", "ts_entropy"], hue="group_short", corner=True, plot_kws={"alpha": 0.6})
plt.show()
"""},
    ],
    "04_feature_engineering",
)

nb(
    "05 — Preprocessing Pipeline",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
from neuro.config import PROCESSED_DIR
from neuro.bids import validate_bids
from neuro.features import get_schaefer_masker
from nilearn.image import smooth_img
import nibabel as nib
from neuro.mlflow_utils import start_run

with start_run("05_preprocessing"):
    report = validate_bids()
    row = report["runs"][report["runs"]["bold_exists"]].iloc[0]
    raw = nib.load(row["bold_path"])
    smoothed = smooth_img(raw, fwhm=6)
    masker = get_schaefer_masker(n_rois=100)
    ts_clean = masker.fit_transform(smoothed)
    np.save(PROCESSED_DIR / "example_preprocessed_ts.npy", ts_clean)
    mlflow.log_param("fwhm_mm", 6)
    print("Preprocessed shape:", ts_clean.shape)
"""},
        {"type": "code", "text": """
mid = raw.shape[3] // 2
z = raw.shape[2] // 2
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(raw.get_fdata()[:, :, z, mid], cmap="gray")
axes[0].set_title("Raw BOLD (mid slice/vol)")
axes[1].imshow(smoothed.get_fdata()[:, :, z, mid], cmap="gray")
axes[1].set_title("Smoothed 6mm FWHM")
plt.show()
import pandas as pd
ts_df = pd.DataFrame(ts_clean, columns=[f"roi_{i}" for i in range(ts_clean.shape[1])])
sns.lineplot(data=ts_df.iloc[:, :5], dashes=False)
plt.title("Preprocessed ROI time series (first 5 ROIs)")
plt.xlabel("Volume")
plt.show()
"""},
    ],
    "05_preprocessing_pipeline",
)

nb(
    "06 — Deep Learning & Clustering (Transformer / LSTM)",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from neuro.config import PROCESSED_DIR
from neuro.models import build_roi_transformer, build_lstm_classifier, make_tf_dataset
from neuro.mlflow_utils import start_run
from neuro import viz

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", gpus)
"""},
        {"type": "code", "text": """
with start_run("06_modeling_dl"):
    X = np.load(PROCESSED_DIR / "roi_ts_stack.npy")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    le = LabelEncoder()
    y = le.fit_transform(labels["group_short"])
    n_time, n_rois = X.shape[1], X.shape[2]

    subj_X, subj_y = [], []
    for sub in labels["subject"].unique():
        idx = labels["subject"] == sub
        subj_X.append(X[idx].mean(axis=0))
        subj_y.append(y[idx][0])
    subj_X = np.stack(subj_X)
    subj_y = np.array(subj_y)

    model = build_roi_transformer(n_rois, n_time, embed_dim=32, num_heads=2, num_layers=2)
    ds = make_tf_dataset(subj_X, subj_y, batch_size=4)
    history = model.fit(ds, epochs=15, validation_split=0.2, verbose=1)
    viz.plot_training_history(history)
    plt.show()
    mlflow.log_metric("final_loss", float(history.history["loss"][-1]))
    mlflow.log_metric("final_accuracy", float(history.history["accuracy"][-1]))
"""},
        {"type": "code", "text": """
    conn = np.load(PROCESSED_DIR / "conn_stack.npy")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(conn)
    mlflow.log_metric("kmeans_inertia", float(kmeans.inertia_))

    preds = np.argmax(model.predict(subj_X, verbose=0), axis=1)
    cm = confusion_matrix(subj_y, preds)
    viz.plot_confusion_matrix(cm, list(le.classes_))
    plt.show()
    print(classification_report(subj_y, preds, target_names=le.classes_))

    lstm = build_lstm_classifier(n_rois, n_time)
    lstm_hist = lstm.fit(make_tf_dataset(subj_X, subj_y, batch_size=4), epochs=10, validation_split=0.2, verbose=0)
    viz.plot_training_history(lstm_hist)
    plt.show()
"""},
    ],
    "06_modeling_dl",
)

nb(
    "07 — Scientific Visualisation & Interpretation",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import pandas as pd
from neuro.config import PROCESSED_DIR
from neuro.bids import load_participants
from neuro.mlflow_utils import start_run

with start_run("07_visual_results"):
    participants = load_participants()
    feat = pd.read_parquet(PROCESSED_DIR / "roi_features.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
"""},
        {"type": "code", "text": """
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.countplot(data=participants, x="group_short", hue="group_short", ax=axes[0, 0], palette="Set2", legend=False)
axes[0, 0].set_title("Cohort balance")
sns.boxplot(data=feat, x="group_short", y="ts_mean", hue="group_short", ax=axes[0, 1], palette="Set2", legend=False)
axes[0, 1].set_title("ROI mean signal by group")
sns.boxplot(data=feat, x="task", y="ts_entropy", hue="group_short", ax=axes[1, 0])
axes[1, 0].set_title("Temporal entropy by task")
sns.scatterplot(data=feat, x="ts_mean", y="ts_std", hue="group_short", style="task", ax=axes[1, 1], alpha=0.7)
axes[1, 1].set_title("Mean vs std by group")
plt.tight_layout()
plt.show()
"""},
    ],
    "07_visual_results",
)

nb(
    "08 — Conclusion & Reproducibility",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import json
import platform
import tensorflow as tf
import pyspark
import nibabel
import nilearn
import sklearn
from neuro.bids import validate_bids
from neuro.mlflow_utils import start_run

with start_run("08_conclusion"):
    report = validate_bids()
    summary = {
        "dataset": "ds000171",
        "subjects": report["n_subjects"],
        "mdd": report["n_mdd"],
        "nd": report["n_nd"],
        "runs_available": report["n_runs_available"],
        "tr_sec": report["tr_json_music"],
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "pyspark": pyspark.__version__,
        "sklearn": sklearn.__version__,
        "nibabel": nibabel.__version__,
        "nilearn": nilearn.__version__,
    }
    for k, v in summary.items():
        if isinstance(v, (int, float, str)):
            mlflow.log_param(k, v)
    print(json.dumps(summary, indent=2))
"""},
        {"type": "code", "text": '''
print("""
## Summary
- BIDS ds000171: emotional music/non-music fMRI in MDD vs controls
- Pipeline: Spark ingestion -> Nilearn preprocessing -> ROI features -> TF transformer
- Visualisation: inline seaborn/matplotlib in notebooks (no HTML/PNG export)
- Tracking: mlflow experiments in mlruns/
- Limitations: small N=39; cross-subject CV essential; TR=3s
""")
'''},
    ],
    "08_conclusion_reproducibility",
)

print("Built notebooks in", NB)