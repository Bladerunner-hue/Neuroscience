#!/usr/bin/env python3
"""Build all ADR notebooks."""

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
"""

EXPORT = """
from pathlib import Path
nb_name = Path.cwd().name if False else "{name}"
!jupyter nbconvert --to html notebooks/{name}.ipynb --output-dir notebooks/html 2>/dev/null || true
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


# 01 PreFlight
nb(
    "01 — PreFlight: Data Intake & Quality Control",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import json, psutil
import pandas as pd
import nibabel as nib
from neuro.config import DATA_ROOT, FIGURES_DIR, TR_SEC
from neuro.bids import validate_bids, group_counts
from neuro.spark_utils import get_spark, participants_spark
from neuro import viz

report = validate_bids()
participants = report["participants"]
runs = report["runs"]
print("=== BIDS Pre-flight ===")
for k, v in report.items():
    if k not in ("missing", "runs", "participants"):
        print(f"{k}: {v}")
print("\\nGroup counts:")
print(group_counts())
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
viz.plot_tr_histogram(runs)
report["missing"].head(10)
"""},
        {"type": "code", "text": EXPORT.format(name="01_pre_flight")},
    ],
    "01_pre_flight",
)

# 02 EDA Univariate
nb(
    "02 — EDA Univariate",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from neuro.bids import validate_bids, load_participants
from neuro.features import parse_events
from neuro import viz

report = validate_bids()
runs = report["runs"]
participants = load_participants()
available = runs[runs["bold_exists"]]
print(f"Available runs: {len(available)}")
"""},
        {"type": "code", "text": """
# T1w intensity histogram (one subject)
t1_row = available.iloc[0]
t1 = nib.load(t1_row["t1_path"] if t1_row["t1_path"] else available[available["t1_path"].notna()].iloc[0]["t1_path"])
t1_data = t1.get_fdata()
plt.figure(figsize=(6,4))
plt.hist(t1_data[t1_data > 0].ravel(), bins=80, color="#8172B3")
plt.title("T1w voxel intensities (non-zero)")
plt.xlabel("Intensity"); plt.ylabel("Count")
viz.savefig("02_t1w_histogram.png")
"""},
        {"type": "code", "text": """
# BOLD mean signal per volume (carpet plot proxy)
bold = nib.load(available.iloc[0]["bold_path"])
data = bold.get_fdata()
mean_ts = data.reshape(-1, data.shape[-1]).mean(axis=0)
plt.figure(figsize=(10,3))
plt.plot(mean_ts, color="#4C72B0")
plt.title(f"Global mean BOLD — {available.iloc[0]['subject']}")
plt.xlabel("Volume"); plt.ylabel("Mean signal")
viz.savefig("02_bold_mean_timeseries.png")
"""},
        {"type": "code", "text": """
# Events / stimulus distribution
ev = parse_events(available.iloc[0]["events_path"])
viz.plot_trial_counts(pd.read_csv(available.iloc[0]["events_path"], sep="\\t"))
participants.groupby(["group_short","sex"]).size().unstack(fill_value=0)
"""},
        {"type": "code", "text": EXPORT.format(name="02_eda_univariate")},
    ],
    "02_eda_univariate",
)

# 03 EDA Multivariate
nb(
    "03 — EDA Multivariate: Group Differences",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
from neuro.bids import validate_bids
from neuro.features import get_schaefer_masker, extract_roi_timeseries, connectivity_matrix
from neuro import viz

report = validate_bids()
runs = report["runs"][report["runs"]["bold_exists"]].head(6)  # subset for speed
masker = get_schaefer_masker(n_rois=50)
conn_mats = []
labels = []
for _, row in runs.iterrows():
    ts = extract_roi_timeseries(row["bold_path"], masker)
    conn_mats.append(connectivity_matrix(ts))
    labels.append(f"{row['subject']}_{row['task']}")
mean_conn = np.mean(conn_mats, axis=0)
viz.plot_connectivity_heatmap(mean_conn, "Mean ROI connectivity (n=6 runs)", "03_connectivity_mean.png")
"""},
        {"type": "code", "text": """
# Group-level mean connectivity (MDD vs ND) — first available per subject
from collections import defaultdict
sub_conn = defaultdict(list)
for _, row in report["runs"][report["runs"]["bold_exists"]].iterrows():
    ts = extract_roi_timeseries(row["bold_path"], masker)
    sub_conn[(row["subject"], row["group_short"])].append(connectivity_matrix(ts))

mdd = [np.mean(v, axis=0) for (_, g), v in sub_conn.items() if g == "MDD"][:5]
nd = [np.mean(v, axis=0) for (_, g), v in sub_conn.items() if g == "ND"][:5]
if mdd and nd:
    diff = np.mean(mdd, axis=0) - np.mean(nd, axis=0)
    viz.plot_connectivity_heatmap(diff, "MDD − ND connectivity diff", "03_connectivity_group_diff.png")
"""},
        {"type": "code", "text": EXPORT.format(name="03_eda_multivariate")},
    ],
    "03_eda_multivariate",
)

# 04 Feature Engineering
nb(
    "04 — Feature Engineering & Spark Scaling",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructType, StructField, StringType, IntegerType
from neuro.bids import validate_bids
from neuro.features import build_feature_table, save_features_parquet, roi_summary_features, extract_roi_timeseries, get_schaefer_masker
from neuro.spark_utils import get_spark

report = validate_bids()
runs = report["runs"]
spark = get_spark("BladerunnerNeuro_Features")
runs_pdf = runs[runs["bold_exists"]].copy()
print(f"Building features for {len(runs_pdf)} runs")
"""},
        {"type": "code", "text": """
# Spark UDF wrapper for ROI mean (distributed metadata aggregation)
masker = get_schaefer_masker(n_rois=50)

@F.udf(returnType=FloatType())
def roi_mean_scalar(path):
    ts = extract_roi_timeseries(path, masker)
    return float(ts.mean())

files_sdf = spark.createDataFrame(runs_pdf[["subject","task","run","bold_path","group_short"]])
files_sdf = files_sdf.withColumn("roi_global_mean", roi_mean_scalar(F.col("bold_path")))
files_sdf.groupBy("group_short").avg("roi_global_mean").show()
"""},
        {"type": "code", "text": """
# Full feature table (local; saves parquet for TF)
feat_df = build_feature_table(runs_pdf, n_rois=50)
path = save_features_parquet(feat_df)
print("Saved:", path)
feat_df[["subject","task","ts_mean","ts_entropy"]].head()
"""},
        {"type": "code", "text": EXPORT.format(name="04_feature_engineering")},
    ],
    "04_feature_engineering",
)

# 05 Preprocessing Pipeline
nb(
    "05 — Preprocessing Pipeline",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
from pathlib import Path
from neuro.config import PROCESSED_DIR
from neuro.bids import validate_bids
from neuro.features import get_schaefer_masker, extract_roi_timeseries
from nilearn.image import smooth_img
import nibabel as nib
import matplotlib.pyplot as plt
from neuro import viz

report = validate_bids()
row = report["runs"][report["runs"]["bold_exists"]].iloc[0]
raw = nib.load(row["bold_path"])
smoothed = smooth_img(raw, fwhm=6)
masker = get_schaefer_masker(n_rois=50)
ts_clean = masker.fit_transform(smoothed)
np.save(PROCESSED_DIR / "example_preprocessed_ts.npy", ts_clean)
print("Preprocessed shape:", ts_clean.shape)
"""},
        {"type": "code", "text": """
# Before/after smoothing — middle volume
mid = raw.shape[3] // 2
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(raw.get_fdata()[:, :, raw.shape[2]//2, mid], cmap="gray")
ax[0].set_title("Raw BOLD (mid slice/vol)")
ax[1].imshow(smoothed.get_fdata()[:, :, smoothed.shape[2]//2, mid], cmap="gray")
ax[1].set_title("Smoothed 6mm FWHM")
viz.savefig("05_preprocessing_before_after.png")
"""},
        {"type": "code", "text": EXPORT.format(name="05_preprocessing_pipeline")},
    ],
    "05_preprocessing_pipeline",
)

# 06 Modeling DL
nb(
    "06 — Deep Learning & Clustering (Transformer / LSTM)",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from neuro.config import PROCESSED_DIR
from neuro.models import build_roi_transformer, build_lstm_classifier, make_tf_dataset
from neuro import viz

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", gpus)
"""},
        {"type": "code", "text": """
X = np.load(PROCESSED_DIR / "roi_ts_stack.npy")
labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
le = LabelEncoder()
y = le.fit_transform(labels["group_short"])
n_time, n_rois = X.shape[1], X.shape[2]
print(X.shape, "classes:", le.classes_)
"""},
        {"type": "code", "text": """
# Subject-level split (mean ROI ts per subject)
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
"""},
        {"type": "code", "text": """
# LSTM baseline + KMeans clustering on connectivity features
conn = np.load(PROCESSED_DIR / "conn_stack.npy")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(conn)
print("KMeans inertia:", kmeans.inertia_)
lstm = build_lstm_classifier(n_rois, n_time)
lstm.fit(make_tf_dataset(subj_X, subj_y, batch_size=4), epochs=10, validation_split=0.2, verbose=0)
"""},
        {"type": "code", "text": EXPORT.format(name="06_modeling_dl")},
    ],
    "06_modeling_dl",
)

# 07 Visual Results
nb(
    "07 — Scientific Visualisation & Interpretation",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from neuro.config import FIGURES_DIR, PROCESSED_DIR
from neuro.bids import load_participants

participants = load_participants()
labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
figs = sorted(FIGURES_DIR.glob("*.png"))
print("Generated figures:", [f.name for f in figs])
"""},
        {"type": "code", "text": """
# Summary dashboard
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
participants.groupby("group_short").size().plot(kind="bar", ax=axes[0,0], color=["#4C72B0","#DD8452"])
axes[0,0].set_title("Cohort balance")
feat = pd.read_parquet(PROCESSED_DIR / "roi_features.parquet")
if "ts_mean" in feat.columns:
    sns.boxplot(data=feat, x="group_short", y="ts_mean", ax=axes[0,1])
    axes[0,1].set_title("ROI mean signal by group")
for i, fname in enumerate(["03_connectivity_mean.png", "06_training_history.png"]):
    p = FIGURES_DIR / fname
    if p.exists():
        axes[1, i].imshow(plt.imread(p))
        axes[1, i].axis("off")
        axes[1, i].set_title(fname)
plt.savefig(FIGURES_DIR / "07_summary_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
"""},
        {"type": "code", "text": EXPORT.format(name="07_visual_results")},
    ],
    "07_visual_results",
)

# 08 Conclusion
nb(
    "08 — Conclusion & Reproducibility",
    [
        {"type": "code", "text": SETUP},
        {"type": "code", "text": """
import json
import platform
from pathlib import Path
import tensorflow as tf
import pyspark
import nibabel
import nilearn
from neuro.bids import validate_bids

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
    "nibabel": nibabel.__version__,
    "nilearn": nilearn.__version__,
}
print(json.dumps(summary, indent=2))
Path("notebooks/html").mkdir(parents=True, exist_ok=True)
"""},
        {"type": "code", "text": '''
summary_md = """
## Summary
- BIDS ds000171: emotional music/non-music fMRI in MDD vs controls
- Pipeline: Spark ingestion -> Nilearn preprocessing -> ROI features -> TF transformer
- Limitations: small N=39; cross-subject CV essential; TR=3s (not 2s)
- Next: fMRIPrep, larger atlas, self-supervised pretrain, TF Serving pass 3
"""
print(summary_md)
'''},
        {"type": "code", "text": EXPORT.format(name="08_conclusion_reproducibility")},
    ],
    "08_conclusion_reproducibility",
)

print("Built notebooks in", NB)