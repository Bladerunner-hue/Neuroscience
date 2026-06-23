"""Feature extraction: ROI time series, connectivity, stimulus-locked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import resample_to_img, smooth_img
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import entropy, skew

from neuro.config import PROCESSED_DIR


SCHAEFER_VALID_ROIS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)


def get_schaefer_masker(n_rois: int = 100) -> NiftiLabelsMasker:
    if n_rois not in SCHAEFER_VALID_ROIS:
        raise ValueError(
            f"n_rois must be one of {SCHAEFER_VALID_ROIS}, got {n_rois}"
        )
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7)
    return NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=3.0,
    )


def extract_roi_timeseries(bold_path: str | Path, masker: NiftiLabelsMasker) -> np.ndarray:
    from nibabel import load

    img = load(str(bold_path))
    smoothed = smooth_img(img, fwhm=6)
    return masker.fit_transform(smoothed)


def roi_summary_features(ts: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(ts)),
        "std": float(np.std(ts)),
        "skew": float(skew(ts)),
        "entropy": float(entropy(np.histogram(ts, bins=32)[0] + 1e-8)),
    }


def connectivity_matrix(ts: np.ndarray) -> np.ndarray:
    return np.corrcoef(ts, rowvar=False)


def parse_events(events_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(events_path, sep="\t")
    stimulus = df[~df["trial_type"].isin(["response", "tones"])].copy()
    stimulus["valence"] = stimulus["trial_type"].str.replace(
        r"_(music|nonmusic)$", "", regex=True
    )
    return stimulus


def stimulus_locked_mean(
    ts: np.ndarray, tr: float, events: pd.DataFrame, window: tuple[float, float] = (0, 15)
) -> dict[str, float]:
    n_vols, _ = ts.shape
    out: dict[str, list[float]] = {}
    for _, row in events.iterrows():
        onset_vol = int(row["onset"] / tr)
        w_start = onset_vol + int(window[0] / tr)
        w_end = onset_vol + int(window[1] / tr)
        if w_end <= n_vols and w_start >= 0:
            valence = row.get("valence", row["trial_type"])
            out.setdefault(valence, []).append(float(np.mean(ts[w_start:w_end])))
    return {k: float(np.mean(v)) for k, v in out.items()}


def build_feature_table(runs_df: pd.DataFrame, n_rois: int = 100) -> pd.DataFrame:
    available = runs_df[runs_df["bold_exists"]].copy()
    masker = get_schaefer_masker(n_rois=n_rois)
    rows = []
    for _, row in available.iterrows():
        ts = extract_roi_timeseries(row["bold_path"], masker)
        summary = roi_summary_features(ts.mean(axis=1))
        conn = connectivity_matrix(ts)
        stim = {}
        if row["events_path"]:
            stim = stimulus_locked_mean(
                ts, row["tr"] or 3.0, parse_events(row["events_path"])
            )
        rows.append(
            {
                "subject": row["subject"],
                "task": row["task"],
                "run": row["run"],
                "group_short": row["group_short"],
                "roi_ts": ts,
                "conn_upper": conn[np.triu_indices_from(conn, k=1)],
                **{f"ts_{k}": v for k, v in summary.items()},
                **{f"stim_{k}": v for k, v in stim.items()},
            }
        )
    return pd.DataFrame(rows)


def save_features_parquet(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or PROCESSED_DIR / "roi_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    export = df.drop(columns=["roi_ts", "conn_upper"], errors="ignore")
    export.to_parquet(path, index=False)
    np.save(PROCESSED_DIR / "roi_ts_stack.npy", np.stack(df["roi_ts"].values))
    np.save(PROCESSED_DIR / "conn_stack.npy", np.stack(df["conn_upper"].values))
    labels = df[["subject", "group_short", "task"]].reset_index(drop=True)
    labels.to_parquet(PROCESSED_DIR / "labels.parquet", index=False)
    return path