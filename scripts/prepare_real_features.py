#!/usr/bin/env python3
"""Extract lightweight features from real OpenNeuro ds000171 BOLD for the book.

Outputs (committed / WASM-friendly):
  data/processed/participants_clean.csv
  data/processed/events_summary.csv
  data/processed/bold_timeseries.csv     # mean whole-brain BOLD per volume
  data/processed/spectral_features.csv   # subject × task Welch features
  data/processed/book_bundle.json        # compact JSON for browser embed

Run from repo root:
  python scripts/prepare_real_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "ds000171"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
TR = 3.0


def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def load_participants() -> pd.DataFrame:
    df = pd.read_csv(DATA / "participants.tsv", sep="\t")
    df["group_short"] = df["group"].map(
        {
            "Major Depressive Disorder": "MDD",
            "Never-Depressed Control": "Control",
        }
    )
    return df


def inventory_events() -> pd.DataFrame:
    rows = []
    for ev in sorted(DATA.glob("sub-*/func/*_events.tsv")):
        parts = ev.name.replace("_events.tsv", "").split("_")
        sub = parts[0]
        task = parts[1].replace("task-", "")
        run = int(parts[2].replace("run-", ""))
        edf = pd.read_csv(ev, sep="\t")
        rows.append(
            {
                "subject": sub,
                "task": task,
                "run": run,
                "n_events": len(edf),
                "trial_types": ",".join(sorted(edf["trial_type"].astype(str).unique())),
                "duration_sum": float(edf["duration"].sum()) if "duration" in edf else np.nan,
            }
        )
    return pd.DataFrame(rows)


def extract_mean_bold(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    # mean over spatial dims → time series
    ts = data.mean(axis=(0, 1, 2))
    # z-score
    ts = (ts - ts.mean()) / (ts.std() + 1e-8)
    return ts.astype(np.float64)


def spectral_feats(ts: np.ndarray, tr: float = TR) -> dict:
    fs = 1.0 / tr
    nper = min(32, max(8, len(ts) // 3))
    f, pxx = signal.welch(ts, fs=fs, nperseg=nper)
    total = trapz(pxx, f) + 1e-12
    bands = {
        "power_low": ((0.01, 0.04),),
        "power_mid": ((0.04, 0.08),),
        "power_high": ((0.08, 0.15),),
    }
    out = {"total_power": total, "spectral_centroid": float(np.sum(f * pxx) / total)}
    for name, ((lo, hi),) in bands.items():
        m = (f >= lo) & (f <= hi)
        out[name] = trapz(pxx[m], f[m]) / total if np.any(m) else 0.0
    # keep PSD sample for plotting (downsample)
    out["psd_f"] = f.tolist()
    out["psd_pxx"] = pxx.tolist()
    return out


def event_aligned_peak(ts: np.ndarray, events_path: Path, tr: float = TR) -> dict:
    if not events_path.exists():
        return {"peak_latency_s": np.nan, "peak_amp": np.nan}
    edf = pd.read_csv(events_path, sep="\t")
    # positive / music-like trials
    mask = edf["trial_type"].astype(str).str.contains("positive|music|pos", case=False, na=False)
    if not mask.any():
        mask = edf["trial_type"].notna()
    onsets = edf.loc[mask, "onset"].astype(float).values
    if len(onsets) == 0:
        return {"peak_latency_s": np.nan, "peak_amp": np.nan}
    # stack peri-stimulus windows 0–30s
    win = int(30 / tr)
    segs = []
    for o in onsets:
        i0 = int(round(o / tr))
        if i0 + win <= len(ts) and i0 >= 0:
            segs.append(ts[i0 : i0 + win])
    if not segs:
        return {"peak_latency_s": np.nan, "peak_amp": np.nan}
    mean_seg = np.mean(np.stack(segs), axis=0)
    peak_i = int(np.argmax(mean_seg))
    return {
        "peak_latency_s": float(peak_i * tr),
        "peak_amp": float(mean_seg[peak_i]),
        "peri_stim": mean_seg.tolist(),
    }


def main() -> None:
    parts = load_participants()
    parts.to_csv(OUT / "participants_clean.csv", index=False)
    print("participants:", len(parts))

    events = inventory_events()
    events.to_csv(OUT / "events_summary.csv", index=False)
    print("event files:", len(events))

    ts_rows = []
    feat_rows = []
    bold_files = sorted(DATA.glob("sub-*/func/*_bold.nii.gz"))
    print("BOLD files:", len(bold_files))

    for bold in bold_files:
        name = bold.name.replace("_bold.nii.gz", "")
        bits = name.split("_")
        sub, task, run_s = bits[0], bits[1].replace("task-", ""), int(bits[2].replace("run-", ""))
        grp = parts.loc[parts.participant_id == sub, "group_short"]
        group = grp.iloc[0] if len(grp) else ("MDD" if "mdd" in sub else "Control")
        print(f"  processing {bold.relative_to(DATA)} …")
        ts = extract_mean_bold(bold)
        for t_i, val in enumerate(ts):
            ts_rows.append(
                {
                    "subject": sub,
                    "group": group,
                    "task": task,
                    "run": run_s,
                    "volume": t_i,
                    "time": t_i * TR,
                    "bold_z": float(val),
                }
            )
        feats = spectral_feats(ts)
        ev_path = bold.with_name(bold.name.replace("_bold.nii.gz", "_events.tsv"))
        peak = event_aligned_peak(ts, ev_path)
        feat_rows.append(
            {
                "subject": sub,
                "group": group,
                "task": task,
                "run": run_s,
                "n_volumes": len(ts),
                "power_low": feats["power_low"],
                "power_mid": feats["power_mid"],
                "power_high": feats["power_high"],
                "spectral_centroid": feats["spectral_centroid"],
                "total_power": feats["total_power"],
                "peak_latency_s": peak["peak_latency_s"],
                "peak_amp": peak["peak_amp"],
                "psd_f": json.dumps(feats["psd_f"]),
                "psd_pxx": json.dumps(feats["psd_pxx"]),
                "peri_stim": json.dumps(peak.get("peri_stim", [])),
            }
        )

    ts_df = pd.DataFrame(ts_rows)
    feat_df = pd.DataFrame(feat_rows)
    ts_df.to_csv(OUT / "bold_timeseries.csv", index=False)
    feat_df.to_csv(OUT / "spectral_features.csv", index=False)

    # Compact bundle for WASM (truncate long series slightly if huge)
    bundle = {
        "source": "OpenNeuro ds000171 (subset processed locally)",
        "tr_sec": TR,
        "n_participants_full": int(len(parts)),
        "n_bold_runs": int(len(feat_df)),
        "participants": parts.to_dict(orient="records"),
        "events_summary": events.to_dict(orient="records"),
        "spectral_features": feat_df.drop(columns=["psd_f", "psd_pxx", "peri_stim"], errors="ignore").to_dict(
            orient="records"
        ),
        # keep one PSD example per group×task for plots
        "psd_examples": {},
        "peri_examples": {},
        "timeseries_examples": {},
    }
    for _, row in feat_df.iterrows():
        key = f"{row['group']}_{row['task']}"
        if key not in bundle["psd_examples"]:
            bundle["psd_examples"][key] = {
                "subject": row["subject"],
                "f": json.loads(row["psd_f"]),
                "pxx": json.loads(row["psd_pxx"]),
            }
        if key not in bundle["peri_examples"] and row["peri_stim"] not in ("[]", ""):
            bundle["peri_examples"][key] = {
                "subject": row["subject"],
                "peri": json.loads(row["peri_stim"]),
            }
    # mean timeseries sample per subject-task (first run only) for chapter 1
    for (sub, task), g in ts_df.groupby(["subject", "task"]):
        key = f"{sub}_{task}"
        if key not in bundle["timeseries_examples"]:
            g2 = g.sort_values("volume")
            bundle["timeseries_examples"][key] = {
                "subject": sub,
                "task": task,
                "group": g2["group"].iloc[0],
                "time": g2["time"].tolist(),
                "bold_z": g2["bold_z"].tolist(),
            }

    (OUT / "book_bundle.json").write_text(json.dumps(bundle))
    print("Wrote", OUT)
    print(feat_df[["subject", "group", "task", "run", "power_high", "peak_latency_s"]])


if __name__ == "__main__":
    main()
