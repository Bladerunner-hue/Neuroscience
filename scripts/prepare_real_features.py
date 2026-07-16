#!/usr/bin/env python3
"""Extract analysis-ready features from real OpenNeuro ds000171 BOLD.

Produces:
  data/processed/participants_clean.csv
  data/processed/events_summary.csv
  data/processed/bold_timeseries.csv
  data/processed/spectral_features.csv      # run-level Welch
  data/processed/condition_features.csv     # trial-type / music-valence effects
  data/processed/subject_features.csv       # subject-level aggregated for ML
  data/processed/book_bundle.json
  + regenerates marimo_notebooks/book_data.py if gen script available

Run:  python scripts/prepare_real_features.py
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

# Canonical stimulus families in this paradigm
STIM_MAP = {
    "positive_music": {"valence": "positive", "domain": "music", "label": "Positive music"},
    "negative_music": {"valence": "negative", "domain": "music", "label": "Negative music"},
    "positive_nonmusic": {"valence": "positive", "domain": "nonmusic", "label": "Positive non-music"},
    "negative_nonmusic": {"valence": "negative", "domain": "nonmusic", "label": "Negative non-music"},
    "tones": {"valence": "neutral", "domain": "tones", "label": "Tones"},
}


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
    # harmonization helpers
    df["age_z"] = (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-8)
    df["sex_m"] = (df["sex"].astype(str).str.upper() == "M").astype(int)
    return df


def inventory_events() -> pd.DataFrame:
    rows = []
    for ev in sorted(DATA.glob("sub-*/func/*_events.tsv")):
        parts = ev.name.replace("_events.tsv", "").split("_")
        sub, task, run = parts[0], parts[1].replace("task-", ""), int(parts[2].replace("run-", ""))
        edf = pd.read_csv(ev, sep="\t")
        types = edf["trial_type"].astype(str)
        row = {
            "subject": sub,
            "task": task,
            "run": run,
            "n_events": len(edf),
            "trial_types": ",".join(sorted(types.unique())),
            "n_positive_music": int((types == "positive_music").sum()),
            "n_negative_music": int((types == "negative_music").sum()),
            "n_positive_nonmusic": int((types == "positive_nonmusic").sum()),
            "n_negative_nonmusic": int((types == "negative_nonmusic").sum()),
            "n_tones": int((types == "tones").sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def extract_mean_bold(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    ts = data.mean(axis=(0, 1, 2))
    return ((ts - ts.mean()) / (ts.std() + 1e-8)).astype(np.float64)


def spectral_feats(ts: np.ndarray, tr: float = TR) -> dict:
    fs = 1.0 / tr
    nper = min(32, max(8, len(ts) // 3))
    f, pxx = signal.welch(ts, fs=fs, nperseg=nper)
    total = trapz(pxx, f) + 1e-12
    out = {
        "total_power": total,
        "spectral_centroid": float(np.sum(f * pxx) / total),
        "psd_f": f.tolist(),
        "psd_pxx": pxx.tolist(),
    }
    for name, lo, hi in [
        ("power_low", 0.01, 0.04),
        ("power_mid", 0.04, 0.08),
        ("power_high", 0.08, 0.15),
    ]:
        m = (f >= lo) & (f <= hi)
        out[name] = trapz(pxx[m], f[m]) / total if np.any(m) else 0.0
    return out


def segment_mean(ts: np.ndarray, onset: float, duration: float, tr: float = TR) -> np.ndarray:
    i0 = max(0, int(round(onset / tr)))
    i1 = min(len(ts), int(round((onset + duration) / tr)))
    if i1 <= i0:
        return np.array([])
    return ts[i0:i1]


def condition_metrics(ts: np.ndarray, events_path: Path) -> list[dict]:
    """Per trial-type summary within one run."""
    if not events_path.exists():
        return []
    edf = pd.read_csv(events_path, sep="\t")
    rows = []
    for trial, g in edf.groupby("trial_type"):
        trial = str(trial)
        if trial == "response":
            continue
        segs = []
        for _, r in g.iterrows():
            seg = segment_mean(ts, float(r["onset"]), float(r.get("duration", 31.5)))
            if len(seg) >= 4:
                segs.append(seg)
        if not segs:
            continue
        cat = np.concatenate(segs)
        meta = STIM_MAP.get(
            trial,
            {"valence": "other", "domain": "other", "label": trial},
        )
        # peak within mean peri-stimulus (align by resampling to same length)
        L = int(np.median([len(s) for s in segs]))
        stacked = np.stack(
            [np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s) for s in segs]
        )
        mean_seg = stacked.mean(axis=0)
        peak_i = int(np.argmax(mean_seg))
        sp = spectral_feats(cat) if len(cat) >= 16 else {
            "power_low": np.nan,
            "power_mid": np.nan,
            "power_high": np.nan,
            "spectral_centroid": np.nan,
            "total_power": np.nan,
            "psd_f": [],
            "psd_pxx": [],
        }
        rows.append(
            {
                "trial_type": trial,
                "valence": meta["valence"],
                "domain": meta["domain"],
                "stim_label": meta["label"],
                "n_epochs": len(segs),
                "mean_bold": float(cat.mean()),
                "std_bold": float(cat.std()),
                "peak_amp": float(mean_seg[peak_i]),
                "peak_latency_s": float(peak_i * TR),
                "power_high": sp["power_high"],
                "power_mid": sp["power_mid"],
                "power_low": sp["power_low"],
                "spectral_centroid": sp["spectral_centroid"],
                "peri_stim": mean_seg.tolist(),
            }
        )
    return rows


def main() -> None:
    parts = load_participants()
    parts.to_csv(OUT / "participants_clean.csv", index=False)
    print("participants:", len(parts))

    events = inventory_events()
    events.to_csv(OUT / "events_summary.csv", index=False)
    print("event files:", len(events))

    ts_rows: list[dict] = []
    feat_rows: list[dict] = []
    cond_rows: list[dict] = []

    bold_files = sorted(DATA.glob("sub-*/func/*_bold.nii.gz"))
    print("BOLD files:", len(bold_files))

    for bold in bold_files:
        name = bold.name.replace("_bold.nii.gz", "")
        bits = name.split("_")
        sub = bits[0]
        task = bits[1].replace("task-", "")
        run = int(bits[2].replace("run-", ""))
        grp = parts.loc[parts.participant_id == sub, "group_short"]
        group = grp.iloc[0] if len(grp) else ("MDD" if "mdd" in sub else "Control")
        age = parts.loc[parts.participant_id == sub, "age"]
        sex = parts.loc[parts.participant_id == sub, "sex"]
        age_v = int(age.iloc[0]) if len(age) else np.nan
        sex_v = str(sex.iloc[0]) if len(sex) else ""
        print(f"  {bold.relative_to(DATA)}")

        ts = extract_mean_bold(bold)
        for t_i, val in enumerate(ts):
            ts_rows.append(
                {
                    "subject": sub,
                    "group": group,
                    "task": task,
                    "run": run,
                    "volume": t_i,
                    "time": t_i * TR,
                    "bold_z": float(val),
                }
            )

        sp = spectral_feats(ts)
        ev_path = bold.with_name(bold.name.replace("_bold.nii.gz", "_events.tsv"))
        # run-level peak on any music-like onset
        peak_lat, peak_amp = np.nan, np.nan
        if ev_path.exists():
            edf = pd.read_csv(ev_path, sep="\t")
            mask = edf["trial_type"].astype(str).str.contains(
                "positive_music|negative_music|positive_nonmusic|negative_nonmusic",
                case=False,
                na=False,
            )
            onsets = edf.loc[mask, "onset"].astype(float).values
            segs = []
            for o in onsets:
                i0 = int(round(o / TR))
                win = int(30 / TR)
                if 0 <= i0 and i0 + win <= len(ts):
                    segs.append(ts[i0 : i0 + win])
            if segs:
                mean_seg = np.mean(np.stack(segs), axis=0)
                peak_i = int(np.argmax(mean_seg))
                peak_lat, peak_amp = float(peak_i * TR), float(mean_seg[peak_i])

        feat_rows.append(
            {
                "subject": sub,
                "group": group,
                "age": age_v,
                "sex": sex_v,
                "task": task,
                "run": run,
                "n_volumes": len(ts),
                "power_low": sp["power_low"],
                "power_mid": sp["power_mid"],
                "power_high": sp["power_high"],
                "spectral_centroid": sp["spectral_centroid"],
                "total_power": sp["total_power"],
                "peak_latency_s": peak_lat,
                "peak_amp": peak_amp,
                "psd_f": json.dumps(sp["psd_f"]),
                "psd_pxx": json.dumps(sp["psd_pxx"]),
            }
        )

        for cm in condition_metrics(ts, ev_path):
            cond_rows.append(
                {
                    "subject": sub,
                    "group": group,
                    "age": age_v,
                    "sex": sex_v,
                    "task": task,
                    "run": run,
                    **{k: v for k, v in cm.items() if k != "peri_stim"},
                    "peri_stim": json.dumps(cm["peri_stim"]),
                }
            )

    ts_df = pd.DataFrame(ts_rows)
    feat_df = pd.DataFrame(feat_rows)
    cond_df = pd.DataFrame(cond_rows)
    ts_df.to_csv(OUT / "bold_timeseries.csv", index=False)
    feat_df.to_csv(OUT / "spectral_features.csv", index=False)
    cond_df.to_csv(OUT / "condition_features.csv", index=False)

    # Subject-level wide features for ML (music contrast effects)
    subj_rows = []
    for sub, g in cond_df.groupby("subject"):
        row = {
            "subject": sub,
            "group": g["group"].iloc[0],
            "age": g["age"].iloc[0],
            "sex": g["sex"].iloc[0],
        }
        for trial in STIM_MAP:
            subg = g[g.trial_type == trial]
            if len(subg):
                row[f"{trial}_mean_bold"] = float(subg["mean_bold"].mean())
                row[f"{trial}_peak_amp"] = float(subg["peak_amp"].mean())
                row[f"{trial}_power_high"] = float(subg["power_high"].mean())
                row[f"{trial}_centroid"] = float(subg["spectral_centroid"].mean())
            else:
                for sfx in ("mean_bold", "peak_amp", "power_high", "centroid"):
                    row[f"{trial}_{sfx}"] = np.nan
        # contrasts: what music does relative to tones / nonmusic
        def _c(a, b):
            if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b]):
                return float(row[a] - row[b])
            return np.nan

        row["pos_music_vs_tones_bold"] = _c("positive_music_mean_bold", "tones_mean_bold")
        row["neg_music_vs_tones_bold"] = _c("negative_music_mean_bold", "tones_mean_bold")
        row["pos_music_vs_neg_music_bold"] = _c(
            "positive_music_mean_bold", "negative_music_mean_bold"
        )
        row["pos_music_vs_pos_nonmusic_bold"] = _c(
            "positive_music_mean_bold", "positive_nonmusic_mean_bold"
        )
        row["pos_music_vs_tones_power_high"] = _c(
            "positive_music_power_high", "tones_power_high"
        )
        row["music_domain_mean_bold"] = np.nanmean(
            [
                row.get("positive_music_mean_bold", np.nan),
                row.get("negative_music_mean_bold", np.nan),
            ]
        )
        row["nonmusic_domain_mean_bold"] = np.nanmean(
            [
                row.get("positive_nonmusic_mean_bold", np.nan),
                row.get("negative_nonmusic_mean_bold", np.nan),
            ]
        )
        row["music_vs_nonmusic_bold"] = (
            row["music_domain_mean_bold"] - row["nonmusic_domain_mean_bold"]
            if pd.notna(row["music_domain_mean_bold"])
            and pd.notna(row["nonmusic_domain_mean_bold"])
            else np.nan
        )
        # run-level spectral means
        sf = feat_df[feat_df.subject == sub]
        if len(sf):
            row["run_power_high_mean"] = float(sf["power_high"].mean())
            row["run_centroid_mean"] = float(sf["spectral_centroid"].mean())
        subj_rows.append(row)

    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(OUT / "subject_features.csv", index=False)

    # Bundle for WASM
    bundle = {
        "source": "OpenNeuro ds000171 — real BOLD subset, trial-type aware features",
        "tr_sec": TR,
        "n_participants_full": int(len(parts)),
        "n_bold_runs": int(len(feat_df)),
        "n_subjects_with_bold": int(feat_df["subject"].nunique()) if len(feat_df) else 0,
        "stim_map": STIM_MAP,
        "participants": parts.to_dict(orient="records"),
        "events_summary": events.to_dict(orient="records"),
        "spectral_features": feat_df.drop(
            columns=["psd_f", "psd_pxx"], errors="ignore"
        ).to_dict(orient="records"),
        "condition_features": cond_df.drop(columns=["peri_stim"], errors="ignore").to_dict(
            orient="records"
        ),
        "subject_features": subj_df.to_dict(orient="records"),
        "psd_examples": {},
        "peri_examples": {},
        "timeseries_examples": {},
        "condition_peri_examples": {},
    }
    for _, row in feat_df.iterrows():
        key = f"{row['group']}_{row['task']}"
        if key not in bundle["psd_examples"] and "psd_f" in row and pd.notna(row.get("psd_f")):
            bundle["psd_examples"][key] = {
                "subject": row["subject"],
                "f": json.loads(row["psd_f"]),
                "pxx": json.loads(row["psd_pxx"]),
            }
    for _, row in cond_df.iterrows():
        key = f"{row['group']}_{row['trial_type']}"
        if key not in bundle["condition_peri_examples"] and row.get("peri_stim"):
            try:
                peri = json.loads(row["peri_stim"]) if isinstance(row["peri_stim"], str) else row["peri_stim"]
            except Exception:
                peri = []
            if peri:
                bundle["condition_peri_examples"][key] = {
                    "subject": row["subject"],
                    "trial_type": row["trial_type"],
                    "peri": peri,
                }
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
    print("\ncondition features sample:")
    print(cond_df.groupby(["group", "trial_type"])["mean_bold"].mean().unstack().round(3))
    print("\nsubject contrasts:")
    print(
        subj_df[
            [
                "subject",
                "group",
                "pos_music_vs_tones_bold",
                "pos_music_vs_neg_music_bold",
                "music_vs_nonmusic_bold",
            ]
        ].round(3)
    )

    # regenerate embedded book_data
    gen = ROOT / "scripts" / "gen_book_data.py"
    if gen.exists():
        import subprocess, sys

        subprocess.check_call([sys.executable, str(gen)], cwd=str(ROOT))


if __name__ == "__main__":
    main()
