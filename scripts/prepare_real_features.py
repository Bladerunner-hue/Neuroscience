#!/usr/bin/env python3
"""Extract analysis-ready features from real OpenNeuro ds000171 BOLD.

Produces:
  data/processed/participants_clean.csv
  data/processed/events_summary.csv
  data/processed/bold_timeseries.csv       # global + spatial slab means
  data/processed/spectral_features.csv     # run-level PSD + spatial
  data/processed/condition_features.csv    # trial-type / music-valence effects
  data/processed/subject_features.csv      # subject-level for ML
  data/processed/spatial_connectivity.csv  # A–P / L–R coherence proxies
  data/processed/cleaned_spectral_features.csv  # QC-gated export for bake-off/TF
  data/processed/book_bundle.json
  + regenerates marimo_notebooks/book_data.py if gen script available

PSD methods (``--psd``):
  welch     classic Hann Welch (legacy baseline)
  uniform   equal-weight DPSS multitaper
  adaptive  Thomson-style adaptive multitaper (default; pure scipy)
  mne       MNE ``psd_array_multitaper(adaptive=True)`` if mne installed

Spatial note: without an atlas in the public book pipeline we define *pseudo-ROIs*
from the native grid (anterior/posterior, left/right, superior/inferior brain mask
by signal variance). This restores coarse spatial specificity that whole-brain means lose.

Run:  python scripts/prepare_real_features.py
      python scripts/prepare_real_features.py --psd adaptive
      python scripts/prepare_real_features.py --from-timeseries  # skip NIfTI reload
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import coherence as msc_coherence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "marimo_notebooks"))
from spectral_methods import (  # noqa: E402
    compute_tsnr,
    spectral_feats as _spectral_feats_core,
)

DATA = ROOT / "data" / "raw" / "ds000171"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
TR = 3.0
# Mutated by main() from CLI
PSD_METHOD = "adaptive"
PSD_NW = 3.5

STIM_MAP = {
    "positive_music": {"valence": "positive", "domain": "music", "label": "Positive music"},
    "negative_music": {"valence": "negative", "domain": "music", "label": "Negative music"},
    "positive_nonmusic": {"valence": "positive", "domain": "nonmusic", "label": "Positive non-music"},
    "negative_nonmusic": {"valence": "negative", "domain": "nonmusic", "label": "Negative non-music"},
    "tones": {"valence": "neutral", "domain": "tones", "label": "Tones"},
}

SPATIAL_KEYS = ("global", "anterior", "posterior", "left", "right", "superior", "inferior")


def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def zscore(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    return ((ts - ts.mean()) / (ts.std() + 1e-8)).astype(np.float64)


def load_participants() -> pd.DataFrame:
    df = pd.read_csv(DATA / "participants.tsv", sep="\t")
    df["group_short"] = df["group"].map(
        {
            "Major Depressive Disorder": "MDD",
            "Never-Depressed Control": "Control",
        }
    )
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
        rows.append(
            {
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
                "n_response": int((types == "response").sum()),
                "has_bold": (ev.with_name(ev.name.replace("_events.tsv", "_bold.nii.gz"))).exists(),
            }
        )
    return pd.DataFrame(rows)


def extract_spatial_bold(path: Path) -> dict[str, np.ndarray]:
    """Global + 6 slab pseudo-ROIs from native grid (brain mask = high temporal variance)."""
    import nibabel as nib

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)  # X,Y,Z,T
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got shape {data.shape}")
    nx, ny, nz, nt = data.shape
    # brain-ish mask: voxels with temporal std above 20th pct of positive stds
    vstd = data.std(axis=3)
    pos = vstd[vstd > 0]
    thr = float(np.percentile(pos, 20)) if pos.size else 0.0
    mask = vstd > thr
    if not np.any(mask):
        mask = np.ones((nx, ny, nz), dtype=bool)

    def mean_ts(m: np.ndarray) -> tuple[np.ndarray, float]:
        raw = data.mean(axis=(0, 1, 2)) if not np.any(m) else data[m].mean(axis=0)
        tsnr = float(np.mean(raw) / (np.std(raw) + 1e-8))
        return zscore(raw), tsnr

    mx, my, mz = nx // 2, ny // 2, nz // 2
    g_ts, g_tsnr = mean_ts(mask)
    out = {
        "global": g_ts,
        "tsnr_global": g_tsnr,
        "left": mean_ts(mask & (np.arange(nx)[:, None, None] < mx))[0],
        "right": mean_ts(mask & (np.arange(nx)[:, None, None] >= mx))[0],
        "posterior": mean_ts(mask & (np.arange(ny)[None, :, None] < my))[0],
        "anterior": mean_ts(mask & (np.arange(ny)[None, :, None] >= my))[0],
        "inferior": mean_ts(mask & (np.arange(nz)[None, None, :] < mz))[0],
        "superior": mean_ts(mask & (np.arange(nz)[None, None, :] >= mz))[0],
    }
    return out


def _mne_multitaper_psd(ts: np.ndarray, tr: float = TR, nw: float = 3.5):
    """MNE adaptive multitaper (production-grade). Requires ``pip install mne``."""
    import mne

    fs = 1.0 / tr
    # bandwidth ≈ 2W; MNE uses Hz full-bandwidth ≈ 2 * (nw / T) roughly
    # Prefer explicit NW via bandwidth = 2 * nw * (fs / n) * n/2 wait:
    # MNE docs: bandwidth is the half-bandwidth in Hz of the multi-taper window.
    # For DPSS, half-bandwidth W = nw / (n / fs) = nw * fs / n
    n = len(ts)
    bandwidth = float(nw) * fs / max(n, 1)
    bandwidth = max(bandwidth, 2.0 * fs / n)  # at least one Rayleigh bin
    pxx, f = mne.time_frequency.psd_array_multitaper(
        np.asarray(ts, dtype=np.float64)[None, :],
        sfreq=fs,
        bandwidth=bandwidth,
        adaptive=True,
        low_bias=True,
        normalization="length",
        verbose=False,
    )
    return f, np.maximum(pxx[0], 1e-20)


def spectral_feats(ts: np.ndarray, tr: float = TR, method: str | None = None) -> dict:
    """Run-level PSD + quality metrics.

    Methods: welch | uniform | adaptive (scipy DPSS) | mne (optional adaptive).
    Quality metrics: spectral_flatness, spectral_entropy, band_snr_high, tsnr.
    """
    method = (method or PSD_METHOD or "adaptive").lower()
    if method == "mne":
        try:
            f, pxx = _mne_multitaper_psd(ts, tr=tr, nw=PSD_NW)
            total = trapz(pxx, f) + 1e-12
            geom = float(np.exp(np.mean(np.log(np.maximum(pxx, 1e-20)))))
            arith = float(np.mean(pxx))
            p_norm = pxx / (pxx.sum() + 1e-20)
            entropy = float(
                -np.sum(p_norm * np.log(p_norm + 1e-20)) / np.log(len(p_norm))
            )
            out = {
                "total_power": float(total),
                "spectral_centroid": float(trapz(f * pxx, f) / total),
                "spectral_flatness": geom / (arith + 1e-20),
                "spectral_entropy": entropy,
                "psd_f": f.tolist(),
                "psd_pxx": pxx.tolist(),
                "psd_method": "mne",
                "tsnr": compute_tsnr(ts),
            }
            for name, lo, hi in [
                ("power_low", 0.01, 0.04),
                ("power_mid", 0.04, 0.08),
                ("power_high", 0.08, 0.15),
            ]:
                m = (f >= lo) & (f <= hi)
                out[name] = trapz(pxx[m], f[m]) / total if np.any(m) else 0.0
            m_hi = (f >= 0.08) & (f <= 0.15)
            m_out = ~m_hi
            p_hi = float(np.mean(pxx[m_hi])) if np.any(m_hi) else 0.0
            p_out = float(np.median(pxx[m_out])) if np.any(m_out) else 1e-12
            out["band_snr_high"] = p_hi / (p_out + 1e-12)
            return out
        except ImportError:
            print("  [warn] mne not installed — falling back to adaptive multitaper")
            method = "adaptive"
    # welch | uniform | adaptive via shared spectral_methods
    sm_method = "welch" if method == "welch" else method
    if sm_method not in ("welch", "uniform", "adaptive"):
        sm_method = "adaptive"
    out = _spectral_feats_core(ts, tr=tr, method=sm_method, nw=PSD_NW)
    out["psd_method"] = sm_method
    return out


def temporal_qc(ts: np.ndarray) -> dict:
    """Simple time-domain QC on z-scored (or raw) series."""
    ts = np.asarray(ts, dtype=np.float64)
    d = np.diff(ts)
    return {
        "ts_std": float(ts.std()),
        "ts_abs_mean": float(np.abs(ts).mean()),
        "ts_spike_frac": float(np.mean(np.abs(d) > 3.0 * (d.std() + 1e-8))),
        "n_volumes": int(len(ts)),
    }


def band_coherence(x: np.ndarray, y: np.ndarray, tr: float = TR, lo=0.03, hi=0.10) -> float:
    """Integrated magnitude-squared coherence in [lo, hi] Hz."""
    n = min(len(x), len(y))
    if n < 16:
        return float("nan")
    x, y = x[:n], y[:n]
    nper = min(28, max(8, n // 2))
    f, cxy = msc_coherence(x, y, fs=1.0 / tr, nperseg=nper)
    m = (f > lo) & (f < hi)
    if not np.any(m):
        return float("nan")
    return trapz(cxy[m], f[m])


def segment_mean(ts: np.ndarray, onset: float, duration: float, tr: float = TR) -> np.ndarray:
    i0 = max(0, int(round(onset / tr)))
    i1 = min(len(ts), int(round((onset + duration) / tr)))
    if i1 <= i0:
        return np.array([])
    return ts[i0:i1]


def condition_metrics(ts: np.ndarray, events_path: Path) -> list[dict]:
    """Per trial-type summary within one run (response events excluded)."""
    if not events_path.exists():
        return []
    edf = pd.read_csv(events_path, sep="\t")
    rows = []
    for trial, g in edf.groupby("trial_type"):
        trial = str(trial)
        if trial == "response":
            continue  # task button-press windows — excluded by design from spectral epochs
        segs = []
        for _, r in g.iterrows():
            seg = segment_mean(ts, float(r["onset"]), float(r.get("duration", 31.5)))
            if len(seg) >= 4:
                segs.append(seg)
        if not segs:
            continue
        cat = np.concatenate(segs)
        meta = STIM_MAP.get(trial, {"valence": "other", "domain": "other", "label": trial})
        L = int(np.median([len(s) for s in segs]))
        stacked = np.stack(
            [np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s) for s in segs]
        )
        mean_seg = stacked.mean(axis=0)
        peak_i = int(np.argmax(mean_seg))
        sp = (
            spectral_feats(cat)
            if len(cat) >= 16
            else {
                "power_low": np.nan,
                "power_mid": np.nan,
                "power_high": np.nan,
                "spectral_centroid": np.nan,
                "total_power": np.nan,
                "psd_f": [],
                "psd_pxx": [],
            }
        )
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ds000171 spectral + QC features")
    p.add_argument(
        "--psd",
        choices=("welch", "uniform", "adaptive", "mne"),
        default="adaptive",
        help="PSD estimator (default: adaptive multitaper)",
    )
    p.add_argument(
        "--nw",
        type=float,
        default=3.5,
        help="DPSS time-bandwidth product for multitaper methods",
    )
    p.add_argument(
        "--from-timeseries",
        action="store_true",
        help="Recompute spectra from bold_timeseries.csv (skip NIfTI I/O)",
    )
    p.add_argument(
        "--no-clean-export",
        action="store_true",
        help="Skip writing cleaned_spectral_features.csv",
    )
    return p.parse_args(argv)


def _recompute_from_timeseries(parts: pd.DataFrame) -> tuple[pd.DataFrame, list, list, list]:
    """Refresh spectral/condition/connectivity from stored global + slab means."""
    ts_path = OUT / "bold_timeseries.csv"
    if not ts_path.exists():
        raise SystemExit("Missing bold_timeseries.csv — run full prepare first.")
    ts_df = pd.read_csv(ts_path)
    feat_rows: list[dict] = []
    cond_rows: list[dict] = []
    conn_rows: list[dict] = []
    keys = [c for c in ts_df.columns if c.startswith("bold_")]
    # group by run
    for (sub, task, run), g in ts_df.groupby(["subject", "task", "run"], sort=False):
        g = g.sort_values("volume")
        group = g["group"].iloc[0]
        age = parts.loc[parts.participant_id == sub, "age"]
        sex = parts.loc[parts.participant_id == sub, "sex"]
        age_v = int(age.iloc[0]) if len(age) else np.nan
        sex_v = str(sex.iloc[0]) if len(sex) else ""
        ts = g["bold_z"].to_numpy(dtype=np.float64)
        spatial = {"global": ts}
        for col in keys:
            k = col.replace("bold_", "")
            spatial[k] = g[col].to_numpy(dtype=np.float64)
        print(f"  recompute {sub} {task} run-{run} (psd={PSD_METHOD})")
        sp = spectral_feats(ts)
        tqc = temporal_qc(ts)
        spat_pow = {}
        for k in SPATIAL_KEYS:
            if k == "global" or k not in spatial:
                continue
            sk = spectral_feats(spatial[k])
            spat_pow[f"power_high_{k}"] = sk["power_high"]
            spat_pow[f"centroid_{k}"] = sk["spectral_centroid"]
        coh_ap = (
            band_coherence(spatial["anterior"], spatial["posterior"])
            if "anterior" in spatial and "posterior" in spatial
            else np.nan
        )
        coh_lr = (
            band_coherence(spatial["left"], spatial["right"])
            if "left" in spatial and "right" in spatial
            else np.nan
        )
        coh_si = (
            band_coherence(spatial["superior"], spatial["inferior"])
            if "superior" in spatial and "inferior" in spatial
            else np.nan
        )
        conn_rows.append(
            {
                "subject": sub,
                "group": group,
                "task": task,
                "run": int(run),
                "coh_ant_post": coh_ap,
                "coh_left_right": coh_lr,
                "coh_sup_inf": coh_si,
            }
        )
        # events for condition metrics
        ev_path = (
            DATA
            / sub
            / "func"
            / f"{sub}_task-{task}_run-{int(run)}_events.tsv"
        )
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
        ant_mean = float(spatial["anterior"].mean()) if "anterior" in spatial else 0.0
        post_mean = float(spatial["posterior"].mean()) if "posterior" in spatial else 0.0
        # tSNR: prefer raw-style if available; use stored ts (z-scored → tSNR~0 useless)
        # Keep previous tsnr from spectral_features if present
        prev = OUT / "spectral_features.csv"
        tsnr_v = np.nan
        if prev.exists():
            old = pd.read_csv(prev)
            m = (
                (old.subject == sub)
                & (old.task == task)
                & (old.run == int(run))
            )
            if m.any() and "tsnr" in old.columns:
                tsnr_v = float(old.loc[m, "tsnr"].iloc[0])
        if not np.isfinite(tsnr_v):
            tsnr_v = compute_tsnr(ts)  # on z-scored series ≈ 0; still a QC column
        feat_rows.append(
            {
                "subject": sub,
                "group": group,
                "age": age_v,
                "sex": sex_v,
                "task": task,
                "run": int(run),
                "n_volumes": len(ts),
                "power_low": sp["power_low"],
                "power_mid": sp["power_mid"],
                "power_high": sp["power_high"],
                "spectral_centroid": sp["spectral_centroid"],
                "spectral_flatness": sp["spectral_flatness"],
                "spectral_entropy": sp["spectral_entropy"],
                "band_snr_high": sp["band_snr_high"],
                "total_power": sp["total_power"],
                "tsnr": tsnr_v,
                "ts_spike_frac": tqc["ts_spike_frac"],
                "peak_latency_s": peak_lat,
                "peak_amp": peak_amp,
                "coh_ant_post": coh_ap,
                "coh_left_right": coh_lr,
                "coh_sup_inf": coh_si,
                "ant_minus_post_mean": ant_mean - post_mean,
                "left_minus_right_mean": (
                    float(spatial["left"].mean() - spatial["right"].mean())
                    if "left" in spatial and "right" in spatial
                    else np.nan
                ),
                **spat_pow,
                "psd_method": sp.get("psd_method", PSD_METHOD),
                "psd_f": json.dumps(sp["psd_f"]),
                "psd_pxx": json.dumps(sp["psd_pxx"]),
            }
        )
        for cm in condition_metrics(ts, ev_path):
            ant_segs = []
            if ev_path.exists() and "anterior" in spatial:
                edf2 = pd.read_csv(ev_path, sep="\t")
                for _, r in edf2[
                    edf2["trial_type"].astype(str) == cm["trial_type"]
                ].iterrows():
                    seg = segment_mean(
                        spatial["anterior"],
                        float(r["onset"]),
                        float(r.get("duration", 31.5)),
                    )
                    if len(seg) >= 4:
                        ant_segs.append(seg)
            ant_mean_c = float(np.concatenate(ant_segs).mean()) if ant_segs else np.nan
            cond_rows.append(
                {
                    "subject": sub,
                    "group": group,
                    "age": age_v,
                    "sex": sex_v,
                    "task": task,
                    "run": int(run),
                    **{k: v for k, v in cm.items() if k != "peri_stim"},
                    "anterior_mean_bold": ant_mean_c,
                    "peri_stim": json.dumps(cm["peri_stim"]),
                }
            )
    return ts_df, feat_rows, cond_rows, conn_rows


def main(argv: list[str] | None = None) -> None:
    global PSD_METHOD, PSD_NW
    args = _parse_args(argv)
    PSD_METHOD = args.psd
    PSD_NW = float(args.nw)
    print(f"PSD method: {PSD_METHOD}  NW={PSD_NW}")

    parts = load_participants()
    parts.to_csv(OUT / "participants_clean.csv", index=False)
    print("participants:", len(parts))

    events = inventory_events()
    events.to_csv(OUT / "events_summary.csv", index=False)
    print("event files:", len(events), "with BOLD:", int(events["has_bold"].sum()) if "has_bold" in events else "?")

    ts_rows: list[dict] = []
    feat_rows: list[dict] = []
    cond_rows: list[dict] = []
    conn_rows: list[dict] = []

    if args.from_timeseries:
        ts_df, feat_rows, cond_rows, conn_rows = _recompute_from_timeseries(parts)
    else:
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

            spatial = extract_spatial_bold(bold)
            ts = spatial["global"]

            for t_i, val in enumerate(ts):
                row_ts = {
                    "subject": sub,
                    "group": group,
                    "task": task,
                    "run": run,
                    "volume": t_i,
                    "time": t_i * TR,
                    "bold_z": float(val),
                }
                for k in SPATIAL_KEYS:
                    if k == "global":
                        continue
                    if t_i < len(spatial[k]):
                        row_ts[f"bold_{k}"] = float(spatial[k][t_i])
                ts_rows.append(row_ts)

            sp = spectral_feats(ts)
            tqc = temporal_qc(ts)
            spat_pow = {}
            for k in SPATIAL_KEYS:
                if k == "global":
                    continue
                sk = spectral_feats(spatial[k])
                spat_pow[f"power_high_{k}"] = sk["power_high"]
                spat_pow[f"centroid_{k}"] = sk["spectral_centroid"]

            coh_ap = band_coherence(spatial["anterior"], spatial["posterior"])
            coh_lr = band_coherence(spatial["left"], spatial["right"])
            coh_si = band_coherence(spatial["superior"], spatial["inferior"])
            conn_rows.append(
                {
                    "subject": sub,
                    "group": group,
                    "task": task,
                    "run": run,
                    "coh_ant_post": coh_ap,
                    "coh_left_right": coh_lr,
                    "coh_sup_inf": coh_si,
                }
            )

            ev_path = bold.with_name(bold.name.replace("_bold.nii.gz", "_events.tsv"))
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

            ant_mean = float(spatial["anterior"].mean())
            post_mean = float(spatial["posterior"].mean())

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
                    "spectral_flatness": sp["spectral_flatness"],
                    "spectral_entropy": sp["spectral_entropy"],
                    "band_snr_high": sp["band_snr_high"],
                    "total_power": sp["total_power"],
                    "tsnr": float(spatial.get("tsnr_global", np.nan)),
                    "ts_spike_frac": tqc["ts_spike_frac"],
                    "peak_latency_s": peak_lat,
                    "peak_amp": peak_amp,
                    "coh_ant_post": coh_ap,
                    "coh_left_right": coh_lr,
                    "coh_sup_inf": coh_si,
                    "ant_minus_post_mean": ant_mean - post_mean,
                    "left_minus_right_mean": float(
                        spatial["left"].mean() - spatial["right"].mean()
                    ),
                    **spat_pow,
                    "psd_method": sp.get("psd_method", PSD_METHOD),
                    "psd_f": json.dumps(sp["psd_f"]),
                    "psd_pxx": json.dumps(sp["psd_pxx"]),
                }
            )

            for cm in condition_metrics(ts, ev_path):
                ant_segs = []
                if ev_path.exists():
                    edf2 = pd.read_csv(ev_path, sep="\t")
                    for _, r in edf2[
                        edf2["trial_type"].astype(str) == cm["trial_type"]
                    ].iterrows():
                        seg = segment_mean(
                            spatial["anterior"],
                            float(r["onset"]),
                            float(r.get("duration", 31.5)),
                        )
                        if len(seg) >= 4:
                            ant_segs.append(seg)
                ant_mean_c = (
                    float(np.concatenate(ant_segs).mean()) if ant_segs else np.nan
                )
                cond_rows.append(
                    {
                        "subject": sub,
                        "group": group,
                        "age": age_v,
                        "sex": sex_v,
                        "task": task,
                        "run": run,
                        **{k: v for k, v in cm.items() if k != "peri_stim"},
                        "anterior_mean_bold": ant_mean_c,
                        "peri_stim": json.dumps(cm["peri_stim"]),
                    }
                )

        ts_df = pd.DataFrame(ts_rows)

    feat_df = pd.DataFrame(feat_rows)
    cond_df = pd.DataFrame(cond_rows)
    conn_df = pd.DataFrame(conn_rows)
    if not isinstance(ts_df, pd.DataFrame):
        ts_df = pd.DataFrame(ts_rows)

    # --- Run-level QC: IsolationForest + rule flags ---
    qc_cols = [
        c
        for c in [
            "tsnr",
            "spectral_flatness",
            "spectral_entropy",
            "band_snr_high",
            "ts_spike_frac",
            "power_high",
            "spectral_centroid",
            "total_power",
        ]
        if c in feat_df.columns
    ]
    if len(feat_df) and qc_cols:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        Xq = feat_df[qc_cols].apply(pd.to_numeric, errors="coerce")
        Xq = Xq.fillna(Xq.median(numeric_only=True))
        Xs = StandardScaler().fit_transform(Xq.values)
        cont = min(0.2, max(0.05, 2.0 / max(len(feat_df), 1)))
        iso = IsolationForest(
            n_estimators=200, contamination=cont, random_state=42
        )
        pred = iso.fit_predict(Xs)  # -1 outlier
        feat_df["qc_iforest"] = pred
        feat_df["qc_outlier"] = (pred == -1).astype(int)
        # Rule-based flags (conservative thresholds for z-scored mean BOLD pipelines)
        feat_df["qc_low_tsnr"] = (
            feat_df["tsnr"] < feat_df["tsnr"].median() - feat_df["tsnr"].std()
        ).astype(int) if "tsnr" in feat_df else 0
        feat_df["qc_high_flatness"] = (
            feat_df["spectral_flatness"]
            > feat_df["spectral_flatness"].quantile(0.9)
        ).astype(int) if "spectral_flatness" in feat_df else 0
        feat_df["qc_high_spike"] = (
            feat_df["ts_spike_frac"] > 0.15
        ).astype(int) if "ts_spike_frac" in feat_df else 0
        feat_df["qc_flag_any"] = (
            (feat_df["qc_outlier"] == 1)
            | (feat_df.get("qc_low_tsnr", 0) == 1)
            | (feat_df.get("qc_high_flatness", 0) == 1)
            | (feat_df.get("qc_high_spike", 0) == 1)
        ).astype(int)
        qc_df = feat_df[
            [
                c
                for c in [
                    "subject",
                    "group",
                    "task",
                    "run",
                    "tsnr",
                    "spectral_flatness",
                    "spectral_entropy",
                    "band_snr_high",
                    "ts_spike_frac",
                    "qc_outlier",
                    "qc_low_tsnr",
                    "qc_high_flatness",
                    "qc_high_spike",
                    "qc_flag_any",
                ]
                if c in feat_df.columns
            ]
        ].copy()
        qc_df.to_csv(OUT / "run_qc.csv", index=False)
        print(
            "QC outliers (IsolationForest):",
            int(feat_df["qc_outlier"].sum()),
            "/",
            len(feat_df),
        )
    else:
        qc_df = pd.DataFrame()

    if not args.from_timeseries:
        ts_df.to_csv(OUT / "bold_timeseries.csv", index=False)
    feat_df.to_csv(OUT / "spectral_features.csv", index=False)
    cond_df.to_csv(OUT / "condition_features.csv", index=False)
    conn_df.to_csv(OUT / "spatial_connectivity.csv", index=False)

    # Cleaned export for bake-off / TF (drop IsolationForest outliers by default)
    if not args.no_clean_export and len(feat_df) and "qc_outlier" in feat_df.columns:
        clean = feat_df[feat_df["qc_outlier"] == 0].drop(
            columns=["psd_f", "psd_pxx"], errors="ignore"
        )
        clean.to_csv(OUT / "cleaned_spectral_features.csv", index=False)
        print(
            "Cleaned features:",
            len(clean),
            "/",
            len(feat_df),
            "→",
            OUT / "cleaned_spectral_features.csv",
        )

    # Subject-level wide features for ML
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
                if "anterior_mean_bold" in subg.columns:
                    row[f"{trial}_anterior"] = float(subg["anterior_mean_bold"].mean())
            else:
                for sfx in ("mean_bold", "peak_amp", "power_high", "centroid", "anterior"):
                    row[f"{trial}_{sfx}"] = np.nan

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
        row["pos_music_vs_tones_anterior"] = _c(
            "positive_music_anterior", "tones_anterior"
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
        # RecSys-style responder score (primary clinical contrast)
        row["responder_score"] = np.nanmean(
            [
                row.get("pos_music_vs_tones_bold", np.nan),
                row.get("music_vs_nonmusic_bold", np.nan),
                row.get("pos_music_vs_tones_anterior", np.nan),
            ]
        )
        sf = feat_df[feat_df.subject == sub]
        if len(sf):
            row["run_power_high_mean"] = float(sf["power_high"].mean())
            row["run_centroid_mean"] = float(sf["spectral_centroid"].mean())
            row["coh_ant_post_mean"] = float(sf["coh_ant_post"].mean())
            row["coh_left_right_mean"] = float(sf["coh_left_right"].mean())
            row["ant_minus_post_mean"] = float(sf["ant_minus_post_mean"].mean())
            # music-task minus nonmusic-task spectral
            sm = sf[sf.task == "music"]
            sn = sf[sf.task == "nonmusic"]
            if len(sm) and len(sn):
                row["music_task_vs_nonmusic_power_high"] = float(
                    sm["power_high"].mean() - sn["power_high"].mean()
                )
                row["music_task_vs_nonmusic_coh_ap"] = float(
                    sm["coh_ant_post"].mean() - sn["coh_ant_post"].mean()
                )
            else:
                row["music_task_vs_nonmusic_power_high"] = np.nan
                row["music_task_vs_nonmusic_coh_ap"] = np.nan
        subj_rows.append(row)

    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(OUT / "subject_features.csv", index=False)

    # Keep per-run PSD arrays in the bundle so WASM chapters (esp. Ch II) can
    # plot multitaper spectra without disk CSVs. Size is modest (~100KB).
    feat_for_bundle = feat_df.copy()
    for col in ("psd_f", "psd_pxx"):
        if col in feat_for_bundle.columns:
            feat_for_bundle[col] = feat_for_bundle[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )
    clean_for_bundle = (
        feat_for_bundle[feat_for_bundle["qc_outlier"] == 0]
        if "qc_outlier" in feat_for_bundle.columns and len(feat_for_bundle)
        else feat_for_bundle
    )
    # drop bulky PSD from cleaned table (tabular only); full PSDs stay on spectral_features
    clean_records = clean_for_bundle.drop(
        columns=["psd_f", "psd_pxx"], errors="ignore"
    ).to_dict(orient="records")

    ml_bakeoff = {}
    if (OUT / "ml_bakeoff.json").exists():
        try:
            ml_bakeoff = json.loads((OUT / "ml_bakeoff.json").read_text())
        except Exception:
            ml_bakeoff = {}
    tf_results = {}
    if (OUT / "tf_results.json").exists():
        try:
            tf_results = json.loads((OUT / "tf_results.json").read_text())
        except Exception:
            tf_results = {}

    bundle = {
        "source": "OpenNeuro ds000171 — expanded BOLD + spatial pseudo-ROIs + trial-type features",
        "tr_sec": TR,
        "psd_method": PSD_METHOD,
        "psd_nw": PSD_NW,
        "n_participants_full": int(len(parts)),
        "n_bold_runs": int(len(feat_df)),
        "n_subjects_with_bold": int(feat_df["subject"].nunique()) if len(feat_df) else 0,
        "n_cleaned_runs": int(len(clean_records)),
        "stim_map": STIM_MAP,
        "spatial_keys": list(SPATIAL_KEYS),
        "participants": parts.to_dict(orient="records"),
        "events_summary": events.to_dict(orient="records"),
        # Full run table WITH psd_f / psd_pxx lists (WASM Ch 0/II)
        "spectral_features": feat_for_bundle.to_dict(orient="records"),
        # QC-gated tabular export (WASM Ch II–IV, bake-off consumers)
        "cleaned_spectral_features": clean_records,
        "condition_features": cond_df.drop(columns=["peri_stim"], errors="ignore").to_dict(
            orient="records"
        ),
        "subject_features": subj_df.to_dict(orient="records"),
        "spatial_connectivity": conn_df.to_dict(orient="records"),
        "run_qc": qc_df.to_dict(orient="records") if len(qc_df) else [],
        "ml_bakeoff": ml_bakeoff,
        "tf_results": tf_results,
        "psd_examples": {},
        "peri_examples": {},
        "timeseries_examples": {},
        "condition_peri_examples": {},
    }
    for _, row in feat_df.iterrows():
        key = f"{row['group']}_{row['task']}"
        if key not in bundle["psd_examples"] and "psd_f" in row and pd.notna(row.get("psd_f")):
            pf = row["psd_f"]
            pp = row["psd_pxx"]
            if isinstance(pf, str):
                pf = json.loads(pf)
            if isinstance(pp, str):
                pp = json.loads(pp)
            bundle["psd_examples"][key] = {
                "subject": row["subject"],
                "f": pf,
                "pxx": pp,
                "psd_method": row.get("psd_method", PSD_METHOD),
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
    print("subjects with BOLD:", feat_df["subject"].nunique())
    print("\ncondition mean_bold:")
    print(cond_df.groupby(["group", "trial_type"])["mean_bold"].mean().unstack().round(3))
    print("\nresponder scores:")
    print(subj_df[["subject", "group", "responder_score", "pos_music_vs_tones_bold"]].round(3))

    gen = ROOT / "scripts" / "gen_book_data.py"
    if gen.exists():
        import subprocess
        import sys

        subprocess.check_call([sys.executable, str(gen)], cwd=str(ROOT))


if __name__ == "__main__":
    main()
