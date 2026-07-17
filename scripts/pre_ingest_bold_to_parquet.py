#!/usr/bin/env python3
"""CUDA-or-CPU pre-ingest: NIfTI BOLD → tabular Parquet for pure Spark.

Honour mode: heavy spectral work happens **outside** the JVM.
- Prefer CuPy FFT if available (optional).
- Default: numpy + scipy multitaper (same as spectral_methods).

Writes:
  data/processed/god_parquet_bold/epoch_ts/     # time series arrays + labels
  data/processed/god_parquet_bold/run_spectral/  # precomputed band powers

Spark (god_mode_bold_to_tfdata.py) only aggregates these tables with Catalyst.

Usage::

    python scripts/pre_ingest_bold_to_parquet.py
    python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725
    python scripts/pre_ingest_bold_to_parquet.py --backend cupy   # if installed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "marimo_notebooks"))
from spectral_methods import spectral_feats  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "god_parquet_bold"
REGISTRY = ROOT / "data" / "processed" / "dataset_registry.json"

# Default TR map (seconds); override via --tr or dataset_description if known
DEFAULT_TR = {
    "ds000171": 3.0,
    "ds002725": 2.0,
    "ds003085": 2.0,
    "ds003720": 2.0,
    "ds004142": 2.0,
    "ds005700": 2.0,
    "ds006564": 2.0,
}


def _backend_fft(ts: np.ndarray, fs: float, backend: str):
    """Optional CuPy path; falls back to spectral_methods (scipy multitaper)."""
    if backend == "cupy":
        try:
            import cupy as cp

            x = cp.asarray(ts, dtype=cp.float64)
            x = x - cp.mean(x)
            # Simple periodogram on GPU (not full multitaper — prototype)
            n = int(x.size)
            freqs = cp.fft.rfftfreq(n, d=1.0 / fs)
            pxx = cp.abs(cp.fft.rfft(x)) ** 2 / (fs * n)
            f = cp.asnumpy(freqs)
            p = cp.asnumpy(pxx)
            # Band powers via trapz
            out = {"psd_f": f.tolist(), "psd_pxx": p.tolist()}
            total = float(np.trapezoid(p, f) + 1e-12) if hasattr(np, "trapezoid") else float(np.trapz(p, f) + 1e-12)
            out["total_power"] = total
            for name, lo, hi in [
                ("power_low", 0.01, 0.04),
                ("power_mid", 0.04, 0.08),
                ("power_high", 0.08, 0.15),
            ]:
                m = (f >= lo) & (f <= hi)
                if np.any(m):
                    band = float(np.trapezoid(p[m], f[m])) if hasattr(np, "trapezoid") else float(np.trapz(p[m], f[m]))
                    out[name] = band / total
                else:
                    out[name] = 0.0
            out["spectral_centroid"] = float(np.trapezoid(f * p, f) / total) if hasattr(np, "trapezoid") else float(np.trapz(f * p, f) / total)
            out["psd_method"] = "cupy_rfft"
            return out
        except Exception as e:
            print(f"  cupy failed ({e}); falling back to scipy multitaper")
    return spectral_feats(ts, tr=1.0 / fs, method="adaptive")


def _zscore(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    return (ts - ts.mean()) / (ts.std() + 1e-8)


def _global_mean_ts(nii_path: Path) -> tuple[np.ndarray, float]:
    import nibabel as nib

    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got {data.shape}")
    # High-variance brainish mask
    vstd = data.std(axis=3)
    pos = vstd[vstd > 0]
    thr = float(np.percentile(pos, 20)) if pos.size else 0.0
    mask = vstd > thr
    raw = data[mask].mean(axis=0) if np.any(mask) else data.mean(axis=(0, 1, 2))
    tsnr = float(np.mean(raw) / (np.std(raw) + 1e-8))
    return _zscore(raw), tsnr


def _discover_bold(ds_root: Path) -> list[Path]:
    # Handle nested ds000171/ds000171/
    roots = [ds_root]
    nested = ds_root / ds_root.name
    if nested.is_dir():
        roots.append(nested)
    files: list[Path] = []
    for r in roots:
        files.extend(sorted(r.glob("sub-*/func/*_bold.nii.gz")))
        files.extend(sorted(r.glob("sub-*/func/*bold.nii.gz")))
    # unique
    seen = set()
    out = []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def _parse_bids_name(path: Path) -> dict:
    name = path.name.replace("_bold.nii.gz", "").replace(".nii.gz", "")
    parts = name.split("_")
    sub = parts[0]
    task = "unknown"
    run = 1
    for p in parts[1:]:
        if p.startswith("task-"):
            task = p.replace("task-", "")
        if p.startswith("run-"):
            try:
                run = int(p.replace("run-", ""))
            except ValueError:
                run = 1
    return {"subject": sub, "task": task, "run": run}


def _group_from_participants(ds_root: Path, subject: str) -> str:
    for base in (ds_root, ds_root / ds_root.name):
        part = base / "participants.tsv"
        if not part.exists():
            continue
        import pandas as pd

        df = pd.read_csv(part, sep="\t")
        col = "participant_id" if "participant_id" in df.columns else df.columns[0]
        row = df[df[col].astype(str) == subject]
        if row.empty:
            continue
        # Prefer short labels for Catalyst joins / TF labels
        # Note: "Never-Depressed" contains "depress" — check never/control first.
        for gcol in ("group_short", "group", "diagnosis", "Group"):
            if gcol not in row.columns:
                continue
            g = str(row.iloc[0][gcol])
            gl = g.lower()
            if "never" in gl or gl in ("control", "hc", "healthy"):
                return "Control"
            if "mdd" in gl or "major depressive" in gl:
                return "MDD"
            if "depress" in gl:
                return "MDD"
            return g
        return "unknown"
    # BIDS subject id fallback (ds000171 naming)
    sl = subject.lower()
    if "mdd" in sl or "patient" in sl:
        return "MDD"
    if "control" in sl:
        return "Control"
    return "unknown"


def ingest_dataset(
    ds_id: str,
    *,
    backend: str = "scipy",
    max_runs: int | None = None,
    tr_override: float | None = None,
) -> tuple[list[dict], list[dict]]:
    ds_root = RAW / ds_id
    if not ds_root.exists():
        print(f"  skip {ds_id}: missing {ds_root}")
        return [], []
    bold_files = _discover_bold(ds_root)
    if max_runs is not None:
        bold_files = bold_files[:max_runs]
    tr = float(tr_override or DEFAULT_TR.get(ds_id, 2.0))
    fs = 1.0 / tr
    print(f"  {ds_id}: {len(bold_files)} BOLD files · TR={tr}s · backend={backend}")

    epoch_rows: list[dict] = []
    spectral_rows: list[dict] = []
    for bold in bold_files:
        meta = _parse_bids_name(bold)
        group = _group_from_participants(ds_root, meta["subject"])
        try:
            ts, tsnr = _global_mean_ts(bold)
        except Exception as e:
            print(f"    fail {bold.name}: {e}")
            continue
        sp = _backend_fft(ts, fs, backend=backend)
        # drop huge psd lists from spectral parquet (keep bands); keep ts in epoch table
        spectral_rows.append(
            {
                "dataset": ds_id,
                "subject": meta["subject"],
                "group": group,
                "task": meta["task"],
                "run": int(meta["run"]),
                "n_volumes": int(len(ts)),
                "tr_sec": tr,
                "tsnr": tsnr,
                "power_low": float(sp.get("power_low", 0.0)),
                "power_mid": float(sp.get("power_mid", 0.0)),
                "power_high": float(sp.get("power_high", 0.0)),
                "spectral_centroid": float(sp.get("spectral_centroid", 0.0)),
                "total_power": float(sp.get("total_power", 0.0)),
                "psd_method": str(sp.get("psd_method", backend)),
            }
        )
        epoch_rows.append(
            {
                "dataset": ds_id,
                "subject": meta["subject"],
                "group": group,
                "task": meta["task"],
                "run": int(meta["run"]),
                "epoch": 0,
                "tr_sec": tr,
                "n_volumes": int(len(ts)),
                "tsnr": tsnr,
                "time_series_array": ts.astype(np.float64).tolist(),
                "power_low": float(sp.get("power_low", 0.0)),
                "power_mid": float(sp.get("power_mid", 0.0)),
                "power_high": float(sp.get("power_high", 0.0)),
                "spectral_centroid": float(sp.get("spectral_centroid", 0.0)),
                "label_group": group,
                "label_task": meta["task"],
            }
        )
        print(f"    ok {meta['subject']} {meta['task']} run-{meta['run']} T={len(ts)}")
    return epoch_rows, spectral_rows


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # empty schema placeholder
        table = pa.table({"dataset": pa.array([], type=pa.string())})
        pq.write_table(table, path / "part-0000.parquet")
        return
    table = pa.Table.from_pylist(rows)
    # single file for simplicity; Spark reads the directory
    out = path / "part-0000.parquet"
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    print(f"  wrote {out} ({out.stat().st_size} bytes, {len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        default="ds000171",
        help="Comma-separated dataset ids under data/raw/",
    )
    p.add_argument(
        "--backend",
        choices=("scipy", "cupy"),
        default="scipy",
        help="FFT backend (cupy optional; scipy multitaper is default honour path)",
    )
    p.add_argument("--max-runs", type=int, default=None, help="Cap BOLD files per dataset")
    p.add_argument("--tr", type=float, default=None, help="Override TR for all")
    args = p.parse_args(argv)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_epoch: list[dict] = []
    all_spec: list[dict] = []
    for ds in ds_list:
        e, s = ingest_dataset(
            ds, backend=args.backend, max_runs=args.max_runs, tr_override=args.tr
        )
        all_epoch.extend(e)
        all_spec.extend(s)

    _write_parquet(all_epoch, OUT / "epoch_ts")
    _write_parquet(all_spec, OUT / "run_spectral")

    manifest = {
        "n_epoch_rows": len(all_epoch),
        "n_spectral_rows": len(all_spec),
        "datasets": ds_list,
        "backend": args.backend,
        "out": str(OUT.relative_to(ROOT)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest", manifest)
    return 0 if all_spec else 1


if __name__ == "__main__":
    raise SystemExit(main())
