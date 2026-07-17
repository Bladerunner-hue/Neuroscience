"""Chapter VII — God-mode Spark (Tungsten/Catalyst honour) + multi-dataset mix.

Local-only chapter (not WASM): launches pure Catalyst aggregation on
pre-ingested Parquet and demos the tf.data hand-off.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "pyspark", "pyarrow", "msgspec", "tensorflow"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="VII · Spark God Mode")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    import json
    import subprocess
    import sys

    ROOT = Path.cwd()
    if not (ROOT / "scripts").exists():
        ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()

    mo.md(
        rf"""
# VII · God-mode Spark (Honour)

**Pure Tungsten / Catalyst** on CPU — no RAPIDS, no JVM GPU, no Python UDFs.

| Stage | Where | What |
|-------|--------|------|
| 1. Spectral pre-ingest | Outside Spark (`pre_ingest_bold_to_parquet.py`) | NIfTI → multitaper bands + `array<double>` TS |
| 2. Catalyst scale-out | Spark local or Connect | `groupBy` / windows / contrasts only |
| 3. Sacred hand-off | TensorFlow only | TFRecords → `tf.data.Dataset` |

Multi-dataset mix: `ds000171` + OpenNeuro cross-refs from the registry.
"""
    )
    return Path, ROOT, json, mo, np, pd, plt, subprocess, sys


@app.cell
def _(ROOT, mo):
    reg_path = ROOT / "data" / "processed" / "dataset_registry.json"
    god_man = ROOT / "data" / "processed" / "god_parquet_bold" / "manifest.json"
    feat_dir = ROOT / "data" / "processed" / "god_features" / "subject_level"

    status = {
        "registry": reg_path.exists(),
        "god_parquet": god_man.exists(),
        "god_features": feat_dir.exists(),
        "tfrecords": (ROOT / "data" / "processed" / "god_tfrecords").exists(),
    }
    mo.md(
        f"""
## 0. Pipeline status

| Artifact | Ready |
|----------|-------|
| dataset_registry.json | **{status['registry']}** |
| god_parquet_bold | **{status['god_parquet']}** |
| god_features (Catalyst) | **{status['god_features']}** |
| god_tfrecords | **{status['tfrecords']}** |

```bash
python scripts/download_openneuro_cohorts.py
python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171
# optional multi-set:
# python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725 --max-runs 4
python scripts/god_mode_bold_to_tfdata.py --smoke-tfdata
# Spark Connect:
# docker compose -f docker-compose.spark.yml up -d
# python scripts/god_mode_bold_to_tfdata.py --remote sc://localhost:15002
```
"""
    )
    return feat_dir, god_man, reg_path, status


@app.cell
def _(ROOT, json, mo, pd, reg_path):
    _blocks = [mo.md("## 1. Dataset registry (mix)")]
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        rows = []
        for k, v in reg.items():
            rows.append(
                {
                    "dataset": k,
                    "role": v.get("role"),
                    "status": v.get("status"),
                    "n_subjects_on_disk": v.get("n_subjects_on_disk"),
                    "why": (v.get("why") or "")[:60],
                }
            )
        _blocks.append(mo.ui.table(pd.DataFrame(rows)))
    else:
        _blocks.append(
            mo.md("*No registry yet — run `python scripts/download_openneuro_cohorts.py`.*")
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(ROOT, feat_dir, mo, pd, plt):
    _blocks = [mo.md("## 2. Catalyst subject features")]
    if feat_dir.exists():
        try:
            subj = pd.read_parquet(feat_dir)
        except Exception:
            # multi-part
            import pyarrow.parquet as pq

            subj = pq.read_table(feat_dir).to_pandas()
        _blocks.append(mo.md(f"**Rows:** {len(subj)} · **Cols:** {list(subj.columns)[:12]}…"))
        _blocks.append(mo.ui.table(subj.head(20).round(4)))
        if "run_power_high_mean" in subj.columns and "dataset" in subj.columns:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            for ds, g in subj.groupby("dataset"):
                ax.hist(
                    g["run_power_high_mean"].dropna(),
                    bins=8,
                    alpha=0.55,
                    label=ds,
                )
            ax.set_xlabel("run_power_high_mean (Catalyst)")
            ax.legend(fontsize=8)
            ax.set_title("Multi-dataset high-band power")
            fig.tight_layout()
            _blocks.append(fig)
    else:
        _blocks.append(
            mo.md("*Run `python scripts/god_mode_bold_to_tfdata.py` to materialize features.*")
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(ROOT, mo, status):
    _blocks = [
        mo.md(
            r"""
## 3. Honour rules (checklist)

1. **Spectral / STFT / multitaper** — only in `pre_ingest_bold_to_parquet.py` (scipy or optional CuPy).  
2. **Spark** — Tungsten columnar + Catalyst AQE; `groupBy` / `Window` / SQL `expr` only.  
3. **No** Pandas UDFs, **no** RAPIDS, **no** JVM GPU.  
4. **tf.data** — only place a GPU is allowed (TensorFlow).  
5. **msgspec** — metadata JSON at `data/processed/god_config_metadata.json`.

## 4. Why this scales

Your LOOCV bake-off stays the scientific gold standard for small *n*.  
God-mode adds a **horizontal** path: same schema, multi-dataset Parquet, Catalyst rollups, TFRecords for larger cohorts — without rewriting the public WASM book.
"""
        )
    ]
    meta_path = ROOT / "data" / "processed" / "god_config_metadata.json"
    if meta_path.exists():
        import msgspec

        meta = msgspec.json.decode(meta_path.read_bytes())
        _blocks.append(mo.md(f"```json\n{msgspec.json.encode(meta).decode()}\n```"))
    mo.vstack(_blocks)
    return


@app.cell
def _(mo):
    mo.md(
        """
## Takeaways

- Download cross-refs → registry JSON  
- Pre-ingest BOLD → `god_parquet_bold/`  
- Catalyst → `god_features/` + TFRecords  
- Live API still serves classic processed CSVs; god tables are the scale path  

**Prev:** [V · TF results](../05_tf_results/) · **Home** [Gallery](../../)
"""
    )
    return


if __name__ == "__main__":
    app.run()
