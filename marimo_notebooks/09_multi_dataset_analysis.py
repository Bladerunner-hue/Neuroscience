"""Chapter IX — Multi-dataset spectral scale (public WASM-safe + local Spark/TF hooks).

Browser: precomputed god_run_summary + datasets registry + TF metrics (no JVM/TF).
Host: optional Spark Catalyst rollups; TF retrain via scripts/run_tf_offline.py.
"""

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="IX · Multi-dataset scale")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        callout,
        hypothesis_card,
        load_multi_dataset_runs,
        load_raw_participants,
        load_tf_results,
        multi_dataset_run_ids,
        multi_studies_overview_md,
        as_table,
        set_global_style,
        studies_dataframe,
    )
    from api_client import connectivity_banner_md, load_datasets_registry, load_table, surface_label
    from multi_dataset_catalog import MULTI_DATASET_CATALOG, integration_roadmap_md

    set_global_style()
    return (
        CONTROL_COLOR,
        MDD_COLOR,
        MULTI_DATASET_CATALOG,
        MUSIC_COLOR,
        book_nav,
        callout,
        connectivity_banner_md,
        hypothesis_card,
        integration_roadmap_md,
        load_datasets_registry,
        load_multi_dataset_runs,
        load_raw_participants,
        load_table,
        load_tf_results,
        mo,
        multi_dataset_run_ids,
        multi_studies_overview_md,
        np,
        pd,
        as_table,
        plt,
        studies_dataframe,
        surface_label,
    )


@app.cell
def _(callout, connectivity_banner_md, hypothesis_card, mo, surface_label):
    mo.vstack(
        [
            mo.md(
                rf"""
# IX · Multi-dataset scale

### Cross-cohort spectral maps · Catalyst path · TensorFlow hand-off

| Layer | Where | WASM? |
|-------|--------|-------|
| Primary feature store | `data/processed/*` + `/api/table/*` | ✅ |
| Multi-set god summary | `god_run_summary.json` / embedded | ✅ |
| Spark Catalyst | host / Connect | ❌ (precompute → JSON) |
| TensorFlow train | `scripts/run_tf_offline.py` | ❌ (metrics only) |

Also open the **HTML explorer** (no Pyodide): [explore/](../../explore/) · live [API](../../api/).

{connectivity_banner_md()}
"""
            ),
            mo.md(f"**Surface:** `{surface_label()}`"),
            mo.md(
                hypothesis_card(
                    "Multi-dataset high-band structure remains comparable after key alignment.",
                    "God-mode run_level rows share `(dataset, subject, task, run)` and spectral bands "
                    "with the primary book store; healthy cross-refs (e.g. ds002725) anchor generalization.",
                )
            ),
            mo.md(
                callout(
                    "method",
                    "PySpark honour (host)",
                    "Spectral pre-ingest **outside** JVM → pure DataFrame `groupBy` / windows → "
                    "optional TFRecords → TensorFlow `tf.data`. "
                    "See `07_spark_god_mode.py` and `scripts/god_mode_bold_to_tfdata.py`.",
                )
            ),
        ]
    )
    return


@app.cell
def _(
    MULTI_DATASET_CATALOG,
    integration_roadmap_md,
    load_datasets_registry,
    load_multi_dataset_runs,
    load_raw_participants,
    mo,
    multi_dataset_run_ids,
    multi_studies_overview_md,
    as_table,
    pd,
    studies_dataframe,
):
    reg = load_datasets_registry()
    studies = studies_dataframe()
    multi = load_multi_dataset_runs()
    ds_ids = multi_dataset_run_ids()
    # Prefer live studies_dataframe (raw + registry + catalog); fall back to API registry
    ds = reg.get("datasets") or {}
    rows = []
    if studies is not None and not getattr(studies, "empty", True):
        rows = studies.to_dict(orient="records")
    elif isinstance(ds, dict):
        for k, v in ds.items():
            if not isinstance(v, dict):
                continue
            rows.append(
                {
                    "dataset": k,
                    "role": v.get("role"),
                    "status": v.get("status"),
                    "match_level": v.get("match_level"),
                    "n_subjects_on_disk": v.get("n_subjects_on_disk"),
                    "n_bold_files": v.get("n_bold_files"),
                    "short_title": (v.get("short_title") or v.get("title") or "")[:48],
                    "url": v.get("url"),
                }
            )
    df = as_table(pd.DataFrame(rows)) if rows else None

    # Per-study scientific + disk detail from catalog
    detail_rows = []
    for ds_id, cat in sorted(
        MULTI_DATASET_CATALOG.items(), key=lambda kv: kv[1].get("priority", 99)
    ):
        live = next((r for r in rows if r.get("dataset") == ds_id), {}) if rows else {}
        n_part = 0
        try:
            p = load_raw_participants(ds_id)
            n_part = int(len(p)) if p is not None and not p.empty else 0
        except Exception:
            n_part = 0
        detail_rows.append(
            {
                "dataset": ds_id,
                "priority": cat.get("priority"),
                "role": cat.get("role"),
                "match": cat.get("match_level"),
                "status": live.get("status") or "—",
                "n_subjects_disk": live.get("n_subjects_on_disk"),
                "n_bold": live.get("n_bold_files"),
                "n_participants_tsv": n_part,
                "in_spectral_multiset": ds_id in ds_ids,
                "short_title": cat.get("short_title"),
                "tasks": ", ".join(cat.get("tasks") or [])[:60],
                "why_neuro": (cat.get("why_neuro") or "")[:100],
            }
        )
    detail = as_table(pd.DataFrame(detail_rows)) if detail_rows else None

    blocks = [
        mo.md(f"## 1. Dataset registry · source `{reg.get('source')}`"),
        mo.md(multi_studies_overview_md()),
        mo.ui.table(df, selection=None, page_size=12) if df is not None else mo.md("*No registry.*"),
        mo.md("### Catalog + disk integration (all studies)"),
        mo.ui.table(detail, selection=None, page_size=12) if detail is not None else mo.md("*No catalog.*"),
        mo.md(integration_roadmap_md()),
    ]
    if multi is not None and not getattr(multi, "empty", True) and "dataset" in multi.columns:
        n_by = multi.groupby("dataset").size().reset_index(name="n_spectral_runs")
        blocks.append(mo.md("### Spectral multi-set run counts"))
        blocks.append(mo.ui.table(as_table(n_by), selection=None))
    mo.vstack(blocks)
    return df, multi, reg, studies


@app.cell
def _(CONTROL_COLOR, MDD_COLOR, MUSIC_COLOR, mo, np, pd, as_table, plt):
    # Load multi-set summary: book_data embed (WASM) → disk → empty
    god = {"n_runs": 0, "records": [], "by_dataset_group": [], "datasets": []}
    try:
        import book_data as bd

        if hasattr(bd, "GOD_RUN_SUMMARY"):
            god = bd.GOD_RUN_SUMMARY
    except Exception:
        pass
    if not god.get("n_runs"):
        from pathlib import Path

        for root in (Path.cwd(), Path.cwd().parent):
            p = root / "data" / "processed" / "god_run_summary.json"
            if p.exists():
                import json

                god = json.loads(p.read_text())
                break

    by = pd.DataFrame(god.get("by_dataset_group") or [])
    rec = pd.DataFrame(god.get("records") or [])
    blocks = [
        mo.md(
            f"""
## 2. Multi-dataset god-mode summary

**Runs (full table):** {god.get('n_runs', 0)} · **Embedded sample:** {god.get('n_records_embedded', len(rec))}  
**Datasets:** `{god.get('datasets') or []}`
"""
        )
    ]
    if len(by):
        blocks.append(mo.ui.table(as_table(by), selection=None))
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        labels = [f"{r.dataset}\n{r.group}" for r in by.itertuples()]
        vals = by["mean_power_high"].astype(float).tolist() if "mean_power_high" in by.columns else []
        colors = []
        for g in by.get("group", []):
            if "Control" in str(g):
                colors.append(CONTROL_COLOR)
            elif "MDD" in str(g):
                colors.append(MDD_COLOR)
            else:
                colors.append(MUSIC_COLOR)
        if vals:
            ax.bar(range(len(vals)), vals, color=colors or MUSIC_COLOR, edgecolor="white")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("mean power_high")
            ax.set_title("Multi-set high-band power (Catalyst / god-mode)")
            fig.tight_layout()
            blocks.append(fig)
    if len(rec):
        show = [c for c in rec.columns if c in (
            "dataset", "subject", "group", "task", "run", "tsnr", "power_high", "spectral_centroid"
        )]
        blocks.append(mo.md("### Sample run-level rows (capped for WASM)"))
        blocks.append(mo.ui.table(as_table(rec[show] if show else rec), selection=None, page_size=12))
    else:
        blocks.append(
            mo.md(
                "*No god summary — on host: "
                "`python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725` "
                "then rebuild summary / export WASM.*"
            )
        )
    mo.vstack(blocks)
    return by, god, rec


@app.cell
def _(load_table, load_tf_results, mo, as_table, pd, plt):
    # Primary book store via unified loader (disk / API / bundle)
    spec = load_table("spectral")
    subj = load_table("subject")
    tf = load_tf_results() or {}

    blocks = [
        mo.md(
            f"""
## 3. Primary book store + TensorFlow metrics

| Table | n | source |
|-------|--:|--------|
| spectral | {spec.get('n')} | `{spec.get('source')}` |
| subject | {subj.get('n')} | `{subj.get('source')}` |
| tf_results | {len(tf) if isinstance(tf, dict) else 0} | offline / API / bundle |
"""
        )
    ]
    srec = spec.get("records") or []
    if srec:
        sdf = pd.DataFrame(srec)
        if "group" in sdf.columns and "power_high" in sdf.columns:
            fig, ax = plt.subplots(figsize=(6.5, 3.4))
            for g, color in (("Control", "#1B4F72"), ("MDD", "#922B21")):
                v = pd.to_numeric(sdf.loc[sdf["group"] == g, "power_high"], errors="coerce").dropna()
                if len(v):
                    ax.hist(v, bins=10, alpha=0.55, label=g, color=color, edgecolor="white")
            ax.set_title("Primary spectral power_high (book store)")
            ax.legend(frameon=False)
            fig.tight_layout()
            blocks.append(fig)

    # TF summary table (no training)
    if isinstance(tf, dict) and tf:
        # flatten models if present
        models = tf.get("models") or tf.get("results") or []
        if isinstance(models, dict):
            mrows = [{"model": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in models.items()]
        elif isinstance(models, list):
            mrows = models
        else:
            mrows = [{"key": k, "value": str(v)[:80]} for k, v in list(tf.items())[:20]]
        blocks.append(mo.md("### TensorFlow offline results (view-only)"))
        blocks.append(mo.ui.table(as_table(pd.DataFrame(mrows)), selection=None, page_size=10))
        blocks.append(
            mo.md(
                "Retrain on host: `python scripts/run_tf_offline.py` · "
                "interactive: `marimo edit marimo_notebooks/06_tf_spectrogram_model.py` · "
                "then re-export WASM so Pages picks up new metrics."
            )
        )
    else:
        blocks.append(mo.md("*No tf_results yet — run offline TF script on the host.*"))

    mo.vstack(blocks)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## 4. Host scale recipes (PySpark + TensorFlow)

```bash
# Multi-set spectral outside JVM
python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725
python scripts/god_mode_bold_to_tfdata.py --smoke-tfdata

# Pure Catalyst rollups (optional Connect)
./scripts/start_local_spark_connect.sh
python scripts/god_mode_bold_to_tfdata.py --remote sc://localhost:15002

# TensorFlow hand-off (GPU allowed only here)
python scripts/run_tf_offline.py

# Refresh Pages tables
python marimo_exports/export_wasm.py --sync-docs
```

**Honour rules:** no Python UDFs in Spark hot paths; no pandas collect of full BOLD;  
TF only after TFRecords / small feature tensors exist.
"""
    )
    return


@app.cell
def _(book_nav, mo):
    mo.md(
        """
## Takeaways

1. **Public WASM** shows multi-set **summaries** + primary tables without Spark/TF in the browser.  
2. **HTML explore** and this chapter share `/api/table/*` + static JSON on GitHub Pages.  
3. **Bigger analysis** grows via god parquet → Catalyst → TF offline → re-export.  

"""
        + book_nav("09_multi_dataset_analysis")
    )
    return


if __name__ == "__main__":
    app.run()
