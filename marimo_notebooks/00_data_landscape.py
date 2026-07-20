"""Chapter 0-local — Data landscape: what we started with, what we added, why, links.

All datasets live under ``data/raw/<OpenNeuro-id>/`` (see ``data/README.md``).
Local-first marimo chapter (not WASM). Polars default; optional Spark via pip + native Connect.
"""

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "marimo", "numpy", "pandas", "matplotlib",
#   "scikit-learn", "scipy", "openneuro-py", "pyspark"
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="0 · Data Landscape")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    from pathlib import Path
    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        callout,
        data_sources_scientific_md,
        hypothesis_card,
        inventory_dataframe,
        load_dataset_registry,
        load_god_run_level_df,
        load_participants_df,
        load_spectral_features_frame,
        load_subject_features,
        as_table,
        primary_cohort_summary,
        set_global_style,
    )
    from multi_dataset_catalog import (
        MULTI_DATASET_CATALOG,
        catalog_rows,
        integration_roadmap_md,
    )
    from spark_session import get_spark, read_parquet_spark, repo_root

    set_global_style()
    ROOT = repo_root()
    DATA = ROOT / "data"
    RAW = DATA / "raw"
    PROC = DATA / "processed"
    return (
        CONTROL_COLOR,
        DATA,
        MDD_COLOR,
        MULTI_DATASET_CATALOG,
        MUSIC_COLOR,
        PROC,
        Path,
        RAW,
        ROOT,
        book_nav,
        callout,
        catalog_rows,
        data_sources_scientific_md,
        get_spark,
        hypothesis_card,
        integration_roadmap_md,
        inventory_dataframe,
        load_dataset_registry,
        load_god_run_level_df,
        load_participants_df,
        load_spectral_features_frame,
        load_subject_features,
        mo,
        np,
        as_table,
        pd,
        pl,
        plt,
        primary_cohort_summary,
        read_parquet_spark,
    )


@app.cell
def _(callout, hypothesis_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
# 0 · Data Landscape

### Where the data live, what we started with, what we added, and why

**All OpenNeuro trees are under the repo data directory:**

```
interviews/Neuroscience/data/
├── README.md
├── raw/
│   ├── ds000171/          ← PRIMARY (full BOLD)
│   ├── ds002725/ … ds006564/
│   └── (one folder per OpenNeuro id)
└── processed/             ← feature store + god multi-set tables
```

| Surface | Path |
|---------|------|
| BIDS raw | `data/raw/<ds_id>/` |
| Feature store (book) | `data/processed/*.csv` |
| Multi-set scale | `data/processed/god_*` |
| Live registry | `data/processed/dataset_registry.json` |
| Scientific catalog | `marimo_notebooks/multi_dataset_catalog.py` |

**Stack:** `pip install -r requirements.txt` (openneuro-py, polars, pyspark, …).  
Spark Connect is **native** (`./scripts/start_local_spark_connect.sh`) when needed — not required for this chapter’s Polars analysis.
"""
            ),
            mo.md(
                hypothesis_card(
                    "Spectral music–reward motifs should be comparable after schema alignment.",
                    "Aligned multitaper / tSNR features and music-related contrasts should share "
                    "direction across affective-music cohorts after harmonizing "
                    "`(dataset, subject, task, run)` — starting from primary ds000171.",
                )
            ),
            mo.md(
                callout(
                    "data",
                    "If folders look empty in the IDE",
                    "Open **`data/raw/`** (not the repo root). Primary BOLD is large (~21 GB) under "
                    "`data/raw/ds000171/` and is often gitignored — it is still on disk locally. "
                    "Cross-refs may be **meta-only** until you run `download_openneuro_cohorts.py`. "
                    "Each folder has a `STATUS.txt` when BOLD is not yet present.",
                )
            ),
        ]
    )
    return


@app.cell
def _(RAW, mo, primary_cohort_summary):
    summary = primary_cohort_summary()
    # list raw children for visibility
    raw_dirs = sorted([p.name for p in RAW.iterdir() if p.is_dir() and p.name.startswith("ds")]) if RAW.exists() else []
    lines = "\n".join(f"- `{d}/`" for d in raw_dirs) or "- *(no ds* folders — run download scripts)*"
    mo.md(
        f"""
## 1. What we **started with** (baseline book)

The original public methods book is built on a **single clinical study**:

### OpenNeuro [ds000171](https://openneuro.org/datasets/ds000171) — Lepping et al.

| | |
|--|--|
| **Title** | Neural Processing of Emotional Musical and Nonmusical Stimuli in Depression |
| **n** | **39** (19 MDD currently depressed, 20 never-depressed controls) |
| **Screening** | SCID; no meds at scan; no comorbid anxiety/mania |
| **Modality** | fMRI BOLD, Siemens Skyra 3T (+ T1w) |
| **Tasks** | Emotional **music** (pos/neg) vs **nonmusic** (tones / non-musical auditory) |
| **Local path** | `data/raw/ds000171/` |
| **Link** | https://openneuro.org/datasets/ds000171 |

### Preprocessing & outputs (already in the book)

| Step | Script / artifact |
|------|-------------------|
| Feature extraction | `scripts/prepare_real_features.py` (nibabel + scipy; adaptive DPSS multitaper default) |
| Run spectral + QC | `data/processed/spectral_features.csv`, `run_qc.csv`, `cleaned_spectral_features.csv` |
| Trial-type epochs | `data/processed/condition_features.csv` |
| Subject table + **R** | `data/processed/subject_features.csv` |
| ML / TF evidence | `ml_bakeoff.json`, `tf_results.json` |
| Marimo chapters | `00_qc` → `01`…`05` public WASM; `06` local TF |

### Live primary stats (this machine)

| Metric | Value |
|--------|------:|
| Participants (clean) | **{summary.get('n_participants_meta', '—')}** |
| Spectral runs (book store) | **{summary.get('n_spectral_runs', '—')}** |
| Subject feature rows | **{summary.get('n_subject_features', '—')}** |
| Runs by group | `{summary.get('runs_by_group', {})}` |
| Runs by task | `{summary.get('runs_by_task', {})}` |
| Mean tSNR | **{(summary.get('mean_tsnr') or 0):.1f}** |
| Mean power_high | **{(summary.get('mean_power_high') or 0):.4f}** |

### Folders under `data/raw/` right now

{lines}
"""
    )
    return summary,


@app.cell
def _(MULTI_DATASET_CATALOG, catalog_rows, load_dataset_registry, mo, as_table, pd, pl):
    reg = load_dataset_registry()
    cat = as_table(pd.DataFrame(catalog_rows()))
    live_rows = []
    for ds, meta in (reg or {}).items():
        live_rows.append(
            {
                "dataset": ds,
                "role": meta.get("role"),
                "status": meta.get("status"),
                "match_level": meta.get("match_level"),
                "n_subjects_on_disk": meta.get("n_subjects_on_disk"),
                "n_bold_files": meta.get("n_bold_files"),
                "has_participants_tsv": meta.get("has_participants_tsv"),
                "n_nominal": meta.get("n_participants_nominal"),
                "short_title": meta.get("short_title") or meta.get("title"),
                "local_path": meta.get("local_path") or f"data/raw/{ds}",
                "url": meta.get("url"),
            }
        )
    live = as_table(pd.DataFrame(live_rows)) if live_rows else pl.DataFrame()
    if live.height and cat.height:
        live = live.join(
            cat.select("dataset", "priority", "modality"),
            on="dataset",
            how="left",
        ).sort("priority")

    # Started vs added
    primary = [k for k, v in MULTI_DATASET_CATALOG.items() if v.get("role") == "primary"]
    added = [k for k, v in MULTI_DATASET_CATALOG.items() if v.get("role") != "primary"]

    mo.vstack(
        [
            mo.md(
                f"""
## 2. What we **added** (multi-dataset extension)

### Why add more OpenNeuro studies?

| Goal | Why |
|------|-----|
| **Cross-validate spectral biomarkers** | Same multitaper bands / tSNR / flatness outside one scanner & clinical cohort |
| **Multimodal** | EEG-fMRI (ds002725), HR–insula (ds004894) |
| **Valence / dynamics** | Happy–sad music time course (ds003085) |
| **Ecological music** | Film + controlled soundtracks (ds006564) |
| **Non-emotional music baseline** | Genre listening (ds003720) |
| **Scale path** | God-mode parquet + optional Spark rollups when *n* grows |

**Started with (primary):** {', '.join(f'`{x}`' for x in primary)}  
**Added to catalog (cross-refs):** {', '.join(f'`{x}`' for x in added)}

Each cross-ref has a dedicated folder under **`data/raw/<id>/`** (meta and/or BOLD).  
Full scientific notes: `multi_dataset_catalog.py`.
"""
            ),
            mo.md("### 2a · Live disk registry (`data/raw/*` + scan)"),
            mo.ui.table(live, selection=None, page_size=12)
            if live.height
            else mo.md("*Run `python scripts/refresh_dataset_registry.py`.*"),
            mo.md("### 2b · Ranked catalog (integration priority)"),
            mo.ui.table(
                cat.select(
                    "priority",
                    "dataset",
                    "role",
                    "match_level",
                    "short_title",
                    "n_nominal",
                    "modality",
                    "url",
                ),
                selection=None,
                page_size=12,
            ),
        ]
    )
    return cat, live, reg


@app.cell
def _(MULTI_DATASET_CATALOG, mo, reg):
    cards = [
        mo.md(
            """
## 3. Per-dataset breakdown (started vs added · why · links)

Open each **`data/raw/<id>/`** in the file tree. `STATUS.txt` explains meta-only vs BOLD-present.
"""
        )
    ]
    for ds_id, m in sorted(
        MULTI_DATASET_CATALOG.items(), key=lambda kv: kv[1].get("priority", 99)
    ):
        live_m = (reg or {}).get(ds_id) or {}
        n_sub = live_m.get("n_subjects_on_disk", 0)
        n_bold = live_m.get("n_bold_files", 0)
        st = live_m.get("status", "?")
        phase = "**STARTED WITH**" if m.get("role") == "primary" else "**ADDED**"
        cards.append(
            mo.md(
                f"""
### {phase} · `{ds_id}` — {m.get('short_title')}

| | |
|--|--|
| **Phase** | {phase} · match **{m.get('match_level')}** · priority {m.get('priority')} |
| **On disk** | status=`{st}` · subjects={n_sub} · BOLD files={n_bold} |
| **Path** | `data/raw/{ds_id}/` |
| **Cohort** | {m.get('cohort')} |
| **Modality** | {', '.join(m.get('modality') or [])} |
| **Tasks** | {'; '.join(m.get('tasks') or [])} |
| **Why (neuroscience)** | {m.get('why_neuro')} |
| **Why (preprocess / Spark path)** | {m.get('why_preprocess')} |
| **How we integrate** | {m.get('integration')} |
| **OpenNeuro** | [{ds_id}]({m.get('url')}) |
"""
            )
        )
    mo.vstack(cards)
    return


@app.cell
def _(inventory_dataframe, mo, as_table, pl):
    inv = as_table(inventory_dataframe())
    by_layer = (
        inv.group_by("layer")
        .agg(
            pl.len().alias("n_artifacts"),
            pl.col("exists").sum().alias("n_present"),
        )
        .sort("layer")
        if inv.height
        else pl.DataFrame()
    )
    mo.vstack(
        [
            mo.md(
                f"""
## 4. Processed artifacts (`data/processed/`)

| Layer | Meaning |
|-------|---------|
| `feature_store` | Original book CSVs/JSON (WASM path) |
| `god_mode` | Multi-dataset parquet (scale path) |
| `raw_bids` | Registry pointers into `data/raw/` |

**Present:** {inv.filter(pl.col('exists')).height if inv.height else 0} / {inv.height}
"""
            ),
            mo.ui.table(by_layer, selection=None) if by_layer.height else mo.md("—"),
            mo.ui.table(
                inv.select(
                    "layer", "name", "role", "exists", "n_rows", "scientific_value", "status"
                ),
                selection=None,
                page_size=18,
            )
            if inv.height
            else mo.md("*Empty inventory.*"),
        ]
    )
    return inv,


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    MUSIC_COLOR,
    load_god_run_level_df,
    load_participants_df,
    load_spectral_features_frame,
    load_subject_features,
    mo,
    as_table,
    pl,
    plt,
):
    blocks = [
        mo.md(
            """
## 5. Live analysis (primary book store + multi-dataset god features)

Compares **what we started with** (book spectral CSV) to **what multi-set ingest added**
(`god_features/run_level` with `dataset` column).
"""
        )
    ]

    runs = load_spectral_features_frame()
    if runs is None:
        runs = pl.DataFrame()
    drop = [c for c in ("psd_f", "psd_pxx") if c in runs.columns]
    runs_s = runs.drop(drop) if drop else runs

    parts = load_participants_df()
    if parts is not None and len(parts):
        gcol = "group_short" if "group_short" in parts.columns else "group"
        if gcol in parts.columns:
            fig_p, ax_p = plt.subplots(figsize=(5.5, 3.2))
            vc = parts[gcol].value_counts()
            colors = [
                CONTROL_COLOR if "Control" in str(i) or "Never" in str(i) else MDD_COLOR
                for i in vc.index
            ]
            ax_p.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white")
            ax_p.set_title("Started with — primary cohort (ds000171)")
            ax_p.set_ylabel("n participants")
            fig_p.tight_layout()
            blocks.append(fig_p)

    if runs_s.height:
        blocks.append(
            mo.md(
                f"**Book spectral (started with):** {runs_s.height} runs × {runs_s.width} cols"
            )
        )
        keys = [c for c in ("group", "task") if c in runs_s.columns]
        if len(keys) == 2:
            blocks.append(
                mo.ui.table(
                    runs_s.group_by(keys).agg(pl.len().alias("n_runs")).sort(keys),
                    selection=None,
                )
            )
        if "power_high" in runs_s.columns and "group" in runs_s.columns:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            for g, color in (("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)):
                vals = (
                    runs_s.filter(pl.col("group") == g)
                    .select("power_high")
                    .to_series()
                    .drop_nulls()
                    .to_list()
                )
                if vals:
                    axes[0].hist(
                        vals, bins=10, alpha=0.55, label=g, color=color, edgecolor="white"
                    )
            axes[0].set_xlabel("power_high")
            axes[0].set_title("Started with — high-band power")
            axes[0].legend(frameon=False)
            if "tsnr" in runs_s.columns:
                for g, color in (("Control", CONTROL_COLOR), ("MDD", MDD_COLOR)):
                    vals = (
                        runs_s.filter(pl.col("group") == g)
                        .select("tsnr")
                        .to_series()
                        .drop_nulls()
                        .to_list()
                    )
                    if vals:
                        axes[1].hist(
                            vals,
                            bins=10,
                            alpha=0.55,
                            label=g,
                            color=color,
                            edgecolor="white",
                        )
                axes[1].set_xlabel("tSNR")
                axes[1].set_title("Started with — tSNR")
                axes[1].legend(frameon=False)
            fig.tight_layout()
            blocks.append(fig)

    subj = load_subject_features()
    subj_pl = as_table(subj) if subj is not None and len(subj) else pl.DataFrame()
    if subj_pl.height:
        show = [
            c
            for c in (
                "subject",
                "group",
                "age",
                "sex",
                "responder_score",
                "R",
                "music_effect_power_high",
            )
            if c in subj_pl.columns
        ]
        blocks.append(mo.md(f"**Subject features + responder R:** {subj_pl.height} rows"))
        blocks.append(
            mo.ui.table(
                subj_pl.select(show) if show else subj_pl.head(20),
                selection=None,
                page_size=12,
            )
        )

    god = load_god_run_level_df()
    if god is not None and god.height:
        blocks.append(
            mo.md(
                f"""
### Added multi-dataset god-mode (`data/processed/god_features/run_level`)

**{god.height}** runs · datasets: `{god['dataset'].unique().to_list() if 'dataset' in god.columns else []}`

Includes primary **ds000171** plus any pre-ingested cross-refs (e.g. **ds002725** sample).
"""
            )
        )
        if set(["dataset", "group", "power_high"]).issubset(god.columns):
            agg = (
                god.group_by("dataset", "group")
                .agg(
                    pl.len().alias("n_runs"),
                    pl.col("power_high").mean().alias("mean_power_high"),
                    pl.col("tsnr").mean().alias("mean_tsnr")
                    if "tsnr" in god.columns
                    else pl.lit(None).alias("mean_tsnr"),
                )
                .sort("dataset", "group")
            )
            blocks.append(mo.ui.table(agg, selection=None))
            pdf = agg.to_pandas()
            fig2, ax2 = plt.subplots(figsize=(7.5, 3.6))
            labels = [f"{r.dataset}\n{r.group}" for r in pdf.itertuples()]
            ax2.bar(
                range(len(pdf)),
                pdf["mean_power_high"],
                color=[
                    CONTROL_COLOR
                    if "Control" in str(g)
                    else (MDD_COLOR if "MDD" in str(g) else MUSIC_COLOR)
                    for g in pdf["group"]
                ],
                edgecolor="white",
            )
            ax2.set_xticks(range(len(pdf)))
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_ylabel("mean power_high")
            ax2.set_title("Added multi-set path — high-band power by dataset×group")
            fig2.tight_layout()
            blocks.append(fig2)
    else:
        blocks.append(
            mo.md(
                """
*God multi-set table not built yet:*

```bash
pip install -r requirements.txt
python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725
python scripts/god_mode_bold_to_tfdata.py --smoke-tfdata
```
"""
            )
        )

    mo.vstack(blocks)
    return god, runs_s, subj_pl


@app.cell
def _(ROOT, get_spark, god, mo, pl, read_parquet_spark):
    spark, sinfo = get_spark(prefer_connect=True, allow_local=True)
    blocks = [
        mo.md(
            f"""
## 6. Consistency + scale path (pip / native Spark)

**Spark session:** `{sinfo.mode}` · `{sinfo.version}` · {sinfo.detail}

Install / run (no containers):

```bash
pip install -r requirements.txt          # pyspark, openneuro-py, polars, …
# optional Connect server if you have a full Spark install:
./scripts/start_local_spark_connect.sh
```
"""
        )
    ]
    god_run = ROOT / "data" / "processed" / "god_features" / "run_level"
    required = {"dataset", "subject", "task", "run", "group", "power_high", "tsnr"}
    spark_summary = None

    if spark is not None and god_run.exists():
        try:
            from pyspark.sql import functions as F

            run_df = read_parquet_spark(spark, god_run)
            cols = set(run_df.columns)
            missing = sorted(required - cols)
            key = [c for c in ("dataset", "subject", "task", "run") if c in run_df.columns]
            dup = (
                run_df.groupBy(*key).count().filter(F.col("count") > 1).count()
                if key
                else -1
            )
            spark_summary = {
                "n_runs": run_df.count(),
                "missing": missing,
                "duplicate_keys": int(dup),
                "n_datasets": run_df.select("dataset").distinct().count()
                if "dataset" in cols
                else 0,
            }
            by = (
                run_df.groupBy("dataset", "group")
                .agg(
                    F.count(F.lit(1)).alias("n_runs"),
                    F.avg("power_high").alias("mean_power_high"),
                    F.avg("tsnr").alias("mean_tsnr"),
                )
                .orderBy("dataset", "group")
            )
            blocks.append(mo.md("### Spark Catalyst rollup"))
            blocks.append(
                mo.md(
                    f"- rows **{spark_summary['n_runs']}** · datasets **{spark_summary['n_datasets']}**\n"
                    f"- missing `{spark_summary['missing'] or 'none'}` · "
                    f"duplicate keys **{spark_summary['duplicate_keys']}**"
                )
            )
            blocks.append(mo.ui.table(by.toPandas().round(4), selection=None))
        except Exception as exc:  # noqa: BLE001
            blocks.append(mo.md(f"*Spark path error → Polars: `{type(exc).__name__}: {exc}`*"))
            spark = None

    if spark_summary is None and god is not None and god.height:
        missing = sorted(required - set(god.columns))
        key = [c for c in ("dataset", "subject", "task", "run") if c in god.columns]
        dup = god.group_by(key).len().filter(pl.col("len") > 1).height if key else 0
        blocks.append(mo.md("### Polars consistency (no Spark required)"))
        blocks.append(
            mo.md(
                f"- rows **{god.height}** · missing `{missing or 'none'}` · duplicate keys **{dup}**"
            )
        )

    mo.vstack(blocks)
    return sinfo, spark, spark_summary


@app.cell
def _(integration_roadmap_md, mo):
    mo.md(
        """
## 7. How to pull / refresh datasets (pip)

```bash
cd interviews/Neuroscience
pip install -r requirements.txt

# Metadata for all catalog ids
python scripts/download_openneuro_cohorts.py

# Limited BOLD for a cross-ref
python scripts/download_openneuro_cohorts.py --with-bold --max-subjects 1 --only ds002725

# Rescan data/raw → dataset_registry.json
python scripts/refresh_dataset_registry.py

# Multi-set spectral pre-ingest + Catalyst tables
python scripts/pre_ingest_bold_to_parquet.py --datasets ds000171,ds002725
python scripts/god_mode_bold_to_tfdata.py --smoke-tfdata
```
"""
        + integration_roadmap_md()
    )
    return


@app.cell
def _(ROOT, mo, sinfo, spark_summary):
    checks = [
        ("`data/raw/ds000171` primary BIDS", (ROOT / "data/raw/ds000171").exists()),
        ("`data/processed/spectral_features.csv`", (ROOT / "data/processed/spectral_features.csv").exists()),
        ("`data/processed/dataset_registry.json`", (ROOT / "data/processed/dataset_registry.json").exists()),
        ("`data/README.md` layout docs", (ROOT / "data/README.md").exists()),
        ("Cross-ref folders under data/raw", any((ROOT / "data/raw").glob("ds00*"))),
        ("God multi-set run_level", (ROOT / "data/processed/god_features/run_level").exists()),
        (
            "Spark optional (connect/local)",
            sinfo.mode in ("connect", "local") if sinfo else False,
        ),
        (
            "No duplicate multi-set keys",
            (spark_summary or {}).get("duplicate_keys", 0) == 0
            if spark_summary
            else None,
        ),
    ]
    lines = []
    for label, ok in checks:
        if ok is None:
            mark, note = "·", "n/a"
        elif ok:
            mark, note = "✓", "ok"
        else:
            mark, note = "✗", "action needed"
        lines.append(f"| {mark} | {label} | {note} |")
    mo.md(
        "## 8. Gate\n\n"
        "| | Check | Status |\n|---|---|---|\n"
        + "\n".join(lines)
        + "\n\n**Next:** [0 · QC Dashboard](../00_qc_dashboard/) · Spark scale: `07_spark_god_mode.py`"
    )
    return


@app.cell
def _(book_nav, mo):
    mo.md(
        """
## Takeaways

| | |
|--|--|
| **Started with** | ds000171 clinical music/nonmusic design + committed feature store under `data/processed/` |
| **Added** | Ranked OpenNeuro cross-refs under `data/raw/<id>/`, catalog, registry, god multi-set path |
| **Where to look** | Always `data/raw/` and `data/processed/` (see `data/README.md`) |
| **Install** | `pip install -r requirements.txt` — openneuro-py for pulls; pyspark optional |
| **Spark** | Native Connect script when scaling; not required for Polars landscape analysis |

"""
        + book_nav("00_data_landscape")
    )
    return


if __name__ == "__main__":
    app.run()
