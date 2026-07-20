"""Shared data browser — marimo view of the same /api/table store as HTML /explore.

Works offline (disk), via FastAPI, or WASM + static JSON / NEURO_API_BASE.
"""

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["marimo", "numpy", "pandas"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="Data browser")


@app.cell
def _():
    import marimo as mo
    from api_client import (
        connectivity_banner_md,
        load_datasets_registry,
        load_local_files_index,
        load_table,
        records_to_frame,
        surface_label,
    )
    from helpers import (
        load_multi_dataset_runs,
        multi_dataset_run_ids,
        multi_studies_overview_md,
        as_table,
        studies_dataframe,
    )

    return (
        connectivity_banner_md,
        load_datasets_registry,
        load_local_files_index,
        load_multi_dataset_runs,
        load_table,
        mo,
        multi_dataset_run_ids,
        multi_studies_overview_md,
        as_table,
        records_to_frame,
        studies_dataframe,
        surface_label,
    )


@app.cell
def _(connectivity_banner_md, mo, surface_label):
    mo.vstack(
        [
            mo.md(
                f"""
# Data browser

One feature store, **three UIs**:

| Surface | How to open |
|---------|-------------|
| **This marimo notebook** | `marimo edit` or `/book/00_data_browser/` |
| **HTML explorer** (no Pyodide) | `/explore/` |
| **WASM chapters** | `/wasm/00_qc_dashboard/` … |

{connectivity_banner_md()}
"""
            ),
            mo.md(f"Active surface label: **`{surface_label()}`**"),
        ]
    )
    return


@app.cell
def _(mo):
    table_name = mo.ui.dropdown(
        options=[
            "spectral",
            "spectral_clean",
            "subject",
            "condition",
            "qc",
            "participants",
            "events",
            "datasets",
            "tf_results",
            "bakeoff",
        ],
        value="spectral",
        label="Table",
    )
    prefer_api = mo.ui.checkbox(value=False, label="Prefer HTTP API over disk")
    mo.hstack([table_name, prefer_api], justify="start")
    return prefer_api, table_name


@app.cell
def _(load_table, mo, prefer_api, records_to_frame, table_name):
    payload = load_table(table_name.value, prefer_api=bool(prefer_api.value))
    recs = payload.get("records") or []
    # object-shaped (bakeoff/tf)
    if recs and not isinstance(recs[0], dict):
        recs = [{"value": r} for r in recs]
    df = records_to_frame(recs)
    n = payload.get("n", getattr(df, "height", len(recs)))
    mo.vstack(
        [
            mo.md(
                f"**Source:** `{payload.get('source')}` · rows **{n}** "
                f"(showing {getattr(df, 'height', len(recs))})"
            ),
            mo.ui.table(df, selection=None, page_size=15)
            if getattr(df, "height", len(recs) if hasattr(recs, "__len__") else 0)
            else mo.md("*Empty.*"),
        ]
    )
    return df, payload


@app.cell
def _(
    load_datasets_registry,
    load_local_files_index,
    load_multi_dataset_runs,
    mo,
    multi_dataset_run_ids,
    multi_studies_overview_md,
    as_table,
    records_to_frame,
    studies_dataframe,
):
    reg = load_datasets_registry()
    files = load_local_files_index()
    studies = studies_dataframe()
    multi = load_multi_dataset_runs()
    ds_ids = multi_dataset_run_ids()
    ds = reg.get("datasets") or {}
    if studies is not None and not getattr(studies, "empty", True):
        ds_rows = studies.to_dict(orient="records")
    elif isinstance(ds, dict):
        ds_rows = [{"dataset": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in ds.items()]
    else:
        ds_rows = []
    # shrink file list for UI
    fl = [
        {k: f.get(k) for k in ("path", "layer", "bytes", "previewable", "table_key", "n_bold_files", "kind")}
        for f in (files.get("files") or [])[:200]
    ]
    blocks = [
        mo.md(multi_studies_overview_md()),
        mo.md(f"## Datasets registry · source `{reg.get('source')}`"),
        mo.ui.table(records_to_frame(ds_rows), selection=None, page_size=12)
        if ds_rows
        else mo.md("*No registry.*"),
    ]
    if multi is not None and not getattr(multi, "empty", True):
        show = [
            c
            for c in (
                "dataset",
                "subject",
                "group",
                "task",
                "run",
                "tsnr",
                "power_high",
                "spectral_centroid",
            )
            if c in multi.columns
        ]
        blocks.append(
            mo.md(
                f"## Multi-set spectral runs · datasets `{', '.join(ds_ids)}` · n={len(multi)}"
            )
        )
        blocks.append(
            mo.ui.table(
                as_table(multi[show].head(50).round(4) if show else multi.head(50)),
                selection=None,
                page_size=15,
            )
        )
    blocks.extend(
        [
            mo.md(f"## Local files index · source `{files.get('source')}` · n={files.get('n', len(fl))}"),
            mo.ui.table(records_to_frame(fl), selection=None, page_size=15)
            if fl
            else mo.md(
                "*No file index (WASM without NEURO_API_BASE). "
                "Run `uvicorn app:app` and set API base for host browse.*"
            ),
        ]
    )
    mo.vstack(blocks)
    return


if __name__ == "__main__":
    app.run()
