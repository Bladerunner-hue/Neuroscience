"""Shared data browser — marimo view of the same /api/table store as HTML /explore.

Works offline (disk), via FastAPI, or WASM + static JSON / NEURO_API_BASE.
"""

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["marimo", "numpy", "pandas", "polars", "pyarrow"]
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
        records_to_polars,
        surface_label,
    )

    return (
        connectivity_banner_md,
        load_datasets_registry,
        load_local_files_index,
        load_table,
        mo,
        records_to_polars,
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
def _(load_table, mo, prefer_api, records_to_polars, table_name):
    payload = load_table(table_name.value, prefer_api=bool(prefer_api.value))
    recs = payload.get("records") or []
    # object-shaped (bakeoff/tf)
    if recs and not isinstance(recs[0], dict):
        recs = [{"value": r} for r in recs]
    df = records_to_polars(recs)
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
def _(load_datasets_registry, load_local_files_index, mo, records_to_polars):
    reg = load_datasets_registry()
    files = load_local_files_index()
    ds = reg.get("datasets") or {}
    if isinstance(ds, dict):
        ds_rows = [{"dataset": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in ds.items()]
    else:
        ds_rows = []
    # shrink file list for UI
    fl = [
        {k: f.get(k) for k in ("path", "layer", "bytes", "previewable", "table_key", "n_bold_files", "kind")}
        for f in (files.get("files") or [])[:200]
    ]
    mo.vstack(
        [
            mo.md(f"## Datasets registry · source `{reg.get('source')}`"),
            mo.ui.table(records_to_polars(ds_rows), selection=None, page_size=12)
            if ds_rows
            else mo.md("*No registry.*"),
            mo.md(f"## Local files index · source `{files.get('source')}` · n={files.get('n', len(fl))}"),
            mo.ui.table(records_to_polars(fl), selection=None, page_size=15)
            if fl
            else mo.md(
                "*No file index (WASM without NEURO_API_BASE). "
                "Run `uvicorn app:app` and set API base for host browse.*"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
