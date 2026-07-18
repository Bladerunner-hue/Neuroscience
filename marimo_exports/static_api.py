#!/usr/bin/env python3
"""Export FastAPI feature endpoints as static JSON for GitHub Pages.

GitHub Pages cannot run a Python process. This module freezes the same payloads
served by ``fastapi_app.py`` into ``docs/api/**`` so the public site exposes:

  https://<user>.github.io/Neuroscience/api/          → interactive explorer
  https://<user>.github.io/Neuroscience/api/health.json
  https://<user>.github.io/Neuroscience/api/meta.json
  https://<user>.github.io/Neuroscience/api/features/spectral.json
  https://<user>.github.io/Neuroscience/api/features/spectral_clean.json
  https://<user>.github.io/Neuroscience/api/features/subject.json
  https://<user>.github.io/Neuroscience/api/features/condition.json
  https://<user>.github.io/Neuroscience/api/qc.json
  https://<user>.github.io/Neuroscience/api/bakeoff.json
  https://<user>.github.io/Neuroscience/api/tf_results.json
  https://<user>.github.io/Neuroscience/api/openapi.json

Local FastAPI still provides live routes at /api/* without the .json suffix.

Usage (repo root):
  python marimo_exports/static_api.py
  python marimo_exports/static_api.py --out docs/api
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DEFAULT_OUT = DOCS / "api"

# Drop bulky columns from tabular API payloads (full PSDs live in WASM bundle)
DROP_COLS = ("psd_f", "psd_pxx", "peri_stim")


def _read_csv_records(name: str) -> list[dict[str, Any]]:
    path = PROCESSED / name
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_csv(path)
    drop = [c for c in DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return json.loads(df.to_json(orient="records"))


def _bundle() -> dict[str, Any]:
    path = PROCESSED / "book_bundle.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build_health(docs: Path | None = None) -> dict[str, Any]:
    docs = docs or DOCS
    b = _bundle()
    meta = {
        "psd_method": b.get("psd_method"),
        "psd_nw": b.get("psd_nw"),
        "n_bold_runs": b.get("n_bold_runs"),
        "n_cleaned_runs": b.get("n_cleaned_runs"),
        "n_subjects_with_bold": b.get("n_subjects_with_bold"),
        "n_participants_full": b.get("n_participants_full"),
        "source": b.get("source"),
    }
    wasm = (
        sorted(p.name for p in (docs / "wasm").iterdir() if p.is_dir())
        if (docs / "wasm").exists()
        else []
    )
    return {
        "status": "ok",
        "surface": "static-github-pages",
        "docs_exists": docs.exists(),
        "wasm_chapters": wasm,
        "processed": sorted(p.name for p in PROCESSED.glob("*")) if PROCESSED.exists() else [],
        "bundle": meta,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endpoints": {
            "health": "api/health.json",
            "meta": "api/meta.json",
            "spectral": "api/features/spectral.json",
            "spectral_clean": "api/features/spectral_clean.json",
            "subject": "api/features/subject.json",
            "condition": "api/features/condition.json",
            "qc": "api/qc.json",
            "bakeoff": "api/bakeoff.json",
            "tf_results": "api/tf_results.json",
            "openapi": "api/openapi.json",
            "architecture": "api/architecture.json",
            "datasets": "api/datasets.json",
            "local_files": "api/local/files.json",
            "surfaces": "api/surfaces.json",
            "table": "api/table/{name}.json",
            "explorer": "explore/",
        },
        "live_server": "uvicorn app:app  # /book/* marimo + /api/* REST",
    }


def build_meta() -> dict[str, Any]:
    b = _bundle()
    return {
        k: b.get(k)
        for k in (
            "source",
            "tr_sec",
            "psd_method",
            "psd_nw",
            "n_participants_full",
            "n_bold_runs",
            "n_cleaned_runs",
            "n_subjects_with_bold",
            "spatial_keys",
            "stim_map",
        )
    }


def build_features_spectral(*, clean: bool = False) -> dict[str, Any]:
    if clean:
        name = "cleaned_spectral_features.csv"
        if (PROCESSED / name).exists():
            rows = _read_csv_records(name)
        else:
            rows = [r for r in _read_csv_records("spectral_features.csv") if not r.get("qc_outlier")]
        return {"n": len(rows), "clean": True, "records": rows}
    rows = _read_csv_records("spectral_features.csv")
    return {"n": len(rows), "clean": False, "records": rows}


def build_features_subject() -> dict[str, Any]:
    rows = _read_csv_records("subject_features.csv")
    return {"n": len(rows), "records": rows}


def build_features_condition() -> dict[str, Any]:
    rows = _read_csv_records("condition_features.csv")
    return {"n": len(rows), "records": rows}


def build_qc() -> dict[str, Any]:
    rows = _read_csv_records("run_qc.csv")
    return {"n": len(rows), "records": rows}


def build_bakeoff() -> dict[str, Any]:
    path = PROCESSED / "ml_bakeoff.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    b = _bundle()
    return b.get("ml_bakeoff") or {}


def build_tf_results() -> dict[str, Any]:
    path = PROCESSED / "tf_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    b = _bundle()
    return b.get("tf_results") or {}


def build_architecture() -> dict[str, Any]:
    """Static copy of the live /api/architecture payload (subset)."""
    return {
        "dataset": {
            "id": "ds000171",
            "url": "https://openneuro.org/datasets/ds000171",
            "tr_sec": 3.0,
        },
        "layers": {
            "notebooks": "marimo_notebooks/",
            "live": "uvicorn app:app → /book/* + /api/* + /explore/",
            "pages": "docs/wasm + docs/api + docs/explore (static)",
            "viz": {
                "marimo_wasm": "/wasm/<chapter>/",
                "marimo_live": "/book/<chapter>/",
                "html_explorer": "/explore/",
            },
        },
        "cross_ref_openneuro": [
            "ds002725",
            "ds003085",
            "ds003720",
            "ds004142",
            "ds004894",
            "ds005700",
            "ds006564",
        ],
        "docs": "ARCHITECTURE.md",
        "github_pages": "https://bladerunner-hue.github.io/Neuroscience/",
    }


def build_features_participants() -> dict[str, Any]:
    rows = _read_csv_records("participants_clean.csv")
    return {"n": len(rows), "records": rows}


def build_features_events() -> dict[str, Any]:
    rows = _read_csv_records("events_summary.csv")
    return {"n": len(rows), "records": rows}


def build_datasets() -> dict[str, Any]:
    """Multi-dataset registry for landscape + explore UI."""
    reg_path = PROCESSED / "dataset_registry.json"
    datasets: Any = {}
    if reg_path.exists():
        datasets = json.loads(reg_path.read_text(encoding="utf-8"))
    else:
        try:
            sys_path_nb = ROOT / "marimo_notebooks"
            if str(sys_path_nb) not in __import__("sys").path:
                __import__("sys").path.insert(0, str(sys_path_nb))
            from multi_dataset_catalog import MULTI_DATASET_CATALOG

            datasets = MULTI_DATASET_CATALOG
        except Exception:
            datasets = {}
    return {
        "n": len(datasets) if isinstance(datasets, dict) else 0,
        "datasets": datasets,
        "data_root": "data/raw/<OpenNeuro-id>/",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_local_files(*, include_raw_niftis: bool = False) -> dict[str, Any]:
    """Index of repo data files for the HTML explorer (sizes only for large NIfTI)."""
    data = ROOT / "data"
    files: list[dict[str, Any]] = []
    if not data.exists():
        return {"n": 0, "files": [], "root": str(ROOT)}
    for sub in ("processed", "raw"):
        base = data / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf in {".nii", ".gz"} and not include_raw_niftis:
                # collapse: count only, skip listing every run (explore uses registry)
                continue
            if p.name in {".git", ".gitattributes"} or ".git" in p.parts:
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                sz = 0
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            files.append(
                {
                    "path": rel,
                    "name": p.name,
                    "layer": sub,
                    "suffix": suf,
                    "bytes": sz,
                    "previewable": suf in {".csv", ".json", ".tsv", ".txt", ".md"}
                    and sz < 12_000_000,
                    "table_key": _table_key_for(rel),
                }
            )
    # raw BIDS summary rows
    raw = data / "raw"
    if raw.exists():
        for ds in sorted(raw.glob("ds*")):
            if not ds.is_dir():
                continue
            n_bold = sum(1 for _ in ds.rglob("*bold.nii.gz"))
            n_sub = sum(1 for _ in ds.glob("sub-*") if _.is_dir())
            files.append(
                {
                    "path": str(ds.relative_to(ROOT)).replace("\\", "/"),
                    "name": ds.name,
                    "layer": "raw",
                    "suffix": "dir",
                    "bytes": 0,
                    "previewable": False,
                    "table_key": None,
                    "n_subjects": n_sub,
                    "n_bold_files": n_bold,
                    "kind": "bids_dataset",
                }
            )
    return {
        "n": len(files),
        "files": files,
        "root": str(ROOT),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _table_key_for(rel: str) -> str | None:
    name = Path(rel).name
    mapping = {
        "spectral_features.csv": "spectral",
        "cleaned_spectral_features.csv": "spectral_clean",
        "subject_features.csv": "subject",
        "condition_features.csv": "condition",
        "run_qc.csv": "qc",
        "participants_clean.csv": "participants",
        "events_summary.csv": "events",
        "ml_bakeoff.json": "bakeoff",
        "tf_results.json": "tf_results",
        "dataset_registry.json": "datasets",
    }
    return mapping.get(name)


def build_table(name: str, *, limit: int | None = None) -> dict[str, Any]:
    """Unified table payload for marimo + HTML explorer."""
    key = name.replace("-", "_").lower().strip()
    builders = {
        "spectral": lambda: build_features_spectral(clean=False),
        "spectral_clean": lambda: build_features_spectral(clean=True),
        "subject": build_features_subject,
        "condition": build_features_condition,
        "qc": build_qc,
        "participants": build_features_participants,
        "events": build_features_events,
        "bakeoff": build_bakeoff,
        "tf_results": build_tf_results,
        "datasets": build_datasets,
    }
    if key not in builders:
        return {"n": 0, "records": [], "error": f"unknown table {name!r}", "key": key}
    data = builders[key]()
    if key in ("bakeoff", "tf_results", "datasets"):
        # not always records-shaped
        if isinstance(data, dict) and "records" not in data:
            return {"n": 1, "records": [data], "key": key, "shape": "object"}
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        records = [data]
    n = len(records)
    if limit is not None and limit > 0:
        records = records[:limit]
    return {
        "n": n,
        "returned": len(records),
        "records": records,
        "key": key,
        "limit": limit,
    }


def build_surfaces() -> dict[str, Any]:
    """Catalog of visualization surfaces (marimo + non-marimo)."""
    notebooks = ROOT / "marimo_notebooks"
    public = [
        ("00_data_browser", "0 · Data browser", "marimo"),
        ("00_qc_dashboard", "0 · QC Dashboard", "marimo"),
        ("01_pre_flight", "I · Cohort & Design", "marimo"),
        ("02_eda_univariate", "II · Spectral Power", "marimo"),
        ("03_eda_multivariate", "III · Algorithm Lab", "marimo"),
        ("04_feature_engineering", "IV · Features", "marimo"),
        ("05_tf_results", "V · Neural Net Results", "marimo"),
        ("09_multi_dataset_analysis", "IX · Multi-dataset scale", "marimo"),
    ]
    local_only = [
        ("00_data_landscape", "0-local · Data Landscape", "marimo-local"),
        ("06_tf_spectrogram_model", "V-local · TF Train", "marimo-local+cli"),
        ("07_spark_god_mode", "VII · Spark God Mode", "marimo-local+cli"),
        ("08_spark_streaming", "VIII · Streaming", "cli+marimo-local"),
    ]
    wasm_stems = {s for s, _, _ in public}
    chapters = []
    for stem, title, kind in public + local_only:
        chapters.append(
            {
                "stem": stem,
                "title": title,
                "kind": kind,
                "exists": (notebooks / f"{stem}.py").exists(),
                "live_url": f"/book/{stem}/" if "marimo" in kind else None,
                "wasm_url": f"/wasm/{stem}/" if stem in wasm_stems else None,
                "viz": "marimo" if "marimo" in kind else "cli",
            }
        )
    return {
        "explorer_url": "/explore/",
        "api_url": "/api/",
        "live_book_url": "/book/",
        "gallery_url": "/",
        "chapters": chapters,
        "non_marimo": [
            {
                "title": "HTML data explorer",
                "url": "/explore/",
                "why": "Browse feature tables without marimo/Pyodide",
            },
            {
                "title": "Static API (JSON)",
                "url": "/api/",
                "why": "Machine-readable tables for any front-end",
            },
            {
                "title": "OpenAPI / Swagger",
                "url": "/docs",
                "why": "Interactive REST when FastAPI is running",
            },
        ],
        "policy": {
            "keep_marimo_py": [
                "00–05 public chapters (reactive viz + WASM export)",
                "00_data_landscape (local multi-set inventory)",
            ],
            "marimo_plus_api": [
                "05_tf_results — viz in marimo; train offline → API/JSON",
                "06_tf_spectrogram_model — train on host; view metrics via /api/tf_results + explore",
            ],
            "cli_or_local_marimo": [
                "07/08 Spark — Catalyst jobs on host; status tables via /api or explore",
            ],
        },
    }


def build_openapi(*, pages_base: str = ".") -> dict[str, Any]:
    """Minimal OpenAPI 3 doc pointing at static JSON (GitHub Pages) paths."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Neuroscience book API (static GitHub Pages mirror)",
            "description": (
                "Frozen snapshot of the FastAPI feature API for OpenNeuro ds000171. "
                "GitHub Pages serves these as static JSON; local `uvicorn app:app` serves live "
                "routes (plus marimo /book/*) without the `.json` suffix."
            ),
            "version": "1.2.0",
        },
        "servers": [
            {"url": pages_base, "description": "This static site (relative)"},
        ],
        "paths": {
            "/health.json": {
                "get": {
                    "summary": "Health + bundle summary",
                    "operationId": "health",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/meta.json": {
                "get": {
                    "summary": "PSD method + cohort counts",
                    "operationId": "meta",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/architecture.json": {
                "get": {
                    "summary": "Stack map + cross-ref datasets",
                    "operationId": "architecture",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/features/spectral.json": {
                "get": {
                    "summary": "Run-level spectral features",
                    "operationId": "spectral",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/features/spectral_clean.json": {
                "get": {
                    "summary": "QC-cleaned spectral features",
                    "operationId": "spectral_clean",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/features/subject.json": {
                "get": {
                    "summary": "Subject-level music-effect table",
                    "operationId": "subject",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/features/condition.json": {
                "get": {
                    "summary": "Trial-type condition features",
                    "operationId": "condition",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/qc.json": {
                "get": {
                    "summary": "Run-level QC flags",
                    "operationId": "qc",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/bakeoff.json": {
                "get": {
                    "summary": "LOOCV sklearn bake-off",
                    "operationId": "bakeoff",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/tf_results.json": {
                "get": {
                    "summary": "Precomputed TensorFlow metrics",
                    "operationId": "tf_results",
                    "responses": {"200": {"description": "OK"}},
                }
            },
        },
    }


API_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>API · Music, Reward &amp; Depression</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600&family=Source+Sans+3:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    body { font-family: "Source Sans 3", system-ui, sans-serif; }
    .serif { font-family: "Cormorant Garamond", Georgia, serif; }
    #swagger-ui .topbar { display: none; }
    #swagger-ui { background: transparent; }
  </style>
</head>
<body class="bg-[#0f1419] text-stone-200 min-h-screen">
  <div class="max-w-5xl mx-auto px-6 py-10">
    <p class="text-emerald-500/90 text-xs tracking-[0.2em] uppercase mb-3">
      GitHub Pages · static FastAPI mirror
    </p>
    <h1 class="serif text-3xl md:text-4xl text-stone-50 mb-2">Public feature API</h1>
    <p class="text-stone-400 text-sm leading-relaxed mb-4 max-w-2xl">
      GitHub Pages cannot run a Python process. These JSON files are a
      <strong class="text-stone-300">frozen mirror</strong> of the local FastAPI app
      (<code class="text-stone-500 text-xs">marimo_exports/fastapi_app.py</code>).
      Same contract, browsable and fetchable from the public site.
    </p>
    <p class="text-xs text-stone-500 mb-8">
      <a class="underline hover:text-emerald-400" href="../">← Book gallery</a>
      · Live local server: <code class="text-stone-600">python marimo_exports/serve.py --fastapi</code>
      · OpenAPI: <a class="underline hover:text-emerald-400" href="openapi.json">openapi.json</a>
    </p>

    <div class="grid md:grid-cols-2 gap-3 mb-10 text-sm" id="links"></div>

    <div class="rounded-2xl border border-stone-800 overflow-hidden bg-stone-900/40 mb-8">
      <div id="swagger-ui"></div>
    </div>

    <footer class="text-xs text-stone-600 border-t border-stone-800 pt-6">
      Endpoints are static snapshots regenerated by
      <code class="text-stone-500">python marimo_exports/static_api.py</code>
      (also run from <code class="text-stone-500">export_wasm.py --sync-docs</code>).
    </footer>
  </div>
  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
  <script>
    const ENDPOINTS = [
      ["health.json", "Health + bundle summary"],
      ["meta.json", "PSD method + cohort counts"],
      ["features/spectral.json", "Run-level spectral features"],
      ["features/spectral_clean.json", "QC-cleaned spectral features"],
      ["features/subject.json", "Subject music-effect table"],
      ["features/condition.json", "Trial-type condition features"],
      ["qc.json", "Run-level QC flags"],
      ["bakeoff.json", "LOOCV sklearn bake-off"],
      ["tf_results.json", "Precomputed TF metrics"],
      ["openapi.json", "OpenAPI 3 schema"],
    ];
    const box = document.getElementById("links");
    ENDPOINTS.forEach(([href, label]) => {
      const a = document.createElement("a");
      a.href = href;
      a.className = "block rounded-xl border border-stone-800 bg-stone-900/60 px-4 py-3 hover:border-emerald-700/50";
      a.innerHTML = `<span class="text-emerald-400/90 text-xs font-mono">${href}</span><br/><span class="text-stone-300">${label}</span>`;
      box.appendChild(a);
    });
    window.ui = SwaggerUIBundle({
      url: "openapi.json",
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis],
      layout: "BaseLayout",
      tryItOutEnabled: true,
      deepLinking: true,
    });
  </script>
</body>
</html>
"""


def export_static_api(out_dir: Path | None = None, *, docs: Path | None = None) -> Path:
    """Write all static API artifacts under out_dir (default docs/api)."""
    out = out_dir or DEFAULT_OUT
    docs = docs or DOCS
    out.mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(parents=True, exist_ok=True)

    payloads: dict[str, Any] = {
        "health.json": build_health(docs),
        "meta.json": build_meta(),
        "architecture.json": build_architecture(),
        "features/spectral.json": build_features_spectral(clean=False),
        "features/spectral_clean.json": build_features_spectral(clean=True),
        "features/subject.json": build_features_subject(),
        "features/condition.json": build_features_condition(),
        "qc.json": build_qc(),
        "bakeoff.json": build_bakeoff(),
        "tf_results.json": build_tf_results(),
        "datasets.json": build_datasets(),
        "surfaces.json": build_surfaces(),
        "local/files.json": build_local_files(),
        "features/participants.json": build_features_participants(),
        "features/events.json": build_features_events(),
        "table/spectral.json": build_table("spectral"),
        "table/subject.json": build_table("subject"),
        "table/condition.json": build_table("condition"),
        "table/qc.json": build_table("qc"),
        "table/participants.json": build_table("participants"),
        "table/tf_results.json": build_table("tf_results"),
        "table/datasets.json": build_table("datasets"),
        "table/bakeoff.json": build_table("bakeoff"),
        "god_run_summary.json": (
            json.loads((PROCESSED / "god_run_summary.json").read_text(encoding="utf-8"))
            if (PROCESSED / "god_run_summary.json").exists()
            else {"n_runs": 0, "records": [], "by_dataset_group": []}
        ),
        "openapi.json": build_openapi(pages_base="."),
    }

    for rel, data in payloads.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    (out / "index.html").write_text(API_INDEX_HTML, encoding="utf-8")
    print(f"  wrote {(out / 'index.html').relative_to(ROOT)}")

    # Directory aliases so /api/health/ works on some static servers
    for name in ("health", "meta", "qc", "bakeoff", "tf_results"):
        alias = out / name
        alias.mkdir(parents=True, exist_ok=True)
        src = out / f"{name}.json"
        if src.exists():
            (alias / "index.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            # tiny HTML that loads JSON for human browsers hitting /api/health/
            (alias / "index.html").write_text(
                f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/><title>{name}</title>
<meta http-equiv="refresh" content="0;url=../{name}.json"/>
</head><body>
<p>Redirecting to <a href="../{name}.json">{name}.json</a>…</p>
<script>location.replace("../{name}.json");</script>
</body></html>
""",
                encoding="utf-8",
            )

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = p.parse_args(argv)
    print(f"Exporting static FastAPI mirror → {args.out}")
    export_static_api(args.out)
    print("Done. Public URL path: /api/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
