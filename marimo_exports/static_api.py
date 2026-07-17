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
            "explorer": "api/",
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
            "live": "uvicorn app:app → /book/* + /api/*",
            "pages": "docs/wasm + docs/api (static)",
        },
        "cross_ref_openneuro": [
            "ds002725",
            "ds003085",
            "ds003720",
            "ds004142",
            "ds005700",
            "ds006564",
        ],
        "docs": "ARCHITECTURE.md",
        "github_pages": "https://bladerunner-hue.github.io/Neuroscience/",
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
