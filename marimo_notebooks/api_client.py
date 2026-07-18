"""Interoperability: marimo (local / WASM) ↔ FastAPI feature & local-file API.

Load order for tables
---------------------
1. Disk under ``data/processed/`` (authoring: ``marimo edit``)
2. HTTP API (live ``uvicorn app:app`` or static ``docs/api/*.json`` on Pages)
3. Embedded ``book_data.BOOK_BUNDLE`` / helpers (WASM offline)

Environment
-----------
- ``NEURO_API_BASE`` — e.g. ``http://127.0.0.1:8000`` so WASM in the browser
  can reach a local FastAPI and see the same tables as the host.
- On GitHub Pages, base is inferred as relative ``../../api`` from a chapter.

Heavy work (TensorFlow train, Spark jobs, NIfTI) stays on the host via FastAPI
or CLI — never inside Pyodide.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def is_wasm() -> bool:
    if sys.platform == "emscripten":
        return True
    if "pyodide" in sys.modules:
        return True
    return bool(os.environ.get("MARIMO_WASM") or os.environ.get("PYODIDE"))


def is_local_disk() -> bool:
    """True when we can read the repo feature store from the filesystem."""
    if is_wasm():
        return False
    for root in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent.parent):
        if (root / "data" / "processed" / "spectral_features.csv").exists():
            return True
        if (root / "data" / "processed" / "book_bundle.json").exists():
            return True
    return False


def api_base() -> Optional[str]:
    """Base URL for REST/static JSON, without trailing slash."""
    env = (os.environ.get("NEURO_API_BASE") or os.environ.get("NEURO_API_URL") or "").strip()
    if env:
        return env.rstrip("/")
    if is_wasm():
        # From docs/wasm/<chapter>/ → docs/api/
        return "../../api"
    # Local authoring: prefer live FastAPI if user set nothing — try common port
    # only when explicitly enabled to avoid hanging offline notebooks
    if os.environ.get("NEURO_PREFER_API") == "1":
        return "http://127.0.0.1:8000"
    return None


def surface_label() -> str:
    if is_wasm():
        return "wasm"
    if is_local_disk():
        return "local-disk"
    if api_base():
        return "api"
    return "bundle-fallback"


# ---------------------------------------------------------------------------
# HTTP (works in CPython; WASM uses urllib if available)
# ---------------------------------------------------------------------------


def _http_get_json(url: str, *, timeout: float = 20.0) -> Any:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — controlled base
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def fetch_api(path: str, *, base: Optional[str] = None) -> Any:
    """GET JSON from live API or static Pages mirror.

    ``path`` examples: ``/api/health``, ``features/spectral.json``, ``datasets``.
    """
    b = (base if base is not None else api_base()) or ""
    if not b:
        raise RuntimeError("No API base configured (set NEURO_API_BASE)")

    p = path.lstrip("/")
    # Live FastAPI: /api/...
    # Static Pages:  /api/....json  (we accept either)
    if b.endswith("/api") or b.endswith("/api/"):
        url = urljoin(b.rstrip("/") + "/", p)
    elif "/api" in b:
        url = urljoin(b.rstrip("/") + "/", p)
    else:
        # base is origin only
        if not p.startswith("api/") and not p.startswith("/api"):
            p = "api/" + p
        url = urljoin(b.rstrip("/") + "/", p.lstrip("/"))

    # Prefer .json on static-style bases when path has no extension
    try:
        return _http_get_json(url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        if not url.endswith(".json") and "features/" in url or url.rstrip("/").endswith(
            ("health", "meta", "qc", "bakeoff", "tf_results", "datasets", "architecture")
        ):
            try:
                return _http_get_json(url.rstrip("/") + ".json")
            except Exception:
                pass
        # Static tree uses nested paths
        alt = url
        if "/api/" in url and not url.endswith(".json"):
            # /api/features/spectral → /api/features/spectral.json
            alt = url + ".json"
            try:
                return _http_get_json(alt)
            except Exception:
                pass
        raise


# ---------------------------------------------------------------------------
# High-level table loaders (records list)
# ---------------------------------------------------------------------------


def _disk_processed() -> Optional[Path]:
    for root in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent.parent):
        p = root / "data" / "processed"
        if p.is_dir():
            return p
    return None


def _csv_records(name: str) -> list[dict[str, Any]]:
    proc = _disk_processed()
    if not proc:
        return []
    path = proc / name
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_csv(path)
    for c in ("psd_f", "psd_pxx", "peri_stim"):
        if c in df.columns:
            df = df.drop(columns=[c])
    return json.loads(df.to_json(orient="records"))


def load_table(
    name: str,
    *,
    clean: bool = False,
    prefer_api: bool = False,
) -> dict[str, Any]:
    """Return ``{n, records, source}`` for a named feature table.

    Names: spectral | spectral_clean | subject | condition | qc | participants
    """
    key = name.replace("-", "_").lower()
    disk_map = {
        "spectral": "spectral_features.csv",
        "spectral_clean": "cleaned_spectral_features.csv",
        "subject": "subject_features.csv",
        "condition": "condition_features.csv",
        "qc": "run_qc.csv",
        "participants": "participants_clean.csv",
        "events": "events_summary.csv",
    }

    # 1) Disk
    if is_local_disk() and not prefer_api:
        if key == "spectral" and clean:
            key = "spectral_clean"
        fname = disk_map.get(key)
        if fname:
            rows = _csv_records(fname)
            if rows or ( _disk_processed() and (_disk_processed() / fname).exists()):
                return {"n": len(rows), "records": rows, "source": f"disk:{fname}"}

    # 2) API / static JSON
    api_paths = {
        "spectral": "features/spectral",
        "spectral_clean": "features/spectral_clean",
        "subject": "features/subject",
        "condition": "features/condition",
        "qc": "qc",
        "participants": "features/participants",
        "events": "features/events",
    }
    if key == "spectral" and clean:
        key = "spectral_clean"
    if api_base() or prefer_api:
        try:
            path = api_paths.get(key, key)
            if key == "spectral" and not clean:
                data = fetch_api(f"features/spectral")
            elif key == "spectral_clean":
                try:
                    data = fetch_api("features/spectral_clean")
                except Exception:
                    data = fetch_api("features/spectral?clean=true")
            else:
                data = fetch_api(path)
            if isinstance(data, dict) and "records" in data:
                data = {**data, "source": data.get("source") or f"api:{path}"}
                return data
            if isinstance(data, list):
                return {"n": len(data), "records": data, "source": f"api:{path}"}
        except Exception as exc:
            api_err = str(exc)
    else:
        api_err = "no api base"

    # 3) Bundle fallback
    try:
        from helpers import load_book_bundle

        b = load_book_bundle()
        bundle_key = {
            "spectral": "spectral_features",
            "spectral_clean": "cleaned_spectral_features",
            "subject": "subject_features",
            "condition": "condition_features",
            "qc": "run_qc",
            "participants": "participants",
            "events": "events_summary",
        }.get(key)
        rows = b.get(bundle_key) or []
        if key == "spectral_clean" and not rows:
            rows = [
                r
                for r in (b.get("spectral_features") or [])
                if not r.get("qc_outlier")
            ]
        return {
            "n": len(rows),
            "records": rows,
            "source": f"bundle:{bundle_key}",
            "api_error": api_err,
        }
    except Exception as exc:
        return {"n": 0, "records": [], "source": "empty", "error": str(exc), "api_error": api_err}


def load_datasets_registry() -> dict[str, Any]:
    """Multi-dataset registry for landscape / WASM."""
    if is_local_disk():
        proc = _disk_processed()
        if proc and (proc / "dataset_registry.json").exists():
            return {
                "source": "disk",
                "datasets": json.loads((proc / "dataset_registry.json").read_text()),
            }
    # WASM embed from export_wasm
    try:
        import book_data as bd

        if hasattr(bd, "DATASETS_REGISTRY") and bd.DATASETS_REGISTRY:
            return {"source": "wasm-embed", "datasets": bd.DATASETS_REGISTRY}
    except Exception:
        pass
    if api_base():
        try:
            data = fetch_api("datasets")
            if isinstance(data, dict):
                return {**data, "source": data.get("source") or "api"}
        except Exception:
            pass
    try:
        from multi_dataset_catalog import MULTI_DATASET_CATALOG

        return {"source": "catalog-static", "datasets": MULTI_DATASET_CATALOG}
    except Exception:
        return {"source": "empty", "datasets": {}}


def load_god_run_summary() -> dict[str, Any]:
    """Multi-set run summary for WASM (embedded) or disk/API."""
    if is_local_disk():
        proc = _disk_processed()
        if proc and (proc / "god_run_summary.json").exists():
            return {
                "source": "disk",
                **json.loads((proc / "god_run_summary.json").read_text()),
            }
    try:
        import book_data as bd

        if hasattr(bd, "GOD_RUN_SUMMARY") and bd.GOD_RUN_SUMMARY:
            return {"source": "wasm-embed", **bd.GOD_RUN_SUMMARY}
    except Exception:
        pass
    if api_base():
        try:
            data = fetch_api("god_run_summary")
            if isinstance(data, dict):
                return {"source": "api", **data}
        except Exception:
            pass
    return {"source": "empty", "n_runs": 0, "records": [], "by_dataset_group": []}


def load_local_files_index() -> dict[str, Any]:
    """List processed (and raw summary) files — from disk or API."""
    if is_local_disk():
        return _scan_local_files()
    if api_base():
        try:
            data = fetch_api("local/files")
            if isinstance(data, dict):
                return {**data, "source": data.get("source") or "api"}
        except Exception as exc:
            return {"source": "error", "error": str(exc), "files": []}
    return {
        "source": "wasm-static",
        "note": "Set NEURO_API_BASE=http://127.0.0.1:8000 to browse host files from WASM",
        "files": [],
    }


def _scan_local_files() -> dict[str, Any]:
    root = None
    for cand in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent.parent):
        if (cand / "data").is_dir():
            root = cand
            break
    if root is None:
        return {"source": "disk", "files": [], "root": None}

    files: list[dict[str, Any]] = []
    data = root / "data"
    for sub in ("processed", "raw"):
        base = data / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            # skip huge nifti in listing body — still show path + size
            rel = str(p.relative_to(root)).replace("\\", "/")
            try:
                sz = p.stat().st_size
            except OSError:
                sz = 0
            files.append(
                {
                    "path": rel,
                    "name": p.name,
                    "layer": sub,
                    "suffix": p.suffix.lower(),
                    "bytes": sz,
                    "previewable": p.suffix.lower()
                    in {".csv", ".json", ".tsv", ".txt", ".md"}
                    and sz < 8_000_000,
                }
            )
    return {
        "source": "disk",
        "root": str(root),
        "n": len(files),
        "files": files,
    }


def records_to_polars(records: list[dict[str, Any]]):
    """Small helper for marimo tables (Polars preferred)."""
    try:
        import polars as pl

        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records)
    except Exception:
        import pandas as pd

        return pd.DataFrame(records)


def connectivity_banner_md() -> str:
    """Short markdown for notebooks showing how data is reached."""
    return (
        f"**Data surface:** `{surface_label()}` · "
        f"API base: `{api_base() or '—'}` · "
        f"disk: `{is_local_disk()}` · wasm: `{is_wasm()}`\n\n"
        "Heavy jobs (TF train, Spark, NIfTI) run on the **host FastAPI/CLI**, not in WASM. "
        "Point WASM at a live server with "
        "`NEURO_API_BASE=http://127.0.0.1:8000` to browse local tables."
    )
