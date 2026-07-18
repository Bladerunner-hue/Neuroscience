#!/usr/bin/env python3
"""
Build the GitHub Pages book from canonical marimo_notebooks/.

Pipeline:
  marimo_notebooks/*.py  (+ helpers.py + spectral_methods.py)
       →  inject helpers + spectral_methods + book_bundle JSON (base64)
       →  marimo export html-wasm
       →  marimo_exports/wasm/<chapter>/
       →  docs/wasm/<chapter>/   (--sync-docs)
       →  GitHub Pages (CI)

Why base64? helpers.py starts with a triple-quoted docstring. Embedding it in a
Python \"\"\" string truncates early and produces a blank WASM page. Base64 avoids
all quote/indent issues.

Usage (repo root):
  python marimo_exports/export_wasm.py --sync-docs
  python marimo_exports/export_wasm.py --sync-docs --verify
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "marimo_notebooks"
HELPERS = NOTEBOOK_DIR / "helpers.py"
SPECTRAL = NOTEBOOK_DIR / "spectral_methods.py"
BOOK_DATA = NOTEBOOK_DIR / "book_data.py"
API_CLIENT = NOTEBOOK_DIR / "api_client.py"
MULTI_CATALOG = NOTEBOOK_DIR / "multi_dataset_catalog.py"
EXPORT_DIR = ROOT / "marimo_exports" / "wasm"
DOCS_DIR = ROOT / "docs"
DOCS_WASM = DOCS_DIR / "wasm"

# WASM book chapters (TF trains offline; 05 shows precomputed results only)
# Multi-set tables come from static /api + embedded book_bundle / god_run_summary.
CANDIDATES = [
    "00_data_browser.py",  # shared tables · same /api/table as HTML explore
    "00_qc_dashboard.py",
    "01_pre_flight.py",
    "02_eda_univariate.py",
    "03_eda_multivariate.py",
    "04_feature_engineering.py",
    "05_tf_results.py",  # precomputed TF/NN page — no tensorflow package in browser
    "09_multi_dataset_analysis.py",  # multi-set god summary + TF metrics (no Spark JVM in browser)
]


def _b64_chunks(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    b64 = base64.b64encode(data).decode("ascii")
    chunks = [b64[i : i + 80] for i in range(0, len(b64), 80)]
    return "(\n" + "".join(f'        "{c}"\n' for c in chunks) + "    )"


def _load_book_bundle_json() -> str | None:
    bundle_json_path = ROOT / "data" / "processed" / "book_bundle.json"
    if bundle_json_path.exists():
        return bundle_json_path.read_text(encoding="utf-8")
    if BOOK_DATA.exists():
        ns: dict = {}
        exec(
            compile(BOOK_DATA.read_text(encoding="utf-8"), "book_data.py", "exec"),
            ns,
        )
        return json.dumps(ns["BOOK_BUNDLE"], separators=(",", ":"))
    return None


def _inject_modules(
    notebook_src: str,
    *,
    helpers_src: str,
    book_bundle_json: str | None,
    spectral_src: str | None = None,
    api_client_src: str | None = None,
    multi_catalog_src: str | None = None,
    god_summary_json: str | None = None,
    datasets_json: str | None = None,
) -> str:
    """Register helpers + spectral_methods + book_data (+ API helpers) for Pyodide.

    helpers/spectral_methods/api_client/catalog: exec Python source.
    book_data / god_summary / datasets: JSON via base64 + json.loads.
    """
    helpers_lit = _b64_chunks(helpers_src)
    loads = f'''
    if "helpers" not in _sys.modules:
        _helpers_b64 = {helpers_lit}
        _helpers_src = _b64.b64decode("".join(_helpers_b64)).decode("utf-8")
        _helpers_mod = _types.ModuleType("helpers")
        exec(compile(_helpers_src, "helpers.py", "exec"), _helpers_mod.__dict__)
        _sys.modules["helpers"] = _helpers_mod
'''
    if spectral_src:
        spectral_lit = _b64_chunks(spectral_src)
        loads += f'''
    if "spectral_methods" not in _sys.modules:
        _spectral_b64 = {spectral_lit}
        _spectral_src = _b64.b64decode("".join(_spectral_b64)).decode("utf-8")
        _spectral_mod = _types.ModuleType("spectral_methods")
        exec(compile(_spectral_src, "spectral_methods.py", "exec"), _spectral_mod.__dict__)
        _sys.modules["spectral_methods"] = _spectral_mod
'''
    if api_client_src:
        api_lit = _b64_chunks(api_client_src)
        loads += f'''
    if "api_client" not in _sys.modules:
        _api_b64 = {api_lit}
        _api_src = _b64.b64decode("".join(_api_b64)).decode("utf-8")
        _api_mod = _types.ModuleType("api_client")
        exec(compile(_api_src, "api_client.py", "exec"), _api_mod.__dict__)
        _sys.modules["api_client"] = _api_mod
'''
    if multi_catalog_src:
        cat_lit = _b64_chunks(multi_catalog_src)
        loads += f'''
    if "multi_dataset_catalog" not in _sys.modules:
        _cat_b64 = {cat_lit}
        _cat_src = _b64.b64decode("".join(_cat_b64)).decode("utf-8")
        _cat_mod = _types.ModuleType("multi_dataset_catalog")
        exec(compile(_cat_src, "multi_dataset_catalog.py", "exec"), _cat_mod.__dict__)
        _sys.modules["multi_dataset_catalog"] = _cat_mod
'''
    if book_bundle_json:
        bundle_lit = _b64_chunks(book_bundle_json.encode("utf-8"))
        loads += f'''
    if "book_data" not in _sys.modules:
        import json as _json
        _book_json_b64 = {bundle_lit}
        _book_json = _b64.b64decode("".join(_book_json_b64)).decode("utf-8")
        _book_mod = _types.ModuleType("book_data")
        _book_mod.BOOK_BUNDLE = _json.loads(_book_json)
        _sys.modules["book_data"] = _book_mod
'''
    # Attach multi-set summaries onto book_data if present
    extras = []
    if god_summary_json:
        extras.append(("GOD_RUN_SUMMARY", god_summary_json))
    if datasets_json:
        extras.append(("DATASETS_REGISTRY", datasets_json))
    if extras and book_bundle_json:
        for attr, raw in extras:
            lit = _b64_chunks(raw.encode("utf-8") if isinstance(raw, str) else raw)
            loads += f'''
    if "book_data" in _sys.modules:
        import json as _json2
        _{attr}_b64 = {lit}
        setattr(_sys.modules["book_data"], "{attr}", _json2.loads(_b64.b64decode("".join(_{attr}_b64)).decode("utf-8")))
'''
    elif extras:
        # create book_data shell
        loads += '''
    if "book_data" not in _sys.modules:
        import json as _json
        _book_mod = _types.ModuleType("book_data")
        _book_mod.BOOK_BUNDLE = {}
        _sys.modules["book_data"] = _book_mod
'''
        for attr, raw in extras:
            lit = _b64_chunks(raw.encode("utf-8") if isinstance(raw, str) else raw)
            loads += f'''
    if "book_data" in _sys.modules:
        import json as _json2
        _{attr}_b64 = {lit}
        setattr(_sys.modules["book_data"], "{attr}", _json2.loads(_b64.b64decode("".join(_{attr}_b64)).decode("utf-8")))
'''

    bootstrap = (
        '''
    # --- injected by export_wasm.py: local modules for WASM/Pyodide (base64) ---
    import base64 as _b64
    import sys as _sys
    import types as _types
'''
        + loads
        + "    # --- end module injection ---\n"
    )
    m = re.search(
        r"(@app\.cell(?:\([^)]*\))?\s*\n"
        r"def\s+\w+\s*\([^)]*\)\s*:\s*\n)",
        notebook_src,
    )
    if not m:
        raise RuntimeError("Could not find first @app.cell to inject modules")
    return notebook_src[: m.end()] + bootstrap + notebook_src[m.end() :]


def _validate_packed(packed: str, path: Path) -> None:
    """Fail fast if injection produces invalid Python."""
    try:
        ast.parse(packed)
    except SyntaxError as e:
        raise RuntimeError(f"Injected notebook is not valid Python: {path}: {e}") from e


def _simulate_wasm_modules(
    helpers_src: str,
    spectral_src: str | None,
    book_bundle_json: str | None,
) -> dict:
    """Execute the same base64→sys.modules path Pyodide uses; return load stats."""
    # Isolate from any pre-imported workspace modules
    saved = {
        k: sys.modules.pop(k)
        for k in ("helpers", "spectral_methods", "book_data")
        if k in sys.modules
    }
    stats: dict = {"ok": True, "errors": []}
    try:
        helpers_mod = types.ModuleType("helpers")
        exec(compile(helpers_src, "helpers.py", "exec"), helpers_mod.__dict__)
        sys.modules["helpers"] = helpers_mod

        if spectral_src:
            spectral_mod = types.ModuleType("spectral_methods")
            exec(compile(spectral_src, "spectral_methods.py", "exec"), spectral_mod.__dict__)
            sys.modules["spectral_methods"] = spectral_mod
            stats["spectral"] = True
        else:
            stats["spectral"] = False

        if book_bundle_json:
            book_mod = types.ModuleType("book_data")
            book_mod.BOOK_BUNDLE = json.loads(book_bundle_json)
            sys.modules["book_data"] = book_mod
            b = book_mod.BOOK_BUNDLE
            stats["bundle_keys"] = sorted(b.keys())
            stats["psd_method"] = b.get("psd_method")
            stats["n_spectral"] = len(b.get("spectral_features") or [])
            stats["n_cleaned"] = len(b.get("cleaned_spectral_features") or [])
            stats["has_psd_arrays"] = bool(
                (b.get("spectral_features") or [{}])[0].get("psd_f")
                if b.get("spectral_features")
                else False
            )
            stats["has_ml_bakeoff"] = bool(b.get("ml_bakeoff"))
            stats["has_tf_results"] = bool(b.get("tf_results"))
            stats["n_ts_examples"] = len(b.get("timeseries_examples") or {})

        # Call loaders as notebooks do (no disk → pure bundle)
        # Temporarily hide processed dir by monkeypatching _find_processed
        helpers = sys.modules["helpers"]
        orig_find = helpers._find_processed
        helpers._find_processed = lambda: None  # type: ignore
        try:
            sf = helpers.load_spectral_features()
            cl = helpers.load_cleaned_spectral_features()
            bake = helpers.load_ml_bakeoff()
            tf = helpers.load_tf_results()
            stats["loader_spectral"] = int(len(sf))
            stats["loader_cleaned"] = int(len(cl))
            stats["loader_bakeoff"] = bool(bake)
            stats["loader_tf"] = bool(tf)
            if "psd_f" in sf.columns and len(sf):
                sample = sf["psd_f"].iloc[0]
                stats["psd_f_type"] = type(sample).__name__
                if isinstance(sample, str):
                    arr = json.loads(sample)
                    stats["psd_len"] = len(arr)
            if spectral_src:
                import numpy as np

                sm = sys.modules["spectral_methods"]
                ts = np.random.default_rng(0).normal(0, 1, 64)
                f, p = sm.multitaper_psd(ts, fs=1 / 3.0)
                stats["multitaper_bins"] = int(len(p))
        finally:
            helpers._find_processed = orig_find  # type: ignore
    except Exception as e:
        stats["ok"] = False
        stats["errors"].append(str(e))
    finally:
        for k in ("helpers", "spectral_methods", "book_data"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return stats


def export_one(notebook: Path, mode: str, tmp_dir: Path) -> bool:
    out_dir = EXPORT_DIR / notebook.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    helpers_src = HELPERS.read_text(encoding="utf-8")
    spectral_src = SPECTRAL.read_text(encoding="utf-8") if SPECTRAL.exists() else None
    api_src = API_CLIENT.read_text(encoding="utf-8") if API_CLIENT.exists() else None
    cat_src = MULTI_CATALOG.read_text(encoding="utf-8") if MULTI_CATALOG.exists() else None
    book_bundle_json = _load_book_bundle_json()
    god_path = ROOT / "data" / "processed" / "god_run_summary.json"
    god_summary_json = god_path.read_text(encoding="utf-8") if god_path.exists() else None
    reg_path = ROOT / "data" / "processed" / "dataset_registry.json"
    datasets_json = reg_path.read_text(encoding="utf-8") if reg_path.exists() else None

    nb_src = notebook.read_text(encoding="utf-8")
    packed = _inject_modules(
        nb_src,
        helpers_src=helpers_src,
        book_bundle_json=book_bundle_json,
        spectral_src=spectral_src,
        api_client_src=api_src,
        multi_catalog_src=cat_src,
        god_summary_json=god_summary_json,
        datasets_json=datasets_json,
    )
    _validate_packed(packed, notebook)
    if re.search(r'BOOK_BUNDLE\s*=\s*\{[^}]*\btrue\b', packed):
        raise RuntimeError(
            f"Refusing to export {notebook.name}: BOOK_BUNDLE still has JSON true/false"
        )

    # Require spectral_methods inject for QC chapter (and always when file exists)
    if spectral_src and "spectral_methods" not in packed:
        print(f"   ❌ spectral_methods missing from packed {notebook.name}")
        return False
    if book_bundle_json and "_book_json_b64" not in packed:
        print(f"   ❌ book_bundle JSON missing from packed {notebook.name}")
        return False
    if api_src and "api_client" not in packed:
        print(f"   ⚠️  api_client not in packed {notebook.name} (optional for older cells)")

    tmp_nb = tmp_dir / notebook.name
    tmp_nb.write_text(packed, encoding="utf-8")

    cmd = [
        "marimo",
        "export",
        "html-wasm",
        str(tmp_nb),
        "-o",
        str(out_dir),
        "--mode",
        mode,
        "--force",
        "--no-show-code",
    ]
    print(f"→ Exporting {notebook.name} ({mode}) …")
    try:
        subprocess.check_call(cmd, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        print(f"   ❌ marimo export failed: {e}")
        return False

    index = out_dir / "index.html"
    if not index.exists():
        print(f"   ❌ missing {index}")
        return False

    html = index.read_text(encoding="utf-8", errors="replace")
    checks = [
        ("injected by export_wasm" in html, "module injection marker"),
        ("b64decode" in html, "base64 module loader"),
        ("_helpers_b64" in html or "helpers" in html, "helpers injection"),
        (
            spectral_src is None or "_spectral_b64" in html or "spectral_methods" in html,
            "spectral_methods injection",
        ),
        (
            book_bundle_json is None
            or "_book_json_b64" in html
            or "book_data" in html,
            "book_data JSON injection",
        ),
        ('exec(compile("""' not in html, "no unsafe triple-quote exec"),
    ]
    for ok, label in checks:
        if not ok:
            print(f"   ❌ export missing: {label}")
            return False
    if re.search(r"has_bold.{0,5}true", html) and "json.loads" not in html:
        print("   ❌ suspicious JSON true in export without json.loads path")
        return False

    # marimo 0.23 html-wasm embeds auto_instantiate=false → blank Pages until Run.
    patched = html.replace('"auto_instantiate": false', '"auto_instantiate": true')
    patched = patched.replace('"show_tracebacks": false', '"show_tracebacks": true')
    if patched != html:
        index.write_text(patched, encoding="utf-8")
        print("   ↺ patched auto_instantiate=true, show_tracebacks=true")
    elif mode == "run" and '"auto_instantiate": true' not in patched:
        print("   ⚠️  could not force auto_instantiate=true")

    (out_dir / ".nojekyll").touch(exist_ok=True)
    for junk in ("CLAUDE.md",):
        p = out_dir / junk
        if p.exists():
            p.unlink()

    # Explicit base for nested GH Pages paths (helps relative dynamic imports)
    html2 = index.read_text(encoding="utf-8", errors="replace")
    if "<base " not in html2.lower():
        html2 = html2.replace("<head>", '<head>\n    <base href="./" />', 1)
        index.write_text(html2, encoding="utf-8")

    # Verify every dynamic import from the entry chunk exists on disk
    if not _verify_chapter_assets(out_dir):
        return False

    print(f"   ✅ {index} ({index.stat().st_size} bytes)")
    return True


def _verify_chapter_assets(chapter_dir: Path, assets_dir: Path | None = None) -> bool:
    """Ensure index.html entry + its dynamic import() targets exist."""
    index = chapter_dir / "index.html"
    if not index.exists():
        print(f"   ❌ missing {index}")
        return False
    html = index.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"""src=["']([^"']*assets/index-[^"']+\.js)["']""", html)
    if not m:
        print(f"   ❌ {chapter_dir.name}: no assets/index-*.js entry in index.html")
        return False
    entry_rel = m.group(1)
    # Resolve relative to chapter_dir (./assets/... or ../shared-assets/...)
    entry = (chapter_dir / entry_rel).resolve()
    if assets_dir is None:
        assets_dir = entry.parent
    if not entry.exists():
        print(f"   ❌ {chapter_dir.name}: missing entry {entry}")
        return False
    js = entry.read_text(encoding="utf-8", errors="replace")
    dyn = re.findall(r"""import\(\s*["']\./([^"']+)["']\s*\)""", js)
    missing = [d for d in dyn if not (assets_dir / d).exists()]
    if missing:
        print(f"   ❌ {chapter_dir.name}: missing dynamic chunks: {missing[:8]}")
        return False
    # run-page is required for public "run" mode
    run_pages = list(assets_dir.glob("run-page-*.js"))
    if not run_pages:
        print(f"   ❌ {chapter_dir.name}: no run-page-*.js in {assets_dir}")
        return False
    return True


def share_wasm_assets(root: Path) -> bool:
    """Collapse per-chapter marimo frontend into one shared-assets/ folder.

    Each chapter duplicates ~680 hashed Vite chunks (~29MB). On GitHub Pages that
    multiplies deploy surface and can leave index.html pointing at a chunk hash
    that never landed → "Failed to fetch dynamically imported module".

    Dynamic imports resolve relative to the *script URL*, so pointing every
    chapter at ``../shared-assets/index-*.js`` keeps chunks coherent.
    """
    chapters = [
        root / Path(name).stem
        for name in CANDIDATES
        if (root / Path(name).stem / "index.html").exists()
    ]
    if not chapters:
        print("   ⚠️  no chapters to share assets for")
        return False

    # Prefer a chapter that still has a local assets/ dir
    donor = next((c for c in chapters if (c / "assets").is_dir()), None)
    if donor is None:
        # Already shared?
        shared = root / "shared-assets"
        if shared.is_dir() and any(shared.glob("run-page-*.js")):
            print(f"   ✅ shared-assets already present ({shared})")
            return True
        print("   ❌ no chapter assets/ to promote to shared-assets")
        return False

    shared = root / "shared-assets"
    if shared.exists():
        shutil.rmtree(shared)
    shutil.copytree(donor / "assets", shared)
    print(f"   ↺ shared-assets ← {donor.name}/assets ({len(list(shared.iterdir()))} files)")

    ok = True
    for ch in chapters:
        index = ch / "index.html"
        html = index.read_text(encoding="utf-8", errors="replace")
        # Rewrite asset URLs to the shared pool (one level up from chapter/)
        new_html = html.replace("./assets/", "../shared-assets/")
        new_html = new_html.replace('"/assets/', '"../shared-assets/')
        new_html = new_html.replace("'/assets/", "'../shared-assets/")
        # base stays ./ so relative navigation within the chapter works
        if new_html == html and "shared-assets" not in html:
            print(f"   ⚠️  {ch.name}: no ./assets/ paths rewritten")
        index.write_text(new_html, encoding="utf-8")
        assets = ch / "assets"
        if assets.is_dir():
            shutil.rmtree(assets)
        if not _verify_chapter_assets(ch, assets_dir=shared):
            ok = False
        else:
            print(f"   ✅ {ch.name} → ../shared-assets/")
    (shared / ".nojekyll").touch(exist_ok=True)
    return ok


def sync_docs() -> None:
    """Copy WASM chapters + shared-assets + static API into docs/ for GitHub Pages.

    Chapter HTML uses ``../shared-assets/…`` relative to ``docs/wasm/<chapter>/``,
    which resolves to **``docs/shared-assets/``** (not ``docs/wasm/shared-assets/``).
    """
    DOCS_WASM.mkdir(parents=True, exist_ok=True)
    if DOCS_WASM.exists():
        for child in list(DOCS_WASM.iterdir()):
            if child.is_dir() and not (EXPORT_DIR / child.name).exists():
                if child.name == "shared-assets":
                    # always refresh from export below
                    shutil.rmtree(child)
                    continue
                shutil.rmtree(child)
                print(f"   removed stale docs/wasm/{child.name}")

    export_shared = EXPORT_DIR / "shared-assets"
    for child in EXPORT_DIR.iterdir():
        if not child.is_dir():
            continue
        if child.name == "shared-assets":
            # Place at docs/shared-assets so ../shared-assets from wasm/* resolves
            dest = DOCS_DIR / "shared-assets"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(child, dest)
            print(f"   synced → docs/shared-assets ({len(list(dest.iterdir()))} files)")
            # Also keep a copy under docs/wasm/ for older links / CI probes
            dest_wasm = DOCS_WASM / "shared-assets"
            if dest_wasm.exists():
                shutil.rmtree(dest_wasm)
            shutil.copytree(child, dest_wasm)
            print("   synced → docs/wasm/shared-assets (mirror)")
            continue
        dest = DOCS_WASM / child.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(child, dest)
        print(f"   synced → docs/wasm/{child.name}")
    (DOCS_DIR / ".nojekyll").touch(exist_ok=True)

    # Freeze FastAPI payloads as static JSON for GitHub Pages (no Python runtime)
    print("Exporting static FastAPI mirror → docs/api/ …")
    try:
        from marimo_exports.static_api import export_static_api
    except ImportError:
        sys.path.insert(0, str(ROOT))
        from marimo_exports.static_api import export_static_api  # type: ignore

    export_static_api(DOCS_DIR / "api", docs=DOCS_DIR)
    print("   synced → docs/api/ (GitHub Pages FastAPI mirror)")

    explore = DOCS_DIR / "explore" / "index.html"
    if explore.exists():
        print(f"   ✅ docs/explore/index.html present ({explore.stat().st_size} B)")
    else:
        print("   ⚠️  docs/explore/index.html missing — non-marimo explore UI won't ship on Pages")


def verify_exports(docs: bool = True) -> bool:
    """Post-export integration checks on HTML + simulated module loaders."""
    ok = True
    helpers_src = HELPERS.read_text(encoding="utf-8")
    spectral_src = SPECTRAL.read_text(encoding="utf-8") if SPECTRAL.exists() else None
    book_bundle_json = _load_book_bundle_json()
    print("\n=== WASM module simulation (disk-hidden loaders) ===")
    stats = _simulate_wasm_modules(helpers_src, spectral_src, book_bundle_json)
    for k, v in stats.items():
        if k != "errors":
            print(f"  {k}: {v}")
    if not stats.get("ok"):
        print("  ERRORS:", stats.get("errors"))
        ok = False
    if stats.get("loader_spectral", 0) < 1:
        print("  ❌ spectral features empty under WASM loaders")
        ok = False
    if stats.get("loader_cleaned", 0) < 1:
        print("  ❌ cleaned features empty under WASM loaders")
        ok = False
    if not stats.get("has_psd_arrays"):
        print("  ⚠️  bundle spectral_features lack psd_f (Ch II plots need them)")
        # soft fail if we at least have psd_examples
        if book_bundle_json:
            b = json.loads(book_bundle_json)
            if not b.get("psd_examples"):
                ok = False
    if not stats.get("spectral"):
        print("  ❌ spectral_methods not loaded")
        ok = False

    base = DOCS_WASM if docs else EXPORT_DIR
    print(f"\n=== HTML export checks under {base} ===")
    # Pages resolves ../shared-assets from docs/wasm/<ch>/ → docs/shared-assets
    if docs:
        shared = DOCS_DIR / "shared-assets"
        if not shared.is_dir():
            shared = DOCS_WASM / "shared-assets"
    else:
        shared = base / "shared-assets"
    if not shared.is_dir() or not any(shared.glob("run-page-*.js")):
        print(f"  ❌ missing shared-assets with run-page-*.js at {shared}")
        ok = False
    else:
        print(f"  ✅ shared-assets ({len(list(shared.iterdir()))} files) at {shared}")

    for name in CANDIDATES:
        stem = Path(name).stem
        idx = base / stem / "index.html"
        if not idx.exists():
            print(f"  ❌ missing {idx}")
            ok = False
            continue
        html = idx.read_text(encoding="utf-8", errors="replace")
        need = [
            ("injected by export_wasm", "injection"),
            ("b64decode", "b64"),
            ("_helpers_b64", "helpers"),
            ("_spectral_b64", "spectral"),
            ("_book_json_b64", "book_json"),
            ('"auto_instantiate": true', "auto_instantiate"),
            ("shared-assets", "shared-assets path"),
        ]
        missing = [lab for tok, lab in need if tok not in html]
        size = idx.stat().st_size
        if missing:
            print(f"  ❌ {stem}: missing {missing} ({size} B)")
            ok = False
        elif not _verify_chapter_assets(base / stem, assets_dir=shared):
            print(f"  ❌ {stem}: asset graph broken")
            ok = False
        else:
            print(f"  ✅ {stem} ({size} B)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", action="store_true", help="Editable WASM (default: run)")
    parser.add_argument(
        "--sync-docs",
        action="store_true",
        help="Copy wasm exports into docs/ for GitHub Pages",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run integration checks after export (also on --sync-docs)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run integration checks (no export)",
    )
    args = parser.parse_args()
    mode = "edit" if args.edit else "run"

    if args.verify_only:
        return 0 if verify_exports(docs=DOCS_WASM.exists()) else 1

    if not HELPERS.exists():
        print(f"Missing canonical helpers: {HELPERS}", file=sys.stderr)
        return 1
    if not SPECTRAL.exists():
        print(f"Missing spectral_methods: {SPECTRAL}", file=sys.stderr)
        return 1

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Canonical notebooks: {NOTEBOOK_DIR}")
    print(f"Export dir: {EXPORT_DIR} (mode={mode})")
    bj = _load_book_bundle_json()
    if bj:
        b = json.loads(bj)
        print(
            f"Bundle: psd={b.get('psd_method')} runs={b.get('n_bold_runs')} "
            f"cleaned={b.get('n_cleaned_runs')} "
            f"bakeoff={'yes' if b.get('ml_bakeoff') else 'no'} "
            f"tf={'yes' if b.get('tf_results') else 'no'}"
        )
    else:
        print("⚠️  No book_bundle.json — WASM chapters will lack embedded data")

    ok = True
    with tempfile.TemporaryDirectory(prefix="marimo_wasm_") as td:
        tmp = Path(td)
        for name in CANDIDATES:
            nb = NOTEBOOK_DIR / name
            if not nb.exists():
                print(f"Missing notebook: {nb}", file=sys.stderr)
                ok = False
                continue
            if not export_one(nb, mode, tmp):
                ok = False

        # One coherent Vite asset pool for all chapters (GitHub Pages-safe)
        print("Sharing marimo frontend assets across chapters …")
        if not share_wasm_assets(EXPORT_DIR):
            ok = False

    if args.sync_docs:
        print("Syncing docs/wasm …")
        sync_docs()

    for name in CANDIDATES:
        stem = Path(name).stem
        idx = EXPORT_DIR / stem / "index.html"
        if not idx.exists():
            print(f"Missing export: {idx}", file=sys.stderr)
            ok = False

    if args.verify or args.sync_docs:
        if not verify_exports(docs=args.sync_docs):
            ok = False
        # Static FastAPI mirror required for GitHub Pages
        api_checks = [
            DOCS_DIR / "api" / "index.html",
            DOCS_DIR / "api" / "health.json",
            DOCS_DIR / "api" / "meta.json",
            DOCS_DIR / "api" / "openapi.json",
            DOCS_DIR / "api" / "features" / "spectral.json",
            DOCS_DIR / "api" / "features" / "spectral_clean.json",
            DOCS_DIR / "api" / "qc.json",
            DOCS_DIR / "api" / "bakeoff.json",
        ]
        for p in api_checks:
            if not p.exists():
                print(f"   ❌ missing static API artifact: {p}")
                ok = False
            else:
                print(f"   ✅ static API: {p.relative_to(ROOT)}")

    print("\nDone. Serve with:  python marimo_exports/serve.py --fastapi")
    print("GitHub Pages API:  …/api/  and  …/api/health.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
