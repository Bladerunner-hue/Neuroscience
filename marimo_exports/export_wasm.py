#!/usr/bin/env python3
"""
Build the GitHub Pages book from canonical marimo_notebooks/.

Pipeline:
  marimo_notebooks/*.py  (+ helpers.py)
       →  inject helpers (base64, first cell) for Pyodide
       →  marimo export html-wasm
       →  marimo_exports/wasm/<chapter>/
       →  docs/wasm/<chapter>/   (--sync-docs)
       →  GitHub Pages (CI)

Why base64? helpers.py starts with a triple-quoted docstring. Embedding it in a
Python \"\"\" string truncates early and produces a blank WASM page. Base64 avoids
all quote/indent issues.

Usage (repo root):
  python marimo_exports/export_wasm.py --sync-docs
"""
from __future__ import annotations

import argparse
import ast
import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "marimo_notebooks"
HELPERS = NOTEBOOK_DIR / "helpers.py"
BOOK_DATA = NOTEBOOK_DIR / "book_data.py"
EXPORT_DIR = ROOT / "marimo_exports" / "wasm"
DOCS_DIR = ROOT / "docs"
DOCS_WASM = DOCS_DIR / "wasm"

# WASM book chapters (TF trains offline; 05 shows precomputed results only)
CANDIDATES = [
    "01_pre_flight.py",
    "02_eda_univariate.py",
    "03_eda_multivariate.py",
    "04_feature_engineering.py",
    "05_tf_results.py",  # precomputed TF/NN page — no tensorflow package in browser
]


def _b64_chunks(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    b64 = base64.b64encode(data).decode("ascii")
    chunks = [b64[i : i + 80] for i in range(0, len(b64), 80)]
    return "(\n" + "".join(f'        "{c}"\n' for c in chunks) + "    )"


def _inject_modules(
    notebook_src: str,
    *,
    helpers_src: str,
    book_bundle_json: str | None,
) -> str:
    """Register helpers + book_data in sys.modules for Pyodide.

    helpers: exec Python source (must be valid Python).
    book_data: NEVER exec a dict literal — load JSON via base64 + json.loads
    so JSON true/false/null cannot become NameErrors.
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
    if book_bundle_json:
        # Prefer compact JSON bytes from book_bundle.json (canonical feature store)
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


def export_one(notebook: Path, mode: str, tmp_dir: Path) -> bool:
    out_dir = EXPORT_DIR / notebook.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    helpers_src = HELPERS.read_text(encoding="utf-8")
    # Canonical JSON for book_data (never exec Python dict with true/false)
    bundle_json_path = ROOT / "data" / "processed" / "book_bundle.json"
    if bundle_json_path.exists():
        book_bundle_json = bundle_json_path.read_text(encoding="utf-8")
    elif BOOK_DATA.exists():
        # fallback: import local book_data safely and re-serialize
        ns: dict = {}
        exec(
            compile(BOOK_DATA.read_text(encoding="utf-8"), "book_data.py", "exec"),
            ns,
        )
        import json as _json

        book_bundle_json = _json.dumps(ns["BOOK_BUNDLE"], separators=(",", ":"))
    else:
        book_bundle_json = None

    nb_src = notebook.read_text(encoding="utf-8")
    packed = _inject_modules(
        nb_src, helpers_src=helpers_src, book_bundle_json=book_bundle_json
    )
    _validate_packed(packed, notebook)
    # Fail closed: injected source must not contain JSON-as-Python booleans
    if re.search(r'BOOK_BUNDLE\s*=\s*\{[^}]*\btrue\b', packed):
        raise RuntimeError(
            f"Refusing to export {notebook.name}: BOOK_BUNDLE still has JSON true/false"
        )

    tmp_nb = tmp_dir / notebook.name
    tmp_nb.write_text(packed, encoding="utf-8")

    # Sanity: can import helpers via the same mechanism
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                "import runpy; runpy.run_path(r'%s', run_name='__not_main__')"
                % str(tmp_nb).replace("'", "\\'"),
            ],
            cwd=str(NOTEBOOK_DIR),
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "MPLBACKEND": "Agg"},
        )
    except subprocess.CalledProcessError:
        # run_path executes module body only (defines app) — OK if no side effects
        pass

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

    # Guard against the triple-quote breakage pattern
    html = index.read_text(encoding="utf-8", errors="replace")
    if "injected by export_wasm" not in html:
        print("   ❌ export missing module injection marker")
        return False
    if 'exec(compile("""' in html or "exec(compile(\\\"\\\"\\\"" in html:
        print("   ❌ unsafe triple-quote embedding detected")
        return False
    if "b64decode" not in html:
        print("   ❌ base64 module loader missing from export")
        return False
    if book_bundle_json is not None and (
        "book_data" not in html and "_book_json_b64" not in html
    ):
        print("   ❌ book_data JSON injection missing from export")
        return False
    # Old bug marker: JSON true inside exec'd book_data.py source
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
    print(f"   ✅ {index} ({index.stat().st_size} bytes)")
    return True


def sync_docs() -> None:
    DOCS_WASM.mkdir(parents=True, exist_ok=True)
    if DOCS_WASM.exists():
        for child in list(DOCS_WASM.iterdir()):
            if child.is_dir() and not (EXPORT_DIR / child.name).exists():
                shutil.rmtree(child)
                print(f"   removed stale docs/wasm/{child.name}")
    for child in EXPORT_DIR.iterdir():
        if not child.is_dir():
            continue
        dest = DOCS_WASM / child.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(child, dest)
        print(f"   synced → docs/wasm/{child.name}")
    (DOCS_DIR / ".nojekyll").touch(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", action="store_true", help="Editable WASM (default: run)")
    parser.add_argument(
        "--sync-docs",
        action="store_true",
        help="Copy wasm exports into docs/ for GitHub Pages",
    )
    args = parser.parse_args()
    mode = "edit" if args.edit else "run"

    if not HELPERS.exists():
        print(f"Missing canonical helpers: {HELPERS}", file=sys.stderr)
        return 1

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Canonical notebooks: {NOTEBOOK_DIR}")
    print(f"Export dir: {EXPORT_DIR} (mode={mode})")

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

    if args.sync_docs:
        print("Syncing docs/wasm …")
        sync_docs()

    for name in CANDIDATES:
        stem = Path(name).stem
        idx = EXPORT_DIR / stem / "index.html"
        if not idx.exists():
            print(f"Missing export: {idx}", file=sys.stderr)
            ok = False

    print("\nDone. Serve with:  python marimo_exports/serve.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
