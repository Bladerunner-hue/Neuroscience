#!/usr/bin/env python3
"""
Build the GitHub Pages book from canonical marimo_notebooks/.

Pipeline:
  marimo_notebooks/*.py  (+ helpers.py)
       →  marimo export html-wasm
       →  marimo_exports/wasm/<chapter>/
       →  docs/wasm/<chapter>/   (--sync-docs)
       →  GitHub Pages (CI)

helpers.py is injected at module scope so WASM/Pyodide can `from helpers import …`
without a real filesystem package.

Usage (repo root):
  python marimo_exports/export_wasm.py --sync-docs
  python marimo_exports/export_wasm.py --edit --sync-docs
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "marimo_notebooks"
HELPERS = NOTEBOOK_DIR / "helpers.py"
EXPORT_DIR = ROOT / "marimo_exports" / "wasm"
DOCS_DIR = ROOT / "docs"
DOCS_WASM = DOCS_DIR / "wasm"

# WASM book chapters only (no TensorFlow)
CANDIDATES = [
    "01_pre_flight.py",
    "02_eda_univariate.py",
    "03_eda_multivariate.py",
    "04_feature_engineering.py",
]


def _inject_helpers(notebook_src: str, helpers_src: str) -> str:
    """Make notebook self-contained for Pyodide by registering helpers in sys.modules.

    Injection must live *inside the first @app.cell* — marimo html-wasm only
    serializes cell bodies, not arbitrary module-level side effects.
    """
    escaped = helpers_src.replace("\\", "\\\\").replace('"""', r'\"\"\"')
    bootstrap = f'''
    # --- injected by export_wasm.py: register helpers for WASM/Pyodide ---
    import sys as _sys
    import types as _types
    if "helpers" not in _sys.modules:
        _helpers = _types.ModuleType("helpers")
        exec(compile("""{escaped}""", "helpers.py", "exec"), _helpers.__dict__)
        _sys.modules["helpers"] = _helpers
    # --- end helpers injection ---
'''
    # Insert at the start of the first @app.cell function body
    m = re.search(
        r"(@app\.cell(?:\([^)]*\))?\s*\n"
        r"def\s+\w+\s*\([^)]*\)\s*:\s*\n)",
        notebook_src,
    )
    if not m:
        raise RuntimeError("Could not find first @app.cell to inject helpers")
    return notebook_src[: m.end()] + bootstrap + notebook_src[m.end() :]


def export_one(notebook: Path, mode: str, tmp_dir: Path) -> bool:
    out_dir = EXPORT_DIR / notebook.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    helpers_src = HELPERS.read_text(encoding="utf-8")
    nb_src = notebook.read_text(encoding="utf-8")
    packed = _inject_helpers(nb_src, helpers_src)
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
    (out_dir / ".nojekyll").touch(exist_ok=True)
    # Drop non-runtime clutter from marimo scaffold if present
    for junk in ("CLAUDE.md",):
        p = out_dir / junk
        if p.exists():
            p.unlink()
    print(f"   ✅ {index} ({index.stat().st_size} bytes)")
    return True


def sync_docs() -> None:
    DOCS_WASM.mkdir(parents=True, exist_ok=True)
    # Remove stale chapter dirs not in current export
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

    # Verify required chapters
    for name in CANDIDATES:
        stem = Path(name).stem
        idx = EXPORT_DIR / stem / "index.html"
        if not idx.exists():
            print(f"Missing export: {idx}", file=sys.stderr)
            ok = False

    print("\nDone. Serve with:  python -m http.server 8765 --directory docs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
