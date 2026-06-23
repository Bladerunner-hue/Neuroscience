#!/usr/bin/env python3
"""
Export marimo notebooks to self-contained WebAssembly HTML.

Usage (from repo root):
    python marimo_exports/export_wasm.py

This produces interactive, browser-only versions under marimo_exports/wasm/

Notes:
- Use `marimo edit --sandbox notebook.py` during development so dependencies are declared.
- Only notebooks using Pyodide-compatible packages (numpy, scipy, pandas, matplotlib via mpl, plotly) will work fully.
- Heavy TensorFlow notebooks (e.g. 06) cannot run in the browser.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "marimo_notebooks"
EXPORT_DIR = ROOT / "marimo_exports" / "wasm"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Notebooks that are good candidates for WASM (no TF, use supported libs)
CANDIDATES = [
    "01_pre_flight.py",
    "02_eda_univariate.py",
    "03_eda_multivariate.py",
    "04_feature_engineering.py",
]

def export_one(notebook: Path, mode: str = "run"):
    out_dir = EXPORT_DIR / notebook.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "index.html"

    cmd = [
        "marimo",
        "export",
        "html-wasm",
        str(notebook),
        "-o",
        str(out_dir),
        "--mode",
        mode,
    ]

    print(f"→ Exporting {notebook.name} ({mode}) ...")
    try:
        subprocess.check_call(cmd, cwd=ROOT)
        print(f"   ✅ Wrote {out_file}")
        # Add .nojekyll so GitHub Pages serves it correctly
        (out_dir / ".nojekyll").touch(exist_ok=True)
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--edit":
        mode = "edit"
    else:
        mode = "run"

    print(f"Exporting to {EXPORT_DIR} (mode={mode})")
    for name in CANDIDATES:
        nb = NOTEBOOK_DIR / name
        if nb.exists():
            export_one(nb, mode)
        else:
            print(f"Skipping missing: {nb}")

    print("\nDone. You can now open any wasm/*/index.html directly in a browser.")
    print("For GitHub Pages, copy the wasm/ contents (or individual dirs) into docs/ or enable Pages on the repo.")