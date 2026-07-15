#!/usr/bin/env python3
"""Serve the GitHub Pages book locally (docs/ after export --sync-docs)."""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT,
        help="Directory to serve (default: docs/)",
    )
    args = p.parse_args()
    root = args.dir.resolve()
    if not root.exists():
        raise SystemExit(f"Missing {root}. Run: python marimo_exports/export_wasm.py --sync-docs")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"Serving {root}")
        print(f"  Book home:  http://127.0.0.1:{args.port}/")
        print(f"  Chapter 01: http://127.0.0.1:{args.port}/wasm/01_pre_flight/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
