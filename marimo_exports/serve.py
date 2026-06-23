#!/usr/bin/env python3
"""Simple static server for marimo HTML dashboards."""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT = int(os.environ.get("PORT", 8765))
DIRECTORY = Path(__file__).parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        # Disable caching so you see changes immediately when re-exporting
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    os.chdir(DIRECTORY)
    print(f"🚀 Serving marimo dashboards at http://localhost:{PORT}")
    print(f"   Static index:   http://localhost:{PORT}/index.html")
    print(f"   WASM gallery:   http://localhost:{PORT}/wasm/ (if exported)")
    print(f"   (Press Ctrl+C to stop)")
    try:
        HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
        sys.exit(0)
