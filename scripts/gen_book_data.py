#!/usr/bin/env python3
"""Regenerate marimo_notebooks/book_data.py from data/processed/book_bundle.json."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
bundle = json.loads((ROOT / "data/processed/book_bundle.json").read_text())
out = ROOT / "marimo_notebooks" / "book_data.py"
out.write_text(
    '"""Auto-generated real-data bundle for the interactive book.\n'
    "Source: OpenNeuro ds000171 (processed subset). Regenerate via scripts/gen_book_data.py\n"
    '"""\n'
    "from __future__ import annotations\n\n"
    f"BOOK_BUNDLE = {json.dumps(bundle, indent=2)}\n"
)
print("Wrote", out, out.stat().st_size, "bytes")
