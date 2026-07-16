#!/usr/bin/env python3
"""Regenerate marimo_notebooks/book_data.py from data/processed/book_bundle.json.

IMPORTANT: Do not embed json.dumps(...) as a Python dict literal — JSON uses
true/false/null which are NameErrors in Python. We base64-encode the JSON and
decode at import time so the module is always valid Python.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bundle_path = ROOT / "data" / "processed" / "book_bundle.json"
out = ROOT / "marimo_notebooks" / "book_data.py"

bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
raw = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
b64 = base64.b64encode(raw).decode("ascii")
chunks = [b64[i : i + 80] for i in range(0, len(b64), 80)]
lit = "(\n" + "".join(f'    "{c}"\n' for c in chunks) + ")"

out.write_text(
    '"""Auto-generated real-data bundle for the interactive book.\n'
    "Source: OpenNeuro ds000171 (processed subset).\n"
    "Regenerate: python scripts/gen_book_data.py\n"
    '"""\n'
    "from __future__ import annotations\n\n"
    "import base64\n"
    "import json\n\n"
    f"_BOOK_BUNDLE_B64 = {lit}\n\n"
    "BOOK_BUNDLE = json.loads(\n"
    '    base64.b64decode("".join(_BOOK_BUNDLE_B64)).decode("utf-8")\n'
    ")\n",
    encoding="utf-8",
)
print("Wrote", out, out.stat().st_size, "bytes")
# sanity: importable Python with True/False, not true/false
ns: dict = {}
exec(compile(out.read_text(encoding="utf-8"), str(out), "exec"), ns)
assert "BOOK_BUNDLE" in ns and isinstance(ns["BOOK_BUNDLE"], dict)
assert ns["BOOK_BUNDLE"].get("n_bold_runs") or ns["BOOK_BUNDLE"].get("participants") is not None
print("Import OK · keys:", list(ns["BOOK_BUNDLE"].keys())[:8])
