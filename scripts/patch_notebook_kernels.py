#!/usr/bin/env python3
"""Set kernelspec on all project notebooks."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL = {
    "display_name": "Python 3.10 (neuro)",
    "language": "python",
    "name": "neuro-310",
}

for path in sorted((REPO / "notebooks").glob("*.ipynb")):
    nb = json.loads(path.read_text())
    nb.setdefault("metadata", {})["kernelspec"] = KERNEL
    path.write_text(json.dumps(nb, indent=1))
    print("patched", path.name)