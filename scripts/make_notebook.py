#!/usr/bin/env python3
"""Generate a minimal Jupyter notebook from code cells."""

import json
import sys
from pathlib import Path


def make_nb(title: str, cells: list[tuple[str, str]], path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]}],
    }
    for kind, src in cells:
        if kind == "md":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [src]})
        else:
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": [line + "\n" for line in src.splitlines()],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    make_nb(sys.argv[1], [], Path(sys.argv[2]))