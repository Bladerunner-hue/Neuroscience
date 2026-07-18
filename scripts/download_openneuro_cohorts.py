#!/usr/bin/env python3
"""Download OpenNeuro multi-dataset catalog for spectral / music-emotion work.

Uses ``marimo_notebooks/multi_dataset_catalog.py`` as the source of truth.
Default: metadata for all catalog datasets; optional limited BOLD.

Usage (repo root)::

    python scripts/download_openneuro_cohorts.py
    python scripts/download_openneuro_cohorts.py --with-bold --max-subjects 1
    python scripts/download_openneuro_cohorts.py --only ds002725,ds004894
    python scripts/refresh_dataset_registry.py   # after download
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REGISTRY = ROOT / "data" / "processed" / "dataset_registry.json"
sys.path.insert(0, str(ROOT / "marimo_notebooks"))

from multi_dataset_catalog import MULTI_DATASET_CATALOG  # noqa: E402

# Lightweight meta patterns (some BIDS trees omit participants.tsv at root)
DEFAULT_META = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "README.txt",
    "CHANGES",
]


def _download(dataset: str, target: Path, include: list[str] | None) -> None:
    import openneuro as on

    target.mkdir(parents=True, exist_ok=True)
    print(f"→ openneuro.download({dataset}) → {target}")
    print(f"  include={include}")
    on.download(
        dataset=dataset,
        target_dir=str(target),
        include=include,
        verify_hash=False,
        max_concurrent_downloads=4,
    )


def _list_subjects(ds_dir: Path) -> list[str]:
    return sorted(p.name for p in ds_dir.glob("sub-*") if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", type=str, default="", help="Comma-separated dataset ids")
    p.add_argument("--with-bold", action="store_true")
    p.add_argument("--max-subjects", type=int, default=1)
    p.add_argument("--skip-existing-meta", action="store_true")
    args = p.parse_args(argv)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    reg: dict[str, dict] = {}
    RAW.mkdir(parents=True, exist_ok=True)

    for ds_id, cat in sorted(
        MULTI_DATASET_CATALOG.items(), key=lambda kv: kv[1].get("priority", 99)
    ):
        if only and ds_id not in only:
            continue
        # Primary often already local — still refresh registry fields
        target = RAW / ds_id
        nested = target / ds_id
        participants = target / "participants.tsv"
        if not participants.exists() and (nested / "participants.tsv").exists():
            participants = nested / "participants.tsv"

        entry = {
            "id": ds_id,
            "title": cat.get("title"),
            "short_title": cat.get("short_title"),
            "role": cat.get("role"),
            "priority": cat.get("priority"),
            "match_level": cat.get("match_level"),
            "why": cat.get("why_neuro"),
            "why_preprocess": cat.get("why_preprocess"),
            "integration": cat.get("integration"),
            "url": cat.get("url"),
            "n_participants_nominal": cat.get("n_participants_nominal"),
            "modality": cat.get("modality"),
            "tasks": cat.get("tasks"),
            "local_path": str(target.relative_to(ROOT)),
            "downloaded_meta": False,
            "downloaded_bold_subjects": [],
            "n_subjects_on_disk": 0,
            "n_bold_files": 0,
            "status": "pending",
        }

        try:
            if ds_id == "ds000171" and participants.exists():
                print(f"skip download {ds_id} (primary already local)")
                entry["downloaded_meta"] = True
            elif args.skip_existing_meta and participants.exists():
                print(f"skip meta {ds_id} (participants present)")
                entry["downloaded_meta"] = True
            else:
                _download(ds_id, target, list(DEFAULT_META))
                entry["downloaded_meta"] = True

            if args.with_bold and cat.get("bold_include_globs") and ds_id != "ds000171":
                subjects: list[str] = []
                part = target / "participants.tsv"
                if part.exists():
                    import pandas as pd

                    pdf = pd.read_csv(part, sep="\t")
                    col = (
                        "participant_id"
                        if "participant_id" in pdf.columns
                        else pdf.columns[0]
                    )
                    subjects = [
                        str(x) for x in pdf[col].tolist() if str(x).startswith("sub-")
                    ]
                subjects = subjects[: max(1, args.max_subjects)]
                include = list(DEFAULT_META)
                for sub in subjects:
                    include.append(f"{sub}/*")
                print(f"  bold subjects: {subjects}")
                _download(ds_id, target, include)
                entry["downloaded_bold_subjects"] = subjects

            entry["n_subjects_on_disk"] = len(_list_subjects(target)) + len(
                _list_subjects(nested)
            )
            entry["n_bold_files"] = sum(1 for _ in target.rglob("*bold.nii.gz"))
            entry["status"] = "ok"
        except Exception as e:
            print(f"  ⚠ {ds_id}: {e}", file=sys.stderr)
            entry["status"] = f"error: {e}"
            entry["error"] = str(e)
            entry["n_subjects_on_disk"] = len(_list_subjects(target))
            entry["n_bold_files"] = (
                sum(1 for _ in target.rglob("*bold.nii.gz")) if target.exists() else 0
            )

        reg[ds_id] = entry
        print(
            f"  status={entry['status']} subjects={entry['n_subjects_on_disk']} "
            f"bold={entry['n_bold_files']}"
        )

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"\nWrote {REGISTRY}")
    # Prefer full disk scan merge for final truth
    try:
        from refresh_dataset_registry import main as refresh

        print("Refreshing registry from disk scan…")
        refresh()
    except Exception as e:
        print(f"(refresh skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
