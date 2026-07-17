#!/usr/bin/env python3
"""Download OpenNeuro cross-ref cohorts for multi-dataset spectral work.

Default: metadata + participants for all registry datasets; optional limited
BOLD for a couple of small pulls. ds000171 is expected already under data/raw/.

Usage (repo root)::

    python scripts/download_openneuro_cohorts.py
    python scripts/download_openneuro_cohorts.py --with-bold --max-subjects 1
    python scripts/download_openneuro_cohorts.py --only ds002725,ds003085
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REGISTRY = ROOT / "data" / "processed" / "dataset_registry.json"

# Cross-ref registry (ARCHITECTURE.md). include_patterns for light pulls.
DATASETS: dict[str, dict] = {
    "ds000171": {
        "title": "Emotional musical vs nonmusical stimuli in depression",
        "role": "primary",
        "why": "Core MDD vs Control music/tones design",
        "url": "https://openneuro.org/datasets/ds000171",
        "meta_include": [
            "dataset_description.json",
            "participants.tsv",
            "README",
            "CHANGES",
        ],
    },
    "ds002725": {
        "title": "Joint EEG-fMRI affective music",
        "role": "cross_ref",
        "why": "Multimodal spectral / ERP validation",
        "url": "https://openneuro.org/datasets/ds002725",
        "meta_include": [
            "dataset_description.json",
            "participants.tsv",
            "README",
        ],
        # Prefer one control subject func if --with-bold
        "bold_include_globs": [
            "sub-*/func/*bold.nii.gz",
            "sub-*/func/*events.tsv",
            "sub-*/anat/*T1w.nii.gz",
        ],
    },
    "ds003085": {
        "title": "Temporal dynamics of emotional music",
        "role": "cross_ref",
        "why": "Happy/sad music block design",
        "url": "https://openneuro.org/datasets/ds003085",
        "meta_include": [
            "dataset_description.json",
            "participants.tsv",
            "README",
        ],
        "bold_include_globs": [
            "sub-*/func/*bold.nii.gz",
            "sub-*/func/*events.tsv",
        ],
    },
    "ds003720": {
        "title": "Music genre fMRI",
        "role": "cross_ref",
        "why": "Genre-specific auditory structure",
        "url": "https://openneuro.org/datasets/ds003720",
        "meta_include": [
            "dataset_description.json",
            "participants.tsv",
            "README",
        ],
    },
    "ds004142": {
        "title": "rt-fMRI neurofeedback reward valence",
        "role": "cross_ref",
        "why": "Reward saliency / valence",
        "url": "https://openneuro.org/datasets/ds004142",
        "meta_include": [
            "dataset_description.json",
            "participants.tsv",
            "README",
        ],
    },
    "ds005700": {
        "title": "NeuroEmo emotion recognition",
        "role": "cross_ref",
        "why": "Emotion labels including depressed",
        "url": "https://openneuro.org/datasets/ds005700",
        "meta_include": [
            "dataset_description.json",
            "participants.tsv",
            "README",
        ],
    },
    "ds006564": {
        "title": "Naturalistic film + musical soundtracks",
        "role": "cross_ref",
        "why": "Depression/anxiety traits + music",
        "url": "https://openneuro.org/datasets/ds006564",
        "meta_include": [
            "dataset_description.json",
            "participants.tsv",
            "README",
        ],
    },
}


def _download(dataset: str, target: Path, include: list[str] | None) -> None:
    import openneuro as on

    target.mkdir(parents=True, exist_ok=True)
    print(f"→ openneuro.download({dataset}) → {target}")
    print(f"  include={include}")
    on.download(
        dataset=dataset,
        target_dir=str(target),
        include=include,
        verify_hash=True,
        max_concurrent_downloads=4,
    )


def _list_subjects(ds_dir: Path) -> list[str]:
    return sorted(p.name for p in ds_dir.glob("sub-*") if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated dataset ids (default: all registry)",
    )
    p.add_argument(
        "--with-bold",
        action="store_true",
        help="Also pull limited BOLD (first N subjects) where globs defined",
    )
    p.add_argument(
        "--max-subjects",
        type=int,
        default=1,
        help="Max subjects per dataset when --with-bold (default 1)",
    )
    p.add_argument(
        "--skip-existing-meta",
        action="store_true",
        help="Skip if participants.tsv already present",
    )
    args = p.parse_args(argv)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    reg: dict[str, dict] = {}
    RAW.mkdir(parents=True, exist_ok=True)

    for ds_id, meta in DATASETS.items():
        if only and ds_id not in only:
            continue
        target = RAW / ds_id
        # Nested layout sometimes used: data/raw/ds000171/ds000171/
        nested = target / ds_id
        participants = target / "participants.tsv"
        if not participants.exists() and (nested / "participants.tsv").exists():
            participants = nested / "participants.tsv"

        entry = {
            **meta,
            "id": ds_id,
            "local_path": str(target.relative_to(ROOT)),
            "downloaded_meta": False,
            "downloaded_bold_subjects": [],
            "n_subjects_on_disk": 0,
            "status": "pending",
        }

        try:
            if args.skip_existing_meta and participants.exists():
                print(f"skip meta {ds_id} (participants present)")
                entry["downloaded_meta"] = True
            else:
                # Always try metadata
                _download(ds_id, target, meta.get("meta_include"))
                entry["downloaded_meta"] = True

            if args.with_bold and meta.get("bold_include_globs"):
                # OpenNeuro include is path prefixes / names — pull one subject if known
                # First download may have created subject dirs only for include matches.
                # Request limited bold via include patterns restricted to first subject after meta.
                # Strategy: download with bold globs (can be large) — cap via include of specific subjects
                # if we can list subjects from API/metadata.
                # Safer: include only first listed subject from participants.tsv if present.
                subjects: list[str] = []
                part = target / "participants.tsv"
                if part.exists():
                    import pandas as pd

                    pdf = pd.read_csv(part, sep="\t")
                    col = "participant_id" if "participant_id" in pdf.columns else pdf.columns[0]
                    subjects = [str(x) for x in pdf[col].tolist() if str(x).startswith("sub-")]
                subjects = subjects[: max(1, args.max_subjects)]
                include = list(meta.get("meta_include") or [])
                for sub in subjects:
                    include.append(f"{sub}/*")
                print(f"  bold subjects: {subjects}")
                _download(ds_id, target, include)
                entry["downloaded_bold_subjects"] = subjects

            entry["n_subjects_on_disk"] = len(_list_subjects(target)) + len(
                _list_subjects(nested)
            )
            entry["status"] = "ok"
        except Exception as e:
            print(f"  ⚠ {ds_id}: {e}", file=sys.stderr)
            entry["status"] = f"error: {e}"
            entry["error"] = str(e)

        reg[ds_id] = entry
        print(f"  status={entry['status']} subjects_on_disk={entry['n_subjects_on_disk']}")

    # Always record primary ds000171 presence
    if "ds000171" not in reg:
        p171 = RAW / "ds000171"
        reg["ds000171"] = {
            **DATASETS["ds000171"],
            "id": "ds000171",
            "local_path": "data/raw/ds000171",
            "status": "local" if p171.exists() else "missing",
            "n_subjects_on_disk": len(_list_subjects(p171))
            + len(_list_subjects(p171 / "ds000171")),
        }

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"\nWrote {REGISTRY}")
    ok = sum(1 for v in reg.values() if str(v.get("status", "")).startswith(("ok", "local")))
    print(f"datasets ok/local: {ok}/{len(reg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
