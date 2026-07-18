#!/usr/bin/env python3
"""Scan data/raw + multi_dataset_catalog → data/processed/dataset_registry.json.

Does not download. Safe offline refresh for the landscape notebook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "dataset_registry.json"
sys.path.insert(0, str(ROOT / "marimo_notebooks"))

from multi_dataset_catalog import MULTI_DATASET_CATALOG  # noqa: E402


def _count_bold(ds_dir: Path) -> int:
    if not ds_dir.exists():
        return 0
    return sum(1 for _ in ds_dir.rglob("*bold.nii.gz"))


def _subjects(ds_dir: Path) -> list[str]:
    if not ds_dir.exists():
        return []
    return sorted(p.name for p in ds_dir.glob("sub-*") if p.is_dir())


def main() -> int:
    reg: dict = {}
    for ds_id, cat in MULTI_DATASET_CATALOG.items():
        ds_dir = RAW / ds_id
        nested = ds_dir / ds_id
        subjects = _subjects(ds_dir) or _subjects(nested)
        part = ds_dir / "participants.tsv"
        if not part.exists() and (nested / "participants.tsv").exists():
            part = nested / "participants.tsv"
        n_bold = _count_bold(ds_dir)
        desc = ds_dir / "dataset_description.json"
        if not desc.exists() and (nested / "dataset_description.json").exists():
            desc = nested / "dataset_description.json"
        status = "ok" if ds_dir.exists() else "missing"
        if ds_dir.exists() and not part.exists() and n_bold == 0 and not desc.exists():
            status = "stub"
        reg[ds_id] = {
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
            "modal_extra": cat.get("modal_extra"),
            "tasks": cat.get("tasks"),
            "cohort": cat.get("cohort"),
            "local_path": f"data/raw/{ds_id}",
            "downloaded_meta": part.exists() or desc.exists(),
            "has_participants_tsv": part.exists(),
            "n_subjects_on_disk": len(subjects),
            "subjects_on_disk": subjects[:20],
            "n_bold_files": n_bold,
            "status": status,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    for k, v in reg.items():
        print(
            f"  {k:10} role={v['role']:10} status={v['status']:8} "
            f"sub={v['n_subjects_on_disk']:3} bold={v['n_bold_files']:4} "
            f"match={v.get('match_level')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
