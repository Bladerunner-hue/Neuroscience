"""BIDS inventory and validation for ds000171."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import pandas as pd

from neuro.config import DATA_ROOT, GROUP_MDD, GROUP_ND, TR_SEC


@dataclass
class RunRecord:
    subject: str
    task: str
    run: int
    bold_path: Path
    events_path: Path | None
    t1_path: Path | None
    group: str
    sex: str
    age: int
    bold_exists: bool
    bold_shape: tuple | None
    tr: float | None


def load_participants(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    df = pd.read_csv(data_root / "participants.tsv", sep="\t")
    df["group_short"] = df["group"].map(
        {GROUP_MDD: "MDD", GROUP_ND: "ND"}
    )
    return df


def task_tr(data_root: Path = DATA_ROOT) -> dict[str, float]:
    out = {}
    for name in ("task-music_bold.json", "task-nonmusic_bold.json"):
        p = data_root / name
        if p.exists():
            out[name.replace("_bold.json", "")] = json.loads(
                p.read_text()
            )["RepetitionTime"]
    return out


def inventory_runs(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    participants = load_participants(data_root).set_index("participant_id")
    rows: list[dict] = []

    for bold_path in sorted(data_root.glob("sub-*/func/*_bold.nii.gz")):
        stem = bold_path.name.replace("_bold.nii.gz", "")
        subject, task, run_part = stem.split("_", 2)
        run = int(run_part.replace("run-", ""))
        events_path = bold_path.with_name(
            bold_path.name.replace("_bold.nii.gz", "_events.tsv")
        )
        t1_path = data_root / subject / "anat" / f"{subject}_T1w.nii.gz"
        meta = participants.loc[subject]
        try:
            bold_exists = bold_path.is_file() and not bold_path.is_symlink() and bold_path.stat().st_size > 1000
        except OSError:
            bold_exists = False
        shape, tr = None, None
        if bold_exists:
            try:
                img = nib.load(str(bold_path))
                shape = img.shape
                tr = float(img.header.get_zooms()[3])
            except Exception:
                bold_exists = False

        rows.append(
            {
                "subject": subject,
                "task": task,
                "run": run,
                "bold_path": str(bold_path),
                "events_path": str(events_path) if events_path.exists() else None,
                "t1_path": str(t1_path) if t1_path.exists() else None,
                "group": meta["group"],
                "group_short": meta["group_short"],
                "sex": meta["sex"],
                "age": int(meta["age"]),
                "bold_exists": bold_exists,
                "n_volumes": shape[3] if shape else None,
                "tr": tr,
            }
        )

    return pd.DataFrame(rows)


def missing_files_report(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["bold_exists"]][["subject", "task", "run", "bold_path"]]


def group_counts(df: pd.DataFrame | None = None) -> pd.DataFrame:
    participants = load_participants()
    return (
        participants.groupby("group_short")
        .agg(n=("participant_id", "count"), age_mean=("age", "mean"), age_std=("age", "std"))
        .reset_index()
    )


def validate_bids(data_root: Path = DATA_ROOT) -> dict:
    participants = load_participants(data_root)
    runs = inventory_runs(data_root)
    tr_map = task_tr(data_root)
    missing = missing_files_report(runs)

    return {
        "n_subjects": participants["participant_id"].nunique(),
        "n_mdd": int((participants["group"] == GROUP_MDD).sum()),
        "n_nd": int((participants["group"] == GROUP_ND).sum()),
        "n_runs_expected": len(runs),
        "n_runs_available": int(runs["bold_exists"].sum()),
        "n_missing_bold": len(missing),
        "tr_json_music": tr_map.get("task-music"),
        "tr_json_nonmusic": tr_map.get("task-nonmusic"),
        "tr_expected": TR_SEC,
        "missing": missing,
        "runs": runs,
        "participants": participants,
    }