from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"


def _resolve_bids_root(raw_dir: Path = RAW_DIR) -> Path:
    """Find BIDS dataset root under data/raw (ds000171/ or raw/ itself)."""
    candidates = [raw_dir / "ds000171", raw_dir]
    for path in candidates:
        if (path / "participants.tsv").exists():
            return path
    return raw_dir / "ds000171"


DATA_ROOT = _resolve_bids_root()
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "figures"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

TR_SEC = 3.0
GROUP_MDD = "Major Depressive Disorder"
GROUP_ND = "Never-Depressed Control"

SPARK_MASTER = "local[*]"
SPARK_REMOTE = "spark://10.66.91.1:7077,10.66.91.2:7077"