# Local raw BIDS (not committed)

Place OpenNeuro [ds000171](https://openneuro.org/datasets/ds000171) here as a BIDS tree:

```text
data/raw/ds000171/
  participants.tsv
  sub-*/anat/*.nii.gz
  sub-*/func/*_bold.nii.gz
  sub-*/func/*_events.tsv
```

Everything under `data/raw/ds000171/` is **gitignored** (including nested DataLad/git-annex clones). Only `data/processed/` is tracked for the public book, WASM chapters, and the static FastAPI mirror under `docs/api/`.

```bash
# After download / unlock:
python scripts/prepare_real_features.py --psd adaptive
python scripts/run_ml_bakeoff.py
python marimo_exports/export_wasm.py --sync-docs --verify
```
