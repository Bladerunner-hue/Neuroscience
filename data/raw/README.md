# `data/raw/` — OpenNeuro BIDS trees

One directory per dataset ID. **Canonical primary:** `ds000171/`.

| Folder | Role | On disk (typical) | OpenNeuro |
|--------|------|-------------------|-----------|
| `ds000171/` | **Primary** MDD music/nonmusic | Full BOLD (~21 G) | https://openneuro.org/datasets/ds000171 |
| `ds002725/` | EEG-fMRI affective music | Meta + sample BOLD | https://openneuro.org/datasets/ds002725 |
| `ds003085/` | Happy/sad music dynamics | Meta / STATUS | https://openneuro.org/datasets/ds003085 |
| `ds003720/` | Genre listening baseline | Meta / STATUS | https://openneuro.org/datasets/ds003720 |
| `ds004142/` | Reward valence nf | Meta | https://openneuro.org/datasets/ds004142 |
| `ds004894/` | Music + HR + insula | Stub / STATUS | https://openneuro.org/datasets/ds004894 |
| `ds005700/` | NeuroEmo visual emotion | Meta / STATUS | https://openneuro.org/datasets/ds005700 |
| `ds006564/` | Naturalistic film+music | Meta / STATUS | https://openneuro.org/datasets/ds006564 |

See each folder’s `STATUS.txt` for download commands. Do **not** place loose `sub-*` trees at the `raw/` root — always under `raw/<ds_id>/`.
