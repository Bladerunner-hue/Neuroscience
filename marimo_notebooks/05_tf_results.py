"""Chapter V — Precomputed TensorFlow neural-net results (public WASM page).

Trained offline by scripts/run_tf_offline.py — this notebook only visualizes JSON.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="V · Neural Net Results")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        key_insight_card,
        load_tf_results,
        set_global_style,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        MDD_COLOR,
        MUSIC_COLOR,
        book_nav,
        clinical_relevance_card,
        data_provenance_md,
        key_insight_card,
        load_tf_results,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(data_provenance_md, mo):
    mo.vstack(
        [
            mo.md(
                r"""
# V · Neural Net Results (precomputed)

### TensorFlow trains **offline** → this page only displays results

Browsers cannot ship TensorFlow/WASM reliably for training. Pipeline:

```text
scripts/run_tf_offline.py   (local CPU/GPU)
    → data/processed/tf_results.json
    → embedded in book_data.py
    → this chapter (matplotlib only, no tensorflow dependency)
```

Local retrain (optional): `python scripts/run_tf_offline.py`  
Interactive train notebook: `marimo edit marimo_notebooks/06_tf_spectrogram_model.py`
"""
            ),
            mo.md(data_provenance_md()),
        ]
    )
    return


@app.cell
def _(load_tf_results, mo, pd):
    tfres = load_tf_results()
    _blocks = [mo.md("## 0. Offline train snapshot")]
    if not tfres:
        _blocks.append(
            mo.md(
                "*No `tf_results.json` yet. Run `python scripts/run_tf_offline.py` "
                "then re-export the book.*"
            )
        )
    else:
        _meta = pd.DataFrame(
            [
                {
                    "tensorflow": tfres.get("tensorflow_version"),
                    "device": tfres.get("device"),
                    "n_subjects": tfres.get("n_subjects"),
                    "protocol": tfres.get("protocol", "")[:80],
                }
            ]
        )
        _models = []
        for _k, _m in (tfres.get("models") or {}).items():
            _models.append(
                {
                    "key": _k,
                    "name": _m.get("name"),
                    "target": _m.get("target"),
                    "split": _m.get("split"),
                    "acc": _m.get("acc"),
                    "f1_macro": _m.get("f1_macro"),
                    "n": _m.get("n") or _m.get("n_val") or _m.get("n_train"),
                }
            )
        _blocks.extend(
            [
                mo.ui.table(_meta),
                mo.md("### Models shipped from offline train"),
                mo.ui.table(pd.DataFrame(_models)),
            ]
        )
    mo.vstack(_blocks)
    return (tfres,)


@app.cell
def _(MUSIC_COLOR, key_insight_card, mo, np, plt, tfres):
    _blocks = [mo.md("## 1. STFT ConvNet (group) — precomputed")]
    _cnn = (tfres or {}).get("models", {}).get("cnn_group")
    if not _cnn:
        _blocks.append(mo.md("*No CNN results in tf_results.json.*"))
    else:
        _hist = _cnn.get("history") or {}
        _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
        if _hist.get("accuracy"):
            _axes[0].plot(_hist["accuracy"], label="train acc")
        if _hist.get("val_accuracy"):
            _axes[0].plot(_hist["val_accuracy"], label="val acc")
        if _hist.get("auc"):
            _axes[0].plot(_hist["auc"], ls="--", label="train AUC")
        if _hist.get("val_auc"):
            _axes[0].plot(_hist["val_auc"], ls="--", label="val AUC")
        _axes[0].set_title("Training curves (offline)")
        _axes[0].set_xlabel("Epoch")
        _axes[0].legend(frameon=False, fontsize=8)
        _axes[0].grid(True, alpha=0.3)

        _cm = np.array(_cnn.get("confusion") or [[0, 0], [0, 0]])
        _labels = _cnn.get("labels") or ["Control", "MDD"]
        _axes[1].imshow(_cm, cmap="Blues")
        _axes[1].set_xticks(range(len(_labels)))
        _axes[1].set_yticks(range(len(_labels)))
        _axes[1].set_xticklabels(_labels)
        _axes[1].set_yticklabels(_labels)
        _axes[1].set_title(
            f"CNN CM · F1={_cnn.get('f1_macro')} · acc={_cnn.get('acc')}"
        )
        _axes[1].set_xlabel("Predicted")
        _axes[1].set_ylabel("True")
        for _i in range(_cm.shape[0]):
            for _j in range(_cm.shape[1]):
                _axes[1].text(
                    _j,
                    _i,
                    int(_cm[_i, _j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if _cm[_i, _j] > _cm.max() / 2 else "black",
                )
        _fig.tight_layout()
        _blocks.extend(
            [
                mo.md(
                    f"**Split:** `{_cnn.get('split')}` · "
                    f"n_train={_cnn.get('n_train')} · n_val={_cnn.get('n_val')} · "
                    f"spectrograms={_cnn.get('n_spectrograms')} · shape={_cnn.get('spectrogram_shape')}"
                ),
                _fig,
                mo.md(
                    key_insight_card(
                        "Subject-holdout CNN metrics come from a real offline train.",
                        "This page never runs TensorFlow — only plots stored history + confusion.",
                    )
                ),
            ]
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(MUSIC_COLOR, key_insight_card, mo, np, plt, tfres):
    _blocks = [mo.md("## 2. Dense MLP LOOCV (group) + domain MLP")]
    _mlp_g = (tfres or {}).get("models", {}).get("mlp_group")
    _mlp_d = (tfres or {}).get("models", {}).get("mlp_domain")
    if not _mlp_g and not _mlp_d:
        _blocks.append(mo.md("*No MLP results.*"))
    else:
        _n = sum(1 for x in (_mlp_g, _mlp_d) if x)
        _fig, _axes = plt.subplots(1, _n, figsize=(4.5 * _n, 3.8))
        if _n == 1:
            _axes = [_axes]
        _ai = 0
        if _mlp_g:
            _cm = np.array(_mlp_g.get("confusion") or [[0, 0], [0, 0]])
            _lab = _mlp_g.get("labels") or ["Control", "MDD"]
            _axes[_ai].imshow(_cm, cmap="Blues")
            _axes[_ai].set_xticks(range(len(_lab)))
            _axes[_ai].set_yticks(range(len(_lab)))
            _axes[_ai].set_xticklabels(_lab)
            _axes[_ai].set_yticklabels(_lab)
            _axes[_ai].set_title(
                f"MLP group LOOCV · F1={_mlp_g.get('f1_macro')}"
            )
            for _i in range(_cm.shape[0]):
                for _j in range(_cm.shape[1]):
                    _axes[_ai].text(
                        _j,
                        _i,
                        int(_cm[_i, _j]),
                        ha="center",
                        va="center",
                        fontsize=13,
                        fontweight="bold",
                        color="white" if _cm[_i, _j] > _cm.max() / 2 else "black",
                    )
            _ai += 1
        if _mlp_d:
            _cm = np.array(_mlp_d.get("confusion") or [[0]])
            _lab = _mlp_d.get("classes") or []
            _axes[_ai].imshow(_cm, cmap="Greens")
            _axes[_ai].set_xticks(range(len(_lab)))
            _axes[_ai].set_yticks(range(len(_lab)))
            _axes[_ai].set_xticklabels(_lab, fontsize=8)
            _axes[_ai].set_yticklabels(_lab, fontsize=8)
            _axes[_ai].set_title(
                f"MLP domain · F1={_mlp_d.get('f1_macro')}"
            )
            for _i in range(_cm.shape[0]):
                for _j in range(_cm.shape[1]):
                    _axes[_ai].text(
                        _j,
                        _i,
                        int(_cm[_i, _j]),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="white" if _cm[_i, _j] > _cm.max() / 2 else "black",
                    )
            _hist = _mlp_d.get("history") or {}
        _fig.tight_layout()
        _blocks.append(_fig)
        if _mlp_d and (_mlp_d.get("history") or {}).get("accuracy"):
            _fig2, _ax2 = plt.subplots(figsize=(7, 3.2))
            _h = _mlp_d["history"]
            _ax2.plot(_h.get("accuracy", []), label="train")
            _ax2.plot(_h.get("val_accuracy", []), label="val")
            _ax2.set_title("Domain MLP accuracy (offline)")
            _ax2.legend(frameon=False)
            _ax2.grid(True, alpha=0.3)
            _fig2.tight_layout()
            _blocks.append(_fig2)
        _blocks.append(
            mo.md(
                key_insight_card(
                    "MLP LOOCV is the deep peer of Ch III group models.",
                    "Compare F1 to sklearn winners in the next section before promoting nets to RecSys.",
                )
            )
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(MUSIC_COLOR, key_insight_card, mo, pd, plt, tfres):
    _blocks = [mo.md("## 3. Sklearn bake-off vs TensorFlow (recommended solver)")]
    _cmp = (tfres or {}).get("comparison") or []
    _rec = (tfres or {}).get("recommended") or {}
    if not _cmp:
        _blocks.append(mo.md("*No comparison table — retrain offline.*"))
    else:
        _df = pd.DataFrame(_cmp)
        _fig, _ax = plt.subplots(figsize=(9, max(3.2, 0.45 * len(_df))))
        _cols = [
            MUSIC_COLOR if str(f) == "tensorflow" else "#5D6D7E"
            for f in _df.get("family", [])
        ]
        _labels = [
            f"{r.target} · {r.model} ({r.family})" for r in _df.itertuples()
        ]
        _ax.barh(
            _labels[::-1],
            _df["f1_macro"].values[::-1],
            color=_cols[::-1],
            edgecolor="white",
        )
        _ax.set_xlabel("macro-F1")
        _ax.set_xlim(0, 1.05)
        _ax.set_title("Classical vs neural (offline scores)")
        _ax.grid(True, axis="x", alpha=0.3)
        _fig.tight_layout()
        _lines = []
        for _t, _b in _rec.items():
            _lines.append(
                f"- **{_t}** → `{_b.get('model')}` ({_b.get('family')}) "
                f"F1={_b.get('f1_macro')}"
            )
        _blocks.extend(
            [
                mo.ui.table(_df),
                _fig,
                mo.md("### Recommended\n\n" + ("\n".join(_lines) if _lines else "*n/a*")),
                mo.md(
                    key_insight_card(
                        "Ship the best family per target — not always deep learning.",
                        "Tabular RecSys fingerprints often stay with RF/GBM; STFT CNNs when raw spectrograms matter.",
                    )
                ),
            ]
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.vstack(
        [
            mo.md(
                r"""
## How to refresh this page after new BOLD / models

```bash
python scripts/prepare_real_features.py
python scripts/run_ml_bakeoff.py
python scripts/run_tf_offline.py          # trains TF locally
python marimo_exports/export_wasm.py --sync-docs
```

Interactive retrain UI (not on Pages): `marimo_notebooks/06_tf_spectrogram_model.py`
"""
            ),
            mo.md(
                clinical_relevance_card(
                    "Neural-net scores only enter playlist / clinical narratives when they beat "
                    "the transparent sklearn bake-off on the same music-effect targets."
                )
            ),
            mo.md(book_nav("05_tf_results")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
