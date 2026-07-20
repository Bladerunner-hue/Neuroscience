"""Chapter VIII — Spark Connect Structured Streaming monitor (local only).

Checkpoint status, lastProgress, dynamic SQL config (msgspec), Plotly from
latest stream_features sink. Not exported to WASM.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "pyspark", "msgspec"]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="VIII · Spark Streaming")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import msgspec
    import pandas as pd

    ROOT = Path.cwd()
    if not (ROOT / "data").exists():
        ROOT = Path(__file__).resolve().parent.parent

    mo.md(
        r"""
# VIII · Structured Streaming monitor

Micro-batch **Spark Connect** streaming for incremental BOLD features.

| Concept | Path / control |
|---------|----------------|
| Inbox (file source) | `data/processed/stream_inbox/` |
| Feature sink | `data/processed/stream_features/` |
| Checkpoints | `data/processed/stream_checkpoints/` |
| Dynamic SQL params | `data/processed/stream_config.json` |
| Live status | `data/processed/stream_status.json` |

```bash
python -m cli.neuro_tal_cli stream-seed
python -m cli.neuro_tal_cli stream-file --once
# continuous:
# python -m cli.neuro_tal_cli stream-file --trigger-seconds 5
# Kafka:
# Prefer native Connect for batch / file streaming:
# ./scripts/start_local_spark_connect.sh
# python -m cli.neuro_tal_cli stream-kafka --bootstrap localhost:9092
```
"""
    )
    return Path, ROOT, json, mo, msgspec, pd, plt


@app.cell
def _(ROOT, mo, msgspec, pd):
    cfg_path = ROOT / "data" / "processed" / "stream_config.json"
    status_path = ROOT / "data" / "processed" / "stream_status.json"
    feat_path = ROOT / "data" / "processed" / "stream_features"

    group = mo.ui.dropdown(
        options=["MDD", "Control", "unknown"],
        value="MDD",
        label="stream_config.current_group (dynamic SQL)",
    )
    min_ph = mo.ui.slider(0.0, 0.5, value=0.0, step=0.01, label="min_power_high filter")
    save_cfg = mo.ui.run_button(label="Write stream_config.json")

    mo.vstack(
        [
            mo.md("## 1. Runtime dynamism (msgspec config)"),
            mo.hstack([group, min_ph, save_cfg], justify="start"),
        ]
    )
    return cfg_path, feat_path, group, min_ph, save_cfg, status_path


@app.cell
def _(ROOT, cfg_path, group, min_ph, mo, msgspec, save_cfg):
    if save_cfg.value:
        cfg = {
            "current_group": group.value,
            "min_power_high": float(min_ph.value),
            "trigger_seconds": 10,
        }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_bytes(msgspec.json.encode(cfg))
        msg = f"Wrote `{cfg_path.relative_to(ROOT)}`: `{cfg}`"
    elif cfg_path.exists():
        cfg = msgspec.json.decode(cfg_path.read_bytes())
        msg = f"Current config: `{cfg}`"
    else:
        msg = "No stream_config.json yet — click **Write** or run a stream once."
    mo.md(msg)
    return


@app.cell
def _(mo, pd, status_path):
    _blocks = [mo.md("## 2. Query status (`stream_status.json`)")]
    if status_path.exists():
        raw = status_path.read_text()
        try:
            obj = pd.read_json(status_path)
            # may be a scalar dict
            import json

            d = json.loads(raw)
            _blocks.append(mo.ui.table(pd.DataFrame([d])))
            if isinstance(d.get("lastProgress"), dict):
                prog = d["lastProgress"]
                flat = {
                    k: prog.get(k)
                    for k in (
                        "id",
                        "runId",
                        "batchId",
                        "numInputRows",
                        "processedRowsPerSecond",
                    )
                    if k in prog
                }
                if flat:
                    _blocks.append(mo.md("**lastProgress (subset)**"))
                    _blocks.append(mo.ui.table(pd.DataFrame([flat])))
        except Exception:
            _blocks.append(mo.md(f"```json\n{raw[:2000]}\n```"))
    else:
        _blocks.append(
            mo.md("*No status yet — run `python -m cli.neuro_tal_cli stream-file --once`.*")
        )
    mo.vstack(_blocks)
    return


@app.cell
def _(feat_path, mo, pd, plt):
    _blocks = [mo.md("## 3. Latest stream_features sink")]
    if feat_path.exists() and any(feat_path.rglob("*.parquet")):
        try:
            df = pd.read_parquet(feat_path)
        except Exception as e:
            df = None
            _blocks.append(mo.md(f"*Read error: {e}*"))
        if df is not None and len(df):
            _blocks.append(
                mo.md(
                    f"**Rows:** {len(df)} · datasets `{df['dataset'].unique().tolist() if 'dataset' in df.columns else []}`"
                )
            )
            show = [c for c in df.columns if c != "time_series_array"][:14]
            _blocks.append(mo.ui.table(df[show].head(25).round(4)))
            if "power_high" in df.columns and "group" in df.columns:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                for g, sub in df.groupby("group"):
                    ax.hist(sub["power_high"].dropna(), bins=12, alpha=0.55, label=str(g))
                ax.set_xlabel("power_high (stream sink)")
                ax.legend(fontsize=8)
                ax.set_title("Streaming feature sink by group")
                fig.tight_layout()
                _blocks.append(fig)
            if "cohort" in df.columns:
                _blocks.append(
                    mo.md(
                        "**Dynamic cohort tag counts:** "
                        + str(df["cohort"].value_counts().to_dict())
                    )
                )
    else:
        _blocks.append(mo.md("*Empty sink — seed inbox and run a micro-batch.*"))
    mo.vstack(_blocks)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## Honour notes

1. **Checkpoint** directory must be durable and stable across restarts.  
2. **foreachBatch** + SQL temp view = runtime dynamism without query restart.  
3. Change `stream_config.json` (or UI above) between batches to retarget `cohort`.  
4. Kafka needs the `spark-sql-kafka` package on the JVM; file source needs no extra jars.  
5. Consume `stream_features/` or god TFRecords with the same `tf.data` path as batch.

**CLI:** `python -m cli.neuro_tal_cli stream-file --once`
"""
    )
    return


if __name__ == "__main__":
    app.run()
