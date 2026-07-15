"""03 — Multivariate spectral coherence. Canonical marimo notebook."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="03 — Coherence")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import coherence
    from scipy.ndimage import gaussian_filter1d

    from helpers import (
        CONTROL_COLOR,
        MDD_COLOR,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        set_global_style,
        trapz_integral,
    )

    set_global_style()

    def simulate_network_ts(df, condition="music", coupling=0.7, seed=0):
        rng = np.random.default_rng(seed)
        base = df[df.condition == condition].groupby("time")["bold"].mean().values
        if len(base) < 8:
            base = df.groupby("time")["bold"].mean().values
        noise = rng.normal(0, 0.4, size=base.shape)
        aud = base + noise
        limb = coupling * base + (1 - coupling) * rng.normal(0, 1, size=base.shape)
        limb = gaussian_filter1d(limb, sigma=1.0)
        return aud, limb

    return (
        CONTROL_COLOR,
        MDD_COLOR,
        book_nav,
        clinical_relevance_card,
        coherence,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        mo,
        np,
        pd,
        plt,
        simulate_network_ts,
        trapz_integral,
    )


@app.cell
def _(mo):
    _title = mo.md(
        r"""
# 03 — Multivariate Coherence

**Chapter 3** · Auditory–limbic spectral coherence during music vs non-music.
"""
    )
    _title
    return


@app.cell
def _(hypothesis_card, mo):
    _hypo = mo.md(
        hypothesis_card(
            "Music increases auditory–limbic coherence in Controls; modulation is absent or reversed in MDD.",
            "Decoupling in MDD is specific to positive music.",
        )
    )
    _hypo
    return


@app.cell
def _(mo):
    focus_ui = mo.ui.dropdown(
        options=["music", "nonmusic"], value="music", label="Focus condition"
    )
    n_subj = mo.ui.slider(8, 20, value=12, step=2, label="Subjects")
    control_coupling = mo.ui.slider(
        0.3, 0.95, value=0.78, step=0.05, label="Control coupling"
    )
    mdd_coupling = mo.ui.slider(
        0.1, 0.8, value=0.35, step=0.05, label="MDD coupling (music)"
    )
    _controls = mo.vstack(
        [
            mo.md("## Reactive controls"),
            mo.hstack([focus_ui, n_subj], justify="start"),
            mo.hstack([control_coupling, mdd_coupling], justify="start"),
        ]
    )
    _controls
    return control_coupling, focus_ui, mdd_coupling, n_subj


@app.cell
def _(
    CONTROL_COLOR,
    MDD_COLOR,
    coherence,
    control_coupling,
    focus_ui,
    key_insight_card,
    make_synthetic_bold_dataset,
    mdd_coupling,
    mo,
    n_subj,
    np,
    plt,
    simulate_network_ts,
    trapz_integral,
):
    synth = make_synthetic_bold_dataset(int(n_subj.value), 105, 3.0)
    cond = focus_ui.value
    coup_c = float(control_coupling.value)
    coup_m = float(mdd_coupling.value) if cond == "music" else 0.55

    aud_c, limb_c = simulate_network_ts(
        synth[synth.group == "Control"], cond, coup_c, seed=1
    )
    aud_m, limb_m = simulate_network_ts(
        synth[synth.group == "MDD"], cond, coup_m, seed=2
    )

    f_c, coh_c = coherence(aud_c, limb_c, fs=1 / 3.0, nperseg=28)
    f_m, coh_m = coherence(aud_m, limb_m, fs=1 / 3.0, nperseg=28)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(f_c, coh_c, color=CONTROL_COLOR, lw=2.5, label="Control")
    ax.plot(f_m, coh_m, color=MDD_COLOR, lw=2.5, ls="--", label="MDD")
    ax.fill_between(f_c, 0, coh_c, alpha=0.12, color=CONTROL_COLOR)
    ax.axvspan(0.03, 0.10, alpha=0.1, color="#7B2CBF", label="Target band")
    ax.set_title(f"Spectral coherence (auditory–limbic) — {cond}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    def integ(f, c):
        mask = (f > 0.03) & (f < 0.10)
        return trapz_integral(c[mask], f[mask]) if np.any(mask) else 0.0

    band_c = integ(f_c, coh_c)
    band_m = integ(f_m, coh_m)

    def band_for(group, condition, coupling, seed):
        aud, limb = simulate_network_ts(
            synth[synth.group == group], condition, coupling, seed
        )
        f, c = coherence(aud, limb, fs=1 / 3.0, nperseg=28)
        return integ(f, c)

    vals = [
        band_for("Control", "music", coup_c, 1),
        band_for("MDD", "music", float(mdd_coupling.value), 2),
        band_for("Control", "nonmusic", 0.5, 3),
        band_for("MDD", "nonmusic", 0.52, 4),
    ]
    fig_bar, axb = plt.subplots(figsize=(7, 3.5))
    x = np.arange(2)
    w = 0.35
    axb.bar(x - w / 2, [vals[0], vals[2]], w, label="Control", color=CONTROL_COLOR)
    axb.bar(x + w / 2, [vals[1], vals[3]], w, label="MDD", color=MDD_COLOR)
    axb.set_xticks(x)
    axb.set_xticklabels(["music", "nonmusic"])
    axb.set_ylabel("Integrated coherence")
    axb.set_title("Auditory–limbic coherence by condition")
    axb.legend()
    axb.grid(True, axis="y", alpha=0.3)

    off = band_c if cond == "music" else 0.45
    conn = np.array([[1.0, off], [off, 1.0]])
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(conn, cmap="RdBu_r", vmin=0, vmax=1)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Auditory", "Limbic"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Auditory", "Limbic"])
    plt.colorbar(im, ax=ax2, label="Coherence")
    ax2.set_title(f"Coupling during {cond}")

    _panel = mo.vstack(
        [
            fig,
            mo.md(
                f"**Integrated coherence 0.03–0.10 Hz** — Control: **{band_c:.3f}** · MDD: **{band_m:.3f}**"
            ),
            fig_bar,
            fig2,
            mo.md(
                key_insight_card(
                    "Music increases auditory–limbic coherence in controls but not MDD.",
                    "Non-music produces comparable (low) coherence in both groups.",
                    effect_size=f"music band Δ ≈ {band_c - band_m:.3f}",
                )
            ),
        ]
    )
    _panel
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    _wrap = mo.vstack(
        [
            mo.md(
                clinical_relevance_card(
                    "Auditory–limbic decoupling during music is a candidate anhedonia mechanism and intervention selector."
                )
            ),
            mo.md(book_nav("03_eda_multivariate")),
        ]
    )
    _wrap
    return


if __name__ == "__main__":
    app.run()
