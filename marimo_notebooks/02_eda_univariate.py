"""02 — Spectral Power (Welch PSD). Canonical marimo notebook."""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="02 — Spectral Power")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    from scipy.signal import welch, stft

    from helpers import (
        CONTROL_COLOR,
        HIGHLIGHT,
        MDD_COLOR,
        band_power,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        set_global_style,
    )

    set_global_style()
    return (
        CONTROL_COLOR,
        HIGHLIGHT,
        MDD_COLOR,
        band_power,
        book_nav,
        clinical_relevance_card,
        hypothesis_card,
        key_insight_card,
        make_synthetic_bold_dataset,
        mo,
        np,
        plt,
        px,
        stft,
        welch,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
# 02 — Spectral Power (Welch PSD)

**Chapter 2** · Univariate spectral biomarkers of anhedonia during positive music.
"""
    )
    return


@app.cell
def _(hypothesis_card, mo):
    mo.md(
        hypothesis_card(
            "MDD shows reduced high-frequency power to positive music — anhedonia biomarker.",
            "Controls have higher power in the target band during positive music; tones show little group difference.",
        )
    )
    return


@app.cell
def _(mo):
    n_subj = mo.ui.slider(6, 18, value=10, step=2, label="Subjects")
    tr = mo.ui.number(start=2.5, stop=3.5, value=3.0, step=0.25, label="TR (s)")
    band_low = mo.ui.slider(0.01, 0.12, value=0.03, step=0.01, label="Band low (Hz)")
    band_high = mo.ui.slider(0.06, 0.18, value=0.10, step=0.01, label="Band high (Hz)")
    nper = mo.ui.slider(16, 48, value=32, step=8, label="Welch nperseg")
    mo.md("## Reactive controls")
    mo.hstack([n_subj, tr, nper], justify="start")
    mo.hstack([band_low, band_high], justify="start")
    return band_high, band_low, n_subj, nper, tr


@app.cell
def _(
    CONTROL_COLOR,
    HIGHLIGHT,
    MDD_COLOR,
    band_high,
    band_low,
    band_power,
    make_synthetic_bold_dataset,
    mo,
    n_subj,
    nper,
    plt,
    tr,
    welch,
):
    synth = make_synthetic_bold_dataset(int(n_subj.value), 105, float(tr.value))
    fs = 1.0 / float(tr.value)

    def get_psd(grp, cond="positive_music"):
        sig = synth[(synth.group == grp) & (synth.trial_type == cond)]["bold"].values
        if len(sig) < 20:
            sig = synth[synth.group == grp]["bold"].values
        nperseg = min(int(nper.value), max(8, len(sig) // 2))
        f, pxx = welch(sig, fs=fs, nperseg=nperseg)
        return f, pxx

    f_c, pxx_c = get_psd("Control")
    f_m, pxx_m = get_psd("MDD")
    f_c_nm, pxx_c_nm = get_psd("Control", "tones")
    f_m_nm, pxx_m_nm = get_psd("MDD", "tones")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].semilogy(f_c, pxx_c, label="Control", color=CONTROL_COLOR, lw=2.5)
    axes[0].semilogy(f_m, pxx_m, label="MDD", color=MDD_COLOR, lw=2.5, ls="--")
    axes[0].axvspan(float(band_low.value), float(band_high.value), alpha=0.15, color=HIGHLIGHT)
    axes[0].set_title("Positive music")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (log)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(f_c_nm, pxx_c_nm, label="Control", color=CONTROL_COLOR, lw=2.5)
    axes[1].semilogy(f_m_nm, pxx_m_nm, label="MDD", color=MDD_COLOR, lw=2.5, ls="--")
    axes[1].axvspan(float(band_low.value), float(band_high.value), alpha=0.15, color=HIGHLIGHT)
    axes[1].set_title("Non-music (tones)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Welch PSD: stimulus-specific group difference", y=1.02)
    mo.output.append(fig)

    bp_c = band_power(f_c, pxx_c, float(band_low.value), float(band_high.value))
    bp_m = band_power(f_m, pxx_m, float(band_low.value), float(band_high.value))
    bp_c_nm = band_power(f_c_nm, pxx_c_nm, float(band_low.value), float(band_high.value))
    bp_m_nm = band_power(f_m_nm, pxx_m_nm, float(band_low.value), float(band_high.value))
    return bp_c, bp_c_nm, bp_m, bp_m_nm, synth


@app.cell
def _(band_high, band_low, bp_c, bp_c_nm, bp_m, bp_m_nm, key_insight_card, mo, px):
    mo.md(
        f"""
**Band power [{float(band_low.value):.2f}–{float(band_high.value):.2f} Hz]**

| Condition | Control | MDD |
|---|---:|---:|
| Positive music | {bp_c:.4f} | {bp_m:.4f} |
| Tones | {bp_c_nm:.4f} | {bp_m_nm:.4f} |
"""
    )
    mo.ui.plotly(
        px.bar(
            {
                "condition": ["music", "music", "tones", "tones"],
                "group": ["Control", "MDD", "Control", "MDD"],
                "band_power": [bp_c, bp_m, bp_c_nm, bp_m_nm],
            },
            x="condition",
            y="band_power",
            color="group",
            barmode="group",
            color_discrete_map={"Control": "#2E86AB", "MDD": "#C73E1D"},
            title="Band power by group × condition",
        )
    )
    ratio = bp_c / max(bp_m, 1e-12)
    mo.md(
        key_insight_card(
            "MDD shows reduced high-frequency power to positive music.",
            "Dissociation is stimulus-specific (little group difference on tones).",
            effect_size=f"~{ratio:.1f}× higher band power in Controls (music)",
        )
    )
    return


@app.cell
def _(mo, np, plt, stft, synth):
    sig = synth.groupby("time")["bold"].mean().values.astype("float64")
    _f, _t, Zxx = stft(sig, fs=1 / 3.0, nperseg=16)
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.imshow(np.abs(Zxx), aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title("SciPy STFT spectrogram")
    ax2.set_xlabel("Time frames")
    ax2.set_ylabel("Frequency bins")
    mo.output.append(fig2)
    return


@app.cell
def _(book_nav, clinical_relevance_card, mo):
    mo.md(
        clinical_relevance_card(
            "Spectral power during positive music is an objective anhedonia biomarker and a RecSys feature."
        )
    )
    mo.md(book_nav("02_eda_univariate"))
    return


if __name__ == "__main__":
    app.run()
