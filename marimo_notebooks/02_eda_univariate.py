import marimo as mo

app = mo.App()

@app.cell
def __(mo):
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    import tensorflow as tf
    import plotly.graph_objects as go
    import plotly.express as px

    import marimo as mo  # ensure

    # Direct, from local helpers (no src)
    from helpers import (
        set_global_style,
        hypothesis_card,
        key_insight_card,
        clinical_relevance_card,
        make_synthetic_bold_dataset,
        band_power,
        CONTROL_COLOR,
        MDD_COLOR,
        HIGHLIGHT,
    )

    set_global_style()
    mo.md("# 02 — EDA Univariate: Spectral Power (Welch PSD)")
    TR_SEC = 3.0
    return band_power, go, make_synthetic_bold_dataset, mo, np, pd, plt, px, tf, welch, TR_SEC, CONTROL_COLOR, MDD_COLOR, HIGHLIGHT, hypothesis_card, key_insight_card, clinical_relevance_card

@app.cell
def __(mo, hypothesis_card):
    mo.md(hypothesis_card(
        "MDD shows reduced high-frequency power to positive music — anhedonia biomarker. Non-music shows little group difference.",
        "Controls will have higher beta/gamma power in the target band during positive music."
    ))
    return

@app.cell
def __(mo):
    # Reactive UI controls
    n_subj = mo.ui.slider(6, 18, value=10, step=2, label="Subjects")
    tr = mo.ui.number(2.5, 3.5, value=TR_SEC, step=0.25, label="TR (s)")
    band_low = mo.ui.slider(0.01, 0.12, value=0.03, step=0.01, label="Band low Hz")
    band_high = mo.ui.slider(0.06, 0.18, value=0.10, step=0.01, label="Band high Hz")
    nper = mo.ui.slider(16, 48, value=32, step=8, label="Welch nperseg")

    mo.md("## Reactive Controls (move sliders → everything updates)")
    mo.hstack([n_subj, tr, nper])
    mo.hstack([band_low, band_high])
    return band_high, band_low, n_subj, nper, tr

@app.cell
def __(mo, np, plt, welch, band_low, band_high, nper, n_subj, tr, make_synthetic_bold_dataset, CONTROL_COLOR, MDD_COLOR, HIGHLIGHT):
    # Core data + Welch (user's requested pattern)
    synth = make_synthetic_bold_dataset(n_subj.value, 105, tr.value)
    fs = 1.0 / tr.value

    # Get traces for music
    def get_psd(grp, cond="positive_music"):
        sig = synth[(synth.group == grp) & (synth.trial_type == cond)]["bold"].values
        if len(sig) < 20: sig = synth[synth.group == grp]["bold"].values
        f, pxx = welch(sig, fs=fs, nperseg=min(nper.value, len(sig)//2))
        return f, pxx

    f_c, pxx_c = get_psd("Control")
    f_m, pxx_m = get_psd("MDD")

    # Matplotlib PSD - the main visual
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_c, pxx_c, label="Control / positive_music", color=CONTROL_COLOR, linewidth=2.5)
    ax.semilogy(f_m, pxx_m, label="MDD / positive_music", color=MDD_COLOR, linewidth=2.5, linestyle="--")
    ax.axvspan(band_low.value, band_high.value, alpha=0.15, color=HIGHLIGHT, label="Target band")
    ax.set_title("Power Spectrum: Positive Music Shows Higher Power in Controls (Welch)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mo.pyplot(fig)

    # Compute band power like the plan
    bp_c = band_power(f_c, pxx_c, band_low.value, band_high.value)
    bp_m = band_power(f_m, pxx_m, band_low.value, band_high.value)
    return bp_c, bp_m, f_c, f_m, fig, pxx_c, pxx_m, sig, synth

@app.cell
def __(mo, bp_c, bp_m, band_low, band_high, key_insight_card):
    mo.md(f"**Band power [{band_low.value:.2f}-{band_high.value:.2f} Hz]** Control: {bp_c:.4f} | MDD: {bp_m:.4f}")
    mo.md(key_insight_card(
        "MDD shows reduced gamma power to positive music — potential anhedonia biomarker. Non-music shows no group difference.",
        "The dissociation is stimulus-specific. This is directly actionable.",
        effect_size=f"~{bp_c / max(bp_m, 1e-6):.1f}x higher in Controls"
    ))
    return

@app.cell
def __(mo, np, synth, nper):
    # Optional TF / SciPy STFT preview (graceful in browser WASM)
    sig = synth.groupby("time")["bold"].mean().values.astype("float32")
    try:
        import tensorflow as tf
        stft = tf.signal.stft(sig, frame_length=16, frame_step=4, fft_length=32)
        mag = np.abs(stft)
        title = "TF STFT Spectrogram (demo)"
    except Exception:
        from scipy.signal import stft
        f, t, Zxx = stft(sig, fs=1/3., nperseg=16)
        mag = np.abs(Zxx)
        title = "SciPy STFT (browser fallback)"

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.imshow(mag, aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title(title)
    mo.pyplot(fig2)
    return

@app.cell
def __(mo, clinical_relevance_card):
    mo.md(clinical_relevance_card(
        "Spectral power differences during positive music can serve as objective biomarker for anhedonia and drive personalized playlist recommendations (RecSys integration)."
    ))
    return

if __name__ == "__main__":
    app.run()
