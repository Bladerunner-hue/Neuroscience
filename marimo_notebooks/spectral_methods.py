"""DPSS multitaper PSD + tSNR helpers for BOLD spectral QC.

WASM-safe: numpy + scipy only (no MNE / TensorFlow).

Methods
-------
uniform   : equal-weight average of K DPSS-tapered periodograms
adaptive  : Thomson-style iterative weights using DPSS eigenvalues
welch     : classic Hann Welch (baseline / comparison)

MNE adaptive multitaper lives in ``scripts/prepare_real_features.py``
(optional ``--psd mne``) because MNE is not shipped to GitHub Pages.
"""
from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
from scipy import signal

PsdMethod = Literal["welch", "uniform", "adaptive"]

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "low": (0.01, 0.04),
    "mid": (0.04, 0.08),
    "high": (0.08, 0.15),
}


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_tsnr(ts: np.ndarray, eps: float = 1e-12) -> float:
    """Temporal SNR: mean / std over time (higher = cleaner)."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0:
        return float("nan")
    return float(np.mean(ts) / (np.std(ts) + eps))


def multitaper_psd(
    x: np.ndarray,
    fs: float,
    nw: float = 3.5,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform multitaper PSD (equal-weight average of DPSS periodograms).

    Parameters
    ----------
    x : 1-D time series
    fs : sampling rate (Hz) = 1/TR for BOLD
    nw : time-bandwidth product (typical 2.5–4)
    k : number of tapers; default int(2*nw - 1)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        f, pxx = signal.periodogram(x, fs=fs)
        return f, np.maximum(pxx, 1e-20)
    if k is None:
        k = max(1, int(2 * nw - 1))
    k = min(k, n - 1)
    tapers = signal.windows.dpss(n, nw, Kmax=k)
    psds = []
    for tap in tapers:
        f, pxx = signal.periodogram(x * tap, fs=fs, window="boxcar")
        psds.append(pxx)
    return f, np.maximum(np.mean(np.asarray(psds), axis=0), 1e-20)


def adaptive_multitaper_psd(
    x: np.ndarray,
    fs: float,
    nw: float = 3.5,
    k: int | None = None,
    max_iter: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thomson-style adaptive multitaper (numpy sketch of MNE adaptive=True).

    Returns
    -------
    f, S, weights
        weights shape (K, n_freq) after last iteration.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        f, pxx = multitaper_psd(x, fs, nw=nw, k=k)
        return f, pxx, np.ones((1, len(pxx)))
    if k is None:
        k = max(1, int(2 * nw - 1))
    k = min(k, n - 1)
    tapers, eigvals = signal.windows.dpss(n, nw, Kmax=k, return_ratios=True)
    periodograms = []
    f = None
    for tap in tapers:
        f, pxx = signal.periodogram(x * tap, fs=fs, window="boxcar")
        periodograms.append(pxx)
    periodograms = np.maximum(np.asarray(periodograms), 1e-20)
    eig = np.asarray(eigvals, dtype=np.float64)[:, None]
    S = np.mean(periodograms, axis=0)
    noise_floor = float(np.median(S))
    weights = np.ones_like(periodograms) / periodograms.shape[0]
    for _ in range(max_iter):
        # w_k(f) ∝ λ_k² / (λ_k² S(f) + (1-λ_k) N̂)
        den = eig**2 * S[None, :] + (1.0 - eig) * noise_floor
        w = (eig**2) / np.maximum(den, 1e-30)
        w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-30)
        S = np.sum(w * periodograms, axis=0)
        weights = w
        noise_floor = float(np.median(S))
    return f, np.maximum(S, 1e-20), weights


def welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if nperseg is None:
        nperseg = min(32, max(8, n // 3))
    nperseg = min(nperseg, n)
    nover = nperseg // 2
    f, pxx = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nover,
        detrend="constant",
        scaling="density",
    )
    return f, np.maximum(pxx, 1e-20)


def compute_psd(
    x: np.ndarray,
    fs: float,
    method: PsdMethod = "adaptive",
    nw: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch PSD estimator."""
    method = (method or "adaptive").lower()  # type: ignore[assignment]
    if method in ("uniform", "multitaper"):
        return multitaper_psd(x, fs, nw=nw)
    if method == "adaptive":
        f, S, _ = adaptive_multitaper_psd(x, fs, nw=nw)
        return f, S
    if method == "welch":
        return welch_psd(x, fs)
    raise ValueError(f"Unknown PSD method: {method!r}")


def _quality_from_psd(
    f: np.ndarray,
    pxx: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    pxx = np.maximum(np.asarray(pxx, dtype=np.float64), 1e-20)
    f = np.asarray(f, dtype=np.float64)
    total = trapz(pxx, f) + 1e-12
    centroid = trapz(f * pxx, f) / total
    # Wiener flatness
    geom = float(np.exp(np.mean(np.log(pxx))))
    arith = float(np.mean(pxx))
    flatness = geom / (arith + 1e-20)
    # Normalized Shannon entropy
    p_norm = pxx / (pxx.sum() + 1e-20)
    entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-20)) / np.log(len(p_norm)))
    out: dict[str, Any] = {
        "total_power": float(total),
        "spectral_centroid": float(centroid),
        "spectral_flatness": float(flatness),
        "spectral_entropy": float(entropy),
        "psd_f": f.tolist(),
        "psd_pxx": pxx.tolist(),
    }
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        out[f"power_{name}"] = (
            trapz(pxx[m], f[m]) / total if np.any(m) else 0.0
        )
    m_hi = (f >= 0.08) & (f <= 0.15)
    m_out = ~m_hi
    p_hi = float(np.mean(pxx[m_hi])) if np.any(m_hi) else 0.0
    p_out = float(np.median(pxx[m_out])) if np.any(m_out) else 1e-12
    out["band_snr_high"] = p_hi / (p_out + 1e-12)
    return out


def spectral_feats(
    ts: np.ndarray,
    tr: float = 3.0,
    method: PsdMethod = "adaptive",
    nw: float = 3.5,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Band powers + QC metrics from a BOLD time series.

    Drop-in upgrade for Welch-only ``spectral_feats`` in prepare_real_features.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    fs = 1.0 / float(tr)
    f, pxx = compute_psd(ts, fs, method=method, nw=nw)
    out = _quality_from_psd(f, pxx, bands)
    out["psd_method"] = method
    out["tsnr"] = compute_tsnr(ts)
    return out


def spectral_feats_multitaper(
    ts: np.ndarray,
    tr: float = 3.0,
    bands: dict[str, tuple[float, float]] | None = None,
    nw: float = 3.5,
    adaptive: bool = False,
) -> dict[str, Any]:
    """Backward-compatible wrapper used by the QC notebook."""
    method: PsdMethod = "adaptive" if adaptive else "uniform"
    return spectral_feats(ts, tr=tr, method=method, nw=nw, bands=bands)


def compare_psd_methods(
    ts: np.ndarray,
    tr: float = 3.0,
    nw: float = 3.5,
) -> dict[str, dict[str, Any]]:
    """Run welch / uniform / adaptive on the same series (for notebook demos)."""
    return {
        m: spectral_feats(ts, tr=tr, method=m, nw=nw)  # type: ignore[arg-type]
        for m in ("welch", "uniform", "adaptive")
    }


def feats_to_json_row(feats: dict[str, Any]) -> dict[str, Any]:
    """Serialize list-valued PSD arrays for CSV storage."""
    row = dict(feats)
    if isinstance(row.get("psd_f"), list):
        row["psd_f"] = json.dumps(row["psd_f"])
    if isinstance(row.get("psd_pxx"), list):
        row["psd_pxx"] = json.dumps(row["psd_pxx"])
    return row
