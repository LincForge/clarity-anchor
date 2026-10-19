"""Cheap in-situ density gauge model (spec §2 'Gauge')."""
from __future__ import annotations

import math

import numpy as np

from .atmosphere import SPECIES

SENSITIVITY = {"N2": 1.00, "O2": 1.00, "O": 0.80, "He": 0.18, "H": 0.45}
DRIFT_SIGMA_WEEK = 0.03
NOISE_REL = 0.02
NOISE_FLOOR = 1e-14  # kg/m^3
WEEK = 7 * 86400.0
TAU_B = 300.0  # round 3: wall-heating time constant of the sunlit offset term (assumption)
T_ZERO = 300.0  # round 3: shutter zero-read period (variant B), 1 s closed per period


def draw_k0(rng) -> float:
    return float(rng.uniform(0.7, 1.3))


def k_random_walk(t, k0: float, sigma_week: float, rng):
    """Multiplicative random walk: ln K diffuses with std sigma_week per sqrt(week)."""
    t = np.asarray(t, float)
    dt = np.diff(t)
    steps = rng.standard_normal(len(dt)) * sigma_week * np.sqrt(dt / WEEK)
    return k0 * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))


def composition_factor(mass_frac, sens: dict | None = None):
    """g(t) = sum_i s_i rho_i / sum_i rho_i, normalised to 1 at the first sample."""
    sens = SENSITIVITY if sens is None else sens
    s = np.array([sens[k] for k in SPECIES])
    mf = np.atleast_2d(mass_frac)
    raw = (mf @ s) / mf.sum(axis=1)
    return raw / raw[0]


def measure(rho_true, K, g, rng, noise: bool = True, rel=NOISE_REL, floor=NOISE_FLOOR, b=None):
    """rho_meas = K g rho_true + b + n,  n ~ N(0, (rel*rho_true + floor)^2).

    b (round 3) is the additive chamber background; b=None is the round-2 model. The noise draw is
    identical with or without b, so b = 0 reproduces round 2 bit for bit.
    """
    out = K * g * rho_true
    if b is not None:
        out = out + b
    if noise:
        out = out + rng.standard_normal(np.shape(rho_true)) * (rel * rho_true + floor)
    return out


def bin_mean(t, x, width: float = 30.0, skip=None, min_valid=None):
    """Average samples into contiguous bins of `width` seconds; returns bin centres and means (full bins only).

    skip (round 3): boolean mask of samples left out of the bin mean (shutter zero-read seconds). The bin
    time stamp stays the mean of all sample times in the bin.
    min_valid (round 3b): bins with fewer than min_valid non-skipped samples are gaps (NaN). None = round 3.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    n_per = int(round(width / (t[1] - t[0])))
    nb = len(t) // n_per
    tb = t[: nb * n_per].reshape(nb, n_per).mean(axis=1)
    xr = x[: nb * n_per].reshape(nb, n_per)
    if skip is None:
        xb = xr.mean(axis=1)
    else:
        keep = ~np.asarray(skip, bool)[: nb * n_per].reshape(nb, n_per)
        nk = keep.sum(axis=1)
        if min_valid is None:
            xb = np.sum(np.where(keep, xr, 0.0), axis=1) / nk
        else:
            ok = nk >= min_valid
            xb = np.full(nb, np.nan)
            xb[ok] = np.sum(np.where(keep, xr, 0.0), axis=1)[ok] / nk[ok]
    return tb, xb


# ---------------------------------------------------------------------------------------------
# Round 3: additive, sun-correlated chamber background and the shutter zero-read
# ---------------------------------------------------------------------------------------------
def sunlit_lag(t, nu, tau: float = TAU_B):
    """First-order lag s of the sunlit indicator nu (exact for a piecewise-constant input on a uniform grid).

    s[0] = nu[0] (the wall starts in equilibrium with the instantaneous illumination).
    """
    from scipy.signal import lfilter

    t = np.asarray(t, float)
    nu = np.asarray(nu, float)
    a = float(np.exp(-(t[1] - t[0]) / tau))
    s, _ = lfilter([1.0 - a], [1.0, -a], nu, zi=[a * nu[0]])
    return s


def offset(s, rho_ref: float, beta0: float, beta1: float):
    """b(t) = rho_ref * (beta0 + beta1 * s(t)). DbC (daemon zone): fail fast on hostile parameters."""
    for name, v in (("beta0", beta0), ("beta1", beta1)):
        if not (isinstance(v, (int, float, np.floating)) and math.isfinite(v) and v >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {v!r}")
    if not (isinstance(rho_ref, (int, float, np.floating)) and math.isfinite(rho_ref) and rho_ref > 0.0):
        raise ValueError(f"rho_ref must be finite and > 0, got {rho_ref!r}")
    s = np.asarray(s, float)
    if not np.all(np.isfinite(s)):
        raise ValueError("sunlit indicator s(t) contains non-finite values")
    return rho_ref * (beta0 + beta1 * s)


ZR_MODES = ("interp", "hold", "fit")  # round 3b: b_hat estimators (interp = round 3, non-causal reference)
MIN_VALID = 15  # round 3b: a 30 s product bin with fewer valid (non-shutter) samples is a gap (NaN)
FIT_MIN_SPREAD = 0.1  # round 3b: "fit" holds until the std of the read regressors s_model reaches this (rank guard)


def _check_n_zero(n_zero, t_zero):
    """DbC (daemon zone): n_zero must be an int (not bool) with 1 <= n_zero < t_zero."""
    if isinstance(n_zero, bool) or not isinstance(n_zero, (int, np.integer)) or not (1 <= n_zero < t_zero):
        raise ValueError(f"n_zero must be an int with 1 <= n_zero < t_zero ({t_zero!r}), got {n_zero!r}")
    return int(n_zero)


def zero_read_mask(t, t_zero: float = T_ZERO, n_zero: int = 1):
    """Shutter-closed seconds t in [k t_zero, k t_zero + n_zero), k >= 1 (none at t = 0).

    n_zero = 1 is the round 3 mask (t = t_zero, 2 t_zero, ...)."""
    n_zero = _check_n_zero(n_zero, t_zero)
    t = np.asarray(t, float)
    if n_zero == 1:
        k = np.round(t / t_zero)
        return (k >= 1) & (np.abs(t - k * t_zero) < 1e-6)
    k = np.floor((t + 1e-6) / t_zero)
    d = t - k * t_zero
    return (k >= 1) & (d > -1e-6) & (d < n_zero - 1e-6)


def _windows(t, t_zero, n_zero):
    """Per closed second: its window index k; plus the complete windows (all n_zero seconds inside the series)."""
    m = zero_read_mask(t, t_zero, n_zero)
    tm = np.asarray(t, float)[m]
    k = np.floor((tm + 1e-6) / t_zero).astype(np.int64)
    ks, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    return m, ks, inv, cnt


def zero_read(t, b, rng, t_zero: float = T_ZERO, noise: bool = True, rel=NOISE_REL, floor=NOISE_FLOOR,
              n_zero: int = 1, return_avail: bool = False):
    """One read per complete window: b_meas = mean over its n_zero closed seconds of b(t) + n_z(t),
    n_z ~ N(0, (rel*b(t) + floor)^2) drawn per second in time order. Time stamp = window centre,
    k t_zero + (n_zero - 1)/2. A window cut by the series end gives no read. n_zero = 1 is round 3 bit for bit.

    return_avail: also return the time each read becomes available, k t_zero + n_zero (after its last closed second).
    """
    n_zero = _check_n_zero(n_zero, t_zero)
    m, ks, inv, cnt = _windows(t, t_zero, n_zero)
    tm = np.asarray(t, float)[m]
    bm = np.asarray(b, float)[m]
    if noise:
        bm = bm + rng.standard_normal(len(bm)) * (rel * bm + floor)
    full = cnt == n_zero
    tz = (np.bincount(inv, weights=tm) / cnt)[full]
    bz = (np.bincount(inv, weights=bm) / cnt)[full]
    if return_avail:
        return tz, bz, ks[full] * t_zero + n_zero
    return tz, bz


def _bhat_causal(t, tz, bz, t_avail, mode, s_model, t_zero, n_zero):
    """Causal b_hat: 'hold' = last completed read (0 before the first); 'fit' = OLS of the reads on the window-mean
    s_model, refit on all completed reads, b_hat = c0 + c1 s_model(t); hold until two reads with regressor std >=
    FIT_MIN_SPREAD exist."""
    j = np.searchsorted(t_avail, t, side="right") - 1  # latest read available at t (-1: none yet)
    have = j >= 0
    jj = np.where(have, j, 0)
    if len(bz) == 0:
        return np.zeros_like(t)
    held = np.where(have, bz[jj], 0.0)
    if mode == "hold":
        return held
    m, ks, inv, cnt = _windows(t, t_zero, n_zero)
    x = (np.bincount(inv, weights=s_model[m]) / cnt)[cnt == n_zero]  # regressor: window mean of s_model
    n = np.arange(1, len(bz) + 1, dtype=float)
    mx, my = np.cumsum(x) / n, np.cumsum(bz) / n
    var = np.cumsum(x * x) / n - mx * mx
    cov = np.cumsum(x * bz) / n - mx * my
    ok = (n >= 2) & (np.sqrt(np.maximum(var, 0.0)) >= FIT_MIN_SPREAD)
    c1 = np.where(ok, cov / np.where(ok, var, 1.0), 0.0)
    c0 = my - c1 * mx
    fitted = c0[jj] + c1[jj] * s_model
    return np.where(have & ok[jj], fitted, held)


def subtract_zero_reads(t, meas, b, rng, t_zero: float = T_ZERO, noise: bool = True, n_zero: int = 1,
                        mode: str = "interp", s_model=None, return_bhat: bool = False):
    """Variant B: rho_corr = rho_meas - b_hat.

    mode "interp" (round 3): b_hat linear between reads (held outside them), non-causal.
    mode "hold" / "fit" (round 3b, causal): see _bhat_causal; "fit" needs s_model (finite, same length as t), the
    lagged sunlit indicator the flight software computes with an assumed lag.

    Returns (rho_corr, skip[, b_hat]). At the skipped (shutter-closed) seconds rho_corr is filled by linear
    interpolation of its neighbours so the anchor integral stays on the 1 Hz Simpson grid; the 30 s
    product leaves those samples out (bin_mean(..., skip=skip)). n_zero = 1, mode = "interp" is round 3 bit for bit.
    """
    if mode not in ZR_MODES:
        raise ValueError(f"mode must be one of {ZR_MODES}, got {mode!r}")
    t = np.asarray(t, float)
    if mode == "fit":
        if s_model is None:
            raise ValueError("mode 'fit' needs s_model (the flight-software lagged sunlit indicator)")
        s_model = np.asarray(s_model, float)
        if s_model.shape != t.shape or not np.all(np.isfinite(s_model)):
            raise ValueError("s_model must be finite and the same length as t")
    tz, bz, t_avail = zero_read(t, b, rng, t_zero, noise=noise, n_zero=n_zero, return_avail=True)
    skip = zero_read_mask(t, t_zero, n_zero)
    bhat = np.interp(t, tz, bz) if mode == "interp" else _bhat_causal(t, tz, bz, t_avail, mode, s_model, t_zero, n_zero)
    rho_corr = np.asarray(meas, float) - bhat
    ok = ~skip
    rho_corr[skip] = np.interp(t[skip], t[ok], rho_corr[ok])
    if return_bhat:
        return rho_corr, skip, bhat
    return rho_corr, skip
