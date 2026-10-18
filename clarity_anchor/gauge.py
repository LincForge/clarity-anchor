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


def bin_mean(t, x, width: float = 30.0, skip=None):
    """Average samples into contiguous bins of `width` seconds; returns bin centres and means (full bins only).

    skip (round 3): boolean mask of samples left out of the bin mean (shutter zero-read seconds). The bin
    time stamp stays the mean of all sample times in the bin.
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
        xb = np.sum(np.where(keep, xr, 0.0), axis=1) / keep.sum(axis=1)
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


def zero_read_mask(t, t_zero: float = T_ZERO):
    """Shutter-closed seconds: t = t_zero, 2 t_zero, ... (none at t = 0; b_hat is held before the first)."""
    t = np.asarray(t, float)
    k = np.round(t / t_zero)
    return (k >= 1) & (np.abs(t - k * t_zero) < 1e-6)


def zero_read(t, b, rng, t_zero: float = T_ZERO, noise: bool = True, rel=NOISE_REL, floor=NOISE_FLOOR):
    """b_meas = b(t_z) + n_z,  n_z ~ N(0, (rel*b(t_z) + floor)^2), at every zero-read second."""
    m = zero_read_mask(t, t_zero)
    tz = np.asarray(t, float)[m]
    bz = np.asarray(b, float)[m]
    if noise:
        bz = bz + rng.standard_normal(len(bz)) * (rel * bz + floor)
    return tz, bz


def subtract_zero_reads(t, meas, b, rng, t_zero: float = T_ZERO, noise: bool = True):
    """Variant B: rho_corr = rho_meas - b_hat, b_hat linear between zero-reads (held outside them).

    Returns (rho_corr, skip). At the skipped (shutter-closed) seconds rho_corr is filled by linear
    interpolation of its neighbours so the anchor integral stays on the 1 Hz Simpson grid; the 30 s
    product leaves those samples out (bin_mean(..., skip=skip)).
    """
    t = np.asarray(t, float)
    tz, bz = zero_read(t, b, rng, t_zero, noise=noise)
    skip = zero_read_mask(t, t_zero)
    rho_corr = np.asarray(meas, float) - np.interp(t, tz, bz)
    ok = ~skip
    rho_corr[skip] = np.interp(t[skip], t[ok], rho_corr[ok])
    return rho_corr, skip
