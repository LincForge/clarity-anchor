"""Cheap in-situ density gauge model (spec §2 'Gauge')."""
from __future__ import annotations

import numpy as np

from .atmosphere import SPECIES

SENSITIVITY = {"N2": 1.00, "O2": 1.00, "O": 0.80, "He": 0.18, "H": 0.45}
DRIFT_SIGMA_WEEK = 0.03
NOISE_REL = 0.02
NOISE_FLOOR = 1e-14  # kg/m^3
WEEK = 7 * 86400.0


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


def measure(rho_true, K, g, rng, noise: bool = True, rel=NOISE_REL, floor=NOISE_FLOOR):
    """rho_meas = K g rho_true + n,  n ~ N(0, (rel*rho_true + floor)^2)."""
    out = K * g * rho_true
    if noise:
        out = out + rng.standard_normal(np.shape(rho_true)) * (rel * rho_true + floor)
    return out


def bin_mean(t, x, width: float = 30.0):
    """Average samples into contiguous bins of `width` seconds; returns bin centres and means (full bins only)."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    n_per = int(round(width / (t[1] - t[0])))
    nb = len(t) // n_per
    tb = t[: nb * n_per].reshape(nb, n_per).mean(axis=1)
    xb = x[: nb * n_per].reshape(nb, n_per).mean(axis=1)
    return tb, xb
