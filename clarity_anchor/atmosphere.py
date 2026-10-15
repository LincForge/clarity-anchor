"""Truth atmosphere (spec §2): NRLMSIS 2.0 x model-error field x storm bump.

pymsis cannot be called inside the RK4 loop efficiently, so the truth is a *tabulated field*:
MSIS is evaluated along a reference trajectory r_ref(t) (and 1 km above it, for the local scale
height H(t)); the truth density anywhere near that trajectory is

    rho(t, r) = rho_msis(t, r_ref(t)) * exp(-(|r| - |r_ref(t)|) / H(t)) * m(t) * storm(t).

The reference trajectory is itself a converged drag propagation (see scenarios.build_truth), so the
exponential correction only spans the small residual offset. Drag and gauge both sample this one
field, which is what the anchor relies on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymsis

from . import orbit

WGS84_F = 1.0 / 298.257223563
F107 = 150.0
F107A = 150.0
AP = 15.0
STORM_DAY = 5.0
STORM_WIDTH_S = 6 * 3600.0
STORM_AMP = 1.5

AMU = 1.66053906660e-27
SPECIES_MASS = {  # kg per particle
    "N2": 28.0134 * AMU,
    "O2": 31.9988 * AMU,
    "O": 15.9994 * AMU,
    "He": 4.002602 * AMU,
    "H": 1.00794 * AMU,
}
SPECIES = ("N2", "O2", "O", "He", "H")
_MSIS_COL = {"N2": pymsis.Variable.N2, "O2": pymsis.Variable.O2, "O": pymsis.Variable.O,
             "He": pymsis.Variable.HE, "H": pymsis.Variable.H}


def model_error(t, T_orb: float, phi1: float, phi2: float):
    t = np.asarray(t, float)
    return 1.0 + 0.15 * np.sin(2 * np.pi * t / T_orb + phi1) + 0.10 * np.sin(2 * np.pi * t / 86400.0 + phi2)


def storm(t, t_s: float = STORM_DAY * 86400.0):
    t = np.asarray(t, float)
    return 1.0 + STORM_AMP * np.exp(-(((t - t_s) / STORM_WIDTH_S) ** 2))


def gmst(t):
    """Greenwich mean sidereal angle (rad) at t seconds past epoch."""
    d = orbit.EPOCH_JD - 2451545.0
    th0 = math.radians((280.46061837 + 360.98564736629 * d) % 360.0)
    return th0 + orbit.OMEGA_E * np.asarray(t, float)


def eci_to_geodetic(t, r):
    """ECI -> geodetic lat (deg), lon (deg, [0,360)), alt (m) on WGS84, Earth rotation only."""
    r = np.atleast_2d(np.asarray(r, float))
    th = gmst(t)
    c, s = np.cos(th), np.sin(th)
    x = c * r[:, 0] + s * r[:, 1]
    y = -s * r[:, 0] + c * r[:, 1]
    z = r[:, 2]
    a = orbit.R_E
    f = WGS84_F
    b = a * (1 - f)
    e2 = f * (2 - f)
    ep2 = (a * a - b * b) / (b * b)
    p = np.hypot(x, y)
    lon = np.degrees(np.arctan2(y, x)) % 360.0
    # Bowring initial guess + 3 Newton-style refinements
    beta = np.arctan2(a * z, b * p)
    lat = np.arctan2(z + ep2 * b * np.sin(beta) ** 3, p - e2 * a * np.cos(beta) ** 3)
    for _ in range(3):
        beta = np.arctan2((1 - f) * np.sin(lat), np.cos(lat))
        lat = np.arctan2(z + ep2 * b * np.sin(beta) ** 3, p - e2 * a * np.cos(beta) ** 3)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    cl = np.cos(lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        alt = np.where(np.abs(cl) > 1e-10, p / cl - N, np.abs(z) - b)
    return np.degrees(lat), lon, alt


def msis_eval(t, r):
    """NRLMSIS 2.0 at ECI positions. Returns total mass density (kg/m^3) and species number densities (m^-3)."""
    t = np.atleast_1d(np.asarray(t, float))
    lat, lon, alt = eci_to_geodetic(t, r)
    n = len(t)
    dates = np.datetime64(orbit.EPOCH_ISO) + np.round(t * 1e3).astype("timedelta64[ms]")
    out = pymsis.calculate(dates, lon, lat, alt / 1e3,
                           f107s=np.full(n, F107), f107as=np.full(n, F107A),
                           aps=np.full((n, 7), AP), version=2.0)
    out = np.asarray(out).reshape(n, -1)
    rho = out[:, pymsis.Variable.MASS_DENSITY]
    species = {k: np.nan_to_num(out[:, _MSIS_COL[k]]) for k in SPECIES}
    return rho, species


@dataclass
class TruthField:
    t0: float
    dt: float
    log_rho: np.ndarray  # ln rho_msis along reference
    H: np.ndarray  # local scale height (m)
    rnorm: np.ndarray  # |r_ref|
    mass_frac: np.ndarray  # (N, 5) species mass fractions along reference
    T_orb: float
    phi1: float
    phi2: float

    @classmethod
    def from_reference(cls, t, r, T_orb: float, seed: int = 1):
        t = np.asarray(t, float)
        dt = float(t[1] - t[0])
        assert np.allclose(np.diff(t), dt), "reference grid must be uniform"
        rn = np.linalg.norm(r, axis=1)
        rho0, sp = msis_eval(t, r)
        rho1, _ = msis_eval(t, r * ((rn + 1000.0) / rn)[:, None])
        H = 1000.0 / np.log(rho0 / rho1)
        md = np.stack([sp[k] * SPECIES_MASS[k] for k in SPECIES], axis=1)
        frac = md / md.sum(axis=1, keepdims=True)
        rng = np.random.default_rng(seed)
        phi1, phi2 = rng.uniform(0, 2 * np.pi, 2)
        return cls(float(t[0]), dt, np.log(rho0), H, rn, frac, T_orb, float(phi1), float(phi2))

    # --- interpolation helpers -------------------------------------------------
    def _idx(self, t):
        u = (np.asarray(t, float) - self.t0) / self.dt
        i = np.clip(np.floor(u).astype(int), 0, len(self.H) - 2)
        return i, (u - i)

    def _interp(self, arr, t):
        i, w = self._idx(t)
        if arr.ndim == 1:
            return arr[i] * (1 - w) + arr[i + 1] * w
        return arr[i] * (1 - w)[:, None] + arr[i + 1] * w[:, None]

    def H_at(self, t):
        return self._interp(self.H, t)

    def multiplier(self, t):
        return model_error(t, self.T_orb, self.phi1, self.phi2) * storm(t)

    def density(self, t, r):
        t = np.atleast_1d(np.asarray(t, float))
        rn = np.linalg.norm(np.atleast_2d(r), axis=1)
        lr = self._interp(self.log_rho, t)
        H = self._interp(self.H, t)
        rr = self._interp(self.rnorm, t)
        return np.exp(lr - (rn - rr) / H) * self.multiplier(t)

    def mass_fractions(self, t):
        return self._interp(self.mass_frac, np.atleast_1d(np.asarray(t, float)))

    def species_number_densities(self, t, r):
        rho = self.density(t, r)
        f = self.mass_fractions(t)
        return {k: f[:, j] * rho / SPECIES_MASS[k] for j, k in enumerate(SPECIES)}

    def density_scalar(self, t, x, y, z):
        u = (t - self.t0) / self.dt
        i = int(u)
        n = len(self.H) - 2
        if i > n:
            i = n
        elif i < 0:
            i = 0
        w = u - i
        lr = self.log_rho[i] * (1 - w) + self.log_rho[i + 1] * w
        H = self.H[i] * (1 - w) + self.H[i + 1] * w
        rr = self.rnorm[i] * (1 - w) + self.rnorm[i + 1] * w
        rn = math.sqrt(x * x + y * y + z * z)
        m = (1.0 + 0.15 * math.sin(2 * math.pi * t / self.T_orb + self.phi1)
             + 0.10 * math.sin(2 * math.pi * t / 86400.0 + self.phi2))
        st = 1.0 + STORM_AMP * math.exp(-(((t - STORM_DAY * 86400.0) / STORM_WIDTH_S) ** 2))
        return math.exp(lr - (rn - rr) / H) * m * st
