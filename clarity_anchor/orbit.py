"""Truth orbit propagator (spec §2): ECI, RK4, point mass + J2 + drag + cannonball SRP.

Hot loop is written with plain Python floats (much faster than tiny numpy arrays).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MU = 3.986004418e14  # m^3/s^2
R_E = 6378137.0  # m (WGS84 equatorial radius, also J2 reference radius)
J2 = 1.08262668e-3
OMEGA_E = 7.2921150e-5  # rad/s
AU = 1.495978707e11
P_SUN = 4.56e-6  # Pa
C_R = 1.3
AREA = 1.0  # m^2
MASS = 150.0  # kg

# Simulation epoch: 2024-03-20 12:00 UTC (Julian date), see DECISIONS.md
EPOCH_JD = 2460390.0
EPOCH_ISO = "2024-03-20T12:00:00"


def cd_of_h(h_m: float) -> float:
    """C_D(h) = 2.2 + 0.4 (h - 300 km)/300 km (spec §2)."""
    return 2.2 + 0.4 * (h_m - 300e3) / 300e3


def b_true(h_m: float) -> float:
    return cd_of_h(h_m) * AREA / MASS


def srp_coef(cr: float = C_R) -> float:
    """|a_srp| in sunlight = P_sun * C_R * A/m."""
    return P_SUN * cr * AREA / MASS


def sun_unit(t: float) -> tuple[float, float, float]:
    """Low-precision solar direction in ECI (Astronomical Almanac), t in s past epoch."""
    n = (EPOCH_JD - 2451545.0) + t / 86400.0
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 4.0e-7 * n)
    return (math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam))


def sun_unit_array(t: np.ndarray) -> np.ndarray:
    n = (EPOCH_JD - 2451545.0) + np.asarray(t, dtype=float) / 86400.0
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4.0e-7 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def shadow_nu(r: np.ndarray, sun: np.ndarray) -> np.ndarray:
    """Cylindrical shadow: 0 if behind Earth within the R_E cylinder, else 1. r (N,3), sun (3,) or (N,3)."""
    r = np.atleast_2d(r)
    sun = np.broadcast_to(sun, r.shape)
    proj = np.sum(r * sun, axis=1)
    perp2 = np.sum(r * r, axis=1) - proj**2
    return np.where((proj < 0) & (perp2 < R_E**2), 0.0, 1.0)


@dataclass
class ForceModel:
    mu: float = MU
    j2: float = J2
    B: float = 0.0  # C_D A / m  (m^2/kg); 0 disables drag
    omega_atm: float = OMEGA_E  # atmosphere co-rotation rate
    density: Callable[[float, float, float, float], float] | None = None  # (t, x, y, z) -> kg/m^3
    srp: bool = True
    cr: float = C_R
    extra: dict = field(default_factory=dict)

    def make_accel(self):
        mu, j2, B, w = self.mu, self.j2, self.B, self.omega_atm
        dens = self.density
        drag_on = B > 0.0 and dens is not None
        srp_on = self.srp
        ksrp = srp_coef(self.cr)
        re2 = R_E * R_E
        rcyl2 = R_E * R_E
        sqrt = math.sqrt

        def accel(t, x, y, z, vx, vy, vz):
            r2 = x * x + y * y + z * z
            r = sqrt(r2)
            mr3 = mu / (r2 * r)
            if j2 != 0.0:
                z2r2 = z * z / r2
                f = 1.5 * j2 * re2 / r2
                cxy = 1.0 + f * (1.0 - 5.0 * z2r2)
                cz = 1.0 + f * (3.0 - 5.0 * z2r2)
                ax = -mr3 * x * cxy
                ay = -mr3 * y * cxy
                az = -mr3 * z * cz
            else:
                ax = -mr3 * x
                ay = -mr3 * y
                az = -mr3 * z
            if drag_on:
                rho = dens(t, x, y, z)
                vrx = vx + w * y
                vry = vy - w * x
                vrz = vz
                vr = sqrt(vrx * vrx + vry * vry + vrz * vrz)
                k = -0.5 * B * rho * vr
                ax += k * vrx
                ay += k * vry
                az += k * vrz
            if srp_on:
                sx, sy, sz = sun_unit(t)
                proj = x * sx + y * sy + z * sz
                if not (proj < 0.0 and (r2 - proj * proj) < rcyl2):
                    ax -= ksrp * sx
                    ay -= ksrp * sy
                    az -= ksrp * sz
            return ax, ay, az

        return accel


def propagate(state0, t_end: float, dt: float, fm: ForceModel, t0: float = 0.0):
    """Fixed-step RK4. Returns t (N,), states (N,6) including t0."""
    acc = fm.make_accel()
    n = int(round((t_end - t0) / dt))
    out = np.empty((n + 1, 6))
    x, y, z, vx, vy, vz = (float(v) for v in state0)
    out[0] = (x, y, z, vx, vy, vz)
    h = dt
    h2 = 0.5 * dt
    h6 = dt / 6.0
    for i in range(n):
        t = t0 + i * dt
        a1 = acc(t, x, y, z, vx, vy, vz)
        x2, y2, z2 = x + h2 * vx, y + h2 * vy, z + h2 * vz
        vx2, vy2, vz2 = vx + h2 * a1[0], vy + h2 * a1[1], vz + h2 * a1[2]
        a2 = acc(t + h2, x2, y2, z2, vx2, vy2, vz2)
        x3, y3, z3 = x + h2 * vx2, y + h2 * vy2, z + h2 * vz2
        vx3, vy3, vz3 = vx + h2 * a2[0], vy + h2 * a2[1], vz + h2 * a2[2]
        a3 = acc(t + h2, x3, y3, z3, vx3, vy3, vz3)
        x4, y4, z4 = x + h * vx3, y + h * vy3, z + h * vz3
        vx4, vy4, vz4 = vx + h * a3[0], vy + h * a3[1], vz + h * a3[2]
        a4 = acc(t + h, x4, y4, z4, vx4, vy4, vz4)
        x += h6 * (vx + 2 * vx2 + 2 * vx3 + vx4)
        y += h6 * (vy + 2 * vy2 + 2 * vy3 + vy4)
        z += h6 * (vz + 2 * vz2 + 2 * vz3 + vz4)
        vx += h6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        vy += h6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        vz += h6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        out[i + 1] = (x, y, z, vx, vy, vz)
    t = t0 + dt * np.arange(n + 1)
    return t, out


def potential(r: np.ndarray, j2: float = J2, mu: float = MU) -> np.ndarray:
    """U = -mu/r + mu J2 R^2 (3 z^2/r^2 - 1)/(2 r^3)."""
    r = np.atleast_2d(r)
    rn = np.linalg.norm(r, axis=1)
    z = r[:, 2]
    return -mu / rn + mu * j2 * R_E**2 * (3 * z**2 / rn**2 - 1) / (2 * rn**3)


def energy(r: np.ndarray, v: np.ndarray, j2: float = J2, mu: float = MU) -> np.ndarray:
    v = np.atleast_2d(v)
    return 0.5 * np.sum(v * v, axis=1) + potential(r, j2, mu)


def sma(r: np.ndarray, v: np.ndarray, mu: float = MU) -> np.ndarray:
    """Osculating Keplerian semi-major axis."""
    rn = np.linalg.norm(np.atleast_2d(r), axis=1)
    v2 = np.sum(np.atleast_2d(v) ** 2, axis=1)
    return 1.0 / (2.0 / rn - v2 / mu)


def hermite(tk: np.ndarray, rk: np.ndarray, vk: np.ndarray, tq: np.ndarray):
    """Cubic Hermite interpolation of position (and its derivative for velocity) on knots tk."""
    tk = np.asarray(tk, float)
    tq = np.asarray(tq, float)
    idx = np.clip(np.searchsorted(tk, tq, side="right") - 1, 0, len(tk) - 2)
    h = (tk[idx + 1] - tk[idx])[:, None]
    s = ((tq - tk[idx])[:, None]) / h
    p0, p1 = rk[idx], rk[idx + 1]
    m0, m1 = vk[idx] * h, vk[idx + 1] * h
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    r = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    d00 = 6 * s2 - 6 * s
    d10 = 3 * s2 - 4 * s + 1
    d01 = -6 * s2 + 6 * s
    d11 = 3 * s2 - 2 * s
    v = (d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1) / h
    return r, v


def circular_state(h_m: float, inc_deg: float = 87.0, raan_deg: float = 0.0) -> np.ndarray:
    """Circular orbit initial state at the ascending node."""
    a = R_E + h_m
    v = math.sqrt(MU / a)
    i = math.radians(inc_deg)
    O = math.radians(raan_deg)
    r = np.array([a * math.cos(O), a * math.sin(O), 0.0])
    vv = np.array([-v * math.cos(i) * math.sin(O), v * math.cos(i) * math.cos(O), v * math.sin(i)])
    return np.concatenate([r, vv])


def kepler_period(h_m: float) -> float:
    a = R_E + h_m
    return 2 * math.pi * math.sqrt(a**3 / MU)
