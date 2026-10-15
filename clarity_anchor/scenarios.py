"""Truth construction and scenario runners (spec §5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import atmosphere as atm
from . import orbit

CACHE = Path(__file__).resolve().parent.parent / "cache"
DT_TRUTH = 5.0
DT_TAB = 10.0
INC_DEG = 87.0


@dataclass
class Truth:
    h_km: float
    days: float
    t: np.ndarray  # 5 s
    states: np.ndarray  # (N,6)
    field: atm.TruthField
    B_true: float
    T_orb: float
    max_ref_offset_m: float
    t1: np.ndarray  # 1 Hz
    rho1: np.ndarray  # truth density at 1 Hz along the truth orbit
    frac1: np.ndarray  # species mass fractions at 1 Hz (N,5)


def _field_from(t, s, T_orb):
    step = max(1, int(round(DT_TAB / (t[1] - t[0]))))
    return atm.TruthField.from_reference(t[::step], s[::step, :3], T_orb=T_orb, seed=1)


def build_truth(h_km: float, days: float = 14.0, iterations: int = 2, use_cache: bool = True,
                srp: bool = True, dt: float = DT_TRUTH) -> Truth:
    """Propagate the truth orbit with a self-consistent tabulated MSIS field.

    pass 0: drag-free reference -> field_0; pass k: drag with field_{k-1} -> traj_k -> field_k.
    The final truth is propagated with field_n whose reference is traj_n (converged to << H).
    """
    path = CACHE / f"truth_h{int(h_km)}_d{days:g}_srp{int(srp)}_dt{dt:g}.npz"
    h = h_km * 1e3
    T_orb = orbit.kepler_period(h)
    B = orbit.b_true(h)
    s0 = orbit.circular_state(h, INC_DEG)
    t_end = days * 86400.0
    if use_cache and path.exists():
        z = np.load(path)
        t, s = z["t"], z["s"]
        field = atm.TruthField(float(z["t0"]), float(z["dt"]), z["log_rho"], z["H"], z["rnorm"],
                               z["mass_frac"], float(z["T_orb"]), float(z["phi1"]), float(z["phi2"]))
        off = float(z["off"])
    else:
        t, s = orbit.propagate(s0, t_end, dt, orbit.ForceModel(B=0.0, srp=srp))
        field = _field_from(t, s, T_orb)
        for _ in range(iterations):
            fm = orbit.ForceModel(B=B, density=field.density_scalar, srp=srp)
            t, s = orbit.propagate(s0, t_end, dt, fm)
            field = _field_from(t, s, T_orb)
        fm = orbit.ForceModel(B=B, density=field.density_scalar, srp=srp)
        t, s = orbit.propagate(s0, t_end, dt, fm)
        stp = max(1, int(round(DT_TAB / dt)))
        off = float(np.max(np.abs(np.linalg.norm(s[::stp, :3], axis=1) - field.rnorm)))
        if use_cache:
            CACHE.mkdir(exist_ok=True)
            np.savez(path, t=t, s=s, t0=field.t0, dt=field.dt, log_rho=field.log_rho, H=field.H,
                     rnorm=field.rnorm, mass_frac=field.mass_frac, T_orb=field.T_orb, phi1=field.phi1,
                     phi2=field.phi2, off=off)
    t1 = np.arange(0.0, t_end + 0.5, 1.0)
    r1, _ = orbit.hermite(t, s[:, :3], s[:, 3:], t1)
    rho1 = field.density(t1, r1)
    frac1 = field.mass_fractions(t1)
    return Truth(h_km, days, t, s, field, B, T_orb, off, t1, rho1, frac1)
