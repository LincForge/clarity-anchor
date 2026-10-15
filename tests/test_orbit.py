"""Analytic tests for the truth propagator (spec §7 tests (a) and (d))."""
import math

import numpy as np

from clarity_anchor import orbit


def _circular_state(h_m, inc_deg=87.0, mu=orbit.MU):
    a = orbit.R_E + h_m
    v = math.sqrt(mu / a)
    i = math.radians(inc_deg)
    return np.array([a, 0.0, 0.0, 0.0, v * math.cos(i), v * math.sin(i)])


def test_d_j2_energy_conserved_14_days():
    """(d) drag and SRP off: J2-inclusive specific energy conserved over 14 d at 5 s."""
    fm = orbit.ForceModel(j2=orbit.J2, B=0.0, srp=False)
    s0 = _circular_state(400e3)
    t, s = orbit.propagate(s0, 14 * 86400.0, 5.0, fm)
    E = orbit.energy(s[:, :3], s[:, 3:], j2=orbit.J2)
    rel = np.max(np.abs(E - E[0])) / abs(E[0])
    print(f"max relative J2 energy error over 14 d: {rel:.3e}")
    assert rel < 1e-9


def test_a_constant_density_decay_rate():
    """(a) constant rho, circular, no J2, non-rotating atmosphere: mean da/dt = -B rho sqrt(mu a)."""
    rho = 1e-11
    B = 2.2 * 1.0 / 150.0
    fm = orbit.ForceModel(j2=0.0, B=B, srp=False, omega_atm=0.0, density=lambda t, x, y, z: rho)
    s0 = _circular_state(400e3)
    a0 = orbit.R_E + 400e3
    T = 2 * math.pi * math.sqrt(a0**3 / orbit.MU)
    t, s = orbit.propagate(s0, 10 * T, 5.0, fm)
    a = orbit.sma(s[:, :3], s[:, 3:])
    slope = np.polyfit(t, a, 1)[0]
    expected = -B * rho * math.sqrt(orbit.MU * np.mean(a))
    print(f"da/dt sim {slope:.6e} analytic {expected:.6e} rel {(slope/expected-1):.3e}")
    assert abs(slope / expected - 1) < 0.01


def test_hermite_interpolation_reproduces_knots_and_is_accurate():
    fm = orbit.ForceModel(j2=orbit.J2, B=0.0, srp=False)
    s0 = _circular_state(400e3)
    t, s = orbit.propagate(s0, 600.0, 1.0, fm)
    tk, sk = t[::10], s[::10]
    tq = t
    r, v = orbit.hermite(tk, sk[:, :3], sk[:, 3:], tq)
    assert np.max(np.abs(r - s[:, :3])) < 1e-2  # cm-level over 10 s knots
    assert np.max(np.abs(v - s[:, 3:])) < 2e-4  # ~1e-8 relative


def test_shadow_cylindrical():
    sun = np.array([1.0, 0.0, 0.0])
    assert orbit.shadow_nu(np.array([[-7e6, 0.0, 0.0]]), sun)[0] == 0.0
    assert orbit.shadow_nu(np.array([[7e6, 0.0, 0.0]]), sun)[0] == 1.0
    assert orbit.shadow_nu(np.array([[-7e6, 6.5e6, 0.0]]), sun)[0] == 1.0
