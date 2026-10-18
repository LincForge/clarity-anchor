"""Orbit-anchored drag-work observable (spec §3).

Per arc:  dE_pod = E(t_end) - E(t_start)   (point mass + J2 energy, same model as truth)
          W_srp  = ∫ a_srp,model · v dt      (estimator's SRP model)
          I_pod  = -2 (dE_pod - W_srp) / B_model
          I_gauge= ∫ rho_meas · w dt,  w = |v_rel| (v_rel · v)
          z_inv  = I_pod / I_gauge  (measures y = 1/K_eff; round 2, reviewer-approved)
          z      = I_gauge / I_pod  (round-1 ratio, kept behind estimator="ratio")

The kernel w = |v_rel|(v_rel·v) is the exact drag power per unit (½Bρ); see DECISIONS.md for why this
replaces the spec's |v_rel|^2 (v_rel·v̂) notation. Both integrals use the identical kernel.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from . import orbit


def arc_indices(tp, arc_len_s: float):
    """Contiguous arcs on the POD grid; arc length rounded to the POD cadence. Returns (n,2) index pairs."""
    tp = np.asarray(tp, float)
    dt = tp[1] - tp[0]
    L = int(round(arc_len_s / dt))
    n = (len(tp) - 1) // L
    starts = np.arange(n) * L
    return np.stack([starts, starts + L], axis=1)


def srp_model_accel(t, r, cr_model: float):
    sun = orbit.sun_unit_array(t)
    nu = orbit.shadow_nu(r, sun)
    return -orbit.srp_coef(cr_model) * sun * nu[:, None]


def kernel(r, v, omega=orbit.OMEGA_E):
    vrel = v - np.cross(np.array([0.0, 0.0, omega]), r)
    return np.linalg.norm(vrel, axis=1) * np.sum(vrel * v, axis=1)


def compute(t1, rho_meas1, tp, rp, vp, arc_len_s, B_model, cr_model, j2=orbit.J2, srp=True,
            omega=orbit.OMEGA_E):
    """Anchor observables for all whole arcs. t1: 1 Hz gauge times (must cover the POD span)."""
    t1 = np.asarray(t1, float)
    arcs = arc_indices(tp, arc_len_s)
    E = orbit.energy(rp, vp, j2=j2)
    dE = E[arcs[:, 1]] - E[arcs[:, 0]]
    # 1 Hz kinematics from POD (cubic Hermite between 10 s epochs)
    t_lo, t_hi = tp[arcs[0, 0]], tp[arcs[-1, 1]]
    i0 = int(np.searchsorted(t1, t_lo))
    i1 = int(np.searchsorted(t1, t_hi))
    assert t1[i0] == t_lo and t1[i1] == t_hi, "gauge grid must contain arc endpoints"
    tq = t1[i0 : i1 + 1]
    r1, v1 = orbit.hermite(tp, rp, vp, tq)
    w = kernel(r1, v1, omega)
    pw_srp = np.sum(srp_model_accel(tq, r1, cr_model) * v1, axis=1) if srp else np.zeros(len(tq))
    f_g = rho_meas1[i0 : i1 + 1] * w
    dt1 = t1[1] - t1[0]
    L = int(round((tp[arcs[0, 1]] - tp[arcs[0, 0]]) / dt1))
    idx = (np.arange(len(arcs)) * L)[:, None] + np.arange(L + 1)[None, :]
    I_gauge = simpson(f_g[idx], dx=dt1, axis=1)
    W_srp = simpson(pw_srp[idx], dx=dt1, axis=1)
    W_drag = dE - W_srp
    I_pod = -2.0 * W_drag / B_model
    return {
        "t_start": tp[arcs[:, 0]],
        "t_end": tp[arcs[:, 1]],
        "dE": dE,
        "W_srp": W_srp,
        "I_pod": I_pod,
        "I_gauge": I_gauge,
        "z": I_gauge / I_pod,  # round-1 ratio (POD noise in the denominator -> biased by ~sigma^2)
        "z_inv": I_pod / I_gauge,  # round-2 measurement of y = 1/K_eff (POD noise in the numerator)
        "W_kernel": simpson(w[idx], dx=dt1, axis=1),  # round 3 (variant C): ∫w dt, the regressor of an additive term
    }


def monte_carlo_R(truth, Kg1, arc_len_s, grade, B_model, cr_model, n=200, seed=0, gauge_noise=True,
                  which="z_inv"):
    """Empirical std (and mean) of the measurement `which` ("z_inv" or "z") on the FIRST arc over n independent POD + gauge noise draws.

    Kg1: noiseless gauge gain K(t)*g(t) at 1 Hz (so the draw only varies the noise).
    """
    from . import gauge, pod

    rng = np.random.default_rng(seed)
    tp_all = truth.t[:: int(round(pod.CADENCE / (truth.t[1] - truth.t[0])))]
    L = int(round(arc_len_s / pod.CADENCE))
    k_pod = int(round(pod.CADENCE / (truth.t[1] - truth.t[0])))
    t_end = tp_all[L]
    n5 = L * k_pod + 1
    t5, s5 = truth.t[:n5], truth.states[:n5]
    n1 = int(round(t_end)) + 1
    t1, rho1, kg = truth.t1[:n1], truth.rho1[:n1], Kg1[:n1]
    zs = np.empty(n)
    for k in range(n):
        tp, rp, vp = pod.observe(t5, s5, grade, rng)
        meas = gauge.measure(rho1, kg, 1.0, rng, noise=gauge_noise)
        zs[k] = compute(t1, meas, tp, rp, vp, arc_len_s, B_model, cr_model)[which][0]
    return float(np.std(zs, ddof=1)), float(np.mean(zs))
