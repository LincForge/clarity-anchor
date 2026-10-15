"""Benchmark metrics (spec §5): accuracy, precision, resolution, convergence time."""
from __future__ import annotations

import math

import numpy as np

ACC_LIMIT = 0.20
PREC_LIMIT = 0.10
RES_LIMIT_S = 30.0
CONV_TOL = 0.05


def convergence_time(t, e, tol=CONV_TOL):
    """Earliest time after which |e| < tol for every remaining sample; NaN if the last sample fails."""
    t = np.asarray(t, float)
    bad = np.nonzero(np.abs(np.asarray(e, float)) >= tol)[0]
    if len(bad) == 0:
        return float(t[0])
    if bad[-1] == len(t) - 1:
        return float("nan")
    return float(t[bad[-1] + 1])


def acc_prec(t, e, t_from):
    m = np.asarray(t) >= t_from
    em = np.asarray(e)[m]
    if len(em) < 2:
        return float("nan"), float("nan"), int(len(em))
    return float(abs(np.mean(em))), float(np.std(em)), int(len(em))


def verdict(value, limit):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNMEASURED"
    return "PASS" if value <= limit else "FAIL"


def decay_signal(truth, seed=0):
    """Observability of the anchor per orbit: median true drag work |W_drag| over 1-orbit arcs (J/kg),
    the equivalent semi-major-axis decay da = 2 a^2 |W| / mu (m), and the arc ENDPOINT energy noise
    sigma(dE) = sqrt(2) * std(E_pod - E_true) per POD grade (J/kg), measured from POD noise draws."""
    from . import anchor, orbit, pod

    rng = np.random.default_rng(seed)
    tp, rp, vp = pod.observe(truth.t, truth.states, "perfect", rng)
    res = anchor.compute(truth.t1, truth.rho1, tp, rp, vp, truth.T_orb, truth.B_true, orbit.C_R)
    W = np.abs(res["dE"] - res["W_srp"])  # exact SRP model, perfect POD: the true drag work per orbit
    wmed = float(np.median(W))
    a = float(np.mean(orbit.sma(rp[:200], vp[:200])))
    E_true = orbit.energy(rp, vp)
    out = {"h_km": truth.h_km, "drag_work_J_per_kg": wmed, "da_m": 2 * a**2 / orbit.MU * wmed}
    for g in ("good", "cots"):
        _, r, v = pod.observe(truth.t, truth.states, g, rng)
        out[f"sigma_dE_{g}"] = float(np.sqrt(2) * np.std(orbit.energy(r, v) - E_true))
    return out
