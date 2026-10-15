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


# ---------------------------------------------------------------------------------------------
# Scenario cells
# ---------------------------------------------------------------------------------------------
GAUGE_SEED = 7
POD_SEED = {"good": 11, "cots": 12, "perfect": 13}
MC_SEED = 101
MC_DRAWS = 200
SRP_MODEL_ERR = 0.10


def _truth_kernel(truth):
    from .anchor import kernel

    r1, v1 = orbit.hermite(truth.t, truth.states[:, :3], truth.states[:, 3:], truth.t1)
    return kernel(r1, v1)


def run_cell(h_km, grade, eps_B=0.0, n_orb=1, days=14.0, use_cache=True, mc_draws=MC_DRAWS,
             composition=True, gauge_seed=GAUGE_SEED, keep_series=False, k_path=None, noise_seed=None,
             pod_seed=None):
    """One scenario cell: truth -> gauge -> POD -> anchor -> KF -> metrics."""
    from scipy.integrate import simpson

    from . import anchor, gauge, metrics, pod
    from . import filter as kf

    truth = build_truth(h_km, days=days, use_cache=use_cache)
    t1, rho1 = truth.t1, truth.rho1
    rng_g = np.random.default_rng(gauge_seed)
    if k_path is None:
        K0 = gauge.draw_k0(rng_g)
        K1 = gauge.k_random_walk(t1, K0, gauge.DRIFT_SIGMA_WEEK, rng_g)
    else:  # shared lot-level K path (F5)
        K1 = np.asarray(k_path)
        K0 = float(K1[0])
    if noise_seed is not None:
        rng_g = np.random.default_rng(noise_seed)
    g1 = gauge.composition_factor(truth.frac1) if composition else np.ones_like(t1)
    meas1 = gauge.measure(rho1, K1, g1, rng_g)
    tp, rp, vp = pod.observe(truth.t, truth.states, grade,
                             np.random.default_rng(POD_SEED[grade] if pod_seed is None else pod_seed))
    B_model = truth.B_true * (1 + eps_B)
    cr_model = orbit.C_R * (1 + SRP_MODEL_ERR)
    arc_len = n_orb * truth.T_orb
    a = anchor.compute(t1, meas1, tp, rp, vp, arc_len, B_model, cr_model)
    r_std, r_mean = anchor.monte_carlo_R(truth, K1 * g1, arc_len, grade, B_model, cr_model,
                                         n=mc_draws, seed=MC_SEED)
    R = r_std**2
    res_kf = kf.run(a["t_end"], a["z"], R)

    # truth K_eff per arc: kernel-weighted K*g over the arc, times (1+eps_B)
    w1 = _truth_kernel(truth)
    L = int(round((a["t_end"][0] - a["t_start"][0])))
    idx = (np.arange(len(a["z"])) * L)[:, None] + np.arange(L + 1)[None, :]
    den = simpson((rho1 * w1)[idx], dx=1.0, axis=1)
    keff_true = simpson((K1 * g1 * rho1 * w1)[idx], dx=1.0, axis=1) / den * (1 + eps_B)
    gbar = simpson((g1 * rho1 * w1)[idx], dx=1.0, axis=1) / den

    # 30 s products
    tb, meas_b = gauge.bin_mean(t1, meas1, 30.0)
    _, rho_b = gauge.bin_mean(t1, rho1, 30.0)
    khat_b = res_kf.at(tb)
    rho_cal = meas_b / khat_b
    e_rho = rho_cal / rho_b - 1.0
    in_arcs = tb < a["t_end"][-1]
    arc_of_bin = np.clip(np.searchsorted(a["t_start"], tb, side="right") - 1, 0, len(a["z"]) - 1)
    e_k = khat_b / keff_true[arc_of_bin] - 1.0
    t_conv = metrics.convergence_time(tb[in_arcs], e_k[in_arcs], metrics.CONV_TOL)
    converged = not np.isnan(t_conv)
    t_from = t_conv if converged else (days - 7.0) * 86400.0
    acc, prec, nwin = metrics.acc_prec(tb, e_rho, t_from)
    mean_err = float(np.mean(e_rho[tb >= t_from]))
    acc_uncal = float(abs(np.mean((meas_b / rho_b - 1.0)[tb >= t_from])))  # ground calibration only (K_hat = 1)
    out = {
        "h_km": h_km, "grade": grade, "eps_B": eps_B, "n_orb": n_orb, "days": days,
        "K0": K0, "R_std": r_std, "R_mc_mean": r_mean, "n_arcs": int(len(a["z"])),
        "conv_days": t_conv / 86400.0 if converged else float("nan"), "converged": converged,
        "window_from_days": t_from / 86400.0,
        "accuracy": acc, "precision": prec, "mean_err": mean_err, "n_bins": nwin,
        "resolution_s": float(np.median(np.diff(tb))),
        "z_rel_noise_vs_truth": float(np.std(a["z"] / keff_true - 1)),
        "gbar_min": float(gbar.min()), "gbar_max": float(gbar.max()),
        "gbar_max_rate_per_day": float(np.max(np.abs(np.diff(gbar))) / (L / 86400.0)) if len(gbar) > 1 else float("nan"),
        "khat_final_err": float(e_k[in_arcs][-1]),
        "acc_uncal": acc_uncal,
        "ksig_final_rel": float(np.sqrt(res_kf.P[-1]) / res_kf.x[-1]),
    }
    if keep_series:
        out["series"] = {"tb": tb, "e_rho": e_rho, "e_k": e_k, "in_arcs": in_arcs, "rho_b": rho_b,
                         "rho_cal": rho_cal, "meas_b": meas_b, "t_arc": a["t_end"], "z": a["z"], "keff_true": keff_true,
                         "gbar": gbar, "khat": res_kf.x, "ksig": np.sqrt(res_kf.P)}
    return out
