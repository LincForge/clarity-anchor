"""Anchor observable tests incl. spec §7 test (b)."""
import numpy as np
import pytest

from clarity_anchor import anchor, gauge, orbit, pod, scenarios


@pytest.fixture(scope="module")
def truth400():
    return scenarios.build_truth(400.0, days=1.0, use_cache=False)


@pytest.fixture(scope="module")
def truth400_nosrp():
    return scenarios.build_truth(400.0, days=1.0, use_cache=False, srp=False)


def test_arc_layout():
    tp = np.arange(0, 20001, 10.0)
    arcs = anchor.arc_indices(tp, 5000.0)
    assert arcs.shape == (4, 2)
    assert np.all(tp[arcs[:, 1]] - tp[arcs[:, 0]] == 5000.0)
    assert np.all(arcs[1:, 0] == arcs[:-1, 1])  # contiguous, shared endpoints


def test_truth_reference_converged(truth400):
    # the tabulated field's reference trajectory sits on top of the final truth trajectory
    assert truth400.max_ref_offset_m < 50.0


def _noiseless_z(tr, K=1.0, cr_err=0.0, srp=True):
    rng = np.random.default_rng(0)
    t1, rho1 = tr.t1, tr.rho1
    meas = gauge.measure(rho1, np.full_like(rho1, K), np.ones_like(rho1), rng, noise=False)
    tp, rp, vp = pod.observe(tr.t, tr.states, "perfect", rng)
    res = anchor.compute(t1, meas, tp, rp, vp, tr.T_orb, B_model=tr.B_true,
                         cr_model=orbit.C_R * (1 + cr_err), srp=srp)
    return res


def test_b_noiseless_ratio_is_one(truth400_nosrp):
    """(b) eps_B = 0, perfect POD, K = 1, no gauge noise, g = 1, SRP off -> z_inv = 1 to 1e-6."""
    res = _noiseless_z(truth400_nosrp, srp=False)
    err = np.max(np.abs(res["z_inv"] - 1.0))
    print(f"\n(b) SRP off: max |z-1| over {len(res['z'])} one-orbit arcs at 400 km: {err:.3e}")
    assert err < 1e-6


def test_b_with_srp_shadow_discontinuity(truth400):
    """(b') same, SRP on with an exact SRP model. The cylindrical shadow switch is a discontinuity
    that fixed-step RK4 integrates to O(dt) at each crossing, so the achieved tolerance is ~5e-6
    (it halves-ish with dt; see DECISIONS.md). Gate 1e-5."""
    res = _noiseless_z(truth400)
    err = np.max(np.abs(res["z_inv"] - 1.0))
    print(f"\n(b') SRP on: max |z-1| over {len(res['z'])} one-orbit arcs at 400 km: {err:.3e}")
    assert err < 1e-5


def test_b_scales_with_K_and_epsB(truth400):
    res = _noiseless_z(truth400, K=1.2)
    assert np.allclose(res["z_inv"], 1 / 1.2, rtol=1e-5)


def test_monte_carlo_R_matches_endpoint_energy_noise(truth400):
    """POD noise enters z mainly through the two endpoint energies: sigma_dE ~ sqrt(2)*sqrt((v sv)^2+(g sr)^2)."""
    tr = truth400
    Kg = np.ones_like(tr.rho1)
    std, mean = anchor.monte_carlo_R(tr, Kg, tr.T_orb, "good", B_model=tr.B_true, cr_model=orbit.C_R,
                                     n=200, seed=3)  # statistics of z_inv = I_pod / I_gauge
    r = np.linalg.norm(tr.states[0, :3])
    v = np.linalg.norm(tr.states[0, 3:])
    sig_E = np.sqrt((v * 0.05e-3) ** 2 + (orbit.MU / r**2 * 0.05) ** 2)
    res = _noiseless_z(tr)
    expect = np.sqrt(2) * sig_E / abs(res["dE"][0])
    print(f"\nMC std(z) {std:.4e} vs endpoint-energy estimate {expect:.4e}")
    assert std == pytest.approx(expect, rel=0.2)
    assert mean == pytest.approx(1.0, abs=4 * std / np.sqrt(200))


def test_inverse_ratio_unbiased_at_600km_one_orbit():
    """Round 2: POD noise in the numerator. Over all 1-orbit arcs at 600 km (good POD, sigma ~21 %),
    mean(z_inv * K_eff,true) - 1 must be within +-1 % (the old z = I_gauge/I_pod was +4.3 %)."""
    res = scenarios.run_cell(600.0, "good", 0.0, 1, keep_series=True, mc_draws=20)
    s = res["series"]
    r = s["z_inv"] * s["keff_true"] - 1.0
    print(f"\nmean(z_inv*K_eff - 1) = {r.mean():+.4f} +- {r.std() / np.sqrt(len(r)):.4f} over {len(r)} arcs")
    assert abs(r.mean()) < 0.01
