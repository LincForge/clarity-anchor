import numpy as np
import pytest

from clarity_anchor import anchor, gauge, orbit, pod, scenarios
from clarity_anchor import filter as kf


def test_c_noiseless_one_arc_unit():
    """(c) K = 1.2, noiseless measurement -> K_hat = 1.2 after one arc."""
    res = kf.run(np.array([5400.0]), np.array([1.2]), R=0.0)
    assert res.x[0] == pytest.approx(1.2, abs=1e-12)


def test_c_noiseless_one_arc_end_to_end():
    tr = scenarios.build_truth(400.0, days=0.2, use_cache=False)
    rng = np.random.default_rng(0)
    meas = gauge.measure(tr.rho1, 1.2, 1.0, rng, noise=False)
    tp, rp, vp = pod.observe(tr.t, tr.states, "perfect", rng)
    a = anchor.compute(tr.t1, meas, tp, rp, vp, tr.T_orb, tr.B_true, orbit.C_R)
    res = kf.run(a["t_end"][:1], a["z_inv"][:1], R=1e-14)  # state y = 1/K_eff
    print(f"\n(c) K_hat = 1/y_hat after one arc = {1 / res.x[0]:.9f}")
    assert 1 / res.x[0] == pytest.approx(1.2, abs=1e-5)


def test_filter_averages_noise_and_holds_between_updates():
    rng = np.random.default_rng(1)
    tm = np.arange(1, 201) * 5400.0
    z = 0.9 + 0.01 * rng.standard_normal(200)
    res = kf.run(tm, z, R=1e-4)
    assert res.x[-1] == pytest.approx(0.9, abs=0.005)
    Q = kf.Q_REL * 0.81 * 5400.0  # per-arc process noise at x ~ 0.9
    assert res.P[-1] == pytest.approx(np.sqrt(Q * 1e-4) - Q / 2, rel=0.1)  # Riccati steady state
    k = res.at(np.array([0.0, 5400.0, 5401.0, 2 * 5400.0 - 1]))
    assert k[0] == pytest.approx(kf.X0)
    assert k[1] == res.x[0] and k[2] == res.x[0] and k[3] == res.x[0]
