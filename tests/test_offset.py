"""Round 3: additive, sun-correlated gauge offset and the shutter zero-read (docs/round3-offset-spec.md)."""
import math

import numpy as np
import pytest

from clarity_anchor import gauge, scenarios


def _t(n=3601):
    return np.arange(0.0, float(n))


def test_offset_zero_betas_reproduce_old_model():
    """beta0 = beta1 = 0 => b = 0 and the gauge output is bit-identical to the round-2 model."""
    t = _t()
    nu = (np.sin(2 * np.pi * t / 1000.0) > 0).astype(float)
    s = gauge.sunlit_lag(t, nu)
    b = gauge.offset(s, rho_ref=3e-12, beta0=0.0, beta1=0.0)
    assert np.all(b == 0.0)
    rho = np.full_like(t, 3e-12)
    K = np.full_like(t, 1.2)
    g = np.full_like(t, 0.95)
    old = gauge.measure(rho, K, g, np.random.default_rng(4))
    new = gauge.measure(rho, K, g, np.random.default_rng(4), b=b)
    assert np.array_equal(old, new)


def test_offset_value_and_lag():
    t = _t(3001)
    nu = np.ones_like(t)
    nu[:1] = 0.0
    s = gauge.sunlit_lag(t, nu, tau=300.0)
    assert s[0] == 0.0  # starts at the instantaneous indicator
    # step response of a first-order lag: 1 - exp(-t/tau) at t = tau
    assert s[300] == pytest.approx(1 - math.exp(-1.0), abs=2e-3)
    assert s[-1] == pytest.approx(1.0, abs=1e-3)
    b = gauge.offset(s, rho_ref=2e-12, beta0=0.1, beta1=0.3)
    assert b[-1] == pytest.approx(2e-12 * (0.1 + 0.3 * s[-1]))


@pytest.mark.parametrize("kw", [
    dict(beta0=float("nan"), beta1=0.1, rho_ref=1e-12),
    dict(beta0=0.1, beta1=float("inf"), rho_ref=1e-12),
    dict(beta0=-0.01, beta1=0.1, rho_ref=1e-12),
    dict(beta0=0.1, beta1=-0.3, rho_ref=1e-12),
    dict(beta0=0.1, beta1=0.1, rho_ref=0.0),
    dict(beta0=0.1, beta1=0.1, rho_ref=-1e-12),
    dict(beta0=0.1, beta1=0.1, rho_ref=float("nan")),
    dict(beta0=0.1, beta1=0.1, rho_ref=float("inf")),
])
def test_offset_guard_rejects_hostile_input(kw):
    """[contract] DbC daemon zone: fail fast with ValueError on non-finite / negative beta or bad rho_ref."""
    s = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError):
        gauge.offset(s, **kw)


def test_offset_guard_rejects_nonfinite_sunlit_indicator():
    """[contract] a NaN in s would silently propagate into every density sample; reject it."""
    s = np.array([0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        gauge.offset(s, rho_ref=1e-12, beta0=0.1, beta1=0.1)


def test_zero_read_noiseless_recovers_b_exactly():
    t = _t(3601)
    b = 1e-13 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 5400.0))
    mask = gauge.zero_read_mask(t, 300.0)
    assert np.array_equal(np.nonzero(mask)[0], np.arange(300, 3601, 300))
    tz, bz = gauge.zero_read(t, b, np.random.default_rng(0), t_zero=300.0, noise=False)
    assert np.array_equal(tz, t[mask])
    assert np.array_equal(bz, b[mask])


def test_zero_read_noise_model():
    t = np.arange(0.0, 300.0 * 20001)
    b = np.full_like(t, 2e-13)
    _, bz = gauge.zero_read(t, b, np.random.default_rng(1), t_zero=300.0)
    assert np.std(bz - 2e-13) == pytest.approx(gauge.NOISE_REL * 2e-13 + gauge.NOISE_FLOOR, rel=0.03)


@pytest.mark.parametrize("beta1", [0.0, "linear"])
def test_rho_corr_noiseless_equals_Kg_rho(beta1):
    """With no gauge noise and noiseless zero-reads, rho_corr = rho_meas - b_hat equals K g rho exactly
    whenever b is piecewise linear between zero-reads (constant floor, or a linear ramp)."""
    t = _t(7201)
    rho = 3e-12 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 5400.0))
    K = np.full_like(t, 1.274)
    g = 1.0 + 0.01 * np.cos(t / 900.0)
    b = np.full_like(t, 0.3 * 3e-12) if beta1 == 0.0 else 1e-13 + 1e-17 * t
    meas = gauge.measure(rho, K, g, np.random.default_rng(0), noise=False, b=b)
    rho_corr, skip = gauge.subtract_zero_reads(t, meas, b, np.random.default_rng(1), noise=False)
    assert np.array_equal(skip, gauge.zero_read_mask(t, gauge.T_ZERO))
    # b_hat is held before the first and after the last zero-read, so a ramp is exact only between them
    inside = (t >= gauge.T_ZERO) & (t <= t[skip][-1]) if beta1 == "linear" else np.ones_like(t, bool)
    m = inside & ~skip
    assert np.allclose(rho_corr[m], (K * g * rho)[m], rtol=1e-12, atol=0.0)


def test_bin_mean_skips_masked_samples():
    t = np.arange(0, 90.0)
    x = t.copy()
    skip = np.zeros(90, bool)
    skip[[0, 59]] = True
    tb, xb = gauge.bin_mean(t, x, 30.0, skip=skip)
    assert np.allclose(tb, [14.5, 44.5, 74.5])  # bin time stamps unchanged
    assert np.allclose(xb, [np.mean(np.arange(1, 30)), np.mean(np.arange(30, 59)), 74.5])


def test_run_cell_offset_zero_reproduces_round2_cell():
    """(0, 0) variant A must reproduce the round-2 cell exactly (R3-1)."""
    kw = dict(n_orb=1, days=1.0, use_cache=False, mc_draws=20)
    old = scenarios.run_cell(400.0, "good", 0.0, **kw)
    new = scenarios.run_cell(400.0, "good", 0.0, beta=(0.0, 0.0), variant="A", **kw)
    for k in ("accuracy", "precision", "conv_days", "mean_err", "khat_final_err", "R_std", "acc_uncal"):
        assert new[k] == old[k], k


NEW_KEYS = ("acc_sunlit", "acc_eclipse", "prec_sunlit", "prec_eclipse", "offset_corr", "beta0", "beta1",
            "variant", "rho_ref", "acc_lowrho_q", "acc_highrho_q")
KW1 = dict(n_orb=1, days=1.0, use_cache=False, mc_draws=20)


def test_run_cell_offset_metrics_plumbing_variant_A():
    """F6 plumbing: the new per-cell metrics exist; with a (0.3, 0.3) offset, A's error correlates with b/rho."""
    a = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="A", **KW1)
    for k in NEW_KEYS:
        assert k in a, k
    assert a["variant"] == "A" and a["beta0"] == 0.3 and a["beta1"] == 0.3
    assert a["offset_corr"] > 0.3  # error tracks b/rho (0.99 on the 14-day cell; 1-day window includes the transient)
    assert 0.0 < a["b_over_rho_mean"] < 1.0


def test_run_cell_offset_metrics_plumbing_variant_B():
    """Variant B (shutter zero-read) removes most of the offset damage seen by A on the same cell."""
    a = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="A", **KW1)
    b = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="B", **KW1)
    for k in NEW_KEYS:
        assert k in b, k
    assert b["variant"] == "B"
    assert b["precision"] < a["precision"]
    assert b["accuracy"] < a["accuracy"]
    assert b["bhat_err_rel_std"] < 0.03
    assert b["skipped_frac"] == pytest.approx(1 / 300, rel=0.01)


def test_run_cell_rejects_unknown_variant():
    with pytest.raises(ValueError):
        scenarios.run_cell(400.0, "good", 0.0, beta=(0.1, 0.1), variant="Z", **KW1)


# --- variant C (stretch): 2-state filter [y, d] with z_inv = y - d * q_k, no zero-reads --------------------------
def test_run2_recovers_both_states_when_q_varies():
    from clarity_anchor import filter as kf

    tm = np.arange(1, 101) * 5400.0
    q = 1.0 + 0.5 * np.sin(np.arange(100) / 3.0)  # regressor varies arc to arc -> both states observable
    z = 0.8 - 0.2 * q
    res = kf.run2(tm, z, q, R=1e-12, q_rel=0.0)
    assert res.x[-1, 0] == pytest.approx(0.8, abs=1e-4)
    assert res.x[-1, 1] == pytest.approx(0.2, abs=1e-4)
    assert abs(res.corr[-1]) < 0.99


def test_run2_constant_regressor_is_degenerate():
    """[contract] With a constant regressor only y - d*q is observed: the posterior correlation goes to +1."""
    from clarity_anchor import filter as kf

    tm = np.arange(1, 101) * 5400.0
    q = np.ones(100)
    res = kf.run2(tm, 0.6 * np.ones(100), q, R=1e-6, q_rel=0.0)
    assert res.corr[-1] > 0.99
    assert res.x[-1, 0] - res.x[-1, 1] == pytest.approx(0.6, abs=1e-3)


def test_run2_rejects_nonfinite_measurement():
    """[contract] a NaN measurement would poison both states for the rest of the run."""
    from clarity_anchor import filter as kf

    with pytest.raises(ValueError):
        kf.run2(np.array([1.0, 2.0]), np.array([1.0, np.nan]), np.array([1.0, 1.0]), R=1e-4)


def test_run_cell_variant_C_plumbing():
    c = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="C", **KW1)
    for k in NEW_KEYS + ("c_post_corr_final", "c_hat_final_over_rho_ref"):
        assert k in c, k
    assert c["variant"] == "C"
    assert -1.0 <= c["c_post_corr_final"] <= 1.0
