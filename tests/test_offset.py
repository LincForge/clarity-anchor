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
    assert np.allclose(rho_corr[~skip], (K * g * rho)[~skip], rtol=1e-12, atol=0.0)


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


def test_run_cell_offset_metrics_plumbing():
    """F6 plumbing: the new per-cell metrics exist and behave (A correlates with b/rho, B removes most of it)."""
    kw = dict(n_orb=1, days=1.0, use_cache=False, mc_draws=20)
    a = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="A", **kw)
    b = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="B", **kw)
    for c in (a, b):
        for k in ("acc_sunlit", "acc_eclipse", "prec_sunlit", "prec_eclipse", "offset_corr", "beta0", "beta1",
                  "variant", "rho_ref", "acc_lowrho_q", "acc_highrho_q"):
            assert k in c, k
        assert c["beta0"] == 0.3 and c["beta1"] == 0.3
    assert a["variant"] == "A" and b["variant"] == "B"
    assert a["offset_corr"] > 0.5
    assert abs(b["offset_corr"]) < abs(a["offset_corr"])
    assert b["precision"] < a["precision"]
