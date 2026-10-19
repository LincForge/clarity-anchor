"""Round 3b: averaged (N-second) and causal shutter zero-reads (docs/round3b-zero-read-spec.md)."""
import json
from pathlib import Path

import numpy as np
import pytest

from clarity_anchor import gauge, scenarios

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _t(n=3601):
    return np.arange(0.0, float(n))


def _old_mask(t, t_zero=300.0):
    """Round 3 zero_read_mask, verbatim (the n_zero = 1 reference)."""
    k = np.round(t / t_zero)
    return (k >= 1) & (np.abs(t - k * t_zero) < 1e-6)


# --- 3.1 the N-second window -------------------------------------------------------------------------------
def test_mask_n1_is_round3_mask():
    t = _t(7201)
    assert np.array_equal(gauge.zero_read_mask(t, 300.0, n_zero=1), _old_mask(t))
    assert np.array_equal(gauge.zero_read_mask(t, 300.0), _old_mask(t))


def test_mask_n10_closes_k_tzero_to_k_tzero_plus_n():
    t = _t(1201)
    m = gauge.zero_read_mask(t, 300.0, n_zero=10)
    expect = np.concatenate([np.arange(300, 310), np.arange(600, 610), np.arange(900, 910), [1200]])
    assert np.array_equal(np.nonzero(m)[0], expect)  # no window at t = 0; the last window is cut by the series end


@pytest.mark.parametrize("n_zero", [0, -1, 300, 301, 2.5, 10.0, True, float("nan"), "3", None])
def test_n_zero_guard(n_zero):
    """[contract] DbC daemon zone: n_zero must be an int with 1 <= n_zero < t_zero."""
    t = _t(601)
    with pytest.raises(ValueError):
        gauge.zero_read_mask(t, 300.0, n_zero=n_zero)
    with pytest.raises(ValueError):
        gauge.zero_read(t, np.zeros_like(t), np.random.default_rng(0), 300.0, n_zero=n_zero)


def test_n_zero_accepts_numpy_int():
    t = _t(601)
    assert gauge.zero_read_mask(t, 300.0, n_zero=np.int64(10)).sum() == 10 + 1


def test_zero_read_n1_bit_identical_to_round3():
    t = _t(7201)
    b = 1e-13 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 5400.0))
    m = _old_mask(t)
    rng = np.random.default_rng(31)
    bz_old = b[m] + rng.standard_normal(m.sum()) * (gauge.NOISE_REL * b[m] + gauge.NOISE_FLOOR)
    tz, bz = gauge.zero_read(t, b, np.random.default_rng(31), 300.0, n_zero=1)
    assert np.array_equal(tz, t[m])
    assert np.array_equal(bz, bz_old)


def test_zero_read_window_mean_and_centre_timestamp():
    t = _t(1801)
    b = 1e-13 * (1.0 + t / 1800.0)
    tz, bz = gauge.zero_read(t, b, np.random.default_rng(0), 300.0, noise=False, n_zero=10)
    ks = np.arange(1, 6)  # k = 6 (t = 1800) is cut by the series end: dropped
    assert np.array_equal(tz, ks * 300.0 + 4.5)
    assert np.allclose(bz, [b[k * 300:k * 300 + 10].mean() for k in ks], rtol=1e-14, atol=0.0)


def test_zero_read_noise_averages_down_as_sqrt_n():
    t = np.arange(0.0, 300.0 * 20001)
    b = np.full_like(t, 2e-13)
    _, bz = gauge.zero_read(t, b, np.random.default_rng(1), 300.0, n_zero=10)
    sig1 = gauge.NOISE_REL * 2e-13 + gauge.NOISE_FLOOR
    assert np.std(bz - 2e-13) == pytest.approx(sig1 / np.sqrt(10), rel=0.03)


# --- 3.2 the three estimators --------------------------------------------------------------------------------
def _scene(n=7201):
    t = _t(n)
    nu = ((t % 5400.0) < 3400.0).astype(float)
    s = gauge.sunlit_lag(t, nu, 300.0)
    rho = 3e-12 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 5400.0))
    b = gauge.offset(s, 3e-12, 0.3, 0.3)
    meas = gauge.measure(rho, np.full_like(t, 1.2), np.ones_like(t), np.random.default_rng(0), noise=False, b=b)
    return t, s, b, meas


def test_subtract_default_is_round3_interp():
    t, s, b, meas = _scene()
    r0, k0 = gauge.subtract_zero_reads(t, meas, b, np.random.default_rng(31))
    r1, k1 = gauge.subtract_zero_reads(t, meas, b, np.random.default_rng(31), n_zero=1, mode="interp")
    # round 3 formula, verbatim
    m = _old_mask(t)
    rng = np.random.default_rng(31)
    bz = b[m] + rng.standard_normal(m.sum()) * (gauge.NOISE_REL * b[m] + gauge.NOISE_FLOOR)
    ref = meas - np.interp(t, t[m], bz)
    ref[m] = np.interp(t[m], t[~m], ref[~m])
    assert np.array_equal(r0, ref) and np.array_equal(r1, ref)
    assert np.array_equal(k0, m) and np.array_equal(k1, m)


def test_hold_values_and_zero_before_first_read():
    t, s, b, meas = _scene()
    _, skip, bhat = gauge.subtract_zero_reads(t, meas, b, None, noise=False, n_zero=10, mode="hold", return_bhat=True)
    assert np.all(bhat[t < 310] == 0.0)  # first read completes after its last closed second t = 309
    assert np.allclose(bhat[(t >= 310) & (t < 610)], b[300:310].mean(), rtol=1e-14, atol=0.0)
    assert np.allclose(bhat[(t >= 610) & (t < 910)], b[600:610].mean(), rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("mode", ["hold", "fit"])
def test_causal_modes_do_not_look_ahead(mode):
    """b_hat(t) must not change when b is changed only after t (interp, by contrast, does)."""
    t, s, b, meas = _scene()
    b2 = b.copy()
    b2[t >= 3000] *= 3.0
    kw = dict(noise=False, n_zero=10, mode=mode, return_bhat=True, s_model=s)
    _, _, h1 = gauge.subtract_zero_reads(t, meas, b, None, **kw)
    _, _, h2 = gauge.subtract_zero_reads(t, meas, b2, None, **kw)
    assert np.array_equal(h1[t < 3000], h2[t < 3000])
    assert not np.array_equal(h1, h2)


def test_interp_does_look_ahead():
    t, s, b, meas = _scene()
    b2 = b.copy()
    b2[t >= 3000] *= 3.0
    _, _, h1 = gauge.subtract_zero_reads(t, meas, b, None, noise=False, n_zero=10, return_bhat=True)
    _, _, h2 = gauge.subtract_zero_reads(t, meas, b2, None, noise=False, n_zero=10, return_bhat=True)
    assert not np.array_equal(h1[t < 3000], h2[t < 3000])


def test_fit_recovers_linear_offset_exactly_once_fitted():
    """Noise-free reads of b = rho_ref (beta0 + beta1 s) with s_model = s: the OLS fit is exact."""
    t, s, b, meas = _scene(16201)
    rho_corr, skip, bhat = gauge.subtract_zero_reads(t, meas, b, None, noise=False, n_zero=10, mode="fit",
                                                     s_model=s, return_bhat=True)
    late = (t > 10800) & ~skip  # two orbits in: reads span sunlit and eclipse
    assert np.allclose(bhat[late], b[late], rtol=1e-9, atol=0.0)
    assert np.allclose(rho_corr[late], (meas - b)[late], rtol=1e-9, atol=0.0)


def test_fit_falls_back_to_hold_before_two_reads():
    t, s, b, meas = _scene()
    kw = dict(noise=False, n_zero=10, return_bhat=True)
    _, _, hf = gauge.subtract_zero_reads(t, meas, b, None, mode="fit", s_model=s, **kw)
    _, _, hh = gauge.subtract_zero_reads(t, meas, b, None, mode="hold", **kw)
    first_two = t < 610  # the second read completes at t = 610
    assert np.array_equal(hf[first_two], hh[first_two])


@pytest.mark.parametrize("kw", [
    dict(mode="lookahead"),
    dict(mode="fit"),  # no s_model
    dict(mode="fit", s_model=np.full(7201, np.nan)),
    dict(mode="fit", s_model=np.ones(10)),
])
def test_mode_guard(kw):
    """[contract] unknown mode, fit without a finite s_model of the right length: ValueError."""
    t, s, b, meas = _scene()
    with pytest.raises(ValueError):
        gauge.subtract_zero_reads(t, meas, b, None, noise=False, n_zero=10, **kw)


def test_bin_mean_min_valid_marks_gaps():
    t = np.arange(0, 90.0)
    x = t.copy()
    skip = np.zeros(90, bool)
    skip[30:60] = True  # a whole bin closed
    skip[60:70] = True  # 20 valid samples left
    tb, xb = gauge.bin_mean(t, x, 30.0, skip=skip, min_valid=15)
    assert np.allclose(tb, [14.5, 44.5, 74.5])
    assert xb[0] == pytest.approx(14.5) and np.isnan(xb[1]) and xb[2] == pytest.approx(np.mean(np.arange(70, 90)))


# --- run_cell plumbing -----------------------------------------------------------------------------------------
KW1 = dict(n_orb=1, days=1.0, use_cache=False, mc_draws=20)


@pytest.mark.parametrize("mode", ["interp", "hold", "fit"])
def test_run_cell_zero_read_modes_plumbing(mode):
    c = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="B", n_zero=10, zr_mode=mode, **KW1)
    for k in ("n_zero", "zr_mode", "tau_model", "gap_frac", "bhat_rms_rel_d1_14", "accuracy", "precision"):
        assert k in c, k
    assert c["n_zero"] == 10 and c["zr_mode"] == mode
    assert c["gap_frac"] == 0.0  # 20 of 30 samples remain in the bins the window touches
    assert np.isfinite(c["accuracy"]) and np.isfinite(c["precision"])
    assert c["skipped_frac"] == pytest.approx(10 / 300, rel=0.02)


def test_run_cell_n30_gaps_one_bin_in_ten_and_excludes_them():
    c = scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="B", n_zero=30, zr_mode="interp", **KW1)
    assert c["gap_frac"] == pytest.approx(0.1, abs=0.002)
    assert np.isfinite(c["accuracy"]) and np.isfinite(c["precision"]) and np.isfinite(c["prec_sunlit"])


def test_run_cell_rejects_unknown_zr_mode():
    with pytest.raises(ValueError):
        scenarios.run_cell(400.0, "good", 0.0, beta=(0.3, 0.3), variant="B", zr_mode="oracle", **KW1)


def test_run_cell_n1_interp_reproduces_F6_variant_B():
    """R3b-6: N = 1, interp reproduces the stored F6 400 km (0.3, 0.3) variant-B cell.

    Exact on the machine that produced F6.json (macOS arm64, 47/47 fields, recorded in summary.md R3b-6);
    within 1e-5 relative across platforms, since Linux x86_64 differs in the last bits of the MSIS evaluation
    (CI 2026-10-18). n_bins is exact.
    """
    f6 = json.loads((RESULTS / "F6.json").read_text())
    ref = next(c for c in f6["cells"] if c["h_km"] == 400.0 and c["beta0"] == 0.3 and c["beta1"] == 0.3
               and c["variant"] == "B")
    c = scenarios.run_cell(400.0, "good", 0.0, 1, beta=(0.3, 0.3), variant="B", n_zero=1, zr_mode="interp")
    for k in ("accuracy", "precision", "conv_days", "acc_sunlit", "acc_eclipse", "offset_corr", "khat_final_err",
              "bhat_err_rel_std", "n_bins"):
        if k == "n_bins":
            assert c[k] == ref[k], k
        else:
            # Linux x86_64 vs macOS arm64 differ by ~1e-6 relative through the whole chain (pymsis last bits).
            # Quoted results carry 0.01 pp = 1e-4; 1e-5 rel / 1e-7 abs is 1000x tighter than any quoted digit.
            assert c[k] == pytest.approx(ref[k], rel=1e-5, abs=1e-7), k
