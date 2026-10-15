import numpy as np
import pytest

from clarity_anchor import gauge


def test_k_random_walk_statistics():
    rng = np.random.default_rng(0)
    t = np.arange(0, 7 * 86400 + 1, 60.0)
    finals = np.array([np.log(gauge.k_random_walk(t, 1.0, 0.03, rng)[-1]) for _ in range(2000)])
    assert np.std(finals) == pytest.approx(0.03, rel=0.08)
    assert abs(np.mean(finals)) < 0.003


def test_k0_draw_range():
    rng = np.random.default_rng(1)
    k0 = np.array([gauge.draw_k0(rng) for _ in range(1000)])
    assert k0.min() >= 0.7 and k0.max() <= 1.3


def test_composition_factor():
    frac = np.array([[0.2, 0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0, 0.0]])
    g = gauge.composition_factor(frac)
    assert g[0] == pytest.approx(1.0)
    assert g[1] == pytest.approx(0.8 / (0.2 + 0.8 * 0.8))
    ones = {k: 1.0 for k in gauge.SENSITIVITY}
    assert np.allclose(gauge.composition_factor(frac, ones), 1.0)


def test_measurement_noise_model():
    rng = np.random.default_rng(2)
    rho = np.full(200000, 3e-13)
    meas = gauge.measure(rho, np.ones_like(rho), np.ones_like(rho), rng)
    sigma = 0.02 * 3e-13 + 1e-14
    assert np.std(meas - rho) == pytest.approx(sigma, rel=0.02)
    clean = gauge.measure(rho, 1.1 * np.ones_like(rho), 0.9 * np.ones_like(rho), rng, noise=False)
    assert np.allclose(clean, 0.99 * rho)


def test_bin30():
    t = np.arange(0, 90.0)
    x = t.copy()
    tb, xb = gauge.bin_mean(t, x, 30.0)
    assert np.allclose(tb, [14.5, 44.5, 74.5])  # mean sample time of each bin
    assert np.allclose(xb, [14.5, 44.5, 74.5])
