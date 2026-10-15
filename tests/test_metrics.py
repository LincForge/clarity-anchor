import numpy as np
import pytest

from clarity_anchor import metrics, scenarios


def test_convergence_time_is_last_entry_into_band():
    t = np.arange(10.0)
    e = np.array([0.5, 0.2, 0.03, 0.06, 0.04, 0.01, 0.0, 0.02, -0.04, 0.01])
    assert metrics.convergence_time(t, e, 0.05) == 4.0
    assert np.isnan(metrics.convergence_time(t, np.append(e[:-1], 0.2), 0.05))
    assert metrics.convergence_time(t, np.zeros(10), 0.05) == 0.0


def test_verdicts():
    assert metrics.verdict(0.19, 0.20) == "PASS"
    assert metrics.verdict(0.21, 0.20) == "FAIL"
    assert metrics.verdict(float("nan"), 0.20) == "UNMEASURED"
    assert metrics.verdict(None, 0.20) == "UNMEASURED"


def test_accuracy_precision_window():
    t = np.arange(100.0)
    e = np.where(t < 50, 1.0, 0.1 + 0.02 * np.sin(t))
    acc, prec, n = metrics.acc_prec(t, e, t_from=50.0)
    assert acc == pytest.approx(0.1, abs=0.005)
    assert prec == pytest.approx(np.std(0.02 * np.sin(t[50:])), rel=1e-9)
    assert n == 50


def test_cell_epsB_bias_short_run():
    """End-to-end: at 400 km with good POD, eps_B = 0.10 biases rho_cal by -eps/(1+eps) = -9.1 %."""
    res = scenarios.run_cell(400.0, "good", eps_B=0.10, n_orb=1, days=1.0, use_cache=False, mc_draws=50)
    assert res["mean_err"] == pytest.approx(-0.10 / 1.10, abs=0.02)
    assert res["resolution_s"] == pytest.approx(30.0)
