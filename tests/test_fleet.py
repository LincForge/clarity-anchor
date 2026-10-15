import numpy as np
import pytest

from clarity_anchor import gauge, scenarios


def test_shared_k_path_and_fleet_mean_cancels_eps():
    """Two hosts sharing one K path with eps_B = +0.1 and -0.1: the fleet-mean K_hat removes most of the bias."""
    tr = scenarios.build_truth(400.0, days=1.0, use_cache=False)
    K1 = gauge.k_random_walk(tr.t1, 1.2, gauge.DRIFT_SIGMA_WEEK, np.random.default_rng(5))
    kw = dict(n_orb=1, days=1.0, use_cache=False, mc_draws=30, keep_series=True, k_path=K1)
    a = scenarios.run_cell(400.0, "good", 0.1, noise_seed=1, pod_seed=21, **kw)
    b = scenarios.run_cell(400.0, "good", -0.1, noise_seed=2, pod_seed=22, **kw)
    assert a["K0"] == b["K0"] == pytest.approx(1.2)
    fleet = 0.5 * (a["series"]["khat"] + b["series"]["khat"])
    own = a["series"]["khat"]
    truth = a["series"]["keff_true"] / 1.1  # K * gbar, no eps
    assert np.mean(own[3:] / truth[3:]) - 1 == pytest.approx(0.1, abs=0.01)
    assert abs(np.mean(fleet[3:] / truth[3:]) - 1) < 0.01
