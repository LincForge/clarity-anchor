import numpy as np
import pytest

from clarity_anchor import pod


def test_pod_subsamples_and_noise_levels():
    t = np.arange(0, 100001, 5.0)
    s = np.zeros((len(t), 6))
    s[:, 0] = t
    rng = np.random.default_rng(0)
    tp, r, v = pod.observe(t, s, "good", rng)
    assert np.allclose(np.diff(tp), 10.0)
    assert np.std(r[:, 0] - tp) == pytest.approx(0.05, rel=0.05)
    assert np.std(v[:, 2]) == pytest.approx(0.05e-3, rel=0.05)
    tp, r, v = pod.observe(t, s, "cots", rng)
    assert np.std(r[:, 1]) == pytest.approx(0.20, rel=0.05)
    assert np.std(v[:, 1]) == pytest.approx(0.2e-3, rel=0.05)
    tp, r, v = pod.observe(t, s, "perfect", rng)
    assert np.all(r[:, 0] == tp) and np.all(v == 0)
