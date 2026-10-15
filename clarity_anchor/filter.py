"""Scalar Kalman filter on K_eff (spec §4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gauge import DRIFT_SIGMA_WEEK, WEEK

X0 = 1.0  # ground-calibration prior: nominal gain
P0 = 0.3**2  # prior variance (generous: ±30 % calibration, plus unknown g and eps_B)
Q_REL = DRIFT_SIGMA_WEEK**2 / WEEK  # relative random-walk variance per second


@dataclass
class KFResult:
    t: np.ndarray  # update times
    x: np.ndarray  # posterior estimate after each update
    P: np.ndarray  # posterior variance

    def at(self, tq):
        """Causal estimate: latest posterior with update time <= tq (prior X0 before the first)."""
        tq = np.asarray(tq, float)
        i = np.searchsorted(self.t, tq, side="right") - 1
        return np.where(i >= 0, self.x[np.clip(i, 0, None)], X0)

    def sigma_at(self, tq):
        tq = np.asarray(tq, float)
        i = np.searchsorted(self.t, tq, side="right") - 1
        return np.sqrt(np.where(i >= 0, self.P[np.clip(i, 0, None)], P0))


def run(t_meas, z, R, x0=X0, p0=P0, q_rel=Q_REL, t0=0.0) -> KFResult:
    x, P, tp = float(x0), float(p0), float(t0)
    xs, Ps = np.empty(len(z)), np.empty(len(z))
    for k, (tk, zk) in enumerate(zip(t_meas, z)):
        P += q_rel * x * x * (tk - tp)  # random-walk predict (state unchanged)
        G = P / (P + R)
        x += G * (zk - x)
        P *= 1.0 - G
        tp = tk
        xs[k], Ps[k] = x, P
    return KFResult(np.asarray(t_meas, float), xs, Ps)
