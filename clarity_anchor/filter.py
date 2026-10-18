"""Scalar Kalman filter on y = 1/K_eff (default inverse estimator) or K_eff (ratio mode); see spec §4."""
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


# ---------------------------------------------------------------------------------------------
# Round 3, variant C (stretch): 2-state filter x = [y, d] on z_inv = y - d * q_k
#   I_gauge = (1/y) I_true + c * W  =>  z_inv = I_pod / I_gauge = y - (y c / rho_s) * (rho_s W / I_gauge)
#   d = y c / rho_s (dimensionless), q_k = rho_s W_k / I_gauge,k, rho_s an estimator-known density scale.
# ---------------------------------------------------------------------------------------------
D0 = 0.0  # prior offset: none
PD0 = 0.3**2  # prior variance of d: offset up to ~30 % of the typical measured density


@dataclass
class KF2Result:
    t: np.ndarray  # update times
    x: np.ndarray  # (n, 2) posterior [y, d]
    P: np.ndarray  # (n, 2, 2) posterior covariance

    @property
    def corr(self):
        return self.P[:, 0, 1] / np.sqrt(self.P[:, 0, 0] * self.P[:, 1, 1])

    def at(self, tq):
        """Causal [y, d] at tq (prior before the first update)."""
        tq = np.asarray(tq, float)
        i = np.searchsorted(self.t, tq, side="right") - 1
        prior = np.array([X0, D0])
        return np.where((i >= 0)[:, None], self.x[np.clip(i, 0, None)], prior)


def run2(t_meas, z, q, R, x0=X0, d0=D0, p0=P0, pd0=PD0, q_rel=Q_REL, t0=0.0) -> KF2Result:
    z = np.asarray(z, float)
    q = np.asarray(q, float)
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(q)) and np.isfinite(R) and R >= 0):
        raise ValueError("run2: measurements, regressors and R must be finite (R >= 0)")
    x = np.array([float(x0), float(d0)])
    P = np.diag([float(p0), float(pd0)])
    tp = float(t0)
    xs, Ps = np.empty((len(z), 2)), np.empty((len(z), 2, 2))
    for k, (tk, zk, qk) in enumerate(zip(t_meas, z, q)):
        dt = tk - tp
        P = P + np.diag([q_rel * x[0] * x[0] * dt, q_rel * dt])  # random walk on both states
        H = np.array([1.0, -qk])
        S = H @ P @ H + R
        G = P @ H / S
        x = x + G * (zk - H @ x)
        P = P - np.outer(G, H @ P)
        P = 0.5 * (P + P.T)
        tp = tk
        xs[k], Ps[k] = x, P
    return KF2Result(np.asarray(t_meas, float), xs, Ps)
