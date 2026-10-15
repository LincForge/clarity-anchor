"""Benchmark metrics (spec §5): accuracy, precision, resolution, convergence time."""
from __future__ import annotations

import math

import numpy as np

ACC_LIMIT = 0.20
PREC_LIMIT = 0.10
RES_LIMIT_S = 30.0
CONV_TOL = 0.05


def convergence_time(t, e, tol=CONV_TOL):
    """Earliest time after which |e| < tol for every remaining sample; NaN if the last sample fails."""
    t = np.asarray(t, float)
    bad = np.nonzero(np.abs(np.asarray(e, float)) >= tol)[0]
    if len(bad) == 0:
        return float(t[0])
    if bad[-1] == len(t) - 1:
        return float("nan")
    return float(t[bad[-1] + 1])


def acc_prec(t, e, t_from):
    m = np.asarray(t) >= t_from
    em = np.asarray(e)[m]
    if len(em) < 2:
        return float("nan"), float("nan"), int(len(em))
    return float(abs(np.mean(em))), float(np.std(em)), int(len(em))


def verdict(value, limit):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNMEASURED"
    return "PASS" if value <= limit else "FAIL"
