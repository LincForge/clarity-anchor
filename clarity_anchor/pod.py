"""Precise orbit determination model (spec §2 'POD'): truth at 10 s + white per-axis noise."""
from __future__ import annotations

import numpy as np

GRADES = {  # (position sigma m, velocity sigma m/s) per axis
    "good": (0.05, 0.05e-3),
    "cots": (0.20, 0.20e-3),
    "perfect": (0.0, 0.0),
}
CADENCE = 10.0


def observe(t, states, grade: str, rng, cadence: float = CADENCE):
    t = np.asarray(t, float)
    step = int(round(cadence / (t[1] - t[0])))
    tp = t[::step]
    sp = states[::step]
    sr, sv = GRADES[grade]
    r = sp[:, :3] + (rng.standard_normal(sp[:, :3].shape) * sr if sr else 0.0)
    v = sp[:, 3:] + (rng.standard_normal(sp[:, 3:].shape) * sv if sv else 0.0)
    return tp, r, v
