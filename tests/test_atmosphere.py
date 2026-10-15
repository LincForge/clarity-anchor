import math

import numpy as np
import pytest

from clarity_anchor import atmosphere as atm
from clarity_anchor import orbit


def test_model_error_and_storm_shapes():
    T = 5500.0
    t = np.array([0.0, T / 4])
    m = atm.model_error(t, T, phi1=0.0, phi2=0.0)
    assert m[0] == pytest.approx(1.0)
    assert m[1] == pytest.approx(1 + 0.15 + 0.10 * math.sin(2 * math.pi * (T / 4) / 86400))
    s = atm.storm(np.array([5 * 86400.0, 5 * 86400.0 + 6 * 3600.0, 0.0]))
    assert s[0] == pytest.approx(2.5)
    assert s[1] == pytest.approx(1 + 1.5 * math.exp(-1))
    assert s[2] == pytest.approx(1.0, abs=1e-12)


def test_geodetic_conversion():
    # At epoch the ECEF and ECI frames differ by GMST; place points via the inverse rotation.
    th = atm.gmst(0.0)
    c, s = math.cos(th), math.sin(th)
    r_eq = np.array([[c * (orbit.R_E + 400e3), s * (orbit.R_E + 400e3), 0.0]])
    lat, lon, alt = atm.eci_to_geodetic(np.array([0.0]), r_eq)
    assert lat[0] == pytest.approx(0.0, abs=1e-9)
    assert lon[0] == pytest.approx(0.0, abs=1e-9)
    assert alt[0] == pytest.approx(400e3, abs=1e-3)
    b = orbit.R_E * (1 - atm.WGS84_F)
    lat, lon, alt = atm.eci_to_geodetic(np.array([0.0]), np.array([[0.0, 0.0, b + 400e3]]))
    assert lat[0] == pytest.approx(90.0, abs=1e-6)
    assert alt[0] == pytest.approx(400e3, abs=1e-3)


def test_msis_plausible_and_species():
    t = np.array([0.0, 3600.0])
    r = np.array([[orbit.R_E + 400e3, 0, 0], [0, orbit.R_E + 400e3, 0]], float)
    rho, species = atm.msis_eval(t, r)
    assert np.all((rho > 5e-13) & (rho < 2e-11))
    assert set(species) == {"N2", "O2", "O", "He", "H"}
    frac = sum(species[k] * atm.SPECIES_MASS[k] for k in species) / rho
    assert np.all((frac > 0.9) & (frac < 1.05))


@pytest.fixture(scope="module")
def field():
    fm = orbit.ForceModel(B=0.0, srp=False)
    t, s = orbit.propagate(orbit.circular_state(400e3), 6 * 3600.0, 10.0, fm)
    return atm.TruthField.from_reference(t, s[:, :3], T_orb=orbit.kepler_period(400e3), seed=1), t, s


def test_truth_field_matches_msis_on_reference(field):
    f, t, s = field
    rho_msis, _ = atm.msis_eval(t, s[:, :3])
    expect = rho_msis * f.multiplier(t)
    got = f.density(t, s[:, :3])
    assert np.max(np.abs(got / expect - 1)) < 1e-6
    # scalar path used by the propagator agrees with the vector path
    i = 37
    assert f.density_scalar(t[i], *s[i, :3]) == pytest.approx(got[i], rel=1e-9)


def test_truth_field_exponential_offset(field):
    f, t, s = field
    r = s[100:101, :3]
    up = r * (1 + 1000.0 / np.linalg.norm(r))
    ratio = f.density(t[100:101], up)[0] / f.density(t[100:101], r)[0]
    assert ratio == pytest.approx(math.exp(-1000.0 / f.H_at(t[100:101])[0]), rel=1e-9)
    assert 30e3 < f.H_at(t[100:101])[0] < 120e3
