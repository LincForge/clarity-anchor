# Implementation decisions

Choices made where `docs/sim-spec.md` is silent, plus every deviation from it.

## Environment
- Python 3.12 via `uv`; resolved versions: numpy 2.5.4, scipy 1.18.1, **pymsis 0.13.0** (NRLMSIS 2.0
  selected with `version=2.0`), matplotlib 3.11.2, pytest 9.1.1.
- pymsis is always called with explicit `f107s=150, f107as=150, aps=15` arrays; it ran fully offline
  (no space-weather download was needed or attempted).

## Orbit
- Constants: μ = 3.986004418e14, R_E = 6378137 m (also the J2 reference radius), J2 = 1.08262668e-3,
  ω_E = 7.2921150e-5 rad/s.
- Epoch 2024-03-20 12:00 UTC (spring equinox). Initial state: circular orbit at the ascending node,
  RAAN = 0, i = 87°. The Sun is then near +x, so β ≈ 0 and the host is eclipsed every orbit (worst
  case for SRP-model error). Sun direction from the low-precision Astronomical Almanac formula.
- RK4 hot loop uses plain Python floats; 14 d at 5 s takes ~0.6 s, so no duration reduction was needed.
- Test (a) is run with the atmosphere co-rotation switched off (ω_atm = 0): the analytic
  da/dt = −Bρ√(μa) assumes v_rel = v. Achieved: relative error 1.6e-7 (gate 1 %).
- Test (d): achieved max relative J2-energy error 2.2e-10 over 14 d at 5 s (gate 1e-9 met).

## Atmosphere
- pymsis cannot be called per RK4 stage at useful speed, so the truth is a **tabulated field**:
  NRLMSIS 2.0 evaluated along a reference trajectory at 10 s (and 1 km radially above it, giving a local
  scale height H(t)), then ρ(t, r) = ρ_MSIS(t, r_ref(t))·exp(−(|r| − |r_ref(t)|)/H(t))·m(t)·storm(t),
  log-linear in time between knots. Drag and gauge sample the same field, so the anchor sees a
  self-consistent truth. The reference trajectory is iterated (see Scenarios) so the offset is small.
- Geodetic altitude on WGS84 (Earth rotation via GMST only, no precession/nutation/polar motion).
- Species: composition is taken as the species **mass fractions** of N2, O2, O, He, H along the reference
  trajectory (held fixed under the small radial offset and the m(t)/storm multipliers). The storm bump is
  a pure density multiplier (spec §2), so it does **not** change composition. Total density is MSIS
  column 0 (total mass density).
- φ1, φ2 drawn from `default_rng(1)`; storm centred at t = 5.0 d; m(t) uses the nominal Keplerian
  period of the scenario altitude. The storm is present in every scenario (it is part of the §2 truth).
