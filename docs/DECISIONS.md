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

## Gauge
- K random walk is multiplicative: ln K diffuses with σ = 0.03 per √week (so 3 %/week 1-σ); K(0) from
  U(0.7, 1.3) with the scenario gauge seed. Same gauge seed (hence same K(t) path) in every scenario,
  so scenarios differ only in the factor under study.
- Noise σ = 0.02·ρ_true + 1e-14 kg/m³ (linear sum, the literal reading of "2 % … plus an absolute floor";
  more conservative than quadrature). No clipping (MSIS density never approaches the floor here).
- g(t) uses species mass fractions (spec writes ρ_i) normalised to 1 at the first 1 Hz sample.
- 30 s bins are contiguous, non-overlapping; each bin is time-stamped at its mean sample time.

## Truth construction
- Pass 0 propagates drag-free and tabulates MSIS along it; then 2 iterations of (propagate with drag
  using the previous field → re-tabulate along the result); the final truth is propagated with the last
  field. Residual max radial offset between the field's reference trajectory and the truth: < 1 cm at
  400 km (1-day check), so the exponential scale-height correction is effectively unused.
- Truth at 1 Hz (for the gauge) is cubic-Hermite-interpolated from the 5 s RK4 states and the field is
  evaluated at those positions.

## Anchor (spec §3) — two corrections to the spec's algebra
1. **Kernel.** The drag power is a_drag·v = −½Bρ|v_rel|(v_rel·v). The spec writes the weighting as
   |v_rel|²(v_rel·v̂), which differs from that by |v_rel|/|v| (≈0.3 % at i = 87°, varying around the
   orbit) and so cannot give z = 1 to 1e-6. Both integrals use the exact kernel w = |v_rel|(v_rel·v);
   the "identical weighting in both integrals" intent is kept.
2. **Sign of the ε_B exponent.** I_pod = −2W_drag/B_model = (B_true/B_model)∫ρw = I_true/(1+ε_B), so
   z = I_gauge/I_pod = K·ḡ·(1+ε_B), not K·ḡ·(1+ε_B)^(−1). Consequently ρ_cal = ρ_meas/K̂ is biased by
   1/(1+ε_B) − 1 = −ε_B/(1+ε_B): −4.8 %, −9.1 %, −16.7 % for ε_B = 0.05/0.10/0.20 (close to, but
   slightly below, the spec's "1:1").
- Arcs: contiguous, non-overlapping, length rounded to the 10 s POD grid, sharing endpoints (so the
  endpoint POD noise of consecutive arcs is anti-correlated; the scalar KF treats arcs as independent,
  which is conservative). 1 Hz kinematics for the kernel and the SRP work come from cubic-Hermite
  interpolation of the (noisy) POD states. Integrals use composite Simpson at 1 s.
- **Test (b)** achieved tolerances (400 km, 15 one-orbit arcs, 5 s truth step): SRP off: max|z−1| =
  4.7e-7 (gate 1e-6 met; 4.0e-8 at 2.5 s step, i.e. RK4 truncation). SRP on with an exact SRP model:
  5.3e-6 — the cylindrical shadow is a discontinuity that fixed-step RK4 integrates only to O(dt) at each
  crossing (~4e-4 J/kg per orbit vs 83 J/kg drag work). Gated at 1e-5; negligible next to POD noise (~1 %).
- **R**: empirical std of z over 200 draws (fresh POD noise + fresh gauge noise) on the FIRST arc of each
  (altitude, POD grade, T_arc) cell, used as a constant R for the whole run. Check at 400 km / good /
  1 orbit: MC std 0.997 % vs endpoint-energy estimate 0.992 %.

## Filter
- State x = K_eff, prior x0 = 1.0 (ground calibration nominal), P0 = 0.3²; random-walk process noise
  Q = (0.03)²/week · x̂² · Δt (relative, matching the gauge drift). Update at each arc end with z_arc.
- ρ_cal(t) = ρ_meas,30s(t)/K̂(t) with the **causal** estimate (latest update at or before t; the prior
  before the first arc). No smoothing — this is the real-time product.
- Test (c): unit test K̂ = 1.2 exactly with R = 0; end-to-end (truth → gauge K = 1.2 → anchor → KF)
  K̂ = 1.1999981 after one arc (error 1.6e-6, the SRP-shadow term of test (b')); gate 1e-5.

## Scenarios and metrics
- Seeds: atmosphere phases 1; gauge 7 (K(0) = 1.075 — close to nominal, so see the ensemble); POD good 11,
  COTS 12; Monte Carlo R 101. Every figure regenerates with `uv run python -m clarity_anchor.run --fig all`
  (~6 min, single process; truths cached in `cache/`, gitignored).
- The SRP model error (C_R·1.10) is on in every scenario; B_model = B_true·(1+ε_B).
- K_eff,true per arc = kernel-weighted mean of K(t)·g(t) over the arc × (1+ε_B) (exactly what a
  noiseless z measures). F1's "K̂ relative error" is K̂(t)/K_eff,true(arc containing t) − 1 at 30 s.
- Convergence time = earliest t after which |K̂/K_eff,true − 1| < 5 % for the rest of the run; NaN if the
  last sample is outside. Because K̂ only updates at arc ends, convergence time is bounded below by one
  T_arc. A cell can "converge" spuriously when its filter barely moves (R huge) but the truth K wanders
  within 5 % of the prior; such cells are flagged † when the final posterior σ exceeds 5 %.
- Accuracy/precision window: from the convergence time to day 14; if not converged, days 7–14.
  Resolution = measured median spacing of the delivered ρ_cal series (30 s by construction).
- An "uncalibrated accuracy" column (K̂ ≡ 1, ground calibration only) is reported so a PASS can be
  attributed to the anchor rather than to a lucky K(0).
- **F4 robustness ensemble** (added; spec silent): every F4 cell is rerun with gauge seeds 1–10 (fresh
  K(0) ~ U(0.7, 1.3) and K path each). The main-seed figure and an ensemble-median panel are both shown.
- No duration reduction was needed: the full F4 grid plus ensemble (264 cell runs) takes ~5.5 min.
- Expectation 4 (composition drift "around the storm") is not testable as specified: the §2 storm is a
  density-only multiplier at fixed ap = 15, so it cannot change composition. Reported UNMEASURED with
  the measured background ḡ drift.

## Findings flagged for the reviewer
- **Ratio-estimator bias.** z = I_gauge/I_pod has POD noise in the denominator, so E[z] ≈ K_eff(1 + σ_z²).
  At 600 km / good / 1-orbit arcs (σ_z ≈ 21 %) the mean of z/K_eff,true − 1 over 208 arcs is +4.3 % ±
  1.8 %, matching σ² = 4.4 %, and K̂ sits ~+5 % high for days. At σ_z ≳ 50 % (600 km COTS 1-orbit,
  MC std(z) ≈ 98, i.e. 9800 %) z is meaningless because I_pod crosses zero. Filtering on 1/K (measurement I_pod/I_gauge,
  noise in the numerator) would remove the bias; not done here because spec §4 fixes x = K_eff, z = ratio.
- **600 km converges much faster than pre-registered** (0.67 d with 10-orbit arcs vs 5–10 d expected):
  the simulated per-orbit drag work at 600 km is 4.7 J/kg median (≈1.2 m/orbit decay via da = 2a²ΔE/μ, at F10.7 = 150 with
  the m(t) field), not the ~0.3 m the spec assumed, and endpoint POD noise is only ~0.55 J/kg (good).
- Every scenario sees the identical truth atmosphere per altitude and one storm; the gauge measures the
  storm directly, so the storm has no visible effect on ρ_cal (K is a gain).
