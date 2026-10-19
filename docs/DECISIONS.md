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
- *(Round 1; superseded by the round-2 inverse estimator below.)* State x = K_eff, prior x0 = 1.0 (ground calibration nominal), P0 = 0.3²; random-walk process noise
  Q = (0.03)²/week · x̂² · Δt (relative, matching the gauge drift). Update at each arc end with z_arc.
- ρ_cal(t) = ρ_meas,30s(t)/K̂(t) with the **causal** estimate (latest update at or before t; the prior
  before the first arc). No smoothing — this is the real-time product.
- Test (c): unit test K̂ = 1.2 exactly with R = 0; end-to-end (truth → gauge K = 1.2 → anchor → KF)
  K̂ = 1.1999981 after one arc (error 1.6e-6, the SRP-shadow term of test (b')); gate 1e-5.

## Scenarios and metrics
- Seeds: atmosphere phases 1; gauge 10 (round 2; K(0) = 1.274, the farthest from 1 among seeds 1–10 —
  round 1 used seed 7, K(0) = 1.075, which flattered the uncalibrated baseline); POD good 11,
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

## F5 (stretch)
- 5 hosts at 400 km, good POD, 1-orbit arcs, all flying the **same** truth orbit/atmosphere (simplest
  choice; hosts differ only in ε_B,j ~ N(0, 0.10), POD noise and gauge noise). One lot-level K(t) path
  (K(0) ~ U(0.7, 1.3)) shared by all hosts' gauges. Host 1's 30 s product is calibrated with the mean of
  the first N hosts' causal K̂; accuracy = RMS over 40 fleet draws of the days 1–14 mean relative error.
  MC R uses 50 draws per host (not 200) to keep the 200 host runs at ~2 min.

## Round 2 (2026-10-15, reviewer-approved)
- **Inverse estimator.** The filter state is y = 1/K_eff and the measurement z_inv = I_pod/I_gauge, so the
  POD noise sits in the numerator (linear, unbiased) and only the ~0.1 %-per-arc gauge noise is in the
  denominator. Prior y0 = 1, P0 = 0.3²; process noise Q = 0.03²/week·ŷ²·Δt (relative random walk, the same
  form as before since ln y = −ln K). R = var(z_inv) from the same 200-draw first-arc Monte Carlo.
  ρ_cal = ρ_meas,30s·ŷ (causal). The round-1 ratio path stays behind `run_cell(estimator="ratio")`
  (it costs one branch). sim-spec §3/§4 amended in place.
- Convergence is now judged on the calibration-factor error ŷ·K_eff,true − 1 (5 % band, same rule).
- Bias check (test): 600 km, good POD, 1-orbit arcs, 208 arcs: mean(z_inv·K_eff,true − 1) = −0.03 %
  (round-1 z: +4.3 %). The standard error of that mean is 1.3 %, larger than the ±1 % gate, so the gate
  passes on the central value, not with statistical power; the paired round-1 number on the same noise
  was +4.3 % ± 1.8 %.
- The † flag now means "final posterior σ ≥ 5 %": the cell converged by the error rule but its own
  filter cannot resolve 5 %.
- **Decay-signal table** (summary.md): median |W_drag| per 1-orbit arc from the 14-day truth with perfect
  POD and an exact SRP model, Δa = 2a²|W|/μ with a the mean osculating a of the first 200 POD epochs, and
  σ(ΔE) = √2·std(E_POD − E_true) from one POD noise draw per grade over all 10 s epochs.
- F5 was rerun unchanged in design (it uses the fleet mean of 1/ŷ_j as K̂).

## Not modeled (proposal must list as risks)
- Gravity-field model error in ΔE: truth and estimator share point mass + J2, so higher harmonics, tides and their mismodelling never enter ΔE_pod.
- Earth albedo and infrared radiation pressure.
- Thermal re-radiation (spacecraft thermal recoil).
- Attitude-control and thruster accelerations (including unloading and leaks; manoeuvre arcs are excluded by assumption).
- Time-correlated POD errors (all POD noise here is white per 10 s epoch).
- Gauge response time (the 1 Hz gauge responds instantly to ρ).

## Round 3 (2026-10-18): additive sun-correlated offset and the shutter zero-read (docs/round3-offset-spec.md)
Choices the round-3 spec left open, plus every deviation from it. All numbers are in `results/F6.json` and
`results/summary.md` (F6 section); none are repeated here.

**Truth.**
- ν(t) is `orbit.shadow_nu` (the same cylindrical shadow as the SRP truth) evaluated along the 1 Hz truth orbit
  (cubic Hermite of the 5 s states) with the almanac Sun direction.
- s(t) is the exact zero-order-hold discretisation of the first-order lag at 1 Hz, s[k] = a s[k−1] + (1 − a) ν[k],
  a = exp(−1 s/τ_b), τ_b = 300 s (assumption, as the spec says). Initial condition s(0) = ν(0): the wall starts in
  equilibrium with the illumination it has at t = 0 (it settles within one orbit, well before any accuracy window).
- ρ_ref = median of the 1 Hz ρ_true along the truth orbit over t < 1 day, so it includes the m(t) model-error field
  (the day-5 storm is outside it). Stored per cell as `rho_ref`.
- **Relative sweep, absolute physics.** The physical offset is an absolute chamber pressure, so the same outgassing
  is a larger fraction of the signal at higher altitude. The sweep is relative to ρ_ref, so one (β0, β1) maps to a
  different absolute pressure at 400 and 600 km; F6 does not say which β a real gauge sits at, at either altitude.
- The gauge noise n is unchanged, σ = 0.02·ρ_true + floor (it does not grow with b). A real gauge's noise would scale
  with the total chamber pressure, including b, so A's precision damage is if anything slightly understated.
- Guard: `gauge.offset` raises ValueError when β0 or β1 is non-finite or negative, when ρ_ref is non-finite or ≤ 0,
  and also when s(t) contains a non-finite value (a NaN would otherwise propagate silently into every sample).
- With b ≡ 0 the gauge adds 0.0 before the unchanged noise draw, so (0, 0) reproduces the round-2 numbers bit for bit
  (regression test `test_run_cell_offset_zero_reproduces_round2_cell`, and R3-1A on the 14-day cells).

**Variant B (shutter zero-read).**
- Zero-reads at t = 300 s, 600 s, … (none at t = 0, so "hold before the first" applies to the first 300 s);
  `np.interp` also holds b̂ after the last zero-read.
- Zero-read noise n_z comes from its own random stream (seed 31, `scenarios.ZR_SEED`), so the gauge noise draws of
  every other sample are identical to variant A on the same cell.
- The shutter second is not a density sample. The 30 s product leaves it out (29-sample bin mean, bin time stamp
  unchanged). The anchor's 1 Hz Simpson integral needs every sample, so ρ_corr at a shutter second is filled by linear
  interpolation of its neighbours. The truth reference bin ρ_true stays the full 30-sample mean: the product is meant
  to estimate the 30 s mean density, so the skipped sample is a real cost, not re-referenced away.
- b̂ is linear between zero-reads, so it uses the NEXT zero-read: the B product has up to 300 s of latency, unlike the
  causal ŷ. This is the spec's estimator; a causal (hold-last) b̂ was not run.
- R (the filter's measurement variance) is the round-2 Monte Carlo, unchanged and computed without the offset or the
  zero-reads, for A, B and C alike: the spec keeps the round-2 estimator, and R is dominated by POD noise.
- Diagnostic (not in the spec): every B cell is rerun with noise-free zero-reads (`zero_noise=False`), stored under
  `diagnostics.B_noiseless_zero_read` in F6.json, to separate the zero-read noise from the interpolation error of the
  lagged sunlit term.

**Variant C (stretch, run).** The spec's measurement I_gauge = (1/y)·I_true + c·∫w dt is nonlinear in [y, c]. It is
reparametrised linearly: d = y·c/ρ_s, z_inv = I_pod/I_gauge = y − d·q_k, q_k = ρ_s·W_k/I_gauge,k, W_k = ∫w dt over the
arc (from the POD kinematics, `anchor.compute` now also returns it as `W_kernel`), ρ_s = median 1 Hz ρ_meas over the
first arc (a scale the estimator knows). Prior d0 = 0, P_d0 = 0.3²; process noise on y as in round 2, on d an absolute
random walk with the same rate (0.03²/week). ε_B = 0, so I_pod = I_true. Product ρ_cal = ŷ·ρ_meas − d̂·ρ_s (causal).
True c per arc = ∫b w dt / ∫w dt. c is not a free per-arc term (that would be unobservable by construction); it is a
slowly varying state, so the test is whether arc-to-arc variation in the measured density separates it from y.

**Metrics.** Sunlit/eclipse split on the 30 s bin mean of s (> 0.5 / < 0.5). offset_corr = Pearson(e_ρ, b_30s/ρ_true,30s)
over the accuracy window; n/a at (0, 0) because b ≡ 0. Added (not in the spec): low/high-ρ_true quartile error (for
R3-3), mean b/ρ_true, eclipse fraction, b̂ error (B), and "baseline corr" = the correlation the no-offset product
(A at (0, 0), same altitude and window) already has with the same b/ρ shape. That column exists because the round-2
error has orbital structure of its own (composition term), so a non-zero offset_corr in B is not by itself leakage.
Convergence is unchanged (ŷ·K_eff,true − 1 inside 5 %); A cells with an offset never meet it because ŷ absorbs the
offset, so their accuracy window is days 7–14 by the main-spec rule. mc_draws stays 200 for F6 (24 cells take ~35 s).
F6(b) shows days 7–9, inside the accuracy window of both A (days 7–14) and B (after convergence).

**Verdict definitions (spec §6 wording left these open).**
- R3-1A "identical to 4 decimals": accuracy and precision equal when printed in % to 4 decimals, and convergence time
  equal to 1e-4 d. R3-1B: accuracy and precision within 0.01 pp of round 2.
- R3-2: all three of accuracy in [2 %, 8 %], precision ≤ 10 %, offset_corr > 0.5.
- R3-3: (precision > 10 % or accuracy > 10 %) AND "error largest where ρ_true is smallest" = RMS error over the
  lowest-ρ_true quartile of bins larger than over the highest quartile. Both parts are reported.
- R3-4: reference is variant A at (0, 0) at the same altitude (= the round-2 cell, the §1 "no-offset baseline");
  every B cell, (0, 0) included, must be within 1 pp on both accuracy and precision.
- R3-5: "worse" = both accuracy and precision larger at 600 km for every β ≠ (0, 0); both ratios are reported.
- C (not numbered in §6): "cannot separate" = |final posterior corr(y, c)| ≥ 0.99 or final σ(ĉ) ≥ true c.
- **Disclosure:** the R3-3 and R3-5 operational definitions and the C criterion were written after a diagnostic run of
  the (0, 0) and (0.3, 0.3) cells at both altitudes had been seen. The thresholds themselves (10 %, 1 pp, 0.5, 2–8 %)
  are the spec's.

**Tests.** The "ρ_corr with noise=False equals K g ρ exactly" test uses an offset that is piecewise linear between
zero-reads (a constant floor, and a linear ramp checked between the first and last zero-read), because linear
interpolation reproduces the lagged sunlit term only at the zero-read instants. Plumbing tests on 1-day runs check
structure and robust inequalities, not the 14-day physics numbers.

## Round 3 findings flagged for the reviewer
- The anchor does NOT leave a mean bias from the offset: it calibrates the drag-weighted arc integral, so the offset is
  absorbed into ŷ and appears as within-orbit error (precision) and a calibration factor far from 1/K_eff. R3-2 and
  R3-3 pre-registered an accuracy hit; the measured damage is in precision.
- B at 600 km does not recover the baseline: a single 1 s zero-read carries the gauge noise floor, which is a few
  percent of ρ at 600 km, and b̂ holds that error for 300 s. With noise-free zero-reads B recovers both altitudes.
  Averaging several shutter seconds per zero-read is the obvious mitigation; it was NOT run, so no claim.
- C separates c from y better than the spec expected (posterior corr high but clearly below 1; ĉ within about one posterior σ of the
  truth), because the storm and the daily m(t) term vary the arc-mean density. It still does not help the product:
  removing a per-arc constant leaves (b − c̄)/ρ_true, so C's precision is worse than A's.

## Round 3 open questions
- None blocking. Untested: zero-read averaging (N shutter seconds per read), a causal b̂, and noise that scales with
  total chamber pressure.

## Round 3b: averaged and causal shutter zero-reads (docs/round3b-zero-read-spec.md)

Every result number for this round is in `results/F7.json` and the generated "## F7" section of
`results/summary.md`; none is repeated here.

**Modeling assumptions (spec §3.2).**
- "fit" assumes the flight software knows the shadow geometry (GNSS position + sun ephemeris). Here s_model is
  computed from the truth orbit's cylindrical-shadow indicator, so GNSS position error and penumbra are ignored.
- "fit" assumes the functional form b = c0 + c1·s_model with s_model a first-order lag of the sunlit indicator with
  an ASSUMED lag τ_model. That form is the modeling assumption, and with τ_model = τ_b it is the truth's own form,
  so the matched fit is a best case for a correctly specified model. Only one mismatched lag, at one altitude, was
  run (the spec's probe). "hold" assumes nothing about b.
- Regressor of each read = mean of s_model over the read's closed window (the read is the window mean of b + n_z).

**Implementation choices the spec left open (fixed before the sweep ran).**
- Causality: a read becomes available at the first second after its last closed second (k·t_zero + N). b̂ at the
  closed seconds is irrelevant: those samples are filled for the anchor and skipped in the product, as in round 3.
- Before the first read, hold and fit use b̂ = 0 as the spec says. The spec calls that the round 3 convention; it
  is not quite: round 3's interp (np.interp) held the FIRST read backwards, which is non-causal. interp keeps the
  round 3 behaviour unchanged so that N = 1 reproduces F6 bit for bit (R3b-6).
- Rank guard: fit keeps holding until at least two reads exist AND the std of their regressors reaches
  `gauge.FIT_MIN_SPREAD` (value in F7.json `config`). Without it, two reads in the same illumination state give an
  ill-posed slope that is extrapolated across the next terminator. This tightens the spec's "two reads" fallback.
- Noise: n_z is drawn per closed second in time order from ZR_SEED, so N = 1 consumes the round 3 draw sequence.
  A window cut by the end of the series gives no read; its seconds are still skipped.
- Gaps: a 30 s bin with fewer than `gauge.MIN_VALID` valid samples is NaN and is excluded from every metric
  (accuracy, precision, sunlit/eclipse splits, offset_corr, mean_err, uncalibrated accuracy). Applied to variant B
  only. The gap fraction is over all 30 s bins of the 14-day run.
- b̂ RMS: RMS of (b̂ − b)/ρ_ref over the 1 Hz non-shutter seconds with t ≥ 1 d. The spec asks it for the causal
  modes; it is also reported for interp as a reference. "duty lost" in the table is the measured skipped fraction.
- R (filter measurement variance) is the round-2 Monte Carlo, unchanged, as in round 3.
- The no-offset baseline is quoted from F6.json (variant A at (0, 0)) and copied into F7.json `baseline_from_F6`.
- R3b-6 is checked in code: every N = 1 interp cell is compared with the stored F6 variant-B cell on every shared
  scalar key, and every file under results/ at efe7628 is compared byte for byte with the working copy (summary.md
  as an unchanged prefix, since the F7 section is appended).

**Verdict definitions.** "Within X pp" = |Δprecision| ≤ X pp. R3b-3 "any N" = every N in the sweep, at both β
(both have β1 > 0); hold must be worse than interp by more than 2 pp in every one of those cells. Each verdict
states the direction of a miss. These were written into `run._f7_section` before the first sweep run; they were
committed together with the results, so the ordering rests on this note, not on git history.

**Decided after seeing data (presentation only, no threshold or cell changed).**
- Verdict-row deltas print to 3 decimals, because a miss just under 2 pp printed as "+2.00 pp" at 2 decimals.
- The one-paragraph reading at the end of the F7 section was written after the sweep; its numbers and its
  PASS/FAIL wording are generated from F7.json and conditioned on the measured margins.

**Timebox.** Every main-sweep cell and the probe ran; no cell was dropped.

## Round 3b findings flagged for the reviewer
- R3b-3 FAILS by margin, not by mechanism: hold is worse than interp at 400 km in every cell, by about the
  pre-registered amount, but under the 2 pp bar in most cells (see the F7 verdict table for the exact deltas).
- R3b-4 PASSES with fit on the BETTER side of interp at both altitudes: the causal fit pools every read under a
  correctly specified model, while interp uses two reads. This is a best case (matched lag, known form).
- R3b-5 PASSES with a thin margin (see the table). One lag mismatch at one altitude does not show the lag can be
  assumed; the proposal should still treat τ_b as a bench-measured parameter.
- R3b-1 and R3b-2 PASS: averaging N closed seconds restores the 600 km product with the non-causal estimator, at the
  duty cost and (at the largest N) the gap fraction reported in the table.

**CI (2026-10-18).** A GitHub Actions gate (`.github/workflows/test.yml`, ubuntu, uv, Python 3.12) runs the full suite from a fresh clone (no truth cache, ~80 s). Two tests were loosened for it: the MSIS reference check from 1e-6 to 1e-5 relative, and the R3b-6 regression from exact equality to 1e-6 relative or 1e-8 absolute on floats (n_bins exact; accuracy is a near-cancelling mean of ~5e-4, where 6e-10 is already 1e-6 relative), because Linux x86_64 and macOS arm64 differ in the last bits of the pymsis evaluation (max 1.05e-6 relative in density). The stored results were produced on macOS arm64 and the exact-equality claims in this file hold there.
