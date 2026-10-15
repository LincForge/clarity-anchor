# Simulation spec — orbit-anchored calibration of a cheap in-situ density gauge

Status: DRAFT for CEO nod · 2026-10-15 · author: Fable 5.1 · executor: Opus 5.5 (TDD) · reviewer: Fable

## 1. The claim this simulation must prove or kill

A cheap gauge has good precision but poor absolute accuracy (±30 % ground calibration) and drifts.
The host spacecraft's GNSS precise orbit (POD) yields a drift-free measure of integrated drag work
over each orbital arc. A filter that continuously fits the gauge's scale factor to the POD drag
work makes the gauge output meet NASA's in-situ density benchmark (accuracy ≤ 20 %, precision
≤ 10 %, resolution ≤ 30 s) at 300–600 km, converging within days, and the only irreducible
accuracy term is the host ballistic-coefficient model error, which the method inherits 1:1.

If the last sentence is false for a realistic host (for example, convergence takes months at
600 km, or the POD noise swamps the per-arc decay), the proposal must not make the claim.
Negative results are a valid deliverable.

## 2. Physics model (keep it this simple, no simpler)

**Frame and orbit.** ECI, Earth rotating at ω_E. Host in a near-circular orbit, inclination 87°,
altitudes h ∈ {300, 400, 500, 600} km. Propagate truth with RK4 at 5 s step: point-mass gravity +
J2 + drag + cannonball SRP with a cylindrical Earth shadow. Duration 14 days.

**Atmosphere (truth).** Tier 1: NRLMSIS 2.0 via `pymsis`, called with explicit `f107=150,
f107a=150, ap=15` (no network at run time). Multiply by a synthetic "model-error field"
m(t) = 1 + 0.15·sin(2π t/T_orb + φ1) + 0.10·sin(2π t/86400 + φ2), and by a storm bump
1 + 1.5·exp(−((t − t_s)/6 h)²) centred on day 5. The truth is deliberately not the clean model.
Return species number densities too (N2, O2, O, He, H) for the gauge composition term.
Tier 2 (stretch): replace the truth series with TU Delft CHAMP or GRACE density files
(https://thermosphere.tudelft.nl/, 10 s cadence) interpolated in time along a propagated orbit
matched in altitude; label figures from this tier clearly as "real-density truth".

**Drag.** a_drag = −½ B ρ |v_rel| v_rel with v_rel = v − ω_E × r (corotating atmosphere).
B_true = C_D A/m with A = 1.0 m², m = 150 kg, C_D(h) = 2.2 + 0.4·(h − 300 km)/300 km.
B_model = B_true·(1 + ε_B) with ε_B ∈ {0, 0.05, 0.10, 0.20}: this is the C_D model error the
retrieval must inherit.

**SRP.** a_srp = −P_sun · C_R · A/m · ŝ · ν (ν = 0 in shadow), P_sun = 4.56e-6 Pa, C_R = 1.3.
The estimator models SRP with C_R·(1 + 0.10) to inject a realistic 10 % model error.

**Gauge.** 1 Hz samples. ρ_meas = K(t)·g(t)·ρ_true + n. K(0) ~ U(0.7, 1.3). K random-walks with
σ = 3 % per week. g(t) is the composition sensitivity factor: Σ_i s_i·ρ_i / Σ_i ρ_i normalised to
1 at t = 0, with cold-cathode relative sensitivities s = {N2: 1.00, O2: 1.00, O: 0.80, He: 0.18,
H: 0.45}. n is white, σ = 2 % of ρ_true plus an absolute floor of 1e-14 kg/m³ (gauge lower
limit after ram compression). Deliver ρ_meas averaged to 30 s bins as well.

**POD.** Truth state at 10 s plus white noise: "good" = 5 cm position, 0.05 mm/s velocity per
axis; "COTS" = 20 cm, 0.2 mm/s. Maneuvers are out of scope (arcs with thrust are excluded in
practice).

## 3. Anchor observable (the heart of it)

Per arc of length T_arc (default one orbit; also 3 and 10 orbits), using the SAME gravity model
as truth:

  ΔE_pod   = E(t_end) − E(t_start),  E = |v|²/2 − μ/r − (J2 potential term)
  W_srp    = ∫ a_srp,model · v dt                      (estimator's SRP, 10 % wrong)
  W_drag   = ΔE_pod − W_srp                            (non-conservative residual)
  I_pod    = −2 W_drag / B_model = (1+ε_B)^(−1) ∫ ρ_true |v_rel| (v_rel·v) dt
  I_gauge  = ∫ ρ_meas |v_rel| (v_rel·v) dt              (same weighting, from the 1 Hz gauge)
  z_inv    = I_pod / I_gauge ≈ 1 / K_eff,   K_eff = K·ḡ_arc·(1 + ε_B)

*(amended 2026-10-15, reviewer-approved)* Three corrections to the original draft:
(i) the weighting kernel is the exact drag power per unit ½Bρ, |v_rel|(v_rel·v); the original
|v_rel|²(v_rel·v̂) differs by |v_rel|/|v| (≈0.3 %) and cannot give a ratio of 1 to 1e-6;
(ii) I_pod = I_true/(1+ε_B), so K_eff = K·ḡ·(1+ε_B), not K·ḡ·(1+ε_B)^(−1), and ρ_cal inherits a
bias −ε_B/(1+ε_B); (iii) the measurement is the inverse ratio z_inv = I_pod/I_gauge, which puts
the POD noise in the numerator. The original z = I_gauge/I_pod carried POD noise in the denominator
and was biased high by ≈σ_z² (+4.3 % at 600 km with 1-orbit arcs).

Both integrals carry the same weighting, so the ratio is exact apart from noise and the
B/SRP model errors. Verify in a test that with ε_B = 0, no POD noise and K ≡ 1, z_inv = 1 to 1e-6.

**POD noise propagation.** Compute the per-arc std of z_inv empirically from 200 Monte Carlo
noise draws on one fixed arc at each altitude and POD grade; use it as the filter's R.

## 4. Estimator

Scalar Kalman filter on y = 1/K_eff (K_eff = K·ḡ·(1+ε_B)), prior y0 = 1, P0 = 0.3², random-walk
process noise matched to the 3 %/week drift, measurement z_inv with variance R from §3. Output
ρ_cal(t) = ρ_meas(t)·ŷ(t) at 30 s. *(amended 2026-10-15, reviewer-approved: state and measurement
inverted to remove the ratio-estimator bias; see §3.)* Stretch: add a bias state b
(ρ_meas = K ρ + b) and show whether it is observable.

No UKF, no attitude, no torques. If the scalar filter cannot meet the benchmark, nothing fancier
will rescue the proposal in four weeks.

## 5. Scenarios and figures

| Fig | Scenario | Shows |
|---|---|---|
| F1 | h ∈ {300,400,500,600}, good POD, ε_B = 0, T_arc = 1 orbit | K̂ relative error vs time; convergence time to 5 % |
| F2 | h = 400, good POD, ε_B = 0.10, with storm | ρ_cal vs ρ_true relative error time series (30 s) + histogram after convergence |
| F3 | h = 400, good POD, ε_B ∈ {0, .05, .10, .20} | post-convergence accuracy vs ε_B (expect a 1:1 line) |
| F4 | h ∈ {300…600} × POD {good, COTS} × T_arc {1,3,10} | convergence time grid (days) |
| F5 (stretch) | 5 hosts, independent ε_B ~ N(0, 0.10), shared K_lot | fleet-mean accuracy vs N (expect 1/√N) |

`results/summary.md` reports, per scenario, accuracy (|mean relative error| of 30 s ρ_cal after
convergence), precision (std of relative error), convergence time, and PASS/FAIL against
20 % / 10 % / 30 s. Mark anything unmeasured as UNMEASURED, never as PASS.

## 6. Pre-registered expectations (write results against these, not after)

- At 400 km with good POD, K̂ converges to < 5 % within 1 day; precision < 5 %.
- At 600 km, per-orbit decay is ~0.3 m, so convergence needs multi-orbit arcs; expect 5–10 days
  with T_arc = 10 orbits.
- Accuracy tracks ε_B 1:1. With ε_B = 0.10, accuracy ≈ 10 %: PASS against 20 %.
  *(Measured, round 1: accuracy follows ε_B/(1+ε_B), not 1:1 — 4.75 / 9.08 / 16.66 % at
  ε_B = 0.05 / 0.10 / 0.20; see the §3 amendment.)*
- The composition term g(t) shows up as a slow K drift of a few percent per day around the storm
  and is tracked, not rejected.

## 7. Repository and process

- New private repo `LincForge/clarity-anchor` (HTTPS remote), Python 3.12, `uv`, `numpy`,
  `scipy`, `pymsis`, `matplotlib`, `pytest`. Package `clarity_anchor/` with modules `orbit`,
  `atmosphere`, `gauge`, `pod`, `anchor`, `filter`, `metrics`, `scenarios`; CLI
  `python -m clarity_anchor.run --fig F1`.
- TDD. Analytic tests before code: (a) constant ρ, circular orbit, no J2 → da/dt = −B ρ √(μ a)
  within 1 %; (b) noiseless anchor ratio = 1 to 1e-6; (c) K = 1.2 noiseless → K̂ = 1.2 after one
  arc; (d) J2 energy is conserved to 1e-9 relative over 14 days with drag and SRP off.
- Deterministic seeds. Every figure regenerable from one command. Commit results PNGs.
- Timebox: two Opus sessions. Stop and report if the Tier 1 figures F1–F4 are not done by then.
- Review gate: Fable reads the physics modules and the summary before anything is quoted in the
  proposal. The proposal cites only numbers that appear in `results/summary.md`.

## 8. Out of scope (do not build)

Attitude dynamics, reaction-wheel torque inversion, chamber gas dynamics, species TOF fitting,
UKF, real GNSS processing, CubeSat bus interfaces.
