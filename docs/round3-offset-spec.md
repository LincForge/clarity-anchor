# Round 3 spec — additive, sun-correlated gauge offset and the shutter zero-read

Date: 2026-10-18. Trigger: Schultz et al., STAIF-1998 (AIP Conf. Proc. 420, 724-728), the WSF-03
November 1996 flight of a ram concentrator over an ion gauge, reports an unexplained
"photodesorption mechanism" on top of the once-per-orbit density swing. The round 1-2 gauge model
(`K g ρ + n`) has no additive term, so that systematic is unmodeled. This round adds it and tests the
one mitigation the hardware already has: the ram-baffle shutter as a periodic zero-point read.

## 1. The claim this round must prove or kill

An additive chamber background b(t) (wall outgassing plus a sunlit photodesorption term) cannot be
removed by the orbit anchor alone, because the anchor yields one scalar per arc and a sun-correlated
offset is partly degenerate with the real day/night density swing. A periodic shutter zero-read
(chamber inflow blocked, residual pressure sampled) removes it, and the calibrated product then
meets the benchmark within about 1 percentage point of the no-offset baseline.

If the zero-read variant does NOT recover the baseline, that is the finding and the proposal must
say so.

## 2. Truth change (gauge model, spec §2 amendment)

    ρ_meas(t) = K(t) g(t) ρ_true(t) + b(t) + n(t)
    b(t)      = ρ_ref · [ β0 + β1 · s(t) ]
    s(t)      = first-order lag of the sunlit indicator ν(t) (orbit.shadow_nu, cylindrical shadow)
                with time constant τ_b = 300 s (wall heating; assumption, label it)
    ρ_ref     = median of ρ_true over day 0-1 at that altitude

β0 is the constant outgassing floor, β1 the sunlit photodesorption term, both as fractions of the
typical chamber density. The 1998 paper gives no number for either, so they are swept:

    (β0, β1) ∈ { (0, 0), (0.1, 0.1), (0.3, 0.3), (0, 0.3) }

(0, 0) must reproduce the round 2 numbers exactly (regression test). Note in DECISIONS.md that the
physical offset is an absolute pressure, so the same outgassing is a larger fraction at higher
altitude; the sweep is relative so one β maps to a different physical pressure per altitude.

Noise n is unchanged. The zero-read measurement (variant B below) carries its own noise draw:
b_meas = b(t_z) + n_z, n_z ~ N(0, (NOISE_REL · b(t_z) + NOISE_FLOOR)²).

Guard: β0, β1 must be finite and ≥ 0, ρ_ref finite and > 0; raise ValueError otherwise (DbC, daemon
zone: fail fast). Add the hostile-input test.

## 3. Estimator variants

- **A (baseline, no mitigation):** round 2 estimator unchanged, fed ρ_meas with the offset. This is
  the damage figure.
- **B (shutter zero-read):** every T_zero = 300 s the shutter closes for 1 s; the 1 Hz sample at
  that second is b_meas (the density product skips that sample, 0.3 % data loss). The estimator
  builds b̂(t) by linear interpolation between zero-reads (hold before the first), forms
  ρ_corr = ρ_meas − b̂, and runs the unchanged round 2 anchor + filter on ρ_corr. ρ_cal = ŷ · ρ_corr.
- **C (stretch, only if A and B are done inside the timebox):** 2-state filter [y, c] with no
  zero-reads, where c is a per-arc additive term in I_gauge (I_gauge = (1/y)·I_true + c·∫w dt).
  Purpose: show the anchor alone cannot separate c from y. Report the posterior correlation and
  the accuracy. If skipped, write "UNMEASURED (not run)" in the table, never a guess.

## 4. Metrics (additions)

Keep accuracy/precision/resolution/convergence as in §5 of the main spec. Add per cell:
- `acc_sunlit`, `acc_eclipse`: |mean| of e_rho over bins with s > 0.5 and s < 0.5 after convergence.
- `prec_sunlit`, `prec_eclipse`: the matching std.
- `offset_corr`: Pearson correlation between e_rho and b(t)/ρ_true(t) after convergence (variant A
  should show a strong positive value; variant B near 0).

## 5. Scenarios and figure

Cells: h = 400 km (headline) and 600 km (offset matters more where ρ is small), good POD, 1-orbit
arcs, ε_B = 0, gauge seed 10, four (β0, β1) pairs, variants A and B (and C if run). 16 cells (24
with C). Reuse the cached truth; do not regenerate F1-F5 JSONs.

F6 (`results/F6.png`, `results/F6.json`):
- (a) 400 km, (0.3, 0.3): ρ_true, ρ_meas, b(t) over orbits 20-23 (sunlit shading).
- (b) 400 km, (0.3, 0.3): 30 s relative error after convergence, variant A vs B, 2 days.
- (c) accuracy and precision vs (β0, β1) for A and B at 400 and 600 km, with the 20 % / 10 % lines.

`results/summary.md`: new section "F6: additive sun-correlated offset" with the cell table
(h, β0, β1, variant, convergence, accuracy, precision, acc_sunlit, acc_eclipse, offset_corr, verdicts)
and the pre-registered expectations below. README "Not modeled" list: replace the absent entry with
"Additive chamber background: modeled in round 3 (F6) as a constant plus a sunlit term, sizes assumed,
see docs/round3-offset-spec.md"; keep every other bullet.

## 6. Pre-registered expectations (write results against these)

| # | Expectation |
|---|---|
| R3-1 | (0, 0), variants A and B: identical to round 2 to 4 decimals (B differs only by the skipped 1 s samples; allow 0.01 pp). |
| R3-2 | A at (0.1, 0.1), 400 km: accuracy 2-8 %, precision still PASS; offset_corr > 0.5. |
| R3-3 | A at (0.3, 0.3), 400 km: precision FAILS (> 10 %) or accuracy > 10 %; the error is largest where ρ_true is smallest. |
| R3-4 | B at every (β0, β1), both altitudes: accuracy and precision within 1 pp of the (0, 0) cell. |
| R3-5 | 600 km variant A is worse than 400 km at the same β (longer eclipse fraction is not the reason; the storm and m(t) field are). State the measured ratio, no claim beyond "worse". |

## 7. Process

- TDD. Tests first: offset generator (0 ⇒ old model; finite/non-negative guard), zero-read
  extraction (noise=False recovers b exactly), ρ_corr with noise=False equals K g ρ exactly, F6
  plumbing. Existing 32 tests stay green.
- Commit after tests, after variant A, after variant B, after F6 + summary. Push each commit to
  origin main (no CI in this repo; the reviewer runs pytest locally).
- Timebox 2 h wall clock. If C does not fit, skip it and say so in summary.md.
- Every number in summary.md comes from a JSON in results/. No number is typed by hand.
