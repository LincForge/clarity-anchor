# Round 3b spec — averaged and causal shutter zero-reads

Date: 2026-10-18. Trigger: round 3 (docs/round3-offset-spec.md, results/F6.json) left two caveats on
the shutter zero-read (variant B). First, a single 1 s zero-read carries the gauge noise floor
(1e-14 kg/m³ = 0.26 % of ρ_ref at 400 km, 5.37 % at 600 km) and b̂ holds that error for 300 s, so
600 km precision ended at 4.9–5.5 % against a 2.89 % no-offset baseline, while noise-free reads
matched the baseline at both altitudes. Second, B's b̂ is linear interpolation between the previous
and the NEXT zero-read, so the product has up to 300 s latency and is not what flight software can
compute. This round fixes both and reports what each fix costs.

## 1. The claim this round must prove or kill

Averaging N shutter-closed seconds per zero-read brings the 600 km product back within 1 percentage
point of the no-offset baseline, and a causal estimate of b̂ (no look-ahead) does so without the
latency. If the causal estimate cannot match the interpolated one, that is the finding and the
proposal says so.

## 2. Truth: unchanged from round 3

Same ρ_meas = K g ρ_true + b(t) + n, same b(t) = ρ_ref (β0 + β1 s(t)), same τ_b = 300 s, same
noise, same seeds (GAUGE_SEED=10, ZR_SEED=31, POD_SEED good). Nothing in `atmosphere.py`,
`orbit.py`, `pod.py`, `anchor.py` or `filter.py` changes. F1–F6 must stay byte-identical (`git diff
--stat` against efe7628 on results/ must list only the new files).

## 3. What changes: the zero-read (gauge.py) and its estimator

### 3.1 Zero-read window of N seconds

    zero_read_mask(t, t_zero=300, n_zero=N): closed seconds t ∈ [k·t_zero, k·t_zero + N), k ≥ 1
    zero_read(...): one b_meas per period = mean over the N closed seconds of (b(t) + n_z(t)),
                    n_z ~ N(0, (NOISE_REL · b(t) + NOISE_FLOOR)²) drawn per second from ZR_SEED
    timestamp of the read = centre of the window, k·t_zero + (N−1)/2

N = 1 must reproduce round 3 variant B bit for bit (regression test on the F6 400 km (0.3, 0.3)
cell: identical precision and accuracy to the stored F6.json values).

Sweep N ∈ {1, 10, 30}. Duty cycle lost to the shutter is N/300 (0.3 %, 3.3 %, 10 %). Report, per
cell, the fraction of 30 s product bins with fewer than 15 valid samples; such bins are reported as
gaps (NaN), not as values, and excluded from acc/prec. At N = 30 one bin in ten is a gap by
construction. Say so in the summary.

Guard: n_zero must be an int with 1 ≤ n_zero < t_zero, else ValueError (DbC, daemon zone).

### 3.2 Three b̂ estimators (`mode`)

    "interp": round 3 behaviour, linear between the previous and the next read (non-causal). Kept as
              the reference.
    "hold":   causal, b̂(t) = last completed read, held until the next completes. Before the first
              read b̂ = 0 (the round 3 convention).
    "fit":    causal, b̂(t) = ĉ0 + ĉ1 · s_model(t), where (ĉ0, ĉ1) are refit by ordinary least squares
              on ALL completed reads so far (recursive, no look-ahead), and s_model is the lagged
              sunlit indicator the flight software can compute from the orbit and the sun vector
              with an ASSUMED lag τ_model. Before two reads exist, fall back to "hold". τ_model =
              300 s (matched to truth) in the main sweep. One probe cell with τ_model = 150 s tests
              the mismatch (see §5).

"fit" assumes the flight software knows the shadow geometry (it does: GNSS position + sun ephemeris)
and the functional form of b(t). The form is the modeling assumption; say so in DECISIONS.md.
"hold" assumes nothing.

In every mode the skipped seconds inside the window are filled for the anchor integral by linear
interpolation of ρ_corr neighbours (round 3 convention) and left out of the 30 s product.

## 4. Metrics

Per cell, as in round 3: accuracy and precision (all / sunlit / eclipse), offset_corr, convergence
time, final calibration-factor error ŷ·K_eff,true − 1, plus: gap fraction (§3.1) and, for the causal
modes, the RMS of (b̂ − b)/ρ_ref over days 1–14 (how good the zero estimate itself is, separate from
the product).

## 5. Cells

Main sweep: altitude ∈ {400, 600} km, good POD, 1-orbit arcs, ε_B = 0, seed 10, β ∈ {(0.3, 0.3),
(0, 0.3)}, N ∈ {1, 10, 30}, mode ∈ {interp, hold, fit}. 36 cells. The no-offset baseline for each
altitude is the stored F6 (0, 0) variant A cell (quote it, do not re-run it).

Probe: 400 km, (0.3, 0.3), N = 10, mode fit, τ_model = 150 s (1 cell).

Priority if the timebox binds: all N for mode fit first, then interp, then hold; drop (0, 0.3) before
dropping an N; never drop the probe.

Figure F7 (results/F7.png, results/F7.json):
  (a) precision vs N, one line per mode, panels for 400 and 600 km, β = (0.3, 0.3); the no-offset
      baseline as a horizontal dashed line;
  (b) same for accuracy;
  (c) time series over two orbits around day 7 at 600 km, (0.3, 0.3), N = 10: (b̂ − b)/ρ_ref for
      hold and fit, with the terminator crossings marked.

Summary: append a "## F7" section to results/summary.md in the round 3 style (table, diagnostics,
reading). Every number from F7.json via write_summary; no hand-typed numbers. Update the README
not-modeled bullet to say what the zero-read now assumes (N closed seconds, causal fit with a known
shadow function). Add a round 3b section to DECISIONS.md naming the assumptions in §3.2 and anything
decided after seeing data.

## 6. Pre-registered expectations (verdicts go in summary.md as PASS/FAIL with the direction)

  R3b-1  interp, N = 10: 600 km precision within 1.0 pp of the no-offset baseline (2.89 %), both β.
  R3b-2  interp, N = 30: 600 km precision within 0.5 pp of the baseline, both β.
  R3b-3  hold, β1 = 0.3, any N: precision WORSE than interp by more than 2 pp at 400 km, because
         s(t) changes by up to (1 − e^{−1}) β1 ρ_ref within one 300 s hold at the terminators. This
         is expected to FAIL the benchmark-level claim and is the reason "fit" exists.
  R3b-4  fit (τ_model = 300 s), N = 10: precision within 0.5 pp of interp at the same N, both
         altitudes, both β.
  R3b-5  fit probe (τ_model = 150 s): precision within 1.5 pp of the matched fit cell. If it fails,
         the proposal must present τ_b as a parameter to be measured on the bench, not assumed.
  R3b-6  Regression: N = 1 interp reproduces F6 variant B exactly; F1–F6 byte-identical.

## 7. Process

TDD: tests first (red), then gauge.py, then scenarios/run, then F7 and summary. Commit after tests,
after the gauge change, after the sweep, after F7 + summary; push each. 2 h timebox. Verdict
definitions are the ones above; if any verdict needs a definition not written here, record that it
was set after seeing data. Do not touch F1–F6 or the round 3 code paths except through the new
`n_zero` and `mode` parameters with defaults that reproduce round 3 (n_zero=1, mode="interp").
