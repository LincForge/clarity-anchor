"""CLI: python -m clarity_anchor.run --fig F1|F2|F3|F4|all   (writes results/*.png, *.json, summary.md)."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics, scenarios  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
ALTS = (300.0, 400.0, 500.0, 600.0)
GRADES = ("good", "cots")
ARCS = (1, 3, 10)
EPS = (0.0, 0.05, 0.10, 0.20)
ENSEMBLE_SEEDS = tuple(range(1, 11))  # gauge seeds for the F4 robustness ensemble (K0 ~ U(0.7, 1.3) each)
DAY = 86400.0


def _scalars(r):
    return {k: v for k, v in r.items() if k not in ("series", "wall_s")}  # wall_s: not deterministic


def _save_json(name, cells):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps([_scalars(c) for c in cells], indent=1, default=float))


def fig_f1():
    cells = [scenarios.run_cell(h, "good", 0.0, 1, keep_series=True) for h in ALTS]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for c in cells:
        s = c["series"]
        m = s["in_arcs"]
        lab = (f"{c['h_km']:.0f} km: conv {c['conv_days']:.2f} d" if c["converged"]
               else f"{c['h_km']:.0f} km: not within 5 % for the rest of 14 d")
        ax.plot(s["tb"][m] / DAY, 100 * s["e_k"][m], lw=0.8, label=lab)
    ax.axhspan(-5, 5, color="0.85", zorder=0, label="±5 %")
    ax.set_ylim(-30, 30)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("ŷ·K_eff,true − 1 (calibration error, %)")
    ax.set_title(f"F1 · calibration-factor error (ŷ = 1/K̂), good POD, ε_B = 0, 1-orbit arcs, K(0) = {cells[0]['K0']:.3f}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "F1.png", dpi=130)
    plt.close(fig)
    _save_json("F1", cells)
    return cells


def fig_f2():
    c = scenarios.run_cell(400.0, "good", 0.10, 1, keep_series=True)
    s = c["series"]
    t = s["tb"] / DAY
    fig = plt.figure(figsize=(10, 7.5))
    ax1 = fig.add_subplot(3, 1, 1)
    w = (t > 4.0) & (t < 6.0)
    ax1.semilogy(t[w], s["rho_b"][w], lw=0.8, label="ρ_true (30 s)")
    ax1.semilogy(t[w], s["rho_cal"][w], lw=0.6, alpha=0.8, label="ρ_cal (30 s)")
    ax1.set_ylabel("kg/m³")
    ax1.set_title("F2 · 400 km, good POD, ε_B = 0.10, storm at day 5 (window days 4–6)")
    ax1.legend(fontsize=8)
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t, 100 * s["e_rho"], lw=0.4)
    ax2.axhline(-100 * 0.10 / 1.10, color="k", ls="--", lw=0.8, label="−ε_B/(1+ε_B) = −9.1 %")
    if c["converged"]:
        ax2.axvline(c["conv_days"], color="r", lw=0.8, label=f"K̂ converged ({c['conv_days']:.2f} d)")
    ax2.set_ylim(-30, 30)
    ax2.set_xlabel("time (days)")
    ax2.set_ylabel("ρ_cal/ρ_true − 1 (%)")
    ax2.legend(fontsize=8)
    ax3 = fig.add_subplot(3, 1, 3)
    post = s["tb"] >= c["window_from_days"] * DAY
    ax3.hist(100 * s["e_rho"][post], bins=120, color="C2")
    ax3.axvline(-100 * 0.10 / 1.10, color="k", ls="--", lw=0.8)
    ax3.set_xlabel("ρ_cal/ρ_true − 1 after convergence (%)")
    ax3.set_ylabel("30 s bins")
    ax3.set_title(f"mean {100*c['mean_err']:.2f} %, std {100*c['precision']:.2f} %", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "F2.png", dpi=130)
    plt.close(fig)
    _save_json("F2", [c])
    return [c]


def fig_f3():
    cells = [scenarios.run_cell(400.0, "good", e, 1) for e in EPS]
    eps = np.array(EPS)
    acc = np.array([c["accuracy"] for c in cells])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    xx = np.linspace(0, 0.22, 50)
    ax.plot(100 * xx, 100 * xx, "k:", lw=1, label="1:1 (spec expectation)")
    ax.plot(100 * xx, 100 * xx / (1 + xx), "k--", lw=1, label="ε_B/(1+ε_B) (exact inheritance)")
    ax.plot(100 * eps, 100 * acc, "o", ms=7, color="C3", label="simulated accuracy")
    ax.axhline(20, color="r", lw=0.8, label="benchmark 20 %")
    ax.set_xlabel("ε_B (%)")
    ax.set_ylabel("post-convergence accuracy |mean rel. error| (%)")
    ax.set_title("F3 · 400 km, good POD, T_arc = 1 orbit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "F3.png", dpi=130)
    plt.close(fig)
    _save_json("F3", cells)
    return cells


def fig_f4():
    cells = []
    for h in ALTS:
        for g in GRADES:
            for n in ARCS:
                t0 = time.time()
                c = scenarios.run_cell(h, g, 0.0, n)
                c["wall_s"] = time.time() - t0
                cells.append(c)
                print(f"F4 cell h={h:.0f} {g} T_arc={n}: conv={c['conv_days']:.2f} d ({c['wall_s']:.1f}s)", flush=True)
    ens = []
    for h in ALTS:
        for g in GRADES:
            for n in ARCS:
                runs = [scenarios.run_cell(h, g, 0.0, n, gauge_seed=sd) for sd in ENSEMBLE_SEEDS]
                cd = np.array([r["conv_days"] for r in runs])
                ok = ~np.isnan(cd)
                ens.append({"h_km": h, "grade": g, "n_orb": n, "n_seeds": len(runs), "n_conv": int(ok.sum()),
                            "median_conv_days": float(np.median(cd[ok])) if ok.any() else float("nan"),
                            "max_conv_days": float(np.max(cd[ok])) if ok.any() else float("nan"),
                            "worst_accuracy": float(max(r["accuracy"] for r in runs)),
                            "worst_precision": float(max(r["precision"] for r in runs)),
                            "worst_acc_uncal": float(max(r["acc_uncal"] for r in runs)),
                            "K0s": [r["K0"] for r in runs]})
                print(f"F4 ensemble h={h:.0f} {g} T_arc={n}: {ok.sum()}/{len(runs)} conv", flush=True)
    cols = [(g, n) for g in GRADES for n in ARCS]
    grid = np.full((len(ALTS), len(cols)), np.nan)
    egrid = np.full((len(ALTS), len(cols)), np.nan)
    elab = np.empty((len(ALTS), len(cols)), dtype=object)
    for c in cells:
        grid[ALTS.index(c["h_km"]), cols.index((c["grade"], c["n_orb"]))] = c["conv_days"]
    for e in ens:
        i, j = ALTS.index(e["h_km"]), cols.index((e["grade"], e["n_orb"]))
        egrid[i, j] = e["median_conv_days"] if e["n_conv"] > e["n_seeds"] / 2 else np.nan
        elab[i, j] = (f"{e['median_conv_days']:.2f} d" if e["n_conv"] else "—") + f"\n{e['n_conv']}/{e['n_seeds']} conv"
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, gr, title, lab in ((axes[0], grid, f"main run (gauge seed {scenarios.GAUGE_SEED}, K(0) = {cells[0]['K0']:.3f})", None),
                              (axes[1], egrid, f"ensemble of {len(ENSEMBLE_SEEDS)} gauge seeds: median over converged runs", elab)):
        shown = np.where(np.isnan(gr), 14.0, gr)
        im = ax.imshow(shown, cmap="viridis_r", vmin=0, vmax=14, aspect="auto")
        for i in range(gr.shape[0]):
            for j in range(gr.shape[1]):
                v = gr[i, j]
                txt = lab[i, j] if lab is not None else ("not conv.\n(14 d)" if np.isnan(v) else f"{v:.2f} d")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if shown[i, j] > 7 else "black")
        ax.set_xticks(range(len(cols)), [f"{g}\n{n} orb" for g, n in cols])
        ax.set_yticks(range(len(ALTS)), [f"{h:.0f} km" for h in ALTS])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, label="days (dark/14 = majority not converged)")
    fig.suptitle("F4 · convergence time to 5 % (days), ε_B = 0, 14-day runs")
    fig.tight_layout()
    fig.savefig(RESULTS / "F4.png", dpi=130)
    plt.close(fig)
    _save_json("F4", cells)
    (RESULTS / "F4_ensemble.json").write_text(json.dumps(ens, indent=1, default=float))
    return cells


F5_HOSTS = 5
F5_DRAWS = 40
F5_SIGMA_EPS = 0.10


def fig_f5():
    """Fleet: 5 hosts at 400 km (good POD, 1-orbit arcs) share one gauge lot K path; eps_B,j ~ N(0, 0.10).
    Host 1 calibrates with the mean K_hat of the first N hosts; accuracy vs N over F5_DRAWS fleets."""
    from . import gauge

    truth = scenarios.build_truth(400.0)
    rng = np.random.default_rng(500)
    signed = np.empty((F5_DRAWS, F5_HOSTS))
    for d in range(F5_DRAWS):
        K1 = gauge.k_random_walk(truth.t1, gauge.draw_k0(rng), gauge.DRIFT_SIGMA_WEEK, rng)
        eps = rng.normal(0.0, F5_SIGMA_EPS, F5_HOSTS)
        hosts = [scenarios.run_cell(400.0, "good", float(eps[j]), 1, keep_series=True, k_path=K1,
                                    noise_seed=10_000 + 100 * d + j, pod_seed=20_000 + 100 * d + j, mc_draws=50)
                 for j in range(F5_HOSTS)]
        s0 = hosts[0]["series"]
        tb, rho_b = s0["tb"], s0["rho_b"]
        meas_b = s0["meas_b"]
        win = tb >= 86400.0  # day 1 onwards (all hosts converged after one orbit at 400 km)
        idx = np.clip(np.searchsorted(s0["t_arc"], tb, side="right") - 1, 0, None)
        for n in range(1, F5_HOSTS + 1):
            kfleet = np.mean([h["series"]["khat"] for h in hosts[:n]], axis=0)[idx]
            signed[d, n - 1] = np.mean((meas_b / kfleet / rho_b - 1.0)[win])
        print(f"F5 fleet draw {d+1}/{F5_DRAWS}", flush=True)
    N = np.arange(1, F5_HOSTS + 1)
    rms = np.sqrt(np.mean(signed**2, axis=0))
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(N, 100 * rms, "o-", color="C3", label=f"simulated RMS accuracy ({F5_DRAWS} fleets)")
    ax.plot(N, 100 * F5_SIGMA_EPS / np.sqrt(N), "k--", lw=1, label="0.10/√N")
    ax.axhline(20, color="r", lw=0.8, label="benchmark 20 %")
    ax.set_xlabel("hosts averaged (N)")
    ax.set_ylabel("RMS of fleet-calibrated mean rel. error (%)")
    ax.set_title("F5 · fleet mean, 400 km, good POD, ε_B ~ N(0, 0.10), shared K_lot")
    ax.set_xticks(N)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "F5.png", dpi=130)
    plt.close(fig)
    out = [{"N": int(n), "rms_accuracy": float(r), "expected": float(F5_SIGMA_EPS / np.sqrt(n)),
            "draws": F5_DRAWS} for n, r in zip(N, rms)]
    (RESULTS / "F5.json").write_text(json.dumps(out, indent=1))
    return out


F6_ALTS = (400.0, 600.0)
F6_BETAS = ((0.0, 0.0), (0.1, 0.1), (0.3, 0.3), (0.0, 0.3))
F6_VARIANTS = ("A", "B", "C")


def _f6_key(c):
    return (c["h_km"], c["beta0"], c["beta1"], c["variant"])


def fig_f6():
    """Round 3: additive sun-correlated offset b(t) (docs/round3-offset-spec.md). 400/600 km, good POD, 1-orbit arcs,
    eps_B = 0, gauge seed 10, four (beta0, beta1) pairs, variants A (no mitigation), B (shutter zero-read) and
    C (stretch: 2-state [y, c] filter, no zero-reads)."""
    from . import gauge

    cells, diag_b0, series = [], [], {}
    for h in F6_ALTS:
        for beta in F6_BETAS:
            for v in F6_VARIANTS:
                keep = beta in ((0.0, 0.0), (0.3, 0.3))
                c = scenarios.run_cell(h, "good", 0.0, 1, beta=beta, variant=v, keep_series=keep)
                if keep:
                    series[(h, beta, v)] = c.pop("series")
                cells.append(c)
                print(f"F6 cell h={h:.0f} beta={beta} {v}: acc={100*c['accuracy']:.2f} % prec={100*c['precision']:.2f} %", flush=True)
            # diagnostic: B with noise-free zero-reads (attributes B's cost to the zero-read noise vs interpolation)
            d = scenarios.run_cell(h, "good", 0.0, 1, beta=beta, variant="B", zero_noise=False)
            diag_b0.append(d)
    # diagnostic: correlation the NO-offset product already has with each b/rho shape (same window as the cell)
    for c in cells:
        s00 = series[(c["h_km"], (0.0, 0.0), "A")]
        _, s_b = gauge.bin_mean(s00["t1"], s00["s1"], 30.0)
        shape = c["rho_ref"] * (c["beta0"] + c["beta1"] * s_b) / s00["rho_b"]
        post = s00["tb"] >= c["window_from_days"] * DAY
        c["offset_corr_baseline"] = (float(np.corrcoef(s00["e_rho"][post], shape[post])[0, 1])
                                     if np.std(shape[post]) > 0 else float("nan"))

    sA, sB = series[(400.0, (0.3, 0.3), "A")], series[(400.0, (0.3, 0.3), "B")]
    T_orb = scenarios.build_truth(400.0).T_orb
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, :])
    t1 = sA["t1"]
    w = (t1 >= 20 * T_orb) & (t1 < 24 * T_orb)
    th = t1[w] / 3600.0
    ax.plot(th, sA["meas1"][w], lw=0.3, color="0.6", label="ρ_meas = K g ρ + b + n (1 Hz)")
    ax.plot(th, sA["rho1"][w], lw=1.0, color="C0", label="ρ_true")
    ax.plot(th, sA["b1"][w], lw=1.2, color="C3", label="b(t) = ρ_ref(β0 + β1 s)")
    ax.fill_between(th, 0, 1, where=sA["nu1"][w] > 0.5, transform=ax.get_xaxis_transform(), color="gold",
                    alpha=0.15, lw=0, label="sunlit (ν = 1)")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("kg/m³")
    ax.set_title("F6 (a) · 400 km, (β0, β1) = (0.3, 0.3), orbits 20–23")
    for ln in ax.legend(fontsize=8, loc="upper right", ncol=2).get_lines():
        ln.set_linewidth(2.0)
    ax = fig.add_subplot(gs[1, :])
    td = sA["tb"] / DAY
    w = (td >= 7.0) & (td < 9.0)
    ax.plot(td[w], 100 * sA["e_rho"][w], lw=0.5, color="C3", label="A: no mitigation")
    ax.plot(td[w], 100 * sB["e_rho"][w], lw=0.5, color="C2", label="B: shutter zero-read every 300 s")
    ax.axhspan(-10, 10, color="0.9", zorder=0, label="±10 % (precision benchmark)")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("ρ_cal/ρ_true − 1 (%, 30 s)")
    ax.set_title("F6 (b) · 400 km, (0.3, 0.3): 30 s relative error, days 7–9 (inside both accuracy windows)")
    for ln in ax.legend(fontsize=8, loc="upper right").get_lines():
        ln.set_linewidth(2.0)
    labels = [f"({b0:g}, {b1:g})" for b0, b1 in F6_BETAS]
    by = {_f6_key(c): c for c in cells}
    for j, (metric, lim, name) in enumerate((("accuracy", 20, "accuracy |mean| (%)"), ("precision", 10, "precision std (%)"))):
        ax = fig.add_subplot(gs[2, j])
        for h, ls in zip(F6_ALTS, ("-", "--")):
            for v, col in zip(F6_VARIANTS, ("C3", "C2", "C0")):
                y = [100 * by[(h, b0, b1, v)][metric] for b0, b1 in F6_BETAS]
                ax.plot(range(len(F6_BETAS)), y, ls, marker="o", color=col, label=f"{v}, {h:.0f} km")
        ax.axhline(lim, color="r", lw=0.8, label=f"benchmark {lim} %")
        ax.set_xticks(range(len(F6_BETAS)), labels)
        ax.set_xlabel("(β0, β1)")
        ax.set_ylabel(name)
        ax.set_title(f"F6 (c) · {metric} vs offset size (solid 400 km, dashed 600 km)", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "F6.png", dpi=130)
    plt.close(fig)
    RESULTS.mkdir(exist_ok=True)
    out = {"cells": [_scalars(c) for c in cells],
           "diagnostics": {"B_noiseless_zero_read": [_scalars(c) for c in diag_b0]}}
    (RESULTS / "F6.json").write_text(json.dumps(out, indent=1, default=float))
    return cells


# ---------------------------------------------------------------------------------------------
# Round 3b: averaged and causal shutter zero-reads (docs/round3b-zero-read-spec.md)
# ---------------------------------------------------------------------------------------------
F7_ALTS = (400.0, 600.0)
F7_BETAS = ((0.3, 0.3), (0.0, 0.3))
F7_N = (1, 10, 30)
F7_MODES = ("fit", "interp", "hold")  # spec §5 priority order
F7_PROBE = {"h_km": 400.0, "beta": (0.3, 0.3), "n_zero": 10, "zr_mode": "fit", "tau_model": 150.0}
F7_REF_COMMIT = "efe7628"  # round 3 results commit: F1-F6 must stay byte-identical to it
F7_SERIES = (600.0, (0.3, 0.3), 10)  # panel (c): hold and fit at this (h, beta, N)


def _f7_key(c):
    return (c["h_km"], c["beta0"], c["beta1"], c["n_zero"], c["zr_mode"], c["tau_model"])


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _results_identical_to(commit):
    """Byte comparison of every results/ file tracked at `commit` (F1-F6, ensembles, decay signal, summary.md is
    checked separately as an append-only prefix)."""
    import subprocess

    root = RESULTS.parent
    names = subprocess.run(["git", "ls-tree", "--name-only", commit, "results/"], cwd=root, check=True,
                           capture_output=True, text=True).stdout.split()
    out = {}
    for n in names:
        old = subprocess.run(["git", "show", f"{commit}:{n}"], cwd=root, check=True, capture_output=True).stdout
        cur = (root / n).read_bytes() if (root / n).exists() else None
        if n.endswith("summary.md"):
            out[n] = cur is not None and cur.startswith(old)
        else:
            out[n] = cur == old
    return out


def f7_sweep():
    """Round 3b sweep: 36 main cells (h x beta x N x mode, tau_model = TAU_B) + 1 probe (tau_model = 150 s).
    Returns (cells, probe, series) where series holds the 1 Hz b_hat of the panel-(c) cells."""
    from . import gauge

    cells, series = [], {}
    for mode in F7_MODES:
        for beta in F7_BETAS:
            for n in F7_N:
                for h in F7_ALTS:
                    keep = (h, beta, n) == F7_SERIES and mode in ("hold", "fit")
                    c = scenarios.run_cell(h, "good", 0.0, 1, beta=beta, variant="B", n_zero=n, zr_mode=mode,
                                           keep_series=keep)
                    if keep:
                        series[mode] = c.pop("series")
                    cells.append(c)
                    print(f"F7 cell h={h:.0f} beta={beta} N={n} {mode}: acc={100*c['accuracy']:.2f} % "
                          f"prec={100*c['precision']:.2f} % gap={100*c['gap_frac']:.1f} %", flush=True)
    pr = F7_PROBE
    probe = scenarios.run_cell(pr["h_km"], "good", 0.0, 1, beta=pr["beta"], variant="B", n_zero=pr["n_zero"],
                               zr_mode=pr["zr_mode"], tau_model=pr["tau_model"])
    print(f"F7 probe: acc={100*probe['accuracy']:.2f} % prec={100*probe['precision']:.2f} %", flush=True)
    if not all(c["tau_model"] == gauge.TAU_B for c in cells):
        raise RuntimeError("F7 main sweep must use tau_model = TAU_B")
    return cells, [probe], series


def fig_f7():
    from . import gauge

    cells, probe, series = f7_sweep()
    f6 = _load("F6")
    f6by = {_f6_key(c): c for c in f6["cells"]}
    baseline = {f"{h:.0f}": {k: f6by[(h, 0.0, 0.0, "A")][k] for k in ("accuracy", "precision", "conv_days", "rho_ref")}
                for h in F7_ALTS}
    # R3b-6 part 1: every N = 1 interp cell vs the stored F6 variant-B cell, all shared scalar keys
    reg = []
    for c in cells:
        if c["n_zero"] != 1 or c["zr_mode"] != "interp":
            continue
        ref = f6by[(c["h_km"], c["beta0"], c["beta1"], "B")]
        keys = [k for k in ref if k in c and k != "offset_corr_baseline"]
        bad = [k for k in keys if not _same(c[k], ref[k])]
        reg.append({"h_km": c["h_km"], "beta0": c["beta0"], "beta1": c["beta1"], "n_keys": len(keys),
                    "mismatched_keys": bad})

    # ---- figure ----
    by = {_f7_key(c): c for c in cells}
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])
    cols = {"interp": "C0", "hold": "C3", "fit": "C2"}
    b0, b1 = F7_BETAS[0]
    for r, (metric, name) in enumerate((("precision", "precision std (%)"), ("accuracy", "accuracy |mean| (%)"))):
        for j, h in enumerate(F7_ALTS):
            ax = fig.add_subplot(gs[r, j])
            for mode in ("interp", "hold", "fit"):
                y = [100 * by[(h, b0, b1, n, mode, gauge.TAU_B)][metric] for n in F7_N]
                ax.plot(F7_N, y, "-o", color=cols[mode], label=mode + (" (non-causal)" if mode == "interp" else " (causal)"))
            if h == F7_PROBE["h_km"] and metric in ("precision", "accuracy"):
                ax.plot([F7_PROBE["n_zero"]], [100 * probe[0][metric]], "x", color=cols["fit"], ms=9, mew=2,
                        label=f"fit, τ_model = {F7_PROBE['tau_model']:.0f} s (probe)")
            ax.axhline(100 * baseline[f"{h:.0f}"][metric], color="k", ls="--", lw=1.0,
                       label="no-offset baseline (F6, A at (0, 0))")
            ax.set_xscale("log")
            ax.set_xticks(F7_N, [str(n) for n in F7_N])
            ax.set_xlabel("N (closed seconds per zero-read)")
            ax.set_ylabel(name)
            ax.set_title(f"F7 ({'ab'[r]}) · {metric}, {h:.0f} km, (β0, β1) = ({b0:g}, {b1:g})", fontsize=10)
            ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[2, :])
    h, beta, n = F7_SERIES
    T_orb = scenarios.build_truth(h).T_orb
    s0 = series["hold"]
    t1 = s0["t1"]
    w = (t1 >= 7 * DAY - T_orb) & (t1 < 7 * DAY + T_orb)
    th = (t1[w] - 7 * DAY) / 60.0
    for mode in ("hold", "fit"):
        s_ = series[mode]
        e = (s_["bhat1"] - s_["b1"]) / s_["rho_ref"]
        e = np.where(s_["skip1"], np.nan, e)
        ax.plot(th, 100 * e[w], lw=0.6, color=cols[mode], label=f"{mode}: (b̂ − b)/ρ_ref")
    nu = s0["nu1"][w]
    cross = np.nonzero(np.diff(nu) != 0)[0]
    for i, k in enumerate(cross):
        ax.axvline(th[k + 1], color="0.4", lw=0.8, ls=":", label="terminator crossing" if i == 0 else None)
    ax.fill_between(th, 0, 1, where=nu > 0.5, transform=ax.get_xaxis_transform(), color="gold", alpha=0.15, lw=0,
                    label="sunlit (ν = 1)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("time from day 7 (min)")
    ax.set_ylabel("(b̂ − b)/ρ_ref (%)")
    ax.set_title(f"F7 (c) · {h:.0f} km, ({beta[0]:g}, {beta[1]:g}), N = {n}: zero-read estimate error, causal modes, "
                 "two orbits around day 7 (shutter seconds blank)", fontsize=10)
    for ln in ax.legend(fontsize=8, loc="upper right", ncol=2).get_lines():
        ln.set_linewidth(2.0)
    fig.tight_layout()
    fig.savefig(RESULTS / "F7.png", dpi=130)
    plt.close(fig)

    # panel (c) numbers for the reading
    pc = {}
    for mode in ("hold", "fit"):
        s_ = series[mode]
        e = ((s_["bhat1"] - s_["b1"]) / s_["rho_ref"])[w & ~s_["skip1"]]
        pc[mode] = {"rms": float(np.sqrt(np.mean(e * e))), "max_abs": float(np.max(np.abs(e)))}
    out = {
        "config": {"t_zero_s": gauge.T_ZERO, "n_zero": list(F7_N), "modes": list(F7_MODES), "betas": [list(b) for b in F7_BETAS],
                   "alts_km": list(F7_ALTS), "tau_b_s": gauge.TAU_B, "tau_model_main_s": gauge.TAU_B,
                   "probe": {**F7_PROBE, "beta": list(F7_PROBE["beta"])}, "min_valid": gauge.MIN_VALID,
                   "fit_min_spread": gauge.FIT_MIN_SPREAD, "gauge_seed": scenarios.GAUGE_SEED,
                   "zr_seed": scenarios.ZR_SEED, "noise_floor": gauge.NOISE_FLOOR, "noise_rel": gauge.NOISE_REL,
                   "ref_commit": F7_REF_COMMIT},
        "baseline_from_F6": baseline,
        "cells": [_scalars(c) for c in cells],
        "probe": [_scalars(c) for c in probe],
        "panel_c": {"h_km": h, "beta": list(beta), "n_zero": n, "window_s": [7 * DAY - T_orb, 7 * DAY + T_orb],
                    "n_terminator_crossings": int(len(cross)), **pc},
        "regression": {"n1_interp_vs_F6_B": reg},
    }
    (RESULTS / "F7.json").write_text(json.dumps(out, indent=1, default=float))
    # R3b-6 part 2 needs the files on disk: compare after writing F7 (F7 is not tracked at the reference commit)
    out["regression"]["results_identical_to_ref"] = _results_identical_to(F7_REF_COMMIT)
    (RESULTS / "F7.json").write_text(json.dumps(out, indent=1, default=float))
    return cells


def decay_signal():
    rows = [metrics.decay_signal(scenarios.build_truth(h)) for h in ALTS]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "decay_signal.json").write_text(json.dumps(rows, indent=1))
    return rows


def _load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.2f} %"


def _days(c):
    d = f"{c['conv_days']:.2f} d" if c["converged"] else "not converged in 14 d"
    if c.get("ksig_final_rel", 0.0) > metrics.CONV_TOL:
        d += f" † (marginal: final posterior σ {100*c['ksig_final_rel']:.1f} % ≥ the 5 % tolerance)"
    return d


def _row(name, c):
    acc_v = metrics.verdict(c["accuracy"], metrics.ACC_LIMIT)
    prec_v = metrics.verdict(c["precision"], metrics.PREC_LIMIT)
    res_v = metrics.verdict(c["resolution_s"], metrics.RES_LIMIT_S)
    win = "after conv." if c["converged"] else "days 7–14 (no conv.)"
    return (f"| {name} | {c['h_km']:.0f} | {c['grade']} | {c['eps_B']:.2f} | {c['n_orb']} | {_days(c)} | "
            f"{_pct(c['accuracy'])} ({win}) | {_pct(c['precision'])} | {c['resolution_s']:.0f} s | "
            f"{acc_v} | {prec_v} | {res_v} | {_pct(c.get('acc_uncal'))} |")


def write_summary():
    f1, f2, f3, f4 = (_load(n) for n in ("F1", "F2", "F3", "F4"))
    L = ["# Results summary — orbit-anchored gauge calibration (Tier 1, synthetic NRLMSIS 2.0 truth)", "",
         "Generated by `python -m clarity_anchor.run --fig all`. Benchmark: accuracy ≤ 20 %, precision ≤ 10 %, "
         "resolution ≤ 30 s (spec §5). Accuracy = |mean| and precision = std of the 30 s ρ_cal/ρ_true − 1 "
         "after convergence of the calibration factor (|ŷ·K_eff,true − 1| < 5 % for the rest of the run; "
         "the filter state is ŷ = 1/K̂_eff, measured by z_inv = I_pod/I_gauge, ρ_cal = ρ_meas·ŷ). If K̂ never converges, "
         "the window is days 7–14 and convergence is reported as not converged. Resolution is the measured "
         f"spacing of the delivered ρ_cal series. One gauge realisation (seed {scenarios.GAUGE_SEED}, chosen as the seed in 1–10 with K(0) farthest from 1; K(0) = "
         f"{(f1 or f2 or f3 or f4 or [{'K0': float('nan')}])[0]['K0']:.3f}) and one POD noise realisation per grade.",
         "", "## Scenario table (spec §5)", "",
         "| Scenario | h (km) | POD | ε_B | T_arc (orb) | convergence | accuracy | precision | resolution | acc ≤20 % | prec ≤10 % | res ≤30 s | uncalibrated accuracy (ŷ ≡ 1) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, cells in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4)):
        if cells is None:
            L.append(f"| {name} | — | — | — | — | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |")
            continue
        for c in cells:
            L.append(_row(name, c))
    f5 = _load("F5")
    if f5:
        L += ["", f"## F5 (stretch): fleet mean, 400 km, good POD, 1-orbit arcs, ε_B ~ N(0, 0.10), shared K_lot, {f5[0]['draws']} fleets", "",
              "Host 1 calibrates with the mean K̂ of N hosts; accuracy = RMS over fleets of the day 1–14 mean relative error.", "",
              "| N | RMS accuracy | 0.10/√N | acc ≤20 % (RMS) |", "|---|---|---|---|"]
        for r in f5:
            L.append(f"| {r['N']} | {_pct(r['rms_accuracy'])} | {_pct(r['expected'])} | {metrics.verdict(r['rms_accuracy'], metrics.ACC_LIMIT)} |")
        nn = np.array([r["N"] for r in f5], float)
        slope = float(np.polyfit(np.log(nn), np.log([r["rms_accuracy"] for r in f5]), 1)[0])
        L += ["", f"Expectation (spec §5): fleet-mean accuracy ∝ 1/√N. Fitted log-log slope {slope:.2f} (−0.5 expected; "
              f"RMS over {f5[0]['draws']} fleets has ~{100/np.sqrt(2*f5[0]['draws']):.0f} % sampling error). "
              f"Verdict: {'PASS' if abs(slope + 0.5) < 0.15 else 'FAIL'}. All hosts share one truth orbit and atmosphere "
              "(see DECISIONS.md)."]
    else:
        L.append("| F5 (fleet, stretch) | — | — | — | — | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |")
    ens = _load("F4_ensemble")
    L += ["", f"## F4 robustness: {len(ENSEMBLE_SEEDS)} gauge seeds per cell (K(0) ~ U(0.7, 1.3) each)", ""]
    if ens:
        L += ["| h (km) | POD | T_arc | converged | median conv. | max conv. | worst accuracy | worst precision | worst-case acc ≤20 % | worst-case prec ≤10 % | worst uncalibrated accuracy |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in ens:
            L.append(f"| {e['h_km']:.0f} | {e['grade']} | {e['n_orb']} | {e['n_conv']}/{e['n_seeds']} | "
                     f"{e['median_conv_days']:.2f} d | {e['max_conv_days']:.2f} d | {_pct(e['worst_accuracy'])} | "
                     f"{_pct(e['worst_precision'])} | {metrics.verdict(e['worst_accuracy'], metrics.ACC_LIMIT)} | "
                     f"{metrics.verdict(e['worst_precision'], metrics.PREC_LIMIT)} | {_pct(e['worst_acc_uncal'])} |".replace("nan d", "—"))
        L += ["", "Accuracy of a run that never converged is taken over days 7–14. A run marked † converged while its "
              "own posterior σ still exceeds the 5 % tolerance, so its convergence time is not resolved by the filter."]
    else:
        L.append("UNMEASURED")
    sig = _load("decay_signal")
    L += ["", "## Per-orbit decay signal vs POD noise", "",
          "Median true drag work per one-orbit arc (14-day truth, F10.7 = F10.7a = 150, ap = 15, with the m(t) field and storm), "
          "the equivalent decay Δa = 2a²|W|/μ, and the endpoint energy noise σ(ΔE) = √2·std(E_POD − E_true) of one arc "
          "(two independent endpoints) per POD grade. SNR = drag work / σ(ΔE) for a 1-orbit arc; it grows ∝ T_arc.", ""]
    if sig:
        L += ["| h (km) | drag work / orbit (J/kg) | Δa / orbit (m) | σ(ΔE) good (J/kg) | σ(ΔE) COTS (J/kg) | SNR good | SNR COTS |",
              "|---|---|---|---|---|---|---|"]
        for r in sig:
            w = r["drag_work_J_per_kg"]
            L.append(f"| {r['h_km']:.0f} | {w:.2f} | {r['da_m']:.2f} | {r['sigma_dE_good']:.3f} | {r['sigma_dE_cots']:.3f} | "
                     f"{w / r['sigma_dE_good']:.1f} | {w / r['sigma_dE_cots']:.1f} |")
    else:
        L.append("UNMEASURED")
    L += ["", "## Pre-registered expectations (spec §6)", "", "| # | Expectation | Measured | Verdict |", "|---|---|---|---|"]

    def cell(cells, **kw):
        if not cells:
            return None
        for c in cells:
            if all(c[k] == v for k, v in kw.items()):
                return c
        return None

    c400 = cell(f1, h_km=400.0) or cell(f4, h_km=400.0, grade="good", n_orb=1)
    if c400:
        ok = c400["converged"] and c400["conv_days"] < 1.0
        L.append(f"| 1a | 400 km, good POD: K̂ within 5 % in < 1 day | {_days(c400)} | {'PASS' if ok else 'FAIL'} |")
        L.append(f"| 1b | 400 km, good POD: precision < 5 % | {_pct(c400['precision'])} | "
                 f"{'PASS' if c400['precision'] < 0.05 else 'FAIL'} |")
    else:
        L += ["| 1a | 400 km K̂ < 5 % within 1 day | — | UNMEASURED |", "| 1b | 400 km precision < 5 % | — | UNMEASURED |"]
    c600_10 = cell(f4, h_km=600.0, grade="good", n_orb=10)
    c600_1 = cell(f4, h_km=600.0, grade="good", n_orb=1) or cell(f1, h_km=600.0)
    if c600_1:
        L.append(f"| 2a | 600 km needs multi-orbit arcs (1-orbit arcs do not converge) | 1-orbit: {_days(c600_1)}, "
                 f"per-arc z_inv noise {_pct(c600_1['R_std'] / abs(c600_1['R_mc_mean']))} | {'PASS' if not c600_1['converged'] else 'FAIL'} |")
    else:
        L.append("| 2a | 600 km needs multi-orbit arcs | — | UNMEASURED |")
    if c600_10:
        inrange = c600_10["converged"] and 5.0 <= c600_10["conv_days"] <= 10.0
        note = "" if inrange else (" (converged faster than expected)" if c600_10["converged"] and c600_10["conv_days"] < 5 else "")
        L.append(f"| 2b | 600 km, good POD, T_arc = 10 orbits: convergence in 5–10 days | {_days(c600_10)}, per-arc z_inv noise "
                 f"{_pct(c600_10['R_std'] / abs(c600_10['R_mc_mean']))} | {'PASS' if inrange else 'FAIL' + note} |")
    else:
        L.append("| 2b | 600 km, 10-orbit arcs: 5–10 days | — | UNMEASURED |")
    if f3:
        eps = np.array([c["eps_B"] for c in f3])
        acc = np.array([c["accuracy"] for c in f3])
        slope = float(np.polyfit(eps, acc, 1)[0])
        dev = float(np.max(np.abs(acc - eps / (1 + eps))))
        c10 = cell(f3, eps_B=0.10)
        meas = ", ".join(f"ε={e:.2f}: {_pct(a)}" for e, a in zip(eps, acc))
        L.append(f"| 3a | accuracy tracks ε_B 1:1 (spec original; measured relation is ε_B/(1+ε_B), amended §3) | {meas}; fitted slope {slope:.3f}; max |acc − ε/(1+ε)| = {_pct(dev)} | "
                 f"{'PASS' if abs(slope - 1) < 0.15 else 'FAIL'} (tracks ε/(1+ε), see DECISIONS) |")
        L.append(f"| 3b | ε_B = 0.10 → accuracy ≈ 10 %, PASS vs 20 % | {_pct(c10['accuracy'])} | "
                 f"{metrics.verdict(c10['accuracy'], metrics.ACC_LIMIT)} |")
    else:
        L += ["| 3a | accuracy tracks ε_B 1:1 (spec original; measured relation is ε_B/(1+ε_B), amended §3) | — | UNMEASURED |", "| 3b | ε_B = 0.10 → ≈10 % | — | UNMEASURED |"]
    c = c400 or (f2[0] if f2 else None)
    if c:
        L.append(f"| 4a | composition g(t) drifts K by a few %/day around the storm | storm is a density-only multiplier in the "
                 f"§2 truth, so it cannot change composition; background arc-mean ḡ at 400 km spans "
                 f"{c['gbar_min']:.4f}–{c['gbar_max']:.4f}, max change {_pct(c['gbar_max_rate_per_day'])}/day | UNMEASURED (not testable as specified) |")
        tracked = c["converged"]
        L.append(f"| 4b | the ḡ drift is tracked, not rejected | ŷ stays within 5 % of 1/(K·ḡ) after {_days(c)} | "
                 f"{'PASS' if tracked else 'FAIL'} |")
    L += ["", "## Per-cell diagnostics", "",
          "| Scenario | h | POD | T_arc | arcs | R std(z_inv), absolute (MC, 200; z_inv ≈ 1/K) | std(z_inv·K_eff,true − 1) | mean(z_inv·K_eff,true − 1) | final calibration error | final posterior σ (rel.) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, cells in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4)):
        for c in cells or []:
            L.append(f"| {name} | {c['h_km']:.0f} | {c['grade']} | {c['n_orb']} | {c['n_arcs']} | {_pct(c['R_std'])} | "
                     f"{_pct(c['z_rel_noise_vs_truth'])} | {_pct(c.get('z_mean_bias'))} | {_pct(c['khat_final_err'])} | {_pct(c.get('ksig_final_rel'))} |")
    L += _f6_section(f1)
    L += _f7_section()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n")


def _pct4(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.4f} %"


def _num(x, fmt=".2f"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def _f6_section(f1):
    """Round 3 section (docs/round3-offset-spec.md §5-§6). Every number is read from results/F6.json (and F1.json
    for the round-2 reference cells)."""
    f6 = _load("F6")
    L = ["", "## F6: additive sun-correlated offset (round 3, docs/round3-offset-spec.md)", ""]
    if not f6:
        return L + ["UNMEASURED (results/F6.json missing; run `python -m clarity_anchor.run --fig F6`)."]
    cells = f6["cells"]
    diag = {_f6_key(c): c for c in f6["diagnostics"]["B_noiseless_zero_read"]}
    by = {_f6_key(c): c for c in cells}
    variants = sorted({c["variant"] for c in cells})
    L += ["Gauge truth ρ_meas = K g ρ_true + b(t) + n with b(t) = ρ_ref·(β0 + β1·s(t)), s the 300 s first-order lag of the "
          "cylindrical-shadow sunlit indicator, ρ_ref the median ρ_true over day 0–1 at that altitude. Good POD, 1-orbit arcs, "
          f"ε_B = 0, gauge seed {scenarios.GAUGE_SEED}. Variant A: round-2 estimator unchanged. Variant B: shutter zero-read every "
          "300 s (1 s sample, own noise draw), b̂ linear between zero-reads, ρ_corr = ρ_meas − b̂ feeds the anchor and the 30 s "
          "product (shutter seconds skipped). Variant C (stretch): no zero-reads, 2-state filter [y, c] with c a slowly "
          "varying additive term in I_gauge. Accuracy/precision/convergence as in the main table; sunlit/eclipse split on "
          "s > 0.5 / s < 0.5; offset_corr = Pearson(e_ρ, b/ρ_true) over the same window. The β sweep is relative to ρ_ref, "
          "so one β is a different absolute pressure at each altitude.", "",
          "| h (km) | β0 | β1 | variant | convergence | accuracy | precision | acc_sunlit | acc_eclipse | offset_corr | acc ≤20 % | prec ≤10 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        win = "after conv." if c["converged"] else "days 7–14, no conv."
        L.append(f"| {c['h_km']:.0f} | {c['beta0']:.1f} | {c['beta1']:.1f} | {c['variant']} | {_days(c)} | "
                 f"{_pct(c['accuracy'])} ({win}) | {_pct(c['precision'])} | {_pct(c['acc_sunlit'])} | {_pct(c['acc_eclipse'])} | "
                 f"{_num(c['offset_corr'])} | {metrics.verdict(c['accuracy'], metrics.ACC_LIMIT)} | "
                 f"{metrics.verdict(c['precision'], metrics.PREC_LIMIT)} |")
    if "C" not in variants:
        L.append("| 400, 600 | all | all | C | UNMEASURED (not run) | UNMEASURED (not run) | UNMEASURED (not run) | — | — | — | UNMEASURED | UNMEASURED |")
    L += ["", "offset_corr is n/a at (0, 0) because b ≡ 0.", "",
          "### F6 diagnostics", "",
          "prec_sunlit/prec_eclipse: std of e_ρ in the sunlit/eclipse bins. Low/high ρ quartile: abs(mean) and RMS of e_ρ over the "
          "bins with ρ_true in the lowest/highest quartile of the window. Baseline corr: Pearson correlation the NO-offset "
          "product (variant A at (0, 0), same altitude and window) already has with this cell's b/ρ shape (orbital structure "
          "of the round-2 error, e.g. the composition term, that is not offset leakage). B, noise-free zero-reads: the same B "
          "cell with n_z = 0, which separates zero-read noise from interpolation error.", "",
          "| h (km) | β0 | β1 | variant | prec_sunlit | prec_eclipse | low-ρ quartile: abs(mean) / RMS | high-ρ quartile: abs(mean) / RMS | mean b/ρ | baseline corr | b̂ error std (rel., B) | B noise-free zero-reads: acc / prec |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def rms(a, p):
        return math.hypot(a, p)

    for c in cells:
        d = diag.get(_f6_key({**c, "variant": "B"})) if c["variant"] == "B" else None
        L.append(f"| {c['h_km']:.0f} | {c['beta0']:.1f} | {c['beta1']:.1f} | {c['variant']} | {_pct(c['prec_sunlit'])} | "
                 f"{_pct(c['prec_eclipse'])} | {_pct(c['acc_lowrho_q'])} / {_pct(rms(c['acc_lowrho_q'], c['prec_lowrho_q']))} | "
                 f"{_pct(c['acc_highrho_q'])} / {_pct(rms(c['acc_highrho_q'], c['prec_highrho_q']))} | {_pct(c['b_over_rho_mean'])} | "
                 f"{_num(c.get('offset_corr_baseline'))} | {_pct(c.get('bhat_err_rel_std')) if c['variant'] == 'B' else '—'} | "
                 f"{(_pct(d['accuracy']) + ' / ' + _pct(d['precision'])) if d else '—'} |")
    cc = [c for c in cells if c["variant"] == "C"]
    L += ["", "### Variant C (stretch): can the anchor alone separate an additive term c from the gain y?", ""]
    if cc:
        L += ["State [y, d], d = y·c/ρ_s (ρ_s = median ρ_meas over the first arc), measurement z_inv = y − d·q_k with "
              "q_k = ρ_s·∫w dt / I_gauge,k. c is observable only through the arc-to-arc spread of q_k (the kernel-weighted "
              "mean measured density), i.e. the daily m(t) term and the storm. True c = arc mean of ∫b w dt / ∫w dt.", "",
              "| h (km) | β0 | β1 | final corr(y, c) | median corr, 2nd half | ĉ/ρ_ref final | true c/ρ_ref (arc mean) | σ(ĉ)/ρ_ref final | arc-to-arc spread of q_k |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in cc:
            L.append(f"| {c['h_km']:.0f} | {c['beta0']:.1f} | {c['beta1']:.1f} | {_num(c['c_post_corr_final'], '.3f')} | "
                     f"{_num(c['c_post_corr_median_2nd_half'], '.3f')} | {_num(c['c_hat_final_over_rho_ref'], '.3f')} | "
                     f"{_num(c['c_true_over_rho_ref_mean'], '.3f')} | {_num(c['c_sigma_final_over_rho_ref'], '.3f')} | "
                     f"{_pct(c['q_reg_rel_std'])} |")
    else:
        L.append("UNMEASURED (not run).")

    # ---- pre-registered expectations R3-1..R3-5 ----
    L += ["", "### Pre-registered expectations (round-3 spec §6)", "", "| # | Expectation | Measured | Verdict |", "|---|---|---|---|"]
    ref = {c["h_km"]: c for c in (f1 or []) if c["grade"] == "good" and c["n_orb"] == 1 and c["eps_B"] == 0.0}
    alts = sorted({c["h_km"] for c in cells})
    betas = [(c["beta0"], c["beta1"]) for c in cells if c["h_km"] == alts[0] and c["variant"] == "A"]
    nz = [b for b in betas if b != (0.0, 0.0)]
    # R3-1
    for v, tol, label in (("A", 0.5e-6, "identical to round 2 to 4 decimals (in %)"),
                          ("B", 1e-4, "within 0.01 pp of round 2 (only the skipped 1 s samples differ)")):
        parts, ok, have = [], True, True
        for h in alts:
            c, r = by.get((h, 0.0, 0.0, v)), ref.get(h)
            if c is None or r is None:
                have = False
                continue
            da, dp = abs(c["accuracy"] - r["accuracy"]), abs(c["precision"] - r["precision"])
            dconv = abs(c["conv_days"] - r["conv_days"]) if c["converged"] and r["converged"] else float("nan")
            ok &= da < tol and dp < tol and (dconv < 1e-4 if v == "A" else True)
            s_ = (f"{h:.0f} km: acc {_pct4(c['accuracy'])} vs {_pct4(r['accuracy'])}, prec {_pct4(c['precision'])} vs "
                  f"{_pct4(r['precision'])} (abs Δ {100*da:.4f} / {100*dp:.4f} pp), conv {c['conv_days']:.4f} vs {r['conv_days']:.4f} d")
            if v == "B" and (h, 0.0, 0.0, "B") in diag:
                d = diag[(h, 0.0, 0.0, "B")]
                s_ += (f"; with noise-free zero-reads: abs Δ {100*abs(d['accuracy'] - r['accuracy']):.4f} / "
                       f"{100*abs(d['precision'] - r['precision']):.4f} pp")
            parts.append(s_)
        L.append(f"| R3-1{v} | (0, 0), variant {v}: {label} | {'; '.join(parts) or '—'} | "
                 f"{('PASS' if ok else 'FAIL') if have else 'UNMEASURED'} |")
    # R3-2
    c = by.get((400.0, 0.1, 0.1, "A"))
    if c:
        ok = 0.02 <= c["accuracy"] <= 0.08 and c["precision"] <= metrics.PREC_LIMIT and c["offset_corr"] > 0.5
        L.append(f"| R3-2 | A at (0.1, 0.1), 400 km: accuracy 2–8 %, precision PASS, offset_corr > 0.5 | accuracy "
                 f"{_pct(c['accuracy'])} ({'after conv.' if c['converged'] else 'days 7–14, no conv.'}), precision "
                 f"{_pct(c['precision'])}, offset_corr {_num(c['offset_corr'])} | {'PASS' if ok else 'FAIL'} |")
    else:
        L.append("| R3-2 | A at (0.1, 0.1), 400 km | — | UNMEASURED |")
    # R3-3
    c = by.get((400.0, 0.3, 0.3, "A"))
    if c:
        fails = c["precision"] > metrics.PREC_LIMIT or c["accuracy"] > 0.10
        lo, hi = rms(c["acc_lowrho_q"], c["prec_lowrho_q"]), rms(c["acc_highrho_q"], c["prec_highrho_q"])
        ok = fails and lo > hi
        L.append(f"| R3-3 | A at (0.3, 0.3), 400 km: precision > 10 % or accuracy > 10 %, and the error is largest where ρ_true "
                 f"is smallest | precision {_pct(c['precision'])}, accuracy {_pct(c['accuracy'])} (benchmark breached: "
                 f"{'yes' if fails else 'no'}); RMS error lowest-ρ quartile {_pct(lo)} vs highest-ρ quartile {_pct(hi)} | "
                 f"{'PASS' if ok else 'FAIL'} |")
    else:
        L.append("| R3-3 | A at (0.3, 0.3), 400 km | — | UNMEASURED |")
    # R3-4
    bc = [c for c in cells if c["variant"] == "B"]
    if bc and all(h in ref for h in alts):
        parts, ok = [], True
        for h in alts:
            base = by[(h, 0.0, 0.0, "A")]
            da = max(abs(c["accuracy"] - base["accuracy"]) for c in bc if c["h_km"] == h)
            dp = max(abs(c["precision"] - base["precision"]) for c in bc if c["h_km"] == h)
            ok &= da <= 0.01 and dp <= 0.01
            worst = max((c for c in bc if c["h_km"] == h), key=lambda c: abs(c["precision"] - base["precision"]))
            parts.append(f"{h:.0f} km: max abs Δacc {100*da:.2f} pp, max abs Δprec {100*dp:.2f} pp (worst ({worst['beta0']:.1f}, "
                         f"{worst['beta1']:.1f}): prec {_pct(worst['precision'])} vs {_pct(base['precision'])})")
        L.append(f"| R3-4 | B at every (β0, β1), both altitudes: accuracy and precision within 1 pp of the (0, 0) no-offset "
                 f"cell (variant A at (0, 0), = round 2) | {'; '.join(parts)} | {'PASS' if ok else 'FAIL'} |")
    else:
        L.append("| R3-4 | B within 1 pp of (0, 0) | — | UNMEASURED |")
    # R3-5
    if len(alts) == 2 and nz:
        lo_h, hi_h = alts
        parts, ok = [], True
        for b0, b1 in nz:
            a4, a6 = by[(lo_h, b0, b1, "A")], by[(hi_h, b0, b1, "A")]
            ra, rp = a6["accuracy"] / a4["accuracy"], a6["precision"] / a4["precision"]
            ok &= ra > 1 and rp > 1
            parts.append(f"({b0:.1f}, {b1:.1f}): accuracy {_pct(a6['accuracy'])}/{_pct(a4['accuracy'])} = ×{ra:.2f}, "
                         f"precision {_pct(a6['precision'])}/{_pct(a4['precision'])} = ×{rp:.2f}")
        ecl = ", ".join(f"{h:.0f} km {_pct(1 - by[(h, 0.0, 0.0, 'A')]['sunlit_frac'])}" for h in alts)
        L.append(f"| R3-5 | 600 km variant A is worse than 400 km at the same β (worse = both accuracy and precision larger) | "
                 f"{'; '.join(parts)}. Eclipse fraction (ν = 0): {ecl} | {'PASS' if ok else 'FAIL'} |")
    else:
        L.append("| R3-5 | 600 km A worse than 400 km | — | UNMEASURED |")
    # C purpose statement (spec §3, not numbered in §6)
    if cc:
        ok_sep = all(abs(c["c_post_corr_final"]) >= 0.99 or c["c_sigma_final_over_rho_ref"] >= c["c_true_over_rho_ref_mean"]
                     for c in cc if (c["beta0"], c["beta1"]) != (0.0, 0.0))
        parts = [f"{c['h_km']:.0f} km ({c['beta0']:.1f}, {c['beta1']:.1f}): corr {c['c_post_corr_final']:.3f}, ĉ/ρ_ref "
                 f"{c['c_hat_final_over_rho_ref']:.3f} ± {c['c_sigma_final_over_rho_ref']:.3f} vs true {c['c_true_over_rho_ref_mean']:.3f}, "
                 f"accuracy {_pct(c['accuracy'])}, precision {_pct(c['precision'])}"
                 for c in cc if (c["beta0"], c["beta1"]) != (0.0, 0.0)]
        L.append(f"| C | (spec §3 purpose, not pre-registered in §6) the anchor alone cannot separate c from y (cannot = "
                 f"abs(final corr) ≥ 0.99 or σ(ĉ) ≥ true c, criterion set in DECISIONS.md) | {'; '.join(parts)} | "
                 f"{'PASS' if ok_sep else 'FAIL'} |")
    else:
        L.append("| C | anchor alone cannot separate c from y | not run | UNMEASURED (not run) |")
    L += ["", "### Reading (every number generated from results/F6.json)", ""]
    from .gauge import NOISE_FLOOR

    for v in ("A", "B", "C"):
        vc = [c for c in cells if c["variant"] == v and (c["beta0"], c["beta1"]) != (0.0, 0.0)]
        if not vc:
            continue
        acc_max = max(vc, key=lambda c: c["accuracy"])
        prec_max = max(vc, key=lambda c: c["precision"])
        head = (f"worst accuracy {_pct(acc_max['accuracy'])} ({acc_max['h_km']:.0f} km, ({acc_max['beta0']:.1f}, {acc_max['beta1']:.1f})), "
                f"worst precision {_pct(prec_max['precision'])} ({prec_max['h_km']:.0f} km, ({prec_max['beta0']:.1f}, {prec_max['beta1']:.1f}))")
        if v == "A":
            nconv = sum(c["converged"] for c in vc)
            kerr = ", ".join(f"{c['h_km']:.0f} km ({c['beta0']:.1f}, {c['beta1']:.1f}) {_pct(c['khat_final_err'])}" for c in vc)
            L.append(f"- **A (no mitigation)**, β ≠ 0: {head}. The anchor still matches the drag-weighted arc integral, so the "
                     f"offset is absorbed into ŷ and shows up as within-orbit structure (precision, offset_corr) rather than as a "
                     f"mean bias. ŷ no longer equals 1/K_eff: {nconv}/{len(vc)} offset cells meet the 5 % convergence rule; final "
                     f"calibration-factor error ŷ·K_eff,true − 1: {kerr}.")
        elif v == "B":
            fl = ", ".join(f"{h:.0f} km {_pct(NOISE_FLOOR / by[(h, 0.0, 0.0, 'A')]['rho_ref'])}" for h in alts)
            dz = ", ".join(f"{h:.0f} km {100*max(abs(diag[_f6_key(c)]['precision'] - by[(h, 0.0, 0.0, 'A')]['precision']) for c in bc if c['h_km'] == h):.2f} pp"
                           for h in alts)
            excess = max(c["offset_corr"] - c["offset_corr_baseline"] for c in vc)
            L.append(f"- **B (shutter zero-read)**, β ≠ 0: {head}. The single 1 s zero-read carries the gauge noise floor "
                     f"({NOISE_FLOOR:g} kg/m³ = {fl} of ρ_ref), and b̂ holds that error for 300 s, so it does not average down "
                     f"in the 30 s bins. With noise-free zero-reads the largest precision change vs the no-offset cell is {dz} "
                     f"(interpolation error of the lagged sunlit term only). B's offset_corr exceeds the baseline corr column "
                     f"(the correlation the no-offset product already has with the same b/ρ shape) by at most {excess:+.2f}.")
        else:
            cr = [c["c_post_corr_final"] for c in vc]
            L.append(f"- **C (2-state, no zero-reads)**, β ≠ 0: {head}. Final posterior corr(y, c) {min(cr):.3f}–{max(cr):.3f}. "
                     f"ĉ tracks the arc-mean offset, but subtracting a per-arc constant leaves (b − c̄)/ρ_true, which is largest "
                     f"where ρ_true is smallest, so C's precision is worse than A's in every offset cell where both were run.")
    worse = all(by[(c["h_km"], c["beta0"], c["beta1"], "C")]["precision"] > c["precision"]
                for c in cells if c["variant"] == "A" and (c["beta0"], c["beta1"]) != (0.0, 0.0)
                and (c["h_km"], c["beta0"], c["beta1"], "C") in by)
    if cc and not worse:
        L[-1] = L[-1].replace(", so C's precision is worse than A's in every offset cell where both were run.",
                              "; C's precision is NOT worse than A's in every offset cell (see table).")
    return L


def _pp(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:+.{nd}f} pp"


def _within(x, ref, tol):
    """|x - ref| <= tol, with the direction of a miss."""
    d = x - ref
    ok = abs(d) <= tol
    return ok, ("" if ok else (" (above: worse)" if d > 0 else " (below: better)"))


def _f7_section():
    """Round 3b section (docs/round3b-zero-read-spec.md §5-§6). Every number is read from results/F7.json."""
    f7 = _load("F7")
    L = ["", "## F7: averaged and causal shutter zero-reads (round 3b, docs/round3b-zero-read-spec.md)", ""]
    if not f7:
        return L + ["UNMEASURED (results/F7.json missing; run `python -m clarity_anchor.run --fig F7`)."]
    cfg, base = f7["config"], f7["baseline_from_F6"]
    cells, probe = f7["cells"], f7["probe"]
    by = {_f7_key(c): c for c in cells}
    tau = cfg["tau_model_main_s"]
    pr = probe[0] if probe else None
    L += [f"Truth as in F6. Variant B only: the shutter closes for N seconds every {cfg['t_zero_s']:g} s "
          f"(N ∈ {{{', '.join(str(n) for n in cfg['n_zero'])}}}); one zero-read per period = mean of b + n_z over the N closed "
          "seconds, time-stamped at the window centre. b̂ estimators: interp = linear between the previous and the NEXT read "
          "(round 3, non-causal, reference); hold = last completed read (causal, b̂ = 0 before the first); fit = causal OLS "
          "of all completed reads on the window-mean lagged sunlit indicator s_model, b̂ = ĉ0 + ĉ1·s_model(t), with s_model "
          f"computed from the known shadow geometry and an ASSUMED lag τ_model ({tau:g} s in the main sweep, matched to "
          f"truth), holding until two reads with regressor std ≥ {cfg['fit_min_spread']:g} exist. 30 s product bins with fewer "
          f"than {cfg['min_valid']} valid samples are gaps (NaN), excluded from every metric. Good POD, 1-orbit arcs, ε_B = 0, "
          f"gauge seed {cfg['gauge_seed']}, zero-read seed {cfg['zr_seed']}. b̂ RMS = RMS of (b̂ − b)/ρ_ref over the 1 Hz "
          "non-shutter seconds of days 1–14. No-offset baseline per altitude = F6 variant A at (0, 0), quoted from F6.json, "
          "not re-run: " + ", ".join(f"{h} km accuracy {_pct(v['accuracy'])}, precision {_pct(v['precision'])}"
                                    for h, v in base.items()) + ".", "",
          "| h (km) | β0 | β1 | N | mode | τ_model (s) | convergence | accuracy | precision | prec_sunlit | prec_eclipse | gap fraction | duty lost | b̂ RMS (rel. ρ_ref) | final cal. error | offset_corr | prec − baseline | acc ≤20 % | prec ≤10 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(cells, key=lambda c: (c["h_km"], -c["beta0"], c["zr_mode"] != "interp", c["zr_mode"], c["n_zero"])) + probe
    for c in rows:
        win = "after conv." if c["converged"] else "days 7–14, no conv."
        b = base[f"{c['h_km']:.0f}"]
        L.append(f"| {c['h_km']:.0f} | {c['beta0']:.1f} | {c['beta1']:.1f} | {c['n_zero']} | {c['zr_mode']} | {c['tau_model']:g} | "
                 f"{_days(c)} | {_pct(c['accuracy'])} ({win}) | {_pct(c['precision'])} | {_pct(c['prec_sunlit'])} | "
                 f"{_pct(c['prec_eclipse'])} | {_pct(c['gap_frac'])} | {_pct(c['skipped_frac'])} | {_pct(c['bhat_rms_rel_d1_14'])} | "
                 f"{_pct(c['khat_final_err'])} | {_num(c['offset_corr'])} | {_pp(c['precision'] - b['precision'])} | "
                 f"{metrics.verdict(c['accuracy'], metrics.ACC_LIMIT)} | {metrics.verdict(c['precision'], metrics.PREC_LIMIT)} |")
    ns = sorted({c["n_zero"] for c in cells})
    gaps = ", ".join(f"N = {n}: {_pct(max(c['gap_frac'] for c in cells if c['n_zero'] == n))}" for n in ns)
    L += ["", f"Gap fraction (largest over cells) by N: {gaps}. At N = {max(ns)} the closed window covers one full 30 s bin "
          "per period, so one bin in ten is a gap by construction; gaps are reported, not filled, and excluded from "
          "accuracy and precision. The last row is the τ_model probe.", ""]

    # ---- pre-registered expectations R3b-1..R3b-6 ----
    L += ["### Pre-registered expectations (round-3b spec §6)", "",
          "Definitions (fixed before the sweep ran, see docs/DECISIONS.md): \"within X pp\" = |Δprecision| ≤ X pp; "
          "R3b-3 \"any N\" = every N in the sweep and both β (both have β1 > 0); each verdict states the direction of a miss.", "",
          "| # | Expectation | Measured | Verdict |", "|---|---|---|---|"]
    betas = [tuple(b) for b in cfg["betas"]]

    def get(h, beta, n, mode, t=tau):
        return by.get((h, beta[0], beta[1], n, mode, t))

    def tolrow(tag, text, pairs, tol):
        parts, ok_all, have = [], True, True
        for lab, x, ref in pairs:
            if x is None or ref is None:
                have = False
                continue
            ok, dirn = _within(x, ref, tol)
            ok_all &= ok
            parts.append(f"{lab}: {_pct(x)} vs {_pct(ref)} (Δ {_pp(x - ref, 3)}){dirn}")
        L.append(f"| {tag} | {text} | {'; '.join(parts) or '—'} | {('PASS' if ok_all else 'FAIL') if have else 'UNMEASURED'} |")

    b6 = base["600"]["precision"]
    for tag, n, tol in (("R3b-1", 10, 0.010), ("R3b-2", 30, 0.005)):
        pairs = [(f"({bb[0]:g}, {bb[1]:g})", (get(600.0, bb, n, "interp") or {}).get("precision"), b6) for bb in betas]
        tolrow(tag, f"interp, N = {n}: 600 km precision within {100*tol:.1f} pp of the no-offset baseline, both β", pairs, tol)
    # R3b-3
    parts, ok_all, have, nmet = [], True, True, 0
    for bb in betas:
        for n in ns:
            ch, ci = get(400.0, bb, n, "hold"), get(400.0, bb, n, "interp")
            if ch is None or ci is None:
                have = False
                continue
            d = ch["precision"] - ci["precision"]
            ok = d > 0.02
            nmet += ok
            ok_all &= ok
            parts.append(f"({bb[0]:g}, {bb[1]:g}) N = {n}: hold {_pct(ch['precision'])} vs interp {_pct(ci['precision'])} "
                         f"(Δ {_pp(d, 3)}){'' if ok else (' (hold worse by less than 2 pp)' if d > 0 else ' (hold not worse)')}")
    L.append(f"| R3b-3 | hold, β1 = 0.3, any N: 400 km precision worse than interp by more than 2 pp | {'; '.join(parts)}. "
             f"Met in {nmet} of {len(parts)} | {('PASS' if ok_all else 'FAIL') if have else 'UNMEASURED'} |")
    # R3b-4
    pairs = []
    for h in (400.0, 600.0):
        for bb in betas:
            cf, ci = get(h, bb, 10, "fit"), get(h, bb, 10, "interp")
            pairs.append((f"{h:.0f} km ({bb[0]:g}, {bb[1]:g}) fit vs interp", cf and cf["precision"], ci and ci["precision"]))
    tolrow("R3b-4", f"fit (τ_model = {tau:g} s), N = 10: precision within 0.5 pp of interp at the same N, both altitudes, both β",
           pairs, 0.005)
    # R3b-5
    pm = get(400.0, (0.3, 0.3), 10, "fit")
    tolrow("R3b-5", f"fit probe (τ_model = {cfg['probe']['tau_model']:g} s): precision within 1.5 pp of the matched fit cell "
           "(400 km, (0.3, 0.3), N = 10)", [("probe vs matched", pr and pr["precision"], pm and pm["precision"])], 0.015)
    # R3b-6
    reg = f7["regression"]["n1_interp_vs_F6_B"]
    ident = f7["regression"].get("results_identical_to_ref", {})
    ok6 = bool(reg) and all(not r["mismatched_keys"] for r in reg) and bool(ident) and all(ident.values())
    regs = "; ".join(f"{r['h_km']:.0f} km ({r['beta0']:g}, {r['beta1']:g}): {r['n_keys'] - len(r['mismatched_keys'])}/{r['n_keys']} "
                     "scalar keys equal" + (f" (differ: {', '.join(r['mismatched_keys'])})" if r["mismatched_keys"] else "")
                     for r in reg)
    nid = sum(ident.values())
    L.append(f"| R3b-6 | N = 1 interp reproduces F6 variant B exactly; F1–F6 byte-identical | {regs}. Files under results/ at "
             f"{cfg['ref_commit']}: {nid}/{len(ident)} byte-identical (summary.md: unchanged prefix) | {'PASS' if ok6 else 'FAIL'} |")

    # ---- reading ----
    L += ["", "### Reading (every number generated from results/F7.json)", ""]
    for h in (400.0, 600.0):
        bb = betas[0]
        line = ", ".join(f"{m} " + "/".join(_pct(get(h, bb, n, m)["precision"]) for n in ns) for m in ("interp", "hold", "fit")
                         if all(get(h, bb, n, m) for n in ns))
        L.append(f"- {h:.0f} km, ({bb[0]:g}, {bb[1]:g}), precision at N = {'/'.join(str(n) for n in ns)}: {line}; no-offset "
                 f"baseline {_pct(base[f'{h:.0f}']['precision'])}.")
    rms = ", ".join(f"{m} N = {n}: {_pct(get(600.0, betas[0], n, m)['bhat_rms_rel_d1_14'])}"
                    for m in ("interp", "hold", "fit") for n in ns if get(600.0, betas[0], n, m))
    L.append(f"- b̂ RMS at 600 km, ({betas[0][0]:g}, {betas[0][1]:g}): {rms}.")
    pc = f7["panel_c"]
    L.append(f"- Panel (c), {pc['h_km']:.0f} km, N = {pc['n_zero']}, two orbits around day 7 ({pc['n_terminator_crossings']} "
             f"terminator crossings): (b̂ − b)/ρ_ref RMS hold {_pct(pc['hold']['rms'])} (max abs {_pct(pc['hold']['max_abs'])}), "
             f"fit {_pct(pc['fit']['rms'])} (max abs {_pct(pc['fit']['max_abs'])}).")
    if pr and pm:
        L.append(f"- τ_model probe: b̂ RMS {_pct(pr['bhat_rms_rel_d1_14'])} vs matched {_pct(pm['bhat_rms_rel_d1_14'])}; "
                 f"precision {_pct(pr['precision'])} vs {_pct(pm['precision'])}.")
    L += ["", _f7_paragraph(f7, get, betas, ns, base, pr, pm)]
    return L


def _f7_paragraph(f7, get, betas, ns, base, pr, pm):
    """One-paragraph reading; numbers from F7.json, wording conditioned on the measured verdict margins."""
    need = [get(h, bb, n, m) for h in (400.0, 600.0) for bb in betas for n in ns for m in ("interp", "hold", "fit")]
    if not all(need) or not (pr and pm):
        return "Reading: UNMEASURED (sweep incomplete; see the table)."
    b4, b6 = base["400"]["precision"], base["600"]["precision"]
    n10, n30 = 10, max(ns)
    d10 = max(get(600.0, bb, n10, "interp")["precision"] - b6 for bb in betas)
    d30 = max(get(600.0, bb, n30, "interp")["precision"] - b6 for bb in betas)
    gap = max(c["gap_frac"] for c in f7["cells"] if c["n_zero"] == n30)
    duty = {n: get(600.0, betas[0], n, "interp")["skipped_frac"] for n in (n10, n30)}
    dh = [get(400.0, bb, n, "hold")["precision"] - get(400.0, bb, n, "interp")["precision"] for bb in betas for n in ns]
    nh = sum(d > 0.02 for d in dh)
    h6 = min(get(600.0, bb, n30, "hold")["precision"] - b6 for bb in betas)
    fb = [get(h, bb, n, "fit")["precision"] - (b4 if h == 400.0 else b6) for h in (400.0, 600.0) for bb in betas for n in ns]
    fi = [get(h, bb, n10, "fit")["precision"] - get(h, bb, n10, "interp")["precision"] for h in (400.0, 600.0) for bb in betas]
    dp = pr["precision"] - pm["precision"]
    hold_dir = ("with hold worse in every cell" if min(dh) > 0 else "and hold is not worse in every cell")
    r4 = max(abs(x) for x in fi) <= 0.005
    side = "fit on the better side" if max(fi) < 0 else ("fit on the worse side" if min(fi) > 0 else "mixed sides")
    claim = (d10 <= 0.01 and r4)
    return (
        f"Reading. Averaging the zero-read works: the non-causal interp estimator brings 600 km precision to within "
        f"{_pp(d10)} of the no-offset baseline at N = {n10} and {_pp(d30)} at N = {n30} (worst β; R3b-1 and R3b-2 "
        f"{'PASS' if d10 <= 0.01 and d30 <= 0.005 else 'see table'}), for a shutter duty cost of {_pct(duty[n10])} and "
        f"{_pct(duty[n30])}, and at N = {n30} {_pct(gap)} of the 30 s bins are gaps. Plain causal hold is not enough: at "
        f"400 km it is {_pp(min(dh))} to {_pp(max(dh))} worse than interp (the 2 pp bar of R3b-3 is cleared in {nh} of "
        f"{len(dh)} cells, so R3b-3 {'PASSES' if nh == len(dh) else 'FAILS by margin, ' + hold_dir}), "
        f"and at 600 km it stays at least {_pp(h6)} above the baseline even at N = {n30}, because b̂ steps at the "
        f"terminators (panel c). The causal fit closes the gap and more: it lands between {_pp(min(fb))} and {_pp(max(fb))} "
        f"of the baseline in every main cell, {_pp(min(fi))} to {_pp(max(fi))} relative to interp at N = {n10} "
        f"(R3b-4 {'PASS' if r4 else 'FAIL'}, {side}), because it pools every read instead of two. That result rests on "
        f"two assumptions the simulation grants: the flight software knows the shadow geometry and the offset is linear "
        f"in a first-order lag of it. Halving the assumed lag (τ_model {f7['config']['probe']['tau_model']:g} s vs "
        f"{f7['config']['tau_b_s']:g} s) costs {_pp(dp)} at 400 km, inside the 1.5 pp bound of R3b-5 by "
        f"{_pp(0.015 - dp)}; one mismatched lag at one altitude was run, so the lag should still be measured on the "
        f"bench, not assumed. " + (
            f"So the §1 claim holds in this model: N ≥ {n10} restores the 600 km product, and a causal estimator "
            f"matches the interpolated one without its latency only when it is given the shadow function and a lag "
            f"close to the truth; without that model (hold), it does not." if claim else
            "So the §1 claim does NOT hold as stated in this model; see the verdict table for which part fails.")
    )


FIGS = {"signal": decay_signal, "F1": fig_f1, "F2": fig_f2, "F3": fig_f3, "F4": fig_f4, "F5": fig_f5, "F6": fig_f6,
        "F7": fig_f7}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig", default="all", choices=[*FIGS, "all", "summary"])
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    todo = list(FIGS) if a.fig == "all" else ([] if a.fig == "summary" else [a.fig])
    for f in todo:
        t0 = time.time()
        FIGS[f]()
        print(f"{f} done in {time.time()-t0:.0f}s", flush=True)
    write_summary()
    print(f"wrote {RESULTS/'summary.md'}")


if __name__ == "__main__":
    main()
