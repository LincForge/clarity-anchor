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
    return {k: v for k, v in r.items() if k != "series"}


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
    ax.set_ylabel("K̂ / K_eff,true − 1 (%)")
    ax.set_title("F1 · K̂ relative error, good POD, ε_B = 0, T_arc = 1 orbit (storm at day 5)")
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
    for ax, gr, title, lab in ((axes[0], grid, "seed 7 (main run)", None),
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


def _load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.2f} %"


def _days(c):
    d = f"{c['conv_days']:.2f} d" if c["converged"] else "not converged in 14 d"
    if c.get("ksig_final_rel", 0.0) > metrics.CONV_TOL:
        d += f" † (filter uninformative: final posterior σ {100*c['ksig_final_rel']:.0f} %)"
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
         "after K̂ convergence (|K̂/K_eff,true − 1| < 5 % for the rest of the run). If K̂ never converges, "
         "the window is days 7–14 and convergence is reported as not converged. Resolution is the measured "
         "spacing of the delivered ρ_cal series. One gauge realisation (seed 7, K(0) = "
         f"{(f1 or f2 or f3 or f4 or [{'K0': float('nan')}])[0]['K0']:.3f}) and one POD noise realisation per grade.",
         "", "## Scenario table (spec §5)", "",
         "| Scenario | h (km) | POD | ε_B | T_arc (orb) | convergence | accuracy | precision | resolution | acc ≤20 % | prec ≤10 % | res ≤30 s | uncalibrated accuracy (K̂ ≡ 1) |",
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
        L += ["", "Accuracy of a run that never converged is taken over days 7–14. A run counted as converged while "
              "its filter is uninformative (†) only reflects the prior/ground calibration happening to sit within 5 %."]
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
                 f"per-arc z noise {_pct(c600_1['R_std'])} | {'PASS' if not c600_1['converged'] else 'FAIL'} |")
    else:
        L.append("| 2a | 600 km needs multi-orbit arcs | — | UNMEASURED |")
    if c600_10:
        inrange = c600_10["converged"] and 5.0 <= c600_10["conv_days"] <= 10.0
        note = "" if inrange else (" (converged faster than expected)" if c600_10["converged"] and c600_10["conv_days"] < 5 else "")
        L.append(f"| 2b | 600 km, good POD, T_arc = 10 orbits: convergence in 5–10 days | {_days(c600_10)}, per-arc z noise "
                 f"{_pct(c600_10['R_std'])} | {'PASS' if inrange else 'FAIL' + note} |")
    else:
        L.append("| 2b | 600 km, 10-orbit arcs: 5–10 days | — | UNMEASURED |")
    if f3:
        eps = np.array([c["eps_B"] for c in f3])
        acc = np.array([c["accuracy"] for c in f3])
        slope = float(np.polyfit(eps, acc, 1)[0])
        dev = float(np.max(np.abs(acc - eps / (1 + eps))))
        c10 = cell(f3, eps_B=0.10)
        meas = ", ".join(f"ε={e:.2f}: {_pct(a)}" for e, a in zip(eps, acc))
        L.append(f"| 3a | accuracy tracks ε_B 1:1 | {meas}; fitted slope {slope:.3f}; max |acc − ε/(1+ε)| = {_pct(dev)} | "
                 f"{'PASS' if abs(slope - 1) < 0.15 else 'FAIL'} (tracks ε/(1+ε), see DECISIONS) |")
        L.append(f"| 3b | ε_B = 0.10 → accuracy ≈ 10 %, PASS vs 20 % | {_pct(c10['accuracy'])} | "
                 f"{metrics.verdict(c10['accuracy'], metrics.ACC_LIMIT)} |")
    else:
        L += ["| 3a | accuracy tracks ε_B 1:1 | — | UNMEASURED |", "| 3b | ε_B = 0.10 → ≈10 % | — | UNMEASURED |"]
    c = c400 or (f2[0] if f2 else None)
    if c:
        L.append(f"| 4a | composition g(t) drifts K by a few %/day around the storm | storm is a density-only multiplier in the "
                 f"§2 truth, so it cannot change composition; background arc-mean ḡ at 400 km spans "
                 f"{c['gbar_min']:.4f}–{c['gbar_max']:.4f}, max change {_pct(c['gbar_max_rate_per_day'])}/day | UNMEASURED (not testable as specified) |")
        tracked = c["converged"]
        L.append(f"| 4b | the ḡ drift is tracked, not rejected | K̂ stays within 5 % of K·ḡ after {_days(c)} | "
                 f"{'PASS' if tracked else 'FAIL'} |")
    L += ["", "## Per-cell diagnostics", "",
          "| Scenario | h | POD | T_arc | arcs | R std(z) (MC, 200) | std(z/K_eff,true − 1) over run | final K̂ error | final K̂ posterior σ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, cells in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4)):
        for c in cells or []:
            L.append(f"| {name} | {c['h_km']:.0f} | {c['grade']} | {c['n_orb']} | {c['n_arcs']} | {_pct(c['R_std'])} | "
                     f"{_pct(c['z_rel_noise_vs_truth'])} | {_pct(c['khat_final_err'])} | {_pct(c.get('ksig_final_rel'))} |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n")


FIGS = {"F1": fig_f1, "F2": fig_f2, "F3": fig_f3, "F4": fig_f4, "F5": fig_f5}


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
