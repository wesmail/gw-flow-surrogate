#!/usr/bin/env python
"""PE with ET 2L: figures and table from the 8 runs of pe/run_pe.py.

Reads results/pe/{nr,fm}_snr<S>/{posterior.csv,summary.json} and makes:

  <out>_corner_snr<S>   corner plot, reference (NRHybSur3dq8, grey) vs our model (blue), truth lines
                        (one file per SNR; the paper shows one or two)
  <out>_vs_snr          (a) shift of our posterior relative to the reference, in units of the
                            reference width: (median_ours - median_ref) / std_ref, per parameter,
                            against SNR (expected to grow ~ linearly with SNR)
                        (b) how much worse our model fits: Delta lnL_max = best lnL(ref) - best lnL(ours),
                            and the log Bayes factor lnZ(ref) - lnZ(ours), against rho^2, with the
                            prediction rho^2 * mm_ET (from results/pe_check.json, or --mm)
  results/pe/pe_table.csv / .tex   medians, 90% intervals, shift/std, JS divergence

We compare with the REFERENCE run, not with the truth: even the exact model's posterior is not
centred on the truth in zero noise (prior and degeneracies), so the model error is the
difference between the two runs.

    python paper_figs/fig_pe.py --results results/pe --out fm_pe
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402

PARAMS = ["chirp_mass", "q", "chi_eff", "luminosity_distance", "theta_jn"]
LABELS = {"chirp_mass": r"$\mathcal{M}_c\,[M_\odot]$", "q": r"$q$", "chi_eff": r"$\chi_{\rm eff}$",
          "luminosity_distance": r"$d_L\,[\mathrm{Gpc}]$", "theta_jn": r"$\theta_{JN}$"}
SCALE = {"luminosity_distance": 1e-3}                      # Mpc -> Gpc for plotting
TEX = {"chirp_mass": r"$\mathcal{M}_c$ [$M_\odot$]", "q": r"$q$", "chi_eff": r"$\chi_{\rm eff}$",
       "luminosity_distance": r"$d_L$ [Gpc]", "theta_jn": r"$\theta_{JN}$"}


def load(results: Path, model: str, snr: str):
    d = results / f"{model}_snr{snr}"
    if not (d / "summary.json").exists():
        return None
    post = np.genfromtxt(d / "posterior.csv", delimiter=",", names=True)
    return dict(post=post, summary=json.loads((d / "summary.json").read_text()))


def js_divergence(a, b, n=256):
    """Jensen-Shannon divergence (bits) between two 1-D sample sets, Gaussian KDE."""
    from scipy.stats import gaussian_kde
    lo, hi = np.min([a.min(), b.min()]), np.max([a.max(), b.max()])
    x = np.linspace(lo, hi, n)
    p = gaussian_kde(a)(x); q = gaussian_kde(b)(x)
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda u, v: np.nansum(np.where(u > 0, u * np.log2(u / v), 0.0))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def corner_plot(ref, ours, truth, snr, name):
    import corner
    import matplotlib.pyplot as plt
    data = lambda r: np.column_stack([r["post"][k] * SCALE.get(k, 1) for k in PARAMS])
    tr = [truth[k] * SCALE.get(k, 1) for k in PARAMS]
    X_ref, X_fm = data(ref), data(ours)
    rng = [(min(a.min(), b.min()), max(a.max(), b.max())) for a, b in zip(X_ref.T, X_fm.T)]
    kw = dict(bins=30, smooth=1.0, levels=(0.5, 0.9), plot_datapoints=False, plot_density=False,
              fill_contours=False, labels=[LABELS[k] for k in PARAMS], range=rng,
              max_n_ticks=3, label_kwargs=dict(fontsize=10))
    # fresh hist_kwargs per call: corner writes the colour into the dict it is given
    fig = corner.corner(X_ref, color=C["gray"], hist_kwargs=dict(density=True, lw=1.2, color=C["gray"]), **kw)
    corner.corner(X_fm, color=C["fm"], fig=fig, hist_kwargs=dict(density=True, lw=1.2, color=C["fm"]), **kw)
    corner.overplot_lines(fig, tr, color="k", lw=0.8, ls="--")
    corner.overplot_points(fig, [tr], marker="s", color="k", ms=3)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C["gray"], lw=1.6, label="NRHybSur3dq8 (reference)"),
                        Line2D([], [], color=C["fm"], lw=1.6, label="our model"),
                        Line2D([], [], color="k", lw=0.8, ls="--", label="injected value")],
               loc="upper right", frameon=False, fontsize=10,
               title=rf"ET 2L, zero noise, SNR $= {float(snr):g}$", title_fontsize=10)
    fig.set_size_inches(7.0, 7.0)
    save(fig, f"{name}_corner_snr{snr}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/pe")
    ap.add_argument("--snrs", nargs="+", default=["15", "30", "60", "120"])
    ap.add_argument("--mm", type=float, default=None,
                    help="mm_ET for the prediction line; default: B_mismatch_ET from results/pe_check.json")
    ap.add_argument("--check-json", default="results/pe_check.json")
    ap.add_argument("--out", default="fm_pe")
    ap.add_argument("--talk", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt

    res = Path(a.results)
    runs = {s: (load(res, "nr", s), load(res, "fm", s)) for s in a.snrs}
    runs = {s: r for s, r in runs.items() if r[0] is not None and r[1] is not None}
    missing = [s for s in a.snrs if s not in runs]
    if missing:
        print(f"note: SNR {missing} not complete (need both nr and fm) - skipped")
    if not runs:
        raise SystemExit("no complete pair of runs found")
    snrs = sorted(runs, key=float)

    mm = a.mm
    if mm is None and Path(a.check_json).exists():
        mm = json.loads(Path(a.check_json).read_text()).get("B_mismatch_ET")
    if mm is None:
        print("note: no mm_ET (give --mm or results/pe_check.json): prediction line omitted")

    # ---- numbers --------------------------------------------------------------------- #
    rows, shift, rho2, dlnl, dlnz = [], {k: [] for k in PARAMS}, [], [], []
    for s in snrs:
        ref, ours = runs[s]
        sr, so = ref["summary"], ours["summary"]
        truth = {k: sr["stats"][k]["truth"] for k in PARAMS}
        rho2.append(sr["snr"] ** 2)
        dlnl.append(sr["lnL_maxphase_best"] - so["lnL_maxphase_best"])
        dlnz.append(sr["log_evidence"] - so["log_evidence"])
        for k in PARAMS:
            xr, xo = ref["post"][k], ours["post"][k]
            sd = float(np.std(xr))
            sh = (np.median(xo) - np.median(xr)) / sd
            shift[k].append(sh)
            rows.append(dict(snr=float(s), param=k, truth=truth[k],
                             ref_median=float(np.median(xr)), ref_q05=float(np.quantile(xr, .05)),
                             ref_q95=float(np.quantile(xr, .95)),
                             fm_median=float(np.median(xo)), fm_q05=float(np.quantile(xo, .05)),
                             fm_q95=float(np.quantile(xo, .95)),
                             shift_over_std=float(sh), js_bits=js_divergence(xr, xo)))
        corner_plot(ref, ours, truth, s, a.out)

    # ---- table -------------------------------------------------------------------------- #
    import csv
    with open(res / "pe_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fmt = lambda v, k: f"{v * SCALE.get(k, 1):.3g}"
    lines = [r"\begin{tabular}{rlcccrr}", r"\toprule",
             r"SNR & parameter & injected & reference & our model & shift/$\sigma$ & JS [bits] \\",
             r"\midrule"]
    for r in rows:
        k = r["param"]
        ci = lambda m, lo, hi: (f"${fmt(m, k)}^{{+{fmt(hi - m, k)}}}_{{-{fmt(m - lo, k)}}}$")
        lines.append(f"{r['snr']:g} & {TEX[k]} & {fmt(r['truth'], k)} & "
                     f"{ci(r['ref_median'], r['ref_q05'], r['ref_q95'])} & "
                     f"{ci(r['fm_median'], r['fm_q05'], r['fm_q95'])} & {r['shift_over_std']:+.2f} & "
                     f"{r['js_bits']:.3f} \\\\")
        if k == PARAMS[-1] and r is not rows[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (res / "pe_table.tex").write_text("\n".join(lines) + "\n")

    # ---- figure: shift and goodness of fit against SNR --------------------------------- #
    S = np.array([float(s) for s in snrs]); rho2 = np.array(rho2)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v"]
    for k, mk in zip(PARAMS, markers):
        ax[0].plot(S, np.abs(shift[k]), marker=mk, lw=1.0, ms=5, label=LABELS[k])
    ax[0].axhline(1.0, color=C["gray"], lw=0.8, ls=":")
    ax[0].text(S[-1], 0.93, r"one reference $\sigma$", color=C["gray"], fontsize="x-small", va="top", ha="right")
    if len(S) > 1:                                             # guide: shift proportional to SNR
        allsh = np.array([np.abs(shift[k]) for k in PARAMS])
        c = np.median(allsh.max(0) / S)
        ax[0].plot(S, c * S, color="k", lw=0.7, ls="--", label=r"$\propto$ SNR")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    from matplotlib.ticker import NullLocator
    ax[0].xaxis.set_minor_locator(NullLocator())
    ax[0].set_xticks(S); ax[0].set_xticklabels([f"{s:g}" for s in S])
    ax[0].set_xlabel("network SNR")
    ax[0].set_ylabel(r"$|\,$median$_{\rm ours}-$median$_{\rm ref}|\,/\,\sigma_{\rm ref}$")
    ax[0].legend(fontsize="x-small", ncol=2, frameon=False)
    ax[0].set_title("(a) shift of our posterior", loc="left", fontsize="medium")

    ax[1].plot(rho2, np.maximum(dlnl, 1e-3), "o-", color=C["fm"], lw=1.2,
               label=r"$\Delta\ln\mathcal{L}_{\max}$ (best fit, ref $-$ ours)")
    ax[1].plot(rho2, np.maximum(dlnz, 1e-3), "s--", color=C["accent"], lw=1.0, mfc="none",
               label=r"$\ln\mathcal{B}$ (ref vs ours)")
    if mm:
        xx = np.geomspace(rho2.min() * 0.7, rho2.max() * 1.4, 50)
        ax[1].plot(xx, xx * mm, color="k", lw=0.8, ls=":", label=rf"$\rho^2\,\mathrm{{mm}}_{{\rm ET}}$, mm $={mm:.1e}$")
    ax[1].axhline(1.0, color=C["gray"], lw=0.8, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\rho^2$"); ax[1].set_ylabel(r"$\Delta\ln\mathcal{L}$, $\ln\mathcal{B}$")
    for x, y, s_ in zip(rho2, np.maximum(dlnl, 1e-3), S):        # label points with their SNR
        ax[1].annotate(f"SNR {s_:g}", (x, y), xytext=(4, -10), textcoords="offset points",
                       fontsize="x-small", color=C["fm"])
    ax[1].legend(fontsize="x-small", frameon=False)
    ax[1].set_title("(b) how much worse our model fits", loc="left", fontsize="medium")
    save(fig, f"{a.out}_vs_snr")

    # ---- printout ---------------------------------------------------------------------- #
    print(f"{'SNR':>5} " + " ".join(f"{k[:10]:>11}" for k in PARAMS) + f" {'dlnL_max':>9} {'lnB':>7}")
    for i, s in enumerate(snrs):
        print(f"{float(s):5g} " + " ".join(f"{shift[k][i]:+11.2f}" for k in PARAMS)
              + f" {dlnl[i]:9.2f} {dlnz[i]:7.2f}" + (f"   (pred {rho2[i] * mm:.2f})" if mm else ""))
    print(f"-> figures {a.out}_corner_snr*, {a.out}_vs_snr;  tables {res}/pe_table.csv, .tex")


if __name__ == "__main__":
    main()