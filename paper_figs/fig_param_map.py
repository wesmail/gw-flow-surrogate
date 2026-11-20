#!/usr/bin/env python
"""Where in parameter space are the errors?  Binned maps over (q, chi_eff).

Panel (a): median full-waveform mismatch per cell (log colour, FIXED range so
           maps of different models are comparable).
Panel (b): share of waveforms above the target (1e-3) per cell, with the
           percentage written in each cell.
Cells with fewer than --min-count waveforms are greyed out (too few to trust).

Input: a fig_context_fraction.py eval CSV (theta-only rows are used).

Usage (repo root)::

    python paper_figs/fig_param_map.py results/headline_K12_M1.csv \
        --rows headline --out fm_param_map

--rows headline : test waveforms idx >= 1000 (the headline rows; default)
--rows all      : every row in the CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402


def binned(d: pd.DataFrame, col: str, q_edges, c_edges, target: float):
    qi = np.digitize(d.q, q_edges) - 1
    ci = np.digitize(d.chi_eff, c_edges) - 1
    nq, nc = len(q_edges) - 1, len(c_edges) - 1
    med = np.full((nc, nq), np.nan); bad = np.full((nc, nq), np.nan)
    cnt = np.zeros((nc, nq), int)
    for i in range(nq):
        for j in range(nc):
            m = d[col].to_numpy()[(qi == i) & (ci == j)]
            cnt[j, i] = m.size
            if m.size:
                med[j, i] = np.median(m); bad[j, i] = (m > target).mean()
    return med, bad, cnt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--col", default="mm_pt_full", help="mismatch column (default: full waveform, phase+time opt.)")
    ap.add_argument("--rows", choices=["headline", "all"], default="headline")
    ap.add_argument("--target", type=float, default=1e-3)
    ap.add_argument("--min-count", type=int, default=15)
    ap.add_argument("--vmin", type=float, default=1e-4)
    ap.add_argument("--vmax", type=float, default=2e-3)
    ap.add_argument("--q-step", type=float, default=1.0)
    ap.add_argument("--chi-step", type=float, default=0.2)
    ap.add_argument("--out", default="fm_param_map")
    ap.add_argument("--talk", action="store_true")
    ap.add_argument("--single", action="store_true",
                    help="one panel: colour = median, number in each cell = %% above target")
    a = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    d = pd.read_csv(a.csv)
    if "context_fraction" in d:
        d = d[d.context_fraction == d.context_fraction.min()]
    if a.rows == "headline":
        d = d[d.idx >= 1000]
    if "chi_eff" not in d:
        d["chi_eff"] = (d.q * d.chi1z + d.chi2z) / (1 + d.q)
    n = len(d)
    if n == 0:
        raise SystemExit("no rows selected (did the eval cover idx >= 1000? try --rows all)")

    q_edges = np.arange(1.0, 8.0 + 1e-9, a.q_step)
    c_edges = np.round(np.arange(-0.8, 0.8 + 1e-9, a.chi_step), 6)
    med, bad, cnt = binned(d, a.col, q_edges, c_edges, a.target)
    few = cnt < a.min_count
    med_m = np.ma.masked_where(few, med); bad_m = np.ma.masked_where(few, bad)

    apply_style(talk=a.talk)
    if a.single:
        fig, ax0 = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
        ax0.set_facecolor("#E6E6E6"); ax0.grid(False)
        norm = LogNorm(vmin=a.vmin, vmax=a.vmax)
        pa = ax0.pcolormesh(q_edges, c_edges, med_m, cmap="viridis", norm=norm, shading="flat")
        cb = fig.colorbar(pa, ax=ax0); cb.set_label("median mismatch")
        for i in range(len(q_edges) - 1):
            for j in range(len(c_edges) - 1):
                if not few[j, i]:
                    light = norm(med[j, i]) > 0.6          # yellow cells: dark text
                    ax0.text(0.5 * (q_edges[i] + q_edges[i + 1]), 0.5 * (c_edges[j] + c_edges[j + 1]),
                             f"{100 * bad[j, i]:.0f}", ha="center", va="center", fontsize=7,
                             color="black" if light else "white")
        ax0.set_xlabel("mass ratio $q$"); ax0.set_ylabel(r"effective spin $\chi_\mathrm{eff}$")
        ax0.set_xlim(q_edges[0], q_edges[-1]); ax0.set_ylim(c_edges[0], c_edges[-1])
        save(fig, a.out)
        ax = None
    else:
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9), constrained_layout=True)
        ext = dict(shading="flat")
        for axi in ax:
            axi.set_facecolor("#E6E6E6")           # grey = too few waveforms
            axi.grid(False)

        cmap_a = plt.get_cmap("viridis").copy()
        pa = ax[0].pcolormesh(q_edges, c_edges, med_m, cmap=cmap_a,
                              norm=LogNorm(vmin=a.vmin, vmax=a.vmax), **ext)
        cb = fig.colorbar(pa, ax=ax[0]); cb.set_label("median mismatch")
        ax[0].set_title("(a) typical error")

        cmap_b = plt.get_cmap("Reds").copy()
        pb = ax[1].pcolormesh(q_edges, c_edges, 100 * bad_m, cmap=cmap_b, vmin=0, vmax=30, **ext)
        cb2 = fig.colorbar(pb, ax=ax[1], extend="max"); e = int(np.round(np.log10(a.target)))
        cb2.set_label(f"share above $10^{{{e}}}$ [%]" if np.isclose(a.target, 10.0 ** e) else f"share above {a.target:g} [%]")
        ax[1].set_title("(b) how often the target is missed")
        for i in range(len(q_edges) - 1):
            for j in range(len(c_edges) - 1):
                if not few[j, i]:
                    v = 100 * bad[j, i]
                    ax[1].text(0.5 * (q_edges[i] + q_edges[i + 1]), 0.5 * (c_edges[j] + c_edges[j + 1]),
                               f"{v:.0f}", ha="center", va="center", fontsize=7,
                               color="white" if v > 18 else "black")

        for axi in ax:
            axi.set_xlabel("mass ratio $q$")
            axi.set_xlim(q_edges[0], q_edges[-1]); axi.set_ylim(c_edges[0], c_edges[-1])
        ax[0].set_ylabel(r"effective spin $\chi_\mathrm{eff}$")

        save(fig, a.out)

    # ---- numbers for the caption / text -----------------------------------
    m = d[a.col]
    print(f"\n{n} waveforms ({a.rows} rows), column {a.col}; grey cells: < {a.min_count} waveforms "
          f"({int(cnt[few].sum())} waveforms in {int(few.sum())} cells)")
    print(f"overall: median {m.median():.3e}, share above target {(m > a.target).mean()*100:.1f}%")
    flat = [(bad[j, i], med[j, i], cnt[j, i], q_edges[i], c_edges[j])
            for i in range(len(q_edges) - 1) for j in range(len(c_edges) - 1) if not few[j, i]]
    flat.sort(reverse=True)
    print("worst cells (share above target):")
    for b, mm, c, q0, c0 in flat[:5]:
        print(f"  q {q0:.0f}-{q0 + a.q_step:.0f}, chi_eff {c0:+.1f} to {c0 + a.chi_step:+.1f}: "
              f"{b*100:4.1f}% above, median {mm:.2e}, n={c}")
    print("median by chi_eff band (all q):")
    for lo, hi in zip(c_edges[:-1], c_edges[1:]):
        s = m[(d.chi_eff >= lo) & (d.chi_eff < hi)]
        if len(s):
            print(f"  {lo:+.1f}..{hi:+.1f}: median {s.median():.2e}, above {((s > a.target).mean())*100:4.1f}%, n={len(s)}")


if __name__ == "__main__":
    main()