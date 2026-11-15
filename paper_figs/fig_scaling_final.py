#!/usr/bin/env python
"""Final data-scaling figure from ``results/paper/scaling_summary.csv``.

No hard-coded mismatch numbers.  Per-seed points faint; geometric mean (per_n
rows) solid; single-seed N as open markers.  Power-law fits on all N and on
N >= 60k with slope uncertainties; floor as a horizontal line.

Usage (from repo root)::

    python paper_figs/fig_scaling_final.py \\
        --summary results/paper/scaling_summary.csv [--talk]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402


def fit_powerlaw(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, slope_stderr) of log y = a log x + b."""
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    # Simple stderr from residual variance (n>2).
    pred = a * lx + b
    n = len(lx)
    if n <= 2:
        return float(a), float(b), float("nan")
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_xx = float(np.sum((lx - lx.mean()) ** 2))
    se = math_sqrt(ss_res / (n - 2) / ss_xx) if ss_xx > 0 else float("nan")
    return float(a), float(b), se


def math_sqrt(v: float) -> float:
    return float(np.sqrt(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/paper/scaling_summary.csv")
    ap.add_argument("--target", type=float, default=1e-3)
    ap.add_argument("--floor-col", default="floor_pt_full_median")
    ap.add_argument("--nmin-fit", type=float, default=60_000,
                    help="second power-law fit uses N >= this")
    ap.add_argument("--talk", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.summary)
    per_n = df[df["row_type"] == "per_n"].copy()
    per_s = df[df["row_type"] == "per_seed"].copy()
    if per_n.empty:
        raise SystemExit(f"no per_n rows in {a.summary}")

    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 5))

    # Faint per-seed points
    if not per_s.empty:
        ax.scatter(
            per_s["n_train"], per_s["mm_pt_full_median"],
            s=28, color=C["fm"], alpha=0.35, zorder=2, label="per-seed median",
        )

    # Solid geometric-mean line; open marker when n_seeds == 1
    multi = per_n[per_n["n_seeds"] > 1]
    single = per_n[per_n["n_seeds"] == 1]
    if not multi.empty:
        ax.plot(
            multi["n_train"], multi["mm_pt_full_median"],
            "o-", color=C["fm"], lw=2.2, ms=9, zorder=3,
            label="geometric mean over seeds",
        )
    if not single.empty:
        ax.plot(
            single["n_train"], single["mm_pt_full_median"],
            "o", mfc="none", mec=C["fm"], ms=9, mew=1.8, zorder=3,
            label="single-seed N",
        )

    x = per_n["n_train"].to_numpy(float)
    y = per_n["mm_pt_full_median"].to_numpy(float)
    xs = np.logspace(np.log10(max(x.min() * 0.8, 1e3)), np.log10(x.max() * 1.5), 80)

    a_all, b_all, se_all = fit_powerlaw(x, y)
    ax.plot(
        xs, np.exp(b_all) * xs ** a_all, ":", color=C["fm"], lw=1.4,
        label=f"fit all N: slope {a_all:.2f}±{se_all:.2f}",
    )

    mask = x >= float(a.nmin_fit)
    if mask.sum() >= 2:
        a2, b2, se2 = fit_powerlaw(x[mask], y[mask])
        ax.plot(
            xs[xs >= float(a.nmin_fit)],
            np.exp(b2) * xs[xs >= float(a.nmin_fit)] ** a2,
            "--", color=C["accent"], lw=1.4,
            label=f"fit N≥{a.nmin_fit:.0e}: slope {a2:.2f}±{se2:.2f}",
        )

    floor = float(per_n[a.floor_col].median()) if a.floor_col in per_n else np.nan
    if np.isfinite(floor) and floor > 0:
        ax.axhline(floor, color=C["gray"], ls="-.", lw=1.3,
                   label=f"floor (median) {floor:.2e}")

    ax.axhline(a.target, color=C["accent"], ls="--", lw=1.2, alpha=0.7)
    ax.text(x.min(), a.target * 1.15, f"target {a.target:.0e}",
            color=C["accent"], fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training waveforms")
    ax.set_ylabel("median mm_pt_full (protocol)")
    ax.set_title("Data scaling of the flow-matching surrogate")
    ax.legend(fontsize="x-small", frameon=False)
    save(fig, "fm_scaling_final")
    print(f"fit all N: slope={a_all:.3f} ± {se_all:.3f}")


if __name__ == "__main__":
    main()
