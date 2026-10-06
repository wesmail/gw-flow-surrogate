#!/usr/bin/env python
"""Final data-scaling figure: the measured law, its bend, and the 120k landing.

Points (all: unified eval, ctx 0.5, n=500, best converged checkpoint):
    6k  -> 0.0110      (flow_6k_small, converged)
    25k -> 0.0023      (flow_25k_small e163, converged)
    120k-> 0.00076     (120k e092; evaluated on the trusted 25k test split
                        after the 120k norm-stats bug — see paper appendix)
The 12k point is shown as an open marker: stopped at a fixed step budget
(best ckpt at wall), plotted for completeness, EXCLUDED from the fit.

Fit: power law through the three converged points. Annotations: local slope
between successive pairs (the bend), the 1e-3 target crossing.

Usage (from repo root)::

    python paper_figs/fig_scaling_final.py [--talk]

Edit the NUMBERS block when new points land (e.g., the 120k resume winner).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402

# ---------------- NUMBERS (edit here) ----------------------------------------
CONVERGED_X = [6000, 25000, 120000]
CONVERGED_Y = [0.0110, 0.0023, 0.00076]
BUDGET_X, BUDGET_Y = [12000], [0.045]      # fixed-budget point, excluded from fit
TARGET = 1e-3
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(); ap.add_argument("--talk", action="store_true")
a = ap.parse_args()
apply_style(talk=a.talk)
import matplotlib.pyplot as plt

x, y = np.array(CONVERGED_X, float), np.array(CONVERGED_Y, float)
c = np.polyfit(np.log(x), np.log(y), 1)
xs = np.logspace(np.log10(5e3), np.log10(1e6), 60)

fig, ax = plt.subplots(figsize=(6.8, 5))
ax.plot(xs, np.exp(c[1]) * xs ** c[0], ":", color=C["fm"], lw=1.4,
        label=f"power-law fit (global slope {c[0]:.2f})")
ax.plot(x, y, "o-", color=C["fm"], lw=2.2, ms=9, label="converged runs (2.5M)")
ax.plot(BUDGET_X, BUDGET_Y, "o", mfc="none", mec=C["gray"], ms=9, mew=1.8,
        label="fixed-budget run (excluded from fit)")
# local slopes between successive converged pairs -> the bend, made explicit
for i in range(len(x) - 1):
    s = (np.log(y[i + 1]) - np.log(y[i])) / (np.log(x[i + 1]) - np.log(x[i]))
    xm, ym = np.sqrt(x[i] * x[i + 1]), np.sqrt(y[i] * y[i + 1])
    ax.annotate(f"slope {s:.2f}", (xm, ym), textcoords="offset points",
                xytext=(8, 8), fontsize=9, color=C["fm"])
ax.axhline(TARGET, color=C["accent"], ls="--", lw=1.5)
ax.text(5.4e3, TARGET * 1.15, "production target 1e-3", color=C["accent"], fontsize=10)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training waveforms")
ax.set_ylabel("mismatch p50 (unified, ctx 0.5, n=500)")
ax.set_title("Data scaling of the 2.5M flow-matching surrogate")
ax.legend()
save(fig, "fm_scaling_final")
print(f"global slope {c[0]:.2f}; local slopes annotated on figure")
