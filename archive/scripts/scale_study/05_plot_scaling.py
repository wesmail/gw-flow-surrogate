#!/usr/bin/env python
"""Panel (a) data scaling + panel (b) model size at 120k.

Reads ``results/scale_study/summary.csv`` and ``fit.json`` / ``prediction.json``.

Usage::

    python scripts/scale_study/05_plot_scaling.py \\
        --results-dir results/scale_study \\
        --out results/scale_study/fig_scale_study.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/scale_study"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--target", type=float, default=1e-4)
    args = ap.parse_args()

    summary = pd.read_csv(args.results_dir / "summary.csv")
    fit = json.loads((args.results_dir / "fit.json").read_text()) if (args.results_dir / "fit.json").is_file() else None
    pred = json.loads((args.results_dir / "prediction.json").read_text()) if (args.results_dir / "prediction.json").is_file() else None

    out = args.out or (args.results_dir / "fig_scale_study.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel (a): mismatch vs N (2.5M / 2p5m only) -------------------------
    scal = summary[(summary["model_tag"] == "2p5m") & summary["median"].notna()].copy()
    scal = scal.sort_values("n_train")
    fit_pts = scal[scal["phase"] == "scaling"]
    held = scal[scal["phase"] == "heldout"]

    if fit is not None and len(fit_pts):
        A = fit["fit_no_floor"]["A"]
        B = fit["fit_no_floor"]["B"]
        xs = np.logspace(np.log10(max(fit_pts["n_train"].min() * 0.8, 1e3)),
                         np.log10(1e6), 80)
        ax_a.plot(xs, A * xs ** B, ":", color="C0", lw=1.5, label=f"power law (slope {B:.2f})")
        if "fit_with_floor" in fit:
            Af, Bf, Cf = fit["fit_with_floor"]["A"], fit["fit_with_floor"]["B"], fit["fit_with_floor"]["C"]
            ax_a.plot(xs, Af * xs ** Bf + Cf, "--", color="C0", lw=1.2, alpha=0.7,
                      label=f"with floor C={Cf:.1e}")

    if len(fit_pts):
        ax_a.plot(fit_pts["n_train"], fit_pts["median"], "o-", color="C0", ms=8, lw=2,
                  label="2.5M scaling (fit)")
    if len(held):
        ax_a.plot(held["n_train"], held["median"], "s", color="C3", ms=10,
                  label="300k held-out check")
        if pred is not None:
            yhat = pred["headline"]["predicted_mismatch_at_300k"]
            ax_a.axhline(yhat, color="C3", ls=":", lw=1.2, alpha=0.8)
            ax_a.annotate(f"pred {yhat:.2e}", xy=(held["n_train"].iloc[0], yhat),
                          textcoords="offset points", xytext=(8, 6), fontsize=9, color="C3")

    ax_a.axhline(args.target, color="0.3", ls="--", lw=1.2)
    ax_a.text(scal["n_train"].min() if len(scal) else 6e3, args.target * 1.2,
              f"target {args.target:.0e}", fontsize=9, color="0.3")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("training waveforms")
    ax_a.set_ylabel("median mismatch (phase+time, future)")
    ax_a.set_title("(a) Data scaling — 2.5M model")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, which="both", ls=":", alpha=0.4)

    # --- Panel (b): mismatch vs model size at 120k ---------------------------
    cap = summary[(summary["n_train"] == 120000) & summary["median"].notna()].copy()
    # Order by approximate param count
    order = {"0p6m": 0.65e6, "2p5m": 3.1e6, "10m": 9.6e6}
    if len(cap):
        cap["n_params"] = cap["model_tag"].map(order)
        cap = cap.dropna(subset=["n_params"]).sort_values("n_params")
        ax_b.plot(cap["n_params"], cap["median"], "o-", color="C1", ms=9, lw=2)
        for _, r in cap.iterrows():
            ax_b.annotate(r["model_tag"], (r["n_params"], r["median"]),
                          textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("model parameters (approx)")
    ax_b.set_ylabel("median mismatch")
    ax_b.set_title("(b) Model size at 120k")
    ax_b.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
