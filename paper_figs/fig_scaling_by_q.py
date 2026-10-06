#!/usr/bin/env python
"""[ITEM 3] Per-bin data-scaling analysis (q bins + high-spin split).

Decision rule (gates the 2x-wider 120k capacity run, §6.0b):
  # capacity limit  => slopes shallow uniformly across bins
  # corner coverage => edge bins flatten, interior keeps ~-1.1
  # => gates whether the 2x-wider 120k run (§6.0b) is worth the GPU-days

Single-command design (not two-stage): the per-waveform CSVs already exist from
eval_fm_unified.py, so there is nothing to recompute on GPU — a plot+CSV
script is enough.

Usage (from repo root)::

    python paper_figs/fig_scaling_by_q.py \\
        --points 6000:results/scale_version_0_epoch=449.csv \\
                 25000:results/scale_version_1_epoch=163.csv \\
                 120000:results/fm_120k_e092.csv

Writes paper_figs/out/fm_scaling_by_q.{png,pdf} and scaling_by_q.csv.
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

# [ITEM 3] 12k measured compute budget, not the data law — excluded by caller.
# [ITEM 3] bins chosen to isolate the corners where mismatch is 100x the interior
Q_EDGES = [1.0, 2.0, 4.0, 6.0, 8.0]
SPIN_THR = 0.4
COL = "mismatch_phase_time_future"
MIN_PER_BIN = 20


def parse_points(specs: list[str]) -> list[tuple[int, Path]]:
  out = []
  for s in specs:
    n_str, path = s.split(":", 1)
    out.append((int(n_str), Path(path)))
  return sorted(out, key=lambda x: x[0])


def bin_label(lo: float, hi: float) -> str:
  return f"q∈[{lo:g},{hi:g})"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--points", nargs="+", required=True,
                  help="N:csv pairs, e.g. 6000:results/a.csv 25000:results/b.csv")
  ap.add_argument("--col", default=COL)
  ap.add_argument("--out-csv", default="scaling_by_q.csv")
  ap.add_argument("--talk", action="store_true")
  args = ap.parse_args()

  points = parse_points(args.points)
  apply_style(talk=args.talk)
  import matplotlib.pyplot as plt

  # Load + tag each row with q-bin and spin class.
  frames = []
  edges = list(Q_EDGES)
  labels = [bin_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
  for N, path in points:
    d = pd.read_csv(path).copy()
    d["N"] = N
    d["mean_spin"] = 0.5 * (d["chi1z"].abs() + d["chi2z"].abs())
    d["spin_hi"] = d["mean_spin"] >= SPIN_THR
    # [ITEM 3] Last bin closed on the right so q=8 is included.
    d["q_bin"] = pd.cut(
      d["q"], bins=edges, right=True, include_lowest=True, labels=labels,
    )
    frames.append(d)
  all_df = pd.concat(frames, ignore_index=True)

  rows = []
  for lab in labels:
    for N, _ in points:
      sub = all_df[(all_df["q_bin"] == lab) & (all_df["N"] == N)]
      n_wf = len(sub)
      if n_wf < MIN_PER_BIN:
        # [ITEM 3] Guard: every bin needs ≥20 waveforms; fail loudly rather than
        # silently plotting noise.
        raise AssertionError(
          f"bin {lab} at N={N} has n={n_wf} < {MIN_PER_BIN}; merge bins and rerun"
        )
      med = float(sub[args.col].median())
      rows.append(dict(bin=str(lab), N=N, n_waveforms=n_wf, median=med, slope=np.nan,
                       spin_split="all"))

  # Per-bin slopes across N (log10–log10).
  slope_by_bin: dict[str, float] = {}
  for lab in labels:
    xs, ys = [], []
    for N, _ in points:
      r = next(r for r in rows if r["bin"] == str(lab) and r["N"] == N and r["spin_split"] == "all")
      xs.append(N); ys.append(r["median"])
    c = np.polyfit(np.log10(xs), np.log10(ys), 1)
    slope_by_bin[str(lab)] = float(c[0])
    for r in rows:
      if r["bin"] == str(lab) and r["spin_split"] == "all":
        r["slope"] = float(c[0])

  # Secondary: high vs low mean-spin (dashed), same q-blind aggregate for clarity
  for hi, tag in [(False, "spin<0.4"), (True, "spin≥0.4")]:
    xs, ys = [], []
    for N, _ in points:
      sub = all_df[(all_df["N"] == N) & (all_df["spin_hi"] == hi)]
      med = float(sub[args.col].median())
      xs.append(N); ys.append(med)
      rows.append(dict(bin="all_q", N=N, n_waveforms=len(sub), median=med,
                       slope=np.nan, spin_split=tag))
    c = np.polyfit(np.log10(xs), np.log10(ys), 1)
    for r in rows:
      if r["spin_split"] == tag:
        r["slope"] = float(c[0])

  out_csv = Path(args.out_csv)
  pd.DataFrame(rows).to_csv(out_csv, index=False)

  # Figure: (a) per-bin curves  (b) slope-vs-bin bars
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
  colors = [C["fm"], C["good"], C["accent"], C["gray"], C["fm_light"]]
  for i, lab in enumerate(labels):
    xs = [r["N"] for r in rows if r["bin"] == str(lab) and r["spin_split"] == "all"]
    ys = [r["median"] for r in rows if r["bin"] == str(lab) and r["spin_split"] == "all"]
    axes[0].plot(xs, ys, "o-", color=colors[i % len(colors)], lw=2, ms=7,
                 label=f"{lab} (slope {slope_by_bin[str(lab)]:.2f})")
  for tag, ls in [("spin<0.4", "--"), ("spin≥0.4", ":")]:
    xs = [r["N"] for r in rows if r["spin_split"] == tag]
    ys = [r["median"] for r in rows if r["spin_split"] == tag]
    axes[0].plot(xs, ys, ls, color=C["gray"], lw=1.6, label=tag)
  axes[0].set_xscale("log"); axes[0].set_yscale("log")
  axes[0].set_xlabel("training waveforms N")
  axes[0].set_ylabel(f"median {args.col}")
  axes[0].set_title("Scaling by q bin (+ spin split)")
  axes[0].legend(fontsize=8)

  labs = list(labels)
  slopes = [slope_by_bin[str(l)] for l in labs]
  axes[1].bar(range(len(labs)), slopes, color=C["fm"], alpha=0.85)
  axes[1].axhline(-1.1, color=C["accent"], ls="--", lw=1.2, label="ref −1.1")
  axes[1].set_xticks(range(len(labs)))
  axes[1].set_xticklabels([str(l) for l in labs], rotation=20, ha="right")
  axes[1].set_ylabel("log10–log10 slope")
  axes[1].set_title("Per-bin slope")
  axes[1].legend()
  fig.tight_layout()
  save(fig, "fm_scaling_by_q")
  print("per-bin slopes:")
  for lab, s in slope_by_bin.items():
    print(f"  {lab}: {s:.3f}")
  print(f"-> {out_csv}")


if __name__ == "__main__":
  main()
