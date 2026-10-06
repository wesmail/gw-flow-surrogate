#!/usr/bin/env python
"""Error anatomy across data scales: the q-inversion + the cycles diagnostic.

Panel (a): mismatch vs q at 25k and at 120k training data — showing the
INVERSION (high-q worst at 25k, best at 120k; q≈1 boundary penalty persists).
Panel (b): the honest mechanism check — mismatch vs accumulated |Δφ| of the
generated segment, per data scale. If the 25k branch correlates with cycles
and the 120k branch does not, the cycle-count penalty was a data-scarcity
effect and the caption may say so.

Usage (from repo root; CSVs are unified-eval outputs at the SAME ctx)::

    python paper_figs/fig_q_anatomy.py \
        --csv-25k results/e3_fm_25k_ctx0.5.csv \
        --csv-120k results/fm120k_ctx0.5.csv \
        --manifest data/120k/manifest.parquet \
        --context-fraction 0.5 [--talk]

The accumulated phase is read from the ground-truth shards (phi track over the
generated region), matched to CSV rows by (q, chi1z, chi2z) — both CSVs must
come from the same test split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402


def accumulated_phase(manifest: str, split: str, ctx_frac: float, n: int) -> pd.DataFrame:
    """Total |Δφ| over the generated region for the first n split waveforms."""
    man = pd.read_parquet(manifest)
    rows = man[man.split == split].reset_index(drop=True).iloc[:n]
    base = Path(manifest).parent
    out = []
    for shard, grp in rows.groupby("file_path"):
        p = base / Path(shard).name
        with h5py.File(p, "r") as f:
            phi = f["phi"]
            for _, r in grp.iterrows():
                ph = phi[int(r.row_index), 0, :].astype(np.float64)
                start = int(round(len(ph) * ctx_frac))
                dphi = np.abs(np.diff(ph[start:]))
                out.append(dict(q=r.q, chi1z=r.chi1z, chi2z=r.chi2z,
                                acc_phase=float(dphi.sum())))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-25k", required=True)
    ap.add_argument("--csv-120k", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--context-fraction", type=float, default=0.5)
    ap.add_argument("--col", default="mismatch_phase_time_future")
    ap.add_argument("--talk", action="store_true")
    a = ap.parse_args()

    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt

    d25 = pd.read_csv(a.csv_25k)
    d120 = pd.read_csv(a.csv_120k)
    acc = accumulated_phase(a.manifest, a.split, a.context_fraction,
                            max(len(d25), len(d120)))

    def join(d):
        m = pd.merge_asof(d.sort_values("q"), acc.sort_values("q"), on="q",
                          direction="nearest", tolerance=1e-4,
                          suffixes=("", "_gt"))
        return m.dropna(subset=["acc_phase"])

    j25, j120 = join(d25), join(d120)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.scatter(d25.q, d25[a.col], s=9, color=C["gray"], alpha=0.5,
               label="trained on 25k")
    ax.scatter(d120.q, d120[a.col], s=9, color=C["fm"], alpha=0.6,
               label="trained on 120k")
    ax.set_yscale("log")
    ax.set_xlabel("mass ratio q"); ax.set_ylabel("mismatch")
    ax.set_title("(a) The q-anatomy inverts with data scale")
    ax.legend()

    ax = axes[1]
    for j, lab, col in ((j25, "25k", C["gray"]), (j120, "120k", C["fm"])):
        r = np.corrcoef(np.log(j.acc_phase), np.log(j[a.col].clip(1e-6)))[0, 1]
        ax.scatter(j.acc_phase, j[a.col], s=9, color=col, alpha=0.5,
                   label=f"{lab}  (log-log r = {r:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("accumulated |Δφ| of generated segment [rad]")
    ax.set_ylabel("mismatch")
    ax.set_title("(b) Cycle-count check: does more phase mean more error?")
    ax.legend()

    save(fig, "fm_q_anatomy")


if __name__ == "__main__":
    main()
