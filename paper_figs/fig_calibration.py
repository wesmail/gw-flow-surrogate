#!/usr/bin/env python
"""UQ calibration: does the model's sample spread predict its actual error?

The flow head is stochastic: S independent generations per waveform give an
ensemble. If the ensemble's internal SPREAD correlates with the TRUE error
(vs. ground truth), the surrogate provides usable uncertainty — the selling
point no deterministic surrogate has. This script measures that.

STAGE 1 (eval, GPU, ~30-60 min for 500x8 samples)::

    python paper_figs/fig_calibration.py eval \
        --checkpoint <ckpt> --manifest data/120k/manifest.parquet \
        --split test --context-fraction 0.5 --n-waveforms 500 --S 8 \
        --out results/calibration.csv

STAGE 2 (figure, CPU, seconds)::

    python paper_figs/fig_calibration.py plot results/calibration.csv [--talk]

Figure: (a) scatter of ensemble spread vs true error with rank correlation;
(b) reliability curve: waveforms binned by spread decile, median true error
per bin — monotone rising = calibrated ordering.

RUN FROM THE REPO ROOT (needs evalute.py + datasets/ + models/ importable).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402


def mismatch_pt(h1: np.ndarray, h2: np.ndarray) -> float:
    """Phase+time-optimized mismatch (unified convention)."""
    h1 = np.asarray(h1, np.complex128).ravel()
    h2 = np.asarray(h2, np.complex128).ravel()
    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
    if n1 == 0 or n2 == 0:
        return 1.0
    n_fft = 1 << int(np.ceil(np.log2(2 * h1.size)))
    xc = np.fft.ifft(np.conj(np.fft.fft(h1, n_fft)) * np.fft.fft(h2, n_fft))
    return 1.0 - float(np.abs(xc).max() / (n1 * n2))


def run_eval(a: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    from datasets.patchify import n_patches_for
    from datasets.waveform_dataset import WaveformPatchDataset
    from models.lightning_module import GWFlowSurrogateLit

    spec = importlib.util.spec_from_file_location("fm_eval", _ROOT / "evalute.py")
    fm_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm_eval)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev).eval().to(dev)
    stats = Path(a.manifest).parent / "norm_stats.npz"
    probe = WaveformPatchDataset(a.manifest, split=a.split, context_patches=1,
                                 norm_stats_path=stats, preload=False)
    N = n_patches_for(probe.n_time, lit.hparams.patch_len)
    ctx = max(1, int(round(N * a.context_fraction)))
    ds = WaveformPatchDataset(a.manifest, split=a.split, context_patches=ctx,
                              norm_stats_path=stats, preload=True)
    n = min(a.n_waveforms, len(ds))
    loader = DataLoader(Subset(ds, range(n)), batch_size=a.batch_size,
                        collate_fn=fm_eval.collate)

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            # S independent single-sample generations (mean_of=1): the ensemble
            preds, h_true, theta = [], None, None
            for _ in range(a.S):
                r = fm_eval.evaluate_batch(lit, batch, n_samples=1, n_steps=a.n_steps,
                                           limit_patches=0, device=torch.device(dev),
                                           mean_of=1)
                preds.append(r["h_pred"].numpy())
                h_true, theta = r["h_true"].numpy(), r["theta"]
            preds = np.stack(preds)                      # (S, B, T)
            for b in range(h_true.shape[0]):
                mms = [mismatch_pt(h_true[b], preds[s, b]) for s in range(a.S)]
                # ensemble spread: mean pairwise mismatch BETWEEN samples
                # (ground-truth-free -> available at deployment time)
                pair = [mismatch_pt(preds[i, b], preds[j, b])
                        for i in range(a.S) for j in range(i + 1, a.S)]
                q, c1, c2 = theta[b, :3].tolist()
                rows.append(dict(
                    idx=len(rows), q=q, chi1z=c1, chi2z=c2,
                    err_mean=float(np.mean(mms)), err_std=float(np.std(mms)),
                    spread=float(np.mean(pair)),
                ))
            print(f"  {len(rows)}/{n}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


def run_plot(a: argparse.Namespace) -> None:
    from scipy.stats import spearmanr

    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt

    d = pd.read_csv(a.csv)
    rho = spearmanr(d.spread, d.err_mean).statistic
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.scatter(d.spread, d.err_mean, s=10, color=C["fm"], alpha=0.55)
    lims = [min(d.spread.min(), d.err_mean.min()) * 0.7,
            max(d.spread.max(), d.err_mean.max()) * 1.4]
    ax.plot(lims, lims, ls=":", color=C["gray"], lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ensemble spread (sample-to-sample mismatch)")
    ax.set_ylabel("true error (mismatch vs ground truth)")
    ax.set_title(f"Spread predicts error  (Spearman ρ = {rho:.2f})")

    ax = axes[1]
    d = d.sort_values("spread").reset_index(drop=True)
    edges = np.linspace(0, len(d), 11, dtype=int)
    bins = [d.iloc[edges[i]:edges[i + 1]] for i in range(10)]
    x = [b.spread.median() for b in bins]
    y50 = [b.err_mean.median() for b in bins]
    y90 = [b.err_mean.quantile(0.9) for b in bins]
    ax.plot(x, y50, "o-", color=C["fm"], lw=2, label="median true error")
    ax.fill_between(x, y50, y90, color=C["band"], alpha=0.7, label="to p90")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ensemble spread (decile bins)")
    ax.set_ylabel("true error")
    ax.set_title("Reliability: binned by predicted uncertainty")
    ax.legend()

    save(fig, "fm_calibration")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--context-fraction", type=float, default=0.5)
    e.add_argument("--n-waveforms", type=int, default=500)
    e.add_argument("--S", type=int, default=8)
    e.add_argument("--n-steps", type=int, default=20)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--out", default="results/calibration.csv")
    p = sub.add_parser("plot")
    p.add_argument("csv")
    p.add_argument("--talk", action="store_true")
    args = ap.parse_args()
    (run_eval if args.cmd == "eval" else run_plot)(args)
