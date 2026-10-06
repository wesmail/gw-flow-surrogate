#!/usr/bin/env python
"""Speed-accuracy frontier: mismatch and wall-clock vs ODE integration steps.

The flow head pays n_sampling_steps network calls per patch; fewer steps =
faster surrogate, possibly worse waveforms. This measures the trade and plots
the frontier, with an optional reference line for the conventional generator.

STAGE 1 (eval, GPU)::

    python paper_figs/fig_speed_frontier.py eval \
        --checkpoint <ckpt> --manifest data/120k/manifest.parquet \
        --steps 5 10 20 50 --n-waveforms 200 --out results/speed_frontier.csv

STAGE 2 (figure)::

    python paper_figs/fig_speed_frontier.py plot results/speed_frontier.csv \
        [--ref-time-ms 120 --ref-label "NRHybSur3dq8"] [--talk]

Timing protocol: batched generation, CUDA-synchronized, 2 warmup batches
excluded; reported as milliseconds per waveform. Measure --ref-time-ms with
the same batch discipline on the same machine (snippet in README) or omit
the reference line.

RUN FROM THE REPO ROOT.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402


def mismatch_pt(h1, h2) -> float:
    h1 = np.asarray(h1, np.complex128).ravel()
    h2 = np.asarray(h2, np.complex128).ravel()
    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
    if n1 == 0 or n2 == 0:
        return 1.0
    n_fft = 1 << int(np.ceil(np.log2(2 * h1.size)))
    xc = np.fft.ifft(np.conj(np.fft.fft(h1, n_fft)) * np.fft.fft(h2, n_fft))
    return 1.0 - float(np.abs(xc).max() / (n1 * n2))


def run_eval(a) -> None:
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
    batches = list(loader)

    rows = []
    with torch.no_grad():
        for n_steps in a.steps:
            # warmup (kernel compilation, allocator) — excluded from timing
            for b in batches[:2]:
                fm_eval.evaluate_batch(lit, b, n_samples=1, n_steps=n_steps,
                                       limit_patches=0, device=torch.device(dev),
                                       mean_of=1)
            mms, t_total, n_wf = [], 0.0, 0
            for b in batches:
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = fm_eval.evaluate_batch(lit, b, n_samples=1, n_steps=n_steps,
                                           limit_patches=0, device=torch.device(dev),
                                           mean_of=1)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t_total += time.perf_counter() - t0
                h_t, h_p = r["h_true"].numpy(), r["h_pred"].numpy()
                mms += [mismatch_pt(h_t[i], h_p[i]) for i in range(h_t.shape[0])]
                n_wf += h_t.shape[0]
            rows.append(dict(n_steps=n_steps,
                             mismatch_p50=float(np.median(mms)),
                             mismatch_p90=float(np.percentile(mms, 90)),
                             ms_per_waveform=1e3 * t_total / n_wf))
            print(rows[-1])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


def run_plot(a) -> None:
    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt

    d = pd.read_csv(a.csv).sort_values("ms_per_waveform")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(d.ms_per_waveform, d.mismatch_p50, "o-", color=C["fm"], lw=2, ms=8,
            label="FM surrogate (median)")
    ax.fill_between(d.ms_per_waveform, d.mismatch_p50, d.mismatch_p90,
                    color=C["band"], alpha=0.7, label="to p90")
    for _, r in d.iterrows():
        ax.annotate(f"{int(r.n_steps)} steps", (r.ms_per_waveform, r.mismatch_p50),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(1e-3, color=C["accent"], ls="--", lw=1.4)
    ax.text(d.ms_per_waveform.min(), 1.15e-3, "production target",
            color=C["accent"], fontsize=9)
    if a.ref_time_ms is not None:
        ax.axvline(a.ref_time_ms, color=C["gray"], ls="-.", lw=1.4)
        ax.text(a.ref_time_ms * 1.05, ax.get_ylim()[1] * 0.5, a.ref_label,
                color=C["gray"], rotation=90, va="top", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("generation time per waveform [ms]")
    ax.set_ylabel("mismatch")
    ax.set_title("Speed-accuracy frontier")
    ax.legend()
    save(fig, "fm_speed_frontier")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--context-fraction", type=float, default=0.5)
    e.add_argument("--steps", nargs="+", type=int, default=[5, 10, 20, 50])
    e.add_argument("--n-waveforms", type=int, default=200)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--out", default="results/speed_frontier.csv")
    p = sub.add_parser("plot")
    p.add_argument("csv")
    p.add_argument("--ref-time-ms", type=float, default=None)
    p.add_argument("--ref-label", default="reference generator")
    p.add_argument("--talk", action="store_true")
    args = ap.parse_args()
    (run_eval if args.cmd == "eval" else run_plot)(args)
