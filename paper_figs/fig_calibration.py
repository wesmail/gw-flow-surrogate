#!/usr/bin/env python
"""UQ calibration: does the model's sample spread predict its actual error?

The flow head is stochastic: S independent generations per waveform give an
ensemble. Two numbers per waveform:
  spread  = mean pairwise mismatch BETWEEN the S samples  (needs no truth)
  error   = mean mismatch of the S samples AGAINST the truth
If the spread ranks waveforms by their true error, the surrogate tells its user
which waveforms to trust -- something a deterministic surrogate cannot do.

Expected relation (small phase errors, mismatch ~ 0.5 <dphi^2>): with a shared
bias b and a random part of variance s^2,  error ~ 0.5 (b^2 + s^2),
spread ~ s^2,  so  error ~ spread/2 + b^2/2.  Points ABOVE the line
error = spread/2 carry a shared (systematic) error the spread cannot see.

FROZEN PROTOCOL: theta-only, same engine and settings as the headline evaluation
(read from its .json: n_steps, mean_of, seed, batch size, amp mode), the
original waveform as truth, headline rows (idx >= 1000). Sample s = 0 uses the
evaluation's own seed, so it IS the headline waveform (checked against the CSV).

STAGE 1 (GPU, ~10 min for 1000 waveforms x 8 samples)::

    python paper_figs/fig_calibration.py eval --checkpoint "$FT" \
        --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
        --eval-csv results/headline_K12_M1.csv --S 8 --n-waveforms 1000 \
        --out results/calibration_K12.csv

STAGE 2 (CPU, seconds)::

    python paper_figs/fig_calibration.py plot results/calibration_K12.csv

Figure: (a) spread vs true error per waveform, with Spearman rank correlation;
(b) reliability: waveforms in spread deciles, median and p90 true error per bin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402

FIG_W = 11.0     # all full-width paper figures share this width (fonts then match)


def run_eval(a: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset
    from datasets.datamodule import _collate_batch
    from datasets.patchify import n_patches_for
    from datasets.waveform_dataset import WaveformPatchDataset
    from models.amp_interp import log_amp_per_sample
    from models.lightning_module import GWFlowSurrogateLit

    csv = pd.read_csv(a.eval_csv)
    if "context_fraction" in csv:
        csv = csv[csv.context_fraction == csv.context_fraction.min()]
    csv_mm = dict(zip(csv.idx.astype(int), csv.mm_pt_full))
    meta_p = Path(a.eval_csv).with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    n_steps = a.n_steps or int(meta.get("n_steps", 12))
    mean_of = int(meta.get("mean_of", 1)); seed = int(meta.get("seed", 1234))
    bs = int(meta.get("batch_size", 64)); temp = float(meta.get("temperature", 1.0))
    amp_mode = a.amp_mode or meta.get("amp_mode", "pchip")
    same_as_eval = n_steps == int(meta.get("n_steps", n_steps))

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev).eval().to(dev)
    P, rscale = int(lit.hparams.patch_len), float(lit.hparams.residual_scale)
    ds = WaveformPatchDataset(a.manifest, split=a.split, patch_len=P, context_patches=1,
                              norm_stats_path=a.norm_stats, preload=False)
    N = n_patches_for(ds.n_time, P); usable = N * P
    n_eval = min(int(meta.get("n_waveforms", len(ds))), len(ds))
    row0 = a.first_row; row1 = min(n_eval, row0 + a.n_waveforms)
    print(f"rows {row0}..{row1 - 1}, S={a.S} samples each; theta-only; n_steps={n_steps} "
          f"mean_of={mean_of} seed={seed} batch={bs} amp={amp_mode}")

    s_frac = torch.arange(P, device=dev, dtype=torch.float64) / P

    def reconstruct(feat, b):
        feat = feat.double()
        la_m = b["norm_log_a_mean"][:, :1].double(); la_s = b["norm_log_a_std"][:, :1].double()
        dp_m = b["norm_dphi_mean"][:, :1].double(); dp_s = b["norm_dphi_std"][:, :1].double()
        la = log_amp_per_sample(feat[..., 0] * la_s + la_m, P, amp_mode)
        dphi = feat[..., 1] * dp_s + dp_m
        res = feat[..., 2:2 + P] * rscale
        bnd = torch.cumsum(dphi, dim=1) - dphi
        phase = (bnd[..., None] + s_frac * dphi[..., None] + res).reshape(feat.shape[0], -1)
        return torch.exp(la) * torch.exp(-1j * phase)

    nfft = 1 << int(np.ceil(np.log2(2 * usable)))

    def mm_pt(F1, n1, F2, n2):
        """phase+time-optimised mismatch from precomputed FFTs, row-wise."""
        xc = torch.fft.ifft(F1.conj() * F2)
        return 1 - xc.abs().amax(dim=1) / (n1 * n2)

    rows = []
    with torch.no_grad():
        for bi in range(row0 // bs, (row1 - 1) // bs + 1):
            lo, hi = bi * bs, min((bi + 1) * bs, n_eval)
            batch = next(iter(DataLoader(Subset(ds, range(lo, hi)), batch_size=bs,
                                         collate_fn=_collate_batch, shuffle=False)))
            b = lit._patchify_batch({k: v.to(dev) for k, v in batch.items()})
            feat = b["token_features"]
            la_o = b["log_a_track"][:, 0, :usable].double(); ph_o = b["phi_track"][:, 0, :usable].double()
            ho = torch.exp(la_o) * torch.exp(-1j * (ph_o - ph_o[:, :1]))
            Fo, no = torch.fft.fft(ho, nfft), torch.linalg.vector_norm(ho, dim=1)
            Fs, ns, err = [], [], []
            for s in range(a.S):
                # s = 0: the evaluation's own seed -> exactly the headline waveform
                torch.manual_seed(seed + bi + 100_000 * s)
                gen = lit.model.generate(feat[:, :1], b["theta"], N - 1, n_steps=n_steps, n_samples=1,
                                         temperature=temp, mean_of=mean_of)[:, 0]
                hp = reconstruct(torch.cat([feat[:, :1], gen], dim=1), b)
                F, n_ = torch.fft.fft(hp, nfft), torch.linalg.vector_norm(hp, dim=1)
                Fs.append(F); ns.append(n_); err.append(mm_pt(Fo, no, F, n_))
            err = torch.stack(err)                                   # (S, B)
            pair = torch.stack([mm_pt(Fs[i], ns[i], Fs[j], ns[j])
                                for i in range(a.S) for j in range(i + 1, a.S)])   # (S(S-1)/2, B)
            err, pair = err.cpu().numpy(), pair.cpu().numpy()
            th = b["theta"].double().cpu().numpy()
            for j in range(hi - lo):
                idx = lo + j
                if idx < row0 or idx >= row1:
                    continue
                q, c1, c2 = th[j, :3]
                rows.append(dict(idx=idx, q=q, chi1z=c1, chi2z=c2, chi_eff=(q * c1 + c2) / (1 + q),
                                 err_deployed=err[0, j], err_mean=err[:, j].mean(),
                                 err_std=err[:, j].std(), spread=pair[:, j].mean(),
                                 err_csv=csv_mm.get(idx, np.nan)))
            print(f"  {min(hi, row1) - row0}/{row1 - row0}")

    d = pd.DataFrame(rows)
    if same_as_eval and d.err_csv.notna().any():
        rel = np.nanmax(np.abs(d.err_deployed / d.err_csv - 1))
        print(f"check: sample 0 vs headline CSV, largest relative difference {rel:.1e} (should be ~0)")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(dict(
        checkpoint=str(a.checkpoint), eval_csv=str(a.eval_csv), rows=[row0, row1], S=a.S,
        n_steps=n_steps, mean_of=mean_of, seed=seed, batch_size=bs, amp_mode=amp_mode,
        seed_rule="sample s uses seed + batch + 100000*s; s=0 equals the headline evaluation"), indent=2))
    print("->", out)


def run_plot(a: argparse.Namespace) -> None:
    from scipy.stats import spearmanr

    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt

    d = pd.read_csv(a.csv)
    rho = spearmanr(d.spread, d.err_mean).statistic
    rho_dep = spearmanr(d.spread, d.err_deployed).statistic
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.2), constrained_layout=True)

    ax = axes[0]
    hi = d.chi_eff > 0.4
    ax.scatter(d.spread[~hi], d.err_mean[~hi], s=9, color=C["fm"], alpha=0.5, lw=0,
               label=r"$\chi_\mathrm{eff}\leq0.4$")
    ax.scatter(d.spread[hi], d.err_mean[hi], s=9, color=C["accent"], alpha=0.6, lw=0,
               label=r"$\chi_\mathrm{eff}>0.4$")
    lims = np.array([min(d.spread.min(), 2 * d.err_mean.min()) * 0.7,
                     max(d.spread.max(), 2 * d.err_mean.max()) * 1.4])
    ax.plot(lims, lims / 2, ls=":", color=C["gray"], lw=1.2, label="error = spread / 2\n(no shared error)")
    ax.axhline(1e-3, color=C["accent"], ls="--", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims)
    ax.set_xlabel("spread (mismatch between samples)")
    ax.set_ylabel("true error (mismatch vs NRHybSur3dq8)")
    ax.set_title(f"(a) spread vs error   (Spearman $\\rho$ = {rho:.2f})")
    ax.legend(loc="upper left", fontsize="x-small", frameon=False)

    ax = axes[1]
    s = d.sort_values("spread").reset_index(drop=True)
    edges = np.linspace(0, len(s), 11, dtype=int)
    bins = [s.iloc[edges[i]:edges[i + 1]] for i in range(10)]
    x = np.array([bb.spread.median() for bb in bins])
    y50 = [bb.err_mean.median() for bb in bins]
    y90 = [bb.err_mean.quantile(0.9) for bb in bins]
    ax.fill_between(x, y50, y90, color=C["band"], alpha=0.8, lw=0, label="median to p90")
    ax.plot(x, y50, "o-", color=C["fm"], lw=1.8, ms=4, label="median true error")
    ax.plot(x, x / 2, ls=":", color=C["gray"], lw=1.2, label="error = spread / 2")
    ax.axhline(1e-3, color=C["accent"], ls="--", lw=0.9, label="target $10^{-3}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("spread (10 bins of equal size)")
    ax.set_ylabel("true error")
    ax.set_title("(b) reliability: error per spread bin")
    ax.legend(loc="upper left", fontsize="x-small", frameon=False)
    save(fig, a.out)

    # ---- numbers for the text ---------------------------------------------
    bad = d.err_deployed > 1e-3
    print(f"\n{len(d)} waveforms")
    print(f"Spearman rho(spread, mean error)     = {rho:.2f}")
    print(f"Spearman rho(spread, deployed error) = {rho_dep:.2f}")
    print(f"median spread {d.spread.median():.2e}; median mean error {d.err_mean.median():.2e}; "
          f"ratio error/(spread/2) median {np.median(d.err_mean / (d.spread / 2)):.2f}")
    print(f"share of waveforms above the line error = spread/2: {(d.err_mean > d.spread / 2).mean() * 100:.0f}%")
    print(f"deployed waveforms above 1e-3: {bad.sum()} of {len(d)}")
    for top in (0.05, 0.10, 0.20):
        flag = d.spread >= d.spread.quantile(1 - top)
        caught = (flag & bad).sum() / max(bad.sum(), 1)
        print(f"  flag the {int(top * 100):2d}% largest spreads -> catches {caught * 100:.0f}% of the failures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--norm-stats", required=True)
    e.add_argument("--eval-csv", required=True, help="headline eval CSV (+ .json): settings and check")
    e.add_argument("--split", default="test")
    e.add_argument("--S", type=int, default=8)
    e.add_argument("--first-row", type=int, default=1000, help="headline rows start at 1000")
    e.add_argument("--n-waveforms", type=int, default=1000)
    e.add_argument("--n-steps", type=int, default=None, help="default: the headline setting")
    e.add_argument("--amp-mode", default=None, choices=("constant", "linear", "pchip"))
    e.add_argument("--device", default=None)
    e.add_argument("--out", default="results/calibration_K12.csv")
    p = sub.add_parser("plot")
    p.add_argument("csv")
    p.add_argument("--out", default="fm_calibration")
    p.add_argument("--talk", action="store_true")
    args = ap.parse_args()
    (run_eval if args.cmd == "eval" else run_plot)(args)