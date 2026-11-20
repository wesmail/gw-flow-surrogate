#!/usr/bin/env python
"""When along the signal does the error build up?  Phase and amplitude error vs time.

For every selected test waveform (default: the 4000 headline rows, idx >= 1000)
the model waveform is regenerated EXACTLY as in the evaluation (same batches, same
seeds, theta-only; settings read from the eval CSV's .json sidecar), and three
error curves are computed against the true waveform:

  (a) accumulated phase error  dphi(t) = phi_model - phi_true, no alignment
      (both start from the same true first patch, so dphi starts at 0)
  (b) phase error after the best constant time + phase shift (the alignment the
      phase+time-optimised mismatch uses)
  (c) relative amplitude error A_model / A_true - 1 after the same shift

Each panel shows the median over waveforms, the 10-90 % band, and the median of
the strongly aligned-spin group (chi_eff > 0.4). Summary curves are written to
paper_figs/out/<out>.csv; key numbers are printed for the text.

Usage (repo root)::

    python paper_figs/fig_dephasing.py --checkpoint "$FT" \
        --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
        --eval-csv results/headline_K12_M1.csv --out fm_dephasing
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--rows", choices=["headline", "all"], default="headline")
    ap.add_argument("--max-waveforms", type=int, default=None, help="limit (for a quick look)")
    ap.add_argument("--stride", type=int, default=4, help="keep every k-th sample for the curves")
    ap.add_argument("--t-max", type=float, default=50.0, help="cut the ringdown tail after this time [M]")
    ap.add_argument("--chi-hi", type=float, default=0.4)
    ap.add_argument("--amp-ylim", type=float, default=0.03, help="panel (c) y-range +-this (ringdown spike goes off-scale)")
    ap.add_argument("--amp-mode", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="fm_dephasing")
    ap.add_argument("--talk", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN edge columns after the shift

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
    n_steps = int(meta.get("n_steps", 12)); mean_of = int(meta.get("mean_of", 1))
    seed = int(meta.get("seed", 1234)); bs = int(meta.get("batch_size", 64))
    temp = float(meta.get("temperature", 1.0))
    amp_mode = a.amp_mode or meta.get("amp_mode", "pchip")

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev).eval().to(dev)
    P, rscale = int(lit.hparams.patch_len), float(lit.hparams.residual_scale)
    ds = WaveformPatchDataset(a.manifest, split=a.split, patch_len=P, context_patches=1,
                              norm_stats_path=a.norm_stats, preload=False)
    N = n_patches_for(ds.n_time, P); usable = N * P
    n_eval = min(int(meta.get("n_waveforms", len(ds))), len(ds))
    row0 = 1000 if a.rows == "headline" else 0
    row1 = n_eval if a.max_waveforms is None else min(n_eval, row0 + a.max_waveforms)
    if row1 <= row0:
        raise SystemExit("no rows to process")
    import h5py
    man = pd.read_parquet(a.manifest)
    shard = Path(man["file_path"].iloc[0])
    if not shard.is_file():
        shard = Path(a.manifest).parent / shard.name
    with h5py.File(shard, "r") as f:
        t = np.asarray(f["t"][:], dtype=np.float64)[:usable]
    keep = np.arange(0, usable, a.stride)
    keep = keep[t[keep] <= a.t_max]
    tk = t[keep]
    print(f"rows {row0}..{row1 - 1}; n_steps={n_steps} mean_of={mean_of} seed={seed} batch={bs} "
          f"amp={amp_mode}; {keep.size} time points (every {a.stride} samples, t <= {a.t_max} M)")

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

    raw_all, al_all, amp_all, ce_all, mm_all, mm_csv = [], [], [], [], [], []
    nfft = 1 << int(np.ceil(np.log2(2 * usable)))
    with torch.no_grad():
        for bi in range(row0 // bs, (row1 - 1) // bs + 1):
            lo, hi = bi * bs, min((bi + 1) * bs, n_eval)
            batch = next(iter(DataLoader(Subset(ds, range(lo, hi)), batch_size=bs,
                                         collate_fn=_collate_batch, shuffle=False)))
            b = lit._patchify_batch({k: v.to(dev) for k, v in batch.items()})
            feat = b["token_features"]
            torch.manual_seed(seed + bi)                     # identical to the evaluation
            gen = lit.model.generate(feat[:, :1], b["theta"], N - 1, n_steps=n_steps, n_samples=1,
                                     temperature=temp, mean_of=mean_of)[:, 0]
            hp = reconstruct(torch.cat([feat[:, :1], gen], dim=1), b)
            la_o = b["log_a_track"][:, 0, :usable].double(); ph_o = b["phi_track"][:, 0, :usable].double()
            ho = torch.exp(la_o) * torch.exp(-1j * (ph_o - ph_o[:, :1]))
            # best time + phase shift for every row at once (FFT cross-correlation)
            xc = torch.fft.ifft(torch.fft.fft(ho, nfft).conj() * torch.fft.fft(hp, nfft))
            k = xc.abs().argmax(dim=1)
            peak = xc.gather(1, k[:, None])[:, 0]
            den = torch.linalg.vector_norm(ho, dim=1) * torch.linalg.vector_norm(hp, dim=1)
            mm = (1 - peak.abs() / den).cpu().numpy()
            shift = torch.where(k > nfft // 2, k - nfft, k).cpu().numpy()
            rot = torch.exp(-1j * peak.angle()).cpu().numpy()
            hp_n, ho_n = hp.cpu().numpy(), ho.cpu().numpy()
            th = b["theta"].double().cpu().numpy()
            for j in range(hi - lo):
                idx = lo + j
                if idx < row0 or idx >= row1:
                    continue
                # (a) accumulated phase error, no alignment: phi_model - phi_true
                raw = -np.unwrap(np.angle(hp_n[j] * np.conj(ho_n[j])))
                # (b, c) after the best shift: out[i] = hp[i + s], no wrap-around
                s = int(shift[j]); ha = np.full(usable, np.nan + 0j)
                if s >= 0:
                    ha[:usable - s] = hp_n[j, s:]
                else:
                    ha[-s:] = hp_n[j, :usable + s]
                ha *= rot[j]
                ok = ~np.isnan(ha)
                al = np.full(usable, np.nan)
                al[ok] = -np.unwrap(np.angle(ha[ok] * np.conj(ho_n[j, ok])))
                amp = np.abs(ha) / np.abs(ho_n[j]) - 1
                raw_all.append(raw[keep]); al_all.append(al[keep]); amp_all.append(amp[keep])
                q, c1, c2 = th[j, :3]
                ce_all.append((q * c1 + c2) / (1 + q)); mm_all.append(mm[j]); mm_csv.append(csv_mm.get(idx, np.nan))
            print(f"  {min(hi, row1) - row0}/{row1 - row0}")

    raw, al, amp = np.array(raw_all), np.array(al_all), np.array(amp_all)
    ce, mm_all, mm_csv = np.array(ce_all), np.array(mm_all), np.array(mm_csv)
    rel = np.nanmax(np.abs(mm_all / mm_csv - 1)) if np.isfinite(mm_csv).any() else np.nan
    print(f"\ncheck vs eval CSV: largest relative mismatch difference {rel:.1e} "
          f"(should be ~0; a few % at most from GPU nondeterminism)")
    hi_g = ce > a.chi_hi

    def bands(x):
        return (np.nanpercentile(x, 10, axis=0), np.nanpercentile(x, 50, axis=0),
                np.nanpercentile(x, 90, axis=0), np.nanpercentile(x[hi_g], 50, axis=0))

    curves = {}
    for name, x in (("phase_raw", raw), ("phase_aligned", al), ("amp_aligned", amp)):
        p10, p50, p90, phi = bands(x)
        curves.update({f"{name}_p10": p10, f"{name}_p50": p50, f"{name}_p90": p90,
                       f"{name}_p50_chi_hi": phi})

    import matplotlib.pyplot as plt
    apply_style(talk=a.talk)
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    spec = (("phase_raw", r"$\delta\varphi(t)$ [rad]", "(a) accumulated phase error\n(no alignment)"),
            ("phase_aligned", r"$\delta\varphi(t)$ [rad]", "(b) phase error after the\nbest time + phase shift"),
            ("amp_aligned", r"$A_\mathrm{model}/A_\mathrm{true}-1$", "(c) amplitude error after\nthe best shift"))
    for axi, (name, ylab, title) in zip(ax, spec):
        axi.fill_between(tk, curves[f"{name}_p10"], curves[f"{name}_p90"], color=C["band"], alpha=0.8,
                         lw=0, label="10–90 % of waveforms")
        axi.plot(tk, curves[f"{name}_p50"], color=C["fm"], lw=1.6, label="median, all")
        axi.plot(tk, curves[f"{name}_p50_chi_hi"], color=C["accent"], lw=1.3, ls="--",
                 label=rf"median, $\chi_\mathrm{{eff}}>{a.chi_hi}$")
        axi.axhline(0, color="k", lw=0.6)
        axi.axvline(0, color=C["gray"], lw=0.7, ls=":")
        axi.set_xlabel(r"$t\,[M]$  (merger at 0)"); axi.set_ylabel(ylab); axi.set_title(title)
        axi.set_xlim(tk[0], tk[-1])
    ax[2].set_ylim(-a.amp_ylim, a.amp_ylim)
    ax[2].annotate("merger:\noff scale", xy=(0, -0.95 * a.amp_ylim), xytext=(-1500, -0.75 * a.amp_ylim),
                   fontsize="x-small", color=C["gray"], arrowprops=dict(arrowstyle="->", color=C["gray"], lw=0.7))
    ax[0].legend(loc="lower left", fontsize="x-small", frameon=False)
    save(fig, a.out)

    out_csv = Path(_PKG / "out" / f"{a.out}.csv")
    pd.DataFrame(dict(t_M=tk, **curves)).to_csv(out_csv, index=False)
    print(f"curves -> {out_csv}")

    # ---- numbers for the text ---------------------------------------------
    def at(x, t0):
        return np.nanmedian(np.abs(x[:, np.argmin(np.abs(tk - t0))]))
    print(f"\n{len(ce)} waveforms ({hi_g.sum()} with chi_eff > {a.chi_hi})")
    for t0 in (-4000, -2500, -1000, -200, 0):
        print(f"  t={t0:6d} M: median |accumulated dphi| {at(raw, t0):.3f} rad "
              f"(chi_eff>{a.chi_hi}: {at(raw[hi_g], t0):.3f}) | after shift {at(al, t0):.3f} rad "
              f"| median |amp error| {100 * at(amp, t0):.2f}%")
    print(f"  median over waveforms of max |dphi| after the shift: {np.nanmedian(np.nanmax(np.abs(al), axis=1)):.3f} rad")
    print(f"  median signed accumulated dphi at merger: {np.nanmedian(raw[:, np.argmin(np.abs(tk))]):+.3f} rad "
          f"(>0: model runs ahead)")


if __name__ == "__main__":
    main()