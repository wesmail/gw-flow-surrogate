#!/usr/bin/env python
"""Model waveform vs true waveform: whole signal, early inspiral, merger zoom.

Two rows by default: the test waveform closest to the MEDIAN mismatch (a typical
case) and the one closest to the 99th PERCENTILE (a hard case), chosen from an
eval CSV of fig_context_fraction.py. The model waveform is regenerated EXACTLY as
in that eval (same batch, same seed, theta-only), and the script checks that the
mismatch it gets agrees with the CSV.

The model waveform is shown after the best constant time and phase shift, i.e.
the alignment the phase+time-optimised mismatch uses.

Usage (repo root)::

    python paper_figs/fig_waveform.py --checkpoint "$FT" \
        --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
        --eval-csv results/headline_K12_M1.csv --out fm_waveform

    # or choose rows yourself:  --idx 1234 4321
The sampler settings (n_steps, mean_of, seed, batch size, amp mode) are read from
the eval CSV's .json sidecar, so they always match the evaluation.
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


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def shift_no_wrap(h: np.ndarray, shift: int) -> np.ndarray:
    """out[i] = h[i + shift]; samples shifted in from outside the signal are NaN
    (not plotted). np.roll would wrap them around and draw a fake spike."""
    out = np.full(h.shape, np.nan + 0j, dtype=complex)
    L = h.size
    if shift >= 0:
        out[:L - shift] = h[shift:]
    else:
        out[-shift:] = h[:L + shift]
    return out


def pick_rows(csv: pd.DataFrame, col: str, picks: list[str]) -> list[tuple[int, str]]:
    d = csv[csv.idx >= 1000] if (csv.idx >= 1000).any() else csv     # headline rows if present
    out = []
    for p in picks:
        qv = {"median": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}[p]
        target = d[col].quantile(qv)
        r = d.iloc[(d[col] - target).abs().argmin()]
        out.append((int(r.idx), {"median": "typical (median)", "p90": "90th percentile",
                                 "p95": "95th percentile", "p99": "hard (99th percentile)"}[p]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--eval-csv", required=True, help="fig_context_fraction.py eval CSV (with .json sidecar)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="explicit test rows (default: median + p99)")
    ap.add_argument("--pick", nargs="+", default=["median", "p99"], choices=["median", "p90", "p95", "p99"])
    ap.add_argument("--early", type=float, nargs=2, default=[-4600.0, -4300.0], help="inspiral zoom [M]")
    ap.add_argument("--merger", type=float, nargs=2, default=[-200.0, 80.0], help="merger zoom [M]")
    ap.add_argument("--out", default="fm_waveform")
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp-mode", default=None, help="override the amplitude mode of the eval json")
    ap.add_argument("--talk", action="store_true")
    a = ap.parse_args()

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
    meta_p = Path(a.eval_csv).with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    n_steps = int(meta.get("n_steps", 12)); mean_of = int(meta.get("mean_of", 1))
    seed = int(meta.get("seed", 1234)); bs = int(meta.get("batch_size", 64))
    temp = float(meta.get("temperature", 1.0))
    amp_mode = a.amp_mode or meta.get("amp_mode", "pchip")
    print(f"settings from {meta_p.name if meta else 'defaults'}: n_steps={n_steps} mean_of={mean_of} "
          f"seed={seed} batch={bs} T={temp} amp={amp_mode}")

    rows = [(i, f"test waveform {i}") for i in a.idx] if a.idx else pick_rows(csv, "mm_pt_full", a.pick)

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev).eval().to(dev)
    P, rscale = int(lit.hparams.patch_len), float(lit.hparams.residual_scale)
    ds = WaveformPatchDataset(a.manifest, split=a.split, patch_len=P, context_patches=1,
                              norm_stats_path=a.norm_stats, preload=False)
    N = n_patches_for(ds.n_time, P); usable = N * P
    import h5py
    man = pd.read_parquet(a.manifest)
    shard = Path(man["file_path"].iloc[0])
    if not shard.is_file():
        shard = Path(a.manifest).parent / shard.name
    with h5py.File(shard, "r") as f:
        t = np.asarray(f["t"][:], dtype=np.float64)[:usable]
    dt = float(t[1] - t[0])
    s_frac = torch.arange(P, device=dev, dtype=torch.float64) / P

    def reconstruct(feat, b):
        feat = feat.double()
        la_m = b["norm_log_a_mean"][:, :1].double(); la_s = b["norm_log_a_std"][:, :1].double()
        dp_m = b["norm_dphi_mean"][:, :1].double(); dp_s = b["norm_dphi_std"][:, :1].double()
        la = log_amp_per_sample(feat[..., 0] * la_s + la_m, P, amp_mode)          # (B, n*P)
        dphi = feat[..., 1] * dp_s + dp_m
        res = feat[..., 2:2 + P] * rscale
        bnd = torch.cumsum(dphi, dim=1) - dphi
        phase = (bnd[..., None] + s_frac * dphi[..., None] + res).reshape(feat.shape[0], -1)
        return torch.exp(la) * torch.exp(-1j * phase)

    results = []
    with torch.no_grad():
        for idx, label in rows:
            bi = idx // bs                                   # rebuild the eval batch exactly
            n_eval = int(meta.get("n_waveforms", len(ds)))   # the eval's last batch may be short
            sub = Subset(ds, range(bi * bs, min((bi + 1) * bs, n_eval, len(ds))))
            batch = next(iter(DataLoader(sub, batch_size=bs, collate_fn=_collate_batch, shuffle=False)))
            b = lit._patchify_batch({k: v.to(dev) for k, v in batch.items()})
            feat = b["token_features"]
            torch.manual_seed(seed + bi)
            gen = lit.model.generate(feat[:, :1], b["theta"], N - 1, n_steps=n_steps, n_samples=1,
                                     temperature=temp, mean_of=mean_of)[:, 0]
            j = idx - bi * bs
            hp = reconstruct(torch.cat([feat[:, :1], gen], dim=1), b)[j].cpu().numpy()
            la_o = b["log_a_track"][j, 0, :usable].double().cpu().numpy()
            ph_o = b["phi_track"][j, 0, :usable].double().cpu().numpy()
            ho = np.exp(la_o) * np.exp(-1j * (ph_o - ph_o[0]))
            # best constant time + phase shift (same as the phase+time-optimised mismatch)
            L = ho.size; nfft = 1 << int(np.ceil(np.log2(2 * L)))
            xc = np.fft.ifft(np.conj(np.fft.fft(ho, nfft)) * np.fft.fft(hp, nfft))
            k = int(np.abs(xc).argmax()); shift = k - nfft if k > nfft // 2 else k
            den = np.linalg.norm(ho) * np.linalg.norm(hp)
            mm = 1 - np.abs(xc[k]) / den
            ha = shift_no_wrap(hp, shift) * np.exp(-1j * np.angle(xc[k]))
            th = b["theta"][j].double().cpu().numpy()
            q, c1, c2 = th[:3]
            ce = (q * c1 + c2) / (1 + q)
            ref = csv.loc[csv.idx == idx, "mm_pt_full"]
            print(f"idx {idx} ({label}): q={q:.2f} chi1={c1:+.2f} chi2={c2:+.2f} chi_eff={ce:+.2f}  "
                  f"mismatch {mm:.3e} (CSV {ref.iloc[0]:.3e})  time shift {shift * dt:+.2f} M"
                  if len(ref) else f"idx {idx}: mismatch {mm:.3e}")
            results.append(dict(idx=idx, label=label, ho=ho, ha=ha, mm=mm, q=q, ce=ce))

    import matplotlib.pyplot as plt
    apply_style(talk=a.talk)
    nr = len(results)
    # Paper-readable: full column shows |h| (Re(h) over ~10k cycles is a smear);
    # zoom columns use thick black NR under dashed FM so both remain visible when
    # they nearly coincide.
    fig, ax = plt.subplots(
        nr, 3, figsize=(11.0, 3.2 * nr), squeeze=False,
        gridspec_kw=dict(width_ratios=[2.15, 1.0, 1.0], wspace=0.35, hspace=0.45),
    )
    for r, res in enumerate(results):
        ho, ha = res["ho"], res["ha"]
        sc = float(np.abs(ho).max())
        # --- full: amplitude envelopes ---
        axi = ax[r, 0]
        axi.plot(t, np.abs(ho) / sc, color="k", lw=1.6, label="NRHybSur3dq8", zorder=2)
        axi.plot(t, np.abs(ha) / sc, color=C["fm"], lw=1.4, ls="--",
                 label="flow-matching surrogate", zorder=3)
        axi.set_xlim(t[0], t[-1])
        axi.set_ylim(-0.02, 1.12)
        axi.set_ylabel(r"$|h_{22}|$ (normalised)")
        axi.set_title(
            f"{res['label']}\n"
            f"$q={res['q']:.1f}$, $\\chi_\\mathrm{{eff}}={res['ce']:+.2f}$, "
            f"mismatch ${sci(res['mm'])}$",
            fontsize=10, pad=8,
        )
        # --- zooms: Re(h) with clear overlay ---
        for c, (lo, hi), title in (
            (1, tuple(a.early), "early inspiral"),
            (2, tuple(a.merger), "merger"),
        ):
            axi = ax[r, c]
            m = (t >= lo) & (t <= hi)
            axi.plot(t[m], np.abs(ho[m]) / sc, color=C["gray"], lw=0.7, zorder=1)
            axi.plot(t[m], -np.abs(ho[m]) / sc, color=C["gray"], lw=0.7, zorder=1)
            axi.plot(t[m], ho.real[m] / sc, color="k", lw=1.5, solid_capstyle="round",
                     label="NRHybSur3dq8", zorder=2)
            axi.plot(t[m], ha.real[m] / sc, color=C["fm"], lw=1.35, ls=(0, (3.2, 1.6)),
                     solid_capstyle="round", label="flow-matching surrogate", zorder=3)
            axi.set_xlim(lo, hi)
            sc_local = float(np.abs(ho[m]).max()) / sc
            axi.set_ylim(-1.2 * sc_local, 1.2 * sc_local)
            if r == 0:
                axi.set_title(title, fontsize=11, pad=8)
            if r == 0 and c == 1:
                axi.set_ylabel(r"$\mathrm{Re}\,h_{22}$ (normalised)")
        if r == nr - 1:
            for c in range(3):
                ax[r, c].set_xlabel(r"$t\,[M]$")
    ax[0, 0].legend(loc="upper left", fontsize=9, frameon=False, handlelength=2.4)
    fig.tight_layout()
    save(fig, a.out)


if __name__ == "__main__":
    main()