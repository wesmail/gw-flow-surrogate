#!/usr/bin/env python
"""Representation-floor test: patch-constant vs interpolated amplitude.

No model, no checkpoint, no norm stats, no change to the repo code.

For each test waveform:
  1. take the ORIGINAL tracks log_a(t), phi(t) from the HDF5 shards,
  2. patchify them with the repo's own datasets.patchify (the true tokens),
  3. rebuild h(t) = A(t) exp(-i phi(t)) from those tokens with three
     amplitude rules (phase is rebuilt the same, exact way in all three):
       constant : one value per patch (what _reconstruct_waveform does today)
       linear   : log_a linearly interpolated between patch centres
       pchip    : monotone cubic (PCHIP) through the patch centres
  4. mismatch of each rebuild against the original, on three regions:
       full (whole waveform), half (last 50%), window (last 25%: merger).

The token is the MEAN of log_a over samples [kP, kP+P-1], so its centre sits
at sample kP + (P-1)/2. Outside the first/last centre the amplitude is held
constant (no extrapolation).

Usage (repo root)::

    python scripts/test_amp_floor.py --manifest data/n1000k/manifest.parquet \\
        --split test --n-waveforms 500 --out results/amp_floor.csv

Prints a table of median / p90 / max floor per rule and region, writes the
per-waveform CSV and a figure paper_figs/out/amp_floor_test.{png,pdf}:
(a) floor CDF on the merger window per rule, (b) amplitude near merger for
the median waveform.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "paper_figs"))
from datasets.patchify import patchify_tracks_batched  # noqa: E402

RULES = ("constant", "linear", "pchip")


def load_tracks(manifest: str, split: str, n: int):
    man = pd.read_parquet(manifest)
    rows = man[man["split"] == split].reset_index(drop=True).iloc[:n]
    if len(rows) == 0:
        raise ValueError(f"no rows with split={split!r} in {manifest}")
    base = Path(manifest).parent
    la_l, ph_l, th_l, t = [], [], [], None
    for shard, grp in rows.groupby("file_path", sort=False):
        p = Path(shard)
        if not p.is_file():
            p = base / p.name
        with h5py.File(p, "r") as f:
            idx = grp["row_index"].to_numpy()
            order = np.argsort(idx)                      # h5py needs increasing indices
            la = f["log_a"][idx[order]][np.argsort(order)]
            ph = f["phi"][idx[order]][np.argsort(order)]
            if t is None:
                t = np.asarray(f["t"][:], dtype=np.float64)
        la_l.append(la[:, 0, :]); ph_l.append(ph[:, 0, :])
        th_l.append(grp[["q", "chi1z", "chi2z"]].to_numpy())
    return (np.concatenate(la_l).astype(np.float64), np.concatenate(ph_l).astype(np.float64),
            np.concatenate(th_l), t)


def rebuild_amp(la_patch: np.ndarray, P: int, rule: str) -> np.ndarray:
    """la_patch (B, n) patch-mean log-amplitude -> log-amplitude per sample (B, n*P)."""
    B, n = la_patch.shape
    if rule == "constant" or n < 2:
        return np.repeat(la_patch, P, axis=1)
    x = np.arange(n * P, dtype=np.float64)              # sample index
    xc = np.arange(n) * P + (P - 1) / 2.0                # patch-mean centres
    xq = np.clip(x, xc[0], xc[-1])                       # hold flat at the edges
    if rule == "linear":
        return np.stack([np.interp(xq, xc, la_patch[b]) for b in range(B)])
    if rule == "pchip":
        return PchipInterpolator(xc, la_patch, axis=1)(xq)
    raise ValueError(rule)


def mismatch(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise (strict, phase+time-optimised) white-noise mismatch."""
    n1 = np.linalg.norm(h1, axis=1); n2 = np.linalg.norm(h2, axis=1)
    den = n1 * n2
    strict = 1 - np.real(np.sum(h1.conj() * h2, axis=1)) / den
    nfft = 1 << int(np.ceil(np.log2(2 * h1.shape[1])))
    xc = np.fft.ifft(np.fft.fft(h1, nfft, axis=1).conj() * np.fft.fft(h2, nfft, axis=1), axis=1)
    pt = 1 - np.abs(xc).max(axis=1) / den
    return strict, pt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-waveforms", type=int, default=500)
    ap.add_argument("--patch-len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", default="results/amp_floor.csv")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    P = a.patch_len

    la_all, ph_all, theta, t = load_tracks(a.manifest, a.split, a.n_waveforms)
    N = (la_all.shape[1] - 1) // P
    U = N * P
    regions = {"full": 0, "half": (N // 2) * P, "window": int(round(0.75 * N)) * P}
    print(f"{len(la_all)} waveforms, {N} patches x {P} samples; "
          f"window starts at t = {t[regions['window']]:.0f} M")

    rows, keep = [], {}
    for b0 in range(0, len(la_all), a.batch):
        la = la_all[b0:b0 + a.batch]; ph = ph_all[b0:b0 + a.batch]
        B = la.shape[0]
        # true tokens, with the repo's own patchify (fp32 inside, like training)
        p = patchify_tracks_batched(torch.from_numpy(la[:, None]), torch.from_numpy(ph[:, None]), P)
        la_p = p["log_a"][..., 0].double().numpy()                     # (B, N)
        dphi = p["delta_phi"][..., 0].double().numpy()                 # (B, N)
        res = p["phi_residual"][..., 0, :].double().numpy()            # (B, N, P)
        # exact phase rebuild (identical for all rules)
        s = np.arange(P) / P
        bnd = np.cumsum(dphi, axis=1) - dphi
        phase = (bnd[:, :, None] + s * dphi[:, :, None] + res).reshape(B, U)
        h_orig = np.exp(la[:, :U]) * np.exp(-1j * (ph[:, :U] - ph[:, :1]))
        out = {k: {} for k in RULES}
        for rule in RULES:
            la_t = rebuild_amp(la_p, P, rule)
            h = np.exp(la_t) * np.exp(-1j * phase)
            for reg, s0 in regions.items():
                out[rule][reg] = mismatch(h_orig[:, s0:], h[:, s0:])
            if b0 == 0:
                keep[rule] = la_t
        if b0 == 0:
            keep["orig"] = la[:, :U]
        for i in range(B):
            r = dict(idx=b0 + i, q=theta[b0 + i, 0], chi1z=theta[b0 + i, 1], chi2z=theta[b0 + i, 2])
            for rule in RULES:
                for reg in regions:
                    r[f"strict_{reg}_{rule}"] = out[rule][reg][0][i]
                    r[f"pt_{reg}_{rule}"] = out[rule][reg][1][i]
            rows.append(r)

    d = pd.DataFrame(rows)
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(outp, index=False)

    tab = []
    for reg in regions:
        for rule in RULES:
            v = d[f"pt_{reg}_{rule}"]
            tab.append(dict(region=reg, rule=rule, median=v.median(),
                            p90=v.quantile(0.9), max=v.max(),
                            gain_vs_constant=d[f"pt_{reg}_constant"].median() / v.median()))
    tab = pd.DataFrame(tab)
    with pd.option_context("display.float_format", "{:.3e}".format):
        print("\nphase+time-optimised floor (true tokens vs original waveform):")
        print(tab.to_string(index=False))
    tab.to_csv(outp.with_name(outp.stem + "_summary.csv"), index=False)
    print(f"\nCSV -> {outp}")

    if a.no_plot:
        return
    from style import C, apply_style, save
    apply_style()
    import matplotlib.pyplot as plt
    cols = {"constant": C["gray"], "linear": C["fm"], "pchip": C["accent"]}
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for rule in RULES:
        v = np.sort(d[f"pt_window_{rule}"].to_numpy())
        ax[0].step(v, np.arange(1, v.size + 1) / v.size, where="post", color=cols[rule], lw=1.6,
                   label=f"{rule}  (median {np.median(v):.1e})")
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("floor on merger window (last 25%)"); ax[0].set_ylabel("fraction of waveforms")
    ax[0].set_title("(a)", loc="left", fontweight="bold"); ax[0].legend(fontsize="x-small", frameon=False)
    # (b) amplitude around merger for the median waveform of the first batch
    nb = keep["orig"].shape[0]
    j = int(np.argsort(d["pt_window_constant"].to_numpy()[:nb])[nb // 2])
    tt = t[:U]; m = (tt > -150) & (tt < 80)
    ax[1].plot(tt[m], np.exp(keep["orig"][j][m]), color="k", lw=1.6, label="original")
    for rule in RULES:
        ax[1].plot(tt[m], np.exp(keep[rule][j][m]), color=cols[rule], lw=1.1,
                   ls="-" if rule != "constant" else "-", label=rule)
    ax[1].set_xlabel("t [M]"); ax[1].set_ylabel("amplitude A(t)")
    ax[1].set_title("(b)", loc="left", fontweight="bold"); ax[1].legend(fontsize="x-small", frameon=False)
    fig.tight_layout()
    save(fig, "amp_floor_test")


if __name__ == "__main__":
    main()