#!/usr/bin/env python
"""Is the phase drift caused by a SYSTEMATIC bias in the per-patch phase advance?

The phase of the waveform is rebuilt by ADDING UP the predicted phase advance
of every patch (dphi). An error in dphi therefore accumulates:

  * random errors (sign changes from patch to patch) grow like  sqrt(N) * std
  * a systematic bias (same sign every patch)       grows like  N * bias

With N ~ 320 patches the bias term wins as soon as  |bias| > std / sqrt(N),
i.e. a bias of only ~6% of the random error is already as harmful.

This script measures, TEACHER-FORCED (every patch predicted from the TRUE
history, so feedback effects are excluded), for each test waveform:
  bias_rad      mean SIGNED dphi error per patch            [rad]
  mae_rad       mean |dphi error| per patch                 [rad]
  std_rad       spread of the signed error around its mean  [rad]
  drift_bias    N_eval * bias        (expected drift from the bias alone)
  drift_random  sqrt(N_eval) * std   (expected drift from random errors alone)
  drift_tf_peak accumulated signed error up to the amplitude peak
and the mean signed error per patch position (where in time the bias sits).

Usage (repo root)::

    python scripts/measure_dphi_bias.py \\
        --checkpoint logs/scale_study/n1000k_2p5m/version_2/checkpoints/last.ckpt \\
        --manifest data/n1000k/manifest.parquet \\
        --norm-stats data/n1000k/norm_stats.npz \\
        --n-waveforms 200 --n-steps 12 --mean-of 1 \\
        --eval-csv results/anneal_last.csv \\
        --out results/dphi_bias_2p5m.csv

--eval-csv (optional): a fig_context_fraction.py CSV of the same checkpoint and
rows; its rollout dphi_peak_rad is compared with the teacher-forced drift.

Reading the result (printed at the end):
  bias_share = drift_bias / drift_random  (median over waveforms)
    >> 1  -> the drift is bias-driven: reweighting dphi in the loss should help
    ~  1  -> both matter
    << 1  -> drift is random-walk-like: look elsewhere (feedback, head noise)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from datasets.datamodule import _collate_batch  # noqa: E402
from datasets.patchify import n_patches_for  # noqa: E402
from datasets.waveform_dataset import WaveformPatchDataset  # noqa: E402
from models.lightning_module import GWFlowSurrogateLit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--norm-stats", required=True, help="the norm_stats.npz the model was trained with")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-waveforms", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--mean-of", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-csv", default=None)
    ap.add_argument("--out", default="results/dphi_bias.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if not Path(a.norm_stats).is_file():
        raise FileNotFoundError(a.norm_stats)
    dev = torch.device(a.device)
    try:
        lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev)
    except RuntimeError:
        lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev,
                                                      use_abs_time=False)
    lit.eval().to(dev)
    P = int(lit.hparams.patch_len)
    D = lit.model.feat_dim

    ds = WaveformPatchDataset(a.manifest, split=a.split, patch_len=P, context_patches=1,
                              norm_stats_path=a.norm_stats, preload=True)
    st = ds.norm_stats
    if np.allclose(st["log_a_mean"], 0) and np.allclose(st["log_a_std"], 1):
        raise RuntimeError("identity norm stats loaded; check --norm-stats")
    N = n_patches_for(ds.n_time, P)
    n = min(a.n_waveforms, len(ds))
    loader = DataLoader(Subset(ds, range(n)), batch_size=a.batch_size,
                        collate_fn=_collate_batch, shuffle=False)
    print(f"{n} waveforms, {N} patches; teacher-forced, n_steps={a.n_steps}, "
          f"mean_of={a.mean_of}, T={a.temperature}")

    rows, err_all, la_err_all = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            b = lit._patchify_batch({k: v.to(dev) for k, v in batch.items()})
            feat, theta = b["token_features"], b["theta"]
            B = feat.shape[0]
            # predict token j+1 from the TRUE tokens 0..j, for all j at once
            z, _ = lit.model.encode(feat, theta)
            cond = (z[:, :N - 1] + lit.model.horizon_emb.weight[0]).reshape(-1, z.shape[-1])
            torch.manual_seed(a.seed + bi)
            rep = cond.repeat_interleave(a.mean_of, dim=0)
            s = lit.model.head.sample(rep, D, a.n_steps, a.temperature)
            pred = s.view(-1, a.mean_of, D).mean(dim=1).view(B, N - 1, D).double()
            true = feat[:, 1:N].double()
            dp_s = b["norm_dphi_std"][:, :1].double()
            la_s = b["norm_log_a_std"][:, :1].double()
            # channel 0: feature 0 = log_a (patch mean), feature 1 = dphi
            err = ((pred[..., 1] - true[..., 1]) * dp_s).cpu().numpy()      # (B, N-1) rad
            la_err = ((pred[..., 0] - true[..., 0]) * la_s).cpu().numpy()   # (B, N-1)
            err_all.append(err); la_err_all.append(la_err)
            peak_patch = (b["log_a_track"][:, 0, :N * P].argmax(dim=1) // P).cpu().numpy()
            th = theta.double().cpu().numpy()
            for i in range(B):
                e = err[i]
                bias, std = e.mean(), e.std()
                k = int(max(1, min(peak_patch[i], N - 1)))
                q, c1, c2 = th[i, :3]
                rows.append(dict(
                    idx=bi * a.batch_size + i, q=q, chi1z=c1, chi2z=c2,
                    chi_eff=(q * c1 + c2) / (1 + q),
                    bias_rad=bias, mae_rad=np.abs(e).mean(), std_rad=std,
                    drift_bias=abs(bias) * (N - 1),
                    drift_random=std * np.sqrt(N - 1),
                    drift_tf_peak=e[:k].sum(),
                    log_a_bias=la_err[i].mean(), log_a_mae=np.abs(la_err[i]).mean(),
                ))
            print(f"  {min((bi + 1) * a.batch_size, n)}/{n}")

    d = pd.DataFrame(rows)
    d["bias_share"] = d.drift_bias / d.drift_random
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    E = np.concatenate(err_all)
    per_patch = pd.DataFrame(dict(patch=np.arange(1, N), mean_signed_err=E.mean(0),
                                  mae=np.abs(E).mean(0), std=E.std(0)))
    per_patch.to_csv(out.with_name(out.stem + "_per_patch.csv"), index=False)

    print("\n=== per-patch dphi error, teacher-forced (median over waveforms) ===")
    print(f"  |bias| per patch      : {d.bias_rad.abs().median():.2e} rad")
    print(f"  mean |error| per patch: {d.mae_rad.median():.2e} rad")
    print(f"  random spread (std)   : {d.std_rad.median():.2e} rad")
    print(f"  expected drift from bias   ({N-1} x |bias|)     : {d.drift_bias.median():.3f} rad")
    print(f"  expected drift from random (sqrt({N-1}) x std) : {d.drift_random.median():.3f} rad")
    print(f"  bias_share = bias drift / random drift (median): {d.bias_share.median():.2f}"
          f"   (p10 {d.bias_share.quantile(.1):.2f}, p90 {d.bias_share.quantile(.9):.2f})")
    print(f"  share of waveforms where bias dominates (>1)   : {(d.bias_share > 1).mean()*100:.0f}%")
    print(f"  sign of bias: {(d.bias_rad > 0).mean()*100:.0f}% positive (model runs ahead)")
    print(f"  log-amplitude: |bias| {d.log_a_bias.abs().median():.2e}, mae {d.log_a_mae.median():.2e}")
    d["q_bin"] = pd.cut(d.q, [1, 2, 4, 6, 8], include_lowest=True)
    print("\nmedian signed bias [rad/patch] and bias_share by q:")
    print(d.groupby("q_bin", observed=True)[["bias_rad", "bias_share"]].median().to_string(
        float_format=lambda x: f"{x:.2e}"))
    third = (N - 1) // 3
    print("\nwhere the bias sits (mean signed error, rad/patch): "
          f"early {E[:, :third].mean():.2e} | middle {E[:, third:2*third].mean():.2e} | "
          f"late {E[:, 2*third:].mean():.2e}")

    if a.eval_csv:
        r = pd.read_csv(a.eval_csv)
        r = r[r.context_fraction == r.context_fraction.min()] if "context_fraction" in r else r
        m = d.merge(r[["idx", "dphi_peak_rad", "mm_pt_full"]], on="idx")
        if len(m) > 10:
            c = np.corrcoef(m.drift_tf_peak, m.dphi_peak_rad)[0, 1]
            cm = np.corrcoef(np.log(m.bias_rad.abs() + 1e-12), np.log(m.mm_pt_full))[0, 1]
            print(f"\nvs rollout ({a.eval_csv}, {len(m)} waveforms):")
            print(f"  corr(teacher-forced drift to peak, rollout dphi at peak) = {c:.2f}")
            print(f"  corr(log |bias|, log full-waveform mismatch)             = {cm:.2f}")
            print(f"  median |TF drift to peak| {m.drift_tf_peak.abs().median():.3f} rad vs "
                  f"median |rollout dphi at peak| {m.dphi_peak_rad.abs().median():.3f} rad")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        x = per_patch.patch
        ax[0].plot(x, per_patch.mean_signed_err, color="#D55E00", lw=1.4, label="mean signed error (bias)")
        ax[0].fill_between(x, -per_patch.mae, per_patch.mae, color="#B7D7EC", alpha=0.7, label="± mean |error|")
        ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xlabel("patch index (merger near the end)")
        ax[0].set_ylabel("dphi error [rad]"); ax[0].set_title("(a) where in time the bias sits"); ax[0].legend(fontsize=7)
        ax[1].hist(d.bias_share, bins=40, color="#0072B2"); ax[1].axvline(1, color="k", ls="--")
        ax[1].set_xscale("symlog", linthresh=0.1); ax[1].set_xlabel("bias drift / random drift")
        ax[1].set_title("(b) is the drift bias-driven? (>1 = yes)")
        ax[2].scatter(d.q, d.bias_rad, s=6, c=d.chi_eff, cmap="coolwarm")
        ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xlabel("q"); ax[2].set_ylabel("bias [rad/patch]")
        ax[2].set_title("(c) bias vs q (colour = chi_eff)")
        fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=150)
        print(f"\nfigure -> {out.with_suffix('.png')}")
    except Exception as exc:  # plotting is optional
        print("plot skipped:", exc)
    print(f"CSV -> {out}")


if __name__ == "__main__":
    main()