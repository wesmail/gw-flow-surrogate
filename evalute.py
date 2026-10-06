#!/usr/bin/env python
"""Evaluate a trained GWFlowSurrogateLit checkpoint on a waveform split.

Metrics
-------
* Free-rollout, phase-maximized mismatch per waveform (mean and best over S
  flow samples), plus aligned relative L2 error of the complex strain.
* Per-patch |Delta-phi| error in RADIANS, both teacher-forced (one-step, true
  history) and free-rollout — the TF-vs-rollout gap is THE diagnostic for
  increment-drift pathologies.
* Cumulative phase-drift curve |phi_pred - phi_true|(t) (global offset removed),
  median and 10-90% band over the evaluated set.
* Amplitude accuracy: median relative amplitude error per patch.

Context protocol (§5.6 / [ITEM 4])
----------------------------------
This script defaults to context_patches=1 (θ-only seed → generate the rest).
`eval_fm_unified.py` defaults to --context-fraction 0.5 (half seeded). These
are different tasks; do not quote mismatch numbers across the two protocols.

Figures (saved to --output-dir)
-------------------------------
  waveform_<i>.png      Re[h] overlay (full + merger zoom), log-amplitude, and
                        the accumulated phase ERROR pred-vs-true (not a signal
                        overlay), for the first --n-plot waveforms
  mismatch_hist.png     log-histogram of per-waveform mismatch
  mismatch_vs_q.png     mismatch vs mass ratio, colored by mean spin
  dphi_error.png        median |Delta-phi error| per patch index, TF vs rollout
  phase_drift.png       median + 10-90% band of accumulated phase error vs time

Outputs: metrics.json (summary), per_waveform.csv (one row per waveform).

Run (from the repo root)
------------------------
  python evalute.py \\
      --checkpoint logs/flow_10k_small/version_0/checkpoints/best.ckpt \\
      --manifest data/generated/manifest.parquet \\
      --split test --n-waveforms 128 --n-samples 4 --output-dir eval_test

The dataset's norm_stats.npz (next to the manifest) MUST be the one used in
training; the script refuses to run with identity stats unless --allow-identity-stats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.waveform_dataset import WaveformPatchDataset
from datasets.patchify import n_patches_for
from models.lightning_module import GWFlowSurrogateLit

_NORM_KEYS = ("norm_log_a_mean", "norm_log_a_std", "norm_dphi_mean", "norm_dphi_std")


def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
  batch = {
    "log_a_track": torch.stack([s["log_a_track"] for s in samples]),
    "phi_track": torch.stack([s["phi_track"] for s in samples]),
    "theta": torch.stack([s["theta"] for s in samples]),
    "ref_phase": torch.stack([s["ref_phase"] for s in samples]),
    "context_patches": torch.stack([s["context_patches"] for s in samples]),
    "target_start": torch.stack([s["target_start"] for s in samples]),
  }
  for k in _NORM_KEYS:
    batch[k] = torch.stack([s[k] for s in samples])
  return batch


def load_time_grid(manifest_path: Path, n_needed: int) -> np.ndarray:
  """Best-effort: read the geometric time grid from the first HDF5 shard."""
  try:
    import h5py
    import pandas as pd
    fp = pd.read_parquet(manifest_path)["file_path"].iloc[0]
    p = Path(fp)
    if not p.is_file():
      p = manifest_path.parent / p.name
    with h5py.File(p, "r") as f:
      t = np.asarray(f["t"][:], dtype=np.float64)
    if t.shape[0] >= n_needed:
      return t
  except Exception:
    pass
  return np.arange(n_needed, dtype=np.float64)


@torch.no_grad()
def evaluate_batch(lit: GWFlowSurrogateLit, batch: dict, n_samples: int,
                   n_steps: int, limit_patches: int, device: torch.device,
                   temperature: float = 1.0, mean_of: int = 1) -> dict:
  batch = {k: v.to(device) for k, v in batch.items()}
  batch = lit._patchify_batch(batch)
  P = lit.hparams.patch_len
  C = lit.hparams.n_channels
  feat = batch["token_features"]
  B, N, _ = feat.shape
  ctx = max(int(batch["context_patches"][0].item()), 1)
  n_gen = N - ctx
  if limit_patches > 0:
    n_gen = min(n_gen, limit_patches)

  la_m = batch["norm_log_a_mean"]; la_s = batch["norm_log_a_std"]
  dp_m = batch["norm_dphi_mean"]; dp_s = batch["norm_dphi_std"]
  dp_s0 = dp_s[:, 0] if dp_s.ndim == 2 else dp_s          # (B,) channel 0

  # ---- ground truth (physical) ------------------------------------------ #
  la_t = batch["log_a"][:, ctx:ctx + n_gen, 0]
  dp_t = batch["delta_phi"][:, ctx:ctx + n_gen, 0]
  res_t = batch["phi_residual"][:, ctx:ctx + n_gen, 0]
  h_true = lit._reconstruct_waveform(la_t, dp_t, res_t, la_m, la_s, dp_m, dp_s, P)

  # ---- free rollout ------------------------------------------------------ #
  gen = lit.model.generate(feat[:, :ctx], batch["theta"], n_gen,
                           n_steps=n_steps, n_samples=n_samples,
                           temperature=temperature, mean_of=mean_of)
  la_g, dp_g, res_g = lit._split_features(gen, C, P)      # (B, S, n_gen, C[, P])

  mism = torch.zeros(B, n_samples)
  rel_l2 = torch.zeros(B, n_samples)
  h_pred0 = None
  for s in range(n_samples):
    h_p = lit._reconstruct_waveform(la_g[:, s, :, 0], dp_g[:, s, :, 0],
                                    res_g[:, s, :, 0], la_m, la_s, dp_m, dp_s, P)
    if s == 0:
      h_pred0 = h_p
    inner = (h_p * h_true.conj()).sum(dim=1)
    num = inner.abs()
    den = torch.sqrt((h_p.abs() ** 2).sum(1) * (h_true.abs() ** 2).sum(1)).clamp_min(1e-30)
    mism[:, s] = (1.0 - num / den).cpu()
    # aligned relative L2: rotate pred by the overlap-maximizing global phase
    delta = torch.angle(inner)
    h_al = h_p * torch.exp(-1j * delta)[:, None]
    rel_l2[:, s] = ((h_al - h_true).abs().pow(2).sum(1)
                    / (h_true.abs().pow(2).sum(1)).clamp_min(1e-30)).cpu()

  # per-patch rollout errors, physical units (sample 0)
  dphi_err_ro = ((dp_g[:, 0, :, 0] - dp_t) * dp_s0[:, None]).abs().cpu()      # (B, n_gen) rad
  la_s0 = la_s[:, 0] if la_s.ndim == 2 else la_s
  amp_rel_err = (torch.exp((la_g[:, 0, :, 0] - la_t) * la_s0[:, None]) - 1).abs().cpu()

  # phase drift vs time (sample 0), global offset removed via first sample
  ang = np.unwrap(np.angle((h_pred0 * h_true.conj()).cpu().numpy()), axis=1)
  drift = np.abs(ang - ang[:, :1])                                            # (B, T_gen)

  # ---- teacher-forced one-step sampling ---------------------------------- #
  z, _ = lit.model.encode(feat, batch["theta"])
  cond = (z[:, ctx - 1:N - 1] + lit.model.horizon_emb.weight[0]).reshape(-1, z.shape[-1])
  # [ITEM 1] Teacher-forced sampling must use the SAME temperature/mean_of as the
  # rollout, otherwise tf_to_rollout_dphi_ratio compares averaged rollout draws
  # against single noisy TF draws and understates drift (§5.3).
  if mean_of > 1:
    rep = cond.repeat_interleave(mean_of, dim=0)
    s = lit.model.head.sample(rep, lit.model.feat_dim, n_steps, temperature)
    tf = s.view(-1, mean_of, lit.model.feat_dim).mean(dim=1).view(B, N - ctx, -1)
  else:
    tf = lit.model.head.sample(cond, lit.model.feat_dim, n_steps,
                               temperature).view(B, N - ctx, -1)
  _, dp_tf, _ = lit._split_features(tf, C, P)
  dphi_err_tf = ((dp_tf[:, :n_gen, 0] - dp_t) * dp_s0[:, None]).abs().cpu()

  return {
    "mismatch": mism, "rel_l2": rel_l2,
    "dphi_err_rollout": dphi_err_ro, "dphi_err_tf": dphi_err_tf,
    "amp_rel_err": amp_rel_err, "drift": drift,
    "theta": batch["theta"].cpu(),
    "h_true": h_true.cpu(), "h_pred": h_pred0.cpu(),
    "la_true": (la_t * la_s0[:, None] + (la_m[:, 0] if la_m.ndim == 2 else la_m)[:, None]).cpu(),
    "la_pred": (la_g[:, 0, :, 0] * la_s0[:, None] + (la_m[:, 0] if la_m.ndim == 2 else la_m)[:, None]).cpu(),
    "ctx": ctx, "n_gen": n_gen,
  }


def plot_waveform(i: int, r: dict, b: int, t: np.ndarray, P: int, out: Path) -> None:
  h_t = r["h_true"][b].numpy(); h_p = r["h_pred"][b].numpy()
  # align global phase for plotting
  delta = np.angle((h_p * np.conj(h_t)).sum())
  h_p = h_p * np.exp(-1j * delta)
  T = h_t.shape[0]
  t0 = r["ctx"] * P
  tt = t[t0:t0 + T] if t.shape[0] >= t0 + T else np.arange(T)

  fig, axes = plt.subplots(4, 1, figsize=(11, 11))
  axes[0].plot(tt, h_t.real, lw=0.6, label="true")
  axes[0].plot(tt, h_p.real, lw=0.6, ls="--", label="pred")
  axes[0].set_title(f"waveform {i}  |  Re[h], full rollout"); axes[0].legend()

  zoom = min(1500, T)
  pk = int(np.argmax(np.abs(h_t)))
  lo = max(0, pk - zoom // 2); hi = min(T, lo + zoom)
  axes[1].plot(tt[lo:hi], h_t.real[lo:hi], lw=0.9, label="true")
  axes[1].plot(tt[lo:hi], h_p.real[lo:hi], lw=0.9, ls="--", label="pred")
  axes[1].set_title("Re[h], merger zoom"); axes[1].legend()

  n_p = r["la_true"].shape[1]
  tp = tt[:n_p * P:P] if tt.shape[0] >= n_p * P else np.arange(n_p)
  axes[2].plot(tp, r["la_true"][b], lw=0.9, label="true")
  axes[2].plot(tp, r["la_pred"][b], lw=0.9, ls="--", label="pred")
  axes[2].set_title("patch-mean log-amplitude"); axes[2].legend()

  axes[3].plot(tt, r["drift"][b], lw=0.8, color="tab:blue",
               label="accumulated phase error |arg(h_pred h_true*)|")
  axes[3].axhline(1.0, color="r", ls=":", lw=0.8,
                  label="1 rad decoherence threshold")
  axes[3].set_title(
    "accumulated phase error pred vs true (rad, constant global offset removed; "
    "post-ringdown region is numerically unreliable)"
  )
  axes[3].set_xlabel("t"); axes[3].set_ylabel("phase error (rad)")
  axes[3].legend()
  fig.tight_layout()
  fig.savefig(out / f"waveform_{i:03d}.png", dpi=130)
  plt.close(fig)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--split", default="test", help="test | val | train | ood")
  ap.add_argument("--norm-stats", default=None,
                  help="default: <manifest dir>/norm_stats.npz")
  ap.add_argument("--n-waveforms", type=int, default=128)
  ap.add_argument("--n-samples", type=int, default=4, help="flow samples per waveform")
  ap.add_argument("--n-steps", type=int, default=None,
                  help="Euler steps (default: training value)")
  ap.add_argument("--limit-patches", type=int, default=0, help="cap rollout length (0=full)")
  # [ITEM 4] default None keeps every existing command line bit-identical
  # (context_patches=1). When set, seed max(1, int(frac * n_patches)) patches.
  ap.add_argument("--context-fraction", type=float, default=None,
                  help="fraction of patches to seed (default: None => context_patches=1)")
  ap.add_argument("--temperature", type=float, default=1.0,
                  help="flow initial-noise scale (0 = deterministic ODE)")
  ap.add_argument("--mean-of", type=int, default=1,
                  help="average K head samples per fed-back patch")
  ap.add_argument("--batch-size", type=int, default=16)
  ap.add_argument("--n-plot", type=int, default=4)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--output-dir", default="eval_out")
  ap.add_argument("--allow-identity-stats", action="store_true")
  args = ap.parse_args()

  device = torch.device(args.device)
  out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

  try:
    lit = GWFlowSurrogateLit.load_from_checkpoint(args.checkpoint, map_location=device)
  except RuntimeError:
    # Checkpoint predates the absolute-time embedding -> rebuild without it.
    print("note: checkpoint has no abs-time embedding, loading with use_abs_time=False")
    lit = GWFlowSurrogateLit.load_from_checkpoint(
      args.checkpoint, map_location=device, use_abs_time=False
    )
  lit.eval().to(device)
  n_steps = args.n_steps or lit.hparams.n_sampling_steps
  P = lit.hparams.patch_len

  manifest = Path(args.manifest)
  stats_path = args.norm_stats or str(manifest.parent / "norm_stats.npz")
  if not Path(stats_path).is_file() and not args.allow_identity_stats:
    raise FileNotFoundError(
      f"{stats_path} not found — evaluation with identity stats does not match "
      "training normalization. Pass --norm-stats or --allow-identity-stats."
    )
  ds_probe = WaveformPatchDataset(
    manifest, split=args.split, patch_len=P,
    context_patches=1, norm_stats_path=stats_path, preload=False,
  )
  n_patches = n_patches_for(ds_probe.n_time, P)
  # [ITEM 4] default None keeps every existing command line bit-identical
  # (θ-only seed). Opt-in fraction selects a longer context for forecasting.
  if args.context_fraction is None:
    ctx_patches = 1
    ctx_frac_reported = None
  else:
    ctx_patches = max(1, int(round(args.context_fraction * n_patches)))
    ctx_frac_reported = float(args.context_fraction)
  n_gen_patches = n_patches - ctx_patches
  # [ITEM 4] Announce the protocol so numbers are never quoted across tasks.
  print(f"context: {ctx_patches} of {n_patches} patches seeded "
        f"(fraction {ctx_frac_reported if ctx_frac_reported is not None else 'default/None'}) "
        f"— generating {n_gen_patches}")

  ds = WaveformPatchDataset(
    manifest, split=args.split, patch_len=P,
    context_patches=ctx_patches, norm_stats_path=stats_path, preload=True,
  )
  n_eval = min(args.n_waveforms, len(ds))
  t_grid = load_time_grid(manifest, ds.n_time)
  print(f"split={args.split}: evaluating {n_eval}/{len(ds)} waveforms, "
        f"S={args.n_samples}, {n_steps} Euler steps, device={device}")

  results, plotted = [], 0
  for start in range(0, n_eval, args.batch_size):
    idx = range(start, min(start + args.batch_size, n_eval))
    batch = collate([ds[i] for i in idx])
    r = evaluate_batch(lit, batch, args.n_samples, n_steps,
                       args.limit_patches, device,
                       temperature=args.temperature, mean_of=args.mean_of)
    results.append(r)
    for b in range(r["mismatch"].shape[0]):
      if plotted < args.n_plot:
        plot_waveform(start + b, r, b, t_grid, P, out)
        plotted += 1
    print(f"  [{min(start + args.batch_size, n_eval):4d}/{n_eval}] "
          f"batch mismatch mean={r['mismatch'].mean():.3e}")

  # ---- aggregate ---------------------------------------------------------- #
  mism = torch.cat([r["mismatch"] for r in results])          # (n, S)
  rel_l2 = torch.cat([r["rel_l2"] for r in results])
  theta = torch.cat([r["theta"] for r in results]).numpy()
  d_ro = torch.cat([r["dphi_err_rollout"] for r in results]).numpy()
  d_tf = torch.cat([r["dphi_err_tf"] for r in results]).numpy()
  amp = torch.cat([r["amp_rel_err"] for r in results]).numpy()
  drift = np.concatenate([r["drift"] for r in results], axis=0)

  mm_mean = mism.mean(dim=1).numpy()
  mm_best = mism.min(dim=1).values.numpy()
  q = lambda x, p: float(np.quantile(x, p))
  summary = {
    "split": args.split, "n_waveforms": int(n_eval),
    "n_samples": args.n_samples, "n_steps": int(n_steps),
    "temperature": args.temperature, "mean_of": args.mean_of,
    # [ITEM 4] Record protocol so metrics.json is self-describing across tasks.
    "context_patches": int(ctx_patches),
    "context_fraction": ctx_frac_reported,
    "n_generated_patches": int(n_gen_patches),
    "mismatch_mean": {"mean": float(mm_mean.mean()), "median": q(mm_mean, 0.5),
                      "p90": q(mm_mean, 0.9), "max": float(mm_mean.max())},
    "mismatch_best_of_S": {"median": q(mm_best, 0.5), "p90": q(mm_best, 0.9)},
    "rel_l2_median": q(rel_l2.mean(dim=1).numpy(), 0.5),
    "dphi_mae_rad_rollout": float(d_ro.mean()),
    "dphi_mae_rad_teacher_forced": float(d_tf.mean()),
    "tf_to_rollout_dphi_ratio": float(d_ro.mean() / max(d_tf.mean(), 1e-12)),
    "amp_rel_err_median": q(amp, 0.5),
    "final_phase_drift_rad_median": q(drift[:, -1], 0.5),
  }
  (out / "metrics.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2))

  header = "q,chi1z,chi2z,mismatch_mean,mismatch_best,rel_l2,final_drift_rad"
  rows = [f"{theta[i,0]:.4f},{theta[i,1]:.4f},{theta[i,2]:.4f},"
          f"{mm_mean[i]:.6e},{mm_best[i]:.6e},"
          f"{rel_l2[i].mean():.6e},{drift[i,-1]:.4f}" for i in range(n_eval)]
  (out / "per_waveform.csv").write_text(header + "\n" + "\n".join(rows) + "\n")

  # ---- summary figures ----------------------------------------------------- #
  fig, ax = plt.subplots(figsize=(6, 4))
  bins = np.logspace(np.log10(max(mm_mean.min(), 1e-7)),
                     np.log10(max(mm_mean.max(), 1e-6)), 30)
  ax.hist(mm_mean, bins=bins, alpha=0.7, label="mean over S")
  ax.hist(mm_best, bins=bins, alpha=0.7, label="best of S")
  ax.set_xscale("log"); ax.set_xlabel("mismatch"); ax.set_ylabel("count")
  ax.legend(); fig.tight_layout(); fig.savefig(out / "mismatch_hist.png", dpi=130)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(6, 4))
  sc = ax.scatter(theta[:, 0], mm_mean, c=0.5 * (theta[:, 1] + theta[:, 2]),
                  cmap="coolwarm", s=14)
  ax.set_yscale("log"); ax.set_xlabel("q"); ax.set_ylabel("mismatch")
  fig.colorbar(sc, label="mean spin")
  fig.tight_layout(); fig.savefig(out / "mismatch_vs_q.png", dpi=130)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(np.median(d_tf, axis=0), label="teacher-forced")
  ax.plot(np.median(d_ro, axis=0), label="free rollout")
  ax.set_yscale("log"); ax.set_xlabel("patch index (from context end)")
  ax.set_ylabel("median |Delta-phi error| (rad)"); ax.legend()
  fig.tight_layout(); fig.savefig(out / "dphi_error.png", dpi=130)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(7, 4))
  med = np.median(drift, axis=0)
  lo, hi = np.quantile(drift, 0.1, axis=0), np.quantile(drift, 0.9, axis=0)
  x = np.arange(drift.shape[1])
  ax.plot(x, med, label="median accumulated phase error")
  ax.fill_between(x, lo, hi, alpha=0.3, label="10-90%")
  ax.axhline(1.0, color="r", ls=":", label="1 rad decoherence threshold")
  ax.set_yscale("log"); ax.set_xlabel("sample (from context end)")
  ax.set_ylabel("accumulated phase error pred vs true (rad)"); ax.legend()
  ax.set_title("free-rollout dephasing (constant global offset removed)")
  fig.tight_layout(); fig.savefig(out / "phase_drift.png", dpi=130)
  plt.close(fig)

  print(f"\nFigures + metrics.json + per_waveform.csv written to {out}/")


if __name__ == "__main__":
  main()