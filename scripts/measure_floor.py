#!/usr/bin/env python
"""[ITEM 2] Representation floor + end-to-end mismatch on real NRHybSur data.

Floor      = mismatch( h_orig , reconstruct(true tokens) )   -- no model involved
End-to-end = mismatch( h_orig , reconstruct(pred tokens) )   -- needs --checkpoint
Token-space = mismatch( reconstruct(true) , reconstruct(pred) ) -- same rollouts

These three are measured separately because floor and model error are both
merger-concentrated and correlated — do NOT estimate end-to-end as floor+model.

Usage:
    python scripts/measure_floor.py --manifest data/25k/manifest.parquet --split test \\
        --n-waveforms 200 [--checkpoint best.ckpt --n-steps 50]

Prints median / p90 for each quantity and writes floor_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# [ITEM 2] Allow `python scripts/measure_floor.py` from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from datasets.datamodule import _collate_batch  # noqa: E402
from datasets.patchify import n_patches_for, patchify_tracks_batched, build_token_features_batched  # noqa: E402
from datasets.waveform_dataset import WaveformPatchDataset  # noqa: E402
from models.lightning_module import GWFlowSurrogateLit  # noqa: E402


def phase_opt_mismatch(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
  """[ITEM 2] Phase-optimised mismatch, same formula as evalute.py evaluate_batch
  (1 - |Σ h1 conj(h2)| / (‖h1‖‖h2‖)). Returns per-row mismatches (B,)."""
  # Work in complex128 for the same reason as h_orig: long unwrapped phases.
  h1 = h1.to(torch.complex128)
  h2 = h2.to(torch.complex128)
  inner = (h1 * h2.conj()).sum(dim=1)
  den = torch.sqrt(
    (h1.abs() ** 2).sum(1) * (h2.abs() ** 2).sum(1)
  ).clamp_min(1e-30)
  return (1.0 - inner.abs() / den).real.float()


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--split", default="test")
  ap.add_argument("--n-waveforms", type=int, default=200)
  ap.add_argument("--patch-len", type=int, default=32)
  ap.add_argument("--batch-size", type=int, default=16)
  ap.add_argument("--checkpoint", default=None,
                  help="If set, also measure end-to-end + token-space mismatch")
  ap.add_argument("--n-steps", type=int, default=20)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--out", default="floor_metrics.json")
  args = ap.parse_args()

  device = torch.device(args.device)
  P = args.patch_len
  manifest = Path(args.manifest)
  stats_path = manifest.parent / "norm_stats.npz"

  # [ITEM 2] Raw tracks; patchify happens locally so floor does not depend on
  # training normalisation. residual_scale=1.0 + identity norms below.
  ds = WaveformPatchDataset(
    manifest, split=args.split, patch_len=P, context_patches=1,
    norm_stats_path=str(stats_path) if stats_path.is_file() else None,
    preload=True,
  )
  n = min(args.n_waveforms, len(ds))
  loader = DataLoader(
    Subset(ds, range(n)), batch_size=args.batch_size, collate_fn=_collate_batch,
  )

  # [ITEM 2] Floor reconstructor: identity stats + residual_scale=1.0 so physical
  # tokens from patchify_tracks_batched pass through unscaled — this is the
  # *exact* _reconstruct_waveform code path used at evaluation time.
  lit_floor = GWFlowSurrogateLit(patch_len=P, residual_scale=1.0, n_channels=1)
  lit_floor.eval()

  lit_model = None
  if args.checkpoint:
    # [ITEM 2] End-to-end path must use the checkpoint's residual_scale / norms.
    try:
      lit_model = GWFlowSurrogateLit.load_from_checkpoint(
        args.checkpoint, map_location=device
      )
    except RuntimeError:
      lit_model = GWFlowSurrogateLit.load_from_checkpoint(
        args.checkpoint, map_location=device, use_abs_time=False
      )
    lit_model.eval().to(device)
    P = lit_model.hparams.patch_len

  floors, e2e, tok = [], [], []
  with torch.no_grad():
    for batch in loader:
      la_track = batch["log_a_track"]   # (B, C, T)
      phi_track = batch["phi_track"]
      B = la_track.shape[0]
      T = la_track.shape[-1]
      n_pat = n_patches_for(T, P)
      usable = n_pat * P

      # [ITEM 2] fp64: phi reaches hundreds of radians; float32 costs ~1e-5 phase noise.
      # Truncate to the samples the patch representation actually covers.
      la64 = la_track[:, 0, :usable].double()
      phi64 = phi_track[:, 0, :usable].double()
      h_orig = torch.exp(la64) * torch.exp(-1j * phi64)   # (B, usable)

      # Physical tokens (no normalisation) — representation of the truth.
      patches = patchify_tracks_batched(la_track.float(), phi_track.float(), P)
      zeros = torch.zeros(B, device=la_track.device)
      ones = torch.ones(B, device=la_track.device)
      h_true_tok = lit_floor._reconstruct_waveform(
        patches["log_a"][:, :, 0],
        patches["delta_phi"][:, :, 0],
        patches["phi_residual"][:, :, 0],
        zeros, ones, zeros, ones, P,
      ).to(torch.complex128)

      floors.append(phase_opt_mismatch(h_orig, h_true_tok).cpu().numpy())

      if lit_model is None:
        continue

      # [ITEM 2] End-to-end: seed 1 context patch, generate the rest, reconstruct
      # with the checkpoint's residual_scale + training norm stats, vs h_orig.
      batch_dev = {k: v.to(device) for k, v in batch.items()}
      batch_dev = lit_model._patchify_batch(batch_dev)
      feat = batch_dev["token_features"]
      ctx = 1
      n_gen = feat.shape[1] - ctx
      gen = lit_model.model.generate(
        feat[:, :ctx], batch_dev["theta"], n_gen,
        n_steps=args.n_steps, n_samples=1,
      )[:, 0]                                           # (B, n_gen, D)
      C = lit_model.hparams.n_channels
      la_g, dp_g, res_g = lit_model._split_features(gen, C, P)
      la_m = batch_dev["norm_log_a_mean"]
      la_s = batch_dev["norm_log_a_std"]
      dp_m = batch_dev["norm_dphi_mean"]
      dp_s = batch_dev["norm_dphi_std"]
      h_pred = lit_model._reconstruct_waveform(
        la_g[:, :, 0], dp_g[:, :, 0], res_g[:, :, 0],
        la_m, la_s, dp_m, dp_s, P,
      ).to(torch.complex128)

      # Full-waveform e2e: concat true seed samples + predicted future.
      # Seed region uses the true reconstruction (ground truth by construction);
      # compare the generated region + seed against h_orig over usable length.
      h_true_full = lit_model._reconstruct_waveform(
        batch_dev["log_a"][:, :, 0],
        batch_dev["delta_phi"][:, :, 0],
        batch_dev["phi_residual"][:, :, 0],
        la_m, la_s, dp_m, dp_s, P,
      ).to(torch.complex128)
      # Build hybrid: true tokens for context, predicted for future.
      seed_samp = ctx * P
      h_e2e = torch.cat([h_true_full[:, :seed_samp], h_pred], dim=1)
      # Truncate/pad to usable if needed (should already match n_pat * P).
      if h_e2e.shape[1] > usable:
        h_e2e = h_e2e[:, :usable]
      elif h_e2e.shape[1] < usable:
        pad = usable - h_e2e.shape[1]
        h_e2e = torch.nn.functional.pad(h_e2e, (0, pad))

      e2e.append(phase_opt_mismatch(h_orig.to(device), h_e2e).cpu().numpy())

      # Token-space: both sides from reconstruction (floor-free scaling metric).
      h_true_gen = lit_model._reconstruct_waveform(
        batch_dev["log_a"][:, ctx:, 0],
        batch_dev["delta_phi"][:, ctx:, 0],
        batch_dev["phi_residual"][:, ctx:, 0],
        la_m, la_s, dp_m, dp_s, P,
      ).to(torch.complex128)
      tok.append(phase_opt_mismatch(h_true_gen, h_pred).cpu().numpy())

  floor_arr = np.concatenate(floors)
  def stats(a: np.ndarray) -> dict:
    return {"median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
            "mean": float(a.mean()), "max": float(a.max())}

  out = {
    "floor": stats(floor_arr),
    "end_to_end": stats(np.concatenate(e2e)) if e2e else None,
    "token_space": stats(np.concatenate(tok)) if tok else None,
    "n_waveforms": int(n),
    "patch_len": int(P),
    "split": args.split,
  }
  Path(args.out).write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  print(f"\n-> {args.out}")


if __name__ == "__main__":
  main()
