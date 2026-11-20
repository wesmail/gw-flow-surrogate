#!/usr/bin/env python
"""Profiler snippet: kernel launches + GPU ms for 20 patches, batch 1, n_steps 12.

Usage::

    python scripts/profile_generate.py --checkpoint last.ckpt [--fast]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from datasets.datamodule import _collate_batch
from datasets.waveform_dataset import WaveformPatchDataset
from models.lightning_module import GWFlowSurrogateLit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="last.ckpt")
    ap.add_argument("--manifest", default="data/n1000k/manifest.parquet")
    ap.add_argument("--norm-stats", default="data/n1000k/norm_stats.npz")
    ap.add_argument("--n-patches", type=int, default=20)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this profiler snippet")
    dev = torch.device("cuda")
    lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev).eval().to(dev)
    P = int(lit.hparams.patch_len)
    ds = WaveformPatchDataset(
        a.manifest, split="test", patch_len=P, context_patches=1,
        norm_stats_path=a.norm_stats, preload=False,
    )
    batch = next(iter(DataLoader(Subset(ds, [0]), batch_size=1, collate_fn=_collate_batch)))
    b = lit._patchify_batch({k: v.to(dev) for k, v in batch.items()})
    seed, theta = b["token_features"][:, :1], b["theta"]
    gen_fn = lit.model.generate_fast if a.fast else lit.model.generate

    # warmup
    with torch.no_grad():
        gen_fn(seed, theta, a.n_patches, n_steps=a.n_steps, n_samples=1, mean_of=1)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with torch.no_grad():
            gen_fn(seed, theta, a.n_patches, n_steps=a.n_steps, n_samples=1, mean_of=1)
        torch.cuda.synchronize()

    launches = sum(1 for e in prof.events() if e.device_type.name == "CUDA" and e.count)
    # Prefer cuda_time_total (us) when present
    gpu_us = sum(getattr(e, "cuda_time_total", 0) or 0 for e in prof.events())
    print(f"path={'generate_fast' if a.fast else 'generate'}  patches={a.n_patches}  "
          f"CUDA events≈{launches}  GPU time={gpu_us/1e3:.2f} ms  "
          f"route={getattr(lit.model, '_fast_capture_route', 'n/a')}")
    print(f"per-patch launches≈{launches/max(a.n_patches,1):.1f}  "
          f"per-patch GPU ms≈{gpu_us/1e3/max(a.n_patches,1):.3f}")


if __name__ == "__main__":
    main()
