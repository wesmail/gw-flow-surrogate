#!/usr/bin/env python
"""Time theta -> h22 for our model the way a PE run uses it (warm, repeated calls).

    python pe/time_fm.py --checkpoint "$FT" --manifest data/n1000k/manifest.parquet \
        --norm-stats data/n1000k/norm_stats.npz --batches 64 256 1024

For each batch size: the first call (includes torch.compile for that shape), then the best of
--repeats warm calls, split into start tokens (NRHybSur3dq8 on CPU) and the model itself.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pe.fm_waveform import FMSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    fm = FMSource(a.checkpoint, a.manifest, a.norm_stats)
    cuda = fm.dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    print(f"device: {fm.dev}{' (' + torch.cuda.get_device_name(fm.dev) + ')' if cuda else ''}, "
          f"torch {torch.__version__}")
    rng = np.random.default_rng(0)

    print(f"{'batch':>6} {'first call':>11} {'warm call':>10} {'of which tokens':>16} {'ms/waveform':>12}  fast")
    for B in a.batches:
        th = np.column_stack([rng.uniform(1.2, 7.5, B), rng.uniform(-.7, .7, B), rng.uniform(-.7, .7, B)])
        t0 = time.perf_counter(); fm.h22(*th.T); sync(); t_first = time.perf_counter() - t0
        t_warm = np.inf
        for _ in range(a.repeats):
            t0 = time.perf_counter(); fm.h22(*th.T); sync(); t_warm = min(t_warm, time.perf_counter() - t0)
        t0 = time.perf_counter(); fm.nr.records(*th.T); t_tok = time.perf_counter() - t0
        print(f"{B:>6} {t_first:>10.1f}s {t_warm:>9.1f}s {t_tok:>15.1f}s {1e3 * t_warm / B:>12.1f}  {fm._fast_ok}")


if __name__ == "__main__":
    main()