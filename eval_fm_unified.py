#!/usr/bin/env python
"""Unified evaluation of the flow-matching surrogate — SAME conventions and
CSV columns as eval_forecaster_gw.py, so every existing figure script works.

Wraps your evalute.py:evaluate_batch and post-processes its h_true/h_pred into
the three mismatch conventions (strict / phase-opt / phase+time-opt).

Context protocol (§5.6 / [ITEM 4])
----------------------------------
This script defaults to --context-fraction 0.5 (half the patches seeded).
`evalute.py` defaults to context_patches=1 (θ-only generation). These are
different tasks; do not quote mismatch numbers across the two protocols.

RUN FROM THE FLOW-MATCHING REPO ROOT (this file + evalute.py side by side):
python eval_fm_unified.py \\
    --checkpoint logs/flow_10k_small/version_2/checkpoints/last.ckpt \\
    --manifest data/25k/manifest.parquet --split test \\
    --context-fraction 0.5 --n-waveforms 500 --n-samples 1 --mean-of 4 \\
    --out results/e3_fm_ctx0.5.csv

Notes on matching the GPT protocol:
* --context-fraction sets the seed to the same fraction of the sequence the
  GPT gets (your old evalute.py used context_patches=1, i.e. theta-only).
* --mean-of 4 with --n-samples 1 approximates the conditional mean, the fair
  partner of GPT greedy decoding. For sampled-vs-sampled use --mean-of 1.
* Mismatches are computed on the GENERATED segment (matches the GPT eval's
  *_future columns; at small context this is also ~ the full waveform).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.cwd()))
from datasets.patchify import n_patches_for  # noqa: E402
from datasets.waveform_dataset import WaveformPatchDataset  # noqa: E402
from models.lightning_module import GWFlowSurrogateLit  # noqa: E402

# import evaluate_batch + collate from evalute.py (filename isn't importable)
spec = importlib.util.spec_from_file_location("fm_eval", Path(__file__).parent / "evalute.py")
fm_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm_eval)


def mismatches(h1: np.ndarray, h2: np.ndarray) -> tuple[float, float, float]:
    """(strict, phase_opt, phase_time_opt) — identical to the GPT eval."""
    h1 = np.asarray(h1, dtype=np.complex128).ravel()
    h2 = np.asarray(h2, dtype=np.complex128).ravel()
    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
    if n1 == 0 or n2 == 0:
        return 1.0, 1.0, 1.0
    inner = np.vdot(h1, h2)
    strict = 1.0 - float(np.real(inner) / (n1 * n2))
    phase_opt = 1.0 - float(np.abs(inner) / (n1 * n2))
    n_fft = 1 << int(np.ceil(np.log2(2 * h1.size)))
    xcorr = np.fft.ifft(np.conj(np.fft.fft(h1, n_fft)) * np.fft.fft(h2, n_fft))
    pt_opt = 1.0 - float(np.abs(xcorr).max() / (n1 * n2))
    return strict, phase_opt, pt_opt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--context-fraction", type=float, default=0.5)
    ap.add_argument("--n-waveforms", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--mean-of", type=int, default=4,
                    help="flows averaged per patch (4 ~ conditional mean ~ GPT greedy)")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/e3_fm.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    lit = GWFlowSurrogateLit.load_from_checkpoint(args.checkpoint, map_location=args.device)
    lit.eval().to(args.device)
    stats_path = Path(args.manifest).parent / "norm_stats.npz"

    # Discover patch count from raw track geometry (dataset returns tracks, not tokens).
    ds_probe = WaveformPatchDataset(args.manifest, split=args.split, context_patches=1,
                                    norm_stats_path=stats_path, preload=False)
    N = n_patches_for(ds_probe.n_time, lit.hparams.patch_len)
    ctx = max(1, int(round(N * args.context_fraction)))
    print(f"sequence: {N} patches -> context {ctx} ({args.context_fraction:.0%})")
    ds = WaveformPatchDataset(args.manifest, split=args.split, context_patches=ctx,
                              norm_stats_path=stats_path, preload=True)
    n = min(args.n_waveforms, len(ds))
    loader = DataLoader(torch.utils.data.Subset(ds, range(n)),
                        batch_size=args.batch_size, collate_fn=fm_eval.collate)

    rows = []
    with torch.no_grad():
        for batch in loader:
            r = fm_eval.evaluate_batch(lit, batch, n_samples=args.n_samples,
                                       n_steps=args.n_steps, limit_patches=0,
                                       device=torch.device(args.device),
                                       mean_of=args.mean_of)
            h_true, h_pred = r["h_true"].numpy(), r["h_pred"].numpy()
            for b in range(h_true.shape[0]):
                s, p, pt = mismatches(h_true[b], h_pred[b])
                q, c1, c2 = r["theta"][b, :3].tolist()
                rows.append(dict(
                    idx=len(rows), q=q, chi1z=c1, chi2z=c2,
                    context_fraction=args.context_fraction,
                    n_context_tokens=r["ctx"], n_generated_tokens=r["n_gen"],
                    mismatch_strict_future=s, mismatch_phase_future=p,
                    mismatch_phase_time_future=pt,
                    # full-waveform columns: seed region is ground truth by
                    # construction, so future == full up to the seed's weight
                    mismatch_strict_full=s, mismatch_phase_time_full=pt,
                ))
            print(f"  {len(rows)}/{n}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    for col in ("mismatch_strict_future", "mismatch_phase_future", "mismatch_phase_time_future"):
        v = np.array([r[col] for r in rows])
        print(f"{col:30s} p50={np.percentile(v,50):.3e}  p90={np.percentile(v,90):.3e}")
    print(f"CSV -> {out}")


if __name__ == "__main__":
    main()
