#!/usr/bin/env python
"""Unified paper evaluation of the flow-matching surrogate.

Computes mismatches of the generated waveform against the ORIGINAL HDF5
waveform (not token reconstruction), with full-then-slice amplitude
reconstruction.  Headline metric under the frozen protocol: ``mm_pt_full``.

Without ``--protocol``, legacy CLI defaults are unchanged (ctx 0.5, etc.).
With ``--protocol``, knobs come from ``PROTOCOL.yaml``; conflicting CLI flags
raise unless ``--override`` is passed (recorded in the sidecar JSON).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "paper_figs"))

from datasets.datamodule import _collate_batch  # noqa: E402
from datasets.patchify import n_patches_for  # noqa: E402
from datasets.waveform_dataset import WaveformPatchDataset  # noqa: E402
from protocol_utils import (  # noqa: E402
    assert_not_identity_stats,
    context_patches_from_fraction,
    enforce_protocol,
    load_checkpoint,
    load_protocol,
    mismatches,
    original_waveform,
    reconstruct_fp64,
    resolve_norm_stats,
    write_sidecar,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=None, help="default: protocol test_manifest or required")
    ap.add_argument("--split", default=None)
    ap.add_argument("--context-fraction", type=float, default=None)
    ap.add_argument("--n-waveforms", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--mean-of", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--amp-mode", default=None, choices=("constant", "linear", "pchip"))
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--protocol", default=None, help="results/protocol/PROTOCOL.yaml")
    ap.add_argument("--override", action="store_true",
                    help="allow CLI knobs to differ from --protocol (recorded in sidecar)")
    ap.add_argument("--out", default="results/e3_fm.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    protocol = None
    overrides: dict = {}
    if args.protocol:
        protocol = load_protocol(args.protocol)
        overrides = enforce_protocol(args, protocol, override=args.override)
        if args.n_waveforms is None:
            args.n_waveforms = int(protocol["n_waveforms_flagship"])
        if args.manifest is None:
            args.manifest = protocol["test_manifest"]
        if args.split is None:
            args.split = protocol["split"]
    else:
        # Legacy defaults (pre-protocol behaviour).
        if args.manifest is None:
            ap.error("--manifest is required without --protocol")
        args.split = args.split or "test"
        args.context_fraction = 0.5 if args.context_fraction is None else args.context_fraction
        args.n_waveforms = 500 if args.n_waveforms is None else args.n_waveforms
        args.n_samples = 1 if args.n_samples is None else args.n_samples
        args.mean_of = 4 if args.mean_of is None else args.mean_of
        args.n_steps = 20 if args.n_steps is None else args.n_steps
        args.batch_size = 16 if args.batch_size is None else args.batch_size
        args.amp_mode = "constant" if args.amp_mode is None else args.amp_mode
        args.temperature = 1.0 if args.temperature is None else args.temperature
        args.seed = 1234 if args.seed is None else args.seed

    window_frac = float(protocol["window_start_fraction"]) if protocol else 0.75
    dt_M = float(protocol["grid"]["dt"]) if protocol else 0.5

    t0 = time.time()
    stats_path = resolve_norm_stats(args.checkpoint, args.norm_stats)
    lit = load_checkpoint(args.checkpoint, map_location=args.device)
    lit.hparams.amp_mode = args.amp_mode
    lit.to(args.device)

    ds_probe = WaveformPatchDataset(
        args.manifest, split=args.split, context_patches=1,
        norm_stats_path=stats_path, preload=False,
    )
    assert_not_identity_stats(ds_probe.norm_stats, path=stats_path)
    P = int(lit.hparams.patch_len)
    N = n_patches_for(ds_probe.n_time, P)
    ctx = context_patches_from_fraction(N, float(args.context_fraction))
    usable = N * P
    s_w = int(round(window_frac * N)) * P
    print(f"sequence: {N} patches -> context {ctx} "
          f"(fraction {args.context_fraction}); amp_mode={args.amp_mode}")

    ds = WaveformPatchDataset(
        args.manifest, split=args.split, context_patches=ctx,
        norm_stats_path=stats_path, preload=True, share_memory=False,
    )
    n = min(int(args.n_waveforms), len(ds))
    loader = DataLoader(
        Subset(ds, range(n)), batch_size=int(args.batch_size),
        collate_fn=_collate_batch, shuffle=False,
    )

    rscale = float(lit.hparams.residual_scale)
    rows: list[dict] = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            torch.manual_seed(int(args.seed) + bi)
            b = {k: v.to(args.device) for k, v in batch.items()}
            b = lit._patchify_batch(b)
            feat = b["token_features"]
            B = feat.shape[0]

            h_orig = original_waveform(b, usable)
            h_tok = reconstruct_fp64(feat, b, rscale, P, amp_mode=args.amp_mode)

            gen = lit.model.generate(
                feat[:, :ctx], b["theta"], N - ctx,
                n_steps=int(args.n_steps), n_samples=int(args.n_samples),
                temperature=float(args.temperature), mean_of=int(args.mean_of),
            )[:, 0]
            h_pred = reconstruct_fp64(
                torch.cat([feat[:, :ctx], gen], dim=1), b, rscale, P,
                amp_mode=args.amp_mode,
            )

            fl_s, _, fl_pt, _ = mismatches(h_orig, h_tok)
            fl_s_w, _, fl_pt_w, _ = mismatches(h_orig[:, s_w:], h_tok[:, s_w:])
            mm_s, _, mm_pt, sh = mismatches(h_orig, h_pred)
            _, _, mm_pt_w, _ = mismatches(h_orig[:, s_w:], h_pred[:, s_w:])

            # |Δφ| at amplitude peak (unwrapped), seed-aligned.
            s_g = ctx * P
            peak = h_orig.abs().argmax(dim=1)
            rel = (h_pred[:, s_g:] * h_orig[:, s_g:].conj()).angle().cpu().numpy()
            rel = np.unwrap(rel, axis=1)
            pk = (peak - s_g).clamp_min(0).cpu().numpy()
            dphi_peak = rel[np.arange(B), pk] - rel[:, 0]

            theta = b["theta"].double().cpu().numpy()
            for i in range(B):
                q, c1, c2 = theta[i, :3]
                rows.append(dict(
                    idx=bi * int(args.batch_size) + i,
                    q=float(q), chi1z=float(c1), chi2z=float(c2),
                    context_fraction=float(args.context_fraction),
                    n_context_tokens=ctx,
                    n_generated_tokens=N - ctx,
                    amp_mode=args.amp_mode,
                    mm_strict_full=float(mm_s[i]),
                    mm_pt_full=float(mm_pt[i]),
                    mm_pt_window=float(mm_pt_w[i]),
                    floor_pt_full=float(fl_pt[i]),
                    floor_pt_window=float(fl_pt_w[i]),
                    floor_strict_full=float(fl_s[i]),
                    dphi_peak_rad=float(dphi_peak[i]),
                    dt_shift_M=float(sh[i].cpu()) * dt_M,
                    # Legacy aliases kept for older figure scripts:
                    mismatch_strict_future=float(mm_s[i]),
                    mismatch_phase_time_future=float(mm_pt[i]),
                ))
            print(f"  {len(rows)}/{n}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    targets = (protocol or {}).get("target_mismatch", [1e-3, 1e-4])
    for col in ("mm_pt_full", "mm_strict_full", "mm_pt_window",
                "floor_pt_full", "floor_pt_window"):
        v = np.array([r[col] for r in rows], dtype=np.float64)
        parts = [
            f"p50={np.percentile(v, 50):.3e}",
            f"p90={np.percentile(v, 90):.3e}",
            f"p99={np.percentile(v, 99):.3e}",
            f"max={v.max():.3e}",
        ]
        for t in targets:
            parts.append(f"<{t:.0e}={(v < float(t)).mean():.1%}")
        print(f"{col:20s} " + "  ".join(parts))
    dphi = np.abs(np.array([r["dphi_peak_rad"] for r in rows], dtype=np.float64))
    dt = np.abs(np.array([r["dt_shift_M"] for r in rows], dtype=np.float64))
    print(f"{'|dphi_peak_rad|':20s} p50={np.percentile(dphi, 50):.3e}  "
          f"p90={np.percentile(dphi, 90):.3e}  max={dphi.max():.3e}")
    print(f"{'|dt_shift_M|':20s} p50={np.percentile(dt, 50):.3e}  "
          f"p90={np.percentile(dt, 90):.3e}  max={dt.max():.3e}")

    write_sidecar(
        out,
        checkpoint=args.checkpoint,
        norm_stats=stats_path,
        manifest=args.manifest,
        protocol=protocol or {},
        overrides=overrides,
        extra={
            "n_waveforms": n,
            "context_patches": ctx,
            "amp_mode": args.amp_mode,
            "n_steps": int(args.n_steps),
            "mean_of": int(args.mean_of),
            "seed": int(args.seed),
            "protocol_sha256": (protocol or {}).get("_protocol_sha256"),
        },
        wall_time_s=time.time() - t0,
        device=str(args.device),
    )
    print(f"CSV -> {out}")


if __name__ == "__main__":
    main()
