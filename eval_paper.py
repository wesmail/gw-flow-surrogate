#!/usr/bin/env python
"""Paper evaluator — the only entry point for quoted mismatches.

Fixes P1–P6, P10, P11 relative to ``eval_fm_unified.py`` / ``evaluate_batch``.
Does **not** call ``evaluate_batch`` (no teacher-forced pass).

Usage::

    python eval_paper.py \\
        --protocol results/protocol/PROTOCOL.yaml \\
        --checkpoint scan_1M/n1000k/seed42/checkpoints/<best>.ckpt \\
        --norm-stats /path/to/train_norm_stats.npz \\
        --n-waveforms 1000 \\
        --context-fraction 0.5 \\
        --n-steps 40 \\
        --out results/paper/flagship.csv

``--n-steps`` and ``--context-fraction`` may come from the protocol file once
Phase 4a/4b fill them; until then they are required on the CLI (or in YAML).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_figs"))

from datasets.datamodule import _collate_batch  # noqa: E402
from datasets.patchify import n_patches_for  # noqa: E402
from datasets.waveform_dataset import WaveformPatchDataset  # noqa: E402
from paper_figs.protocol_utils import (  # noqa: E402
    assert_grid_matches,
    assert_not_identity_stats,
    audited_training_grid,
    load_checkpoint,
    load_protocol,
    mismatches,
    original_waveform,
    reconstruct_fp64,
    resolve_norm_stats,
    write_sidecar,
)


def _require(proto: dict, key: str, cli_val):
    if cli_val is not None:
        return cli_val
    v = proto.get(key)
    if v is None:
        raise SystemExit(
            f"Missing {key}: set it in PROTOCOL.yaml (after Phase 4 gate) or pass --{key.replace('_', '-')}"
        )
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="results/protocol/PROTOCOL.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--norm-stats", default=None, help="training norm_stats.npz (else hparams)")
    ap.add_argument("--manifest", default=None, help="override protocol test_manifest")
    ap.add_argument("--split", default=None)
    ap.add_argument("--n-waveforms", type=int, default=None)
    ap.add_argument("--context-fraction", type=float, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--mean-of", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proto = load_protocol(args.protocol)
    overrides = {}

    def take(key, cli, cast=lambda x: x):
        if cli is not None:
            overrides[key] = cast(cli)
            return cast(cli)
        return cast(_require(proto, key, None)) if key in (
            "n_steps", "context_fraction"
        ) else cast(proto.get(key) if cli is None else cli)

    # Required protocol knobs (n_steps / context_fraction must be set)
    n_steps = int(_require(proto, "n_steps", args.n_steps))
    if args.n_steps is not None:
        overrides["n_steps"] = n_steps
    ctx_frac = float(_require(proto, "context_fraction", args.context_fraction))
    if args.context_fraction is not None:
        overrides["context_fraction"] = ctx_frac

    manifest = Path(args.manifest or proto["test_manifest"])
    split = args.split or proto.get("split", "test")
    n_waveforms = int(args.n_waveforms or proto.get("n_waveforms_flagship") or 5000)
    mean_of = int(args.mean_of if args.mean_of is not None else proto.get("mean_of", 4))
    temperature = float(args.temperature if args.temperature is not None else proto.get("temperature", 1.0))
    seed = int(args.seed if args.seed is not None else proto.get("seed", 1234))
    batch_size = int(args.batch_size if args.batch_size is not None else proto.get("batch_size", 64))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = Path(args.checkpoint)
    stats_path = resolve_norm_stats(ckpt, args.norm_stats)
    lit = load_checkpoint(ckpt, map_location=device)
    lit.to(device)
    P = int(lit.hparams.patch_len)
    rscale = float(lit.hparams.residual_scale)
    if int(lit.hparams.n_channels) != 1:
        raise NotImplementedError("eval_paper assumes n_channels == 1")

    # Dataset + norm-stats guards (P1/P2)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    ds = WaveformPatchDataset(
        manifest, split=split, patch_len=P, context_patches=1,
        norm_stats_path=str(stats_path), preload=True,
    )
    assert_not_identity_stats(ds.norm_stats, path=stats_path)

    # Grid guard (P11)
    man = pd.read_parquet(manifest)
    shard = Path(man["file_path"].iloc[0])
    if not shard.is_file():
        shard = Path(manifest).parent / shard.name
    with h5py.File(shard, "r") as f:
        t_grid = np.asarray(f["t"][:], dtype=np.float64)
    assert_grid_matches(t_grid, audited_training_grid(), label="test manifest")

    N = n_patches_for(ds.n_time, P)
    usable = N * P
    ctx = 1 if ctx_frac <= 0 else max(1, int(round(ctx_frac * N)))
    n = min(n_waveforms, len(ds))
    if n < n_waveforms:
        raise RuntimeError(f"split {split!r} has only {len(ds)} rows (asked {n_waveforms})")

    print(
        f"eval_paper: N={N} patches, ctx={ctx} ({ctx_frac}), n_steps={n_steps}, "
        f"mean_of={mean_of}, seed={seed}, n={n}, device={device}"
    )
    print(f"  ckpt={ckpt}")
    print(f"  norm={stats_path}")
    print(f"  manifest={manifest} split={split}")

    loader = DataLoader(
        Subset(ds, range(n)), batch_size=batch_size, collate_fn=_collate_batch, shuffle=False
    )
    rows: list[dict] = []
    gen_feats: list[np.ndarray] = []
    curve_dphi: list[np.ndarray] = []
    curve_damp: list[np.ndarray] = []
    t0 = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            torch.manual_seed(seed + bi)  # P5: common random numbers
            b = {k: v.to(device) for k, v in batch.items()}
            b = lit._patchify_batch(b)
            feat = b["token_features"]  # (B, N, D)
            B = feat.shape[0]
            n_gen = N - ctx

            h_orig = original_waveform(b, usable)
            h_tok = reconstruct_fp64(feat, b, rscale, P)

            gen = lit.model.generate(
                feat[:, :ctx], b["theta"], n_gen,
                n_steps=n_steps, n_samples=1,
                temperature=temperature, mean_of=mean_of,
            )[:, 0]
            feat_full = torch.cat([feat[:, :ctx], gen], dim=1)
            h_pred = reconstruct_fp64(feat_full, b, rscale, P)

            s_g = ctx * P
            # vs original — generated segment (headline)
            g = mismatches(h_orig[:, s_g:], h_pred[:, s_g:])
            # vs original — full waveform (real, P4)
            full = mismatches(h_orig, h_pred)
            # vs token reconstruction on generated segment
            tok = mismatches(h_tok[:, s_g:], h_pred[:, s_g:])
            # representation floor on generated segment
            fl = mismatches(h_orig[:, s_g:], h_tok[:, s_g:])

            # dephasing / amp at peak (no alignment)
            peak = h_orig.abs().argmax(dim=1)
            rel = (h_pred[:, s_g:] * h_orig[:, s_g:].conj()).angle().cpu().numpy()
            rel = np.unwrap(rel, axis=1)
            pk = (peak - s_g).clamp_min(0).cpu().numpy()
            dphi_peak = rel[np.arange(B), pk] - rel[:, 0]
            a_pred = h_pred.abs()
            a_true = h_orig.abs()
            damp_peak = (
                (a_pred[torch.arange(B, device=device), peak]
                 / a_true[torch.arange(B, device=device), peak].clamp_min(1e-30) - 1.0)
                .cpu().numpy()
            )

            # curves at patch boundaries on the generated region
            pb = np.arange(n_gen, dtype=np.int64) * P
            # relative phase / amp at start of each generated patch
            dphi_c = rel[:, pb] - rel[:, :1]
            damp_c = (
                (a_pred[:, s_g:][:, pb] / a_true[:, s_g:][:, pb].clamp_min(1e-30) - 1.0)
                .cpu().numpy()
            )

            theta = b["theta"].double().cpu().numpy()
            to = lambda x: x.double().cpu().numpy()  # noqa: E731
            for i in range(B):
                q, c1, c2 = map(float, theta[i, :3])
                rows.append(dict(
                    idx=bi * batch_size + i,
                    q=q, chi1z=c1, chi2z=c2,
                    chi_eff=(q * c1 + c2) / (1.0 + q),
                    context_patches=ctx,
                    t_seed_end_M=float(t_grid[min(s_g, len(t_grid) - 1)]),
                    mm_strict_gen=float(to(g[0])[i]),
                    mm_phase_gen=float(to(g[1])[i]),
                    mm_pt_gen=float(to(g[2])[i]),
                    dt_shift_gen_M=float(to(g[3])[i]) * float(t_grid[1] - t_grid[0]),
                    mm_strict_full=float(to(full[0])[i]),
                    mm_pt_full=float(to(full[2])[i]),
                    mm_pt_gen_tokens=float(to(tok[2])[i]),
                    floor_pt_gen=float(to(fl[2])[i]),
                    dphi_peak_rad=float(dphi_peak[i]),
                    damp_peak_rel=float(damp_peak[i]),
                ))
            gen_feats.append(gen.float().cpu().numpy())
            curve_dphi.append(dphi_c.astype(np.float32))
            curve_damp.append(damp_c.astype(np.float32))

            done = min((bi + 1) * batch_size, n)
            el = time.time() - t0
            eta = el / done * (n - done) if done else 0.0
            print(f"  {done}/{n}  ({el / 60:.1f} min, ETA {eta / 60:.1f} min)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    np.savez_compressed(
        out.with_name(out.stem + "_tokens.npz"),
        tokens=np.concatenate(gen_feats, axis=0),
        idx=df["idx"].to_numpy(),
    )
    np.savez_compressed(
        out.with_name(out.stem + "_curves.npz"),
        dphi=np.concatenate(curve_dphi, axis=0),
        damp=np.concatenate(curve_damp, axis=0),
        idx=df["idx"].to_numpy(),
    )

    wall = time.time() - t0
    write_sidecar(
        out,
        checkpoint=ckpt,
        norm_stats=stats_path,
        manifest=manifest,
        protocol=proto,
        overrides=overrides,
        device=str(device),
        wall_time_s=wall,
        extra={
            "n_waveforms": n,
            "context_fraction": ctx_frac,
            "context_patches": ctx,
            "n_steps": n_steps,
            "mean_of": mean_of,
            "temperature": temperature,
            "seed": seed,
            "batch_size": batch_size,
            "split": split,
            "row_range": [0, n],
            "residual_scale": rscale,
            "patch_len": P,
            "n_patches": N,
            "headline_metric": proto.get("headline_metric", "mm_pt_gen"),
            "mm_pt_gen_median": float(df["mm_pt_gen"].median()),
            "mm_pt_gen_tokens_median": float(df["mm_pt_gen_tokens"].median()),
            "floor_pt_gen_median": float(df["floor_pt_gen"].median()),
        },
    )
    print(f"CSV → {out}")
    print(f"median mm_pt_gen={df['mm_pt_gen'].median():.4e}  "
          f"tokens={df['mm_pt_gen_tokens'].median():.4e}  "
          f"floor={df['floor_pt_gen'].median():.4e}")


if __name__ == "__main__":
    main()
