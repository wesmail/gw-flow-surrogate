#!/usr/bin/env python
"""Evaluate a trained GWSurrogate checkpoint: metrics + diagnostic plots.

Computes per-sample teacher-forced patch MSE and free-running rollout mismatch,
writes a summary report, histograms, and predicted-vs-true waveform overlays.

Example:
    python scripts/evaluate_model.py \\
        --ckpt-path logs/stage1/version_0/checkpoints/epoch=010-val_mismatch=0.1234.ckpt \\
        --manifest-path data/25k/manifest.parquet \\
        --split test \\
        --output-dir eval/test

    python scripts/evaluate_model.py --ckpt-path ... --split val --n-samples 32 \\
        --rollout-max-patches 64 --n-plot 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/evaluate_model.py` from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from datasets.datamodule import _collate_batch
from datasets.waveform_dataset import WaveformPatchDataset, resolve_shard_path
from models.lightning_module import GWSurrogateLit


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", type=Path, required=True, help="Lightning checkpoint (.ckpt)")
    p.add_argument("--manifest-path", type=Path, default=Path("data/25k/manifest.parquet"))
    p.add_argument("--norm-stats-path", type=Path, default=None, help="Default: <manifest-dir>/norm_stats.npz")
    p.add_argument("--split", choices=("train", "val", "test", "ood"), default="test")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: eval/<split> beside manifest")
    p.add_argument("--n-samples", type=int, default=50, help="Max waveforms to score (0 = all)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-plot", type=int, default=6, help="Number of waveform overlay figures")
    p.add_argument("--rollout-max-patches", type=int, default=64,
                   help="Cap autoregressive rollout length (0 = full sequence)")
    p.add_argument("--context-patches", type=int, default=1,
                   help="Context patches seeded before rollout")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                   help="Preload split into RAM (fast; needs memory for large splits)")
    p.add_argument("--plot-t-min", type=float, default=-80.0, help="Merger-window plot left edge (t/M)")
    p.add_argument("--plot-t-max", type=float, default=80.0, help="Merger-window plot right edge (t/M)")
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def _teacher_forced_mse(
    lit: GWSurrogateLit, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample teacher-forced MSE on horizon-0 next-patch targets."""
    batch = lit._patchify_batch(batch)
    out = lit.model(batch["token_features"], batch["theta"], batch["mask"])
    ctx = int(batch["context_patches"][0].item())
    h = 0
    n = batch["mask"].shape[1]
    lo, hi = ctx - 1, n - 1
    if hi <= lo:
        z = batch["log_a"].new_zeros(batch["log_a"].shape[0])
        return z, z.clone(), z.clone(), z.clone()

    m = batch["mask"][:, ctx:]
    mse_la = (out["mu_a"][:, lo:hi, h] - batch["log_a"][:, ctx:]).pow(2)
    mse_dp = (out["mu_dphi"][:, lo:hi, h] - batch["delta_phi"][:, ctx:]).pow(2)
    mse_res = (out["mu_res"][:, lo:hi, h] - batch["phi_residual"][:, ctx:]).pow(2)
    while m.ndim < mse_la.ndim:
        m = m.unsqueeze(-1)
    m = m.expand_as(mse_la)
    mse_la = (mse_la * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1)
    mse_dp = (mse_dp * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1)
    mse_res = (mse_res * m.unsqueeze(-1)).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2)).clamp_min(1)
    total = mse_la + lit.hparams.lambda_phi * (mse_dp + mse_res)
    return mse_la, mse_dp, mse_res, total


@torch.no_grad()
def _rollout_mismatch_per_sample(
    lit: GWSurrogateLit,
    batch: dict[str, torch.Tensor],
    rollout_max_patches: int,
) -> torch.Tensor:
    """Per-sample rollout mismatch (1 - phase-aligned overlap), shape (B,)."""
    batch = lit._patchify_batch(batch)
    p = lit.hparams.patch_len
    ctx = max(int(batch["context_patches"][0].item()), 1)
    b = batch["token_features"].shape[0]
    n_avail = int(batch["mask"].sum(dim=1).min().item())
    n_gen = n_avail - ctx
    if rollout_max_patches > 0:
        n_gen = min(n_gen, rollout_max_patches)
    if n_gen <= 0:
        return batch["log_a"].new_ones(b)

    seed = batch["token_features"][:, :ctx]
    gen = lit.model.generate(seed, batch["theta"], n_gen)

    la_m = batch["norm_log_a_mean"]
    la_s = batch["norm_log_a_std"]
    dp_m = batch["norm_dphi_mean"]
    dp_s = batch["norm_dphi_std"]

    h_pred = lit._reconstruct_waveform(
        gen["mu_a"], gen["mu_dphi"], gen["mu_res"], la_m, la_s, dp_m, dp_s, p,
    )
    h_true = lit._reconstruct_waveform(
        batch["log_a"][:, ctx : ctx + n_gen],
        batch["delta_phi"][:, ctx : ctx + n_gen],
        batch["phi_residual"][:, ctx : ctx + n_gen],
        la_m, la_s, dp_m, dp_s, p,
    )
    num = torch.abs((h_pred * h_true.conj()).sum(dim=1))
    den = torch.sqrt((h_pred.abs() ** 2).sum(1) * (h_true.abs() ** 2).sum(1)).clamp_min(1e-30)
    return 1.0 - num / den


def _evaluate_split(
    lit: GWSurrogateLit,
    loader: DataLoader,
    device: torch.device,
    rollout_max_patches: int,
) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = {
        "mse_log_a": [],
        "mse_dphi": [],
        "mse_residual": [],
        "mse_total_tf": [],
        "mismatch_rollout": [],
    }
    lit.eval()
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mse_la, mse_dp, mse_res, mse_tot = _teacher_forced_mse(lit, batch)
        mm = _rollout_mismatch_per_sample(lit, batch, rollout_max_patches)
        for name, tensor in (
            ("mse_log_a", mse_la),
            ("mse_dphi", mse_dp),
            ("mse_residual", mse_res),
            ("mse_total_tf", mse_tot),
            ("mismatch_rollout", mm),
        ):
            buckets[name].extend(tensor.detach().cpu().tolist())
    return buckets


def _plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        ("mismatch_rollout", "Rollout mismatch (1 − overlap)", "#dc2626"),
        ("mse_total_tf", "Teacher-forced total MSE", "#2563eb"),
        ("mse_log_a", "Teacher-forced MSE log A", "#7c3aed"),
        ("mse_dphi", "Teacher-forced MSE Δφ", "#0891b2"),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        vals = df[col].to_numpy()
        ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(vals), color="k", ls="--", lw=1, label=f"median={np.median(vals):.4g}")
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"Evaluation metrics — {df['split'].iloc[0]} (n={len(df)})", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "metrics_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_mismatch_vs_params(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df["q"], df["mismatch_rollout"], s=18, alpha=0.65, c="#dc2626")
    axes[0].set(xlabel="q", ylabel="rollout mismatch")
    axes[1].scatter(df["chi1z"], df["chi2z"], c=df["mismatch_rollout"], s=22, cmap="viridis")
    axes[1].set(xlabel=r"$\chi_{1z}$", ylabel=r"$\chi_{2z}$")
    cb = fig.colorbar(axes[1].collections[0], ax=axes[1], fraction=0.046)
    cb.set_label("mismatch")
    fig.suptitle("Rollout mismatch vs parameters", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "mismatch_vs_params.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


@torch.no_grad()
def _rollout_waveforms_single(
    lit: GWSurrogateLit,
    batch: dict[str, torch.Tensor],
    rollout_max_patches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (|h_pred|, |h_true|) for one sample (batch size 1)."""
    batch = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch = lit._patchify_batch(batch)
    p = lit.hparams.patch_len
    ctx = max(int(batch["context_patches"][0].item()), 1)
    n_avail = int(batch["mask"].sum(dim=1).min().item())
    n_gen = n_avail - ctx
    if rollout_max_patches > 0:
        n_gen = min(n_gen, rollout_max_patches)

    seed = batch["token_features"][:, :ctx]
    gen = lit.model.generate(seed, batch["theta"], n_gen)
    la_m, la_s = batch["norm_log_a_mean"], batch["norm_log_a_std"]
    dp_m, dp_s = batch["norm_dphi_mean"], batch["norm_dphi_std"]

    h_pred = lit._reconstruct_waveform(
        gen["mu_a"], gen["mu_dphi"], gen["mu_res"], la_m, la_s, dp_m, dp_s, p,
    )
    h_true = lit._reconstruct_waveform(
        batch["log_a"][:, ctx : ctx + n_gen],
        batch["delta_phi"][:, ctx : ctx + n_gen],
        batch["phi_residual"][:, ctx : ctx + n_gen],
        la_m, la_s, dp_m, dp_s, p,
    )
    return h_pred[0].abs().cpu().numpy(), h_true[0].abs().cpu().numpy()


def _load_stored_waveform(
    manifest_row: pd.Series, data_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp = resolve_shard_path(manifest_row["file_path"], data_dir)
    ri = int(manifest_row["row_index"])
    with h5py.File(fp, "r") as f:
        t = f["t"][:].astype(np.float64)
        h = f["h_re"][ri, 0] + 1j * f["h_im"][ri, 0]
        log_a = f["log_a"][ri, 0].astype(np.float64)
        phi = f["phi"][ri, 0].astype(np.float64)
    return t, np.abs(h), log_a, phi


def _plot_waveform_examples(
    lit: GWSurrogateLit,
    ds: WaveformPatchDataset,
    meta: pd.DataFrame,
    metrics: pd.DataFrame,
    device: torch.device,
    out_dir: Path,
    n_plot: int,
    rollout_max_patches: int,
    t_min: float,
    t_max: float,
) -> None:
    """Plot rollout |h| and stored log_a / φ for best, worst, and random samples."""
    if n_plot <= 0:
        return
    ranked = metrics.sort_values("mismatch_rollout")
    picks: list[int] = []
    if len(ranked) >= 1:
        picks.append(int(ranked.index[0]))   # best
    if len(ranked) >= 2:
        picks.append(int(ranked.index[-1]))  # worst
    mid = len(ranked) // 2
    if len(ranked) >= 3:
        picks.append(int(ranked.index[mid]))
    rng = np.random.default_rng(0)
    rest = [i for i in ranked.index if i not in picks]
    if rest:
        extra = rng.choice(rest, size=min(n_plot - len(picks), len(rest)), replace=False)
        picks.extend(int(i) for i in extra)
    picks = picks[:n_plot]

    for rank, row_idx in enumerate(picks):
        sample = ds[row_idx]
        batch = _collate_batch([sample])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mm = float(metrics.loc[row_idx, "mismatch_rollout"])
        row = meta.iloc[row_idx]

        t_store, amp_store, log_a_store, phi_store = _load_stored_waveform(row, ds.data_dir)
        amp_pred, amp_true = _rollout_waveforms_single(lit, batch, rollout_max_patches)

        # Map rollout samples onto the fixed geometric time grid.
        ctx_p = max(int(sample["context_patches"]), 1)
        p = lit.hparams.patch_len
        i0 = int(row["peak_index"]) + ctx_p * p
        i1 = min(i0 + len(amp_pred), len(t_store))
        t_roll = t_store[i0:i1]
        amp_pred = amp_pred[: len(t_roll)]
        amp_true = amp_true[: len(t_roll)]

        mask = (t_store >= t_min) & (t_store <= t_max)
        mer = (t_roll >= t_min) & (t_roll <= t_max)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t_store[mask], amp_store[mask], color="#94a3b8", lw=0.8, label="stored |h|")
        axes[0].plot(t_roll[mer], amp_true[mer], color="#16a34a", lw=1.0, label="true rollout")
        axes[0].plot(t_roll[mer], amp_pred[mer], color="#dc2626", ls="--", lw=1.0, label="pred rollout")
        axes[0].axvline(0, color="k", ls=":", alpha=0.4)
        axes[0].set_ylabel(r"$|h_{22}|$")
        axes[0].legend(fontsize=8, loc="upper right")

        axes[1].plot(t_store[mask], log_a_store[mask], color="#7c3aed", lw=0.8)
        axes[1].set_ylabel(r"$\log A$ (stored)")

        axes[2].plot(t_store[mask], phi_store[mask], color="#0891b2", lw=0.8)
        axes[2].set_ylabel(r"$\phi$ (stored)")
        axes[2].set_xlabel(r"$t/M$")

        fig.suptitle(
            f"Sample {row_idx}: q={row['q']:.2f}, χ₁={row['chi1z']:.2f}, χ₂={row['chi2z']:.2f}, "
            f"mismatch={mm:.4f}",
            fontsize=10,
            fontweight="bold",
        )
        fig.tight_layout()
        path = out_dir / f"waveform_{rank:02d}_idx{row_idx:05d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


def _write_report(df: pd.DataFrame, out_dir: Path, ckpt: Path, rollout_max_patches: int) -> None:
    lines = [
        "GWSurrogate evaluation report",
        "=" * 40,
        f"checkpoint: {ckpt}",
        f"split: {df['split'].iloc[0]}",
        f"n_samples: {len(df)}",
        f"rollout_max_patches: {rollout_max_patches or 'full'}",
        "",
    ]
    for col, label in (
        ("mismatch_rollout", "Rollout mismatch"),
        ("mse_total_tf", "Teacher-forced total MSE"),
        ("mse_log_a", "MSE log A"),
        ("mse_dphi", "MSE Δφ"),
        ("mse_residual", "MSE residual"),
    ):
        v = df[col].to_numpy()
        lines.append(
            f"{label:24s}  mean={v.mean():.6g}  median={np.median(v):.6g}  "
            f"std={v.std():.6g}  p10={np.percentile(v,10):.6g}  p90={np.percentile(v,90):.6g}"
        )
    lines.append("")
    lines.append("Per-sample metrics: metrics.parquet")
    path = out_dir / "report.txt"
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    manifest_path = args.manifest_path.resolve()
    stats_path = args.norm_stats_path or (manifest_path.parent / "norm_stats.npz")
    out_dir = args.output_dir or (manifest_path.parent / "eval" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt_path}")
    lit = GWSurrogateLit.load_from_checkpoint(str(args.ckpt_path), map_location="cpu")
    lit.eval()
    lit.to(device)

    ctx = args.context_patches

    ds = WaveformPatchDataset(
        manifest_path,
        split=args.split,
        patch_len=int(lit.hparams.patch_len),
        context_patches=ctx,
        norm_stats_path=stats_path,
        preload=args.preload,
        share_memory=False,
    )
    meta = pd.read_parquet(manifest_path)
    meta = meta[meta["split"] == args.split].reset_index(drop=True)

    n = len(ds) if args.n_samples <= 0 else min(args.n_samples, len(ds))
    indices = list(range(n))
    subset = Subset(ds, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
        num_workers=0,
    )

    print(f"Evaluating {n} {args.split} samples on {device} ...")
    buckets = _evaluate_split(lit, loader, device, args.rollout_max_patches)
    metrics = meta.iloc[indices].copy().reset_index(drop=True)
    for key, vals in buckets.items():
        metrics[key] = vals
    metrics.to_parquet(out_dir / "metrics.parquet", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(f"  wrote {out_dir / 'metrics.parquet'}")

    _write_report(metrics, out_dir, args.ckpt_path, args.rollout_max_patches)
    _plot_histograms(metrics, out_dir)
    _plot_mismatch_vs_params(metrics, out_dir)
    _plot_waveform_examples(
        lit, ds, meta.iloc[indices].reset_index(drop=True), metrics, device,
        out_dir, args.n_plot, args.rollout_max_patches, args.plot_t_min, args.plot_t_max,
    )

    summary = {
        "checkpoint": str(args.ckpt_path),
        "split": args.split,
        "n_samples": n,
        "mismatch_rollout_mean": float(metrics["mismatch_rollout"].mean()),
        "mismatch_rollout_median": float(metrics["mismatch_rollout"].median()),
        "mse_total_tf_mean": float(metrics["mse_total_tf"].mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
