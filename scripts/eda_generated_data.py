#!/usr/bin/env python
"""Exploratory analysis for generated GWSurrogate HDF5 datasets.

Reads manifest.parquet (+ optional norm_stats.json) and produces plots under
<data-dir>/eda/ by default.

Example:
    python scripts/eda_generated_data.py --data-dir data/generated
    python scripts/eda_generated_data.py --data-dir data/generated --n-examples 6 --n-envelope 200
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

SPLITS = ("train", "val", "test", "ood")
SPLIT_COLORS = {
    "train": "#2563eb",
    "val": "#16a34a",
    "test": "#ca8a04",
    "ood": "#dc2626",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def load_manifest(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "manifest.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    if not df["valid"].all():
        df = df[df["valid"]].copy()
    return df


def load_norm_stats(data_dir: Path) -> dict | None:
    path = data_dir / "norm_stats.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def open_first_h5(data_dir: Path, manifest: pd.DataFrame) -> tuple[Path, dict]:
    h5_path = Path(manifest.iloc[0]["file_path"])
    if not h5_path.is_file():
        h5_path = data_dir / h5_path.name
    with h5py.File(h5_path, "r") as f:
        attrs = {k: (json.loads(v) if k == "modes" and isinstance(v, str) else v) for k, v in f.attrs.items()}
        times = f["t"][:]
    return h5_path, {"attrs": attrs, "times": times}


def load_waveform_row(h5_path: Path, row_index: int) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        return {
            "t": f["t"][:],
            "log_a": f["log_a"][row_index, 0],
            "phi": f["phi"][row_index, 0],
            "h_re": f["h_re"][row_index, 0],
            "h_im": f["h_im"][row_index, 0],
            "theta": f["theta"][row_index],
        }


def plot_split_counts(manifest: pd.DataFrame, out_dir: Path) -> None:
    counts = manifest["split"].value_counts().reindex([s for s in SPLITS if s in manifest["split"].values])
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [SPLIT_COLORS.get(s, "gray") for s in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Samples per split")
    ax.set_xlabel("split")
    ax.set_ylabel("count")
    ax.bar_label(ax.containers[0])
    _save(fig, out_dir, "01_split_counts.png")


def plot_parameter_histograms(manifest: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    params = [("q", "mass ratio q"), ("chi1z", r"$\chi_{1z}$"), ("chi2z", r"$\chi_{2z}$")]
    for ax, (col, label) in zip(axes, params):
        for split in manifest["split"].unique():
            sub = manifest[manifest["split"] == split]
            ax.hist(
                sub[col],
                bins=40,
                alpha=0.45,
                density=True,
                label=split,
                color=SPLIT_COLORS.get(split, "gray"),
                histtype="stepfilled",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Parameter distributions by split", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "02_parameter_histograms.png")


def plot_parameter_scatter(manifest: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    pairs = [("q", "chi1z"), ("chi1z", "chi2z")]
    for ax, (x, y) in zip(axes, pairs):
        for split in manifest["split"].unique():
            sub = manifest[manifest["split"] == split]
            ax.scatter(
                sub[x], sub[y], s=6, alpha=0.35, label=split, c=SPLIT_COLORS.get(split, "gray")
            )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.axvline(8.0, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0.8, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-0.8, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.legend(fontsize=8, markerscale=2)
    fig.suptitle("Parameter coverage (dashed: NRHybSur nominal box edges)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "03_parameter_scatter.png")


def plot_metadata_panel(meta: dict, norm_stats: dict | None, manifest: pd.DataFrame, out_dir: Path) -> None:
    attrs = meta["attrs"]
    times = meta["times"]
    lines = [
        f"N manifest rows: {len(manifest):,}",
        f"Unique HDF5 shards: {manifest['file_path'].nunique()}",
        f"Surrogate: {attrs.get('surrogate', '?')}",
        f"Storage format: {attrs.get('storage_format', '?')}",
        f"t/M ∈ [{times[0]:.1f}, {times[-1]:.1f}], dt={float(attrs.get('dt_M', np.median(np.diff(times)))):.3g} M",
        f"T = {len(times)} samples, peak index (median) = {manifest['peak_index'].median():.0f}",
        f"Modes: {attrs.get('modes', '?')}",
    ]
    if norm_stats:
        la = norm_stats.get("log_a", {})
        dp = norm_stats.get("dphi", {})
        lines.append(f"norm_stats log_a mean/std: {la.get('mean')} / {la.get('std')}")
        lines.append(f"norm_stats dphi mean/std: {dp.get('mean')} / {dp.get('std')}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(
        0.02, 0.98, "\n".join(lines),
        va="top", ha="left", fontsize=11, family="monospace",
        transform=ax.transAxes,
    )
    ax.set_title("Dataset summary")
    _save(fig, out_dir, "00_dataset_summary.png")


def plot_example_waveforms(
    manifest: pd.DataFrame, out_dir: Path, n_per_split: int, rng: np.random.Generator
) -> None:
    """One panel per split: |h|, log A, φ for a random example."""
    fig = plt.figure(figsize=(14, 3 * len(SPLITS)))
    gs = GridSpec(len(SPLITS), 3, figure=fig, hspace=0.45, wspace=0.28)

    for row, split in enumerate(SPLITS):
        sub = manifest[manifest["split"] == split]
        if sub.empty:
            continue
        pick = sub.sample(n=min(n_per_split, len(sub)), random_state=int(rng.integers(1e9))).iloc[0]
        wf = load_waveform_row(Path(pick["file_path"]), int(pick["row_index"]))
        t = wf["t"]
        h = wf["h_re"] + 1j * wf["h_im"]
        amp = np.abs(h)

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.plot(t, amp, color=SPLIT_COLORS.get(split, "C0"), lw=0.8)
        ax0.axvline(0, color="k", ls="--", alpha=0.4)
        ax0.set_ylabel("|h₂₂|")
        ax0.set_title(f"{split}: q={pick['q']:.2f}, χ₁={pick['chi1z']:.2f}, χ₂={pick['chi2z']:.2f}")

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.plot(t, wf["log_a"], color="#7c3aed", lw=0.8)
        ax1.axvline(0, color="k", ls="--", alpha=0.4)
        ax1.set_ylabel("log A")

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.plot(t, wf["phi"], color="#0891b2", lw=0.8)
        ax2.axvline(0, color="k", ls="--", alpha=0.4)
        ax2.set_ylabel(r"$\phi$ (rad)")

        if row == len(SPLITS) - 1:
            for ax in (ax0, ax1, ax2):
                ax.set_xlabel(r"$t/M$")

    fig.suptitle("Example waveforms per split (mode 22)", y=1.01)
    _save(fig, out_dir, "04_example_waveforms.png")


def plot_merger_zoom(manifest: pd.DataFrame, out_dir: Path, rng: np.random.Generator) -> None:
    """Zoom on merger+ringdown region for random train examples."""
    sub = manifest[manifest["split"] == "train"]
    if sub.empty:
        return
    picks = sub.sample(n=min(5, len(sub)), random_state=int(rng.integers(1e9)))
    fig, ax = plt.subplots(figsize=(10, 4))
    for _, row in picks.iterrows():
        wf = load_waveform_row(Path(row["file_path"]), int(row["row_index"]))
        mask = (wf["t"] >= -50) & (wf["t"] <= 80)
        h = wf["h_re"][mask] + 1j * wf["h_im"][mask]
        ax.plot(wf["t"][mask], np.abs(h), alpha=0.75, lw=1)
    ax.axvline(0, color="k", ls="--", alpha=0.5, label="t=0 (peak)")
    ax.set_xlabel(r"$t/M$")
    ax.set_ylabel("|h₂₂|")
    ax.set_title("Merger + ringdown zoom (random train samples)")
    ax.legend()
    _save(fig, out_dir, "05_merger_zoom.png")


def plot_amplitude_envelope(
    manifest: pd.DataFrame, out_dir: Path, n_samples: int, rng: np.random.Generator
) -> None:
    """Median and 10–90% band of |h| over a train subsample."""
    sub = manifest[manifest["split"] == "train"]
    if sub.empty:
        sub = manifest
    picks = sub.sample(n=min(n_samples, len(sub)), random_state=int(rng.integers(1e9)))
    amps = []
    t_ref = None
    for _, row in picks.iterrows():
        wf = load_waveform_row(Path(row["file_path"]), int(row["row_index"]))
        if t_ref is None:
            t_ref = wf["t"]
        h = wf["h_re"] + 1j * wf["h_im"]
        amps.append(np.abs(h))
    stack = np.stack(amps, axis=0)
    med = np.median(stack, axis=0)
    p10, p90 = np.percentile(stack, [10, 90], axis=0)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(t_ref, p10, p90, alpha=0.25, color="#2563eb", label="10–90% band")
    ax.plot(t_ref, med, color="#1e40af", lw=1.2, label="median")
    ax.axvline(0, color="k", ls="--", alpha=0.4)
    ax.set_xlabel(r"$t/M$")
    ax.set_ylabel("|h₂₂|")
    ax.set_title(f"Train amplitude envelope (n={len(picks)} subsample)")
    ax.legend()
    _save(fig, out_dir, "06_amplitude_envelope.png")


def plot_phase_increments(manifest: pd.DataFrame, out_dir: Path, n_samples: int, rng: np.random.Generator) -> None:
    """Distribution of per-sample Δφ on the time grid (train subsample)."""
    sub = manifest[manifest["split"] == "train"]
    if sub.empty:
        sub = manifest
    picks = sub.sample(n=min(n_samples, len(sub)), random_state=int(rng.integers(1e9)))
    dphi = []
    for _, row in picks.iterrows():
        wf = load_waveform_row(Path(row["file_path"]), int(row["row_index"]))
        dphi.append(np.diff(wf["phi"]))
    cat = np.concatenate(dphi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(cat, bins=80, color="#0891b2", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel(r"$\Delta\phi$ per time step (rad)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Phase increment distribution (train subsample)")

    # Merger-region only
    wf0 = load_waveform_row(Path(picks.iloc[0]["file_path"]), int(picks.iloc[0]["row_index"]))
    merger_mask = (wf0["t"][:-1] >= -20) & (wf0["t"][:-1] <= 20)
    merger_dphi = []
    for _, row in picks.iterrows():
        wf = load_waveform_row(Path(row["file_path"]), int(row["row_index"]))
        merger_dphi.append(np.diff(wf["phi"])[merger_mask])
    merger_cat = np.concatenate(merger_dphi)
    axes[1].hist(merger_cat, bins=60, color="#dc2626", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel(r"$\Delta\phi$ per step, $t\in[-20,20]M$")
    axes[1].set_title("Merger-region increments")
    fig.tight_layout()
    _save(fig, out_dir, "07_phase_increments.png")


def plot_gauge_and_peak_checks(manifest: pd.DataFrame, meta: dict, out_dir: Path, n_samples: int, rng: np.random.Generator) -> None:
    """ref_phase distribution + φ(t_ref) after storage should be ~0."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(manifest["ref_phase"], bins=50, color="#6366f1", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel(r"stored ref_phase ($\alpha$)")
    axes[0].set_title("Applied gauge rotation α")

    t_ref = float(meta["attrs"].get("t_ref", -5000.0))
    sub = manifest[manifest["split"] == "train"].sample(
        n=min(n_samples, len(manifest)), random_state=int(rng.integers(1e9))
    )
    phi_at_ref = []
    for _, row in sub.iterrows():
        wf = load_waveform_row(Path(row["file_path"]), int(row["row_index"]))
        idx = int(np.argmin(np.abs(wf["t"] - t_ref)))
        phi_at_ref.append(wf["phi"][idx])
    wrapped = (np.array(phi_at_ref) + np.pi) % (2 * np.pi) - np.pi
    axes[1].hist(wrapped, bins=40, color="#16a34a", edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel(r"$\phi_{22}(t_{\mathrm{ref}})$ (wrapped)")
    axes[1].set_title(f"Gauge check at t_ref={t_ref:.0f} M (should peak at 0)")
    fig.tight_layout()
    _save(fig, out_dir, "08_gauge_and_peak.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(manifest["peak_index"], bins=40, color="#f59e0b", edgecolor="white", linewidth=0.3)
    peak_t = meta["times"][manifest["peak_index"].astype(int).clip(0, len(meta["times"]) - 1)]
    ax2 = ax.twinx()
    ax2.hist(peak_t, bins=40, alpha=0.35, color="#b45309")
    ax.set_xlabel("peak_index")
    ax.set_ylabel("count (index)")
    ax2.set_ylabel(f"count (t/M at index)")
    ax.set_title("Merger peak index / time (post alignment)")
    _save(fig, out_dir, "09_peak_index.png")


def write_text_report(
    manifest: pd.DataFrame, meta: dict, norm_stats: dict | None, out_dir: Path
) -> None:
    lines = ["GWSurrogate dataset EDA report", "=" * 40, ""]
    for split in SPLITS:
        sub = manifest[manifest["split"] == split]
        if sub.empty:
            continue
        lines.append(
            f"{split:5s}  n={len(sub):6d}  "
            f"q [{sub['q'].min():.3f}, {sub['q'].max():.3f}]  "
            f"χ1z [{sub['chi1z'].min():.3f}, {sub['chi1z'].max():.3f}]  "
            f"χ2z [{sub['chi2z'].min():.3f}, {sub['chi2z'].max():.3f}]"
        )
    lines.append("")
    lines.append("HDF5 attributes:")
    for k, v in meta["attrs"].items():
        if k == "generation_config":
            continue
        lines.append(f"  {k}: {v}")
    if norm_stats:
        lines.append("")
        lines.append("norm_stats.json:")
        lines.append(textwrap.indent(json.dumps(norm_stats, indent=2), "  "))
    path = out_dir / "report.txt"
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/generated"), help="Dataset root with manifest.parquet")
    p.add_argument("--output-dir", type=Path, default=None, help="Plot output (default: <data-dir>/eda)")
    p.add_argument("--n-examples", type=int, default=1, help="Random waveform examples per split")
    p.add_argument("--n-envelope", type=int, default=150, help="Train subsample size for envelope / Δφ plots")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = (args.output_dir or data_dir / "eda").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {data_dir}")
    manifest = load_manifest(data_dir)
    norm_stats = load_norm_stats(data_dir)
    _, meta = open_first_h5(data_dir, manifest)
    meta["times"] = np.asarray(meta["times"])
    rng = np.random.default_rng(args.seed)

    print(f"Generating EDA plots → {out_dir}")
    write_text_report(manifest, meta, norm_stats, out_dir)
    plot_metadata_panel(meta, norm_stats, manifest, out_dir)
    plot_split_counts(manifest, out_dir)
    plot_parameter_histograms(manifest, out_dir)
    plot_parameter_scatter(manifest, out_dir)
    plot_example_waveforms(manifest, out_dir, args.n_examples, rng)
    plot_merger_zoom(manifest, out_dir, rng)
    plot_amplitude_envelope(manifest, out_dir, args.n_envelope, rng)
    plot_phase_increments(manifest, out_dir, args.n_envelope, rng)
    plot_gauge_and_peak_checks(manifest, meta, out_dir, min(args.n_envelope, len(manifest)), rng)

    print("Done.")


if __name__ == "__main__":
    main()
