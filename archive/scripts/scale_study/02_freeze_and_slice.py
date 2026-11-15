#!/usr/bin/env python
"""Freeze shared norm stats + residual_scale, and carve nested train manifests.

Run from the repo root (or rely on --data-dir). Typical usage::

    python scripts/scale_study/02_freeze_and_slice.py \\
        --data-dir data/scale300k \\
        --sizes 6000 12000 25000 50000 120000 300000 \\
        --results-dir results/scale_study

Writes
------
* ``manifest_train_<N>.parquet`` next to the parent ``manifest.parquet``
  (same directory as HDF5 shards — required because file_path is a basename).
  Each file keeps the first N *train* rows (Sobol prefix) plus the full
  val / test / ood splits from the parent.
* ``norm_stats.npz`` — patch-level stats from the **full** train pool.
* ``results/scale_study/frozen.json`` — residual_scale, paths, sizes, seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.datamodule import WaveformDataModule  # noqa: E402
from datasets.waveform_dataset import compute_norm_stats  # noqa: E402
from models.lightning_module import compute_residual_scale  # noqa: E402


DEFAULT_SIZES = (6000, 12000, 25000, 50000, 120000, 300000)


def slice_manifest(parent: pd.DataFrame, n_train: int) -> pd.DataFrame:
    train = parent[parent["split"] == "train"].reset_index(drop=True)
    if n_train > len(train):
        raise ValueError(
            f"Requested n_train={n_train} but parent has only {len(train)} train rows"
        )
    # Preserve Sobol order: first N rows of the train split.
    train_n = train.iloc[:n_train]
    other = parent[parent["split"] != "train"]
    out = pd.concat([train_n, other], ignore_index=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/scale300k"))
    ap.add_argument("--parent-manifest", type=Path, default=None,
                    help="Default: <data-dir>/manifest.parquet")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--patch-len", type=int, default=32)
    ap.add_argument("--results-dir", type=Path, default=Path("results/scale_study"))
    ap.add_argument("--residual-batches", type=int, default=8)
    ap.add_argument("--skip-residual-scale", action="store_true",
                    help="Only slice + norm stats (no torch residual_scale).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing subset manifests.")
    args = ap.parse_args()

    data_dir = args.data_dir
    parent_path = args.parent_manifest or (data_dir / "manifest.parquet")
    if not parent_path.is_file():
        raise SystemExit(f"Parent manifest not found: {parent_path}")

    parent = pd.read_parquet(parent_path)
    n_train_full = int((parent["split"] == "train").sum())
    print(f"Parent: {parent_path}")
    print(f"  rows={len(parent)}  train={n_train_full}  "
          f"val={(parent['split']=='val').sum()}  "
          f"test={(parent['split']=='test').sum()}  "
          f"ood={(parent['split']=='ood').sum()}")

    # --- Nested manifests ---------------------------------------------------
    written = []
    for n in args.sizes:
        out = data_dir / f"manifest_train_{n}.parquet"
        if out.exists() and not args.force:
            print(f"  skip existing {out.name}")
            written.append(str(out))
            continue
        sliced = slice_manifest(parent, n)
        sliced.to_parquet(out, index=False)
        print(f"  wrote {out.name}  train={n}  total_rows={len(sliced)}")
        written.append(str(out))

    # --- Frozen norm stats on FULL train pool -------------------------------
    stats_path = data_dir / "norm_stats.npz"
    print(f"Computing frozen norm_stats on full train ({n_train_full} rows) → {stats_path}")
    compute_norm_stats(
        parent_path,
        stats_path,
        patch_len=args.patch_len,
        max_samples=0,  # all train rows
    )

    residual_scale = None
    if not args.skip_residual_scale:
        print("Computing residual_scale on full-train DataModule …")
        dm = WaveformDataModule(
            manifest_path=str(parent_path),
            norm_stats_path=str(stats_path),
            batch_size=64,
            num_workers=0,
            patch_len=args.patch_len,
            compute_stats_if_missing=False,
            preload=True,
        )
        residual_scale = float(compute_residual_scale(dm, patch_len=args.patch_len,
                                                      n_batches=args.residual_batches))
        print(f"  residual_scale = {residual_scale:.8f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "data_dir": str(data_dir),
        "parent_manifest": str(parent_path),
        "norm_stats_path": str(stats_path),
        "n_train_pool": n_train_full,
        "train_sizes": list(args.sizes),
        "subset_manifests": written,
        "patch_len": args.patch_len,
        "residual_scale": residual_scale,
        "n_val": int((parent["split"] == "val").sum()),
        "n_test": int((parent["split"] == "test").sum()),
        "n_ood": int((parent["split"] == "ood").sum()),
        "note": (
            "All scale-study runs MUST use this norm_stats_path and residual_scale. "
            "Subset manifests share the same val/test/ood rows."
        ),
    }
    frozen_path = args.results_dir / "frozen.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"Wrote {frozen_path}")
    if residual_scale is None:
        print("WARNING: residual_scale not computed; re-run without --skip-residual-scale before training.")


if __name__ == "__main__":
    main()
