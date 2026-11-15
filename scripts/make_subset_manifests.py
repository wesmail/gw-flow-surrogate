#!/usr/bin/env python
"""Nested training subsets of ONE generated pool, for the data-scaling study.

For every N it writes  <pool>/manifest_train_<N>.parquet  containing
  * ALL validation and test (and ood) rows of the pool manifest, unchanged
  * the FIRST N training rows in generation order
so  N1 < N2  implies  train(N1) is a subset of train(N2).

Generation order = (shard number, row_index) unless the manifest has an explicit
index column (sobol_index / gen_index / global_index), which is then used.
Because training points come from a scrambled Sobol sequence, the first N
points are themselves evenly spread over the parameter box (printed below).

Waveform files are NOT copied: the new manifests point to the same shards.
Use the SAME norm_stats.npz (of the full pool) and residual_scale for every N.

Usage (repo root):
    python scripts/make_subset_manifests.py --manifest data/n1000k/manifest.parquet \\
        --sizes 16000 40000 100000 251000 631000 1000000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ORDER_COLS = ("sobol_index", "gen_index", "global_index")


def shard_number(path: str) -> int:
    m = re.findall(r"(\d+)", Path(str(path)).stem)
    return int(m[-1]) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", default=None, help="default: next to the pool manifest")
    a = ap.parse_args()

    man = pd.read_parquet(a.manifest)
    if "split" not in man:
        raise ValueError("manifest has no 'split' column")
    train = man[man.split == "train"].copy()
    rest = man[man.split != "train"]
    print(f"pool: {len(train)} train rows; kept unchanged in every subset: "
          + ", ".join(f"{k}={v}" for k, v in rest.split.value_counts().items()))

    order_col = next((c for c in ORDER_COLS if c in train), None)
    if order_col:
        train = train.sort_values(order_col, kind="stable")
        print(f"generation order from column '{order_col}'")
    else:
        train["_shard"] = train["file_path"].map(shard_number)
        train = train.sort_values(["_shard", "row_index"], kind="stable").drop(columns="_shard")
        print("generation order from (shard number, row_index)")

    out_dir = Path(a.out_dir) if a.out_dir else Path(a.manifest).parent
    sizes = sorted(set(a.sizes))
    if sizes[-1] > len(train):
        raise ValueError(f"largest size {sizes[-1]} > available train rows {len(train)}")
    prev = None
    cols = [c for c in ("q", "chi1z", "chi2z") if c in train]
    for n in sizes:
        sub = train.iloc[:n]
        keys = set(zip(sub.file_path, sub.row_index))
        if prev is not None and not prev <= keys:
            raise AssertionError("subsets are not nested")   # cannot happen with a prefix
        prev = keys
        out = pd.concat([sub, rest], ignore_index=True)
        p = out_dir / f"manifest_train_{n}.parquet"
        out.to_parquet(p, index=False)
        span = "  ".join(f"{c}[{sub[c].min():.2f},{sub[c].max():.2f}] mean {sub[c].mean():+.2f}"
                         for c in cols)
        print(f"N={n:>8d}: {p.name}  ({len(out)} rows)  {span}")
    print("\nnested: OK. Same val/test rows in every file. Point every run at the SAME norm_stats.npz.")


if __name__ == "__main__":
    main()
