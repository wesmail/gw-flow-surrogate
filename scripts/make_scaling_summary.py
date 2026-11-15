#!/usr/bin/env python
"""Build ``results/paper/scaling_summary.csv`` from per-run eval CSVs.

Expects files named ``results/scaling/N{N}_seed{S}.csv`` (from
``cluster/train_scaling.sbatch``). Writes one ``per_seed`` row per CSV and one
``per_n`` row per training size (geometric mean of medians over seeds).

Usage (repo root)::

    python scripts/make_scaling_summary.py
    python scripts/make_scaling_summary.py --glob 'results/scaling/N*_seed*.csv' \\
        --out results/paper/scaling_summary.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

_NAME = re.compile(r"N(?P<n>\d+)_seed(?P<seed>\d+)\.csv$")


def _median_frac(df: pd.DataFrame, col: str = "mm_pt_full", thr: float = 1e-3) -> dict:
    y = df[col].to_numpy(float)
    out = {
        f"{col}_median": float(np.median(y)),
        f"frac_lt_{thr:g}": float(np.mean(y < thr)),
    }
    if "floor_pt_full" in df.columns:
        out["floor_pt_full_median"] = float(np.median(df["floor_pt_full"].to_numpy(float)))
    if "chi_eff" in df.columns:
        m = df["chi_eff"].to_numpy(float) > 0.4
        if m.any():
            out[f"frac_lt_{thr:g}_chieff_gt0.4"] = float(np.mean(y[m] < thr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/scaling/N*_seed*.csv")
    ap.add_argument("--out", default="results/paper/scaling_summary.csv")
    a = ap.parse_args()

    paths = sorted(Path().glob(a.glob))
    paths = [p for p in paths if "_summary" not in p.name]
    if not paths:
        raise SystemExit(f"no files matched {a.glob!r}")

    per_seed: list[dict] = []
    for p in paths:
        m = _NAME.search(p.name)
        if not m:
            print(f"skip (name): {p}")
            continue
        df = pd.read_csv(p)
        if "mm_pt_full" not in df.columns:
            raise SystemExit(f"{p} missing mm_pt_full")
        row = {
            "row_type": "per_seed",
            "n_train": int(m.group("n")),
            "seed": int(m.group("seed")),
            "n_seeds": 1,
            "path": str(p),
            **_median_frac(df),
        }
        per_seed.append(row)

    if not per_seed:
        raise SystemExit("no per-seed rows parsed")

    sdf = pd.DataFrame(per_seed)
    per_n: list[dict] = []
    for n, g in sdf.groupby("n_train", sort=True):
        meds = g["mm_pt_full_median"].to_numpy(float)
        row = {
            "row_type": "per_n",
            "n_train": int(n),
            "seed": -1,
            "n_seeds": int(len(g)),
            "mm_pt_full_median": float(np.exp(np.mean(np.log(meds)))),
            "mm_pt_full_median_min": float(meds.min()),
            "mm_pt_full_median_max": float(meds.max()),
            "path": "",
        }
        if "floor_pt_full_median" in g.columns:
            row["floor_pt_full_median"] = float(g["floor_pt_full_median"].median())
        for c in g.columns:
            if c.startswith("frac_lt_"):
                row[c] = float(g[c].mean())
        per_n.append(row)

    out = pd.concat([sdf, pd.DataFrame(per_n)], ignore_index=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(sdf)} per_seed, {len(per_n)} per_n)")
    print(out[out.row_type == "per_n"][
        ["n_train", "n_seeds", "mm_pt_full_median"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
