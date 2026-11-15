#!/usr/bin/env python
"""Generate the paper evaluation set ``data/test5k`` (test + OOD only).

Does **not** edit ``data_generation.generate.generate_dataset``. Calls
``uniform_parameters`` / ``extrapolation_parameters`` and ``_write_shard``
directly so ``--n-train 0`` works cleanly and **no** ``norm_stats.*`` is written
(test data never defines normalisation).

Default seed **7919** — unused by scan_1M (42/43/44) or scale-study (42).
Grid matches the audited training defaults: dt=0.5, t in [-5000, 130].
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data_generation.config import GenerationConfig, time_grid  # noqa: E402
from data_generation.generate import _write_shard, combine_manifests  # noqa: E402
from data_generation.sampling import extrapolation_parameters, uniform_parameters  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("generate_test5k")

# Recorded training grid from Phase 0 audit (GenerationConfig defaults).
TRAIN_T_MIN, TRAIN_T_MAX, TRAIN_DT = -5000.0, 130.0, 0.5
TRAIN_T = time_grid(TRAIN_T_MIN, TRAIN_T_MAX, TRAIN_DT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("data/test5k"))
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7919)
    ap.add_argument("--dt", type=float, default=TRAIN_DT)
    ap.add_argument("--t-min", type=float, default=TRAIN_T_MIN)
    ap.add_argument("--t-max", type=float, default=TRAIN_T_MAX)
    ap.add_argument("--n-files", type=int, default=1)
    args = ap.parse_args()

    out = args.output_dir
    if (out / "manifest.parquet").is_file():
        raise SystemExit(f"{out}/manifest.parquet already exists — refuse to overwrite")

    cfg = GenerationConfig(
        output_dir=str(out),
        n_files=args.n_files,
        seed=args.seed,
        dt=args.dt,
        t_min=args.t_min,
        t_max=args.t_max,
        waveforms_per_file=args.n_test + args.n_ood,
    )
    if not np.array_equal(cfg.times, TRAIN_T):
        raise RuntimeError(
            f"Configured time grid differs from audited training grid.\n"
            f"  cfg: n={cfg.t_full} t[0]={cfg.times[0]} t[-1]={cfg.times[-1]} dt={cfg.dt}\n"
            f"  train: n={TRAIN_T.size} t[0]={TRAIN_T[0]} t[-1]={TRAIN_T[-1]} dt={TRAIN_DT}"
        )

    rng = np.random.default_rng(args.seed)
    params_test = uniform_parameters(cfg, args.n_test, rng)
    params_ood = extrapolation_parameters(args.n_ood, rng)
    all_params = np.vstack([params_test, params_ood])
    splits = ["test"] * args.n_test + ["ood"] * args.n_ood

    log.info("Generating %d test + %d ood → %s (seed=%d)", args.n_test, args.n_ood, out, args.seed)
    total = all_params.shape[0]
    per_file = int(np.ceil(total / cfg.n_files))
    for file_idx in range(cfg.n_files):
        start = file_idx * per_file
        end = min(start + per_file, total)
        if start >= total:
            break
        _write_shard(file_idx, cfg, all_params[start:end], splits[start:end])

    manifest = combine_manifests(out)
    # Hard rule: test set must not define normalisation.
    for junk in ("norm_stats.npz", "norm_stats.json"):
        p = out / junk
        if p.is_file():
            p.unlink()
            log.warning("Removed unexpected %s", p)

    # ---- checks ----------------------------------------------------------- #
    with h5py.File(out / "waveforms_00000.h5", "r") as f:
        t = np.asarray(f["t"][:], dtype=np.float64)
        peak = np.asarray(f["peak_index"][:])
        la = f["log_a"][:]
        phi = f["phi"][:]
    if not np.array_equal(t, TRAIN_T):
        raise RuntimeError("P11 fail: written t array != audited training grid")
    man = pd.read_parquet(manifest)
    test = man[man["split"] == "test"]
    ood = man[man["split"] == "ood"]
    assert len(test) == args.n_test, (len(test), args.n_test)
    assert len(ood) == args.n_ood
    assert ((test["q"] >= 1) & (test["q"] <= 8)).all()
    assert ((test["chi1z"].abs() <= 0.8) & (test["chi2z"].abs() <= 0.8)).all()
    if not np.isfinite(la).all() or not np.isfinite(phi).all():
        raise RuntimeError("NaN/inf in tracks")
    # peak index should be identical after alignment (same grid → same peak sample)
    if peak.min() != peak.max():
        log.warning("peak_index not constant: min=%d max=%d (report in WORKLOG)", peak.min(), peak.max())
    else:
        log.info("peak_index constant = %d", int(peak[0]))

    meta = dict(
        output_dir=str(out),
        n_test=args.n_test,
        n_ood=args.n_ood,
        seed=args.seed,
        dt=args.dt,
        t_min=args.t_min,
        t_max=args.t_max,
        n_time=int(t.size),
        t0=float(t[0]),
        t_end=float(t[-1]),
        grid_matches_train_audit=True,
        note="No norm_stats written. Do not place norm_stats.npz here.",
    )
    (out / "TEST5K_META.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s (%d rows). Meta → %s", manifest, len(man), out / "TEST5K_META.json")


if __name__ == "__main__":
    main()
