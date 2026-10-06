#!/usr/bin/env python
"""CLI for waveform dataset generation.

Smoke test:
    python -m data_generation.cli generate \\
        --output-dir data/smoke \\
        --n-files 1 \\
        --waveforms-per-file 8 \\
        --n-train 6 --n-test 1 --n-ood 1
"""

from __future__ import annotations

import argparse
import logging

from data_generation.config import GenerationConfig
from data_generation.generate import combine_manifests, generate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate HDF5 shards + manifest")
    gen.add_argument("--output-dir", default="data/generated")
    gen.add_argument("--n-files", type=int, default=32)
    gen.add_argument("--waveforms-per-file", type=int, default=2_000)
    gen.add_argument("--n-train", type=int, default=None)
    gen.add_argument("--n-test", type=int, default=1_000)
    gen.add_argument("--n-ood", type=int, default=500)
    gen.add_argument("--val-fraction", type=float, default=0.1)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--dt", type=float, default=0.5)
    gen.add_argument("--t-min", type=float, default=-5000.0)
    gen.add_argument("--t-max", type=float, default=130.0)
    gen.add_argument("--q-min", type=float, default=1.0)
    gen.add_argument("--q-max", type=float, default=8.0)
    gen.add_argument("--chi-min", type=float, default=-0.8)
    gen.add_argument("--chi-max", type=float, default=0.8)
    gen.add_argument("--jobs", type=int, default=1)
    gen.add_argument(
        "--show-surrogate-warnings",
        action="store_true",
        help="Show gwsurrogate 'outside training range' warnings (noisy for OOD samples).",
    )

    comb = sub.add_parser("combine", help="Combine per-shard manifests")
    comb.add_argument("--output-dir", required=True)
    comb.add_argument("--name", default="manifest.parquet")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        cfg = GenerationConfig(
            output_dir=args.output_dir,
            n_files=args.n_files,
            waveforms_per_file=args.waveforms_per_file,
            seed=args.seed,
            dt=args.dt,
            t_min=args.t_min,
            t_max=args.t_max,
            q_min=args.q_min,
            q_max=args.q_max,
            chi_min=args.chi_min,
            chi_max=args.chi_max,
            suppress_surrogate_warnings=not args.show_surrogate_warnings,
        )
        generate_dataset(
            cfg,
            n_train=args.n_train,
            n_test=args.n_test,
            n_ood=args.n_ood,
            val_fraction=args.val_fraction,
            jobs=args.jobs,
        )
    elif args.command == "combine":
        combine_manifests(args.output_dir, args.name)


if __name__ == "__main__":
    main()
