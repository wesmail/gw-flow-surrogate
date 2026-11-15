#!/usr/bin/env python
"""Re-evaluate every ``scan_1M/n*/seed*`` best checkpoint under the frozen protocol.

Picks the checkpoint with the lowest ``val_mismatch`` in its filename (never
``last.ckpt``), evaluates test rows ``[0, n_waveforms_scaling)`` via
``eval_fm_unified.py --protocol``, writes per-run CSVs + sidecars, then a
summary with geometric means over seeds.

Usage (from repo root; do not invoke until norms are recovered)::

    python scripts/eval_scaling.py \\
        --protocol results/protocol/PROTOCOL.yaml \\
        --scan-root scan_1M \\
        --out-dir results/paper/scaling

Does not train or modify checkpoints. Skips runs with no usable checkpoint
and lists them at the end.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "paper_figs"))

from protocol_utils import load_protocol  # noqa: E402

_VAL_RE = re.compile(r"val_mismatch=([0-9.]+)")
_N_RE = re.compile(r"^n(\d+)k$", re.I)
_SEED_RE = re.compile(r"^seed(\d+)$", re.I)


def parse_n_train(folder_name: str) -> int:
    m = _N_RE.match(folder_name)
    if not m:
        raise ValueError(f"cannot parse n_train from {folder_name!r}")
    return int(m.group(1)) * 1000


def best_checkpoint(ckpt_dir: Path) -> tuple[Path | None, float | None]:
    """Lowest ``val_mismatch`` in filename; exclude ``last.ckpt``."""
    if not ckpt_dir.is_dir():
        return None, None
    best_path, best_val = None, None
    for p in ckpt_dir.glob("*.ckpt"):
        if p.name == "last.ckpt":
            continue
        m = _VAL_RE.search(p.name)
        if not m:
            continue
        v = float(m.group(1))
        if best_val is None or v < best_val:
            best_val, best_path = v, p
    return best_path, best_val


def geometric_mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def summarize_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    out = {
        "mm_pt_full_median": float(df["mm_pt_full"].median()),
        "mm_pt_window_median": float(df["mm_pt_window"].median()),
        "floor_pt_full_median": float(df["floor_pt_full"].median()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="results/protocol/PROTOCOL.yaml")
    ap.add_argument("--scan-root", default="scan_1M")
    ap.add_argument("--out-dir", default="results/paper/scaling")
    ap.add_argument("--summary", default="results/paper/scaling_summary.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--norm-stats", default=None,
                    help="optional shared override; else each run's hparams.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="list planned runs / skips without calling eval")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    proto = load_protocol(args.protocol)
    n_wave = int(proto["n_waveforms_scaling"])
    scan = Path(args.scan_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    per_seed_rows: list[dict] = []

    n_dirs = sorted(p for p in scan.glob("n*") if p.is_dir())
    if not n_dirs:
        print(f"no n* directories under {scan}; nothing to do")
        return

    for n_dir in n_dirs:
        try:
            n_train = parse_n_train(n_dir.name)
        except ValueError as exc:
            print(f"skip {n_dir}: {exc}")
            skipped.append(str(n_dir))
            continue
        seed_dirs = sorted(p for p in n_dir.glob("seed*") if p.is_dir())
        for seed_dir in seed_dirs:
            m = _SEED_RE.match(seed_dir.name)
            if not m:
                skipped.append(str(seed_dir))
                continue
            seed = int(m.group(1))
            ckpt, val_mm = best_checkpoint(seed_dir / "checkpoints")
            run_id = f"{n_dir.name}/{seed_dir.name}"
            if ckpt is None:
                print(f"SKIP {run_id}: no checkpoint with val_mismatch in filename")
                skipped.append(run_id)
                continue

            out_csv = out_dir / f"n{n_train}_s{seed}.csv"
            print(f"EVAL {run_id} -> {out_csv.name}  "
                  f"ckpt={ckpt.name}  val_mismatch={val_mm}")
            if args.dry_run:
                per_seed_rows.append(dict(
                    row_type="per_seed",
                    n_train=n_train,
                    seed=seed,
                    run_id=run_id,
                    checkpoint=str(ckpt),
                    val_mismatch_filename=val_mm,
                    mm_pt_full_median=float("nan"),
                    mm_pt_window_median=float("nan"),
                    floor_pt_full_median=float("nan"),
                    csv=str(out_csv),
                    skipped=False,
                ))
                continue

            cmd = [
                args.python, str(_ROOT / "eval_fm_unified.py"),
                "--protocol", str(Path(args.protocol).resolve()),
                "--checkpoint", str(ckpt.resolve()),
                "--n-waveforms", str(n_wave),
                "--out", str(out_csv),
            ]
            if args.norm_stats:
                cmd.extend(["--norm-stats", str(args.norm_stats)])
            if args.device:
                cmd.extend(["--device", args.device])
            # --n-waveforms is not a protocol conflict key; scaling size comes
            # from protocol["n_waveforms_scaling"] and is passed explicitly.
            subprocess.check_call(cmd, cwd=_ROOT)

            stats = summarize_csv(out_csv)
            per_seed_rows.append(dict(
                row_type="per_seed",
                n_train=n_train,
                seed=seed,
                run_id=run_id,
                checkpoint=str(ckpt),
                val_mismatch_filename=val_mm,
                **stats,
                csv=str(out_csv),
                skipped=False,
            ))

    # Aggregate per N.
    summary_rows = list(per_seed_rows)
    if per_seed_rows and not args.dry_run:
        df = pd.DataFrame(per_seed_rows)
        for n_train, g in df.groupby("n_train"):
            meds = g["mm_pt_full_median"].tolist()
            wins = g["mm_pt_window_median"].tolist()
            floors = g["floor_pt_full_median"].tolist()
            summary_rows.append(dict(
                row_type="per_n",
                n_train=int(n_train),
                seed="",
                run_id=f"n{int(n_train)}",
                checkpoint="",
                val_mismatch_filename="",
                mm_pt_full_median=geometric_mean(meds),
                mm_pt_window_median=geometric_mean(wins),
                floor_pt_full_median=float(np.median(floors)),
                mm_pt_full_min=float(np.min(meds)),
                mm_pt_full_max=float(np.max(meds)),
                n_seeds=int(len(g)),
                csv="",
                skipped=False,
            ))
    elif per_seed_rows and args.dry_run:
        # still emit per_n skeleton for dry-run visibility
        by_n: dict[int, list] = {}
        for r in per_seed_rows:
            by_n.setdefault(int(r["n_train"]), []).append(r)
        for n_train, rs in sorted(by_n.items()):
            summary_rows.append(dict(
                row_type="per_n",
                n_train=n_train,
                seed="",
                run_id=f"n{n_train}",
                checkpoint="",
                val_mismatch_filename="",
                mm_pt_full_median=float("nan"),
                mm_pt_window_median=float("nan"),
                floor_pt_full_median=float("nan"),
                mm_pt_full_min=float("nan"),
                mm_pt_full_max=float("nan"),
                n_seeds=len(rs),
                csv="",
                skipped=False,
            ))

    # Write summary (even if empty / dry-run).
    fieldnames = [
        "row_type", "n_train", "seed", "run_id", "checkpoint",
        "val_mismatch_filename", "mm_pt_full_median", "mm_pt_window_median",
        "floor_pt_full_median", "mm_pt_full_min", "mm_pt_full_max", "n_seeds",
        "csv", "skipped",
    ]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)

    print(f"\nsummary -> {summary_path}  ({len(per_seed_rows)} per-seed rows)")
    if skipped:
        print("SKIPPED runs (no usable checkpoint):")
        for s in skipped:
            print(f"  - {s}")
    else:
        print("no skipped runs")


if __name__ == "__main__":
    main()
