#!/usr/bin/env python
"""Aggregate scale-study eval CSVs, fit power laws, write prediction.json.

Expects per-run CSVs produced by ``04_eval_one.sh`` / ``eval_fm_unified.py``::

    results/scale_study/eval_<run_id>.csv

Usage::

    # After the five scaling evals (6k…120k), write the prediction BEFORE 300k:
    python scripts/scale_study/04_eval_and_fit.py \\
        --results-dir results/scale_study \\
        --fit-phases scaling \\
        --write-prediction

    # After everything, refresh summary + include heldout / capacity:
    python scripts/scale_study/04_eval_and_fit.py --results-dir results/scale_study
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MISMATCH_DEFAULT = "mismatch_phase_time_future"


def _load_runs_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("run_id"):
                headers = line.split("\t")
                continue
            parts = line.split("\t")
            rows.append(dict(zip(headers, parts)))
    return rows


def _median_mismatch(csv_path: Path, col: str) -> dict:
    vals = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals.append(float(row[col]))
    v = np.asarray(vals, dtype=float)
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
    }


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """y = A * x^B  via log-log OLS."""
    lx, ly = np.log(x), np.log(y)
    B, logA = np.polyfit(lx, ly, 1)
    A = float(np.exp(logA))
    yhat = A * np.power(x, B)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"A": A, "B": float(B), "r2": r2, "form": "A * N**B"}


def _fit_power_law_with_floor(x: np.ndarray, y: np.ndarray) -> dict:
    """y = A * x^B + C, C >= 0. Grid-search C then OLS on (y-C)."""
    best = None
    # Floor candidates from near-zero up to 90% of the smallest y.
    y_min = float(np.min(y))
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                np.geomspace(max(y_min * 1e-4, 1e-8), y_min * 0.9, 40),
            ]
        )
    )
    for C in candidates:
        y_adj = y - C
        if np.any(y_adj <= 0):
            continue
        fit = _fit_power_law(x, y_adj)
        yhat = fit["A"] * np.power(x, fit["B"]) + C
        ss_res = float(np.sum((y - yhat) ** 2))
        if best is None or ss_res < best["ss_res"]:
            best = {
                "A": fit["A"],
                "B": fit["B"],
                "C": float(C),
                "ss_res": ss_res,
                "form": "A * N**B + C",
            }
    if best is None:
        plain = _fit_power_law(x, y)
        plain["C"] = 0.0
        plain["form"] = "A * N**B + C"
        plain["ss_res"] = float("nan")
        return plain
    yhat = best["A"] * np.power(x, best["B"]) + best["C"]
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    best["r2"] = 1.0 - best["ss_res"] / ss_tot if ss_tot > 0 else float("nan")
    del best["ss_res"]
    return best


def _predict_n(fit: dict, target: float) -> float | None:
    """Invert y(N)=target. Returns None if unreachable (floor >= target)."""
    A, B = fit["A"], fit["B"]
    C = fit.get("C", 0.0)
    if target <= C:
        return None
    # target = A * N^B + C  →  N = ((target - C) / A) ** (1/B)
    if A <= 0 or B == 0:
        return None
    ratio = (target - C) / A
    if ratio <= 0:
        return None
    return float(ratio ** (1.0 / B))


def _predict_y(fit: dict, n: float) -> float:
    return float(fit["A"] * (n ** fit["B"]) + fit.get("C", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/scale_study"))
    ap.add_argument("--runs-tsv", type=Path, default=Path("scripts/scale_study/runs.tsv"))
    ap.add_argument("--mismatch-col", default=MISMATCH_DEFAULT)
    ap.add_argument("--fit-phases", nargs="+", default=["scaling"],
                    help="Which phases enter the power-law fit (default: scaling only).")
    ap.add_argument("--exclude-run-ids", nargs="*", default=[],
                    help="Run ids to drop from the fit (e.g. hold out 300k by phase instead).")
    ap.add_argument("--target-mismatch", type=float, default=1e-4)
    ap.add_argument("--predict-at-n", type=int, default=300000)
    ap.add_argument("--write-prediction", action="store_true",
                    help="Write prediction.json (required gate before evaluating heldout 300k).")
    args = ap.parse_args()

    runs = _load_runs_tsv(args.runs_tsv)
    summary_rows = []
    for r in runs:
        run_id = r["run_id"]
        csv_path = args.results_dir / f"eval_{run_id}.csv"
        row = {
            "run_id": run_id,
            "n_train": int(r["n_train"]),
            "model_tag": r["model_tag"],
            "phase": r["phase"],
            "csv": str(csv_path) if csv_path.is_file() else "",
            "median": None,
            "p95": None,
            "max": None,
        }
        if csv_path.is_file():
            stats = _median_mismatch(csv_path, args.mismatch_col)
            row.update({k: stats[k] for k in ("median", "p95", "max", "n", "mean")})
            print(f"{run_id:16s} N={row['n_train']:<7d}  "
                  f"median={stats['median']:.4e}  p95={stats['p95']:.4e}  max={stats['max']:.4e}")
        else:
            print(f"{run_id:16s}  (no CSV yet)")
        summary_rows.append(row)

    summary_path = args.results_dir / "summary.csv"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "n_train", "model_tag", "phase", "median", "p95", "max", "mean", "n", "csv"]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"Wrote {summary_path}")

    # --- Power-law fit on selected points with measured medians -------------
    fit_points = [
        r for r in summary_rows
        if r["phase"] in args.fit_phases
        and r["run_id"] not in args.exclude_run_ids
        and r["median"] is not None
    ]
    if len(fit_points) < 2:
        print("Not enough points to fit a power law yet.")
        return

    x = np.array([r["n_train"] for r in fit_points], dtype=float)
    y = np.array([r["median"] for r in fit_points], dtype=float)
    fit_plain = _fit_power_law(x, y)
    fit_floor = _fit_power_law_with_floor(x, y)

    pred_plain_y = _predict_y(fit_plain, args.predict_at_n)
    pred_floor_y = _predict_y(fit_floor, args.predict_at_n)
    n_star_plain = _predict_n(fit_plain, args.target_mismatch)
    n_star_floor = _predict_n(fit_floor, args.target_mismatch)

    print("\n--- Power law (no floor): y = A * N^B ---")
    print(f"  A={fit_plain['A']:.6e}  B={fit_plain['B']:.4f}  R²={fit_plain['r2']:.4f}")
    print(f"  predicted median @ N={args.predict_at_n}: {pred_plain_y:.4e}")
    print(f"  predicted N for {args.target_mismatch:.0e}: "
          f"{n_star_plain:.3e}" if n_star_plain else "  unreachable")

    print("\n--- Power law + floor: y = A * N^B + C ---")
    print(f"  A={fit_floor['A']:.6e}  B={fit_floor['B']:.4f}  C={fit_floor['C']:.4e}  "
          f"R²={fit_floor['r2']:.4f}")
    print(f"  predicted median @ N={args.predict_at_n}: {pred_floor_y:.4e}")
    print(f"  predicted N for {args.target_mismatch:.0e}: "
          f"{n_star_floor:.3e}" if n_star_floor else "  unreachable (floor ≥ target)")

    payload = {
        "mismatch_col": args.mismatch_col,
        "fit_phases": args.fit_phases,
        "points": [
            {"run_id": r["run_id"], "n_train": r["n_train"], "median": r["median"]}
            for r in fit_points
        ],
        "fit_no_floor": fit_plain,
        "fit_with_floor": fit_floor,
        "predict_at_n": args.predict_at_n,
        "predicted_mismatch_at_n_no_floor": pred_plain_y,
        "predicted_mismatch_at_n_with_floor": pred_floor_y,
        "target_mismatch": args.target_mismatch,
        "predicted_n_for_target_no_floor": n_star_plain,
        "predicted_n_for_target_with_floor": n_star_floor,
        # Convenience: the two headline numbers from the plan
        "headline": {
            "predicted_mismatch_at_300k": pred_plain_y,
            "predicted_n_for_1e-4": n_star_plain,
            "predicted_mismatch_at_300k_with_floor": pred_floor_y,
            "predicted_n_for_1e-4_with_floor": n_star_floor,
        },
    }

    fit_path = args.results_dir / "fit.json"
    fit_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {fit_path}")

    if args.write_prediction:
        pred_path = args.results_dir / "prediction.json"
        # Strip points that are not yet known — this file is the gate.
        gate = {
            "created_from": [r["run_id"] for r in fit_points],
            "headline": payload["headline"],
            "fit_no_floor": fit_plain,
            "fit_with_floor": fit_floor,
            "note": (
                "Write this BEFORE evaluating / looking at the 300k held-out run. "
                "Compare 300k median to predicted_mismatch_at_300k afterwards."
            ),
        }
        pred_path.write_text(json.dumps(gate, indent=2) + "\n")
        print(f"Wrote GATE file {pred_path}")
        print("You may now train/eval 300k_2p5m.")


if __name__ == "__main__":
    main()
