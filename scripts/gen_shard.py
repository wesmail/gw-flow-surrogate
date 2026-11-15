#!/usr/bin/env python
"""Generate the waveform pool ONE SHARD AT A TIME (for cluster array jobs),
reproducing the laptop pool exactly.

data_generation.generate.generate_dataset() builds one parameter table
(Sobol train+val, uniform test, OOD) from the seed and splits it into n_files
equal shards, then generates the shards one after another. This script builds
the SAME table and generates only shard number --index, so 64 shards can run
as 64 independent jobs. Same seed + same counts  =>  same parameters, same
order, same shard layout as the laptop pool.

Subcommands
  plan     print the table layout; with --reference, check that every
           (shard, row) has the same q, chi1z, chi2z and split as the laptop
           manifest. Needs no surrogate; run it BEFORE submitting the array.
  shard    generate one shard (index from --index or $SLURM_ARRAY_TASK_ID)
  compare  after generation + combine: compare the new manifest with the
           laptop manifest row by row
  download fetch the NRHybSur3dq8 data file once (login node, has internet)

Counts: either --n-train/--n-val/--n-test/--n-ood/--n-files, or
--infer-from <laptop manifest.parquet> to read them from that manifest.

Examples (repo root):
  python scripts/gen_shard.py download
  python scripts/gen_shard.py plan  --infer-from data/ref/n1000k_manifest.parquet \
                                    --reference  data/ref/n1000k_manifest.parquet
  python scripts/gen_shard.py shard --infer-from data/ref/n1000k_manifest.parquet \
                                    --output-dir data/n1000k --index 0
  python scripts/gen_shard.py compare --manifest data/n1000k/manifest.parquet \
                                    --reference data/ref/n1000k_manifest.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from data_generation.config import GenerationConfig  # noqa: E402
from data_generation.sampling import (  # noqa: E402
    extrapolation_parameters,
    sobol_parameters,
    uniform_parameters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("gen_shard")


# --------------------------------------------------------------------------- #
# the parameter table: same steps, same order as generate_dataset()           #
# --------------------------------------------------------------------------- #
def param_table(cfg: GenerationConfig, n_train: int, n_val: int, n_test: int, n_ood: int):
    params_sobol = sobol_parameters(cfg, n_train + n_val, seed=cfg.seed, skip=256)
    rng = np.random.default_rng(cfg.seed + 1)
    params_test = uniform_parameters(cfg, n_test, rng)
    params_ood = extrapolation_parameters(n_ood, rng)
    params = np.vstack([params_sobol, params_test, params_ood])
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test + ["ood"] * n_ood
    return params, splits


def shard_slice(index: int, total: int, n_files: int) -> slice:
    per_file = int(np.ceil(total / n_files))
    start = index * per_file
    return slice(start, min(start + per_file, total))


def shard_number(path: str) -> int:
    import re
    m = re.findall(r"(\d+)", Path(str(path)).stem)
    return int(m[-1]) if m else 0


# --------------------------------------------------------------------------- #
def counts_from_args(a) -> dict:
    if a.infer_from:
        ref = pd.read_parquet(a.infer_from, columns=["file_path", "split"])
        vc = ref.split.value_counts()
        c = dict(n_train=int(vc.get("train", 0)), n_val=int(vc.get("val", 0)),
                 n_test=int(vc.get("test", 0)), n_ood=int(vc.get("ood", 0)),
                 n_files=int(ref.file_path.map(shard_number).max()) + 1)
        log.info("counts inferred from %s: %s", a.infer_from, c)
    else:
        need = ("n_train", "n_test", "n_ood", "n_files")
        if any(getattr(a, k) is None for k in need):
            raise SystemExit("give --infer-from, or all of --n-train --n-test --n-ood --n-files")
        n_val = a.n_val if a.n_val is not None else max(1, int(a.n_train * a.val_fraction))
        c = dict(n_train=a.n_train, n_val=n_val, n_test=a.n_test, n_ood=a.n_ood,
                 n_files=a.n_files)
    for k in ("n_train", "n_val", "n_test", "n_ood", "n_files"):   # explicit flags win
        v = getattr(a, k, None)
        if v is not None:
            c[k] = v
    return c


def make_cfg(a, n_files: int, output_dir: str = "unused") -> GenerationConfig:
    return GenerationConfig(output_dir=output_dir, n_files=n_files, seed=a.seed, dt=a.dt,
                            t_min=a.t_min, t_max=a.t_max, q_min=a.q_min, q_max=a.q_max,
                            chi_min=a.chi_min, chi_max=a.chi_max)


def compare_rows(new: pd.DataFrame, ref: pd.DataFrame) -> bool:
    """Match rows by (shard number, row_index); compare split and parameters."""
    def key(d):
        d = d.copy()
        d["_shard"] = d.file_path.map(shard_number)
        return d.set_index(["_shard", "row_index"]).sort_index()
    n, r = key(new), key(ref)
    ok = True
    if len(n) != len(r):
        log.error("row count differs: new %d vs reference %d", len(n), len(r)); ok = False
    common = n.index.intersection(r.index)
    if len(common) < max(len(n), len(r)):
        log.error("%d (shard,row) keys are not in both", max(len(n), len(r)) - len(common)); ok = False
    n, r = n.loc[common], r.loc[common]
    bad_split = int((n.split != r.split).sum())
    dmax = max(float(np.abs(n[c].to_numpy() - r[c].to_numpy()).max()) for c in ("q", "chi1z", "chi2z"))
    log.info("compared %d rows: split mismatches %d, max |parameter difference| %.2e",
             len(common), bad_split, dmax)
    if bad_split or dmax > 1e-6:
        ok = False
        diff = (np.abs(n.q.to_numpy() - r.q.to_numpy()) > 1e-6) | (n.split.to_numpy() != r.split.to_numpy())
        log.error("first differing rows:\n%s\nvs reference\n%s",
                  n[diff][["split", "q", "chi1z", "chi2z"]].head(3),
                  r[diff][["split", "q", "chi1z", "chi2z"]].head(3))
    return ok


# --------------------------------------------------------------------------- #
def cmd_plan(a) -> None:
    c = counts_from_args(a)
    cfg = make_cfg(a, c["n_files"])
    params, splits = param_table(cfg, c["n_train"], c["n_val"], c["n_test"], c["n_ood"])
    total = len(splits)
    per = int(np.ceil(total / c["n_files"]))
    print(f"total {total} waveforms = train {c['n_train']} + val {c['n_val']} + test {c['n_test']} "
          f"+ ood {c['n_ood']};  {c['n_files']} shards x {per} rows")
    if a.reference:
        rows = []
        for i in range(c["n_files"]):
            sl = shard_slice(i, total, c["n_files"])
            for j, g in enumerate(range(sl.start, sl.stop)):
                rows.append((f"waveforms_{i:05d}.h5", j, splits[g], *params[g]))
        new = pd.DataFrame(rows, columns=["file_path", "row_index", "split", "q", "chi1z", "chi2z"])
        ref = pd.read_parquet(a.reference, columns=["file_path", "row_index", "split", "q", "chi1z", "chi2z"])
        if compare_rows(new, ref):
            print("PLAN MATCHES THE REFERENCE: same parameters, splits and shard layout.")
        else:
            print("PLAN DOES NOT MATCH. Check --seed / ranges / counts. (If some laptop rows "
                  "failed to generate, row indices shift inside that shard. If the last laptop shard "
                  "was empty, pass --n-files explicitly.)")
            sys.exit(1)


def cmd_shard(a) -> None:
    idx = a.index if a.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    if idx < 0:
        raise SystemExit("give --index or run inside a SLURM array job")
    c = counts_from_args(a)
    cfg = make_cfg(a, c["n_files"], a.output_dir)
    params, splits = param_table(cfg, c["n_train"], c["n_val"], c["n_test"], c["n_ood"])
    sl = shard_slice(idx, len(splits), c["n_files"])
    if sl.start >= len(splits):
        log.info("shard %d is empty; nothing to do", idx); return
    out = Path(a.output_dir)
    man = out / f"waveforms_{idx:05d}.manifest.parquet"
    if man.is_file() and not a.overwrite:
        log.info("shard %d already done (%s); use --overwrite to redo", idx, man); return

    import data_generation.generate as G
    try:   # load the local data file directly: no download from compute nodes
        import gwsurrogate
        G._SURROGATE = gwsurrogate.LoadSurrogate(cfg.surrogate_name)
    except Exception as exc:
        raise SystemExit(f"could not load {cfg.surrogate_name} ({exc}). "
                         "Run `python scripts/gen_shard.py download` once on the login node.")
    log.info("shard %d: rows %d..%d of %d", idx, sl.start, sl.stop - 1, len(splits))
    G._write_shard(idx, cfg, params[sl], splits[sl])


def cmd_compare(a) -> None:
    ok = compare_rows(pd.read_parquet(a.manifest), pd.read_parquet(a.reference))
    print("POOL MATCHES THE REFERENCE." if ok else "POOL DIFFERS FROM THE REFERENCE.")
    sys.exit(0 if ok else 1)


def cmd_download(a) -> None:
    import gwsurrogate
    gwsurrogate.catalog.pull("NRHybSur3dq8")
    gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    print("NRHybSur3dq8 downloaded and loads.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def gen_args(p):
        p.add_argument("--infer-from", default=None, help="laptop manifest.parquet to read counts from")
        p.add_argument("--n-train", type=int); p.add_argument("--n-val", type=int)
        p.add_argument("--n-test", type=int); p.add_argument("--n-ood", type=int)
        p.add_argument("--n-files", type=int); p.add_argument("--val-fraction", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=42)          # data_generation.cli defaults
        p.add_argument("--dt", type=float, default=0.5)
        p.add_argument("--t-min", type=float, default=-5000.0); p.add_argument("--t-max", type=float, default=130.0)
        p.add_argument("--q-min", type=float, default=1.0); p.add_argument("--q-max", type=float, default=8.0)
        p.add_argument("--chi-min", type=float, default=-0.8); p.add_argument("--chi-max", type=float, default=0.8)

    p = sub.add_parser("plan"); gen_args(p); p.add_argument("--reference", default=None)
    p = sub.add_parser("shard"); gen_args(p)
    p.add_argument("--output-dir", required=True); p.add_argument("--index", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("compare"); p.add_argument("--manifest", required=True); p.add_argument("--reference", required=True)
    sub.add_parser("download")
    a = ap.parse_args()
    {"plan": cmd_plan, "shard": cmd_shard, "compare": cmd_compare, "download": cmd_download}[a.cmd](a)


if __name__ == "__main__":
    main()
