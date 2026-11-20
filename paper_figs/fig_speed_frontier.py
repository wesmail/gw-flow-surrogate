#!/usr/bin/env python
"""How fast is the surrogate?  Timing vs batch size and steps, plus the
speed-accuracy frontier, compared with NRHybSur3dq8 on the same time grid.

What is timed (stage "time"): exactly "theta in, waveform out" for the FROZEN
protocol path: theta-only autoregressive generation of all patches (n_steps flow
steps per patch, one sample), reconstruction of h(t) (PCHIP amplitude), and the
copy of h(t) to the CPU. Data loading and mismatch computation are NOT timed.
The first true patch (the start token, see the guide) is prepared beforehand.
CUDA is synchronised around every timed call; warm-up calls are excluded.

STAGE 1, our model (GPU, a few minutes)::

    python paper_figs/fig_speed_frontier.py time --checkpoint "$FT" \
        --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
        --steps 6 8 10 12 15 20 40 80 --batch-sizes 1 16 64 256 1024 \
        --out results/speed_model.csv

STAGE 2, the reference (CPU, environment WITH gwsurrogate; one core)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python paper_figs/fig_speed_frontier.py ref --manifest data/n1000k/manifest.parquet \
        --n 50 --out results/speed_nrhybsur.json

STAGE 3, the figure (CPU, seconds); accuracy is read from the step-sweep CSVs::

    python paper_figs/fig_speed_frontier.py plot results/speed_model.csv \
        --ref results/speed_nrhybsur.json \
        --acc-glob "results/sampler*/K*_M1.csv" --frozen-steps 12

Figure: (a) time per waveform vs batch size (frozen steps) with the NRHybSur3dq8
line; (b) speed-accuracy frontier at one batch size (median and p90 mismatch vs
time per waveform, one point per number of steps).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))

FIG_W = 11.0     # shared width of full-width paper figures


def time_grid(manifest: str, n_time: int | None = None) -> np.ndarray:
    import h5py
    man = pd.read_parquet(manifest)
    shard = Path(man["file_path"].iloc[0])
    if not shard.is_file():
        shard = Path(manifest).parent / shard.name
    with h5py.File(shard, "r") as f:
        t = np.asarray(f["t"][:], dtype=np.float64)
    return t if n_time is None else t[:n_time]


# --------------------------------------------------------------------------- #
def run_time(a: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset
    from datasets.datamodule import _collate_batch
    from datasets.patchify import n_patches_for
    from datasets.waveform_dataset import WaveformPatchDataset
    from models.amp_interp import log_amp_per_sample
    from models.lightning_module import GWFlowSurrogateLit

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = dev.type == "cuda"
    lit = GWFlowSurrogateLit.load_from_checkpoint(a.checkpoint, map_location=dev).eval().to(dev)
    P, rscale = int(lit.hparams.patch_len), float(lit.hparams.residual_scale)
    ds = WaveformPatchDataset(a.manifest, split=a.split, patch_len=P, context_patches=1,
                              norm_stats_path=a.norm_stats, preload=False)
    N = n_patches_for(ds.n_time, P)
    Bmax = max(a.batch_sizes)
    rows_avail = min(len(ds), Bmax)
    batch = next(iter(DataLoader(Subset(ds, range(rows_avail)), batch_size=rows_avail,
                                 collate_fn=_collate_batch, shuffle=False)))
    b = lit._patchify_batch({k: v.to(dev) for k, v in batch.items()})
    reps = int(np.ceil(Bmax / rows_avail))                      # tile if the split is small
    first = b["token_features"][:, :1].repeat(reps, 1, 1)[:Bmax].contiguous()
    theta = b["theta"].repeat(reps, 1)[:Bmax].contiguous()
    norm = {k: b[k].repeat(reps, 1)[:Bmax].contiguous()
            for k in ("norm_log_a_mean", "norm_log_a_std", "norm_dphi_mean", "norm_dphi_std")}
    s_frac = torch.arange(P, device=dev, dtype=torch.float64) / P

    def theta_to_h(Bn: int, n_steps: int) -> np.ndarray:
        gen = lit.model.generate(first[:Bn], theta[:Bn], N - 1, n_steps=n_steps, n_samples=1,
                                 temperature=1.0, mean_of=1)[:, 0]
        feat = torch.cat([first[:Bn], gen], dim=1).double()
        la = log_amp_per_sample(feat[..., 0] * norm["norm_log_a_std"][:Bn, :1].double()
                                + norm["norm_log_a_mean"][:Bn, :1].double(), P, a.amp_mode)
        dphi = feat[..., 1] * norm["norm_dphi_std"][:Bn, :1].double() + norm["norm_dphi_mean"][:Bn, :1].double()
        res = feat[..., 2:2 + P] * rscale
        bnd = torch.cumsum(dphi, dim=1) - dphi
        ph = (bnd[..., None] + s_frac * dphi[..., None] + res).reshape(Bn, -1)
        return (torch.exp(la) * torch.exp(-1j * ph)).cpu().numpy()     # waveform on the host

    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    print(f"device {gpu}; {N} patches; timing theta -> h(t) on host; amp={a.amp_mode}")
    rows = []
    with torch.inference_mode():
        for n_steps in a.steps:
            for Bn in a.batch_sizes:
                torch.manual_seed(0)
                try:
                    for _ in range(a.warmup):
                        theta_to_h(Bn, n_steps)
                    if cuda:
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    n_calls = max(a.min_calls, int(np.ceil(a.min_waveforms / Bn)))
                    ts = []
                    t_start = time.perf_counter()
                    for c in range(n_calls):
                        if c >= a.min_calls and time.perf_counter() - t_start > a.max_seconds:
                            break
                        if cuda:
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        theta_to_h(Bn, n_steps)
                        if cuda:
                            torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                except torch.cuda.OutOfMemoryError:
                    print(f"  steps={n_steps:3d} batch={Bn:5d}: out of GPU memory, skipped")
                    torch.cuda.empty_cache(); continue
                ts = np.array(ts)
                r = dict(n_steps=n_steps, batch=Bn, calls=len(ts),
                         s_per_call_median=float(np.median(ts)),
                         ms_per_waveform=1e3 * float(np.median(ts)) / Bn,
                         ms_per_waveform_min=1e3 * float(ts.min()) / Bn,
                         peak_mem_GB=(torch.cuda.max_memory_allocated() / 1e9 if cuda else np.nan),
                         device=gpu)
                rows.append(r)
                print(f"  steps={n_steps:3d} batch={Bn:5d}: {r['s_per_call_median']:.3f} s per call, "
                      f"{r['ms_per_waveform']:.2f} ms per waveform  (peak {r['peak_mem_GB']:.1f} GB)")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


# --------------------------------------------------------------------------- #
_SUR = None
_T = None


def _worker_init(t):
    global _SUR, _T
    import warnings
    warnings.simplefilter("ignore")
    import gwsurrogate
    _SUR = gwsurrogate.LoadSurrogate("NRHybSur3dq8"); _T = t
    _SUR(2.0, [0, 0, 0.1], [0, 0, -0.1], times=_T, f_low=0, mode_list=[(2, 2)])   # warm-up


def _worker_call(th):
    q, c1, c2 = th
    _SUR(float(q), [0, 0, float(c1)], [0, 0, float(c2)], times=_T, f_low=0, mode_list=[(2, 2)])
    return 0


def run_ref(a: argparse.Namespace) -> None:
    import os
    if a.processes > 1:                      # throughput with several CPU workers in parallel
        import multiprocessing as mp
        t = time_grid(a.manifest); t = t[t <= 134.0]
        rng = np.random.default_rng(0)
        th = np.column_stack([rng.uniform(1, 8, a.n), rng.uniform(-0.8, 0.8, a.n), rng.uniform(-0.8, 0.8, a.n)])
        with mp.get_context("spawn").Pool(a.processes, initializer=_worker_init, initargs=(t,)) as pool:
            pool.map(_worker_call, th[: a.processes * 2])           # make sure all workers are loaded
            t0 = time.perf_counter(); pool.map(_worker_call, th, chunksize=4)
            wall = time.perf_counter() - t0
        res = dict(label=f"NRHybSur3dq8, {a.processes} CPU processes", ms_per_waveform=1e3 * wall / a.n,
                   n=int(a.n), n_time=int(t.size), processes=a.processes,
                   threads=os.environ.get("OMP_NUM_THREADS", "not set"),
                   note="throughput: wall time / number of waveforms, all workers busy")
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"NRHybSur3dq8 with {a.processes} processes: {res['ms_per_waveform']:.2f} ms per waveform "
              f"(throughput over {a.n} waveforms) -> {out}")
        return
    import gwsurrogate
    t = time_grid(a.manifest)
    t = t[t <= 134.0]                                   # NRHybSur3dq8 domain
    from pathlib import Path
    h5 = Path(gwsurrogate.catalog.download_path()) / "NRHybSur3dq8.h5"
    if not h5.is_file():
        gwsurrogate.catalog.pull("NRHybSur3dq8")
    sur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    rng = np.random.default_rng(0)
    th = np.column_stack([rng.uniform(1, 8, a.n + a.warmup), rng.uniform(-0.8, 0.8, a.n + a.warmup),
                          rng.uniform(-0.8, 0.8, a.n + a.warmup)])
    import warnings
    ts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (q, c1, c2) in enumerate(th):
            t0 = time.perf_counter()
            sur(float(q), [0, 0, float(c1)], [0, 0, float(c2)], times=t, f_low=0, mode_list=[(2, 2)])
            dt = time.perf_counter() - t0
            if i >= a.warmup:
                ts.append(dt)
    ts = np.array(ts)
    res = dict(label="NRHybSur3dq8", ms_per_waveform=1e3 * float(np.median(ts)),
               ms_p10=1e3 * float(np.percentile(ts, 10)), ms_p90=1e3 * float(np.percentile(ts, 90)),
               n=int(ts.size), n_time=int(t.size), mode_list="[(2,2)]", f_low=0,
               threads=os.environ.get("OMP_NUM_THREADS", "not set"),
               note="one call per waveform on the same time grid as the model; CPU")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"NRHybSur3dq8: median {res['ms_per_waveform']:.1f} ms per waveform "
          f"(p10 {res['ms_p10']:.1f}, p90 {res['ms_p90']:.1f}; {res['n']} calls, {res['n_time']} samples, "
          f"OMP_NUM_THREADS={res['threads']}) -> {out}")


# --------------------------------------------------------------------------- #
def accuracy_table(pattern: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"K(\d+)_M1\.csv$", f)
        if not m:
            continue
        d = pd.read_csv(f)
        if "context_fraction" in d:
            d = d[d.context_fraction == d.context_fraction.min()]
        x = d.mm_pt_full
        rows.append(dict(n_steps=int(m.group(1)), median=x.median(), p90=x.quantile(0.9),
                         below=(x < 1e-3).mean(), file=f))
    return pd.DataFrame(rows).drop_duplicates("n_steps", keep="last").sort_values("n_steps")


def run_plot(a: argparse.Namespace) -> None:
    from style import C, apply_style, save
    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt

    tm = pd.read_csv(a.csv)
    ref = json.loads(Path(a.ref).read_text()) if a.ref else None
    acc = accuracy_table(a.acc_glob) if a.acc_glob else pd.DataFrame()
    fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 4.0), constrained_layout=True)

    # (a) time per waveform vs batch size, frozen steps
    f = tm[tm.n_steps == a.frozen_steps].sort_values("batch")
    ax[0].plot(f.batch, f.ms_per_waveform, "o-", color=C["fm"], lw=1.8, ms=5,
               label=f"flow-matching surrogate ({a.frozen_steps} steps, GPU)")
    if ref:
        ax[0].axhline(ref["ms_per_waveform"], color=C["gray"], ls="-.", lw=1.3,
                      label=f"NRHybSur3dq8 (one CPU core)")
    ref_all = json.loads(Path(a.ref_all).read_text()) if a.ref_all else None
    if ref_all:
        ax[0].axhline(ref_all["ms_per_waveform"], color=C["gray"], ls=":", lw=1.3,
                      label=f"NRHybSur3dq8 ({ref_all['processes']} CPU cores)")
    fast = None
    if a.fast_csv:
        fast = pd.read_csv(a.fast_csv)
        ff = fast[fast.n_steps == a.frozen_steps].sort_values("batch")
        ax[0].plot(ff.batch, ff.ms_per_waveform, "s--", color=C["fm"], lw=1.3, ms=4, alpha=0.75,
                   label=f"same, compiled step (torch.compile)")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax[0].xaxis.set_major_locator(FixedLocator(sorted(f.batch.unique())))
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax[0].set_xlabel("waveforms generated together (batch size)")
    ax[0].set_ylabel("time per waveform [ms]")
    ax[0].set_title("(a) time per waveform")
    ax[0].legend(loc="upper right", fontsize="x-small", frameon=False)

    # (b) speed-accuracy frontier at one batch size
    g = tm[tm.batch == a.frontier_batch].merge(acc, on="n_steps", how="inner").sort_values("n_steps")
    if len(g):
        ax[1].fill_between(g.ms_per_waveform, g["median"], g.p90, color=C["band"], alpha=0.8, lw=0,
                           label="median to p90")
        ax[1].plot(g.ms_per_waveform, g["median"], "o-", color=C["fm"], lw=1.8, ms=5, label="median")
        for _, r in g.iterrows():
            ax[1].annotate(f"{int(r.n_steps)}", (r.ms_per_waveform, r["median"]),
                           textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8,
                           color=C["accent"] if r.n_steps == a.frozen_steps else "k")
        if ref:
            ax[1].axvline(ref["ms_per_waveform"], color=C["gray"], ls="-.", lw=1.3)
        ax[1].axhline(1e-3, color=C["accent"], ls="--", lw=0.9, label="target $10^{-3}$")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel(f"time per waveform [ms]  (batch {a.frontier_batch})")
        ax[1].set_ylabel("mismatch (full waveform)")
        ax[1].set_title("(b) speed and accuracy (numbers: steps per patch)")
        ax[1].legend(loc="upper right", fontsize="x-small", frameon=False)
    save(fig, a.out)

    # ---- numbers for the text ---------------------------------------------
    print(f"\ntime per waveform at {a.frozen_steps} steps (GPU {tm.device.iloc[0]}):")
    for _, r in f.iterrows():
        sp = ""
        if ref:
            ratio = ref["ms_per_waveform"] / r.ms_per_waveform
            sp = (f"  -> {ratio:.1f}x faster than NRHybSur3dq8" if ratio >= 1
                  else f"  -> {1 / ratio:.1f}x SLOWER than NRHybSur3dq8")
        print(f"  batch {int(r.batch):5d}: {r.ms_per_waveform:8.2f} ms{sp}")
    if ref:
        print(f"NRHybSur3dq8: {ref['ms_per_waveform']:.1f} ms per waveform (CPU, threads={ref['threads']})")
    if fast is not None:
        m = f.merge(fast[fast.n_steps == a.frozen_steps][["batch", "ms_per_waveform"]], on="batch",
                    suffixes=("", "_fast"))
        print("\ncompiled step vs original (ms per waveform):")
        for _, r in m.iterrows():
            print(f"  batch {int(r.batch):5d}: {r.ms_per_waveform:8.2f} -> {r.ms_per_waveform_fast:8.2f}  "
                  f"({r.ms_per_waveform / r.ms_per_waveform_fast:.1f}x)")
    if ref_all:
        print(f"NRHybSur3dq8, {ref_all['processes']} processes: {ref_all['ms_per_waveform']:.2f} ms per waveform "
              f"(throughput); best model batch: {f.ms_per_waveform.min():.2f} ms "
              f"-> {f.ms_per_waveform.min() / ref_all['ms_per_waveform']:.1f}x the reference time")
    if len(g):
        print(f"\nfrontier at batch {a.frontier_batch}:")
        print(g[["n_steps", "ms_per_waveform", "median", "p90", "below"]].to_string(
            index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--manifest", required=True)
    t.add_argument("--norm-stats", required=True)
    t.add_argument("--split", default="test")
    t.add_argument("--steps", nargs="+", type=int, default=[6, 8, 10, 12, 15, 20, 40, 80])
    t.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 256, 1024])
    t.add_argument("--amp-mode", default="pchip")
    t.add_argument("--warmup", type=int, default=1)
    t.add_argument("--min-calls", type=int, default=3)
    t.add_argument("--min-waveforms", type=int, default=0, help="time at least this many waveforms per setting")
    t.add_argument("--max-seconds", type=float, default=30.0,
                   help="stop timing a setting after this long (at least --min-calls calls are always made)")
    t.add_argument("--device", default=None)
    t.add_argument("--out", default="results/speed_model.csv")
    r = sub.add_parser("ref")
    r.add_argument("--manifest", required=True, help="to read the same time grid")
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--processes", type=int, default=1, help=">1: throughput with this many CPU worker processes")
    r.add_argument("--out", default="results/speed_nrhybsur.json")
    p = sub.add_parser("plot")
    p.add_argument("csv")
    p.add_argument("--ref", default=None)
    p.add_argument("--ref-all", default=None, help="json of the multi-process reference (all CPU cores)")
    p.add_argument("--fast-csv", default=None, help="timing CSV of the compiled path (time --fast)")
    p.add_argument("--acc-glob", default="results/sampler*/K*_M1.csv")
    p.add_argument("--frozen-steps", type=int, default=12)
    p.add_argument("--frontier-batch", type=int, default=1024)
    p.add_argument("--out", default="fm_speed")
    p.add_argument("--talk", action="store_true")
    args = ap.parse_args()
    {"time": run_time, "ref": run_ref, "plot": run_plot}[args.cmd](args)