#!/usr/bin/env python
"""Context-fraction study: how much does the model gain from a TRUE seed?

Question answered
-----------------
A surrogate is used from theta alone. Our model can also be given the first
part of the true waveform ("seed") and generate the rest. This script measures
the accuracy as a function of the seeded fraction, from theta-only (1 patch)
to 75% seeded.

Fair comparison: the COMMON WINDOW
----------------------------------
"Mismatch of the generated part" is NOT comparable across context fractions:
at 0.75 only the merger is generated, at theta-only the whole waveform is.
"Full-waveform mismatch" is not comparable either: the seeded part is exact,
so more context trivially dilutes the error.

The headline quantity is therefore the mismatch on ONE fixed time window that
is generated in EVERY run: [window_start, end] (default window_start = the
largest context fraction, i.e. the last 25% = late inspiral + merger +
ringdown). Same samples, same truth, only the amount of seed changes.

All mismatches are against the ORIGINAL waveform (exp(log_a - i phi) from the
HDF5 tracks, fp64), so the patch representation floor is included, i.e. this is
end-to-end accuracy. The floor itself (true tokens -> reconstruction vs
original, no model) is stored per waveform and drawn as a reference line.

STAGE 1 (eval, GPU)::

    python paper_figs/fig_context_fraction.py eval \\
        --checkpoint scan_1M/n1000k/seed42/checkpoints/<best>.ckpt \\
        --manifest data/test5k/manifest.parquet \\
        --norm-stats data/scale300k/norm_stats.npz \\
        --split test --n-waveforms 1000 \\
        --fractions 0 0.25 0.5 0.75 \\
        --n-steps 12 --mean-of 1 --seed 1234 \\
        --out results/context_fraction.csv

    --norm-stats MUST be the file the checkpoint was trained with. If omitted,
    it is read from <run>/hparams.yaml (norm_stats_path); if that fails the
    script stops. It never falls back to identity statistics.

STAGE 2 (figure, CPU, seconds)::

    python paper_figs/fig_context_fraction.py plot results/context_fraction.csv [--talk]

Figure: (a) median (+ p10-p90 band) mismatch on the common window vs context
fraction, phase+time-optimised and strict, with the representation floor;
(b) per-waveform CDF on the common window, one curve per context fraction.
A summary table (median / p90 / p99 / fraction below target) is printed and
written next to the CSV as *_summary.csv.

RUN FROM THE REPO ROOT.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PKG))
from style import C, apply_style, save  # noqa: E402
from protocol_utils import (  # noqa: E402
    mismatches,
    reconstruct_fp64,
    resolve_norm_stats,
    sha256_file,
)

THETA_ONLY_LABEL = "θ only"


# --------------------------------------------------------------------------- #
# STAGE 1: evaluation                                                         #
# --------------------------------------------------------------------------- #
def run_eval(a: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    from datasets.datamodule import _collate_batch
    from datasets.patchify import n_patches_for
    from datasets.waveform_dataset import WaveformPatchDataset
    from models.lightning_module import GWFlowSurrogateLit

    protocol = getattr(a, "protocol_data", None)
    overrides = getattr(a, "protocol_overrides", {}) or {}

    dev = torch.device(a.device)
    ckpt = Path(a.checkpoint)
    stats_path = resolve_norm_stats(ckpt, a.norm_stats)

    try:
        lit = GWFlowSurrogateLit.load_from_checkpoint(str(ckpt), map_location=dev)
    except RuntimeError:
        print("note: checkpoint has no abs-time embedding -> use_abs_time=False")
        lit = GWFlowSurrogateLit.load_from_checkpoint(str(ckpt), map_location=dev,
                                                      use_abs_time=False)
    lit.hparams.amp_mode = a.amp_mode
    lit.eval().to(dev)
    hp = lit.hparams
    P, rscale = int(hp.patch_len), float(hp.residual_scale)
    if int(hp.n_channels) != 1:
        raise NotImplementedError("script assumes n_channels == 1 (the (2,2) mode)")

    # share_memory=False: full-split preload of n1000k test OOMs /dev/shm on WSL.
    ds = WaveformPatchDataset(
        a.manifest, split=a.split, patch_len=P, context_patches=1,
        norm_stats_path=str(stats_path), preload=True, share_memory=False,
    )
    N = n_patches_for(ds.n_time, P)
    usable = N * P
    n = min(a.n_waveforms, len(ds))
    if n < a.n_waveforms:
        print(f"WARNING: split {a.split!r} has only {len(ds)} rows (asked {a.n_waveforms})")

    # time grid (for the "seed ends at t = ... M" annotation)
    import h5py
    man = pd.read_parquet(a.manifest)
    shard = Path(man["file_path"].iloc[0])
    if not shard.is_file():
        shard = Path(a.manifest).parent / shard.name
    with h5py.File(shard, "r") as f:
        t_grid = np.asarray(f["t"][:], dtype=np.float64)[:usable]

    # context patches per requested fraction; fraction <= 0 means theta-only (1 patch)
    ctxs = [1 if fr <= 0 else max(1, int(round(fr * N))) for fr in a.fractions]
    w_frac = a.window_start if a.window_start is not None else max(a.fractions)
    w0 = max(int(round(w_frac * N)), max(ctxs))           # window must be generated in every run
    s_w = w0 * P
    print(f"{N} patches x {P} samples; contexts {ctxs}; common window = patches "
          f"[{w0}, {N}) = t in [{t_grid[s_w]:.0f}, {t_grid[-1]:.0f}] M")
    print(f"norm stats: {stats_path}  residual_scale={rscale}  n_steps={a.n_steps}  "
          f"mean_of={a.mean_of}  T={a.temperature}  seed={a.seed}  device={dev}")

    def reconstruct(feat: torch.Tensor, b: dict) -> torch.Tensor:
        return reconstruct_fp64(feat, b, rscale, P, amp_mode=a.amp_mode)

    loader = DataLoader(Subset(ds, range(n)), batch_size=a.batch_size,
                        collate_fn=_collate_batch, shuffle=False)
    rows: list[dict] = []
    t_start = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            b = {k: v.to(dev) for k, v in batch.items()}
            b = lit._patchify_batch(b)
            feat = b["token_features"]                              # (B, N, D)
            B = feat.shape[0]

            # original waveform, fp64, phase referenced to its first sample
            la_o = b["log_a_track"][:, 0, :usable].double()
            ph_o = b["phi_track"][:, 0, :usable].double()
            h_orig = torch.exp(la_o) * torch.exp(-1j * (ph_o - ph_o[:, :1]))
            h_tok = reconstruct(feat, b)                            # true tokens
            peak = h_orig.abs().argmax(dim=1)

            fl_s, fl_p, fl_pt, _ = mismatches(h_orig[:, s_w:], h_tok[:, s_w:])
            theta = b["theta"].double().cpu().numpy()

            for fr, ctx in zip(a.fractions, ctxs):
                # common random numbers: same noise stream for every fraction
                torch.manual_seed(a.seed + bi)
                gen_fn = lit.model.generate_fast if getattr(a, "fast", False) else lit.model.generate
                gen = gen_fn(feat[:, :ctx], b["theta"], N - ctx,
                             n_steps=a.n_steps, n_samples=1,
                             temperature=a.temperature,
                             mean_of=a.mean_of)[:, 0]
                h_pred = reconstruct(torch.cat([feat[:, :ctx], gen], dim=1), b)
                s_g = ctx * P

                w = mismatches(h_orig[:, s_w:], h_pred[:, s_w:])
                w_tok = mismatches(h_tok[:, s_w:], h_pred[:, s_w:])
                g = mismatches(h_orig[:, s_g:], h_pred[:, s_g:])
                full = mismatches(h_orig, h_pred)

                # accumulated (unwrapped) dephasing at the peak, no alignment:
                # the seed fixes time and phase, so this is a real error
                rel = (h_pred[:, s_g:] * h_orig[:, s_g:].conj()).angle().cpu().numpy()
                rel = np.unwrap(rel, axis=1)
                pk = (peak - s_g).clamp_min(0).cpu().numpy()
                dphi_peak = rel[np.arange(B), pk] - rel[:, 0]

                to = lambda x: x.double().cpu().numpy()  # noqa: E731
                cols = dict(
                    mm_strict_window=to(w[0]), mm_phase_window=to(w[1]),
                    mm_pt_window=to(w[2]), dt_shift_window_M=to(w[3]) * a_dt(t_grid),
                    mm_pt_window_tokens=to(w_tok[2]),
                    mm_strict_gen=to(g[0]), mm_pt_gen=to(g[2]),
                    mm_strict_full=to(full[0]), mm_pt_full=to(full[2]),
                    floor_strict_window=to(fl_s), floor_pt_window=to(fl_pt),
                    dphi_peak_rad=dphi_peak,
                )
                for i in range(B):
                    q, c1, c2 = theta[i, :3]
                    rows.append(dict(
                        idx=bi * a.batch_size + i, q=q, chi1z=c1, chi2z=c2,
                        chi_eff=(q * c1 + c2) / (1 + q),
                        context_fraction=fr, context_patches=ctx,
                        t_seed_end_M=float(t_grid[s_g]),
                        **{k: float(v[i]) for k, v in cols.items()},
                    ))
            done = min((bi + 1) * a.batch_size, n)
            el = time.time() - t_start
            print(f"  {done}/{n}  ({el / 60:.1f} min, ETA {el / done * (n - done) / 60:.1f} min)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    meta = dict(
        script=Path(__file__).name, checkpoint=str(ckpt), checkpoint_sha256=sha256_file(ckpt),
        manifest=str(a.manifest), split=a.split, n_waveforms=int(n),
        norm_stats=str(stats_path), norm_stats_sha256=sha256_file(stats_path),
        norm_stats_values={k: np.load(stats_path)[k].tolist()
                           for k in ("log_a_mean", "log_a_std", "delta_phi_mean", "delta_phi_std")},
        residual_scale=rscale, patch_len=P, n_patches=N, dt_M=a_dt(t_grid),
        fractions=list(a.fractions), context_patches=ctxs,
        window_patches=[w0, N], window_t_M=[float(t_grid[s_w]), float(t_grid[-1])],
        n_steps=a.n_steps, mean_of=a.mean_of, temperature=a.temperature, seed=a.seed,
        amp_mode=a.amp_mode,
        batch_size=a.batch_size, device=str(dev),
        gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
        torch=torch.__version__, inference_dtype="fp32 network, fp64 reconstruction/mismatch",
        mismatch="white-noise (flat PSD) time-domain overlap; against ORIGINAL waveform",
        wall_time_min=(time.time() - t_start) / 60,
        protocol=protocol,
        overrides=overrides,
        protocol_sha256=(protocol or {}).get("_protocol_sha256"),
    )
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"CSV  -> {out}\nmeta -> {out.with_suffix('.json')}")
    summarize(df, out, a.target)


def a_dt(t_grid: np.ndarray) -> float:
    return float(t_grid[1] - t_grid[0])


# --------------------------------------------------------------------------- #
# Summary table                                                               #
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame, out: Path, target: float) -> pd.DataFrame:
    rows = []
    for fr, g in df.groupby("context_fraction", sort=True):
        r = dict(context_fraction=fr, context_patches=int(g.context_patches.iloc[0]),
                 n=len(g))
        for col in ("mm_pt_window", "mm_strict_window", "mm_pt_gen", "mm_pt_full"):
            v = g[col].to_numpy()
            r[f"{col}_p50"] = np.median(v)
            r[f"{col}_p90"] = np.quantile(v, 0.9)
            r[f"{col}_p99"] = np.quantile(v, 0.99)
            r[f"{col}_frac_below_target"] = float((v < target).mean())
        r["floor_pt_window_p50"] = g.floor_pt_window.median()
        r["abs_dphi_peak_p50"] = g.dphi_peak_rad.abs().median()
        r["abs_dt_shift_window_p50_M"] = g.dt_shift_window_M.abs().median()
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    with pd.option_context("display.float_format", "{:.3e}".format, "display.width", 200):
        print(s[["context_fraction", "context_patches", "n",
                 "mm_pt_window_p50", "mm_pt_window_p90", "mm_pt_window_p99",
                 "mm_pt_window_frac_below_target", "mm_strict_window_p50",
                 "floor_pt_window_p50", "abs_dphi_peak_p50"]].to_string(index=False))
    return s


# --------------------------------------------------------------------------- #
# STAGE 2: figure                                                             #
# --------------------------------------------------------------------------- #
def bootstrap_median_ci(v: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    lv = np.log10(np.clip(v, 1e-16, None))
    meds = np.median(rng.choice(lv, size=(n_boot, lv.size), replace=True), axis=1)
    return 10 ** np.quantile(meds, 0.16), 10 ** np.quantile(meds, 0.84)


def run_plot(a: argparse.Namespace) -> None:
    apply_style(talk=a.talk)
    import matplotlib.pyplot as plt
    from matplotlib import cm

    d = pd.read_csv(a.csv)
    meta_p = Path(a.csv).with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    fr = np.sort(d.context_fraction.unique())
    t_end = d.groupby("context_fraction").t_seed_end_M.first()

    def xlab(f):
        return THETA_ONLY_LABEL if f <= 0 else f"{f:g}"

    # two-column journal width by default; --single stacks for one column
    if a.single:
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    axes = np.atleast_1d(axes).ravel()

    # (a) summary vs context fraction --------------------------------------
    ax = axes[0]
    med = np.array([d[d.context_fraction == f].mm_pt_window.median() for f in fr])
    p10 = np.array([d[d.context_fraction == f].mm_pt_window.quantile(0.1) for f in fr])
    p90 = np.array([d[d.context_fraction == f].mm_pt_window.quantile(0.9) for f in fr])
    ci = np.array([bootstrap_median_ci(d[d.context_fraction == f].mm_pt_window.to_numpy())
                   for f in fr])
    med_s = np.array([d[d.context_fraction == f].mm_strict_window.median() for f in fr])
    floor = d.drop_duplicates("idx").floor_pt_window.median()

    ax.fill_between(fr, p10, p90, color=C["band"], alpha=0.8, lw=0, label="p10–p90")
    ax.errorbar(fr, med, yerr=[med - ci[:, 0], ci[:, 1] - med], fmt="o-", color=C["fm"],
                lw=1.8, ms=5, capsize=2.5, label="phase+time opt.")
    ax.plot(fr, med_s, "s--", color=C["fm"], mfc="white", lw=1.2, ms=4,
            label="strict")
    ax.axhline(floor, color=C["gray"], ls=":", lw=1.2, label="floor")
    if a.target:
        ax.axhline(a.target, color=C["accent"], ls="--", lw=1.0, label="target")
    ax.set_yscale("log")
    ax.set_xticks(fr)
    ax.set_xticklabels([xlab(f) for f in fr])
    ax.set_xlabel("seeded fraction of the waveform")
    ax.set_ylabel("mismatch on common window")
    ax.set_title("(a)", loc="left", fontweight="bold")
    # caption: markers = median (error bar: bootstrap 68% CI), band = p10-p90
    top = ax.secondary_xaxis("top")
    top.set_xticks(fr, labels=[f"{t_end[f]:.0f}" for f in fr])
    top.tick_params(labelsize="x-small")
    top.set_xlabel("seed ends at t [M]", fontsize="small")
    # headroom above the data for the legend (log axis: multiply)
    ax.set_ylim(min(floor, p10.min()) / 2, max(p90.max(), med_s.max(), a.target or 0) * 40)
    ax.legend(loc="upper right", fontsize="x-small", frameon=True, framealpha=0.92,
              edgecolor="none", ncol=2, columnspacing=1.0, handlelength=1.8)

    # (b) CDF per context fraction -----------------------------------------
    ax = axes[1]
    cols = cm.Blues(np.linspace(0.45, 0.95, len(fr)))
    for f, c in zip(fr, cols):
        v = np.sort(d[d.context_fraction == f].mm_pt_window.to_numpy())
        y = np.arange(1, v.size + 1) / v.size
        lab = xlab(f) + ("" if f <= 0 else " seeded")
        if a.target:
            lab += f"  ({100 * (v < a.target).mean():.0f}% < {a.target:g})"
        ax.step(v, y, where="post", color=c, lw=1.6, label=lab)
    fl = np.sort(d.drop_duplicates("idx").floor_pt_window.to_numpy())
    ax.step(fl, np.arange(1, fl.size + 1) / fl.size, where="post", color=C["gray"],
            ls=":", lw=1.2, label="floor")
    if a.target:
        ax.axvline(a.target, color=C["accent"], ls="--", lw=1.0)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("mismatch on common window")
    ax.set_ylabel("fraction of test waveforms")
    ax.set_title("(b)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize="x-small", frameon=True, framealpha=0.92,
              edgecolor="none")

    if meta:
        n_wf = meta.get("n_waveforms")
        w = meta.get("window_t_M")
        print(f"caption facts: n={n_wf}, common window t in [{w[0]:.0f}, {w[1]:.0f}] M, "
              f"n_steps={meta.get('n_steps')}, mean_of={meta.get('mean_of')}, "
              f"seed={meta.get('seed')}, ckpt={meta.get('checkpoint')}")
    fig.tight_layout()
    save(fig, a.name)
    summarize(d, Path(a.csv), a.target or 1e-3)


if __name__ == "__main__":
    use_proto = "--protocol" in sys.argv
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--manifest", default=None, required=not use_proto)
    e.add_argument("--norm-stats", default=None,
                   help="training norm_stats.npz (default: from the run's hparams.yaml)")
    e.add_argument("--split", default=None if use_proto else "test")
    e.add_argument("--n-waveforms", type=int, default=None if use_proto else 1000)
    e.add_argument("--fractions", type=float, nargs="+",
                   default=None if use_proto else [0.0, 0.25, 0.5, 0.75],
                   help="seeded fractions; 0 = theta-only (1 patch)")
    e.add_argument("--window-start", type=float, default=None,
                   help="start of common window as a fraction (default: max of --fractions)")
    e.add_argument("--n-steps", type=int, default=None if use_proto else None,
                   required=not use_proto, help="Euler steps (frozen protocol)")
    e.add_argument("--mean-of", type=int, default=None if use_proto else 1)
    e.add_argument("--temperature", type=float, default=None if use_proto else 1.0)
    e.add_argument("--amp-mode", default=None if use_proto else "constant",
                   choices=("constant", "linear", "pchip"),
                   help="amplitude reconstruction (default: constant = legacy)")
    e.add_argument("--seed", type=int, default=None if use_proto else 1234)
    e.add_argument("--batch-size", type=int, default=None if use_proto else 64)
    e.add_argument("--target", type=float, default=1e-3)
    e.add_argument("--device", default=None)
    e.add_argument("--out", default="results/context_fraction.csv")
    e.add_argument("--protocol", default=None)
    e.add_argument("--override", action="store_true")
    e.add_argument("--fast", action="store_true",
                   help="opt-in generate_fast (static KV + capture); default off")
    p = sub.add_parser("plot")
    p.add_argument("csv")
    p.add_argument("--target", type=float, default=1e-3)
    p.add_argument("--name", default="fm_context_fraction")
    p.add_argument("--single", action="store_true", help="single-column (stacked) layout")
    p.add_argument("--talk", action="store_true")
    args = ap.parse_args()
    if args.cmd == "eval":
        args.protocol_data = None
        args.protocol_overrides = {}
        if args.protocol:
            from protocol_utils import enforce_protocol, load_protocol
            args.protocol_data = load_protocol(args.protocol)
            args.protocol_overrides = enforce_protocol(
                args, args.protocol_data, override=args.override,
            )
            proto = args.protocol_data
            if args.manifest is None:
                args.manifest = proto["test_manifest"]
            if args.split is None:
                args.split = proto["split"]
            if args.n_waveforms is None:
                args.n_waveforms = int(proto["n_waveforms_flagship"])
            if args.n_steps is None:
                args.n_steps = int(proto["n_steps"])
            if args.mean_of is None:
                args.mean_of = int(proto["mean_of"])
            if args.temperature is None:
                args.temperature = float(proto["temperature"])
            if args.amp_mode is None:
                args.amp_mode = proto["amp_mode"]
            if args.seed is None:
                args.seed = int(proto["seed"])
            if args.batch_size is None:
                args.batch_size = int(proto["batch_size"])
            if args.fractions is None:
                args.fractions = [float(proto["context_fraction"])]
            if args.window_start is None:
                args.window_start = float(proto["window_start_fraction"])
        if args.device is None:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        run_eval(args)
    else:
        run_plot(args)
