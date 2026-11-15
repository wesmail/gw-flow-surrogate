# Phase 0 — Evaluation protocol audit

Date: 2026-11-09  
Repo: `gw-flow-surrogate` @ `git rev-parse HEAD` → see sidecar when runs start  
Scope: `scan_1M/` (32 Lightning runs) + eval / figure entry points

Companion CSVs:

- `results/protocol/norm_stats_audit.csv` — one row per run (hparams fields)
- `results/protocol/checkpoint_inventory.csv` — best ckpt by lowest `val_mismatch` in filename

---

## 1. Norm-stats audit

### What the scan runs recorded

| Field | Value across all 32 runs |
|-------|--------------------------|
| `norm_stats_path` | **`null` / `None` every run** |
| `compute_stats_if_missing` | `true` (from checkpoint `datamodule_hyper_parameters`) |
| `norm_stats_max_samples` | `1000` |
| `manifest_path` | `/data/hammad/GW/gw_data/n{N}/manifest.parquet` (per size) |
| `residual_scale` | varies per $N$: **0.07967 … 0.08489** |
| `patch_len` | 32 |
| `context_patches` | 1 (training) |
| `n_sampling_steps` | 20 |
| `use_abs_time` | `true` |
| Model | `d_model=128`, `n_layers=6` (~2.5M) |

### Expected vs actual

- **Scale-study design** (`scripts/scale_study/`): one frozen `norm_stats.npz` for all $N$.
- **`scan_1M` actual:** **not** one frozen file. Each size pool computed its own patch-level stats at train time (subsample ≤1000 train rows) and wrote them next to that pool’s remote manifest as `norm_stats.npz`.
- **Groups:** one logical group **per $N$** (11 groups: 10k…1000k). Seeds within an $N$ share the same manifest path, so they should share the same npz *if* the file was computed once and reused; with `compute_stats_if_missing: true`, the first seed to train would create it and later seeds would load it — **unless** concurrent first writes raced. Treat **per-$N$ stats** as the recovery unit.
- **Local availability:** `/data/hammad/GW/gw_data` is **not mounted** here. Zero of 32 runs have a resolvable local npz. Checkpoints do **not** embed the four norm arrays (only `residual_scale` + datamodule hparams with `norm_stats_path: None`).

### How to recover (required before Phase 4 on scan ckpts)

1. Copy each `/data/hammad/GW/gw_data/n*/norm_stats.npz` (and ideally the manifests) onto this machine, **or**
2. Recompute with `compute_norm_stats(manifest, out, patch_len=32, max_samples=1000, seed=0)` on the original train split — **only exact if the original subsample seed was 0** (default in `compute_norm_stats`). If a different seed was used, bit-identical recovery needs that seed from training logs.

Until then, `resolve_norm_stats` will correctly **raise** for scan checkpoints when `hparams.yaml` has `norm_stats_path: null` and no `--norm-stats` is passed.

---

## 2. Checkpoint inventory

Rule (unchanged): **lowest `val_mismatch` in the filename**; never quote that number as a paper metric (P7).

| Flag | Runs |
|------|------|
| `NO_CHECKPOINT` | `n1000k/seed43` (events + hparams only) |
| `EARLY_BEST_EPOCH` (epoch ≤ 20) | `n631k/seed44` best = epoch **17**, val_mismatch=0.0111 (also has epoch 11) |
| Usable best ckpts | **31** |

Full table: `checkpoint_inventory.csv`. Flagship for Phase 4:  
`scan_1M/n1000k/seed42/checkpoints/epoch=062-val_mismatch=0.0021.ckpt`.

---

## 3. Grid audit

Training HDF5 not available locally. Geometry inferred from `data_generation.config.GenerationConfig` defaults, which match generation CLI defaults and are consistent with scan training (`patch_len=32`, absolute-time embedding on):

| Quantity | Value |
|----------|-------|
| `dt` | 0.5 M |
| `t[0]` | −5000.0 M |
| `t[-1]` | 130.0 M |
| `n_time` | 10261 |
| `n_patches` (`patch_len=32`) | 320 |
| Peak alignment | merger at $t=0$ after generation align |

**Action for Phase 1:** generate `data/test5k` with the same `dt/t_min/t_max` and assert `np.array_equal(t_test, t_train_recorded)` against this recorded grid (and against any recovered training shard’s `t` when data arrive).

---

## 4. Problem register P1–P11

| # | Verdict | Evidence | Required fix status |
|---|---|---|---|
| **P1** | **Confirmed** | `eval_fm_unified.py:86` `stats_path = Path(args.manifest).parent / "norm_stats.npz"`; same pattern `paper_figs/fig_calibration.py:69`, `fig_speed_frontier.py:69`, `scripts/measure_floor.py:70`; `run_prefreeze_pipeline.sh:52` comment. | Fix in new `eval_paper.py` / `protocol_utils.resolve_norm_stats` (already sketched in `fig_context_fraction.py:90–112`). Do not change training. |
| **P2** | **Confirmed** | `datasets/waveform_dataset.py:104–110` returns mean 0 / std 1 if missing. `evalute.py:269–272` guards; `eval_fm_unified.py` does **not**. | Guard in every new eval entry point; leave dataset class unchanged. |
| **P3** | **Confirmed** | `evalute.py:116–119` builds `h_true` from true tokens via `_reconstruct_waveform`, not from HDF5 `log_a`/`phi` tracks. | New evaluator: original waveform truth + token-space secondary + floor column. (`fig_context_fraction.py` already does original-waveform mismatch.) |
| **P4** | **Confirmed** | `eval_fm_unified.py:117–119` assigns `mismatch_*_full =` future values with a comment claiming seed dilution. | Compute real full-waveform mismatch. |
| **P5** | **Confirmed** | No `torch.manual_seed` in `eval_fm_unified.py` / `evalute.py` generation loops. (`fig_context_fraction.py:223` already seeds `seed + bi`.) | Require protocol seed + `manual_seed(seed + batch_index)`. |
| **P6** | **Confirmed** | `eval_fm_unified.py:78` `default=20`; `results/prefreeze/PROTOCOL.txt` has `n-steps=40`; training `n_sampling_steps=20`. | `--n-steps` required via `PROTOCOL.yaml`. |
| **P7** | **Confirmed** | Training configs use `context_patches: 1`; `_rollout_mismatch` (`lightning_module.py:201–251`) uses `rollout_mean_of` (default 1), phase-abs inner product, mean over ≤`rollout_n_waveforms` (4). | Document only; never quote filename metric. |
| **P8** | **Confirmed** | `paper_figs/fig_q_anatomy.py:79–81` `merge_asof` on `q`, `tolerance=1e-4`. | Join on `idx`. |
| **P9** | **Confirmed** | `paper_figs/fig_scaling_final.py:34–35` hard-coded `CONVERGED_X/Y`. | Read scaling CSVs. |
| **P10** | **Confirmed** | `evalute.py:154–168` always runs teacher-forced sampling inside `evaluate_batch`. | New evaluator must not call `evaluate_batch`. |
| **P11** | **Confirmed (requirement)** | Grid must match; no assert today in eval scripts. Generator `n_train=0` is awkward (`n_val = max(1, int(0*val_fraction))` still emits a val split). | Assert `t` equality in `eval_paper.py`; Phase 1 uses a thin wrapper without editing `generate_dataset`. |

---

## 5. Already-present scaffolding (do not reinvent blindly)

| Path | Relevance |
|------|-----------|
| `paper_figs/fig_context_fraction.py` | `resolve_norm_stats`, fp64 reconstruct, original-waveform mismatch, seeding, sidecars — extract into `protocol_utils.py` |
| `scripts/make_smoke_cf.py` | Synthetic chirps + tiny ckpt for CPU tests |
| `fm_context_fraction_PREVIEW_dummy_data.png` | Preview figure only |

---

## 6. Local data on this machine (found after Phase 0)

| Pool | Path (via symlink under this repo) | Splits | `norm_stats.npz` |
|------|--------------------------------------|--------|------------------|
| 25k | `data/25k` → `../GWSurrogate/data/25k` | 25k / 2.5k / 500 test | yes |
| 120k | `data/120k` → `../GWSurrogate/data/120k` | 120k / 12k / 1000 test / 500 ood | yes |
| test5k | `data/test5k` (generated Phase 1) | 5000 test / 1000 ood | **no** (by design) |

All three share the audited grid. Prefreeze historically quoted the **120k test** set.

## 7. Blockers before Phase 4 on `scan_1M`

1. **Missing per-$N$ scan training `norm_stats.npz`** from `/data/hammad/GW/gw_data/n*/`. Local 25k/120k norms are **not** substitutes for those checkpoints (P1).
2. GPU for 4a/4c/4d.
3. Alternative: run Phase 4a on a local 25k/120k-trained checkpoint with the matching local npz to freeze `context_fraction` / `n_steps`, then re-eval `scan_1M` when (1) arrives.
